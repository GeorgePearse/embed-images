#include "inference.hpp"

#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <cstring>
#include <numeric>
#include <stdexcept>

namespace embed {

// ImageNet normalization constants.
static constexpr float MEAN[] = {0.485f, 0.456f, 0.406f};
static constexpr float STD[] = {0.229f, 0.224f, 0.225f};

/// Load a thumbnail and convert to NCHW float32 with ImageNet normalization.
/// Output buffer must hold 3*size*size floats.
static void load_and_normalize(const std::filesystem::path& path, int size, float* out) {
    auto img = cv::imread(path.string(), cv::IMREAD_COLOR);  // BGR
    if (img.empty()) throw std::runtime_error("Cannot read " + path.string());

    if (img.rows != size || img.cols != size) {
        cv::resize(img, img, cv::Size(size, size));
    }

    // Convert BGR -> RGB, HWC -> NCHW, normalize
    const int hw = size * size;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            auto pixel = img.at<cv::Vec3b>(y, x);
            int idx = y * size + x;
            // OpenCV is BGR, CLIP expects RGB
            out[0 * hw + idx] = (pixel[2] / 255.0f - MEAN[0]) / STD[0];  // R
            out[1 * hw + idx] = (pixel[1] / 255.0f - MEAN[1]) / STD[1];  // G
            out[2 * hw + idx] = (pixel[0] / 255.0f - MEAN[2]) / STD[2];  // B
        }
    }
}

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess)                                           \
            throw std::runtime_error(std::string("CUDA error: ") +        \
                                     cudaGetErrorString(err));            \
    } while (0)

EmbeddingResult run_batched(Engine& engine,
                            const ThumbnailManifest& manifest,
                            int batch_size,
                            int size) {
    const size_t n = manifest.size();
    spdlog::info("Running inference: {} images, batch_size={}", n, batch_size);

    auto& ctx = engine.context();
    auto& eng = engine.engine();

    // Discover input/output tensor names and shapes
    const int num_io = eng.getNbIOTensors();
    std::string input_name, output_name;
    for (int i = 0; i < num_io; ++i) {
        auto name = eng.getIOTensorName(i);
        if (eng.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_name = name;
        } else {
            output_name = name;
        }
    }
    if (input_name.empty() || output_name.empty()) {
        throw std::runtime_error("Could not find input/output tensors");
    }

    // Get output embedding dimension from the engine's output shape
    auto out_dims = eng.getTensorShape(output_name.c_str());
    // Shape is typically (-1, embed_dim) or (batch, embed_dim)
    int embed_dim = out_dims.d[out_dims.nbDims - 1];
    spdlog::info("Input: '{}', Output: '{}' (embed_dim={})", input_name, output_name, embed_dim);

    const size_t input_elems_per_image = 3 * size * size;
    const size_t input_bytes_per_image = input_elems_per_image * sizeof(float);
    const size_t output_bytes_per_image = embed_dim * sizeof(float);

    // Allocate host buffers
    std::vector<float> host_input(batch_size * input_elems_per_image);
    std::vector<float> host_output(batch_size * embed_dim);

    // Allocate device buffers
    void* d_input = nullptr;
    void* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, batch_size * input_bytes_per_image));
    CUDA_CHECK(cudaMalloc(&d_output, batch_size * output_bytes_per_image));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    EmbeddingResult result;
    result.embed_dim = embed_dim;
    result.data.reserve(n * embed_dim);
    result.paths.reserve(n);

    for (size_t offset = 0; offset < n; offset += batch_size) {
        const size_t batch_n = std::min(static_cast<size_t>(batch_size), n - offset);

        // Load and normalize batch
        for (size_t i = 0; i < batch_n; ++i) {
            load_and_normalize(manifest[offset + i].thumbnail, size,
                               host_input.data() + i * input_elems_per_image);
        }

        // Set dynamic batch dimension
        nvinfer1::Dims input_dims;
        auto base_dims = eng.getTensorShape(input_name.c_str());
        input_dims.nbDims = base_dims.nbDims;
        input_dims.d[0] = static_cast<int>(batch_n);
        for (int d = 1; d < base_dims.nbDims; ++d) {
            input_dims.d[d] = base_dims.d[d];
        }
        ctx.setInputShape(input_name.c_str(), input_dims);

        // Copy input to device
        CUDA_CHECK(cudaMemcpyAsync(d_input, host_input.data(),
                                   batch_n * input_bytes_per_image,
                                   cudaMemcpyHostToDevice, stream));

        // Set tensor addresses
        ctx.setTensorAddress(input_name.c_str(), d_input);
        ctx.setTensorAddress(output_name.c_str(), d_output);

        // Infer
        if (!ctx.enqueueV3(stream)) {
            throw std::runtime_error("Inference failed at batch offset " + std::to_string(offset));
        }

        // Copy output back
        CUDA_CHECK(cudaMemcpyAsync(host_output.data(), d_output,
                                   batch_n * output_bytes_per_image,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Collect results
        for (size_t i = 0; i < batch_n; ++i) {
            result.paths.push_back(manifest[offset + i].original);
            result.data.insert(result.data.end(),
                               host_output.data() + i * embed_dim,
                               host_output.data() + (i + 1) * embed_dim);
        }

        if ((offset + batch_n) % 1000 < static_cast<size_t>(batch_size) || offset + batch_n == n) {
            spdlog::info("  inference: {}/{}", offset + batch_n, n);
        }
    }

    result.n = n;

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    spdlog::info("Inference complete: {} embeddings, dim={}", result.n, result.embed_dim);
    return result;
}

}  // namespace embed
