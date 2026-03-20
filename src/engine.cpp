#include "engine.hpp"

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <stdexcept>

namespace embed {

void TrtLogger::log(Severity severity, const char* msg) noexcept {
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            spdlog::error("[TRT] {}", msg);
            break;
        case Severity::kWARNING:
            spdlog::warn("[TRT] {}", msg);
            break;
        case Severity::kINFO:
            spdlog::info("[TRT] {}", msg);
            break;
        default:
            break;
    }
}

static std::vector<char> read_file(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open " + path.string());
    auto size = f.tellg();
    std::vector<char> buf(size);
    f.seekg(0);
    f.read(buf.data(), size);
    return buf;
}

static void write_file(const std::filesystem::path& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write " + path.string());
    f.write(static_cast<const char*>(data), size);
}

Engine Engine::build(const std::filesystem::path& onnx_path,
                     const std::filesystem::path& cache_dir,
                     bool fp16) {
    Engine eng;
    std::filesystem::create_directories(cache_dir);

    auto cache_path = cache_dir / (onnx_path.stem().string() + ".engine");

    if (std::filesystem::exists(cache_path)) {
        spdlog::info("Loading cached engine: {}", cache_path.string());
        eng.engine_data_ = read_file(cache_path);

        eng.runtime_.reset(nvinfer1::createInferRuntime(eng.logger_));
        eng.engine_.reset(eng.runtime_->deserializeCudaEngine(
            eng.engine_data_.data(), eng.engine_data_.size()));
    } else {
        spdlog::info("Building engine from ONNX: {}", onnx_path.string());

        auto builder = TrtUniquePtr<nvinfer1::IBuilder>(
            nvinfer1::createInferBuilder(eng.logger_));
        auto network = TrtUniquePtr<nvinfer1::INetworkDefinition>(
            builder->createNetworkV2(1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
        auto parser = TrtUniquePtr<nvonnxparser::IParser>(
            nvonnxparser::createParser(*network, eng.logger_));

        if (!parser->parseFromFile(onnx_path.c_str(),
                                   static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            throw std::runtime_error("Failed to parse ONNX: " + onnx_path.string());
        }

        auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

        // Add optimization profile for dynamic batch dimension
        auto profile = builder->createOptimizationProfile();
        for (int32_t i = 0; i < network->getNbInputs(); ++i) {
            auto input = network->getInput(i);
            auto dims = input->getDimensions();
            // Set min/opt/max for batch dimension (dim 0)
            auto min_dims = dims; min_dims.d[0] = 1;
            auto opt_dims = dims; opt_dims.d[0] = 32;
            auto max_dims = dims; max_dims.d[0] = 64;
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, min_dims);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, max_dims);
        }
        config->addOptimizationProfile(profile);

        if (fp16 && builder->platformHasFastFp16()) {
            spdlog::info("Enabling FP16 precision");
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }

        auto serialized = TrtUniquePtr<nvinfer1::IHostMemory>(
            builder->buildSerializedNetwork(*network, *config));
        if (!serialized) throw std::runtime_error("Engine build failed");

        write_file(cache_path, serialized->data(), serialized->size());
        spdlog::info("Cached engine to: {}", cache_path.string());

        eng.engine_data_.assign(
            static_cast<const char*>(serialized->data()),
            static_cast<const char*>(serialized->data()) + serialized->size());

        eng.runtime_.reset(nvinfer1::createInferRuntime(eng.logger_));
        eng.engine_.reset(eng.runtime_->deserializeCudaEngine(
            eng.engine_data_.data(), eng.engine_data_.size()));
    }

    if (!eng.engine_) throw std::runtime_error("Failed to create engine");

    eng.context_.reset(eng.engine_->createExecutionContext());
    if (!eng.context_) throw std::runtime_error("Failed to create execution context");

    spdlog::info("Engine ready — {} I/O tensors", eng.engine_->getNbIOTensors());

    return eng;
}

int32_t Engine::binding_index(const std::string& name) const {
    // TensorRT 10+ uses name-based tensor API
    for (int32_t i = 0; i < engine_->getNbIOTensors(); ++i) {
        if (std::string(engine_->getIOTensorName(i)) == name) return i;
    }
    return -1;
}

nvinfer1::Dims Engine::binding_dims(int32_t index) const {
    auto name = engine_->getIOTensorName(index);
    return engine_->getTensorShape(name);
}

}  // namespace embed
