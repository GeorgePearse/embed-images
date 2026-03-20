#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace embed {

/// Custom deleter for TensorRT objects.
struct TrtDeleter {
    template <typename T>
    void operator()(T* p) const noexcept {
        delete p;
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter>;

/// TensorRT logger implementation.
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

/// Wraps a TensorRT engine + execution context.
class Engine {
public:
    /// Build or load a cached TensorRT engine from an ONNX model.
    /// The serialized engine is cached at `cache_dir/<model_stem>.engine`.
    static Engine build(const std::filesystem::path& onnx_path,
                        const std::filesystem::path& cache_dir,
                        bool fp16 = true);

    nvinfer1::ICudaEngine& engine() { return *engine_; }
    nvinfer1::IExecutionContext& context() { return *context_; }

    /// Returns the binding index for a given tensor name.
    int32_t binding_index(const std::string& name) const;

    /// Returns the shape of a binding.
    nvinfer1::Dims binding_dims(int32_t index) const;

private:
    TrtLogger logger_;
    TrtUniquePtr<nvinfer1::IRuntime> runtime_;
    TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
    TrtUniquePtr<nvinfer1::IExecutionContext> context_;
    std::vector<char> engine_data_;
};

}  // namespace embed
