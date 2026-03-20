#pragma once

#include "engine.hpp"
#include "thumbnail.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace embed {

struct EmbeddingResult {
    std::vector<std::filesystem::path> paths;
    std::vector<float> data;  // flat (N * embed_dim)
    size_t n = 0;
    size_t embed_dim = 0;

    /// Access embedding for image i.
    const float* row(size_t i) const { return data.data() + i * embed_dim; }
};

/// Run batched inference over all thumbnails.
/// Loads each thumbnail, normalises to NCHW float32 (ImageNet stats),
/// and feeds through the TensorRT engine in batches.
EmbeddingResult run_batched(Engine& engine,
                            const ThumbnailManifest& manifest,
                            int batch_size,
                            int size);

}  // namespace embed
