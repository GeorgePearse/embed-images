#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace embed {

struct ThumbnailEntry {
    std::filesystem::path original;
    std::filesystem::path thumbnail;
};

using ThumbnailManifest = std::vector<ThumbnailEntry>;

/// Discover all images under `images_dir` (recursively) and resize them to
/// `size x size` PNGs under `thumb_dir`. Uses OpenMP for parallelism.
ThumbnailManifest generate_thumbnails(const std::filesystem::path& images_dir,
                                      const std::filesystem::path& thumb_dir,
                                      int size);

}  // namespace embed
