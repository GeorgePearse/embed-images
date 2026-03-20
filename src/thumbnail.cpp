#include "thumbnail.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <stdexcept>

namespace embed {

static const std::vector<std::string> IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"};

static bool is_image(const std::filesystem::path& p) {
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(IMAGE_EXTENSIONS.begin(), IMAGE_EXTENSIONS.end(), ext) !=
           IMAGE_EXTENSIONS.end();
}

ThumbnailManifest generate_thumbnails(const std::filesystem::path& images_dir,
                                      const std::filesystem::path& thumb_dir,
                                      int size) {
    // Discover images
    std::vector<std::filesystem::path> image_paths;
    for (auto& entry : std::filesystem::recursive_directory_iterator(images_dir)) {
        if (entry.is_regular_file() && is_image(entry.path())) {
            image_paths.push_back(entry.path());
        }
    }
    std::sort(image_paths.begin(), image_paths.end());

    if (image_paths.empty()) {
        throw std::runtime_error("No images found in " + images_dir.string());
    }

    spdlog::info("Found {} images, generating {}x{} thumbnails", image_paths.size(), size, size);
    std::filesystem::create_directories(thumb_dir);

    ThumbnailManifest manifest;
    manifest.reserve(image_paths.size());

    size_t done = 0;
    size_t failed = 0;

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < image_paths.size(); ++i) {
        auto& src = image_paths[i];
        auto dest = thumb_dir / (src.stem().string() + "_thumb.png");

        auto img = cv::imread(src.string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            #pragma omp atomic
            ++failed;
            continue;
        }

        cv::Mat thumb;
        cv::resize(img, thumb, cv::Size(size, size), 0, 0, cv::INTER_LANCZOS4);
        cv::imwrite(dest.string(), thumb);

        #pragma omp critical
        {
            manifest.push_back({src, dest});
            ++done;
            if (done % 500 == 0) {
                spdlog::info("  thumbnails: {}/{}", done, image_paths.size());
            }
        }
    }

    if (failed > 0) {
        spdlog::warn("{} images failed to load", failed);
    }

    // Sort manifest to match original order
    std::sort(manifest.begin(), manifest.end(),
              [](const auto& a, const auto& b) { return a.original < b.original; });

    spdlog::info("Generated {} thumbnails", manifest.size());
    return manifest;
}

}  // namespace embed
