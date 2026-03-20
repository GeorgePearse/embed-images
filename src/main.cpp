#include "duplicates.hpp"
#include "engine.hpp"
#include "inference.hpp"
#include "output.hpp"
#include "thumbnail.hpp"

#include <CLI/CLI.hpp>
#include <spdlog/spdlog.h>

#include <future>
#include <string>

int main(int argc, char** argv) {
    CLI::App app{"embed-images: thumbnailing + TensorRT embedding extraction"};

    std::string model_path;
    std::string images_dir;
    std::string output_dir;
    int thumbnail_size = 224;
    int batch_size = 32;
    std::string format_str = "npz";
    bool find_dupes = false;
    size_t top_k = 50;
    float dupe_threshold = 0.9f;

    app.add_option("--model", model_path, "Path to ONNX embedding model")->required();
    app.add_option("--images-dir", images_dir, "Directory containing input images")->required();
    app.add_option("--output-dir", output_dir, "Output directory")->required();
    app.add_option("--thumbnail-size", thumbnail_size, "Thumbnail size (NxN)");
    app.add_option("--batch-size", batch_size, "Inference batch size");
    app.add_option("--output-format", format_str, "Output format: npz or json");
    app.add_flag("--find-duplicates", find_dupes, "Find most similar image pairs");
    app.add_option("--top-k", top_k, "Number of duplicate pairs to report");
    app.add_option("--duplicate-threshold", dupe_threshold, "Cosine similarity threshold");

    CLI11_PARSE(app, argc, argv);

    auto format = (format_str == "json") ? embed::OutputFormat::Json : embed::OutputFormat::Npz;

    spdlog::info("embed-images pipeline starting");
    spdlog::info("  model:          {}", model_path);
    spdlog::info("  images:         {}", images_dir);
    spdlog::info("  output:         {}", output_dir);
    spdlog::info("  thumbnail_size: {}", thumbnail_size);
    spdlog::info("  batch_size:     {}", batch_size);

    namespace fs = std::filesystem;
    auto cache_dir = fs::path(output_dir) / "engine_cache";
    auto thumb_dir = fs::path(output_dir) / "thumbnails";

    // Stage 1 + 2 in parallel: build engine and generate thumbnails
    auto engine_future = std::async(std::launch::async, [&]() {
        return embed::Engine::build(model_path, cache_dir);
    });

    auto thumb_future = std::async(std::launch::async, [&]() {
        return embed::generate_thumbnails(images_dir, thumb_dir, thumbnail_size);
    });

    auto engine = engine_future.get();
    auto manifest = thumb_future.get();

    // Stage 3: batched inference
    auto embeddings = embed::run_batched(engine, manifest, batch_size, thumbnail_size);

    // Write embeddings
    embed::write_embeddings(embeddings, output_dir, format);

    // Optionally find duplicates
    if (find_dupes) {
        auto pairs = embed::find_duplicates(embeddings, top_k, dupe_threshold);
        embed::write_duplicates(pairs, embeddings, fs::path(output_dir) / "duplicates.json");

        // Print top results to stdout
        spdlog::info("Top duplicate pairs:");
        for (size_t i = 0; i < std::min(pairs.size(), size_t{10}); ++i) {
            auto& p = pairs[i];
            spdlog::info("  {:.4f}  {} <-> {}",
                         p.similarity,
                         embeddings.paths[p.i].filename().string(),
                         embeddings.paths[p.j].filename().string());
        }
    }

    spdlog::info("Pipeline complete");
    return 0;
}
