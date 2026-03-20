#include "output.hpp"

#include <cnpy.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

namespace embed {

void write_embeddings(const EmbeddingResult& result,
                      const std::filesystem::path& output_dir,
                      OutputFormat format) {
    std::filesystem::create_directories(output_dir);

    if (format == OutputFormat::Npz) {
        auto npz_path = output_dir / "embeddings.npz";
        cnpy::npz_save(npz_path.string(), "embeddings",
                        result.data.data(),
                        {result.n, result.embed_dim}, "w");
        spdlog::info("Wrote {}", npz_path.string());

        // Path manifest as JSON sidecar
        auto manifest_path = output_dir / "paths.json";
        nlohmann::json paths = nlohmann::json::array();
        for (auto& p : result.paths) {
            paths.push_back(p.string());
        }
        std::ofstream f(manifest_path);
        f << paths.dump(2);
        spdlog::info("Wrote {}", manifest_path.string());

    } else {
        auto json_path = output_dir / "embeddings.json";
        nlohmann::json arr = nlohmann::json::array();
        for (size_t i = 0; i < result.n; ++i) {
            nlohmann::json entry;
            entry["path"] = result.paths[i].string();
            entry["embedding"] = std::vector<float>(
                result.row(i), result.row(i) + result.embed_dim);
            arr.push_back(std::move(entry));
        }
        std::ofstream f(json_path);
        f << arr.dump();
        spdlog::info("Wrote {}", json_path.string());
    }
}

}  // namespace embed
