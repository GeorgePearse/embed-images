#include "duplicates.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <queue>

namespace embed {

/// Cosine similarity between two vectors.
static float cosine_sim(const float* a, const float* b, size_t dim) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 0.0f ? dot / denom : 0.0f;
}

std::vector<DuplicatePair> find_duplicates(const EmbeddingResult& result,
                                           size_t top_k,
                                           float threshold) {
    const size_t n = result.n;
    const size_t dim = result.embed_dim;

    spdlog::info("Computing pairwise cosine similarities for {} images (threshold={:.2f})",
                 n, threshold);

    // Min-heap to keep top-K highest similarities
    auto cmp = [](const DuplicatePair& a, const DuplicatePair& b) {
        return a.similarity > b.similarity;
    };
    std::priority_queue<DuplicatePair, std::vector<DuplicatePair>, decltype(cmp)> heap(cmp);

    size_t pairs_above_threshold = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:pairs_above_threshold)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            float sim = cosine_sim(result.row(i), result.row(j), dim);
            if (sim >= threshold) {
                ++pairs_above_threshold;
                #pragma omp critical
                {
                    if (heap.size() < top_k) {
                        heap.push({i, j, sim});
                    } else if (sim > heap.top().similarity) {
                        heap.pop();
                        heap.push({i, j, sim});
                    }
                }
            }
        }

        if (i % 500 == 0 && i > 0) {
            spdlog::info("  similarity scan: {}/{}", i, n);
        }
    }

    spdlog::info("Found {} pairs above threshold {:.2f}, returning top {}",
                 pairs_above_threshold, threshold, std::min(top_k, pairs_above_threshold));

    // Extract sorted results (highest similarity first)
    std::vector<DuplicatePair> pairs;
    pairs.reserve(heap.size());
    while (!heap.empty()) {
        pairs.push_back(heap.top());
        heap.pop();
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const auto& a, const auto& b) { return a.similarity > b.similarity; });

    return pairs;
}

void write_duplicates(const std::vector<DuplicatePair>& pairs,
                      const EmbeddingResult& result,
                      const std::filesystem::path& output_path) {
    nlohmann::json arr = nlohmann::json::array();
    for (auto& p : pairs) {
        arr.push_back({
            {"image_a", result.paths[p.i].string()},
            {"image_b", result.paths[p.j].string()},
            {"cosine_similarity", p.similarity},
        });
    }

    std::ofstream f(output_path);
    f << arr.dump(2);
    spdlog::info("Wrote {} duplicate pairs to {}", pairs.size(), output_path.string());
}

}  // namespace embed
