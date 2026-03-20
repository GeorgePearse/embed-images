#pragma once

#include "inference.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace embed {

struct DuplicatePair {
    size_t i;
    size_t j;
    float similarity;  // cosine similarity
};

/// Find the top-K most similar image pairs by cosine similarity.
std::vector<DuplicatePair> find_duplicates(const EmbeddingResult& result,
                                           size_t top_k = 50,
                                           float threshold = 0.9f);

/// Write duplicate pairs to a JSON file.
void write_duplicates(const std::vector<DuplicatePair>& pairs,
                      const EmbeddingResult& result,
                      const std::filesystem::path& output_path);

}  // namespace embed
