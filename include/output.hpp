#pragma once

#include "inference.hpp"

#include <filesystem>
#include <string>

namespace embed {

enum class OutputFormat { Npz, Json };

/// Write embeddings to disk in the given format.
void write_embeddings(const EmbeddingResult& result,
                      const std::filesystem::path& output_dir,
                      OutputFormat format);

}  // namespace embed
