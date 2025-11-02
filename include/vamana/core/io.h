#pragma once

#include <cstdint>
#include <string>

namespace vamana {
namespace io {

std::string resolve_dataset_path(const std::string& filename);

void load_data(const char* filename, float*& data, uint32_t& num_points, uint32_t& dimension);
void save_data(const char* filename, const float* data, uint32_t num_points, uint32_t dimension);
void free_data(float* data);

void load_fbin(const std::string& filename, float*& data, uint32_t& num_points, uint32_t& dimension);
void save_fbin(const std::string& filename, const float* data, uint32_t num_points, uint32_t dimension);

void load_ibin(const std::string& filename, uint32_t*& data, uint32_t& num_queries, uint32_t& k);
void save_ibin(const std::string& filename, const uint32_t* data, uint32_t num_queries, uint32_t k);

} // namespace io
} // namespace vamana