#include "vamana/core/io.h"
#include "vamana/core/distance.h"
#include <cstdio>
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>

namespace vamana {
namespace io {

std::string resolve_dataset_path(const std::string& filename) {
    // If it's already an absolute path or contains '/', use as-is
    if (filename[0] == '/' || filename.find('/') != std::string::npos) {
        return filename;
    }
    
    // First try datasets/indexes directory
    std::string datasets_path = "datasets/indexes/" + filename;
    std::ifstream test_file(datasets_path);
    if (test_file.good()) {
        return datasets_path;
    }
    
    // Fall back to current directory
    return filename;
}

void load_data(const char* filename, float*& data, uint32_t& num_points, uint32_t& dimension) {
    std::string resolved_path = resolve_dataset_path(filename);
    std::FILE* file = std::fopen(resolved_path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error(std::string("Cannot open file: ") + resolved_path);
    }
    
    // Read header
    if (std::fread(&num_points, sizeof(uint32_t), 1, file) != 1 ||
        std::fread(&dimension, sizeof(uint32_t), 1, file) != 1) {
        std::fclose(file);
        throw std::runtime_error("Failed to read file header");
    }
    
    std::cout << "Loading " << num_points << " points of dimension " << dimension << std::endl;
    
    // Allocate aligned memory for SIMD operations
    size_t total_size = static_cast<size_t>(num_points) * dimension * sizeof(float);
    data = static_cast<float*>(aligned_alloc_wrapper(32, total_size));
    if (!data) {
        std::fclose(file);
        throw std::runtime_error("Failed to allocate aligned memory");
    }
    
    // Read data
    size_t elements_read = std::fread(data, sizeof(float), 
                                    static_cast<size_t>(num_points) * dimension, file);
    std::fclose(file);
    
    if (elements_read != static_cast<size_t>(num_points) * dimension) {
        aligned_free_wrapper(data);
        data = nullptr;
        throw std::runtime_error("Failed to read all data from file");
    }
    
    std::cout << "Successfully loaded dataset" << std::endl;
}

void save_data(const char* filename, const float* data, uint32_t num_points, uint32_t dimension) {
    std::FILE* file = std::fopen(filename, "wb");
    if (!file) {
        throw std::runtime_error(std::string("Cannot create file: ") + filename);
    }
    
    // Write header
    if (std::fwrite(&num_points, sizeof(uint32_t), 1, file) != 1 ||
        std::fwrite(&dimension, sizeof(uint32_t), 1, file) != 1) {
        std::fclose(file);
        throw std::runtime_error("Failed to write file header");
    }
    
    // Write data
    size_t elements_written = std::fwrite(data, sizeof(float), 
                                        static_cast<size_t>(num_points) * dimension, file);
    std::fclose(file);
    
    if (elements_written != static_cast<size_t>(num_points) * dimension) {
        throw std::runtime_error("Failed to write all data to file");
    }
    
    std::cout << "Successfully saved dataset to " << filename << std::endl;
}

void free_data(float* data) {
    if (data) {
        aligned_free_wrapper(data);
    }
}

void load_fbin(const std::string& filename, float*& data, uint32_t& num_points, uint32_t& dimension) {
    std::string resolved_path = resolve_dataset_path(filename);
    std::FILE* file = std::fopen(resolved_path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error(std::string("Cannot open fbin file: ") + resolved_path);
    }
    
    // Read header - DiskANN .fbin format
    if (std::fread(&num_points, sizeof(uint32_t), 1, file) != 1 ||
        std::fread(&dimension, sizeof(uint32_t), 1, file) != 1) {
        std::fclose(file);
        throw std::runtime_error("Failed to read fbin file header");
    }
    
    std::cout << "Loading " << num_points << " points of dimension " << dimension << " from .fbin" << std::endl;
    
    // Allocate aligned memory for SIMD operations
    size_t total_size = static_cast<size_t>(num_points) * dimension * sizeof(float);
    data = static_cast<float*>(aligned_alloc_wrapper(32, total_size));
    if (!data) {
        std::fclose(file);
        throw std::runtime_error("Failed to allocate aligned memory for fbin data");
    }
    
    // Read data
    size_t elements_read = std::fread(data, sizeof(float), 
                                    static_cast<size_t>(num_points) * dimension, file);
    std::fclose(file);
    
    if (elements_read != static_cast<size_t>(num_points) * dimension) {
        aligned_free_wrapper(data);
        data = nullptr;
        throw std::runtime_error("Failed to read all data from fbin file");
    }
    
    std::cout << "Successfully loaded fbin dataset" << std::endl;
}

void save_fbin(const std::string& filename, const float* data, uint32_t num_points, uint32_t dimension) {
    std::string resolved_path = resolve_dataset_path(filename);
    std::FILE* file = std::fopen(resolved_path.c_str(), "wb");
    if (!file) {
        throw std::runtime_error(std::string("Cannot create fbin file: ") + resolved_path);
    }
    
    // Write header - DiskANN .fbin format
    if (std::fwrite(&num_points, sizeof(uint32_t), 1, file) != 1 ||
        std::fwrite(&dimension, sizeof(uint32_t), 1, file) != 1) {
        std::fclose(file);
        throw std::runtime_error("Failed to write fbin header");
    }
    
    // Write data
    size_t elements_written = std::fwrite(data, sizeof(float), 
                                         static_cast<size_t>(num_points) * dimension, file);
    std::fclose(file);
    
    if (elements_written != static_cast<size_t>(num_points) * dimension) {
        throw std::runtime_error("Failed to write all data to fbin file");
    }
    
    std::cout << "Successfully saved fbin dataset to " << resolved_path << std::endl;
}

void load_ibin(const std::string& filename, uint32_t*& data, uint32_t& num_queries, uint32_t& k) {
    std::string resolved_path = resolve_dataset_path(filename);
    std::FILE* file = std::fopen(resolved_path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error(std::string("Cannot open ibin file: ") + resolved_path);
    }
    
    // Read header - DiskANN .ibin format
    if (std::fread(&num_queries, sizeof(uint32_t), 1, file) != 1 ||
        std::fread(&k, sizeof(uint32_t), 1, file) != 1) {
        std::fclose(file);
        throw std::runtime_error("Failed to read ibin file header");
    }
    
    std::cout << "Loading ground truth: " << num_queries << " queries, k=" << k << " from .ibin" << std::endl;
    
    // Allocate memory
    size_t total_size = static_cast<size_t>(num_queries) * k * sizeof(uint32_t);
    data = static_cast<uint32_t*>(std::malloc(total_size));
    if (!data) {
        std::fclose(file);
        throw std::runtime_error("Failed to allocate memory for ibin data");
    }
    
    // Read data
    size_t elements_read = std::fread(data, sizeof(uint32_t), 
                                    static_cast<size_t>(num_queries) * k, file);
    std::fclose(file);
    
    if (elements_read != static_cast<size_t>(num_queries) * k) {
        std::free(data);
        data = nullptr;
        throw std::runtime_error("Failed to read all data from ibin file");
    }
    
    std::cout << "Successfully loaded ibin ground truth" << std::endl;
}

void save_ibin(const std::string& filename, const uint32_t* data, uint32_t num_queries, uint32_t k) {
    std::string resolved_path = resolve_dataset_path(filename);
    std::FILE* file = std::fopen(resolved_path.c_str(), "wb");
    if (!file) {
        throw std::runtime_error(std::string("Cannot create ibin file: ") + resolved_path);
    }
    
    // Write header - DiskANN .ibin format
    if (std::fwrite(&num_queries, sizeof(uint32_t), 1, file) != 1 ||
        std::fwrite(&k, sizeof(uint32_t), 1, file) != 1) {
        std::fclose(file);
        throw std::runtime_error("Failed to write ibin header");
    }
    
    // Write data
    size_t elements_written = std::fwrite(data, sizeof(uint32_t), 
                                         static_cast<size_t>(num_queries) * k, file);
    std::fclose(file);
    
    if (elements_written != static_cast<size_t>(num_queries) * k) {
        throw std::runtime_error("Failed to write all data to ibin file");
    }
    
    std::cout << "Successfully saved ibin ground truth to " << resolved_path << std::endl;
}

} // namespace io
} // namespace vamana