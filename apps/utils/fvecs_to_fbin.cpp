#include "vamana/core/io.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " --input_file <fvecs_file> --output_file <fbin_file>" << std::endl;
    std::cout << "Convert .fvecs file to DiskANN .fbin format" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --input_file    Input .fvecs file path" << std::endl;
    std::cout << "  --output_file   Output .fbin file path" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string input_file, output_file;
    
    // Simple argument parsing
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            print_usage(argv[0]);
            return 1;
        }
        
        std::string arg = argv[i];
        std::string value = argv[i + 1];
        
        if (arg == "--input_file") {
            input_file = value;
        } else if (arg == "--output_file") {
            output_file = value;
        } else {
            std::cout << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (input_file.empty() || output_file.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        std::cout << "Converting " << input_file << " to " << output_file << std::endl;
        
        // Read .fvecs file
        std::ifstream file(input_file, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open input file: " + input_file);
        }
        
        // Read first vector to get dimension
        uint32_t dimension;
        file.read(reinterpret_cast<char*>(&dimension), sizeof(uint32_t));
        if (!file) {
            throw std::runtime_error("Failed to read dimension from fvecs file");
        }
        
        // Reset to beginning and count vectors
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        size_t vector_size = sizeof(uint32_t) + dimension * sizeof(float);
        uint32_t num_points = file_size / vector_size;
        
        std::cout << "Input file: " << num_points << " vectors of dimension " << dimension << std::endl;
        
        // Allocate memory for data
        std::vector<float> data(static_cast<size_t>(num_points) * dimension);
        
        // Read all vectors
        for (uint32_t i = 0; i < num_points; i++) {
            uint32_t vec_dim;
            file.read(reinterpret_cast<char*>(&vec_dim), sizeof(uint32_t));
            if (vec_dim != dimension) {
                throw std::runtime_error("Inconsistent dimension in fvecs file");
            }
            
            file.read(reinterpret_cast<char*>(data.data() + i * dimension), 
                     dimension * sizeof(float));
        }
        
        file.close();
        
        // Save as .fbin
        vamana::io::save_fbin(output_file, data.data(), num_points, dimension);
        
        std::cout << "Conversion completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}