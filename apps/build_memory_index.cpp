#include "vamana/core/index.h"
#include "vamana/core/io.h"
#include <iostream>
#include <string>
#include <chrono>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " --data_type float --dist_fn l2 --data_path <file> --index_path_prefix <prefix> -R <R> -L <L> --alpha <alpha>" << std::endl;
    std::cout << "Build Vamana memory index" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --data_type         Data type (currently only 'float' supported)" << std::endl;
    std::cout << "  --dist_fn           Distance function (currently only 'l2' supported)" << std::endl;
    std::cout << "  --data_path         Input dataset file (.fbin format)" << std::endl;
    std::cout << "  --index_path_prefix Output index path prefix" << std::endl;
    std::cout << "  -R                  Max degree of each node in the graph" << std::endl;
    std::cout << "  -L                  Search list size during construction" << std::endl;
    std::cout << "  --alpha             Alpha parameter for pruning (e.g., 1.2)" << std::endl;
    std::cout << "  -T, --num_threads   Number of threads (0 = use all cores)" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 15) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string data_type, dist_fn, data_path, index_path_prefix;
    uint32_t R = 0, L = 0;
    float alpha = 0.0f;
    uint32_t num_threads = 0; // 0 means use all cores
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--data_type" && i + 1 < argc) {
            data_type = argv[++i];
        } else if (arg == "--dist_fn" && i + 1 < argc) {
            dist_fn = argv[++i];
        } else if (arg == "--data_path" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--index_path_prefix" && i + 1 < argc) {
            index_path_prefix = argv[++i];
        } else if (arg == "-R" && i + 1 < argc) {
            R = std::stoul(argv[++i]);
        } else if (arg == "-L" && i + 1 < argc) {
            L = std::stoul(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            alpha = std::stof(argv[++i]);
        } else if (arg == "--num_threads" || arg == "-T" && i + 1 < argc){
            num_threads = std::stoul(argv[++i]);
        } else {
            std::cout << "Unknown argument: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate arguments
    if (data_type != "float") {
        std::cout << "Error: Currently only 'float' data type is supported" << std::endl;
        return 1;
    }
    
    if (dist_fn != "l2") {
        std::cout << "Error: Currently only 'l2' distance function is supported" << std::endl;
        return 1;
    }
    
    if (data_path.empty() || index_path_prefix.empty() || R == 0 || L == 0 || alpha <= 0.0f) {
        std::cout << "Error: Missing or invalid required arguments" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        std::cout << "=== BUILD VAMANA INDEX ===" << std::endl;
        std::cout << "Data type: " << data_type << std::endl;
        std::cout << "Distance function: " << dist_fn << std::endl;
        std::cout << "Data path: " << data_path << std::endl;
        std::cout << "Index prefix: " << index_path_prefix << std::endl;
        std::cout << "R: " << R << std::endl;
        std::cout << "L: " << L << std::endl;
        std::cout << "Alpha: " << alpha << std::endl;
        std::cout << "SIMD optimizations: " << (use_simd ? "ENABLED" : "DISABLED") << std::endl;
        
        // Load dataset
        float* data = nullptr;
        uint32_t num_points, dimension;
        vamana::io::load_fbin(data_path, data, num_points, dimension);
        
        // Create and build index
        VamanaIndex index(dimension, R, L, alpha, 750, num_threads);  // maxc = 750
        
        std::cout << "Building Vamana index..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        index.build(data, num_points);
        
        auto build_time = std::chrono::high_resolution_clock::now() - start_time;
        auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_time).count();
        
        std::cout << "Index built in " << build_ms << " ms" << std::endl;
        
        // Save index
        std::cout << "Saving index to: " << index_path_prefix << std::endl;
        index.save_index(index_path_prefix);
        std::cout << "Index saved successfully" << std::endl;
        
        // Clean up
        vamana::io::free_data(data);
        
        std::cout << "Build completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}