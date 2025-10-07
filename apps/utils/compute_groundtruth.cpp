#include "vamana/core/io.h"
#include "vamana/core/distance.h"
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <queue>

struct NeighborResult {
    uint32_t id;
    float distance;
    
    NeighborResult(uint32_t id, float dist) : id(id), distance(dist) {}
    
    bool operator<(const NeighborResult& other) const {
        // For max heap (priority queue)
        return distance < other.distance;
    }
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " --data_type float --dist_fn l2 --base_file <file> --query_file <file> --gt_file <file> --K <k>" << std::endl;
    std::cout << "Compute brute-force ground truth for nearest neighbor search" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --data_type     Data type (currently only 'float' supported)" << std::endl;
    std::cout << "  --dist_fn       Distance function (currently only 'l2' supported)" << std::endl;
    std::cout << "  --base_file     Base dataset file (.fbin format)" << std::endl;
    std::cout << "  --query_file    Query dataset file (.fbin format)" << std::endl;
    std::cout << "  --gt_file       Output ground truth file (.ibin format)" << std::endl;
    std::cout << "  --K             Number of nearest neighbors to compute" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 13) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string data_type, dist_fn, base_file, query_file, gt_file;
    uint32_t K = 0;
    
    // Parse arguments
    for (int i = 1; i < argc; i += 2) {
        if (i + 1 >= argc) {
            print_usage(argv[0]);
            return 1;
        }
        
        std::string arg = argv[i];
        std::string value = argv[i + 1];
        
        if (arg == "--data_type") {
            data_type = value;
        } else if (arg == "--dist_fn") {
            dist_fn = value;
        } else if (arg == "--base_file") {
            base_file = value;
        } else if (arg == "--query_file") {
            query_file = value;
        } else if (arg == "--gt_file") {
            gt_file = value;
        } else if (arg == "--K") {
            K = std::stoul(value);
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
    
    if (base_file.empty() || query_file.empty() || gt_file.empty() || K == 0) {
        std::cout << "Error: Missing required arguments" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        std::cout << "=== COMPUTE GROUND TRUTH ===" << std::endl;
        std::cout << "Data type: " << data_type << std::endl;
        std::cout << "Distance function: " << dist_fn << std::endl;
        std::cout << "Base file: " << base_file << std::endl;
        std::cout << "Query file: " << query_file << std::endl;
        std::cout << "Ground truth file: " << gt_file << std::endl;
        std::cout << "K: " << K << std::endl;
        
        // Load base dataset
        float* base_data = nullptr;
        uint32_t num_base_points, base_dim;
        vamana::io::load_fbin(base_file, base_data, num_base_points, base_dim);
        
        // Load query dataset
        float* query_data = nullptr;
        uint32_t num_queries, query_dim;
        vamana::io::load_fbin(query_file, query_data, num_queries, query_dim);
        
        if (base_dim != query_dim) {
            throw std::runtime_error("Base and query dimensions don't match");
        }
        
        std::cout << "Computing ground truth for " << num_queries 
                  << " queries against " << num_base_points << " base points..." << std::endl;
        
        // Allocate result buffer
        std::vector<uint32_t> ground_truth(static_cast<size_t>(num_queries) * K);
        
        // Compute ground truth for each query
        for (uint32_t q = 0; q < num_queries; q++) {
            if (q % 100 == 0) {
                std::cout << "Processing query " << q << "/" << num_queries << std::endl;
            }
            
            const float* query = query_data + q * base_dim;
            
            // Use max heap to maintain top K closest
            std::priority_queue<NeighborResult> heap;
            
            // Compute distance to all base points
            for (uint32_t i = 0; i < num_base_points; i++) {
                const float* base_point = base_data + i * base_dim;
                float dist = adaptive_l2_distance(query, base_point, base_dim);
                
                if (heap.size() < K) {
                    heap.emplace(i, dist);
                } else if (dist < heap.top().distance) {
                    heap.pop();
                    heap.emplace(i, dist);
                }
            }
            
            // Extract results in ascending order of distance
            std::vector<NeighborResult> results;
            while (!heap.empty()) {
                results.push_back(heap.top());
                heap.pop();
            }
            
            // Reverse to get ascending order
            std::reverse(results.begin(), results.end());
            
            // Store in ground truth array
            for (uint32_t k = 0; k < K && k < results.size(); k++) {
                ground_truth[q * K + k] = results[k].id;
            }
        }
        
        // Save ground truth
        vamana::io::save_ibin(gt_file, ground_truth.data(), num_queries, K);
        
        // Clean up
        vamana::io::free_data(base_data);
        vamana::io::free_data(query_data);
        
        std::cout << "Ground truth computation completed successfully!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}