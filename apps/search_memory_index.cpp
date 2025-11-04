#include "vamana/core/index.h"
#include "vamana/core/io.h"
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>

#ifdef _OPENMP
#include <omp.h>
#endif

struct SearchResult {
    std::vector<std::vector<uint32_t>> results;  // [query][neighbor]
    double qps;
    double average_recall;
};

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " --data_type float --dist_fn l2 --data_path <file> --index_path_prefix <prefix> --query_file <file> --gt_file <file> -K <K> -L <L1> <L2> ... --result_path <path>" << std::endl;
    std::cout << "Search Vamana memory index" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --data_type         Data type (currently only 'float' supported)" << std::endl;
    std::cout << "  --dist_fn           Distance function (currently only 'l2' supported)" << std::endl;
    std::cout << "  --data_path         Base dataset file (.fbin format)" << std::endl;
    std::cout << "  --index_path_prefix Index path prefix" << std::endl;
    std::cout << "  --query_file        Query dataset file (.fbin format)" << std::endl;
    std::cout << "  --gt_file           Ground truth file (.ibin format)" << std::endl;
    std::cout << "  -K                  Number of nearest neighbors to return" << std::endl;
    std::cout << "  -L                  Search list sizes (space-separated list)" << std::endl;
    std::cout << "  -T, --num_threads   Number of threads (0 = use all cores, default = 1)" << std::endl;
    std::cout << "  --result_path       Output directory for results" << std::endl;
}

double calculate_recall(const std::vector<uint32_t>& results, const uint32_t* ground_truth, uint32_t k) {
    if (k == 0) return 0.0;
    
    size_t hits = 0;
    for (size_t i = 0; i < std::min(results.size(), static_cast<size_t>(k)); i++) {
        uint32_t result_id = results[i];
        for (uint32_t j = 0; j < k; j++) {
            if (ground_truth[j] == result_id) {
                hits++;
                break;
            }
        }
    }
    
    return (double)hits / (double)std::min(results.size(), static_cast<size_t>(k));
}

SearchResult search_with_L(VamanaIndex& index, float* query_data, uint32_t num_queries, 
                          uint32_t dimension, uint32_t K, uint32_t L, 
                          const uint32_t* ground_truth_data, uint32_t gt_k, size_t num_threads) {
    SearchResult result;
    result.results.resize(num_queries);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    double total_recall = 0.0;
    
    // Set number of threads for parallel search (DiskANN pattern)
    #ifdef _OPENMP
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    #endif
    
    // Parallelize query processing - each thread handles one query at a time
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:total_recall)
    for (int64_t q = 0; q < (int64_t)num_queries; q++) {
        const float* query = query_data + q * dimension;
        // search() is single-threaded per query
        // Parallelism is achieved here at the batch level
        auto neighbors = index.search(query, K, L);
        
        // Extract IDs
        result.results[q].resize(neighbors.size());
        for (size_t i = 0; i < neighbors.size(); i++) {
            result.results[q][i] = neighbors[i].id;
        }
        
        // Calculate recall if ground truth available
        if (ground_truth_data && gt_k > 0) {
            const uint32_t* gt_for_query = ground_truth_data + q * gt_k;
            double recall = calculate_recall(result.results[q], gt_for_query, std::min(K, gt_k));
            total_recall += recall;
        }
    }
    
    auto search_time = std::chrono::high_resolution_clock::now() - start_time;
    auto search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_time).count();
    
    result.qps = (double)num_queries / ((double)search_ms / 1000.0);
    result.average_recall = ground_truth_data ? (total_recall / num_queries) : 0.0;
    
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 15) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string data_type, dist_fn, data_path, index_path_prefix, query_file, gt_file, result_path;
    uint32_t K = 0;
    std::vector<uint32_t> L_values;
    uint32_t num_threads = 1; // by default just keep 1 thread for search
    
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
        } else if (arg == "--query_file" && i + 1 < argc) {
            query_file = argv[++i];
        } else if (arg == "--gt_file" && i + 1 < argc) {
            gt_file = argv[++i];
        } else if (arg == "-K" && i + 1 < argc) {
            K = std::stoul(argv[++i]);
        } else if (arg == "-L") {
            // Parse multiple L values
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                L_values.push_back(std::stoul(argv[++i]));
            }
        } else if (arg == "--num_threads" || arg == "-T" && i + 1 < argc) {
            num_threads = std::stoul(argv[++i]);

        } else if (arg == "--result_path" && i + 1 < argc) {
            result_path = argv[++i];
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
    
    if (data_path.empty() || index_path_prefix.empty() || query_file.empty() || K == 0 || L_values.empty() || result_path.empty()) {
        std::cout << "Error: Missing required arguments" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    try {
        std::cout << "=== SEARCH VAMANA INDEX ===" << std::endl;
        std::cout << "Data type: " << data_type << std::endl;
        std::cout << "Distance function: " << dist_fn << std::endl;
        std::cout << "Data path: " << data_path << std::endl;
        std::cout << "Index prefix: " << index_path_prefix << std::endl;
        std::cout << "Query file: " << query_file << std::endl;
        std::cout << "Ground truth file: " << gt_file << std::endl;
        std::cout << "K: " << K << std::endl;
        std::cout << "L values: ";
        for (uint32_t L : L_values) {
            std::cout << L << " ";
        }
        std::cout << std::endl;
        std::cout << "Result path: " << result_path << std::endl;
        
        // Load queries
        float* query_data = nullptr;
        uint32_t num_queries, query_dim;
        vamana::io::load_fbin(query_file, query_data, num_queries, query_dim);
        
        // Load ground truth (optional)
        uint32_t* ground_truth_data = nullptr;
        uint32_t num_gt_queries = 0, gt_k = 0;
        if (!gt_file.empty()) {
            try {
                vamana::io::load_ibin(gt_file, ground_truth_data, num_gt_queries, gt_k);
            } catch (const std::exception& e) {
                std::cout << "Warning: Could not load ground truth: " << e.what() << std::endl;
            }
        }
        
        // Load index and data for search
        VamanaIndex index(query_dim, 32, 64, 1.2f, 500);  // Default params, will be overwritten by load
        
        // Determine number of threads for parallel batch search
        #ifdef _OPENMP
        if (num_threads == 0) {
            num_threads = omp_get_max_threads();
        }
        #else
        num_threads = 1;
        #endif
        
        // Load index and initialize search scratch spaces
        index.load_index(index_path_prefix, num_threads);
        
        // Load the original data for distance calculations
        float* base_data = nullptr;
        uint32_t num_base, base_dim;
        vamana::io::load_fbin(data_path, base_data, num_base, base_dim);
        
        if (base_dim != query_dim) {
            std::cerr << "Error: Base data dimension (" << base_dim << ") doesn't match query dimension (" << query_dim << ")" << std::endl;
            return 1;
        }
        
        // Set the data in the index for distance calculations
        index.set_data(base_data, num_base);
        
        // Create result output
        std::ofstream results_file(result_path);
        results_file << "L,QPS,Recall@" << K << std::endl;
        
        std::cout << "\nRunning searches with different L values..." << std::endl;
        std::cout << "L\tQPS\tRecall@" << K << std::endl;
        
        for (uint32_t L : L_values) {
            SearchResult result = search_with_L(index, query_data, num_queries, query_dim, 
                                              K, L, ground_truth_data, gt_k, num_threads);
            
            std::cout << L << "\t" << result.qps << "\t" << result.average_recall << std::endl;
            results_file << L << "," << result.qps << "," << result.average_recall << std::endl;
        }
        
        results_file.close();
        
        // Clean up
        vamana::io::free_data(query_data);
        vamana::io::free_data(base_data);
        if (ground_truth_data) {
            std::free(ground_truth_data);
        }
        
        std::cout << "Search completed successfully! Results saved to " << result_path << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}