#pragma once

#include "vamana/core/types.h"
#include <vector>
#include <string>

class Graph {
private:
    std::vector<std::vector<location_t>> adj_list;
    
public:
    // Constructor
    Graph(size_t num_nodes = 0);
    
    // Basic graph operations
    void resize(size_t num_nodes);
    void add_edge(location_t from, location_t to);
    void set_neighbors(location_t node, const std::vector<location_t>& neighbors);
    const std::vector<location_t>& get_neighbors(location_t node) const;
    
    // Graph properties
    size_t size() const { return adj_list.size(); }
    size_t degree(location_t node) const;
    
    // I/O operations
    void save(const std::string& filename) const;
    void load(const std::string& filename);
    
    // Utility
    void clear();
};
