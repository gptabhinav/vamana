#include "vamana/core/graph.h"
#include <fstream>
#include <iostream>
#include <algorithm>

Graph::Graph(size_t num_nodes) {
    resize(num_nodes);
}

void Graph::resize(size_t num_nodes) {
    adj_list.resize(num_nodes);
}

void Graph::add_edge(location_t from, location_t to) {
    if (from >= adj_list.size()) {
        resize(from + 1);
    }
    
    // Avoid duplicates
    auto& neighbors = adj_list[from];
    if (std::find(neighbors.begin(), neighbors.end(), to) == neighbors.end()) {
        neighbors.push_back(to);
    }
}

void Graph::set_neighbors(location_t node, const std::vector<location_t>& neighbors) {
    if (node >= adj_list.size()) {
        resize(node + 1);
    }
    adj_list[node] = neighbors;
}

const std::vector<location_t>& Graph::get_neighbors(location_t node) const {
    if (node >= adj_list.size()) {
        static const std::vector<location_t> empty;
        return empty;
    }
    return adj_list[node];
}

// const at the end just means, that calling this function wont change modify anything
// inside the object. so kinda signifies, the operations being performed on the object
// would be read only
size_t Graph::degree(location_t node) const {
    if (node >= adj_list.size()) {
        return 0;
    }
    return adj_list[node].size();
}

void Graph::save(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for writing" << std::endl;
        return;
    }
    
    // Save number of nodes
    size_t num_nodes = adj_list.size();
    file.write(reinterpret_cast<const char*>(&num_nodes), sizeof(num_nodes));
    
    // Save adjacency list
    for (const auto& neighbors : adj_list) {
        size_t degree = neighbors.size();
        file.write(reinterpret_cast<const char*>(&degree), sizeof(degree));
        file.write(reinterpret_cast<const char*>(neighbors.data()), degree * sizeof(location_t));
    }
    
    file.close();
}

void Graph::load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << " for reading" << std::endl;
        return;
    }
    
    // Load number of nodes
    size_t num_nodes;
    file.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
    
    // Clear and resize
    adj_list.clear();
    adj_list.resize(num_nodes);
    
    // Load adjacency list
    for (size_t i = 0; i < num_nodes; i++) {
        size_t degree;
        file.read(reinterpret_cast<char*>(&degree), sizeof(degree));
        
        adj_list[i].resize(degree);
        file.read(reinterpret_cast<char*>(adj_list[i].data()), degree * sizeof(location_t));
    }
    
    file.close();
}

void Graph::clear() {
    for (auto& neighbors : adj_list) {
        neighbors.clear();
    }
}
