#pragma once
// we use types.h to standardize types used throughout the codebase
// or define constants that are being used everywhere in vamana
// like for example SIMD_ALIGNMENT, or our default R value, etc

#include <cstdint>

// defining location and distance types
typedef std::uint32_t location_t;   // location ids are always non-negative and we dont expect more than 4 billion points for this implementation (can change to uint64_t if needed)
typedef float distance_t;           // we dont need to worry about float not being 32 bit, its guaranteed by the standard in most modern systems

// defining SIMD alignment constant
constexpr std::uint32_t SIMD_ALIGNMENT = 32; // AVX2 requires 32-byte alignment

// defining default VAMANA parameters
constexpr std::uint32_t DEFAULT_R = 32;     // default number of neighbors per node
constexpr std::uint32_t DEFAULT_L = 100;    // default candidate list size during search
constexpr float DEFAULT_ALPHA = 1.2f;       // default alpha parameter
constexpr std::uint32_t DEFAULT_MAXC = 750; // default max candidates during construction

// Graph constants, dont really understand them right now, will update later
constexpr float GRAPH_SLACK_FACTOR = 1.3f;
constexpr float OVERHEAD_FACTOR = 1.1f;
constexpr bool DEFAULT_SATURATE_GRAPH = false;