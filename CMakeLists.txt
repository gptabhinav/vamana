cmake_minimum_required(VERSION 3.12)
project(Vamana)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mavx2 -mfma")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -march=native -mavx2 -mfma")
# Add AVX2 flags for all builds
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mavx2 -mfma")

# Try to find OpenMP, but make it optional for now
find_package(OpenMP)

# Core library sources
set(VAMANA_CORE_SOURCES
    src/core/neighbor.cpp
    src/core/distance.cpp  
    src/core/scratch.cpp
    src/core/graph.cpp
    src/core/index.cpp
    src/core/io.cpp
)

# Create directories for organized output
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/apps)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/apps/utils)

# Main applications
add_executable(build_memory_index
    apps/build_memory_index.cpp
    ${VAMANA_CORE_SOURCES}
)
target_include_directories(build_memory_index PRIVATE include)
set_target_properties(build_memory_index PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(build_memory_index OpenMP::OpenMP_CXX)
endif()

add_executable(search_memory_index
    apps/search_memory_index.cpp
    ${VAMANA_CORE_SOURCES}
)
target_include_directories(search_memory_index PRIVATE include)
set_target_properties(search_memory_index PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(search_memory_index OpenMP::OpenMP_CXX)
endif()

# Utility applications
add_executable(fvecs_to_fbin
    apps/utils/fvecs_to_fbin.cpp
    ${VAMANA_CORE_SOURCES}
)
target_include_directories(fvecs_to_fbin PRIVATE include)
set_target_properties(fvecs_to_fbin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps/utils
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fvecs_to_fbin OpenMP::OpenMP_CXX)
endif()

add_executable(ivecs_to_ibin
    apps/utils/ivecs_to_ibin.cpp
    ${VAMANA_CORE_SOURCES}
)
target_include_directories(ivecs_to_ibin PRIVATE include)
set_target_properties(ivecs_to_ibin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps/utils
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ivecs_to_ibin OpenMP::OpenMP_CXX)
endif()

add_executable(compute_groundtruth
    apps/utils/compute_groundtruth.cpp
    ${VAMANA_CORE_SOURCES}
)
target_include_directories(compute_groundtruth PRIVATE include)
set_target_properties(compute_groundtruth PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps/utils
)
if(OpenMP_CXX_FOUND)
    target_link_libraries(compute_groundtruth OpenMP::OpenMP_CXX)
endif()

# Enable testing
enable_testing()

# Add tests subdirectory
add_subdirectory(test)