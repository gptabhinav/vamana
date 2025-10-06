cmake_minimum_required(VERSION 3.12)
project(Vamana)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mavx2 -mfma")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Try to find OpenMP, but make it optional for now
find_package(OpenMP)

# Create the executable
add_executable(vamana
    src/main.cpp
    src/core/neighbor.cpp
    src/core/distance.cpp  
    src/core/scratch.cpp
    src/core/graph.cpp
    src/core/index.cpp
)

# Include directories
target_include_directories(vamana PRIVATE include)

# Link libraries if OpenMP is found
if(OpenMP_CXX_FOUND)
    target_link_libraries(vamana OpenMP::OpenMP_CXX)
endif()

# Enable testing
enable_testing()

# Add tests subdirectory
add_subdirectory(test)