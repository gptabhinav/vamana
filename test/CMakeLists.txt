# Test CMakeLists.txt

# Common test sources (the actual implementation)
set(VAMANA_SOURCES
    ../src/core/neighbor.cpp
    ../src/core/distance.cpp  
    ../src/core/scratch.cpp
    ../src/core/graph.cpp
    ../src/core/index.cpp
    ../src/core/io.cpp
)

# Individual Unit Test Executables
add_executable(test_neighbor
    test_neighbor.cpp
    ../src/core/neighbor.cpp
)

add_executable(test_distance
    test_distance.cpp
    ../src/core/distance.cpp
)

add_executable(test_graph
    test_graph.cpp
    ../src/core/graph.cpp
)

add_executable(test_scratch
    test_scratch.cpp
    ../src/core/scratch.cpp
)

add_executable(test_index
    test_index.cpp
    ${VAMANA_SOURCES}
)

# Set common properties for all unit tests
foreach(test_target test_neighbor test_distance test_graph test_scratch test_index)
    target_include_directories(${test_target} PRIVATE ../include)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${test_target} OpenMP::OpenMP_CXX)
    endif()
endforeach()

# Integration tests
add_executable(integration_tests
    integration_tests.cpp
    ${VAMANA_SOURCES}
)

target_include_directories(integration_tests PRIVATE ../include)

if(OpenMP_CXX_FOUND)
    target_link_libraries(integration_tests OpenMP::OpenMP_CXX)
endif()

# Add tests to CTest
add_test(NAME TestNeighbor COMMAND test_neighbor)
add_test(NAME TestDistance COMMAND test_distance)
add_test(NAME TestGraph COMMAND test_graph)
add_test(NAME TestScratch COMMAND test_scratch)
add_test(NAME TestIndex COMMAND test_index)
add_test(NAME IntegrationTests COMMAND integration_tests)