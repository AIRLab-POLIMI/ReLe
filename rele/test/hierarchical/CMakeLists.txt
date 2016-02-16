#Hierarchical test
add_executable(taxi_hierarchical taxi/TaxiFuelTest.cpp
                                 taxi/TaxiOptions.cpp)
target_link_libraries(taxi_hierarchical rele)

add_executable(rocky_hierarchical Rocky/HierarchicalRockyTest.cpp
                                  Rocky/RockyOptions.cpp)
target_link_libraries(rocky_hierarchical rele)