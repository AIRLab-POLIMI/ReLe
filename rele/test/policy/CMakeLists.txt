#Policy tests
add_executable(rand_Policy RandomPolicyTest.cpp)
target_link_libraries(rand_Policy rele)

add_executable(mvn_Policy GenericNormalPolicyTest.cpp)
target_link_libraries(mvn_Policy rele)

add_executable(gibbs_Policy GenericParametricGibbsPolicyTest.cpp)
target_link_libraries(gibbs_Policy rele)
