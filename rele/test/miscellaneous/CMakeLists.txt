#Matlab tests (miscellaneous)
add_executable(chol_FIM CholeskyFIMTest.cpp)
target_link_libraries(chol_FIM rele)

add_executable(mle_Policy MLEPolicyTest.cpp)
target_link_libraries(mle_Policy rele)

add_executable(range_test ModularRangeTest.cpp)
target_link_libraries(range_test rele)