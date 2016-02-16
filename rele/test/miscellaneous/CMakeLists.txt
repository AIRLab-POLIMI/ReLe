#Matlab tests (miscellaneous)
add_executable(chol_FIM CholeskyFIMTest.cpp)
target_link_libraries(chol_FIM rele)

add_executable(dist2mat Distribution2MatlabTest.cpp)
target_link_libraries(dist2mat rele)

add_executable(pol2mat Policy2MatlabTest.cpp)
target_link_libraries(pol2mat rele)

add_executable(mle_Policy MLEPolicyTest.cpp)
target_link_libraries(mle_Policy rele)

add_executable(range_test ModularRangeTest.cpp)
target_link_libraries(range_test rele)

#miscellaneous/policy_search tests
add_executable(datadiff2mat policy_search/DataDiff2Matlab.cpp)
target_link_libraries(datadiff2mat rele)