#Batch test

add_executable(deep_off DeepBatchTest.cpp)
target_link_libraries(deep_off rele)

add_executable(nls_off NlsBatchTest.cpp)
target_link_libraries(nls_off rele)

add_executable(portfolio_off PortfolioBatchTest.cpp)
target_link_libraries(portfolio_off rele)

add_executable(mc_LSPI McLSPITest.cpp)
target_link_libraries(mc_LSPI rele)
