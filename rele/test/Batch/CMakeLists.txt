#Batch test

add_executable(deep_off DeepBatchTest.cpp)
target_link_libraries(deep_off rele)

add_executable(nls_off NlsBatchTest.cpp)
target_link_libraries(nls_off rele)

add_executable(portfolio_off PortfolioBatchTest.cpp)
target_link_libraries(portfolio_off rele)

add_executable(car_batch CarOnHillBatchTest.cpp)
target_link_libraries(car_batch rele)

add_executable(acro_batch AcrobotBatchTest.cpp)
target_link_libraries(acro_batch rele)