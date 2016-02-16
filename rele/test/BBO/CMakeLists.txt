#BBO test
add_executable(deep_BBO DeepBBOTest.cpp)
target_link_libraries(deep_BBO rele)

add_executable(portfolio_BBO PortfolioBBOTest.cpp)
target_link_libraries(portfolio_BBO rele)

add_executable(lqr_BBO LqrBBOTest.cpp)
target_link_libraries(lqr_BBO rele)

add_executable(nls_BBO NlsBBOTest.cpp)
target_link_libraries(nls_BBO rele)

add_executable(segway_BBO SegwayBBOTest.cpp)
target_link_libraries(segway_BBO rele)

add_executable(ucp_BBO UnicyclePolarBBOTest.cpp)
target_link_libraries(ucp_BBO rele)

add_executable(mh_BBO MultiHeatBBOTest.cpp)
target_link_libraries(mh_BBO rele)