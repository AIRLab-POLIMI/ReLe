#PG test
add_executable(nls_PG NlsPGTest.cpp)
target_link_libraries(nls_PG rele)

add_executable(lqr_PG LqrPGTest.cpp)
target_link_libraries(lqr_PG rele)

add_executable(portfolio_PG PortfolioPGTest.cpp)
target_link_libraries(portfolio_PG rele)

add_executable(deep_PG DeepPGTest.cpp)
target_link_libraries(deep_PG rele)

add_executable(dam_PG DamPGTest.cpp)
target_link_libraries(dam_PG rele)

add_executable(ship_PG ShipSteeringPGTest.cpp)
target_link_libraries(ship_PG rele)

add_executable(pursuer_PG PursuerPGTest.cpp)
target_link_libraries(pursuer_PG rele)

add_executable(taxi_PG TaxiFuelPGTest.cpp)
target_link_libraries(taxi_PG rele)
