#IRL tests
add_executable(pfa_test PFA/PFATest.cpp)
target_link_libraries(pfa_test rele)

add_executable(simple_chain_MWAL MWAL/SimpleChainMWAL.cpp)
target_link_libraries(simple_chain_MWAL rele)

add_executable(lqr_MWAL MWAL/LQRMWAL.cpp)
target_link_libraries(lqr_MWAL rele)

add_executable(lqr_GIRL GIRL/GIRL_LQR.cpp)
target_link_libraries(lqr_GIRL rele)

add_executable(nls_GIRL GIRL/GIRL_NLS.cpp)
target_link_libraries(nls_GIRL rele)

add_executable(ship_GIRL GIRL/GIRL_ShipSteering.cpp)
target_link_libraries(ship_GIRL rele)

add_executable(gaussian_GIRL GIRL/GIRL_Gaussian.cpp)
target_link_libraries(gaussian_GIRL rele)

add_executable(nls_PGIRL PGIRL/PGIRL_NLS.cpp)
target_link_libraries(nls_PGIRL rele)

add_executable(simple_PGIRL PGIRL/SimpleTest.cpp)
target_link_libraries(simple_PGIRL rele)

add_executable(lqr_MGIRL GIRL/MGIRL_LQR.cpp)
target_link_libraries(lqr_MGIRL rele)

add_executable(lqr_EMIRL EMIRL/EMIRL_LQR.cpp)
target_link_libraries(lqr_EMIRL rele)

add_executable(nls_EMIRL EMIRL/EMIRL_NLS.cpp)
target_link_libraries(nls_EMIRL rele)

add_executable(ship_EMIRL EMIRL/EMIRL_ShipSteering.cpp)
target_link_libraries(ship_EMIRL rele)

add_executable(gmm_features_test GMM/GMMClusteringTest.cpp)
target_link_libraries(gmm_features_test rele)

add_executable(mleH2O_test SmartH2O/mleH2OTest.cpp)
target_link_libraries(mleH2O_test rele)