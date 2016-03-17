#IRL tests

#MWAL
add_executable(simple_chain_MWAL MWAL/SimpleChainMWAL.cpp)
target_link_libraries(simple_chain_MWAL rele)

add_executable(lqr_MWAL MWAL/LQRMWAL.cpp)
target_link_libraries(lqr_MWAL rele)

# SMART H2O
add_executable(mleH2O_test SmartH2O/mleH2OTest.cpp)
target_link_libraries(mleH2O_test rele)

# step based linear
find_package(Boost REQUIRED COMPONENTS  program_options)

add_executable(lqr_print step_based_linear/LQRPrintTest.cpp)
target_link_libraries(lqr_print rele)

add_executable(lqr_LinearIRL step_based_linear/LQRTest.cpp)
target_link_libraries(lqr_LinearIRL rele ${Boost_LIBRARIES})

add_executable(nls_LinearIRL step_based_linear/NLSTest.cpp)
target_link_libraries(nls_LinearIRL rele ${Boost_LIBRARIES})

add_executable(gaussian_LinearIRL step_based_linear/GaussianTest.cpp)
target_link_libraries(gaussian_LinearIRL rele ${Boost_LIBRARIES})

add_executable(ship_LinearIRL step_based_linear/ShipSteeringTest.cpp)
target_link_libraries(ship_LinearIRL rele ${Boost_LIBRARIES})

add_executable(gradientIRL_test step_based_linear/GradientIRLTest.cpp)
target_link_libraries(gradientIRL_test rele)

add_executable(hessianIRL_test step_based_linear/HessianIRLTest.cpp)
target_link_libraries(hessianIRL_test rele)

# episode based linear
add_executable(lqr_EMIRL episode_based_linear/EMIRL_LQR.cpp)
target_link_libraries(lqr_EMIRL rele)

add_executable(nls_EMIRL episode_based_linear/EMIRL_NLS.cpp)
target_link_libraries(nls_EMIRL rele)

add_executable(ship_EMIRL episode_based_linear/EMIRL_ShipSteering.cpp)
target_link_libraries(ship_EMIRL rele)
