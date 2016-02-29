#IRL tests

#MWAL
add_executable(simple_chain_MWAL MWAL/SimpleChainMWAL.cpp)
target_link_libraries(simple_chain_MWAL rele)

add_executable(lqr_MWAL MWAL/LQRMWAL.cpp)
target_link_libraries(lqr_MWAL rele)

#EMIRL
add_executable(lqr_EMIRL EMIRL/EMIRL_LQR.cpp)
target_link_libraries(lqr_EMIRL rele)

add_executable(nls_EMIRL EMIRL/EMIRL_NLS.cpp)
target_link_libraries(nls_EMIRL rele)

add_executable(ship_EMIRL EMIRL/EMIRL_ShipSteering.cpp)
target_link_libraries(ship_EMIRL rele)

# SMART H2O
add_executable(mleH2O_test SmartH2O/mleH2OTest.cpp)
target_link_libraries(mleH2O_test rele)

# gradient
find_package(Boost REQUIRED COMPONENTS  program_options)

add_executable(lqr_print gradient/LQRPrintTest.cpp)
target_link_libraries(lqr_print rele)

add_executable(lqr_LinearIRL gradient/LQRTest.cpp)
target_link_libraries(lqr_LinearIRL rele ${Boost_LIBRARIES})

add_executable(nls_LinearIRL gradient/NLSTest.cpp)
target_link_libraries(nls_LinearIRL rele ${Boost_LIBRARIES})

add_executable(gaussian_LinearIRL gradient/GaussianTest.cpp)
target_link_libraries(gaussian_LinearIRL rele ${Boost_LIBRARIES})

add_executable(ship_LinearIRL gradient/ShipSteeringTest.cpp)
target_link_libraries(ship_LinearIRL rele ${Boost_LIBRARIES})

add_executable(gradientIRL_test gradient/GradientIRLTest.cpp)
target_link_libraries(gradientIRL_test rele)

add_executable(hessianIRL_test gradient/HessianIRLTest.cpp)
target_link_libraries(hessianIRL_test rele)
