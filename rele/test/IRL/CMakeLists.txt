#IRL tests
find_package(Boost REQUIRED COMPONENTS  program_options)

add_executable(test_hessian_gradient TestGradientHess.cpp)
target_link_libraries(test_hessian_gradient rele)

#NIPS
add_executable(nips_car nips/Car.cpp)
target_link_libraries(nips_car rele)

add_executable(nips_ship nips/Ship.cpp)
target_link_libraries(nips_ship rele)

add_executable(nips_lqr_comp nips/LQR_comp.cpp)
target_link_libraries(nips_lqr_comp rele)

add_executable(nips_lqr_map nips/LQR_MAP.cpp)
target_link_libraries(nips_lqr_map rele)

add_executable(nips_lqr_map2 nips/LQR_MAP2.cpp)
target_link_libraries(nips_lqr_map2 rele)

add_executable(nips_lqr_mle nips/LQR_MLE.cpp)
target_link_libraries(nips_lqr_mle rele)

add_executable(nips_lqr nips/LQR.cpp)
target_link_libraries(nips_lqr rele)

add_executable(nips_nls nips/NLS.cpp)
target_link_libraries(nips_nls rele)

add_executable(nips_nls_fe nips/NLS_FE.cpp)
target_link_libraries(nips_nls_fe rele)

add_executable(nips_segway nips/Segway.cpp)
target_link_libraries(nips_segway rele)


#MWAL
add_executable(simple_chain_MWAL MWAL/SimpleChainMWAL.cpp)
target_link_libraries(simple_chain_MWAL rele)

add_executable(lqr_MWAL MWAL/LQRMWAL.cpp)
target_link_libraries(lqr_MWAL rele)

# SMART H2O
add_executable(mleH2O_test SmartH2O/mleH2OTest.cpp)
target_link_libraries(mleH2O_test rele)

# step based linear
add_executable(lqr_sblirl step_based_linear/LQRTest.cpp)
target_link_libraries(lqr_sblirl rele ${Boost_LIBRARIES})

add_executable(nls_sblirl step_based_linear/NLSTest.cpp)
target_link_libraries(nls_sblirl rele ${Boost_LIBRARIES})

add_executable(ship_sblirl step_based_linear/ShipSteeringTest.cpp)
target_link_libraries(ship_sblirl rele ${Boost_LIBRARIES})

add_executable(lqr_print step_based_linear/LQRPrintTest.cpp)
target_link_libraries(lqr_print rele)

add_executable(lqr_exact_print step_based_linear/LQRExactPrint.cpp)
target_link_libraries(lqr_exact_print rele)

add_executable(gradientIRL_test step_based_linear/GradientIRLTest.cpp)
target_link_libraries(gradientIRL_test rele)

add_executable(hessianIRL_test step_based_linear/HessianIRLTest.cpp)
target_link_libraries(hessianIRL_test rele)

# episode based linear
add_executable(lqr_eblirl episode_based_linear/LQRTest.cpp)
target_link_libraries(lqr_eblirl rele ${Boost_LIBRARIES})

add_executable(nls_eblirl episode_based_linear/NLSTest.cpp)
target_link_libraries(nls_eblirl rele ${Boost_LIBRARIES})

add_executable(ship_eblirl episode_based_linear/ShipSteeringTest.cpp)
target_link_libraries(ship_eblirl rele ${Boost_LIBRARIES})

# non linear
add_executable(gaussian_NoGIRL GaussianTestNoGIRL.cpp)
target_link_libraries(gaussian_NoGIRL rele)

# bayesian
add_executable(lqr_bayesian bayesian/LQRBayesianTest.cpp)
target_link_libraries(lqr_bayesian rele)

add_executable(lqr_approximate_bayesian bayesian/LQRApproximateBayesianTest.cpp)
target_link_libraries(lqr_approximate_bayesian rele)

add_executable(nls_bayesian bayesian/NLSBayesianTest.cpp)
target_link_libraries(nls_bayesian rele)

add_executable(lqr_bayesian_batch bayesian/BatchTest.cpp)
target_link_libraries(lqr_bayesian_batch rele)

# classification based
add_executable(car_cbased classification_based/Car.cpp)
target_link_libraries(car_cbased rele)



