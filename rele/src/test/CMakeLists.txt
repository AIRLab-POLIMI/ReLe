#Test Makefile

#nlopt
add_executable(nlopt_test nlopt.cpp)
target_link_libraries(nlopt_test  ${NLOPT_LIBRARY})

#Core test
add_executable(basis_test core/BasisTest.cpp)
target_link_libraries(basis_test rele)

add_executable(features_test core/FeaturesTest.cpp)
target_link_libraries(features_test rele)

add_executable(tiles_test core/TilesTest.cpp)
target_link_libraries(tiles_test rele)

add_executable(pr_mdp core/ParametricRewardMDPTest.cpp)
target_link_libraries(pr_mdp rele)

add_executable(dataset_test core/DatasetTest.cpp)
target_link_libraries(dataset_test rele)

add_executable(nn_test core/NeuralNetworksTest.cpp)
target_link_libraries(nn_test rele)

add_executable(trees_test core/TreesTest.cpp)
target_link_libraries(trees_test rele)

add_executable(gaussian_test core/GaussianMixtureModelsTest.cpp)
target_link_libraries(gaussian_test rele)

#Matlab tests (miscellaneous)
add_executable(chol_FIM miscellaneous/CholeskyFIMTest.cpp)
target_link_libraries(chol_FIM rele)

add_executable(dist2mat miscellaneous/Distribution2MatlabTest.cpp)
target_link_libraries(dist2mat rele)

add_executable(pol2mat miscellaneous/Policy2MatlabTest.cpp)
target_link_libraries(pol2mat rele)

add_executable(mle_Policy miscellaneous/MLEPolicyTest.cpp)
target_link_libraries(mle_Policy rele)

add_executable(rand_Policy miscellaneous/RandomPolicyTest.cpp)
target_link_libraries(rand_Policy rele)

add_executable(range_test miscellaneous/ModularRangeTest.cpp)
target_link_libraries(range_test rele)

#miscellaneous/policy_search tests
add_executable(datadiff2mat miscellaneous/policy_search/DataDiff2Matlab.cpp)
target_link_libraries(datadiff2mat rele)

#TD tests
add_executable(simple_chain TD/SimpleChain.cpp)
target_link_libraries(simple_chain rele)

add_executable(simple_chain_mean TD/SimpleChainMeanReward.cpp)
target_link_libraries(simple_chain_mean rele)

add_executable(grid_world TD/GridWorldTest.cpp)
target_link_libraries(grid_world rele)

add_executable(mountain_car TD/MountainCarTest.cpp)
target_link_libraries(mountain_car rele)

#IRL tests

add_executable(simple_chain_MWAL IRL/MWAL/SimpleChainMWAL.cpp)
target_link_libraries(simple_chain_MWAL rele)

add_executable(lqr_MWAL IRL/MWAL/LQRMWAL.cpp)
target_link_libraries(lqr_MWAL rele)

add_executable(lqr_GIRL IRL/GIRL/GIRL_LQR.cpp)
target_link_libraries(lqr_GIRL rele)

add_executable(nls_GIRL IRL/GIRL/GIRL_NLS.cpp)
target_link_libraries(nls_GIRL rele)

add_executable(ship_GIRL IRL/GIRL/GIRL_ShipSteering.cpp)
target_link_libraries(ship_GIRL rele)

add_executable(gaussian_GIRL IRL/GIRL/GIRL_Gaussian.cpp)
target_link_libraries(gaussian_GIRL rele)

add_executable(lqr_MGIRL IRL/GIRL/MGIRL_LQR.cpp)
target_link_libraries(lqr_MGIRL rele)


#add_executable(lqr_GIRL IRL/LQRGIRL.cpp)
#target_link_libraries(lqr_GIRL rele)

#add_executable(lqr_GMGIRL IRL/GMLQRGIRL.cpp)
#target_link_libraries(lqr_GMGIRL rele)

#add_executable(lqr_MLEGIRL IRL/MLELQRGIRL.cpp)
#target_link_libraries(lqr_MLEGIRL rele)

#add_executable(lqr_MLEGIRLALL IRL/MLELQRGIRLALL.cpp)
#target_link_libraries(lqr_MLEGIRLALL rele)

#add_executable(lqr_MLEGIRLALL1D IRL/MLELQRGIRLALL1D.cpp)
#target_link_libraries(lqr_MLEGIRLALL1D rele)

#add_executable(lqr_GIRLALL IRL/LQRGIRLALL.cpp)
#target_link_libraries(lqr_GIRLALL rele)

#add_executable(deep_GIRL IRL/DeepGIRL.cpp)
#target_link_libraries(deep_GIRL rele)

#add_executable(dam_GIRL IRL/DamGIRL.cpp)
#target_link_libraries(dam_GIRL rele)

#add_executable(mc_GIRL IRL/McGIRLTest.cpp)
#target_link_libraries(mc_GIRL rele)

#add_executable(hw_GIRL IRL/HwGIRLTest.cpp)
#target_link_libraries(hw_GIRL rele)

#Solvers tests
add_executable(lqr_solver_test solvers/LQRsolverTest.cpp)
target_link_libraries(lqr_solver_test rele)

add_executable(dynamic_programming_test solvers/dynamic_programming/DynamicProgrammingTest.cpp)
target_link_libraries(dynamic_programming_test rele)

#Rocky test
add_executable(rocky_REPS Rocky/RockyTest.cpp
                          Rocky/RockyPolicy.cpp)
target_link_libraries(rocky_REPS rele)

add_executable(rocky_hierarchical Rocky/HierarchicalRockyTest.cpp
                                  Rocky/RockyOptions.cpp)
target_link_libraries(rocky_hierarchical rele)

#Hierarchical test
add_executable(taxi_hierarchical hierarchical/taxi/TaxiFuelTest.cpp
                                 hierarchical/taxi/TaxiOptions.cpp)
target_link_libraries(taxi_hierarchical rele)


#REPS test
add_executable(lqr_REPS REPS/LqrREPSTest.cpp)
target_link_libraries(lqr_REPS rele)

add_executable(nls_REPS REPS/NlsREPSTest.cpp)
target_link_libraries(nls_REPS rele)

#BBO test
add_executable(deep_BBO BBO/DeepBBOTest.cpp)
target_link_libraries(deep_BBO rele)

add_executable(portfolio_BBO BBO/PortfolioBBOTest.cpp)
target_link_libraries(portfolio_BBO rele)

add_executable(lqr_BBO BBO/LqrBBOTest.cpp)
target_link_libraries(lqr_BBO rele)

add_executable(nls_BBO BBO/NlsBBOTest.cpp)
target_link_libraries(nls_BBO rele)

add_executable(segway_BBO BBO/SegwayBBOTest.cpp)
target_link_libraries(segway_BBO rele)

add_executable(ucp_BBO BBO/UnicyclePolarBBOTest.cpp)
target_link_libraries(ucp_BBO rele)

add_executable(mh_BBO BBO/MultiHeatBBOTest.cpp)
target_link_libraries(mh_BBO rele)

#PG test
add_executable(nls_PG PG/NlsPGTest.cpp)
target_link_libraries(nls_PG rele)

add_executable(lqr_PG PG/LqrPGTest.cpp)
target_link_libraries(lqr_PG rele)

add_executable(portfolio_PG PG/PortfolioPGTest.cpp)
target_link_libraries(portfolio_PG rele)

add_executable(deep_PG PG/DeepPGTest.cpp)
target_link_libraries(deep_PG rele)

add_executable(dam_PG PG/DamPGTest.cpp)
target_link_libraries(dam_PG rele)

add_executable(ship_PG PG/ShipSteeringPGTest.cpp)
target_link_libraries(ship_PG rele)

add_executable(taxi_PG PG/TaxiFuelPGTest.cpp)
target_link_libraries(taxi_PG rele)

#Batch test
add_executable(deep_off Batch/DeepBatchTest.cpp)
target_link_libraries(deep_off rele)

add_executable(nls_off Batch/NlsBatchTest.cpp)
target_link_libraries(nls_off rele)

add_executable(mc_LSPI Batch/McLSPITest.cpp)
target_link_libraries(mc_LSPI rele)

#Others tests
add_executable(nls_OptParams Others/NlsOptimalParamsTest.cpp)
target_link_libraries(nls_OptParams rele)

#MultiHeat tests
add_executable(mh_test MultiHeat/MultiHeatTest.cpp)
target_link_libraries(mh_test rele)
