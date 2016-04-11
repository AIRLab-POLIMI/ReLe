#Core test
add_executable(basis_test BasisTest.cpp)
target_link_libraries(basis_test rele)

add_executable(features_test FeaturesTest.cpp)
target_link_libraries(features_test rele)

add_executable(tiles_test TilesTest.cpp)
target_link_libraries(tiles_test rele)

add_executable(pr_mdp ParametricRewardMDPTest.cpp)
target_link_libraries(pr_mdp rele)

add_executable(dataset_test DatasetTest.cpp)
target_link_libraries(dataset_test rele)

add_executable(nn_test NeuralNetworksTest.cpp)
target_link_libraries(nn_test rele)

add_executable(ae_test AutoencoderTest.cpp)
target_link_libraries(ae_test rele)

add_executable(trees_test TreesTest.cpp)
target_link_libraries(trees_test rele)

add_executable(gaussian_test GaussianMixtureModelsTest.cpp)
target_link_libraries(gaussian_test rele)

add_executable(gp_test GaussianProcessTest.cpp)
target_link_libraries(gp_test rele)

add_executable(em_test ExpectationMaximizationTest.cpp)
target_link_libraries(em_test rele)

add_executable(bi_test BayesianInferenceTest.cpp)
target_link_libraries(bi_test rele)

add_executable(nene_test NearestNeighbourTest.cpp)
target_link_libraries(nene_test rele)