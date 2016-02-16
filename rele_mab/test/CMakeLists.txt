# Test Makefile

if(${GSL_FOUND})
	add_executable(mab_q_learning MAB/MABQLearningTest.cpp)
	target_link_libraries(mab_q_learning rele_mab rele ${GSL_LIBRARIES})
endif()

#UCB1
add_executable(ucb1_test MAB/UCB1Test.cpp)
target_link_libraries(ucb1_test rele_mab rele)
