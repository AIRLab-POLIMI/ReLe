#Solvers tests
add_executable(exact_plot lqr/ExactPlot.cpp)
target_link_libraries(exact_plot rele)

add_executable(lqr_solver_test lqr/LQRsolverTest.cpp)
target_link_libraries(lqr_solver_test rele)

add_executable(lqr_exact_test lqr/LQRExactTest.cpp)
target_link_libraries(lqr_exact_test rele)

add_executable(lqr_joint_test lqr/LQRjointTest.cpp)
target_link_libraries(lqr_joint_test rele)

add_executable(dynamic_programming_test dynamic_programming/DynamicProgrammingTest.cpp)
target_link_libraries(dynamic_programming_test rele)