#Core tests
add_executable(rrbot_test RRBotTest.cpp)
target_link_libraries(rrbot_test releROS ${catkin_LIBRARIES})