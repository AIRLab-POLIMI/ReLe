#Emotions learning tests
add_executable(emotions Emotions.cpp CompressedPolicy.cpp)
target_link_libraries(emotions releROS ${catkin_LIBRARIES})