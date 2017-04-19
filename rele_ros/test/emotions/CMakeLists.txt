#Emotions learning tests
add_executable(emotions Emotions.cpp CompressedPolicy.cpp)
target_link_libraries(emotions releROS ${catkin_LIBRARIES})

add_executable(emotions_features EmotionsFeatures.cpp)
target_link_libraries(emotions_features releROS ${catkin_LIBRARIES})

add_executable(emotions_generator EmotionsGenerator.cpp)
target_link_libraries(emotions_generator releROS ${catkin_LIBRARIES})

add_executable(emotions_runner EmotionsRunner.cpp)
target_link_libraries(emotions_runner releROS ${catkin_LIBRARIES})