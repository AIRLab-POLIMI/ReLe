#Emotions learning tests
add_executable(emotions Emotions.cpp CompressedPolicy.cpp)
target_link_libraries(emotions releROS ${catkin_LIBRARIES})

add_executable(emotions_classification EmotionsClassification.cpp)
target_link_libraries(emotions_classification releROS ${catkin_LIBRARIES})

add_executable(emotions_performances EmotionPerformances.cpp)
target_link_libraries(emotions_performances releROS ${catkin_LIBRARIES})

add_executable(emotions_features EmotionsFeatures.cpp)
target_link_libraries(emotions_features releROS ${catkin_LIBRARIES})