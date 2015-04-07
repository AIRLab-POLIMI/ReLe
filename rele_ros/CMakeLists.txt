cmake_minimum_required(VERSION 2.8.3)


find_package(catkin REQUIRED COMPONENTS rele 
                                        roscpp 
                                        vrep_common)
project(rele_ros)

catkin_package(INCLUDE_DIRS include/rele_ros
               LIBRARIES releROS)


# ----------------------- INCLUDES ----------------------------

include_directories(include/rele_ros/core
                    include/rele_ros/envirorments
                    ${catkin_INCLUDE_DIRS})

# ----------------------- LIBRARIES ---------------------------

  

# ----------------------- GCC FLAGS ----------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


if(CMAKE_BUILD_TYPE MATCHES Debug)	
	set(BUILD_TYPE_MSG "Debug") 

else()	
	set(BUILD_TYPE_MSG "Release")
endif()

# ------------------------ MESSAGES ----------------------------

message(STATUS "Build type       : " ${BUILD_TYPE_MSG})


# ------------------------ BUILD -------------------------------

#releROS library
file(GLOB_RECURSE ReLeROS_SOURCE 
                  src/envirorments/*.cpp)
                  
add_library(releROS ${ReLeROS_SOURCE})
target_link_libraries(releROS ${catkin_LIBRARIES})
add_dependencies(releROS ${catkin_EXPORTED_TARGETS})


#simulator handlers
add_executable(basketbot_sim src/simulator_handlers/basketbot_sim.cpp)
target_link_libraries(basketbot_sim ${catkin_LIBRARIES})
add_dependencies(basketbot_sim ${catkin_EXPORTED_TARGETS})

#Test nodes
add_executable(basketbot src/test/BasketBotTest.cpp)
target_link_libraries(basketbot releROS ${catkin_LIBRARIES})
add_dependencies(basketbot ${catkin_EXPORTED_TARGETS})

add_executable(room_environment src/test/RoomEnvironmentTest.cpp)
target_link_libraries(room_environment releROS ${catkin_LIBRARIES})
add_dependencies(room_environment ${catkin_EXPORTED_TARGETS})


# ------------------------ INSTALLATION ------------------------

## Mark executables and/or libraries for installation
#install(TARGETS rele_ros
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

# --------------------------------------------------------------


