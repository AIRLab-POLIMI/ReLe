cmake_minimum_required(VERSION 2.8.3)


find_package(catkin REQUIRED COMPONENTS rele
                                        roscpp
                                        std_srvs)
project(rele_ros)

catkin_package(INCLUDE_DIRS include
               LIBRARIES releROS)


# ----------------------- INCLUDES ----------------------------

include_directories(include
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
                  src/environments/*.cpp)

add_library(releROS ${ReLeROS_SOURCE})
target_link_libraries(releROS ${catkin_LIBRARIES})
add_dependencies(releROS ${catkin_EXPORTED_TARGETS})

#releROS test

add_executable(rrbot_test test/RRBotTest.cpp)
target_link_libraries(rrbot_test releROS ${catkin_LIBRARIES})


# ------------------------ INSTALLATION ------------------------

## Mark executables and/or libraries for installation
install(TARGETS releROS
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# --------------------------------------------------------------
