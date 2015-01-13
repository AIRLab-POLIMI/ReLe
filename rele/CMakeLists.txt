cmake_minimum_required(VERSION 2.8.3)

set(CATKIN_MSG FALSE)
find_package(catkin QUIET)
project(rele)

if(catkin_FOUND)
	catkin_package()
	set(CATKIN_MSG TRUE)
else()
	set(CATKIN_MSG FALSE)
endif()

# ----------------------- INCLUDES ----------------------------

include_directories(include/core
                    include/utils
                    include/algorithms)

# ----------------------- LIBRARIES ---------------------------

find_package(Boost 1.46 REQUIRED COMPONENTS random)
find_package(Armadillo REQUIRED)
  

# ----------------------- GCC FLAGS ----------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


if(CMAKE_BUILD_TYPE MATCHES Debug)	
	set(BUILD_TYPE_MSG "Debug") 

else()	
	set(BUILD_TYPE_MSG "Release")
endif()

# ------------------------ MESSAGES ----------------------------

message(STATUS "Using catkin?    : " ${CATKIN_MSG})
message(STATUS "Build type       : " ${BUILD_TYPE_MSG})


# ------------------------ BUILD -------------------------------

file(GLOB_RECURSE ReLe_SOURCE src/core/*.cpp src/algorithms/*.cpp src/utils/*.cpp)

add_library(rele ${ReLe_SOURCE})

add_executable(simple_chain src/test/SimpleChain.cpp)
target_link_libraries(simple_chain rele)


# ------------------------ INSTALLATION ------------------------

## Mark executables and/or libraries for installation
#install(TARGETS rele
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

# --------------------------------------------------------------


