cmake_minimum_required(VERSION 2.8.3)

set(CATKIN_MSG FALSE)
find_package(catkin QUIET)
project(rele)

if(catkin_FOUND)
	catkin_package(INCLUDE_DIRS include
	                            include/rele/core
	                            include/rele/policy
                                include/rele/utils
                                include/rele/generators
                                include/rele/algorithms
                                include/rele/envirorments
                                include/rele/approximators
	               LIBRARIES rele)
	set(CATKIN_MSG TRUE)
else()
	set(CATKIN_MSG FALSE)
endif()

# ----------------------- INCLUDES ----------------------------

include_directories(include/rele/core
                    include/rele/policy
                    include/rele/utils
                    include/rele/generators
                    include/rele/algorithms
                    include/rele/envirorments
                    include/rele/approximators)

# ----------------------- LIBRARIES ---------------------------

#find_package(Boost REQUIRED COMPONENTS random)
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

file(GLOB_RECURSE ReLe_SOURCE 
                  src/core/*.cpp 
                  src/policy/*.cpp
                  src/algorithms/*.cpp
                  src/envirorments/*.cpp  
                  src/utils/*.cpp
                  src/generators/*.cpp
                  src/approximators/*.cpp)

add_library(rele ${ReLe_SOURCE})
target_link_libraries(rele ${ARMADILLO_LIBRARIES})

add_executable(simple_chain src/test/SimpleChain.cpp)
target_link_libraries(simple_chain rele)

add_executable(grid_world src/test/GridWorldTest.cpp)
target_link_libraries(grid_world rele)

add_executable(mountain_car src/test/MountainCarTest.cpp)
target_link_libraries(mountain_car rele)

# ------------------------ INSTALLATION ------------------------

if(catkin_FOUND)
	# Mark executables and/or libraries for installation
	install(TARGETS rele
	ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
	RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
	)

	# Install headers
	install(DIRECTORY include/rele
	DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
	PATTERN ".svn" EXCLUDE
	)
endif()

# --------------------------------------------------------------


