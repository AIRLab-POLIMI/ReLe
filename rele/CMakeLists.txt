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
                                include/rele/statistics
	               LIBRARIES rele)
	set(CATKIN_MSG TRUE)
else()
	set(CATKIN_MSG FALSE)
endif()

#----------------------- CMAKE MODULES ------------------------

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# ----------------------- INCLUDES ----------------------------

include_directories(include/rele/core
                    include/rele/policy
                    include/rele/utils
                    include/rele/generators
                    include/rele/algorithms
                    include/rele/envirorments
                    include/rele/approximators
                    include/rele/statistics)

# ----------------------- LIBRARIES ---------------------------

#find_package(Boost REQUIRED COMPONENTS random)
if(WIN32)
	find_path(ARMADILLO_INCLUDE_DIR
		NAMES armadillo
		PATHS "C:/usr/include"
	)
	#include_directories("C:/usr/include")
else()
	find_package(Armadillo REQUIRED)
        find_package(NLopt REQUIRED)
        #Boost.Numeric.odeint used for integration (odeint)
        set(Boost_USE_STATIC_LIBS        OFF)
        set(Boost_USE_MULTITHREADED      ON)
        set(Boost_USE_STATIC_RUNTIME    OFF)
        find_package(Boost 1.53 REQUIRED)
        if(Boost_FOUND)
                include_directories(${Boost_INCLUDE_DIRS})
        endif()
endif()
include_directories(${ARMADILLO_INCLUDE_DIR})

# ----------------------- GCC FLAGS ----------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


if(CMAKE_BUILD_TYPE MATCHES Debug)	
	set(BUILD_TYPE_MSG "Debug") 

else()	
	set(BUILD_TYPE_MSG "Release")
endif()

add_definitions( -DARMA_DONT_PRINT_ERRORS )

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
                  src/approximators/*.cpp
                  src/statistics/*.cpp)

file(GLOB_RECURSE ReLe_INCLUDE
                  include/*.h)

add_library(rele ${ReLe_SOURCE} ${ReLe_INCLUDE})
if(WIN32)
	target_link_libraries(rele -L"C:/usr/bin" -L"C:/usr/lib" -larmadillo -lnlopt -lgfortran)
else()
	target_link_libraries(rele ${ARMADILLO_LIBRARIES}
                                   ${NLOPT_LIBRARY}
                                   ${Boost_LIBRARIES})
endif()

add_executable(simple_chain src/test/SimpleChain.cpp)
target_link_libraries(simple_chain rele)

add_executable(grid_world src/test/GridWorldTest.cpp)
target_link_libraries(grid_world rele)

add_executable(mountain_car src/test/MountainCarTest.cpp)
target_link_libraries(mountain_car rele)

add_executable(lqr_PGPE src/test/LqrPGPETest.cpp)
target_link_libraries(lqr_PGPE rele)

add_executable(lqr_NES src/test/LqrNESTest.cpp)
target_link_libraries(lqr_NES rele)

add_executable(lqr_xNES src/test/LqrxNESTest.cpp)
target_link_libraries(lqr_xNES rele)

add_executable(lqr_REPS src/test/LqrEpisodicREPSTest.cpp)
target_link_libraries(lqr_REPS rele)

add_executable(rocky_REPS src/test/RockyTest.cpp)
target_link_libraries(rocky_REPS rele)

add_executable(nls_BBO src/test/NlsBBOTest.cpp)
target_link_libraries(nls_BBO rele)

add_executable(nls_REPS src/test/NlsEpisodicREPSTest.cpp)
target_link_libraries(nls_REPS rele)

add_executable(basisMatrixTest src/test/BasisMatrixTest.cpp)
target_link_libraries(basisMatrixTest rele)

add_executable(nls_OptParams src/test/NlsOptimalParamsTest.cpp)
target_link_libraries(nls_OptParams rele)

add_executable(deep_BBO src/test/DeepBBOTest.cpp)
target_link_libraries(deep_BBO rele)
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


