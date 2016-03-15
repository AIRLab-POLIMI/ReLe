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

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.7")
        message(FATAL_ERROR "Unsupported GCC/G++ version (>=4.7, it is better >=4.8)")
    elseif(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8")
        message(STATUS "C++11 is partially supported in this version, deactiving C++11 for armadillo")
        add_definitions(-DARMA_DONT_USE_CXX11)
        add_definitions(-DARMA_BLAS_LONG) #required for matlab compatibility
        message(STATUS "Added defs: -DARMA_DONT_USE_CXX11 -DARMA_BLAS_LONG")
    endif()

    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "5.0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wsuggest-override")
    endif()
endif()
add_definitions( -DARMA_DONT_PRINT_ERRORS )



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
