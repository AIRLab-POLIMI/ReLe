cmake_minimum_required(VERSION 2.8.3)

set(CATKIN_MSG FALSE)
find_package(catkin QUIET)
project(rele)

if(catkin_FOUND)
    catkin_package(INCLUDE_DIRS include
        LIBRARIES rele)
    set(CATKIN_MSG TRUE)
else()
    set(CATKIN_MSG FALSE)
endif() 

#----------------------- CMAKE MODULES ------------------------

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# ----------------------- INCLUDES ----------------------------

include_directories(include)

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
    find_package(Boost 1.53 REQUIRED system timer)
    if(Boost_FOUND)
        include_directories(${Boost_INCLUDE_DIRS})
    endif()
endif()
include_directories(${ARMADILLO_INCLUDE_DIR})

# ----------------------- GCC FLAGS ----------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")


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

message(STATUS "Using catkin?    : " ${CATKIN_MSG})
message(STATUS "Build type       : " ${BUILD_TYPE_MSG})

# ------------------------ BUILD -------------------------------

file(GLOB_RECURSE ReLe_SOURCE
    src/*.cpp)

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

add_subdirectory(test)

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

else()
    # Non-catkin installation
    # Install libraries
    install(TARGETS rele
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        )
    # Install headers
    install(DIRECTORY include/rele
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include
        )

endif()

# ------------------------ DOCUMENTATION ------------------------
add_subdirectory(doc EXCLUDE_FROM_ALL)




