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
    #GSL
    find_package(GSL QUIET)
    if(${GSL_FOUND})
        include_directories(${GSL_INCLUDE_DIRS})
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
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wsuggest-override")
    endif()
endif()
add_definitions( -DARMA_DONT_PRINT_ERRORS )

# ------------------------ MESSAGES ----------------------------

message(STATUS "Using catkin?    : " ${CATKIN_MSG})
message(STATUS "Build type       : " ${BUILD_TYPE_MSG})

# ------------------------ BUILD -------------------------------

file(GLOB_RECURSE ReLe_SOURCE
    src/*.cpp)

if(NOT ${GSL_FOUND})
    list(REMOVE_ITEM ReLe_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/algorithms/td/WQ-Learning.cpp)
endif()

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

add_custom_target(doc
    ${CMAKE_CURRENT_SOURCE_DIR}/doc/build.py ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Building html documentation with Doxygen, Sphinx and Breathe")

## For building Documentation (uses Sphinx)
## REFER TO https://github.com/akheron/jansson/tree/master/doc
#option(BUILD_DOCS "Build documentation (uses python-sphinx)." ON)
#if (BUILD_DOCS)
#    find_package(Doxygen)
#    if (NOT DOXYGEN_FOUND)
#        message(WARNING "Doxygen not found. Cannot generate documentation!
#            Set -DBUILD_DOCS=OFF to get rid of this message.")

#    else()

#        add_custom_target(doxygen
#            ${DOXYGEN_EXECUTABLE}

#            )


#        find_package(Sphinx)

#        if (NOT SPHINX_FOUND)
#            message(WARNING "Sphinx not found. Cannot generate documentation!
#                Set -DBUILD_DOCS=OFF to get rid of this message.")
#        else()
#            if (Sphinx_VERSION_STRING VERSION_LESS 1.0)
#                message(WARNING "Your Sphinx version is too old!
#                    This project requires Sphinx v1.0 or above to produce
#                    proper documentation (you have v${Sphinx_VERSION_STRING}).
#                    You will get output but it will have errors.")
#            endif()

#            # configured documentation tools and intermediate build results
#            set(BINARY_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/_build")

#            # Sphinx cache with pickled ReST documents
#            set(SPHINX_CACHE_DIR "${CMAKE_CURRENT_BINARY_DIR}/_doctrees")

#            # CMake could be used to build the conf.py file too,
#            # eg it could automatically write the version of the program or change the theme.
#            # if(NOT DEFINED SPHINX_THEME)
#            #    set(SPHINX_THEME default)
#            # endif()
#            #
#            # if(NOT DEFINED SPHINX_THEME_DIR)
#            #    set(SPHINX_THEME_DIR)
#            # endif()
#            #
#            # configure_file(
#            #    "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
#            #    "${BINARY_BUILD_DIR}/conf.py"
#            #    @ONLY)

#            # TODO: Add support for all sphinx builders: http://sphinx-doc.org/builders.html

#            # Add documentation targets.
#            set(DOC_TARGETS html)

#            #      option(RELE_BUILD_MAN "Create a target for building man pages." ON)

#            #      if (RELE_BUILD_MAN)
#            #         if (Sphinx_VERSION_STRING VERSION_LESS 1.0)
#            #            message(WARNING "Sphinx version 1.0 > is required to build man pages. You have v${Sphinx_VERSION_STRING}.")
#            #         else()
#            #            list(APPEND DOC_TARGETS man)
#            #         endif()
#            #      endif()

#            #      option(RELE_BUILD_LATEX "Create a target for building latex docs (to create PDF)." OFF)

#            #      if (RELE_BUILD_LATEX)
#            #         find_package(LATEX)

#            #         if (NOT LATEX_COMPILER)
#            #            message("Couldn't find Latex, can't build latex docs using Sphinx")
#            #         else()
#            #            message("Latex found! If you have problems building, see Sphinx documentation for required Latex packages.")
#            #            list(APPEND DOC_TARGETS latex)
#            #         endif()
#            #      endif()

#            # The doc target will build all documentation targets.
#            add_custom_target(doc)

#            foreach (DOC_TARGET ${DOC_TARGETS})
#                add_custom_target(${DOC_TARGET}
#                    ${SPHINX_EXECUTABLE}
#                    # -q   # Enable for quiet mode
#                    -b ${DOC_TARGET}
#                    -d "${SPHINX_CACHE_DIR}"
#                    # -c "${BINARY_BUILD_DIR}" # enable if using cmake-generated conf.py
#                    "${CMAKE_CURRENT_SOURCE_DIR}/doc/source"
#                    "${CMAKE_CURRENT_BINARY_DIR}/doc/${DOC_TARGET}"
#                    COMMAND
#                    COMMENT "Building ${DOC_TARGET} documentation with Sphinx")

#                add_dependencies(doc ${DOC_TARGET})
#            endforeach()

#            message("Building documentation enabled for: ${DOC_TARGETS}")
#        endif()
#    endif()
#endif ()

# --------------------------------------------------------------


