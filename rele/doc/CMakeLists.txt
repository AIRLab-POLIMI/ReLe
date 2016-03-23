#Build tutorials
file(GLOB TUTORIAL_SOURCES tutorials/code/*.cpp)
foreach(tutorial_source ${TUTORIAL_SOURCES})
    get_filename_component(tutorial ${tutorial_source} NAME_WE)
    add_executable(${tutorial} ${tutorial_source})
    target_link_libraries(${tutorial} rele)
    set(TUTORIALS ${TUTORIALS} ${tutorial})
endforeach()

#Build documentation
add_custom_target(doc
    ${PROJECT_SOURCE_DIR}/doc/build.py ${PROJECT_SOURCE_DIR}
    DEPENDS ${TUTORIALS} 
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