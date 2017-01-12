include(ARMA_FindMKL)
include(ARMA_FindACMLMP)
include(ARMA_FindACML)
include(ARMA_FindOpenBLAS)
include(ARMA_FindATLAS)
include(ARMA_FindBLAS)
include(ARMA_FindLAPACK)

message(STATUS "     MKL_FOUND = ${MKL_FOUND}"     )
message(STATUS "  ACMLMP_FOUND = ${ACMLMP_FOUND}"  )
message(STATUS "    ACML_FOUND = ${ACML_FOUND}"    )
message(STATUS "OpenBLAS_FOUND = ${OpenBLAS_FOUND}")
message(STATUS "   ATLAS_FOUND = ${ATLAS_FOUND}"   )
message(STATUS "    BLAS_FOUND = ${BLAS_FOUND}"    )
message(STATUS "  LAPACK_FOUND = ${LAPACK_FOUND}"  )

if(MKL_FOUND OR ACMLMP_FOUND OR ACML_FOUND)

  set(ARMA_USE_LAPACK true)
  set(ARMA_USE_BLAS   true)

  message(STATUS "")
  message(STATUS "*** If the MKL or ACML libraries are installed in non-standard locations such as")
  message(STATUS "*** /opt/intel/mkl, /opt/intel/composerxe/, /usr/local/intel/mkl")
  message(STATUS "*** make sure the run-time linker can find them.")
  message(STATUS "*** On Linux systems this can be done by editing /etc/ld.so.conf")
  message(STATUS "*** or modifying the LD_LIBRARY_PATH environment variable.")
  message(STATUS "")
  message(STATUS "*** On systems with SELinux enabled (eg. Fedora, RHEL),")
  message(STATUS "*** you may need to change the SELinux type of all MKL/ACML libraries")
  message(STATUS "*** to fix permission problems that may occur during run-time.")
  message(STATUS "*** See README.txt for more information")
  message(STATUS "")

  if(MKL_FOUND)
    set(LA_LIBS ${LA_LIBS} ${MKL_LIBRARIES})

    if(ACMLMP_FOUND OR ACML_FOUND)
      message(STATUS "*** Intel MKL as well as AMD ACML libraries were found.")
      message(STATUS "*** Using only the MKL library to avoid linking conflicts.")
      message(STATUS "*** If you wish to use ACML instead, please link manually with")
      message(STATUS "*** acml or acml_mp instead of the armadillo wrapper library.")
      message(STATUS "*** Alternatively, remove MKL from your system and rerun")
      message(STATUS "*** Armadillo's configuration using ./configure")
    endif()

  else()

    if(ACMLMP_FOUND)
      set(LA_LIBS ${LA_LIBS} ${ACMLMP_LIBRARIES})

      message(STATUS "*** Both single-core and multi-core ACML libraries were found.")
      message(STATUS "*** Using only the multi-core library to avoid linking conflicts.")
    else()
      if(ACML_FOUND)
        set(LA_LIBS ${LA_LIBS} ${ACML_LIBRARIES})
      endif()
    endif()

  endif()

else()

  if(OpenBLAS_FOUND AND ATLAS_FOUND)
    message(STATUS "")
    message(STATUS "*** WARNING: found both OpenBLAS and ATLAS; ATLAS will not be used")
  endif()

  if(OpenBLAS_FOUND AND BLAS_FOUND)
    message(STATUS "")
    message(STATUS "*** WARNING: found both OpenBLAS and BLAS; BLAS will not be used")
  endif()

  if(OpenBLAS_FOUND)

    set(ARMA_USE_BLAS true)
    set(LA_LIBS ${LA_LIBS} ${OpenBLAS_LIBRARIES})

    message(STATUS "")
    message(STATUS "*** If the OpenBLAS library is installed in")
    message(STATUS "*** /usr/local/lib or /usr/local/lib64")
    message(STATUS "*** make sure the run-time linker can find it.")
    message(STATUS "*** On Linux systems this can be done by editing /etc/ld.so.conf")
    message(STATUS "*** or modifying the LD_LIBRARY_PATH environment variable.")
    message(STATUS "")

  else()

    if(ATLAS_FOUND)
      set(ARMA_USE_ATLAS true)
      set(LA_ATLAS_INCLUDE_DIR ${ATLAS_INCLUDE_DIR})
      set(LA_LIBS ${LA_LIBS} ${ATLAS_LIBRARIES})

      message(STATUS "ATLAS_INCLUDE_DIR = ${ATLAS_INCLUDE_DIR}")
    endif()

    if(BLAS_FOUND)
      set(ARMA_USE_BLAS true)
      set(LA_LIBS ${LA_LIBS} ${BLAS_LIBRARIES})
    endif()

  endif()

  if(LAPACK_FOUND)
    set(ARMA_USE_LAPACK true)
    set(LA_LIBS ${LA_LIBS} ${LAPACK_LIBRARIES})
  endif()

endif()
