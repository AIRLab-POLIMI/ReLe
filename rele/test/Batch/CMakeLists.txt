#Batch test

#add_executable(deep_off DeepBatchTest.cpp)
#target_link_libraries(deep_off rele)

if(${GSL_FOUND})
	add_executable(gridFQI2 GridFQITest2.cpp)
	target_link_libraries(gridFQI2 rele ${GSL_LIBRARIES})
endif()

#if(${GSL_FOUND})
#	add_executable(forex ForexTest.cpp)
#	target_link_libraries(forex rele ${GSL_LIBRARIES})
#endif()

add_executable(gridFQI GridFQITest.cpp)
target_link_libraries(gridFQI rele)

#add_executable(nls_off NlsBatchTest.cpp)
#target_link_libraries(nls_off rele)

add_executable(mc_LSPI McLSPITest.cpp)
target_link_libraries(mc_LSPI rele)