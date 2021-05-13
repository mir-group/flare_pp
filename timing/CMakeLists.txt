# make executable
#add_executable(time time_single_bond.cpp)
#target_include_directories(time PUBLIC ${ACE_INCLUDE_DIR})
#target_link_libraries(time PUBLIC flare_pp)
#
#
#add_executable(benchmark benchmark_B2.cpp)
#target_include_directories(benchmark PUBLIC ${ACE_INCLUDE_DIR})
#target_link_libraries(benchmark PUBLIC flare_pp)

add_executable(time_mpi mpi_construction.cpp)
#target_include_directories(time_mpi PUBLIC ${ACE_INCLUDE_DIR})
include_directories(../src/flare_pp)
target_link_libraries(time_mpi PUBLIC flare)
