add_executable(mpi_build mpi_build.cpp)
include_directories(../src/flare_pp)
target_link_libraries(mpi_build PUBLIC flare)

add_executable(omp_build omp_build.cpp)
include_directories(../src/flare_pp)
target_link_libraries(omp_build PUBLIC flare)
