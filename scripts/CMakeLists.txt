add_executable(mpi_build mpi_build.cpp)
include_directories(../src/flare_pp)
target_link_libraries(mpi_build PUBLIC flare)

add_executable(omp_build omp_build.cpp)
include_directories(../src/flare_pp)
target_link_libraries(omp_build PUBLIC flare)

add_executable(mpi_matrix_test mpi_matrix_test.cpp)
include_directories(../src/flare_pp)
target_link_libraries(mpi_matrix_test PUBLIC flare)
