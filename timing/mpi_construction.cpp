#include <chrono>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include "omp.h"
#include "mpi.h"

#include "b2.h"
#include "parallel_sgp.h"
#include "sparse_gp.h"
#include "structure.h"
#include "normalized_dot_product.h"

int main(int argc, char* argv[]) {
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  int n_atoms = 100; 
  int n_envs = 5;
  int n_strucs = std::stoi(argv[1]);
  int n_species = 2;

  double cell_size = 10;
  double cutoff = cell_size / 2;

  int N = 8;
  int L = 3;
  std::string radial_string = "chebyshev";
  std::string cutoff_string = "cosine";
  std::vector<double> radial_hyps{0, cutoff};
  std::vector<double> cutoff_hyps;
  std::vector<int> descriptor_settings{n_species, N, L};

  std::vector<Descriptor *> dc;
  B2 ps(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
        descriptor_settings);
  dc.push_back(&ps);

  blacs::initialize();

  double sigma = 2.0;
  int power = 2;
  NormalizedDotProduct kernel_norm = NormalizedDotProduct(sigma, power);
  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel_norm);
  ParallelSGP parallel_sgp = ParallelSGP(kernels, sigma_e, sigma_f, sigma_s);

  // Build kernel matrices for paralle sgp
  std::vector<Eigen::MatrixXd> training_cells, training_positions;
  std::vector<std::vector<int>> training_species;
  std::vector<Eigen::VectorXd> training_labels;
  std::vector<std::vector<std::vector<int>>> sparse_indices = {{}};

  for (int t = 0; t < n_strucs; t++) {
    Eigen::MatrixXd cell = Eigen::MatrixXd::Identity(3, 3) * cell_size;
    training_cells.push_back(cell);

    // Make random positions
    Eigen::MatrixXd positions = Eigen::MatrixXd::Random(n_atoms, 3) * cell_size / 2;
    MPI_Bcast(positions.data(), n_atoms * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    training_positions.push_back(positions);
  
    // Make random labels
    Eigen::VectorXd labels = Eigen::VectorXd::Random(1 + n_atoms * 3 + 6);
    MPI_Bcast(labels.data(), 1 + n_atoms * 3 + 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    training_labels.push_back(labels);
  
    // Make random species.
    std::vector<int> species;
    for (int i = 0; i < n_atoms; i++) {
      species.push_back(rand() % n_species);
    }
    MPI_Bcast(species.data(), n_atoms, MPI_INT, 0, MPI_COMM_WORLD);
    training_species.push_back(species);

    // Make random sparse envs
    std::vector<int> env_inds;
    for (int i = 0; i < n_atoms; i++) env_inds.push_back(i);
    std::random_shuffle( env_inds.begin(), env_inds.end() );
    std::vector<int> sparse_inds;
    for (int i = 0; i < n_envs; i++) sparse_inds.push_back(env_inds[i]);
    MPI_Bcast(sparse_inds.data(), n_envs, MPI_INT, 0, MPI_COMM_WORLD);
    sparse_indices[0].push_back(sparse_inds);
  }

  std::cout << "Start building" << std::endl;
  double duration = 0;
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  parallel_sgp.build(training_cells, training_species, training_positions, 
        training_labels, cutoff, dc, sparse_indices);

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Rank: " << blacs::mpirank << ", time: " << duration << " ms" << std::endl;

  return 0;
}
