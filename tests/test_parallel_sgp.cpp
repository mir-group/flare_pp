#include "parallel_sgp.h"
#include "test_structure.h"
#include "omp.h"
#include <thread>
#include <chrono>
#include <numeric> // Iota


TEST_F(StructureTest, BuildPMatrix){
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel);
  ParallelSGP sparse_gp = ParallelSGP(kernels, sigma_e, sigma_f, sigma_s);

  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);
  test_struc.energy = energy;
  test_struc.forces = forces;
  test_struc.stresses = stresses;

  // Make positions.
  Eigen::MatrixXd cell_1, cell_2;
  std::vector<int> species_1, species_2;
  Eigen::MatrixXd positions_1, positions_2;
  Eigen::VectorXd labels_1, labels_2;

  cell_1 = Eigen::MatrixXd::Identity(3, 3) * cell_size;
  cell_2 = Eigen::MatrixXd::Identity(3, 3) * cell_size;

  positions_1 = Eigen::MatrixXd::Random(n_atoms, 3) * cell_size / 2;
  positions_2 = Eigen::MatrixXd::Random(n_atoms, 3) * cell_size / 2;

  labels_1 = Eigen::VectorXd::Random(1 + n_atoms * 3 + 6);
  labels_2 = Eigen::VectorXd::Random(1 + n_atoms * 3 + 6);

  // Make random species.
  for (int i = 0; i < n_atoms; i++) {
    species_1.push_back(rand() % n_species);
    species_2.push_back(rand() % n_species);
  }

  std::vector<Eigen::MatrixXd> training_cells = {cell_1, cell_2};
  std::vector<std::vector<int>> training_species = {species_1, species_2};
  std::vector<Eigen::MatrixXd> training_positions = {positions_1, positions_2};
  std::vector<Eigen::VectorXd> training_labels = {labels_1, labels_2};
  std::vector<std::vector<std::vector<int>>> sparse_indices = {{{0, 1}, {3}}};

  std::cout << "Start building" << std::endl;
  sparse_gp.build(training_cells, training_species, training_positions, 
          training_labels, cutoff, dc, sparse_indices);
}
