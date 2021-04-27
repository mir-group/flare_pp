#include "parallel_sgp.h"
#include "sparse_gp.h"
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
  ParallelSGP parallel_sgp = ParallelSGP(kernels, sigma_e, sigma_f, sigma_s);
  SparseGP sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);

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

  // Build kernel matrices for paralle sgp
  std::vector<Eigen::MatrixXd> training_cells = {cell_1, cell_2};
  std::vector<std::vector<int>> training_species = {species_1, species_2};
  std::vector<Eigen::MatrixXd> training_positions = {positions_1, positions_2};
  std::vector<Eigen::VectorXd> training_labels = {labels_1, labels_2};
  std::vector<std::vector<std::vector<int>>> sparse_indices = {{{0, 1}, {3}}};

  std::cout << "Start building" << std::endl;
  parallel_sgp.build(training_cells, training_species, training_positions, 
          training_labels, cutoff, dc, sparse_indices);

  parallel_sgp.write_mapping_coefficients("beta.txt", "Me", 0);
  parallel_sgp.write_varmap_coefficients("beta_var.txt", "Me", 0);

  // Build sparse_gp (non parallel)
  Structure train_struc_1 = Structure(cell_1, species_1, positions_1, cutoff, dc);
  train_struc_1.energy = labels_1.segment(0, 1);
  train_struc_1.forces = labels_1.segment(1, n_atoms * 3);
  train_struc_1.stresses = labels_1.segment(1 + n_atoms * 3, 6);

  Structure train_struc_2 = Structure(cell_2, species_2, positions_2, cutoff, dc);
  train_struc_2.energy = labels_2.segment(0, 1);
  train_struc_2.forces = labels_2.segment(1, n_atoms * 3);
  train_struc_2.stresses = labels_2.segment(1 + n_atoms * 3, 6);

  sparse_gp.add_training_structure(train_struc_1);
  sparse_gp.add_specific_environments(train_struc_1, sparse_indices[0][0]);
  sparse_gp.add_training_structure(train_struc_2);
  sparse_gp.add_specific_environments(train_struc_2, sparse_indices[0][1]);
  sparse_gp.update_matrices_QR();

  // Check the kernel matrices are consistent
  EXPECT_EQ(parallel_sgp.sparse_descriptors[0].n_clusters, sparse_gp.Sigma.rows());
  EXPECT_EQ(sparse_gp.sparse_descriptors[0].n_clusters,
            parallel_sgp.Kuu_inverse.rows());
//  EXPECT_NEAR(parallel_sgp.alpha, sparse_gp.alpha, 1e-6);
  for (int r = 0; r < parallel_sgp.Kuu_inverse.rows(); r++) {
    for (int c = 0; c < parallel_sgp.Kuu_inverse.rows(); c++) {
      EXPECT_NEAR(parallel_sgp.Kuu_inverse(r, c), sparse_gp.Kuu_inverse(r, c), 1e-6);
    }
  }

  // Compare predictions on testing structure are consistent
//  parallel_sgp.predict_local_uncertainties(test_struc);
//  test_struc_copy = Structure(test_struc.cell, test_struc.species, test_struc.positions, cutoff, dc);
//  sparse_sgp.predict_local_uncertainties(test_struc_copy);
//
//  EXPECT_NEAR(test_struc.mean_efs, test_struc_copy.mean_efs, 1e-6);
//  EXPECT_NEAR(test_struc.local_uncertainties, test_struc_copy.local_uncertainties, 1e-6);

}
