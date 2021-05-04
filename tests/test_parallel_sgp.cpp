#include "parallel_sgp.h"
#include "sparse_gp.h"
#include "test_structure.h"
#include "omp.h"
#include "mpi.h"
#include <thread>
#include <chrono>
#include <numeric> // Iota


TEST_F(StructureTest, BuildPMatrix){
  double sigma_e = 1;
  double sigma_f = 2;
  double sigma_s = 3;

  blacs::initialize();

  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel);
  ParallelSGP parallel_sgp = ParallelSGP(kernels, sigma_e, sigma_f, sigma_s);
  SparseGP sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);

  // Generate random labels
  Eigen::VectorXd energy = Eigen::VectorXd::Random(1);
  Eigen::VectorXd forces = Eigen::VectorXd::Random(n_atoms * 3);
  Eigen::VectorXd stresses = Eigen::VectorXd::Random(6);

  // Broadcast data such that different procs won't generate different random numbers
  MPI_Bcast(energy.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(forces.data(), n_atoms *  3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(stresses.data(), 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

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
  MPI_Bcast(positions_1.data(), n_atoms * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(positions_2.data(), n_atoms * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  labels_1 = Eigen::VectorXd::Random(1 + n_atoms * 3 + 6);
  labels_2 = Eigen::VectorXd::Random(1 + n_atoms * 3 + 6);
  MPI_Bcast(labels_1.data(), 1 + n_atoms * 3 + 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(labels_2.data(), 1 + n_atoms * 3 + 6, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // Make random species.
  for (int i = 0; i < n_atoms; i++) {
    species_1.push_back(rand() % n_species);
    species_2.push_back(rand() % n_species);
  }
  MPI_Bcast(species_1.data(), n_atoms, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(species_2.data(), n_atoms, MPI_INT, 0, MPI_COMM_WORLD);

  // Build kernel matrices for paralle sgp
  std::vector<Eigen::MatrixXd> training_cells = {cell_1, cell_2};
  std::vector<std::vector<int>> training_species = {species_1, species_2};
  std::vector<Eigen::MatrixXd> training_positions = {positions_1, positions_2};
  std::vector<Eigen::VectorXd> training_labels = {labels_1, labels_2};
  std::vector<std::vector<std::vector<int>>> sparse_indices = {{{0, 1}, {2}}}; //{{{0, 1, 4, 5, 6, 7, 9}, {0, 2, 3, 4, 6, 7, 8}}};

  std::cout << "Start building" << std::endl;
  parallel_sgp.build(training_cells, training_species, training_positions, 
          training_labels, cutoff, dc, sparse_indices);

  if (blacs::mpirank == 0) {
    //parallel_sgp.write_mapping_coefficients("beta.txt", "Me", 0);
    //parallel_sgp.write_varmap_coefficients("beta_var.txt", "Me", 0);
  
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
    std::cout << "Done QR for sparse_gp" << std::endl;
  
    // Check the kernel matrices are consistent
    EXPECT_EQ(parallel_sgp.sparse_descriptors[0].n_clusters, sparse_gp.Sigma.rows());
    EXPECT_EQ(sparse_gp.sparse_descriptors[0].n_clusters,
              parallel_sgp.Kuu_inverse.rows());
    EXPECT_EQ(parallel_sgp.sparse_descriptors[0].n_clusters, sparse_gp.sparse_descriptors[0].n_clusters);
  //  for (int t = 0; t < parallel_sgp.sparse_descriptors[0].n_types; t++) {
  //    for (int r = 0; r < parallel_sgp.sparse_descriptors[0].descriptors[t].rows(); r++) {
  //      for (int c = 0; c < parallel_sgp.sparse_descriptors[0].descriptors[t].cols(); c++) {
  //        double par_desc = parallel_sgp.sparse_descriptors[0].descriptors[t](r, c);
  //        double sgp_desc = sparse_gp.sparse_descriptors[0].descriptors[t](r, c);
  //        std::cout << "descriptor par ser r=" << r << " c=" << c << " " << par_desc << " " << sgp_desc << std::endl;
  //      }
  //    }
  //  }
    std::cout << "Checked matrix shape" << std::endl;
  
    for (int r = 0; r < sparse_gp.y.size(); r++) {
      std::cout << "y(" << r << ")=" << parallel_sgp.b_vec(r) << " " << sparse_gp.y(r) << std::endl;
      EXPECT_NEAR(parallel_sgp.b_vec(r), sparse_gp.y(r), 1e-6); // * sqrt(sparse_gp.noise_vector(r)), 1e-6);
    }

//    int cum_f = 0;
//    for (int i = 0; i < training_labels.size(); i++) {
//      for (int j = 0; j < training_labels[i].size(); j++) {
//        std::cout << "label(" << cum_f << ")=" << training_labels[i](j) << std::endl;
//        cum_f += 1;
//      }
//    }
  
  
    // TODO: Kuu is just for debugging, not stored
    for (int r = 0; r < parallel_sgp.Kuu.rows(); r++) {
      for (int c = 0; c < parallel_sgp.Kuu.rows(); c++) {
        std::cout << "parallel_sgp.Kuu(" << r << "," << c << ")=" << parallel_sgp.Kuu(r, c);
        std::cout << " " << sparse_gp.Kuu(r, c) << std::endl;
        EXPECT_NEAR(parallel_sgp.Kuu(r, c), sparse_gp.Kuu(r, c), 1e-6);
      }
    }
  
//    // TODO: Kuf is just for debugging, not stored
//    Eigen::VectorXd noise_vector_sqrt = sqrt(sparse_gp.noise_vector.array());
//    Eigen::MatrixXd sgp_Kuf_noise = sparse_gp.Kuf * noise_vector_sqrt.asDiagonal();
//    for (int r = 0; r < parallel_sgp.Kuf.rows(); r++) {
//      for (int c = 0; c < parallel_sgp.Kuf.cols(); c++) {
//        std::cout << "parallel_sgp.Kuf(" << r << "," << c << ")=" << parallel_sgp.Kuf(r, c);
//        std::cout << " " << sgp_Kuf_noise(r, c) << std::endl; 
//        EXPECT_NEAR(parallel_sgp.Kuf(r, c), sgp_Kuf_noise(r, c), 1e-6);
//      }
//    }
  
  
  //  for (int r = 0; r < parallel_sgp.Kuu_inverse.rows(); r++) {
  //    for (int c = 0; c < parallel_sgp.Kuu_inverse.rows(); c++) {
  //      EXPECT_NEAR(parallel_sgp.Kuu_inverse(r, c), sparse_gp.Kuu_inverse(r, c), 1e-6);
  //    }
  //  }
  //
  //  for (int r = 0; r < parallel_sgp.alpha.size(); r++) {
  //    EXPECT_NEAR(parallel_sgp.alpha(r), sparse_gp.alpha(r), 1e-6);
  //  }
  //  // Compare predictions on testing structure are consistent
  //  parallel_sgp.predict_local_uncertainties(test_struc);
  //  Structure test_struc_copy(test_struc.cell, test_struc.species, test_struc.positions, cutoff, dc);
  //  sparse_gp.predict_local_uncertainties(test_struc_copy);
  //
  //  EXPECT_NEAR(test_struc.mean_efs(0), test_struc_copy.mean_efs(0), 1e-5);
  ////  for (int r = 0; r < test_struc.mean_efs.size(); r++) {
  ////    EXPECT_NEAR(test_struc.mean_efs(r), test_struc_copy.mean_efs(r), 1e-5);
  ////  }
  //
  //  for (int i = 0; i < test_struc.local_uncertainties.size(); i++) {
  //    for (int r = 0; r < test_struc.local_uncertainties[i].size(); r++) {
  //      EXPECT_NEAR(test_struc.local_uncertainties[i](r), test_struc_copy.local_uncertainties[i](r), 1e-5);
  //    }
  //  }
  }
}
