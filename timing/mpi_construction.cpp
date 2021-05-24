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
  int n_types = n_species;

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
  SparseGP sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s);

  // Build kernel matrices for paralle sgp
  std::vector<Eigen::MatrixXd> training_cells, training_positions;
  std::vector<std::vector<int>> training_species;
  std::vector<Eigen::VectorXd> training_labels;
  std::vector<std::vector<std::vector<int>>> sparse_indices = {{}};
  Eigen::MatrixXd cell, positions;
  Eigen::VectorXd labels; 
  std::vector<int> species, sparse_inds; 

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
        training_labels, cutoff, dc, sparse_indices, n_types);

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Rank: " << blacs::mpirank << ", time: " << duration << " ms" << std::endl;

  if (blacs::mpirank == 0) {
    // Build sparse_gp (non parallel)
    for (int t = 0; t < n_strucs; t++) {
      cell = training_cells[t];
      positions = training_positions[t];
      labels = training_labels[t];
      species = training_species[t];
      sparse_inds = sparse_indices[0][t];

      Structure train_struc(cell, species, positions, cutoff, dc);
      train_struc.energy = labels.segment(0, 1);
      train_struc.forces = labels.segment(1, n_atoms * 3);
      train_struc.stresses = labels.segment(1 + n_atoms * 3, 6);
 
      sparse_gp.add_training_structure(train_struc);
      sparse_gp.add_specific_environments(train_struc, sparse_inds);
    }
    sparse_gp.update_matrices_QR();
    std::cout << "Done QR for sparse_gp" << std::endl;
  
    // Check the kernel matrices are consistent
    std::cout << "begin comparing n_clusters" << std::endl;
    assert(parallel_sgp.sparse_descriptors[0].n_clusters == sparse_gp.Sigma.rows());
    assert(parallel_sgp.sparse_descriptors[0].n_clusters == sparse_gp.sparse_descriptors[0].n_clusters);
    std::cout << "done comparing n_clusters" << std::endl;
    for (int t = 0; t < parallel_sgp.sparse_descriptors[0].n_types; t++) {
      for (int r = 0; r < parallel_sgp.sparse_descriptors[0].descriptors[t].rows(); r++) {
        double par_desc_norm = parallel_sgp.sparse_descriptors[0].descriptor_norms[t](r);
        double sgp_desc_norm = sparse_gp.sparse_descriptors[0].descriptor_norms[t](r);

        if (std::abs(par_desc_norm - sgp_desc_norm) > 1e-6) {
          std::cout << "++ t=" << t << ", r=" << r;
          std::cout << ", par_desc_norm=" << par_desc_norm << ", sgp_desc_norm=" << sgp_desc_norm << std::endl;
//          throw std::runtime_error("descriptors does not match");
        }

        for (int c = 0; c < parallel_sgp.sparse_descriptors[0].descriptors[t].cols(); c++) {
          double par_desc = parallel_sgp.sparse_descriptors[0].descriptors[t](r, c);
          double sgp_desc = sparse_gp.sparse_descriptors[0].descriptors[t](r, c);
          std::cout << "t=" << t << ", r=" << r << " c=" << c;
          std::cout << ", par_desc=" << par_desc << ", sgp_desc=" << sgp_desc << std::endl;

          if (std::abs(par_desc - sgp_desc) > 1e-6) {
            std::cout << "*** t=" << t << ", r=" << r << " c=" << c;
            std::cout << ", par_desc=" << par_desc << ", sgp_desc=" << sgp_desc << std::endl;
//            throw std::runtime_error("descriptors does not match");
          }
        }
      }
    }
    std::cout << "Checked matrix shape" << std::endl;
    std::cout << "parallel_sgp.Kuu(0, 0)=" << parallel_sgp.Kuu(0, 0) << std::endl;
  
    for (int r = 0; r < parallel_sgp.Kuu.rows(); r++) {
      for (int c = 0; c < parallel_sgp.Kuu.rows(); c++) {
        // Sometimes the accuracy is between 1e-6 ~ 1e-5        
        if (std::abs(parallel_sgp.Kuu(r, c) - sparse_gp.Kuu(r, c)) > 1e-6) {
          throw std::runtime_error("Kuu does not match");
        }
      }
    }
    std::cout << "Kuu matches" << std::endl;
 
    for (int r = 0; r < parallel_sgp.Kuu_inverse.rows(); r++) {
      for (int c = 0; c < parallel_sgp.Kuu_inverse.rows(); c++) {
        if (std::abs(parallel_sgp.Kuu_inverse(r, c) - sparse_gp.Kuu_inverse(r, c)) > 1e-5) {
          throw std::runtime_error("Kuu_inverse does not match");
        }
      }
    }
    std::cout << "Kuu_inverse matches" << std::endl;
  
    for (int r = 0; r < parallel_sgp.alpha.size(); r++) {
      if (std::abs(parallel_sgp.alpha(r) - sparse_gp.alpha(r)) > 1e-6) {
        std::cout << "alpha: r=" << r << " " << parallel_sgp.alpha(r) << " " << sparse_gp.alpha(r) << std::endl;
        throw std::runtime_error("alpha does not match");
      }
    }
    std::cout << "alpha matches" << std::endl;

    // Compare predictions on testing structure are consistent
    cell = Eigen::MatrixXd::Identity(3, 3) * cell_size;
    positions = Eigen::MatrixXd::Random(n_atoms, 3) * cell_size / 2;
    // Make random species.
    species.clear(); 
    for (int i = 0; i < n_atoms; i++) {
      species.push_back(rand() % n_species);
    }
    Structure test_struc(cell, species, positions, cutoff, dc);
    parallel_sgp.predict_local_uncertainties(test_struc);
    Structure test_struc_copy(test_struc.cell, test_struc.species, test_struc.positions, cutoff, dc);
    sparse_gp.predict_local_uncertainties(test_struc_copy);
  
    for (int r = 0; r < test_struc.mean_efs.size(); r++) {
      if (std::abs(test_struc.mean_efs(r) - test_struc_copy.mean_efs(r)) > 1e-5) {
        std::cout << "mean_efs: r=" << r << " " << test_struc.mean_efs(r) << " " << test_struc_copy.mean_efs(r) << std::endl;
        throw std::runtime_error("mean_efs does not match");
      }
    }
    std::cout << "mean_efs matches" << std::endl;
  
    for (int i = 0; i < test_struc.local_uncertainties.size(); i++) {
      for (int r = 0; r < test_struc.local_uncertainties[i].size(); r++) {
        if (std::abs(test_struc.local_uncertainties[i](r) - test_struc_copy.local_uncertainties[i](r)) > 1e-5) {
          throw std::runtime_error("local_unc does not match");
        }
      }
    }
    std::cout << "local_unc matches" << std::endl;
  }

  return 0;
}
