#include "parallel_sgp.h"
#include <algorithm> // Random shuffle
#include <chrono>
#include <fstream> // File operations
#include <iomanip> // setprecision
#include <iostream>
#include <numeric> // Iota

ParallelSGP ::ParallelSGP() {}

ParallelSGP ::ParallelSGP(std::vector<Kernel *> kernels, double energy_noise,
                    double force_noise, double stress_noise) {

  this->kernels = kernels;
  n_kernels = kernels.size();
  Kuu_jitter = 1e-8; // default value
  label_count = Eigen::VectorXd::Zero(1);

  // Count hyperparameters.
  int n_hyps = 0;
  for (int i = 0; i < kernels.size(); i++) {
    n_hyps += kernels[i]->kernel_hyperparameters.size();
  }

  // Set the kernel hyperparameters.
  hyperparameters = Eigen::VectorXd::Zero(n_hyps + 3);
  Eigen::VectorXd hyps_curr;
  int hyp_counter = 0;
  for (int i = 0; i < kernels.size(); i++) {
    hyps_curr = kernels[i]->kernel_hyperparameters;

    for (int j = 0; j < hyps_curr.size(); j++) {
      hyperparameters(hyp_counter) = hyps_curr(j);
      hyp_counter++;
    }
  }

  // Set the noise hyperparameters.
  hyperparameters(n_hyps) = energy_noise;
  hyperparameters(n_hyps + 1) = force_noise;
  hyperparameters(n_hyps + 2) = stress_noise;

  this->energy_noise = energy_noise;
  this->force_noise = force_noise;
  this->stress_noise = stress_noise;

  // Initialize kernel lists.
  Eigen::MatrixXd empty_matrix;
  for (int i = 0; i < kernels.size(); i++) {
    Kuu_kernels.push_back(empty_matrix);
    Kuf_kernels.push_back(empty_matrix);
  }
}

void ParallelSGP ::initialize_sparse_descriptors(const Structure &structure) {
  if (sparse_descriptors.size() != 0)
    return;

  for (int i = 0; i < structure.descriptors.size(); i++) {
    ClusterDescriptor empty_descriptor;
    empty_descriptor.initialize_cluster(structure.descriptors[i].n_types,
                                        structure.descriptors[i].n_descriptors);
    sparse_descriptors.push_back(empty_descriptor);
    std::vector<std::vector<int>> empty_indices;
    sparse_indices.push_back(empty_indices); // NOTE: the sparse_indices should be of size n_kernels
  }
};


void ParallelSGP ::initialize_local_sparse_descriptors(const Structure &structure) {
  if (local_sparse_descriptors.size() != 0)
    return;

  for (int i = 0; i < structure.descriptors.size(); i++) {
    ClusterDescriptor empty_descriptor;
    empty_descriptor.initialize_cluster(structure.descriptors[i].n_types,
                                        structure.descriptors[i].n_descriptors);
    local_sparse_descriptors.push_back(empty_descriptor);
    std::vector<std::vector<int>> empty_indices;
    local_sparse_indices.push_back(empty_indices); // NOTE: the sparse_indices should be of size n_kernels
  }
};


void ParallelSGP::build(const std::vector<Eigen::MatrixXd> &training_cells,
        const std::vector<std::vector<int>> &training_species,
        const std::vector<Eigen::MatrixXd> &training_positions,
        const std::vector<Eigen::VectorXd> &training_labels,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &sparse_indices) {

  // initialize BLACS
  blacs::initialize();

  // Compute the dimensions of the matrices
  int f_size = 0;
  for (int i = 0; i < training_labels.size(); i++) {
     f_size += training_labels[i].size();
  }
  f_size *= n_kernels;
  f_size = 1;

  int u_size = 0;
  for (int i = 0; i < sparse_indices.size(); i++) {
     u_size += sparse_indices[i].size();
  }
  u_size *= n_kernels;
  u_size = 1;

  // Create distributed matrices
  DistMatrix<double> A(f_size + u_size, u_size, -1, u_size);
  DistMatrix<double> y(f_size + u_size, 1,      -1, 1);
  DistMatrix<double> Kuu(u_size,        u_size, -1, u_size);

//  Structure struc;
//  int cum_f = 0;
//  int cum_u = 0;
//  int cum_y = 0;
//  std::vector<Structure> local_training_structures;
//  std::vector<Eigen::VectorXd> local_training_labels;
//  // The kernel matrix is built differently from the serial version 
//  // The outer loop is the training set, the inner loop is kernels
//  for (int t = 0; t < training_cells.size(); t++) {
//    struc = Structure(training_cells[t], training_species[t], 
//            training_positions[t], cutoff, descriptor_calculators);
//    int label_size = 1 + struc.noa * 3 + 6;
//
//    initialize_sparse_descriptors(struc);
//    initialize_local_sparse_descriptors(struc);
//    for (int i = 0; i < n_kernels; i++) { 
//      // Collect all sparse envs u
//      sparse_descriptors[i].add_clusters(
//              struc.descriptors[i], sparse_indices[i][t]);
//
//      // Distribute the training structures and training labels
//      for (int l = 0; l < label_size; l++) {
//        // Collect local training structures for A
//        if (A.islocal(cum_f + l, 0)) {
//          local_training_structures.push_back(struc);
//          cum_f += label_size;
//          break;
//        }
//      }
//
//      // Collect local sparse descriptors for Kuu
//      for (int s = 0; s < sparse_indices[i][t].size(); s++) {
//        if (Kuu.islocal(cum_u + s, 0)) {
//          local_sparse_descriptors[i].add_clusters(
//                  struc.descriptors[i], sparse_indices[i][t]);
//          cum_u += sparse_indices[i][t].size();
//          break;
//        }
//      }
//    }
//
//  }
//
//  // Build block of A, y, Kuu using distributed training structures
//  std::vector<Eigen::MatrixXd> kuf, kuu;
//  for (int i = 0; i < n_kernels; i++) {
//    kuf.push_back(kernels[i]->envs_strucs(sparse_descriptors, local_training_structures));
//    kuu.push_back(kernels[i]->envs_envs(sparse_descriptors, local_sparse_descriptors));
//  }
//
//  cum_f = 0;
//  cum_y = 0;
//  cum_u = 0;
//  int local_f = 0;
//  int local_u = 0;
//  for (int t = 0; t < training_cell.size(); t++) {
//    int n_atoms = training_positions[t].size();
//    int label_size = 1 + n_atoms * 3 + 6;
//    for (int i = 0; i < n_kernels; i++) { 
//      for (int l = 0; l < label_size; l++) {
//        // Assign a column of kuf to a row of A
//        if (A.islocal(cum_f, 0)) { // cum_f is the global index
//          for (int c = 0; c < u_size; c++) { 
//            A.set(cum_f, c, kuf[i](c, local_f)); 
//          }
//          local_f += 1;
//        }
//        cum_f += 1;
//
//        // Assign training label to y 
//        if (y.islocal(cum_y, 0)) { // cum_y is the global index
//          y.set(cum_y, 0, training_labels[t](l)); 
//        }
//        cum_y += 1;
//      }
//
//      for (int s = 0; s < sparse_indices[i][t].size(); s++) {
//        // Assign sparse set kernel matrix Kuu
//        if (Kuu.islocal(cum_u, 0)) { // cum_u is the global index
//          for (int c = 0; c < u_size; c++) {
//            if (cum_u == c) {
//              Kuu.set(cum_u, c, kuu[i](c, local_u) + Kuu_jitter);
//            } else {
//              Kuu.set(cum_u, c, kuu[i](c, local_u)); 
//            }
//          }
//          local_u += 1;
//        }
//        cum_u += 1;
//      }
//    }
//  }
//
//  // Synchronize
//  blacs::barrier();
//  // Call fencd() ??
//
//  // Store square root of noise vector.
//  Eigen::VectorXd noise_vector_sqrt = sqrt(noise_vector.array());
//
//  // Cholesky decompose Kuu.
//  DistMatrix<double> L = Kuu.cholesky();
//  DistMatrix<double> L_inv = L.inv();
//  //L_diag = L_inv.diagonal();
//  
//
//  // Get the inverse of Kuu from Cholesky decomposition.
//  Eigen::MatrixXd Kuu_eye = Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());
//  L_inv = chol.matrixL().solve(Kuu_eye);
//  L_diag = L_inv.diagonal();
//  Kuu_inverse = L_inv.transpose() * L_inv;
//
//  // Form A matrix.
//  Eigen::MatrixXd A =
//      Eigen::MatrixXd::Zero(Kuf.cols() + Kuu.cols(), Kuu.cols());
//  A.block(0, 0, Kuf.cols(), Kuu.cols()) =
//      noise_vector_sqrt.asDiagonal() * Kuf.transpose();
//  A.block(Kuf.cols(), 0, Kuu.cols(), Kuu.cols()) = chol.matrixL().transpose();
//
//  // Form b vector.
//  Eigen::VectorXd b = Eigen::VectorXd::Zero(Kuf.cols() + Kuu.cols());
//  b.segment(0, Kuf.cols()) = noise_vector_sqrt.asDiagonal() * y;
//
//  // QR decompose A.
//  Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
//  Eigen::VectorXd Q_b = qr.householderQ().transpose() * b;
//  R_inv = qr.matrixQR().block(0, 0, Kuu.cols(), Kuu.cols())
//                       .triangularView<Eigen::Upper>()
//                       .solve(Kuu_eye);
//  R_inv_diag = R_inv.diagonal();
//  alpha = R_inv * Q_b;
//  Sigma = R_inv * R_inv.transpose();

  // finalize BLACS
  blacs::finalize();


}


