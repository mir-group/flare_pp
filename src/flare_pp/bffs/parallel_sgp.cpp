#include "parallel_sgp.h"
#include <algorithm> // Random shuffle
#include <chrono>
#include <fstream> // File operations
#include <iomanip> // setprecision
#include <iostream>
#include <numeric> // Iota

ParallelSGP ::ParallelSGP() {}

ParallelSGP ::ParallelSGP(std::vector<Kernel *> kernels, double energy_noise,
                    double force_noise, double stress_noise, bool isDistributed_) 
            : isDistributed(isDistributed_) {

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


void ParallelSGP::build(const std::vector<Eigen::MatrixXd> &training_cells,
        const std::vector<std::vector<int>> &training_species,
        const std::vector<Eigen::MatrixXd> &training_positions,
        const std::vector<Eigen::VectorXd> &training_labels,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<Eigen::VectorXd> &sparse_indices) {

  // initialize BLACS
  blacs::initialize();

  // Compute the dimensions of the matrices
  int f_size = 0;
  for (int i = 0; i < training_labels.size(); i++) {
     f_size += training_labels[i].size();
  }

  int u_size = 0;
  for (int i = 0; i < sparse_indices.size(); i++) {
     u_size += sparse_indices[i].size();
  }

  // Create distributed matrices
  DistMatrix<int> A(f_size + u_size, u_size);
  DistMatrix<int> y(f_size + u_size, 1);
  DistMatrix<int> Kuu(u_size, u_size);

//  int A_numBlockRows = std::min((int)mpi->getSize(), f_size + u_size);
//  A = Matrix<double>(f_size + u_size, u_size, A_numBlockRows, 1, isDistributed);
//  y = Matrix<double>(f_size + u_size, 1, A_numBlockRows, 1, isDistributed);
//  int Kuu_numBlockRows = std::min((int)mpi->getSize(), f_size + u_size);
//  Kuu = Matrix<double>(u_size, u_size, Kuu_numBlockRows, 1, isDistributed);
  

//  // Store square root of noise vector.
//  Eigen::VectorXd noise_vector_sqrt = sqrt(noise_vector.array());
//
//  // Cholesky decompose Kuu.
//  Eigen::LLT<Eigen::MatrixXd> chol(
//      Kuu + Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols()));
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
}


