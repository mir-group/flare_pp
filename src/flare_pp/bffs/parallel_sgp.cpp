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


void ParallelSGP ::initialize_global_sparse_descriptors(const Structure &structure) {
  if (global_sparse_descriptors.size() != 0)
    return;

  for (int i = 0; i < structure.descriptors.size(); i++) {
    ClusterDescriptor empty_descriptor;
    empty_descriptor.initialize_cluster(structure.descriptors[i].n_types,
                                        structure.descriptors[i].n_descriptors);
    global_sparse_descriptors.push_back(empty_descriptor);
    std::vector<std::vector<int>> empty_indices;
    global_sparse_indices.push_back(empty_indices); // NOTE: the sparse_indices should be of size n_kernels
  }
};

void ParallelSGP ::add_training_structure(const Structure &structure) {

  int n_energy = structure.energy.size();
  int n_force = structure.forces.size();
  int n_stress = structure.stresses.size();
  int n_struc_labels = n_energy + n_force + n_stress;
  int n_atoms = structure.noa;

  // No updating Kuf

  // Update labels.
  label_count.conservativeResize(training_structures.size() + 2);
  label_count(training_structures.size() + 1) = n_labels + n_struc_labels;
  y.conservativeResize(n_labels + n_struc_labels);
  y.segment(n_labels, n_energy) = structure.energy;
  y.segment(n_labels + n_energy, n_force) = structure.forces;
  y.segment(n_labels + n_energy + n_force, n_stress) = structure.stresses;

  // Update noise.
  noise_vector.conservativeResize(n_labels + n_struc_labels);
  noise_vector.segment(n_labels, n_energy) =
      Eigen::VectorXd::Constant(n_energy, 1 / (energy_noise * energy_noise));
  noise_vector.segment(n_labels + n_energy, n_force) =
      Eigen::VectorXd::Constant(n_force, 1 / (force_noise * force_noise));
  noise_vector.segment(n_labels + n_energy + n_force, n_stress) =
      Eigen::VectorXd::Constant(n_stress, 1 / (stress_noise * stress_noise));

  // Update label count.
  n_energy_labels += n_energy;
  n_force_labels += n_force;
  n_stress_labels += n_stress;
  n_labels += n_struc_labels;

  // Store training structure.
  training_structures.push_back(structure);
  n_strucs += 1;
}

void ParallelSGP ::add_global_noise(const Structure &structure) {

  int n_energy = structure.energy.size();
  int n_force = structure.forces.size();
  int n_stress = structure.stresses.size();
  int n_struc_labels = n_energy + n_force + n_stress;
  int n_atoms = structure.noa;

  // Update noise.
  global_noise_vector.conservativeResize(n_labels + n_struc_labels);
  global_noise_vector.segment(n_labels, n_energy) =
      Eigen::VectorXd::Constant(n_energy, 1 / (energy_noise * energy_noise));
  global_noise_vector.segment(n_labels + n_energy, n_force) =
      Eigen::VectorXd::Constant(n_force, 1 / (force_noise * force_noise));
  global_noise_vector.segment(n_labels + n_energy + n_force, n_stress) =
      Eigen::VectorXd::Constant(n_stress, 1 / (stress_noise * stress_noise));

}



void ParallelSGP::build(const std::vector<Eigen::MatrixXd> &training_cells,
        const std::vector<std::vector<int>> &training_species,
        const std::vector<Eigen::MatrixXd> &training_positions,
        const std::vector<Eigen::VectorXd> &training_labels,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &sparse_indices) {

  // initialize BLACS
  blacs::initialize();

  // Compute the dimensions of the matrices
  int f_size_single_kernel = 0;
  for (int i = 0; i < training_labels.size(); i++) {
     f_size_single_kernel += training_labels[i].size();
  }
  int f_size = f_size_single_kernel * n_kernels;

  int u_size_single_kernel = 0;
  for (int k = 0; k < n_kernels; k++) {
    for (int i = 0; i < sparse_indices.size(); i++) {
      u_size_single_kernel += sparse_indices[k][i].size();
    }
  }
  int u_size = u_size_single_kernel * n_kernels;

  // Create distributed matrices
  { // specify the scope of the DistMatrix
  DistMatrix<double> A(f_size + u_size, u_size, -1, u_size);
  DistMatrix<double> b(f_size + u_size, 1,      -1, 1);
  DistMatrix<double> Kuu_dist(  u_size, u_size, -1, u_size);

  Structure struc;
  int cum_f = 0;
  int cum_u = 0;
  int cum_b = 0;
  // The kernel matrix is built differently from the serial version 
  // The outer loop is the training set, the inner loop is kernels
  
  std::cout << "Start looping training set" << std::endl; 
  for (int t = 0; t < training_cells.size(); t++) {
    struc = Structure(training_cells[t], training_species[t], 
            training_positions[t], cutoff, descriptor_calculators);
    int label_size = 1 + struc.noa * 3 + 6;
    std::cout << "Training data created" << std::endl; 
    
    add_global_noise(struc); // for b
    std::cout << "added global noise" << std::endl; 

    initialize_sparse_descriptors(struc);
    std::cout << "initialized sparse descriptors" << std::endl; 
    initialize_global_sparse_descriptors(struc);
    std::cout << "initialized global sparse descriptors" << std::endl; 
    for (int i = 0; i < n_kernels; i++) { 
      // Collect all sparse envs u
      std::vector<std::vector<int>> sparse_ind_i = sparse_indices[i];
      std::cout << sparse_ind_i[0][1] << std::endl; 
      global_sparse_descriptors[i].add_clusters(
              struc.descriptors[i], sparse_ind_i[t]);
      std::cout << "added clusters to global sparse desc" << std::endl; 

      // Distribute the training structures and training labels
      for (int l = 0; l < label_size; l++) {
        // Collect local training structures for A
        if (A.islocal(cum_f + l, 0)) {
          add_training_structure(struc);
          std::cout << "added local training structures" << std::endl; 
          cum_f += label_size;
          break;
        }
      }

      // Collect local sparse descriptors for Kuu
      for (int s = 0; s < sparse_indices[i][t].size(); s++) {
        if (Kuu_dist.islocal(cum_u + s, 0)) {
          sparse_descriptors[i].add_clusters(
                  struc.descriptors[i], sparse_indices[i][t]);
          std::cout << "added local sparse descriptors" << std::endl; 
          cum_u += sparse_indices[i][t].size();
          break;
        }
      }
    }

  }

  // Build block of A, y, Kuu using distributed training structures
  std::vector<Eigen::MatrixXd> kuf, kuu;
  for (int i = 0; i < n_kernels; i++) {
    Eigen::MatrixXd kuf_i = Eigen::MatrixXd::Zero(u_size_single_kernel, f_size_single_kernel);
    for (int t = 0; t < training_structures.size(); t++) {
      kuf_i.block(0, t, u_size_single_kernel, 1) = kernels[i]->envs_struc(
                  global_sparse_descriptors[i], 
                  training_structures[t].descriptors[i], 
                  kernels[i]->kernel_hyperparameters);
    }
    kuf.push_back(kuf_i);
    kuu.push_back(kernels[i]->envs_envs(
                global_sparse_descriptors[i], 
                sparse_descriptors[i],
                kernels[i]->kernel_hyperparameters));
  }

  // Store square root of noise vector.
  Eigen::VectorXd noise_vector_sqrt = sqrt(noise_vector.array());
  Eigen::VectorXd global_noise_vector_sqrt = sqrt(global_noise_vector.array());

  cum_f = 0;
  cum_b = 0;
  cum_u = 0;
  int local_f = 0;
  int local_b = 0;
  int local_u = 0;
  //for (int t = 0; t < training_cell.size(); t++) {
  //  int n_atoms = training_positions[t].size();
  //  int label_size = 1 + n_atoms * 3 + 6;
  //  for (int i = 0; i < n_kernels; i++) { 
  //    for (int l = 0; l < training_labels[t].size(); l++) {
  //      // Assign a column of kuf to a row of A
  //      if (A.islocal(cum_f, 0)) { // cum_f is the global index
  //        for (int c = 0; c < u_size; c++) { 
  //          A.set(cum_f, c, kuf[i](c, local_f) * noise_vector_sqrt(local_f));
  //        }
  //        local_f += 1;
  //      }
  //      cum_f += 1;

  //      // Assign training label to y 
  //      if (b.islocal(cum_b, 0)) { // cum_b is the global index
  //        b.set(cum_b, 0, training_labels[t](l) * global_noise_vector[l]); // Do A and b have the same division? probably not
  //      }
  //      cum_b += 1;
  //    }

  //    for (int s = 0; s < sparse_indices[i][t].size(); s++) {
  //      // Assign sparse set kernel matrix Kuu
  //      if (Kuu_dist.islocal(cum_u, 0)) { // cum_u is the global index
  //        for (int c = 0; c < u_size; c++) {
  //          if (cum_u == c) {
  //            Kuu_dist.set(cum_u, c, kuu[i](c, local_u) + Kuu_jitter);
  //          } else {
  //            Kuu_dist.set(cum_u, c, kuu[i](c, local_u)); 
  //          }
  //        }
  //        local_u += 1;
  //      }
  //      cum_u += 1;
  //    }
  //  }
  //}

  //// Synchronize
  //blacs::barrier();
  //// Call fence() ??

  //// Cholesky decomposition of Kuu and its inverse.
  //DistMatrix<double> L = Kuu_dist.cholesky();
  //DistMatrix<double> L_inv_dist = L.inv();
  ////L_diag = L_inv.diagonal();
  //L.fence(); // Is this correct? I want other processors able to access elements of L
  //Kuu_inverse = L_inv_dist.matmul(L_inv_dist, 1.0, "T", "N"); // Kuu_inverse = L_inv.transpose() * L_inv; 
  //
  //// Assign L.T to A matrix
  //cum_f = f_size;
  //for (int r = 0; r < u_size; r++) {
  //  if (A.islocal(cum_f, 0)) {
  //    for (int c = 0; c < u_size; c++) {
  //      A.set(cum_f, c, L(c, r)); // the local_f is actually a global index of L.T
  //    }
  //  }

  //  if (b.islocal(cum_f, 0)) {
  //    b.set(cum_f, 0, 0.0); // set chunk f_size ~ f_size + u_size to 0 
  //  }
  //  cum_f += 1;

  //}

  //DistMatrix<double> R_inv_QT = A.qr_invert();
  //DistMatrix<double> alpha_dist = R_inv_QT.matmul(b);

  }

  // finalize BLACS
  blacs::finalize();


}


