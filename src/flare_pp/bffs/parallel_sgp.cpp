#include "parallel_sgp.h"
#include <mpi.h>
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

//std::vector<ClusterDescriptor>
//ParallelSGP ::initialize_sparse_descriptors(const Structure &structure, std::vector<ClusterDescriptor> sparse_desc) {
//  if (sparse_desc.size() != 0)
//    return sparse_desc;
//
//  for (int i = 0; i < structure.descriptors.size(); i++) {
//    ClusterDescriptor empty_descriptor;
//    empty_descriptor.initialize_cluster(structure.descriptors[i].n_types,
//                                        structure.descriptors[i].n_descriptors);
//    sparse_desc.push_back(empty_descriptor);
//    std::vector<std::vector<int>> empty_indices;
//    sparse_indices.push_back(empty_indices); // NOTE: the sparse_indices should be of size n_kernels
//  }
//
//  return sparse_desc;
//};

void ParallelSGP ::initialize_local_sparse_descriptors(const Structure &structure) {
  if (local_sparse_descriptors.size() != 0)
    return;

  for (int i = 0; i < structure.descriptors.size(); i++) {
    ClusterDescriptor empty_descriptor;
    empty_descriptor.initialize_cluster(structure.descriptors[i].n_types,
                                        structure.descriptors[i].n_descriptors);
    local_sparse_descriptors.push_back(empty_descriptor);
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

std::vector<std::vector<std::vector<int>>>
ParallelSGP ::sparse_indices_by_type(const Structure &structure,
                                     const std::vector<int> atoms) {

  // Gather clusters with central atom in the given list.
  std::vector<std::vector<std::vector<int>>> indices_1;
  for (int i = 0; i < n_kernels; i++){
    // TODO: change to local/global sparse_indices
//    sparse_indices[i].push_back(atoms); // for each kernel the added atoms are the same

    int n_types = structure.descriptors[i].n_types;
    std::vector<std::vector<int>> indices_2;
    for (int j = 0; j < n_types; j++){
      int n_clusters = structure.descriptors[i].n_clusters_by_type[j];
      std::vector<int> indices_3;
      for (int k = 0; k < n_clusters; k++){
        int atom_index_1 = structure.descriptors[i].atom_indices[j](k);
        for (int l = 0; l < atoms.size(); l++){
          int atom_index_2 = atoms[l];
          if (atom_index_1 == atom_index_2){
            indices_3.push_back(k);
          }
        }
      }
      indices_2.push_back(indices_3);
    }
    indices_1.push_back(indices_2);
  }

  return indices_1;
}

void ParallelSGP ::add_local_specific_environments(const Structure &structure,
                                          const std::vector<int> atoms) {

  std::vector<std::vector<std::vector<int>>> indices_1 = 
      sparse_indices_by_type(structure, atoms);

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < n_kernels; i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i], indices_1[i]);
    cluster_descriptors.push_back(cluster_descriptor);
  }

  // Store sparse environments.
  for (int i = 0; i < n_kernels; i++) {
    local_sparse_descriptors[i].add_clusters_by_type(structure.descriptors[i],
                                               indices_1[i]);
  }
}

void ParallelSGP ::add_global_specific_environments(const Structure &structure,
                                          const std::vector<int> atoms) {

  std::vector<std::vector<std::vector<int>>> indices_1 = 
      sparse_indices_by_type(structure, atoms);

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < n_kernels; i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i], indices_1[i]);
    cluster_descriptors.push_back(cluster_descriptor);
  }

  // Store sparse environments.
  for (int i = 0; i < n_kernels; i++) {
    global_sparse_descriptors[i].add_clusters_by_type(structure.descriptors[i],
                                               indices_1[i]);
  }
}

void ParallelSGP ::add_global_noise(const Structure &structure) {

  int n_energy = structure.energy.size();
  int n_force = structure.forces.size();
  int n_stress = structure.stresses.size();
  int n_struc_labels = n_energy + n_force + n_stress;
  int n_atoms = structure.noa;

  // Update noise.
  global_noise_vector.conservativeResize(global_n_labels + n_struc_labels);
  global_noise_vector.segment(global_n_labels, n_energy) =
      Eigen::VectorXd::Constant(n_energy, 1 / (energy_noise * energy_noise));
  global_noise_vector.segment(global_n_labels + n_energy, n_force) =
      Eigen::VectorXd::Constant(n_force, 1 / (force_noise * force_noise));
  global_noise_vector.segment(global_n_labels + n_energy + n_force, n_stress) =
      Eigen::VectorXd::Constant(n_stress, 1 / (stress_noise * stress_noise));

  global_n_labels += n_struc_labels;

}



void ParallelSGP::build(const std::vector<Eigen::MatrixXd> &training_cells,
        const std::vector<std::vector<int>> &training_species,
        const std::vector<Eigen::MatrixXd> &training_positions,
        const std::vector<Eigen::VectorXd> &training_labels,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices) {

  load_local_training_data(training_cells, training_species, training_positions,
        training_labels, cutoff, descriptor_calculators, training_sparse_indices);

  compute_matrices(training_labels, descriptor_calculators, training_sparse_indices);

}

void ParallelSGP::load_local_training_data(const std::vector<Eigen::MatrixXd> &training_cells,
        const std::vector<std::vector<int>> &training_species,
        const std::vector<Eigen::MatrixXd> &training_positions,
        const std::vector<Eigen::VectorXd> &training_labels,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices) {

  // initialize BLACS
  blacs::initialize();

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Compute the dimensions of the matrices
  f_size = 0;
  for (int i = 0; i < training_labels.size(); i++) {
     f_size += training_labels[i].size();
  }
  f_size_per_proc = f_size / world_size;

  u_size = 0;
  for (int k = 0; k < n_kernels; k++) {
    u_size_single_kernel = 0; // TODO: here assumes that each kernel has the same set of sparse envs
    for (int i = 0; i < training_sparse_indices[k].size(); i++) {
      u_size += training_sparse_indices[k][i].size();
      u_size_single_kernel += training_sparse_indices[k][i].size();
    }
  }
  u_size_per_proc = u_size / world_size;

  Structure struc;
  int cum_f = 0;
  int cum_u = 0;
  int cum_b = 0;
 
  nmin_struc = world_rank * f_size_per_proc;
  nmin_envs = world_rank * u_size_per_proc;
  if (world_rank == world_size - 1) {
    nmax_struc = f_size;
    nmin_envs = u_size_single_kernel; 
  } else {
    nmax_struc = (world_rank + 1) * f_size_per_proc;
    nmax_envs = (world_rank + 1) * u_size_per_proc;
  }

  // Distribute the training structures and sparse envs
  global_n_labels = 0;
  for (int t = 0; t < training_cells.size(); t++) {
    struc = Structure(training_cells[t], training_species[t], 
            training_positions[t], cutoff, descriptor_calculators);
    int label_size = 1 + struc.noa * 3 + 6;
    assert(label_size == training_labels[t].size());
    struc.energy = training_labels[t].segment(cum_f, 1);
    struc.forces = training_labels[t].segment(cum_f + 1, 3 * struc.noa);
    struc.stresses = training_labels[t].segment(cum_f + 3 * struc.noa, 6); 
    
    add_global_noise(struc); // for b

    //global_sparse_descriptors = initialize_sparse_descriptors(struc, global_sparse_descriptors);
    initialize_global_sparse_descriptors(struc);

    if (nmin_struc < cum_f + label_size && cum_f < nmax_struc) {
      // Collect local training structures for A
      add_training_structure(struc);
    }

    // TODO: now all the kernels share the same sparse envs
    add_global_specific_environments(struc, training_sparse_indices[0][t]);
    if (nmin_struc < cum_f + label_size && cum_f < nmax_struc) {
      initialize_local_sparse_descriptors(struc); 
      add_local_specific_environments(struc, training_sparse_indices[0][t]);
    }

    cum_f += label_size;

  }
  // Assign global sparse descritors
  sparse_descriptors = global_sparse_descriptors;
  blacs::barrier();

}

void ParallelSGP::compute_matrices(
        const std::vector<Eigen::VectorXd> &training_labels,
        std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices) {

  // Build block of A, y, Kuu using distributed training structures
  std::vector<Eigen::MatrixXd> kuf, kuu;
  int cum_f = 0;
  int cum_u = 0;
  for (int i = 0; i < n_kernels; i++) {
    assert(u_size_single_kernel == global_sparse_descriptors[i].n_clusters);
    Eigen::MatrixXd kuf_i = Eigen::MatrixXd::Zero(u_size_single_kernel, f_size);
    for (int t = 0; t < training_structures.size(); t++) {
      int f_size_i = 1 + training_structures[t].noa * 3 + 6;
      kuf_i.block(0, cum_f, u_size_single_kernel, f_size_i) = kernels[i]->envs_struc(
                  global_sparse_descriptors[i], 
                  training_structures[t].descriptors[i], 
                  kernels[i]->kernel_hyperparameters);
      cum_f += f_size_i;
    }
    kuf.push_back(kuf_i);

    if (blacs::mpirank == 0) {
      kuu.push_back(kernels[i]->envs_envs(
                global_sparse_descriptors[i], 
                global_sparse_descriptors[i],
                kernels[i]->kernel_hyperparameters));
    }
  }

  // Store square root of noise vector.
  Eigen::VectorXd noise_vector_sqrt = sqrt(noise_vector.array());
  Eigen::VectorXd global_noise_vector_sqrt = sqrt(global_noise_vector.array());

  // Synchronize, wait until all training structures are ready on all processors
  blacs::barrier();

  // Create distributed matrices
  { // specify the scope of the DistMatrix
  DistMatrix<double> A(f_size + u_size, u_size); // use the default blocking
  DistMatrix<double> b(f_size + u_size, 1);
  DistMatrix<double> Kuu_dist(  u_size, u_size);
  A = [](int i, int j){return 0.0;};
  b = [](int i, int j){return 0.0;};
  Kuu_dist = [](int i, int j){return 0.0;};
  blacs::barrier();

  int cum_u = 0;
  // Assign sparse set kernel matrix Kuu
  if (blacs::mpirank == 0) {
    for (int i = 0; i < n_kernels; i++) { 
      for (int r = 0; r < kuu[i].rows(); r++) {
        for (int c = 0; c < kuu[i].cols(); c++) {
          Kuu_dist.set(r + cum_u, c + cum_u, kuu[i](r, c));
        }
      }
      cum_u += kuu[i].rows();
    }
  }
  Kuu_dist.fence();

  int cum_f = 0;
  int local_f = 0;
  for (int t = 0; t < training_labels.size(); t++) { // training_structures is local subset
    int label_size = training_labels[t].size();

    if (nmin_struc < cum_f + label_size && cum_f < nmax_struc) {
      for (int l = 0; l < label_size; l++) {
        if (cum_f + l >= nmin_struc && cum_f + l < nmax_struc) {
          for (int i = 0; i < n_kernels; i++) { 
            // Assign a column of kuf to a row of A
            int u_size_single_kernel = global_sparse_descriptors[i].n_clusters;
            for (int c = 0; c < u_size_single_kernel; c++) { 
              A.set(cum_f + l, c + i * u_size_single_kernel, kuf[i](c, local_f + l) * noise_vector_sqrt(local_f + l));
            }
          }
    
          // Assign training label to y 
          b.set(cum_f + l, 0, training_labels[t](l) * global_noise_vector_sqrt(cum_f + l)); 
        }
      }
      local_f += label_size;
    }

    cum_f += label_size;
  }

  // Wait until the communication is done
  A.fence();
  b.fence();

  // Cholesky decomposition of Kuu and its inverse.
  DistMatrix<double> L = Kuu_dist.cholesky();
  L.fence();
  DistMatrix<double> L_inv_dist = L.triangular_invert('L');
  L_inv_dist.fence();
  DistMatrix<double> Kuu_inv_dist = L_inv_dist.matmul(L_inv_dist, 1.0, 'T', 'N'); 
  Kuu_inv_dist.fence();

  // Assign value to Kuu_inverse for varmap
  Kuu_inverse = Eigen::MatrixXd::Zero(u_size, u_size);
  for (int u = 0; u < u_size; u++) {
    for (int v = 0; v < u_size; v++) {
      Kuu_inverse(u, v) = Kuu_inv_dist(u, v);
    }
  }

  // Assign L.T to A matrix
  cum_f = f_size;
  for (int r = 0; r < u_size; r++) {
    if (blacs::mpirank == 0) {
      for (int c = 0; c < u_size; c++) {
        A.set(cum_f, c, L(c, r)); // the local_f is actually a global index of L.T
      }
    }
    cum_f += 1;
  }

  A.fence();
  b.fence();

  // QR factorize A to compute alpha
  DistMatrix<double> QR(u_size + f_size, u_size);
  std::vector<double> tau;
  std::tie(QR, tau) = A.qr();
  QR.fence();

  DistMatrix<double> R(u_size, u_size);                                 // Upper triangular R from QR
  R = [&QR](int i, int j) {return i > j ? 0 : QR(i, j);};
  blacs::barrier();
  DistMatrix<double> Rinv_dist = R.triangular_invert('U');              // Compute the inverse of R
  DistMatrix<double> Q_b = QR.QT_matmul(b, tau);                        // Q_b = Q^T * b
  DistMatrix<double> alpha_dist = Rinv_dist.matmul(Q_b, 1.0, 'N', 'N'); // alpha = R^-1 * Q_b

  // Assign value to alpha for mapping
  alpha = Eigen::VectorXd::Zero(u_size);
  for (int u = 0; u < u_size; u++) {
    alpha(u) = alpha_dist(u, 0);
  }

  }

  // finalize BLACS
  blacs::finalize();

}

void ParallelSGP ::predict_local_uncertainties(Structure &test_structure) {
  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;

  int n_sparse = Kuu_inverse.rows();
  Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(n_sparse, n_out);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = sparse_descriptors[i].n_clusters; 
    kernel_mat.block(count, 0, size, n_out) = kernels[i]->envs_struc(
        sparse_descriptors[i], test_structure.descriptors[i],
        kernels[i]->kernel_hyperparameters);
    count += size;
  }

  test_structure.mean_efs = kernel_mat.transpose() * alpha;
  std::vector<Eigen::VectorXd> local_uncertainties =
    this->compute_cluster_uncertainties(test_structure);
  test_structure.local_uncertainties = local_uncertainties;

}
