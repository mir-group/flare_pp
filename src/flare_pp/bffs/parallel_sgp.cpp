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
  std::cout << "noise vector n_labels, n_struc_labels " << n_labels << " " << n_struc_labels << std::endl;
  noise_vector.segment(n_labels, n_energy) =
      Eigen::VectorXd::Constant(n_energy, 1 / (energy_noise * energy_noise));
  noise_vector.segment(n_labels + n_energy, n_force) =
      Eigen::VectorXd::Constant(n_force, 1 / (force_noise * force_noise));
  noise_vector.segment(n_labels + n_energy + n_force, n_stress) =
      Eigen::VectorXd::Constant(n_stress, 1 / (stress_noise * stress_noise));
  std::cout << "noise vector size" << noise_vector.size() << std::endl;

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
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices) {

  load_local_training_data(training_cells, training_species, training_positions,
        training_labels, cutoff, descriptor_calculators, training_sparse_indices);

  std::cout << "Done loading data" << std::endl; 
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
    for (int i = 0; i < training_sparse_indices[k].size(); i++) {
      u_size += training_sparse_indices[k][i].size();
    }
  }

  Structure struc;
  int cum_f = 0;
  int cum_u = 0;
  int cum_b = 0;
 
  std::cout << "Start looping training set" << std::endl; 
  nmin_struc = world_rank * f_size_per_proc;
  if (world_rank == world_size) {
    nmax_struc = f_size;
  } else {
    nmax_struc = (world_rank + 1) * f_size_per_proc;
  }

  // Distribute the training structures and sparse envs
  for (int t = 0; t < training_cells.size(); t++) {
    struc = Structure(training_cells[t], training_species[t], 
            training_positions[t], cutoff, descriptor_calculators);
    int label_size = 1 + struc.noa * 3 + 6;
    struc.energy = training_labels[t].segment(cum_f, cum_f + 1);
    struc.forces = training_labels[t].segment(cum_f + 1, cum_f + 3 * struc.noa);
    struc.stresses = training_labels[t].segment(cum_f + 3 * struc.noa, label_size); 
    std::cout << "Training data created" << std::endl; 
    
    add_global_noise(struc); // for b
    std::cout << "added global noise" << std::endl; 

    initialize_global_sparse_descriptors(struc);
    std::cout << "initialized global sparse descriptors" << std::endl; 

    if (nmin_struc < cum_f + label_size || cum_f < nmax_struc) {
      // Collect local training structures for A
      std::cout << "initialized sparse descriptors" << std::endl;
      add_training_structure(struc);
      std::cout << "added local training structures" << std::endl; 
      std::cout << "noise vector size" << noise_vector.size() << std::endl;
    }
    cum_f += label_size;

    for (int i = 0; i < n_kernels; i++) { 
      // Collect all sparse envs u
      global_sparse_descriptors[i].add_clusters(
              struc.descriptors[i], training_sparse_indices[i][t]);
      std::cout << "added clusters to global sparse desc" << std::endl; 

      if (nmin_struc < cum_f + label_size || cum_f < nmax_struc) {
        // Collect local sparse descriptors for Kuu, 
        // use the local training structure's sparse descriptors
        initialize_sparse_descriptors(struc);
        sparse_descriptors[i].add_clusters(
                struc.descriptors[i], training_sparse_indices[i][t]);
        std::cout << "added local sparse descriptors" << std::endl; 
      }
    }

  }
}

void ParallelSGP::compute_matrices(
        const std::vector<Eigen::VectorXd> &training_labels,
        std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices) {

  // Build block of A, y, Kuu using distributed training structures
  std::cout << "\nStart compute_matrices" << std::endl; 
  std::vector<Eigen::MatrixXd> kuf, kuu;
  for (int i = 0; i < n_kernels; i++) {
    int u_size_single_kernel = global_sparse_descriptors[i].n_clusters;
    Eigen::MatrixXd kuf_i = Eigen::MatrixXd::Zero(u_size_single_kernel, f_size);
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
  std::cout << "computed kuf, kuu" << std::endl; 

  // Store square root of noise vector.
  Eigen::VectorXd noise_vector_sqrt = sqrt(noise_vector.array());
  Eigen::VectorXd global_noise_vector_sqrt = sqrt(global_noise_vector.array());

  // Synchronize, wait until all training structures are ready on all processors
  blacs::barrier();
  std::cout << "done barrier" << std::endl; 

  // Create distributed matrices
  { // specify the scope of the DistMatrix
  DistMatrix<double> A(f_size + u_size, u_size); // use the default blocking
  DistMatrix<double> b(f_size + u_size, 1);
  DistMatrix<double> Kuu_dist(  u_size, u_size);
  std::cout << "Created distmatrix" << std::endl; 

  int cum_f = 0;
  int local_f = 0;
  Eigen::VectorXd cum_u = Eigen::VectorXd::Zero(n_kernels);
  Eigen::VectorXd local_u = Eigen::VectorXd::Zero(n_kernels);
  for (int t = 0; t < training_structures.size(); t++) {
    int n_atoms = training_structures[t].noa;
    int label_size = 1 + n_atoms * 3 + 6;
    std::cout << "Start structure " << t << std::endl; 

//    if (cum_f >= nmin_struc && cum_f < nmax_struc) {
//      int c = 0;
//      int i = 0;
//      Kuu_dist.set(cum_u, c, kuu[i](c, local_u(i)) + Kuu_jitter);
//      break;
//    }
    
    // Assign sparse set kernel matrix Kuu
    for (int i = 0; i < n_kernels; i++) { 
      if (cum_f >= nmin_struc && cum_f < nmax_struc) {
      // if the 1st atom is local, then the structure and sparse envs are also local
        int u_size_single_kernel = global_sparse_descriptors[i].n_clusters;
        int head = i * u_size_single_kernel;
        for (int r = 0; r < training_sparse_indices[i][t].size(); r++) {
          for (int c = 0; c < u_size_single_kernel; c++) {
            if (cum_u(i) + r == c) {
              Kuu_dist.set(cum_u(i) + r + head, c + head, kuu[i](c, local_u(i) + r) + Kuu_jitter);
            } else {
              Kuu_dist.set(cum_u(i) + r + head, c + head, kuu[i](c, local_u(i) + r)); 
            }
          }
        }
        local_u(i) += training_sparse_indices[i][t].size();
      }
      cum_u(i) += training_sparse_indices[i][t].size();
    }
    std::cout << "Assigned kuu" << std::endl; 

    for (int l = 0; l < training_labels[t].size(); l++) {
      std::cout << "start label " << l << std::endl; 
      if (cum_f >= nmin_struc && cum_f < nmax_struc) {
        std::cout << "on current proc" << std::endl; 
        for (int i = 0; i < n_kernels; i++) { 
          // Assign a column of kuf to a row of A
          int u_size_single_kernel = global_sparse_descriptors[i].n_clusters;
          std::cout << "begin setting A" << std::endl; 
          for (int c = 0; c < u_size_single_kernel; c++) { 
            std::cout << "cum_f " << cum_f << std::endl; 
            std::cout << "ind_u " << c + i * u_size_single_kernel << std::endl; 
            std::cout << "kuf value " << kuf[i](c, local_f) << std::endl;
            std::cout << "noise vector size" << noise_vector_sqrt.size() << std::endl;
            std::cout << "local_f " << local_f << std::endl; 
            A.set(cum_f, c + i * u_size_single_kernel, kuf[i](c, local_f) * noise_vector_sqrt(local_f));
          }
        }
        local_f += 1;
  
        // Assign training label to y 
        std::cout << "begin setting b" << std::endl; 
        b.set(cum_f, 0, training_labels[t](l) * global_noise_vector[l]); 
      }
      cum_f += 1;
    }
    std::cout << "Assigned A, b" << std::endl; 

  }

  // Wait until the communication is done
  std::cout << "Start A fence" << std::endl; 
  A.fence();
  std::cout << "Start b fence" << std::endl; 
  b.fence();
  std::cout << "Start Kuu fence" << std::endl; 
  Kuu_dist.fence();
  std::cout << "Done Kuu fence" << std::endl; 

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
  std::cout << "start finalize" << std::endl; 
  blacs::finalize();
  std::cout << "done finalize" << std::endl; 

}


