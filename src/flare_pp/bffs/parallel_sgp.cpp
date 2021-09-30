#include "parallel_sgp.h"
#include <mpi.h>
#include <algorithm> // Random shuffle
#include <chrono>
#include <fstream> // File operations
#include <iomanip> // setprecision
#include <iostream>
#include <numeric> // Iota

#include <blacs.h>
#include <distmatrix.h>
#include <matrix.h>


#define MAXLINE 1024


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

//void ParallelSGP ::initialize_local_sparse_descriptors(const Structure &structure) {
//  if (local_sparse_descriptors.size() != 0)
//    return;
//
//  for (int i = 0; i < structure.descriptors.size(); i++) {
//    ClusterDescriptor empty_descriptor;
//    empty_descriptor.initialize_cluster(structure.descriptors[i].n_types,
//                                        structure.descriptors[i].n_descriptors);
//    local_sparse_descriptors.push_back(empty_descriptor);
//    std::vector<std::vector<int>> empty_indices;
//    sparse_indices.push_back(empty_indices); // NOTE: the sparse_indices should be of size n_kernels
//  }
//};


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

  // Save "1" vector for energy, force and stress noise, for likelihood gradient calculation
  e_noise_one.conservativeResize(n_labels + n_struc_labels);
  f_noise_one.conservativeResize(n_labels + n_struc_labels);
  s_noise_one.conservativeResize(n_labels + n_struc_labels);

  e_noise_one.segment(n_labels, n_struc_labels) = Eigen::VectorXd::Zero(n_struc_labels);
  f_noise_one.segment(n_labels, n_struc_labels) = Eigen::VectorXd::Zero(n_struc_labels);
  s_noise_one.segment(n_labels, n_struc_labels) = Eigen::VectorXd::Zero(n_struc_labels);

  e_noise_one.segment(n_labels, n_energy) = Eigen::VectorXd::Ones(n_energy);
  f_noise_one.segment(n_labels + n_energy, n_force) = Eigen::VectorXd::Ones(n_force);
  s_noise_one.segment(n_labels + n_energy + n_force, n_stress) = Eigen::VectorXd::Ones(n_stress);
 
  // Update label count.
  n_energy_labels += n_energy;
  n_force_labels += n_force;
  n_stress_labels += n_stress;
  n_labels += n_struc_labels;

  // Store training structure.
  training_structures.push_back(structure);
  n_strucs += 1;
}

Eigen::VectorXi ParallelSGP ::sparse_indices_by_type(int n_types, 
        std::vector<int> species, const std::vector<int> atoms) {
  // Compute the number of sparse envs of each type  
  // TODO: support two-body/three-body descriptors
  Eigen::VectorXi n_envs_by_type = Eigen::VectorXi::Zero(n_types);
  for (int a = 0; a < atoms.size(); a++) {
    int s = species[atoms[a]];
    n_envs_by_type(s)++;
  }
  return n_envs_by_type;
}

void ParallelSGP ::add_specific_environments(const Structure &structure,
                                      const std::vector<int> atoms) {

  // Gather clusters with central atom in the given list.
  std::vector<std::vector<std::vector<int>>> indices_1;
  for (int i = 0; i < n_kernels; i++){
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

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < n_kernels; i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i], indices_1[i]);
    cluster_descriptors.push_back(cluster_descriptor);
  }
  local_sparse_descriptors.push_back(cluster_descriptors);

}

void ParallelSGP ::add_global_noise(int n_energy, int n_force, int n_stress) {

  int n_struc_labels = n_energy + n_force + n_stress;

  // Update noise.
  global_noise_vector.conservativeResize(global_n_labels + n_struc_labels);
  global_noise_vector.segment(global_n_labels, n_energy) =
      Eigen::VectorXd::Constant(n_energy, 1 / (energy_noise * energy_noise));
  global_noise_vector.segment(global_n_labels + n_energy, n_force) =
      Eigen::VectorXd::Constant(n_force, 1 / (force_noise * force_noise));
  global_noise_vector.segment(global_n_labels + n_energy + n_force, n_stress) =
      Eigen::VectorXd::Constant(n_stress, 1 / (stress_noise * stress_noise));

  global_n_energy_labels += n_energy;
  global_n_force_labels += n_force;
  global_n_stress_labels += n_stress;
  global_n_labels += n_struc_labels;

}

void ParallelSGP::build(const std::vector<Structure> &training_strucs,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices,
        int n_types) {
 
  // initialize BLACS
  std::cout << "start blacs" << std::endl;
  blacs::initialize();
  std::cout << "done blacs" << std::endl;

  timer.tic();

  // Get the number of processes
  std::cout << "MPI comm size" << std::endl;
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  std::cout << "MPI comm rank" << std::endl;
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Compute the dimensions of the matrices Kuf and Kuu
  std::cout << "start computing f_size" << std::endl;
  f_size = 0;
  for (int i = 0; i < training_strucs.size(); i++) {
     f_size += training_strucs[i].energy.size() + training_strucs[i].forces.size() + training_strucs[i].stresses.size();
  }
  if (f_size % world_size == 0) {
    f_size_per_proc = f_size / world_size;
  } else {
    f_size_per_proc = f_size / world_size + 1;
  }

  std::cout << "start computing u_size" << std::endl;
  u_size = 0;
  for (int k = 0; k < n_kernels; k++) {
    std::cout << "start kernel " << k << std::endl;
    int u_kern = 0;
    for (int i = 0; i < training_sparse_indices[k].size(); i++) {
      u_kern += training_sparse_indices[k][i].size();
    }
    std::cout << "get u_kern" << std::endl;
    u_size += u_kern;
    u_size_single_kernel.push_back(u_kern);
    std::cout << "push_back " << u_kern << std::endl;
  }
  u_size_per_proc = u_size / world_size;
  std::cout << "done computing u_size" << std::endl;

  // Compute the range of structures covered by the current rank
  nmin_struc = world_rank * f_size_per_proc;
  nmin_envs = world_rank * u_size_per_proc;
  if (world_rank == world_size - 1) {
    nmax_struc = f_size;
    nmax_envs = u_size; 
  } else {
    nmax_struc = (world_rank + 1) * f_size_per_proc;
    nmax_envs = (world_rank + 1) * u_size_per_proc;
  }
  std::cout << "Start loading data" << std::endl;

  // load and distribute training structures, compute descriptors
  load_local_training_data(training_strucs, cutoff, descriptor_calculators, training_sparse_indices, n_types);
  timer.toc("load_local_training_data", blacs::mpirank);

  // compute kernel matrices from training data
  timer.tic();
  compute_kernel_matrices(training_strucs);
  timer.toc("compute_kernel_matrice", blacs::mpirank);

  timer.tic();
  update_matrices_QR(); 
  timer.toc("update_matrices_QR", blacs::mpirank);

  // TODO: finalize BLACS
  //blacs::finalize();

}

/* -------------------------------------------------------------------------
 *                Load training data and compute descriptors 
 * ------------------------------------------------------------------------- */

void ParallelSGP::load_local_training_data(const std::vector<Structure> &training_strucs,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices,
        int n_types) {

  // Distribute the training structures and sparse envs
  Structure struc;
  int cum_f = 0;
  global_n_labels = 0;

  // Compute the total number of clusters of each type of each kernel
  std::vector<std::vector<int>> n_clusters_by_type;
  for (int i = 0; i < n_kernels; i++) {
    std::vector<int> n_clusters_kern;
    for (int s = 0; s < n_types; s++) {
      n_clusters_kern.push_back(0);
    }
    n_clusters_by_type.push_back(n_clusters_kern);
    n_struc_clusters_by_type.push_back({});
  }

  for (int t = 0; t < training_strucs.size(); t++) {

    int label_size = training_strucs[t].n_labels();
    int noa = training_strucs[t].noa;
    assert (label_size == 1 + 3 * noa + 6); 
   
    int n_energy = 1;
    int n_forces = 3 * noa;
    int n_stress = 6;
    add_global_noise(n_energy, n_forces, n_stress); // for b

    // Compute the total number of clusters of each type
    for (int i = 0; i < n_kernels; i++) {
      Eigen::VectorXi n_envs_by_type = sparse_indices_by_type(n_types,
              training_strucs[t].species, training_sparse_indices[i][t]);
      n_struc_clusters_by_type[i].push_back(n_envs_by_type);
      for (int s = 0; s < n_types; s++) n_clusters_by_type[i][s] += n_envs_by_type(s);
    }

    // Collect local training structures for A, Kuf
    if (nmin_struc < cum_f + label_size && cum_f < nmax_struc) {
      std::cout << "Rank: " << blacs::mpirank << ", training struc " << t << std::endl;
      struc = Structure(training_strucs[t].cell, training_strucs[t].species, 
              training_strucs[t].positions, cutoff, descriptor_calculators);

      struc.energy = training_strucs[t].energy;
      struc.forces = training_strucs[t].forces;
      struc.stresses = training_strucs[t].stresses;
 
      add_training_structure(struc);

      std::vector<int> label_inds;
      for (int l = 0; l < label_size; l++) {
        if (cum_f + l >= nmin_struc && cum_f + l < nmax_struc) {
          label_inds.push_back(l);
        }
      }
      local_label_indices.push_back(label_inds);
      local_label_size += label_size;

      if (nmin_struc <= cum_f && cum_f < nmax_struc) {
        // avoid multiple procs add the same sparse envs
        add_specific_environments(struc, training_sparse_indices[0][t]);
      }
    }

    cum_f += label_size;
  }
  for (int i = 0; i < n_kernels; i++) {
    for (int s = 0; s < n_types; s++) {
      assert(n_clusters_by_type[i][s] >= world_size);
    }
  }

  blacs::barrier();
  
  gather_sparse_descriptors(n_clusters_by_type, training_strucs); 
}

/* -------------------------------------------------------------------------
 *                   Gather distributed sparse descriptors
 * ------------------------------------------------------------------------- */

void ParallelSGP::gather_sparse_descriptors(std::vector<std::vector<int>> n_clusters_by_type,
        const std::vector<Structure> &training_strucs) {

  for (int i = 0; i < n_kernels; i++) {
    // Assign global sparse descritors
    int n_descriptors = training_structures[0].descriptors[i].n_descriptors;
    int n_types = training_structures[0].descriptors[i].n_types;

    int cum_f, local_u, cum_u;
    std::vector<Eigen::MatrixXd> descriptors;
    std::vector<Eigen::VectorXd> descriptor_norms, cutoff_values;
    std::cout << "begin distmat" << std::endl;
    for (int s = 0; s < n_types; s++) {
      DistMatrix<double> dist_descriptors(n_clusters_by_type[i][s], n_descriptors);
      DistMatrix<double> dist_descriptor_norms(n_clusters_by_type[i][s], 1);
      DistMatrix<double> dist_cutoff_values(n_clusters_by_type[i][s], 1);
      std::cout << "create distmat" << std::endl;
      dist_descriptors = [](int i, int j){return 0.0;};
      dist_descriptor_norms = [](int i, int j){return 0.0;};
      dist_cutoff_values = [](int i, int j){return 0.0;};
      blacs::barrier();
      std::cout << "barrier" << std::endl;

      cum_f = 0;
      cum_u = 0;
      local_u = 0;
      bool lock = true;
      std::cout << "begin set element" << std::endl;
      for (int t = 0; t < training_strucs.size(); t++) {
        if (nmin_struc <= cum_f && cum_f < nmax_struc) {
          ClusterDescriptor cluster_descriptor = local_sparse_descriptors[local_u][i];
          for (int j = 0; j < n_struc_clusters_by_type[i][t](s); j++) {
            for (int d = 0; d < n_descriptors; d++) {
              dist_descriptors.set(cum_u + j, d, 
                      cluster_descriptor.descriptors[s](j, d), lock);
              dist_descriptor_norms.set(cum_u + j, 0, 
                      cluster_descriptor.descriptor_norms[s](j), lock);
              dist_cutoff_values.set(cum_u + j, 0, 
                      cluster_descriptor.cutoff_values[s](j), lock);
            }
          }
          local_u += 1;
        }
        cum_u += n_struc_clusters_by_type[i][t](s);
        int label_size = training_strucs[t].n_labels();
        cum_f += label_size;
      }
      std::cout << "finish setting" << std::endl;
      dist_descriptors.fence();
      dist_descriptor_norms.fence();
      dist_cutoff_values.fence();
      std::cout << "fence" << std::endl;

      int nrows = n_clusters_by_type[i][s];
      int ncols = n_descriptors;
      Eigen::MatrixXd type_descriptors = Eigen::MatrixXd::Zero(nrows, ncols);
      Eigen::VectorXd type_descriptor_norms = Eigen::VectorXd::Zero(nrows);
      Eigen::VectorXd type_cutoff_values = Eigen::VectorXd::Zero(nrows);

      std::cout << "Rank: " << blacs::mpirank << ", descriptor size: " << s << " " << n_clusters_by_type[i][s] << " " << n_descriptors << " " << std::endl;
      Matrix<double> descriptors_array(nrows, ncols);
      Matrix<double> descriptor_norms_array(nrows, 1);
      Matrix<double> cutoff_values_array(nrows, 1);
      std::cout << "created serial matrix" << std::endl;

      // TODO: change to &type_descriptors(0, 0) 
      dist_descriptors.allgather(descriptors_array.array.get());
      dist_descriptor_norms.allgather(descriptor_norms_array.array.get());
      dist_cutoff_values.allgather(cutoff_values_array.array.get());
      std::cout << "done allgather" << std::endl;
      for (int r = 0; r < n_clusters_by_type[i][s]; r++) {
        for (int c = 0; c < n_descriptors; c++) {
          type_descriptors(r, c) = descriptors_array(r, c);
        }
        type_descriptor_norms(r) = descriptor_norms_array(r, 0); 
        type_cutoff_values(r) = cutoff_values_array(r, 0);
      }
      std::cout << "begin push_back" << std::endl;
      descriptors.push_back(type_descriptors);
      descriptor_norms.push_back(type_descriptor_norms);
      cutoff_values.push_back(type_cutoff_values);
      std::cout << "added to local descriptor" << std::endl;

    }

    // Store sparse environments. 
    std::vector<int> cumulative_type_count = {0};
    int n_clusters = 0;
    for (int s = 0; s < n_types; s++) {
      cumulative_type_count.push_back(cumulative_type_count[s] + n_clusters_by_type[i][s]);
      n_clusters += n_clusters_by_type[i][s];
    }

    ClusterDescriptor cluster_desc;
    cluster_desc.initialize_cluster(n_types, n_descriptors);
    cluster_desc.descriptors = descriptors;
    cluster_desc.descriptor_norms = descriptor_norms;
    cluster_desc.cutoff_values = cutoff_values;
    cluster_desc.n_clusters_by_type = n_clusters_by_type[i];
    cluster_desc.cumulative_type_count = cumulative_type_count;
    cluster_desc.n_clusters = n_clusters;
    sparse_descriptors.push_back(cluster_desc);
  }

}

/* -------------------------------------------------------------------------
 *                   Compute kernel matrices and alpha 
 * ------------------------------------------------------------------------- */
void ParallelSGP ::stack_Kuf() {
  // Update Kuf kernels.
  int local_f_size = nmax_struc - nmin_struc;
  Kuf_local = Eigen::MatrixXd::Zero(u_size, local_f_size);

  int count = 0;
  for (int i = 0; i < Kuf_kernels.size(); i++) {
    int size = Kuf_kernels[i].rows();
    Kuf_local.block(count, 0, size, local_f_size) = Kuf_kernels[i];
    count += size;
  }
}

void ParallelSGP::compute_kernel_matrices(const std::vector<Structure> &training_strucs) {
  timer.tic();

  // Build block of A, y, Kuu using distributed training structures
  std::vector<Eigen::MatrixXd> kuf, kuu;
  int cum_f = 0;
  int cum_u = 0;
  int local_f_size = nmax_struc - nmin_struc;
  Kuf_local = Eigen::MatrixXd::Zero(u_size, local_f_size);

  for (int i = 0; i < n_kernels; i++) {
    int u_kern = u_size_single_kernel[i]; // sparse set size of kernel i
    assert(u_kern == sparse_descriptors[i].n_clusters);
    Eigen::MatrixXd kuf_i = Eigen::MatrixXd::Zero(u_kern, local_f_size);
    std::cout << "Rank: " << blacs::mpirank << ", starting computing kern_t" << std::endl;
    cum_f = 0;
    for (int t = 0; t < training_structures.size(); t++) {
      std::cout << "start training data " << t << std::endl;
      int f_size_i = local_label_indices[t].size();
      Eigen::MatrixXd kern_t = kernels[i]->envs_struc(
                  sparse_descriptors[i], 
                  training_structures[t].descriptors[i], 
                  kernels[i]->kernel_hyperparameters);

      std::cout << "Rank: " << blacs::mpirank << ", starting assigning kern_local" << std::endl;
      // Remove columns of kern_t that is not assigned to the current processor
      Eigen::MatrixXd kern_local = Eigen::MatrixXd::Zero(u_kern, f_size_i);
      for (int l = 0; l < f_size_i; l++) {
        kern_local.block(0, l, u_kern, 1) = kern_t.block(0, local_label_indices[t][l], u_kern, 1);
      }
      kuf_i.block(0, cum_f, u_kern, f_size_i) = kern_local; 
      cum_f += f_size_i;
      std::cout << "Rank: " << blacs::mpirank << ", done assigning kern_local" << std::endl;
    }
    std::cout << "push back" << std::endl;
    //kuf.push_back(kuf_i);

    // TODO: change here into stack_Kuf
    Kuf_local.block(cum_u, 0, u_kern, local_f_size) = kuf_i; 

    Kuf_kernels[i] = kuf_i;

    Kuu_kernels[i] = kernels[i]->envs_envs(
              sparse_descriptors[i], 
              sparse_descriptors[i],
              kernels[i]->kernel_hyperparameters);
    cum_u += u_kern;
  }

  // Only keep the chunk of noise_vector assigned to the current proc
  std::cout << "get noise" << std::endl;
  cum_f = 0;
  cum_u = 0;
  int cum_f_struc = 0;
  local_noise_vector = Eigen::VectorXd::Zero(local_f_size);
  local_e_noise_one = Eigen::VectorXd::Zero(local_f_size);
  local_f_noise_one = Eigen::VectorXd::Zero(local_f_size);
  local_s_noise_one = Eigen::VectorXd::Zero(local_f_size);
  local_labels = Eigen::VectorXd::Zero(local_f_size);
  for (int t = 0; t < training_structures.size(); t++) {
    int f_size_i = local_label_indices[t].size();
    for (int l = 0; l < f_size_i; l++) {
      local_noise_vector(cum_f + l) = noise_vector(cum_f_struc + local_label_indices[t][l]);
      local_e_noise_one(cum_f + l) = e_noise_one(cum_f_struc + local_label_indices[t][l]);
      local_f_noise_one(cum_f + l) = f_noise_one(cum_f_struc + local_label_indices[t][l]);
      local_s_noise_one(cum_f + l) = s_noise_one(cum_f_struc + local_label_indices[t][l]);
      local_labels(cum_f + l) = y(cum_f_struc + local_label_indices[t][l]);
    }
    cum_f += f_size_i;
    cum_f_struc += training_strucs[t].n_labels();
  }
}

void ParallelSGP::update_matrices_QR() {
  // Store square root of noise vector.
  Eigen::VectorXd noise_vector_sqrt = sqrt(local_noise_vector.array());
  Eigen::VectorXd global_noise_vector_sqrt = sqrt(global_noise_vector.array());

  // Synchronize, wait until all training structures are ready on all processors
  blacs::barrier();
  timer.toc("build local kuf kuu", blacs::mpirank);

  // Create distributed matrices
  // specify the scope of the DistMatrix
  timer.tic();

  std::cout << "f_size=" << f_size << " , u_size=" << u_size << std::endl;
  DistMatrix<double> A(f_size + u_size, u_size); // use the default blocking
  DistMatrix<double> b(f_size + u_size, 1);
  DistMatrix<double> Kuu_dist(u_size, u_size);
  std::cout << "Created A, b, Kuu_dist" << std::endl;
  A = [](int i, int j){return 0.0;};
  b = [](int i, int j){return 0.0;};
  Kuu_dist = [](int i, int j){return 0.0;};
  blacs::barrier();

  bool lock = true;
  cum_u = 0;
  // Assign sparse set kernel matrix Kuu
  for (int i = 0; i < n_kernels; i++) { 
    for (int r = 0; r < Kuu_kernels[i].rows(); r++) {
      for (int c = 0; c < Kuu_kernels[i].cols(); c++) {
        if (Kuu_dist.islocal(r + cum_u, c + cum_u)) { // only set the local part
          Kuu_dist.set(r + cum_u, c + cum_u, Kuu_kernels[i](r, c), lock);
        }
      }
    }
    cum_u += Kuu_kernels[i].rows();
  }
  Kuu_dist.fence();

  // TODO: Kuu is only needed for debug and unit test 
  Matrix<double> Kuu_array(u_size, u_size);
  Kuu_dist.allgather(Kuu_array.array.get());
  Kuu = Eigen::Map<Eigen::MatrixXd>(Kuu_array.array.get(), u_size, u_size);
  std::cout << "Allgathered Kuu" << std::endl;

  timer.toc("build Kuu_dist", blacs::mpirank);

  // Cholesky decomposition of Kuu and its inverse.
  timer.tic();
  Eigen::LLT<Eigen::MatrixXd> chol(
      Kuu + Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols()));

  // Get the inverse of Kuu from Cholesky decomposition.
  Eigen::MatrixXd Kuu_eye = Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());
  Eigen::MatrixXd L = chol.matrixL();
  L_inv = chol.matrixL().solve(Kuu_eye);
  Kuu_inverse = L_inv.transpose() * L_inv;
  L_diag = L_inv.diagonal();

  timer.toc("cholestky, tri_inv, matmul", blacs::mpirank);

  // Assign Lambda * Kfu to A
  timer.tic();

  cum_f = 0;
  int local_f_full = 0;
  int local_f = 0;
  Eigen::MatrixXd noise_kfu = noise_vector_sqrt.asDiagonal() * Kuf_local.transpose();
  A.collect(&noise_kfu(0, 0), 0, 0, f_size, u_size, f_size_per_proc, u_size, nmax_struc - nmin_struc); 
  Eigen::VectorXd noise_labels = noise_vector_sqrt.asDiagonal() * local_labels;
  b.collect(&noise_labels(0), 0, 0, f_size, 1, f_size_per_proc, 1, nmax_struc - nmin_struc); 
  // TODO: for debug
  b_debug = Eigen::VectorXd::Zero(f_size+u_size);
  b.gather(&b_debug(0));
  std::cout << "Collected Kuf to A" << std::endl;

  // Wait until the communication is done
  A.fence();
  b.fence();

  // TODO: Kuf is for debugging
  Kuf = Eigen::MatrixXd::Zero(u_size, f_size);
  for (int r = 0; r < u_size; r++) {
    for (int c = 0; c < f_size; c++) {
      Kuf(r, c) = A(c, r, lock);
    }
  }
  timer.toc("set A & b", blacs::mpirank);

  // Copy L.T to A using scatter function
  timer.tic();
  Eigen::MatrixXd L_T;
  int mb, nb, lld;
  if (blacs::mpirank == 0) {
    L_T = L.transpose();
    mb = nb = lld = u_size;
  } else {
    mb = nb = lld = 0; 
  }
  A.scatter(&L_T(0,0), f_size, 0, u_size, u_size);

  A.fence();

  timer.toc("set L.T to A", blacs::mpirank);

  // QR factorize A to compute alpha
  timer.tic();
  DistMatrix<double> QR(u_size + f_size, u_size);
  std::vector<double> tau;
  std::tie(QR, tau) = A.qr();
  QR.fence();

  timer.toc("QR", blacs::mpirank);

  timer.tic();

  DistMatrix<double> R_dist(u_size, u_size);                                 // Upper triangular R from QR
  R_dist = [&QR](int i, int j) {return i > j ? 0 : QR(i, j, true);};
  blacs::barrier();

  // Directly use triangular_solve to get alpha: R * alpha = Q_b
  DistMatrix<double> Qb_dist = QR.QT_matmul(b, tau);                        // Q_b = Q^T * b

  Matrix<double> Qb_array(f_size + u_size, 1);
  std::cout << "Begin allgather Qb" << std::endl;
  Qb_dist.allgather(Qb_array.array.get());
  Q_b = Eigen::Map<Eigen::VectorXd>(Qb_array.array.get(), f_size + u_size).segment(0, u_size);
  std::cout << "Allgathered Qb" << std::endl;

  Matrix<double> R_array(u_size, u_size);
  R_dist.allgather(R_array.array.get());
  R = Eigen::Map<Eigen::MatrixXd>(R_array.array.get(), u_size, u_size);
  std::cout << "Allgathered R" << std::endl;

  // Using Lapack triangular solver to temporarily avoid the numerical issue 
  // with Scalapack block algorithm with ill-conditioned matrix
  R_inv = R.triangularView<Eigen::Upper>().solve(Kuu_eye);
  R_inv_diag = R_inv.diagonal();
  alpha = R_inv * Q_b;

  timer.toc("get alpha", blacs::mpirank);
}

void ParallelSGP ::set_hyperparameters(Eigen::VectorXd hyps) {
  timer.tic();

  // Reset Kuu and Kuf matrices.
  int n_hyps, hyp_index = 0;
  Eigen::VectorXd new_hyps;

  std::vector<Eigen::MatrixXd> Kuu_grad, Kuf_grad;
  for (int i = 0; i < n_kernels; i++) {
    n_hyps = kernels[i]->kernel_hyperparameters.size();
    new_hyps = hyps.segment(hyp_index, n_hyps);

    Kuu_grad = kernels[i]->Kuu_grad(sparse_descriptors[i], Kuu_kernels[i], new_hyps);
    Kuf_grad = kernels[i]->Kuf_grad(sparse_descriptors[i], training_structures,
                                    i, Kuf_kernels[i], new_hyps);

    Kuu_kernels[i] = Kuu_grad[0];
    Kuf_kernels[i] = Kuf_grad[0];

    kernels[i]->set_hyperparameters(new_hyps);
    hyp_index += n_hyps;
  }
  timer.toc("set_hyp: update Kuf Kuu");

  // Stack Kuf_local
  timer.tic();
  stack_Kuf();
  timer.toc("set_hyp: stack Kuf Kuu");

  // Update noise vector
  timer.tic();
  hyperparameters = hyps;
  energy_noise = hyps(hyp_index);
  force_noise = hyps(hyp_index + 1);
  stress_noise = hyps(hyp_index + 2);

  Eigen::VectorXd local_noise_vector = 1 / (energy_noise * energy_noise) * local_e_noise_one 
               + 1 / (force_noise * force_noise) * local_f_noise_one 
               + 1 / (stress_noise * stress_noise) * local_s_noise_one; 
   
  // TODO: global_n_labels == f_size
  global_noise_vector = Eigen::VectorXd::Zero(global_n_labels);
  DistMatrix<double> noise_vector_dist(f_size, 1);
  noise_vector_dist.collect(&local_noise_vector(0), 0, 0, f_size, 1, f_size_per_proc, 1, nmax_struc - nmin_struc);
  noise_vector_dist.gather(&global_noise_vector(0));

  timer.toc("set_hyp: update noise");

  // Update remaining matrices.
  timer.tic();
  update_matrices_QR();
  timer.toc("set_hyp: update_matrices_QR");
}

/* -------------------------------------------------------------------------
 *                    Predict mean and uncertainties 
 * ------------------------------------------------------------------------- */

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

/* -------------------------------------------------------------------------
 *                          Compute likelihood
 * ------------------------------------------------------------------------- */

void ParallelSGP ::compute_likelihood_stable() {
  // initialize BLACS
  blacs::initialize();

  std::cout << "build Kuf_dist" << std::endl;
  DistMatrix<double> Kfu_dist(f_size, u_size);
  Eigen::MatrixXd Kfu_local = Kuf_local.transpose();
  Kfu_dist.collect(&Kfu_local(0, 0), 0, 0, f_size, u_size, f_size_per_proc, u_size, nmax_struc - nmin_struc); 

  std::cout << "build alpha_dist" << std::endl;
  DistMatrix<double> alpha_dist(u_size, 1);
  alpha_dist.scatter(&alpha(0), 0, 0, u_size, 1);

  std::cout << "build K_alpha_dist" << std::endl;
  DistMatrix<double> K_alpha_dist(f_size, 1);
  K_alpha_dist = Kfu_dist.matmul(alpha_dist, 1.0, 'N', 'N');
  Eigen::VectorXd K_alpha = Eigen::VectorXd::Zero(f_size);
  K_alpha_dist.gather(&K_alpha(0));

  std::cout << "build y_dist" << std::endl;
  DistMatrix<double> y_dist(f_size, 1);
  y_dist.collect(&local_labels(0), 0, 0, f_size, 1, f_size_per_proc, 1, nmax_struc - nmin_struc);
  Eigen::VectorXd y_global = Eigen::VectorXd::Zero(f_size);
  y_dist.gather(&y_global(0));

  y_K_alpha = Eigen::VectorXd::Zero(f_size);

  // Compute log marginal likelihood
  log_marginal_likelihood = 0;
  if (blacs::mpirank == 0) {
    y_K_alpha = y_global - K_alpha;
    data_fit =
        -(1. / 2.) * y_global.transpose() * global_noise_vector.cwiseProduct(y_K_alpha);
    constant_term = -(1. / 2.) * global_n_labels * log(2 * M_PI);

    // Compute complexity penalty.
    double noise_det = - 2 * (global_n_energy_labels * log(abs(energy_noise))
            + global_n_force_labels * log(abs(force_noise))
            + global_n_stress_labels * log(abs(stress_noise)));

    assert(L_diag.size() == R_inv_diag.size());
    double Kuu_inv_det = 0;
    double sigma_inv_det = 0;
    for (int i = 0; i < L_diag.size(); i++) {
      Kuu_inv_det -= 2 * log(abs(L_diag(i)));
      sigma_inv_det += 2 * log(abs(R_inv_diag(i)));
    }

    complexity_penalty = (1. / 2.) * (noise_det + Kuu_inv_det + sigma_inv_det);
    log_marginal_likelihood = complexity_penalty + data_fit + constant_term;
  }
}

/* -------------------------------------------------------------------------
 *              Compute likelihood gradient of hyperparameters
 * ------------------------------------------------------------------------- */

double ParallelSGP ::compute_likelihood_gradient_stable(bool precomputed_KnK) {
  // initialize BLACS
  blacs::initialize();

  // Compute likelihood
  compute_likelihood_stable();

  Sigma = R_inv * R_inv.transpose();

  // Compute likelihood gradient of kernel hyps such as signal variance
  int n_hyps_total = hyperparameters.size();
  likelihood_gradient = Eigen::VectorXd::Zero(n_hyps_total);
  likelihood_gradient += compute_like_grad_of_kernel_hyps();

  // Compute likelihood gradient of energy, force, stress noises
  likelihood_gradient.segment(n_hyps_total - 3, 3) += compute_like_grad_of_noise();

  // finalize BLACS
  std::cout << "finalize" << std::endl;
  //blacs::finalize();
  
  return log_marginal_likelihood;
}

Eigen::VectorXd ParallelSGP ::compute_like_grad_of_kernel_hyps() {
  std::cout << "build Kuf_dist" << std::endl;
  DistMatrix<double> Kfu_dist(f_size, u_size);
  Eigen::MatrixXd Kfu_local = Kuf_local.transpose();
  Kfu_dist.collect(&Kfu_local(0, 0), 0, 0, f_size, u_size, f_size_per_proc, u_size, nmax_struc - nmin_struc); 

  std::cout << "build alpha_dist" << std::endl;
  DistMatrix<double> alpha_dist(u_size, 1);
  alpha_dist.scatter(&alpha(0), 0, 0, u_size, 1);

  // Compute Kuu and Kuf matrices and gradients.
  int n_hyps_total = hyperparameters.size();
  std::vector<Eigen::MatrixXd> Kuu_grad, Kuf_grad, Kuu_grads, Kuf_grads;
  int n_hyps, hyp_index = 0, grad_index = 0;
  Eigen::VectorXd hyps_curr;

  int count = 0;
  Eigen::VectorXd complexity_grad = Eigen::VectorXd::Zero(n_hyps_total);
  Eigen::VectorXd datafit_grad = Eigen::VectorXd::Zero(n_hyps_total);
  Eigen::VectorXd likelihood_grad = Eigen::VectorXd::Zero(n_hyps_total);
  for (int i = 0; i < n_kernels; i++) {
    n_hyps = kernels[i]->kernel_hyperparameters.size();
    hyps_curr = hyperparameters.segment(hyp_index, n_hyps);
    int size = Kuu_kernels[i].rows();
    Kuu_grad = kernels[i]->Kuu_grad(sparse_descriptors[i], Kuu_kernels[i], hyps_curr);
    Kuf_grad = kernels[i]->Kuf_grad(sparse_descriptors[i], 
              training_structures, i, Kuf_kernels[i], hyps_curr);

    Eigen::MatrixXd Kuu_i = Kuu_grad[0];

    for (int j = 0; j < n_hyps; j++) {
      Eigen::MatrixXd dKuu = Eigen::MatrixXd::Zero(u_size, u_size);
      dKuu.block(count, count, size, size) = Kuu_grad[j + 1];

      // Compute and return the derivative of Kuf_grad * e/f/s_noise_one * Kuf.transpose()
      Eigen::MatrixXd dKuf_local = Kuf_grad[1]; // TODO: only apply for inner product kernel
      Eigen::VectorXd noise_one_local = 
          local_e_noise_one / (energy_noise * energy_noise) + 
          local_f_noise_one / (force_noise * force_noise) + 
          local_s_noise_one / (stress_noise * stress_noise);
      Eigen::MatrixXd dnK_local = noise_one_local.asDiagonal() * dKuf_local.transpose();
    
      DistMatrix<double> dnK_dist(f_size, u_size); // Build a large matrix
      dnK_dist.collect(&dnK_local(0, 0), 0, count, f_size, size, f_size_per_proc, size, nmax_struc - nmin_struc);
      DistMatrix<double> dKnK_dist(u_size, u_size);
      dKnK_dist = dnK_dist.matmul(Kfu_dist, 1.0, 'T', 'N');
    
      Eigen::MatrixXd dK_noise_K = Eigen::MatrixXd::Zero(u_size, u_size); 
      dKnK_dist.gather(&dK_noise_K(0, 0));

      // Derivative of complexity over sigma
      if (blacs::mpirank == 0) {
        Eigen::MatrixXd Pi_mat = dK_noise_K + dK_noise_K.transpose() + dKuu;

        // TODO: the 2nd term is not very stable numerically, because dK_noise_K is very large, and Kuu_grads is small
        complexity_grad(hyp_index + j) += 1./2. * (Kuu_i.inverse() * Kuu_grad[j + 1]).trace() - 1./2. * (Pi_mat * Sigma).trace(); 
      }

      // Derivative of data_fit over sigma
      Eigen::MatrixXd dKfu_local = Kuf_grad[j + 1].transpose();
      DistMatrix<double> dKfu_dist(f_size, u_size);
      dKfu_dist.collect(&dKfu_local(0, 0), 0, 0, f_size, u_size, f_size_per_proc, u_size, nmax_struc - nmin_struc); 

      DistMatrix<double> dK_alpha_dist(f_size, 1);
      dK_alpha_dist = dKfu_dist.matmul(alpha_dist, 1.0, 'N', 'N');
      Eigen::VectorXd dK_alpha = Eigen::VectorXd::Zero(f_size);
      dK_alpha_dist.gather(&dK_alpha(0));

      if (blacs::mpirank == 0) {
        datafit_grad(hyp_index + j) +=
            dK_alpha.transpose() * global_noise_vector.cwiseProduct(y_K_alpha);
        datafit_grad(hyp_index + j) += 
            - 1./2. * alpha.transpose() * dKuu * alpha;
        likelihood_grad(hyp_index + j) += complexity_grad(hyp_index + j) + datafit_grad(hyp_index + j); 
      }
    }
    count += size;
    hyp_index += n_hyps;
  } 
  assert(hyp_index == n_hyps_total - 3);
  return likelihood_grad;
}

Eigen::VectorXd ParallelSGP ::compute_like_grad_of_noise() {
  // Derivative of complexity over noise
  double en3 = energy_noise * energy_noise * energy_noise;
  double fn3 = force_noise * force_noise * force_noise;
  double sn3 = stress_noise * stress_noise * stress_noise;
 
  std::cout << "compute_KnK" << std::endl;
  compute_KnK();
  blacs::barrier(); 

  Eigen::VectorXd complexity_grad = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd datafit_grad = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd likelihood_grad = Eigen::VectorXd::Zero(3);

  if (blacs::mpirank == 0) {
    complexity_grad(0) = - global_n_energy_labels / energy_noise 
        + (KnK_e * Sigma).trace() / en3;
    complexity_grad(1) = - global_n_force_labels / force_noise 
        + (KnK_f * Sigma).trace() / fn3;
    complexity_grad(2) = - global_n_stress_labels / stress_noise 
        + (KnK_s * Sigma).trace() / sn3;
  }
 
  // Derivative of data_fit over noise  
  DistMatrix<double> e_noise_one_dist(f_size, 1);
  e_noise_one_dist.collect(&local_e_noise_one(0), 0, 0, f_size, 1, f_size_per_proc, 1, nmax_struc - nmin_struc);
  Eigen::VectorXd global_e_noise_one = Eigen::VectorXd::Zero(f_size);
  e_noise_one_dist.gather(&global_e_noise_one(0));

  DistMatrix<double> f_noise_one_dist(f_size, 1);
  f_noise_one_dist.collect(&local_f_noise_one(0), 0, 0, f_size, 1, f_size_per_proc, 1, nmax_struc - nmin_struc);
  Eigen::VectorXd global_f_noise_one = Eigen::VectorXd::Zero(f_size);
  f_noise_one_dist.gather(&global_f_noise_one(0));

  DistMatrix<double> s_noise_one_dist(f_size, 1);
  s_noise_one_dist.collect(&local_s_noise_one(0), 0, 0, f_size, 1, f_size_per_proc, 1, nmax_struc - nmin_struc);
  Eigen::VectorXd global_s_noise_one = Eigen::VectorXd::Zero(f_size);
  s_noise_one_dist.gather(&global_s_noise_one(0));

  if (blacs::mpirank == 0) {
    datafit_grad(0) = y_K_alpha.transpose() * global_e_noise_one.cwiseProduct(y_K_alpha);
    datafit_grad(0) /= en3;
    datafit_grad(1) = y_K_alpha.transpose() * global_f_noise_one.cwiseProduct(y_K_alpha);
    datafit_grad(1) /= fn3;
    datafit_grad(2) = y_K_alpha.transpose() * global_s_noise_one.cwiseProduct(y_K_alpha);
    datafit_grad(2) /= sn3;

    likelihood_grad(0) += complexity_grad(0) + datafit_grad(0);
    likelihood_grad(1) += complexity_grad(1) + datafit_grad(1);
    likelihood_grad(2) += complexity_grad(2) + datafit_grad(2);
  }
  return likelihood_grad;
}

void ParallelSGP ::compute_KnK() {
  // Compute and return Kuf * e/f/s_noise_one * Kuf.transpose()
  std::cout << "build Kuf_dist" << std::endl;
  DistMatrix<double> Kfu_dist(f_size, u_size);
  Eigen::MatrixXd Kfu_local = Kuf_local.transpose();
  Kfu_dist.collect(&Kfu_local(0, 0), 0, 0, f_size, u_size, f_size_per_proc, u_size, nmax_struc - nmin_struc); 
  
  // Compute Kuf * e/f/s_noise_one_local and collect to distributed matrix
  Eigen::MatrixXd enK_local = local_e_noise_one.asDiagonal() * Kuf_local.transpose();
  DistMatrix<double> enK_dist(f_size, u_size);
  enK_dist.collect(&enK_local(0, 0), 0, 0, f_size, u_size, f_size_per_proc, u_size, nmax_struc - nmin_struc);

  Eigen::MatrixXd fnK_local = local_f_noise_one.asDiagonal() * Kuf_local.transpose();
  DistMatrix<double> fnK_dist(f_size, u_size);
  fnK_dist.collect(&fnK_local(0, 0), 0, 0, f_size, u_size, f_size_per_proc, u_size, nmax_struc - nmin_struc);

  Eigen::MatrixXd snK_local = local_s_noise_one.asDiagonal() * Kuf_local.transpose();
  DistMatrix<double> snK_dist(f_size, u_size);
  snK_dist.collect(&snK_local(0, 0), 0, 0, f_size, u_size, f_size_per_proc, u_size, nmax_struc - nmin_struc);

  // Compute Kn * Kuf.tranpose() 
  DistMatrix<double> KnK_e_dist(u_size, u_size);
  KnK_e_dist = Kfu_dist.matmul(enK_dist, 1.0, 'T', 'N');

  DistMatrix<double> KnK_f_dist(u_size, u_size);
  KnK_f_dist = Kfu_dist.matmul(fnK_dist, 1.0, 'T', 'N');

  DistMatrix<double> KnK_s_dist(u_size, u_size);
  KnK_s_dist = Kfu_dist.matmul(snK_dist, 1.0, 'T', 'N');

  // Gather to get the serial matrix
  KnK_e = Eigen::MatrixXd::Zero(u_size, u_size);
  KnK_e_dist.gather(&KnK_e(0,0));

  KnK_f = Eigen::MatrixXd::Zero(u_size, u_size);
  KnK_f_dist.gather(&KnK_f(0,0));

  KnK_s = Eigen::MatrixXd::Zero(u_size, u_size);
  KnK_s_dist.gather(&KnK_s(0,0));
}
