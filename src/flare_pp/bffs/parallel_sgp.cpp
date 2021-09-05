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

  global_n_labels += n_struc_labels;

}

void ParallelSGP::build(const std::vector<Structure> &training_strucs,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices,
        int n_types) {
 
  // initialize BLACS
  blacs::initialize();

  double duration = 0;
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Compute the dimensions of the matrices Kuf and Kuu
  f_size = 0;
  for (int i = 0; i < training_strucs.size(); i++) {
     f_size += training_strucs[i].energy.size() + training_strucs[i].forces.size() + training_strucs[i].stresses.size();
  }
  if (f_size % world_size == 0) {
    f_size_per_proc = f_size / world_size;
  } else {
    f_size_per_proc = f_size / world_size + 1;
  }

  u_size = 0;
  for (int k = 0; k < n_kernels; k++) {
    u_size_single_kernel = 0; // TODO: here assumes that each kernel has the same set of sparse envs
    for (int i = 0; i < training_sparse_indices[k].size(); i++) {
      u_size += training_sparse_indices[k][i].size();
      u_size_single_kernel += training_sparse_indices[k][i].size();
    }
  }
  u_size_per_proc = u_size / world_size;

  // Compute the range of structures covered by the current rank
  nmin_struc = world_rank * f_size_per_proc;
  nmin_envs = world_rank * u_size_per_proc;
  if (world_rank == world_size - 1) {
    nmax_struc = f_size;
    nmin_envs = u_size_single_kernel; 
  } else {
    nmax_struc = (world_rank + 1) * f_size_per_proc;
    nmax_envs = (world_rank + 1) * u_size_per_proc;
  }
  std::cout << "Start loading data" << std::endl;

  // load and distribute training structures, compute descriptors
  load_local_training_data(training_strucs, cutoff, descriptor_calculators, training_sparse_indices, n_types);
  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Rank: " << blacs::mpirank << ", load_local_training_data: " << duration << " ms" << std::endl;

  // compute kernel matrices from training data
  duration = 0;
  t1 = std::chrono::high_resolution_clock::now();
  compute_matrices(training_strucs);
  t2 = std::chrono::high_resolution_clock::now();
  duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Rank: " << blacs::mpirank << ", compute_matrices: " << duration << " ms" << std::endl;

  // finalize BLACS
  blacs::finalize();

}

void ParallelSGP::load_local_training_data(const std::vector<Structure> &training_strucs,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices,
        int n_types) {

  double duration = 0;
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  double time_build, time_noise, time_add_struc, time_add_env;
  std::chrono::high_resolution_clock::time_point t1_inner, t2_inner, t3_inner, t4_inner, t5_inner;

  // Distribute the training structures and sparse envs
  Structure struc;
  int cum_f = 0;
  global_n_labels = 0;

  // Compute the total number of clusters of each type
  std::vector<int> n_clusters_by_type;
  for (int s = 0; s < n_types; s++) n_clusters_by_type.push_back(0);

  for (int t = 0; t < training_strucs.size(); t++) {
    t1_inner = std::chrono::high_resolution_clock::now();

    int label_size = training_strucs[t].n_labels();
    int noa = training_strucs[t].noa;
    assert (label_size == 1 + 3 * noa + 6); 
   
    int n_energy = 1;
    int n_forces = 3 * noa;
    int n_stress = 6;
    add_global_noise(n_energy, n_forces, n_stress); // for b

    Eigen::VectorXi n_envs_by_type = sparse_indices_by_type(n_types,
            training_strucs[t].species, training_sparse_indices[0][t]);
    n_struc_clusters_by_type.push_back(n_envs_by_type);
    for (int s = 0; s < n_types; s++) n_clusters_by_type[s] += n_envs_by_type(s);

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
  for (int s = 0; s < n_types; s++) assert(n_clusters_by_type[s] >= world_size);

  blacs::barrier();
  
  gather_sparse_descriptors(n_clusters_by_type, training_strucs, 
          training_sparse_indices);
}

void ParallelSGP::gather_sparse_descriptors(std::vector<int> n_clusters_by_type,
        const std::vector<Structure> &training_strucs,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices) {

  // Assign global sparse descritors
  // TODO: allow multiple descriptors
  int n_descriptors = training_structures[0].descriptors[0].n_descriptors;
  int n_types = training_structures[0].descriptors[0].n_types;

  int cum_f, local_u, cum_u;
  std::vector<Eigen::MatrixXd> descriptors;
  std::vector<Eigen::VectorXd> descriptor_norms, cutoff_values;
  int kernel_ind = 0;
  std::cout << "begin distmat" << std::endl;
  for (int s = 0; s < n_types; s++) {
    std::cout << "type " << s << " of " << n_types << std::endl;
    DistMatrix<double> dist_descriptors(n_clusters_by_type[s], n_descriptors);
    DistMatrix<double> dist_descriptor_norms(n_clusters_by_type[s], 1);
    DistMatrix<double> dist_cutoff_values(n_clusters_by_type[s], 1);
    std::cout << "create distmat" << std::endl;
    dist_descriptors = [](int i, int j){return 0.0;};
    dist_descriptor_norms = [](int i, int j){return 0.0;};
    dist_cutoff_values = [](int i, int j){return 0.0;};
    std::cout << "assigned 0" << std::endl;
    blacs::barrier();
    std::cout << "barrier" << std::endl;

    cum_f = 0;
    cum_u = 0;
    local_u = 0;
    bool lock = true;
    std::cout << "begin set element" << std::endl;
    for (int t = 0; t < training_strucs.size(); t++) {
      if (nmin_struc <= cum_f && cum_f < nmax_struc) {
        ClusterDescriptor cluster_descriptor = local_sparse_descriptors[local_u][kernel_ind];
        for (int j = 0; j < n_struc_clusters_by_type[t](s); j++) {
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
      cum_u += n_struc_clusters_by_type[t](s);
      int label_size = training_strucs[t].n_labels();
      cum_f += label_size;
    }
    std::cout << "finish setting" << std::endl;
    dist_descriptors.fence();
    dist_descriptor_norms.fence();
    dist_cutoff_values.fence();
    std::cout << "fence" << std::endl;

    int nrows = n_clusters_by_type[s];
    int ncols = n_descriptors;
    Eigen::MatrixXd type_descriptors = Eigen::MatrixXd::Zero(nrows, ncols);
    Eigen::VectorXd type_descriptor_norms = Eigen::VectorXd::Zero(nrows);
    Eigen::VectorXd type_cutoff_values = Eigen::VectorXd::Zero(nrows);

    std::cout << "Rank: " << blacs::mpirank << ", descriptor size: " << s << " " << n_clusters_by_type[s] << " " << n_descriptors << " " << std::endl;
    Matrix<double> descriptors_array(nrows, ncols);
    Matrix<double> descriptor_norms_array(nrows, 1);
    Matrix<double> cutoff_values_array(nrows, 1);
    std::cout << "created serial matrix" << std::endl;

    dist_descriptors.allgather(descriptors_array.array.get());
    std::cout << "something" << std::endl;
    dist_descriptor_norms.allgather(descriptor_norms_array.array.get());
    std::cout << "something 1" << std::endl;
    dist_cutoff_values.allgather(cutoff_values_array.array.get());
    std::cout << "done allgather" << std::endl;
    // TODO: use Eigen::Map to save memory
    for (int r = 0; r < n_clusters_by_type[s]; r++) {
      for (int c = 0; c < n_descriptors; c++) {
        type_descriptors(r, c) = descriptors_array(r, c);//dist_descriptors(r, c, lock);
      }
      type_descriptor_norms(r) = descriptor_norms_array(r, 0); //dist_descriptor_norms(r, 0, lock);
      type_cutoff_values(r) = cutoff_values_array(r, 0); //dist_cutoff_values(r, 0, lock);
    }
    std::cout << "begin push_back" << std::endl;
    descriptors.push_back(type_descriptors);
    descriptor_norms.push_back(type_descriptor_norms);
    cutoff_values.push_back(type_cutoff_values);
    std::cout << "added to local descriptor" << std::endl;

  }

  // Store sparse environments. 
  // TODO: support multiple kernels
  std::vector<int> cumulative_type_count = {0};
  int n_clusters = 0;
  for (int s = 0; s < n_types; s++) {
    cumulative_type_count.push_back(cumulative_type_count[s] + n_clusters_by_type[s]);
    n_clusters += n_clusters_by_type[s];
  }

  for (int i = 0; i < n_kernels; i++) {
    ClusterDescriptor cluster_desc;
    cluster_desc.initialize_cluster(n_types, n_descriptors);
    cluster_desc.descriptors = descriptors;
    cluster_desc.descriptor_norms = descriptor_norms;
    cluster_desc.cutoff_values = cutoff_values;
    cluster_desc.n_clusters_by_type = n_clusters_by_type;
    cluster_desc.cumulative_type_count = cumulative_type_count;
    cluster_desc.n_clusters = n_clusters;
    sparse_descriptors.push_back(cluster_desc);
  }

}

void ParallelSGP::compute_matrices(const std::vector<Structure> &training_strucs) {
  double duration = 0;
  std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
  std::chrono::high_resolution_clock::time_point t1_inner, t2_inner;

  // Build block of A, y, Kuu using distributed training structures
  std::vector<Eigen::MatrixXd> kuf, kuu;
  int cum_f = 0;
  int cum_u = 0;
  int local_f_size = nmax_struc - nmin_struc;

  for (int i = 0; i < n_kernels; i++) {
    t1_inner = std::chrono::high_resolution_clock::now();
    assert(u_size_single_kernel == sparse_descriptors[i].n_clusters);
    Eigen::MatrixXd kuf_i = Eigen::MatrixXd::Zero(u_size_single_kernel, local_f_size);
    for (int t = 0; t < training_structures.size(); t++) {
      int f_size_i = local_label_indices[t].size();
      Eigen::MatrixXd kern_t = kernels[i]->envs_struc(
                  sparse_descriptors[i], 
                  training_structures[t].descriptors[i], 
                  kernels[i]->kernel_hyperparameters);

      // Remove columns of kern_t that is not assigned to the current processor
      Eigen::MatrixXd kern_local = Eigen::MatrixXd::Zero(u_size_single_kernel, f_size_i);
      for (int l = 0; l < f_size_i; l++) {
        kern_local.block(0, l, u_size_single_kernel, 1) = kern_t.block(0, local_label_indices[t][l], u_size_single_kernel, 1);
      }
      kuf_i.block(0, cum_f, u_size_single_kernel, f_size_i) = kern_local; 
      cum_f += f_size_i;
    }
    kuf.push_back(kuf_i);
    t2_inner = std::chrono::high_resolution_clock::now();
    duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2_inner - t1_inner ).count();
    std::cout << "Rank: " << blacs::mpirank << ", build local kuf: " << duration << " ms" << std::endl;


    t1_inner = std::chrono::high_resolution_clock::now();

    kuu.push_back(kernels[i]->envs_envs(
              sparse_descriptors[i], 
              sparse_descriptors[i],
              kernels[i]->kernel_hyperparameters));

    t2_inner = std::chrono::high_resolution_clock::now();
    duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2_inner - t1_inner ).count();
    std::cout << "Rank: " << blacs::mpirank << ", build local kuu: " << duration << " ms" << std::endl;

  }

  // Only keep the chunk of noise_vector assigned to the current proc
  cum_f = 0;
  int cum_f_struc = 0;
  local_noise_vector = Eigen::VectorXd::Zero(local_f_size);
  local_labels = Eigen::MatrixXd::Zero(local_f_size, 1); // TODO: change back to VectorXd
  for (int t = 0; t < training_structures.size(); t++) {
    int f_size_i = local_label_indices[t].size();
    for (int l = 0; l < f_size_i; l++) {
      local_noise_vector(cum_f + l) = noise_vector(cum_f_struc + local_label_indices[t][l]);
      local_labels(cum_f + l, 0) = y(cum_f_struc + local_label_indices[t][l]);
    }
    cum_f += f_size_i;
    cum_f_struc += training_strucs[t].n_labels();
  }

  // Store square root of noise vector.
  Eigen::VectorXd noise_vector_sqrt = sqrt(local_noise_vector.array());
  Eigen::VectorXd global_noise_vector_sqrt = sqrt(global_noise_vector.array());

  std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
  duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Rank: " << blacs::mpirank << ", build local kuf kuu: " << duration << " ms" << std::endl;

  // Synchronize, wait until all training structures are ready on all processors
  blacs::barrier();

  // Create distributed matrices
  // specify the scope of the DistMatrix
  duration = 0;
  t1 = std::chrono::high_resolution_clock::now();

  std::cout << "f_size=" << f_size << " , u_size=" << u_size << std::endl;
  DistMatrix<double> A(f_size + u_size, u_size); // use the default blocking
  DistMatrix<double> b(f_size + u_size, 1);
  DistMatrix<double> Kuu_dist(u_size, u_size);
  std::cout << "Created A, b, Kuu_dist" << std::endl;
  A = [](int i, int j){return 0.0;};
  b = [](int i, int j){return 0.0;};
  Kuu_dist = [](int i, int j){return 0.0;};
  blacs::barrier();
  std::cout << "Initialize with 0" << std::endl;

  bool lock = true;
  cum_u = 0;
  // Assign sparse set kernel matrix Kuu
  for (int i = 0; i < n_kernels; i++) { 
    for (int r = 0; r < kuu[i].rows(); r++) {
      for (int c = 0; c < kuu[i].cols(); c++) {
        if (Kuu_dist.islocal(r + cum_u, c + cum_u)) { // only set the local part
          Kuu_dist.set(r + cum_u, c + cum_u, kuu[i](r, c), lock);
        }
      }
    }
    cum_u += kuu[i].rows();
  }
  Kuu_dist.fence();

  t2 = std::chrono::high_resolution_clock::now();
  duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Rank: " << blacs::mpirank << ", build Kuu_dist: " << duration << " ms" << std::endl;

  duration = 0;
  t1 = std::chrono::high_resolution_clock::now();

  // TODO: Kuu and L_inv are only needed for debug and unit test 
  Matrix<double> Kuu_array(u_size, u_size);
  std::cout << "Begin allgather Kuu" << std::endl;
  Kuu_dist.allgather(Kuu_array.array.get());
  Kuu = Eigen::Map<Eigen::MatrixXd>(Kuu_array.array.get(), u_size, u_size);
  std::cout << "Allgathered Kuu" << std::endl;

  t2 = std::chrono::high_resolution_clock::now();
  duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Rank: " << blacs::mpirank << ", set Kuu_dist: " << duration << " ms" << std::endl;

  duration = 0;
  t1 = std::chrono::high_resolution_clock::now();

  // Cholesky decomposition of Kuu and its inverse.
  Eigen::LLT<Eigen::MatrixXd> chol(
      Kuu + Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols()));

  // Get the inverse of Kuu from Cholesky decomposition.
  Eigen::MatrixXd Kuu_eye = Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());
  Eigen::MatrixXd L = chol.matrixL();
  L_inv = chol.matrixL().solve(Kuu_eye);
  Kuu_inverse = L_inv.transpose() * L_inv;

  t2 = std::chrono::high_resolution_clock::now();
  duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Rank: " << blacs::mpirank << ", cholestky, tri_inv, matmul: " << duration << " ms" << std::endl;


  duration = 0;
  t1 = std::chrono::high_resolution_clock::now();

  cum_f = 0;
  int local_f_full = 0;
  int local_f = 0;
  // Assign Lambda * Kfu to A
  Eigen::MatrixXd noise_kfu = noise_vector_sqrt.asDiagonal() * kuf[0].transpose();
  A.collect(&noise_kfu(0, 0), 0, 0, f_size, u_size, f_size_per_proc, u_size, nmax_struc - nmin_struc); 
  Eigen::MatrixXd noise_labels = noise_vector_sqrt.asDiagonal() * local_labels;
  b.collect(&noise_labels(0, 0), 0, 0, f_size, 1, f_size_per_proc, 1, nmax_struc - nmin_struc); 
  b_debug = Eigen::VectorXd::Zero(f_size+u_size);
  b.gather(&b_debug(0));
  std::cout << "Collected Kuf to A" << std::endl;

  // Wait until the communication is done
  A.fence();
  b.fence();

//  // TODO: Kuf is for debugging
//  Kuf = Eigen::MatrixXd::Zero(u_size, f_size);
//  for (int r = 0; r < u_size; r++) {
//    for (int c = 0; c < f_size; c++) {
//      Kuf(r, c) = A(c, r);
//    }
//  }


  t2 = std::chrono::high_resolution_clock::now();
  duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Rank: " << blacs::mpirank << ", set A & b: " << duration << " ms" << std::endl;

  duration = 0;
  t1 = std::chrono::high_resolution_clock::now();

  // Copy L.T to A using scatter function
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

  t2 = std::chrono::high_resolution_clock::now();
  duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Rank: " << blacs::mpirank << ", set L.T to A: " << duration << " ms" << std::endl;

  duration = 0;
  t1 = std::chrono::high_resolution_clock::now();

  // QR factorize A to compute alpha
  DistMatrix<double> QR(u_size + f_size, u_size);
  std::vector<double> tau;
  std::tie(QR, tau) = A.qr();
  QR.fence();

  t2 = std::chrono::high_resolution_clock::now();
  duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Rank: " << blacs::mpirank << ", QR: " << duration << " ms" << std::endl;

  duration = 0;
  t1 = std::chrono::high_resolution_clock::now();

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
  alpha = R.triangularView<Eigen::Upper>().solve(Q_b);


  t2 = std::chrono::high_resolution_clock::now();
  duration += (double) std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
  std::cout << "Rank: " << blacs::mpirank << ", get alpha: " << duration << " ms" << std::endl;

}

void ParallelSGP ::predict_local_uncertainties(Structure &test_structure) {
  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;

  int n_sparse = Kuu_inverse.rows();
  Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(n_sparse, n_out);
  int count = 0;
  std::cout << "In predict: sparse_desc size " << sparse_descriptors.size() << std::endl;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = sparse_descriptors[i].n_clusters; 
    kernel_mat.block(count, 0, size, n_out) = kernels[i]->envs_struc(
        sparse_descriptors[i], test_structure.descriptors[i],
        kernels[i]->kernel_hyperparameters);
    count += size;
  }
  std::cout << "kernel_mat size " << kernel_mat.rows() << " " << kernel_mat.cols() << std::endl;
  std::cout << "alpha size " << alpha.size() << std::endl;

  test_structure.mean_efs = kernel_mat.transpose() * alpha;
  std::cout << "Begin compute_cluster" << std::endl;
  std::vector<Eigen::VectorXd> local_uncertainties =
    this->compute_cluster_uncertainties(test_structure);
  test_structure.local_uncertainties = local_uncertainties;

}

