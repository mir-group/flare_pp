#include "sparse_gp.h"
#include <algorithm> // Random shuffle
#include <chrono>
#include <fstream> // File operations
#include <iomanip> // setprecision
#include <iostream>
#include <numeric> // Iota

SparseGP ::SparseGP() {}

SparseGP ::SparseGP(std::vector<Kernel *> kernels, double energy_noise,
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

void SparseGP ::initialize_sparse_descriptors(const Structure &structure) {
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

std::vector<std::vector<int>>
SparseGP ::sort_clusters_by_uncertainty(const Structure &structure) {

  // Compute cluster uncertainties.
  std::vector<Eigen::VectorXd> variances =
      compute_cluster_uncertainties(structure);

  std::vector<std::vector<int>> sorted_indices;
  for (int i = 0; i < n_kernels; i++) {
    // Sort cluster indices by decreasing uncertainty.
    std::vector<int> indices(variances[i].size());
    iota(indices.begin(), indices.end(), 0);
    Eigen::VectorXd v = variances[i];
    stable_sort(indices.begin(), indices.end(),
                [&v](int i1, int i2) { return v(i1) > v(i2); });
    sorted_indices.push_back(indices);
  }

  return sorted_indices;
}

std::vector<Eigen::VectorXd>
SparseGP ::compute_cluster_uncertainties(const Structure &structure) {
  // TODO: this only computes the energy-energy variance, and the Sigma matrix is not considered?

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < structure.descriptors.size(); i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i]);
    cluster_descriptors.push_back(cluster_descriptor);
  }

  // Compute cluster uncertainties.
  std::vector<Eigen::VectorXd> K_self, Q_self, variances;
  std::vector<Eigen::MatrixXd> sparse_kernels;
  int sparse_count = 0;
  for (int i = 0; i < n_kernels; i++) {
    K_self.push_back(
        (kernels[i]->envs_envs(cluster_descriptors[i], cluster_descriptors[i],
                               kernels[i]->kernel_hyperparameters))
            .diagonal());

    sparse_kernels.push_back(
        kernels[i]->envs_envs(cluster_descriptors[i], sparse_descriptors[i],
                              kernels[i]->kernel_hyperparameters));

    int n_clusters = sparse_descriptors[i].n_clusters;
    Eigen::MatrixXd L_inverse_block =
        L_inv.block(sparse_count, sparse_count, n_clusters, n_clusters);
    sparse_count += n_clusters;

    Eigen::MatrixXd Q1 = L_inverse_block * sparse_kernels[i].transpose();
    Q_self.push_back((Q1.transpose() * Q1).diagonal());

    variances.push_back(K_self[i] - Q_self[i]); // it is sorted by clusters, not the original atomic order 
    // TODO: If the environment is empty, the assigned uncertainty should be
    // set to zero.
  }

  return variances;
}

void SparseGP ::add_specific_environments(const Structure &structure,
                                          const std::vector<int> atoms) {

  initialize_sparse_descriptors(structure);

  // Gather clusters with central atom in the given list.
  std::vector<std::vector<std::vector<int>>> indices_1;
  for (int i = 0; i < n_kernels; i++){
    sparse_indices[i].push_back(atoms); // for each kernel the added atoms are the same

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

  // Update Kuu and Kuf.
  update_Kuu(cluster_descriptors);
  update_Kuf(cluster_descriptors);
  stack_Kuu();
  stack_Kuf();

  // Store sparse environments.
  for (int i = 0; i < n_kernels; i++) {
    sparse_descriptors[i].add_clusters_by_type(structure.descriptors[i],
                                               indices_1[i]);
  }
}

void SparseGP ::add_uncertain_environments(const Structure &structure,
                                           const std::vector<int> &n_added) {

  initialize_sparse_descriptors(structure);

  // Compute cluster uncertainties.
  std::vector<std::vector<int>> sorted_indices =
      sort_clusters_by_uncertainty(structure);

  std::vector<std::vector<int>> n_sorted_indices;
  for (int i = 0; i < n_kernels; i++) {
    // Take the first N indices.
    int n_curr = n_added[i];
    if (n_curr > sorted_indices[i].size())
      n_curr = sorted_indices[i].size();
    std::vector<int> n_indices(n_curr);
    for (int j = 0; j < n_curr; j++) {
      n_indices[j] = sorted_indices[i][j];
    }
    n_sorted_indices.push_back(n_indices);
  }

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < n_kernels; i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i], n_sorted_indices[i]);
    cluster_descriptors.push_back(cluster_descriptor);
  }

  // Update Kuu and Kuf.
  update_Kuu(cluster_descriptors);
  update_Kuf(cluster_descriptors);
  stack_Kuu();
  stack_Kuf();

  // Store sparse environments.
  for (int i = 0; i < n_kernels; i++) {
    sparse_descriptors[i].add_clusters(structure.descriptors[i],
                                       n_sorted_indices[i]);

    // find the atom index of added sparse env
    std::vector<int> added_indices;
    for (int k = 0; k < n_sorted_indices[i].size(); k++) {
      int cluster_val = n_sorted_indices[i][k];
      int atom_index, val;
      for (int j = 0; j < structure.descriptors[i].n_types; j++) {
        int ccount = structure.descriptors[i].cumulative_type_count[j];
        int ccount_p1 = structure.descriptors[i].cumulative_type_count[j + 1];
        if ((cluster_val >= ccount) && (cluster_val < ccount_p1)) {
          val = cluster_val - ccount;
          atom_index = structure.descriptors[i].atom_indices[j][val];
          added_indices.push_back(atom_index);
          break;
        }
      }
    }

    sparse_indices[i].push_back(added_indices);
  }
}

void SparseGP ::add_random_environments(const Structure &structure,
                                        const std::vector<int> &n_added) {

  initialize_sparse_descriptors(structure);

  // Randomly select environments without replacement.
  std::vector<std::vector<int>> envs1;
  for (int i = 0; i < structure.descriptors.size(); i++) { // NOTE: n_kernels might be diff from descriptors number
    std::vector<int> envs2;
    int n_clusters = structure.descriptors[i].n_clusters;
    std::vector<int> clusters(n_clusters);
    std::iota(clusters.begin(), clusters.end(), 0);
    std::random_shuffle(clusters.begin(), clusters.end());
    int n_curr = n_added[i];
    if (n_curr > n_clusters)
      n_curr = n_clusters;
    for (int k = 0; k < n_curr; k++) {
      envs2.push_back(clusters[k]);
    }
    envs1.push_back(envs2);
  }

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < structure.descriptors.size(); i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i], envs1[i]);
    cluster_descriptors.push_back(cluster_descriptor);
  }

  // Update Kuu and Kuf.
  update_Kuu(cluster_descriptors);
  update_Kuf(cluster_descriptors);
  stack_Kuu();
  stack_Kuf();

  // Store sparse environments.
  for (int i = 0; i < n_kernels; i++) {
    sparse_descriptors[i].add_clusters(structure.descriptors[i], envs1[i]);

    // find the atom index of added sparse env
    std::vector<int> added_indices;
    for (int k = 0; k < envs1[i].size(); k++) {
      int cluster_val = envs1[i][k];
      int atom_index, val;
      for (int j = 0; j < structure.descriptors[i].n_types; j++) {
        int ccount = structure.descriptors[i].cumulative_type_count[j];
        int ccount_p1 = structure.descriptors[i].cumulative_type_count[j + 1];
        if ((cluster_val >= ccount) && (cluster_val < ccount_p1)) {
          val = cluster_val - ccount;
          atom_index = structure.descriptors[i].atom_indices[j][val];
          added_indices.push_back(atom_index);
          break;
        }
      }
    }
    sparse_indices[i].push_back(added_indices);
  }
}

void SparseGP ::add_all_environments(const Structure &structure) {
  initialize_sparse_descriptors(structure);

  // Create cluster descriptors.
  std::vector<ClusterDescriptor> cluster_descriptors;
  for (int i = 0; i < structure.descriptors.size(); i++) {
    ClusterDescriptor cluster_descriptor =
        ClusterDescriptor(structure.descriptors[i]);
    cluster_descriptors.push_back(cluster_descriptor);
  }

  // Update Kuu and Kuf.
  update_Kuu(cluster_descriptors);
  update_Kuf(cluster_descriptors);
  stack_Kuu();
  stack_Kuf();

  // Store sparse environments.
  std::vector<int> added_indices;
  for (int j = 0; j < structure.noa; j++) {
    added_indices.push_back(j);
  }
  for (int i = 0; i < n_kernels; i++) {
    sparse_descriptors[i].add_all_clusters(structure.descriptors[i]);
    sparse_indices[i].push_back(added_indices);
  }
}

void SparseGP ::update_Kuu(
    const std::vector<ClusterDescriptor> &cluster_descriptors) {

  // Update Kuu matrices.
  for (int i = 0; i < n_kernels; i++) {
    Eigen::MatrixXd prev_block =
        kernels[i]->envs_envs(sparse_descriptors[i], cluster_descriptors[i],
                              kernels[i]->kernel_hyperparameters);
    Eigen::MatrixXd self_block =
        kernels[i]->envs_envs(cluster_descriptors[i], cluster_descriptors[i],
                              kernels[i]->kernel_hyperparameters);

    int n_sparse = sparse_descriptors[i].n_clusters;
    int n_envs = cluster_descriptors[i].n_clusters;
    int n_types = cluster_descriptors[i].n_types;

    Eigen::MatrixXd kern_mat =
        Eigen::MatrixXd::Zero(n_sparse + n_envs, n_sparse + n_envs);

    int n1 = 0; // Sparse descriptor counter 1
    int n2 = 0; // Cluster descriptor counter 1

    // TODO: Generalize to allow comparisons across types.
    for (int j = 0; j < n_types; j++) {
      int n3 = 0; // Sparse descriptor counter 2
      int n4 = 0; // Cluster descriptor counter 2
      int n5 = sparse_descriptors[i].n_clusters_by_type[j];
      int n6 = cluster_descriptors[i].n_clusters_by_type[j];

      for (int k = 0; k < n_types; k++){
        int n7 = sparse_descriptors[i].n_clusters_by_type[k];
        int n8 = cluster_descriptors[i].n_clusters_by_type[k];

        Eigen::MatrixXd prev_vals_1 = prev_block.block(n1, n4, n5, n8);
        Eigen::MatrixXd prev_vals_2 = prev_block.block(n3, n2, n7, n6);
        Eigen::MatrixXd self_vals = self_block.block(n2, n4, n6, n8);

        kern_mat.block(n1 + n2, n3 + n4, n5, n7) =
          Kuu_kernels[i].block(n1, n3, n5, n7);
        kern_mat.block(n1 + n2, n3 + n4 + n7, n5, n8) =
          prev_vals_1;
        kern_mat.block(n1 + n2 + n5, n3 + n4, n6, n7) =
          prev_vals_2.transpose();
        kern_mat.block(n1 + n2 + n5, n3 + n4 + n7, n6, n8) =
          self_vals;

        n3 += n7;
        n4 += n8;
      }
      n1 += n5;
      n2 += n6;
    }
    Kuu_kernels[i] = kern_mat;

    // Update sparse count.
    this->n_sparse += n_envs;
    }
  }

void SparseGP ::update_Kuf(
    const std::vector<ClusterDescriptor> &cluster_descriptors) {

  // Compute kernels between new sparse environments and training structures.
  for (int i = 0; i < n_kernels; i++) {
    int n_sparse = sparse_descriptors[i].n_clusters;
    int n_envs = cluster_descriptors[i].n_clusters;
    int n_types = cluster_descriptors[i].n_types;

    // Precompute indices.
    Eigen::ArrayXi inds = Eigen::ArrayXi::Zero(n_types + 1);
    int counter = 0;
    for (int j = 0; j < n_types; j++) {
      int t1 = sparse_descriptors[i].n_clusters_by_type[j];
      int t2 = cluster_descriptors[i].n_clusters_by_type[j];
      counter += t1 + t2;
      inds(j + 1) = counter;
    }

    Eigen::MatrixXd kern_mat =
        Eigen::MatrixXd::Zero(n_sparse + n_envs, n_labels);

#pragma omp parallel for
    for (int j = 0; j < n_strucs; j++) {
      int n_atoms = training_structures[j].noa;
      Eigen::MatrixXd envs_struc_kernels = kernels[i]->envs_struc(
          cluster_descriptors[i], training_structures[j].descriptors[i],
          kernels[i]->kernel_hyperparameters);

      int n1 = 0; // Sparse descriptor count
      int n2 = 0; // Cluster descriptor count
      for (int k = 0; k < n_types; k++) {
        int current_count = 0;
        int u_ind = inds(k);
        int n3 = sparse_descriptors[i].n_clusters_by_type[k];
        int n4 = cluster_descriptors[i].n_clusters_by_type[k];

        if (training_structures[j].energy.size() != 0) {
          kern_mat.block(u_ind, label_count(j), n3, 1) =
              Kuf_kernels[i].block(n1, label_count(j), n3, 1);
          kern_mat.block(u_ind + n3, label_count(j), n4, 1) =
              envs_struc_kernels.block(n2, 0, n4, 1);

          current_count += 1;
        }

        if (training_structures[j].forces.size() != 0) {
          std::vector<int> atom_indices = training_atom_indices[j];
          for (int a = 0; a < atom_indices.size(); a++) {
            kern_mat.block(u_ind, label_count(j) + current_count, n3, 3) =
                Kuf_kernels[i].block(n1, label_count(j) + current_count, n3, 3);
            kern_mat.block(u_ind + n3, label_count(j) + current_count, n4, 3) =
                envs_struc_kernels.block(n2, 1 + atom_indices[a] * 3, n4, 3);
            current_count += 3;
          }
        }

        if (training_structures[j].stresses.size() != 0) {
          kern_mat.block(u_ind, label_count(j) + current_count, n3, 6) =
              Kuf_kernels[i].block(n1, label_count(j) + current_count, n3, 6);
          kern_mat.block(u_ind + n3, label_count(j) + current_count, n4, 6) =
              envs_struc_kernels.block(n2, 1 + n_atoms * 3, n4, 6);
        }

        n1 += n3;
        n2 += n4;
      }
    }
    Kuf_kernels[i] = kern_mat;
  }
}

void SparseGP ::add_training_structure(const Structure &structure,
                                       const std::vector<int> atom_indices) {
  initialize_sparse_descriptors(structure);

  int n_atoms = structure.noa;
  int n_energy = structure.energy.size();
  int n_force = 0;
  std::vector<int> atoms;
  if (atom_indices[0] == -1) { // add all atoms
    n_force = structure.forces.size();
    for (int i = 0; i < n_atoms; i++) {
      atoms.push_back(i);
    }
  } else {
    atoms = atom_indices;
    n_force = atoms.size() * 3;
  }
  training_atom_indices.push_back(atoms);
  int n_stress = structure.stresses.size();
  int n_struc_labels = n_energy + n_force + n_stress;

  // Update Kuf kernels.
  Eigen::MatrixXd envs_struc_kernels;
  for (int i = 0; i < n_kernels; i++) {
    int n_sparse = sparse_descriptors[i].n_clusters;

    envs_struc_kernels = // contain all atoms
        kernels[i]->envs_struc(sparse_descriptors[i], structure.descriptors[i],
                               kernels[i]->kernel_hyperparameters);

    Kuf_kernels[i].conservativeResize(n_sparse, n_labels + n_struc_labels);
    Kuf_kernels[i].block(0, n_labels, n_sparse, n_energy) =
        envs_struc_kernels.block(0, 0, n_sparse, n_energy);
    Kuf_kernels[i].block(0, n_labels + n_energy + n_force, n_sparse, n_stress) =
        envs_struc_kernels.block(0, 1 + n_atoms * 3, n_sparse, n_stress);

    // Only add forces from `atoms`
    for (int a = 0; a < atoms.size(); a++) {
      Kuf_kernels[i].block(0, n_labels + n_energy + a * 3, n_sparse, 3) =
          envs_struc_kernels.block(0, 1 + atoms[a] * 3, n_sparse, 3); // if n_energy=0, we can not use n_energy but 1
    }
  }

  // Update labels.
  label_count.conservativeResize(training_structures.size() + 2);
  label_count(training_structures.size() + 1) = n_labels + n_struc_labels;
  y.conservativeResize(n_labels + n_struc_labels);
  y.segment(n_labels, n_energy) = structure.energy;
  y.segment(n_labels + n_energy + n_force, n_stress) = structure.stresses;
  for (int a = 0; a < atoms.size(); a++) {
    y.segment(n_labels + n_energy + a * 3, 3) = structure.forces.segment(atoms[a] * 3, 3);
  }
  e_noise_one.conservativeResize(n_labels + n_struc_labels);
  f_noise_one.conservativeResize(n_labels + n_struc_labels);
  s_noise_one.conservativeResize(n_labels + n_struc_labels);

  e_noise_one.segment(n_labels, n_struc_labels) = Eigen::VectorXd::Zero(n_struc_labels);
  f_noise_one.segment(n_labels, n_struc_labels) = Eigen::VectorXd::Zero(n_struc_labels);
  s_noise_one.segment(n_labels, n_struc_labels) = Eigen::VectorXd::Zero(n_struc_labels);

  e_noise_one.segment(n_labels, n_energy) = Eigen::VectorXd::Ones(n_energy);
  f_noise_one.segment(n_labels + n_energy, n_force) = Eigen::VectorXd::Ones(n_force);
  s_noise_one.segment(n_labels + n_energy + n_force, n_stress) = Eigen::VectorXd::Ones(n_stress);

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

  // Update Kuf.
  stack_Kuf();
}

void SparseGP ::stack_Kuu() {
  // Update Kuu.
  Kuu = Eigen::MatrixXd::Zero(n_sparse, n_sparse);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = Kuu_kernels[i].rows();
    Kuu.block(count, count, size, size) = Kuu_kernels[i];
    count += size;
  }
}

void SparseGP ::stack_Kuf() {
  // Update Kuf kernels.
  Kuf = Eigen::MatrixXd::Zero(n_sparse, n_labels);
  int count = 0;
  for (int i = 0; i < Kuf_kernels.size(); i++) {
    int size = Kuf_kernels[i].rows();
    Kuf.block(count, 0, size, n_labels) = Kuf_kernels[i];
    count += size;
  }
}

void SparseGP ::update_matrices_QR() {
  // Store square root of noise vector.
  Eigen::VectorXd noise_vector_sqrt = sqrt(noise_vector.array());

  // Cholesky decompose Kuu.
  Eigen::LLT<Eigen::MatrixXd> chol(
      Kuu + Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols()));

  // Get the inverse of Kuu from Cholesky decomposition.
  Eigen::MatrixXd Kuu_eye = Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols());
  L_inv = chol.matrixL().solve(Kuu_eye);
  L_diag = L_inv.diagonal();
  Kuu_inverse = L_inv.transpose() * L_inv;

  // Form A matrix.
  Eigen::MatrixXd A =
      Eigen::MatrixXd::Zero(Kuf.cols() + Kuu.cols(), Kuu.cols());
  A.block(0, 0, Kuf.cols(), Kuu.cols()) =
      noise_vector_sqrt.asDiagonal() * Kuf.transpose();
  A.block(Kuf.cols(), 0, Kuu.cols(), Kuu.cols()) = chol.matrixL().transpose();

  // Form b vector.
  Eigen::VectorXd b = Eigen::VectorXd::Zero(Kuf.cols() + Kuu.cols());
  b.segment(0, Kuf.cols()) = noise_vector_sqrt.asDiagonal() * y;

  // QR decompose A.
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
  Eigen::VectorXd Q_b = qr.householderQ().transpose() * b;
  R_inv = qr.matrixQR().block(0, 0, Kuu.cols(), Kuu.cols())
                       .triangularView<Eigen::Upper>()
                       .solve(Kuu_eye);
  R_inv_diag = R_inv.diagonal();
  alpha = R_inv * Q_b;
  Sigma = R_inv * R_inv.transpose();
}

void SparseGP ::predict_mean(Structure &test_structure) {

  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;

  Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(n_sparse, n_out);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = Kuu_kernels[i].rows();
    kernel_mat.block(count, 0, size, n_out) = kernels[i]->envs_struc(
        sparse_descriptors[i], test_structure.descriptors[i],
        kernels[i]->kernel_hyperparameters);
    count += size;
  }

  test_structure.mean_efs = kernel_mat.transpose() * alpha;
}

void SparseGP ::predict_SOR(Structure &test_structure) {

  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;

  Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(n_sparse, n_out);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = Kuu_kernels[i].rows();
    kernel_mat.block(count, 0, size, n_out) = kernels[i]->envs_struc(
        sparse_descriptors[i], test_structure.descriptors[i],
        kernels[i]->kernel_hyperparameters);
    count += size;
  }

  test_structure.mean_efs = kernel_mat.transpose() * alpha;
  Eigen::MatrixXd variance_sqrt = kernel_mat.transpose() * R_inv;
  test_structure.variance_efs =
      (variance_sqrt * variance_sqrt.transpose()).diagonal();
}

void SparseGP ::predict_DTC(Structure &test_structure) {

  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;

  Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(n_sparse, n_out);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = Kuu_kernels[i].rows();
    kernel_mat.block(count, 0, size, n_out) = kernels[i]->envs_struc(
        sparse_descriptors[i], test_structure.descriptors[i],
        kernels[i]->kernel_hyperparameters);
    count += size;
  }

  test_structure.mean_efs = kernel_mat.transpose() * alpha;

  // Compute variances.
  Eigen::VectorXd V_SOR, Q_self, K_self = Eigen::VectorXd::Zero(n_out);

  for (int i = 0; i < n_kernels; i++) {
    K_self += kernels[i]->self_kernel_struc(test_structure.descriptors[i],
                                            kernels[i]->kernel_hyperparameters);
  }

  Q_self = (kernel_mat.transpose() * Kuu_inverse * kernel_mat).diagonal();
  V_SOR = (kernel_mat.transpose() * Sigma * kernel_mat).diagonal();

  test_structure.variance_efs = K_self - Q_self + V_SOR;
}

void SparseGP ::predict_local_uncertainties(Structure &test_structure) {
  int n_atoms = test_structure.noa;
  int n_out = 1 + 3 * n_atoms + 6;

  Eigen::MatrixXd kernel_mat = Eigen::MatrixXd::Zero(n_sparse, n_out);
  int count = 0;
  for (int i = 0; i < Kuu_kernels.size(); i++) {
    int size = Kuu_kernels[i].rows();
    kernel_mat.block(count, 0, size, n_out) = kernels[i]->envs_struc(
        sparse_descriptors[i], test_structure.descriptors[i],
        kernels[i]->kernel_hyperparameters);
    count += size;
  }

  test_structure.mean_efs = kernel_mat.transpose() * alpha;

  std::vector<Eigen::VectorXd> local_uncertainties =
    compute_cluster_uncertainties(test_structure);
  test_structure.local_uncertainties = local_uncertainties;

}

void SparseGP ::compute_likelihood_stable() {
  // Compute inverse of Qff from Sigma.
  Eigen::MatrixXd noise_diag = noise_vector.asDiagonal();

  data_fit =
      -(1. / 2.) * y.transpose() * noise_diag * (y - Kuf.transpose() * alpha);
  constant_term = -(1. / 2.) * n_labels * log(2 * M_PI);

  // Compute complexity penalty.
  double noise_det = 0;
  for (int i = 0; i < noise_vector.size(); i++) {
    noise_det += log(noise_vector(i));
  }

  double Kuu_inv_det = 0;
  for (int i = 0; i < L_diag.size(); i++) {
    Kuu_inv_det -= 2 * log(abs(L_diag(i)));
  }

  double sigma_inv_det = 0;
  for (int i = 0; i < R_inv_diag.size(); i++) {
    sigma_inv_det += 2 * log(abs(R_inv_diag(i)));
  }

  complexity_penalty = (1. / 2.) * (noise_det + Kuu_inv_det + sigma_inv_det);
  log_marginal_likelihood = complexity_penalty + data_fit + constant_term;
}

double SparseGP ::compute_likelihood_gradient_stable() {

  // Compute inverse of Qff from Sigma.
  Eigen::MatrixXd noise_diag = noise_vector.asDiagonal();
  Eigen::MatrixXd e_noise_one_diag = e_noise_one.asDiagonal();
  Eigen::MatrixXd f_noise_one_diag = f_noise_one.asDiagonal();
  Eigen::MatrixXd s_noise_one_diag = s_noise_one.asDiagonal();

  Eigen::VectorXd y_K_alpha = y - Kuf.transpose() * alpha;
  data_fit =
      -(1. / 2.) * y.transpose() * noise_diag * y_K_alpha;
  constant_term = -(1. / 2.) * n_labels * log(2 * M_PI);

  // Compute complexity penalty.
  double noise_det = 0;
  for (int i = 0; i < noise_vector.size(); i++) {
    noise_det += log(noise_vector(i));
  }

  double Kuu_inv_det = 0;
  for (int i = 0; i < L_diag.size(); i++) {
    Kuu_inv_det -= 2 * log(abs(L_diag(i)));
  }

  double sigma_inv_det = 0;
  for (int i = 0; i < R_inv_diag.size(); i++) {
    sigma_inv_det += 2 * log(abs(R_inv_diag(i)));
  }

  complexity_penalty = (1. / 2.) * (noise_det + Kuu_inv_det + sigma_inv_det);
  log_marginal_likelihood = complexity_penalty + data_fit + constant_term;
  std::cout << "computed likelihood" << std::endl;

  // Compute Kuu and Kuf matrices and gradients.
  int n_hyps_total = hyperparameters.size();

  //Eigen::MatrixXd Kuu_mat = Eigen::MatrixXd::Zero(n_sparse, n_sparse);
  //Eigen::MatrixXd Kuf_mat = Eigen::MatrixXd::Zero(n_sparse, n_labels);

  std::vector<Eigen::MatrixXd> Pi_grads;

  std::vector<Eigen::MatrixXd> Kuu_grad, Kuf_grad, Kuu_grads, Kuf_grads;

  int n_hyps, hyp_index = 0, grad_index = 0;
  Eigen::VectorXd hyps_curr;

  int count = 0;
  Eigen::VectorXd complexity_grad = Eigen::VectorXd::Zero(n_hyps_total);
  Eigen::VectorXd datafit_grad = Eigen::VectorXd::Zero(n_hyps_total);
  likelihood_gradient = Eigen::VectorXd::Zero(n_hyps_total);
  std::cout << "enter for loop" << std::endl;
  for (int i = 0; i < n_kernels; i++) {
    n_hyps = kernels[i]->kernel_hyperparameters.size();
    hyps_curr = hyperparameters.segment(hyp_index, n_hyps);
    int size = Kuu_kernels[i].rows();

    Kuu_grad = kernels[i]->Kuu_grad(sparse_descriptors[i], Kuu, hyps_curr);
    Kuf_grad = kernels[i]->Kuf_grad(sparse_descriptors[i], training_structures,
                                    i, Kuf, hyps_curr);

    //Kuu_mat.block(count, count, size, size) = Kuu_grad[0];
    //Kuf_mat.block(count, 0, size, n_labels) = Kuf_grad[0];
    Eigen::MatrixXd Kuu_i = Kuu_grad[0];

    std::cout << "enter for loop" << std::endl;
    for (int j = 0; j < n_hyps; j++) {
      Kuu_grads.push_back(Eigen::MatrixXd::Zero(n_sparse, n_sparse));
      Kuf_grads.push_back(Eigen::MatrixXd::Zero(n_sparse, n_labels));

      Kuu_grads[hyp_index + j].block(count, count, size, size) =
          Kuu_grad[j + 1];
      Kuf_grads[hyp_index + j].block(count, 0, size, n_labels) =
          Kuf_grad[j + 1];

      std::cout << "computed Kuu_grad Kuf_grad" << std::endl;

      // Compute Pi matrix and save as an intermediate variable
      Eigen::MatrixXd dK_noise_K = Kuf_grads[hyp_index + j] * noise_diag * Kuf.transpose();
      Eigen::MatrixXd Pi_mat = dK_noise_K + dK_noise_K.transpose() + Kuu_grads[hyp_index + j]; 
      Pi_grads.push_back(Pi_mat);

      // Derivative of complexity over sigma
      complexity_grad(hyp_index + j) += 1./2. * (Kuu_i.inverse() * Kuu_grad[j + 1]).trace() - 1./2. * (Pi_mat * Sigma).trace(); 

      // Derivative of data_fit over sigma
      datafit_grad(hyp_index + j) += y.transpose() * noise_diag * Kuf_grads[hyp_index + j].transpose() * alpha;
      datafit_grad(hyp_index + j) += - 1./2. * alpha.transpose() * Pi_mat * alpha;
      likelihood_gradient(hyp_index + j) += complexity_grad(hyp_index + j) + datafit_grad(hyp_index + j); 
    }

    count += size;
    hyp_index += n_hyps;
  }

  // Derivative of complexity over noise  

  // Derivative of data_fit over noise  
  datafit_grad(hyp_index + 0) = y_K_alpha.transpose() * e_noise_one_diag * y_K_alpha;
  datafit_grad(hyp_index + 1) = y_K_alpha.transpose() * f_noise_one_diag * y_K_alpha;
  datafit_grad(hyp_index + 2) = y_K_alpha.transpose() * s_noise_one_diag * y_K_alpha;
  datafit_grad(hyp_index + 0) /= energy_noise * energy_noise * energy_noise;
  datafit_grad(hyp_index + 1) /= force_noise * force_noise * force_noise;
  datafit_grad(hyp_index + 2) /= stress_noise * stress_noise * stress_noise;

  likelihood_gradient(hyp_index + 0) += datafit_grad(hyp_index + 0);
  likelihood_gradient(hyp_index + 1) += datafit_grad(hyp_index + 1);
  likelihood_gradient(hyp_index + 2) += datafit_grad(hyp_index + 2);

  return log_marginal_likelihood;

}

void SparseGP ::compute_likelihood() {
  if (n_labels == 0) {
    std::cout << "Warning: The likelihood is being computed without any "
                 "labels in the training set. The result won't be meaningful."
              << std::endl;
    return;
  }

  // Construct noise vector.
  Eigen::VectorXd noise = 1 / noise_vector.array();

  Eigen::MatrixXd Qff_plus_lambda =
      Kuf.transpose() * Kuu_inverse * Kuf +
      noise.asDiagonal() * Eigen::MatrixXd::Identity(n_labels, n_labels);

  // Decompose the matrix. Use QR decomposition instead of LLT/LDLT becaues Qff
  // becomes nonpositive when the training set is large.
  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(Qff_plus_lambda);
  Eigen::VectorXd Q_inv_y = qr.solve(y);
  Eigen::MatrixXd qr_mat = qr.matrixQR();
  // Compute the complexity penalty.
  complexity_penalty = 0;
  for (int i = 0; i < qr_mat.rows(); i++) {
    complexity_penalty += -log(abs(qr_mat(i, i)));
  }
  complexity_penalty /= 2;

  double half = 1.0 / 2.0;
  data_fit = -half * y.transpose() * Q_inv_y;
  constant_term = -half * n_labels * log(2 * M_PI);
  log_marginal_likelihood = complexity_penalty + data_fit + constant_term;
}

double
SparseGP ::compute_likelihood_gradient(const Eigen::VectorXd &hyperparameters) {

  // Compute Kuu and Kuf matrices and gradients.
  int n_hyps_total = hyperparameters.size();

  Eigen::MatrixXd Kuu_mat = Eigen::MatrixXd::Zero(n_sparse, n_sparse);
  Eigen::MatrixXd Kuf_mat = Eigen::MatrixXd::Zero(n_sparse, n_labels);

  std::vector<Eigen::MatrixXd> Kuu_grad, Kuf_grad, Kuu_grads, Kuf_grads;

  int n_hyps, hyp_index = 0, grad_index = 0;
  Eigen::VectorXd hyps_curr;

  int count = 0;
  for (int i = 0; i < n_kernels; i++) {
    n_hyps = kernels[i]->kernel_hyperparameters.size();
    hyps_curr = hyperparameters.segment(hyp_index, n_hyps);
    int size = Kuu_kernels[i].rows();

    Kuu_grad = kernels[i]->Kuu_grad(sparse_descriptors[i], Kuu, hyps_curr);
    Kuf_grad = kernels[i]->Kuf_grad(sparse_descriptors[i], training_structures,
                                    i, Kuf, hyps_curr);

    Kuu_mat.block(count, count, size, size) = Kuu_grad[0];
    Kuf_mat.block(count, 0, size, n_labels) = Kuf_grad[0];

    for (int j = 0; j < n_hyps; j++) {
      Kuu_grads.push_back(Eigen::MatrixXd::Zero(n_sparse, n_sparse));
      Kuf_grads.push_back(Eigen::MatrixXd::Zero(n_sparse, n_labels));

      Kuu_grads[hyp_index + j].block(count, count, size, size) =
          Kuu_grad[j + 1];
      Kuf_grads[hyp_index + j].block(count, 0, size, n_labels) =
          Kuf_grad[j + 1];
    }

    count += size;
    hyp_index += n_hyps;
  }

  Eigen::MatrixXd Kuu_inverse =
      (Kuu_mat + Kuu_jitter * Eigen::MatrixXd::Identity(Kuu.rows(), Kuu.cols()))
          .inverse();

  // Construct updated noise vector and gradients.
  Eigen::VectorXd noise_vec = Eigen::VectorXd::Zero(n_labels);
  Eigen::VectorXd e_noise_grad = Eigen::VectorXd::Zero(n_labels);
  Eigen::VectorXd f_noise_grad = Eigen::VectorXd::Zero(n_labels);
  Eigen::VectorXd s_noise_grad = Eigen::VectorXd::Zero(n_labels);

  double sigma_e = hyperparameters(hyp_index);
  double sigma_f = hyperparameters(hyp_index + 1);
  double sigma_s = hyperparameters(hyp_index + 2);

  int current_count = 0;
  for (int i = 0; i < training_structures.size(); i++) {
    int n_atoms = training_structures[i].noa;

    if (training_structures[i].energy.size() != 0) {
      noise_vec(current_count) = sigma_e * sigma_e;
      e_noise_grad(current_count) = 2 * sigma_e;
      current_count += 1;
    }

    if (training_structures[i].forces.size() != 0) {
      for (int a = 0; a < training_atom_indices[i].size(); a++) {
        noise_vec.segment(current_count, 3) =
            Eigen::VectorXd::Constant(3, sigma_f * sigma_f);
        f_noise_grad.segment(current_count, 3) =
            Eigen::VectorXd::Constant(3, 2 * sigma_f);
        current_count += 3;
      }
    }

    if (training_structures[i].stresses.size() != 0) {
      noise_vec.segment(current_count, 6) =
          Eigen::VectorXd::Constant(6, sigma_s * sigma_s);
      s_noise_grad.segment(current_count, 6) =
          Eigen::VectorXd::Constant(6, 2 * sigma_s);
      current_count += 6;
    }
  }

  // Compute Qff and Qff grads.
  Eigen::MatrixXd Qff_plus_lambda =
      Kuf_mat.transpose() * Kuu_inverse * Kuf_mat +
      noise_vec.asDiagonal() * Eigen::MatrixXd::Identity(n_labels, n_labels);

  std::vector<Eigen::MatrixXd> Qff_grads;
  grad_index = 0;
  for (int i = 0; i < n_kernels; i++) {
    n_hyps = kernels[i]->kernel_hyperparameters.size();
    for (int j = 0; j < n_hyps; j++) {
      Qff_grads.push_back(
          Kuf_grads[grad_index].transpose() * Kuu_inverse * Kuf_mat -
          Kuf_mat.transpose() * Kuu_inverse * Kuu_grads[grad_index] *
              Kuu_inverse * Kuf_mat +
          Kuf_mat.transpose() * Kuu_inverse * Kuf_grads[grad_index]);

      grad_index++;
    }
  }

  Qff_grads.push_back(e_noise_grad.asDiagonal() *
                      Eigen::MatrixXd::Identity(n_labels, n_labels));
  Qff_grads.push_back(f_noise_grad.asDiagonal() *
                      Eigen::MatrixXd::Identity(n_labels, n_labels));
  Qff_grads.push_back(s_noise_grad.asDiagonal() *
                      Eigen::MatrixXd::Identity(n_labels, n_labels));

  // Perform LU decomposition inplace and compute the inverse.
  Eigen::PartialPivLU<Eigen::Ref<Eigen::MatrixXd>> lu(Qff_plus_lambda);
  Eigen::MatrixXd Qff_inverse = lu.inverse();

  // Compute log determinant from the diagonal of U.
  complexity_penalty = 0;
  for (int i = 0; i < Qff_plus_lambda.rows(); i++) {
    complexity_penalty += -log(abs(Qff_plus_lambda(i, i)));
  }
  complexity_penalty /= 2;

  // Compute log marginal likelihood.
  Eigen::VectorXd Q_inv_y = Qff_inverse * y;
  data_fit = -(1. / 2.) * y.transpose() * Q_inv_y;
  constant_term = -n_labels * log(2 * M_PI) / 2;
  log_marginal_likelihood = complexity_penalty + data_fit + constant_term;

  // Compute likelihood gradient.
  likelihood_gradient = Eigen::VectorXd::Zero(n_hyps_total);
  Eigen::MatrixXd Qff_inv_grad;
  for (int i = 0; i < n_hyps_total; i++) {
    Qff_inv_grad = Qff_inverse * Qff_grads[i];
    likelihood_gradient(i) =
        -Qff_inv_grad.trace() + y.transpose() * Qff_inv_grad * Q_inv_y;
    likelihood_gradient(i) /= 2;
  }

  return log_marginal_likelihood;
}

void SparseGP ::set_hyperparameters(Eigen::VectorXd hyps) {
  // Reset Kuu and Kuf matrices.
  int n_hyps, hyp_index = 0;
  Eigen::VectorXd new_hyps;

  std::vector<Eigen::MatrixXd> Kuu_grad, Kuf_grad;
  for (int i = 0; i < n_kernels; i++) {
    n_hyps = kernels[i]->kernel_hyperparameters.size();
    new_hyps = hyps.segment(hyp_index, n_hyps);

    Kuu_grad = kernels[i]->Kuu_grad(sparse_descriptors[i], Kuu, new_hyps);
    Kuf_grad = kernels[i]->Kuf_grad(sparse_descriptors[i], training_structures,
                                    i, Kuf, new_hyps);

    Kuu_kernels[i] = Kuu_grad[0];
    Kuf_kernels[i] = Kuf_grad[0];

    kernels[i]->set_hyperparameters(new_hyps);
    hyp_index += n_hyps;
  }

  stack_Kuu();
  stack_Kuf();

  hyperparameters = hyps;
  energy_noise = hyps(hyp_index);
  force_noise = hyps(hyp_index + 1);
  stress_noise = hyps(hyp_index + 2);

  int current_count = 0;
  for (int i = 0; i < training_structures.size(); i++) {
    int n_atoms = training_structures[i].noa;

    if (training_structures[i].energy.size() != 0) {
      noise_vector(current_count) = 1 / (energy_noise * energy_noise);
      current_count += 1;
    }

    if (training_structures[i].forces.size() != 0) {
      for (int a = 0; a < training_atom_indices[i].size(); a++) {
        noise_vector.segment(current_count, 3) =
            Eigen::VectorXd::Constant(3, 1 / (force_noise * force_noise));
        current_count += 3;
      }
    }

    if (training_structures[i].stresses.size() != 0) {
      noise_vector.segment(current_count, 6) =
          Eigen::VectorXd::Constant(6, 1 / (stress_noise * stress_noise));
      current_count += 6;
    }
  }

  // Update remaining matrices.
  update_matrices_QR();
}

void SparseGP::write_mapping_coefficients(std::string file_name,
    std::string contributor, std::vector<int> kernel_indices, 
    std::string map_type) {

  // Make beta file.
  std::ofstream coeff_file;
  coeff_file.open(file_name);

  // Record the date.
  time_t now = std::time(0);
  std::string t(ctime(&now));
  coeff_file << "DATE: ";
  coeff_file << t.substr(0, t.length() - 1) << " ";

  // Record the contributor.
  coeff_file << "CONTRIBUTOR: ";
  coeff_file << contributor << "\n";

  // Write the number of kernels/descriptors to map
  coeff_file << kernel_indices.size();

  for (int k = 0; k < kernel_indices.size(); k++) {
    int kernel_index = kernel_indices[k];

    // Compute mapping coefficients.
    Eigen::MatrixXd mapping_coeffs;
    if (map_type == std::string("potential")) {
      mapping_coeffs =
          kernels[kernel_index]->compute_mapping_coefficients(*this, kernel_index);
    } else if (map_type == std::string("uncertainty")) {
      mapping_coeffs =
          kernels[kernel_index]->compute_varmap_coefficients(*this, kernel_index);
    }
    
    // Write descriptor information to file.
    int coeff_size = mapping_coeffs.row(0).size();
    training_structures[0].descriptor_calculators[kernel_index]->write_to_file(
        coeff_file, coeff_size);
    
    // Write beta vectors to file.
    coeff_file << std::scientific << std::setprecision(16);
    
    int count = 0;
    for (int i = 0; i < mapping_coeffs.rows(); i++) {
      Eigen::VectorXd coeff_vals = mapping_coeffs.row(i);
    
      // Start a new line for each beta.
      if (count != 0) {
        count = 0;
        coeff_file << "\n";
      }
    
      for (int j = 0; j < coeff_vals.size(); j++) {
        double coeff_val = coeff_vals[j];
    
        // Pad with 2 spaces if positive, 1 if negative.
        if (coeff_val > 0) {
          coeff_file << "  ";
        } else {
          coeff_file << " ";
        }
    
        coeff_file << coeff_vals[j];
        count++;
    
        // New line if 5 numbers have been added.
        if (count == 5) {
          count = 0;
          coeff_file << "\n";
        }
      }
    }
  }
  coeff_file.close();
}

void SparseGP ::to_json(std::string file_name, const SparseGP & sgp){
  std::ofstream sgp_file(file_name);
  nlohmann::json j = sgp;
  sgp_file << j;
}

SparseGP SparseGP ::from_json(std::string file_name){
  std::ifstream sgp_file(file_name);
  nlohmann::json j;
  sgp_file >> j;
  return j;
}
