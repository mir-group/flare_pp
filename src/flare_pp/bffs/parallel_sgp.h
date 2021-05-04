#ifndef PARALLEL_SGP_H
#define PARALLEL_SGP_H

#include "descriptor.h"
#include "kernel.h"
#include "structure.h"
#include "sparse_gp.h"
#include <vector>
#include <nlohmann/json.hpp>
#include "json.h"

#include <blacs.h>
#include <distmatrix.h>

class ParallelSGP : public SparseGP {
public:
  Eigen::VectorXd hyperparameters;

  // Kernel attributes.
  std::vector<Kernel *> kernels;
  std::vector<Eigen::MatrixXd> Kuu_kernels, Kuf_kernels;
  Eigen::MatrixXd Kuu, Kuf;
  int n_kernels = 0;
  double Kuu_jitter;

  // Solution attributes.
  Eigen::MatrixXd Sigma, Kuu_inverse, R_inv, L_inv;
  Eigen::VectorXd alpha, R_inv_diag, L_diag;

  // Training and sparse points.
  std::vector<ClusterDescriptor> local_sparse_descriptors, global_sparse_descriptors, sparse_descriptors;
  std::vector<Structure> training_structures;
  std::vector<std::vector<std::vector<int>>> sparse_indices, global_sparse_indices;

  // Label attributes.
  Eigen::VectorXd noise_vector, y, label_count;
  int n_energy_labels = 0, n_force_labels = 0, n_stress_labels = 0,
      n_sparse = 0, n_labels = 0, n_strucs = 0, global_n_labels = 0;
  double energy_noise, force_noise, stress_noise;

  // Parallel parameters
  int u_size, f_size, f_size_single_kernel, f_size_per_proc;
  int nmin_struc, nmax_struc;
  Eigen::VectorXd b_vec; // TODO: for debugging, remove

  // Likelihood attributes.
  double log_marginal_likelihood, data_fit, complexity_penalty, trace_term,
      constant_term;
  Eigen::VectorXd likelihood_gradient;

  // Constructors.
  ParallelSGP();
  ParallelSGP(std::vector<Kernel *> kernels, double energy_noise,
           double force_noise, double stress_noise);

//  std::vector<ClusterDescriptor>
//  initialize_sparse_descriptors(const Structure &structure, std::vector<ClusterDescriptor> sparse_desc);
  void initialize_local_sparse_descriptors(const Structure &structure);
  void initialize_global_sparse_descriptors(const Structure &structure);
  void add_all_environments(const Structure &structure);

  void add_specific_environments(const Structure &structure,
                                 const std::vector<int> atoms);
  void add_random_environments(const Structure &structure,
                               const std::vector<int> &n_added);
  void add_uncertain_environments(const Structure &structure,
                                  const std::vector<int> &n_added);
  std::vector<std::vector<int>>
  sort_clusters_by_uncertainty(const Structure &structure);

  void add_global_noise(const Structure &structure); 
  Eigen::VectorXd global_noise_vector;

  void add_training_structure(const Structure &structure);
  std::vector<std::vector<std::vector<int>>>
  sparse_indices_by_type(const Structure &structure, const std::vector<int> atoms);    
  void add_local_specific_environments(const Structure &structure, const std::vector<int> atoms);
  void add_global_specific_environments(const Structure &structure, const std::vector<int> atoms);
  void update_Kuu(const std::vector<ClusterDescriptor> &cluster_descriptors);
  void update_Kuf(const std::vector<ClusterDescriptor> &cluster_descriptors);
  void stack_Kuu();
  void stack_Kuf();

  void update_matrices_QR();

  void predict_SOR(Structure &structure);
  void predict_DTC(Structure &structure);
  void predict_local_uncertainties(Structure &structure);

  void compute_likelihood_stable();
  void compute_likelihood();

  double compute_likelihood_gradient(const Eigen::VectorXd &hyperparameters);
  void set_hyperparameters(Eigen::VectorXd hyps);

  void build(const std::vector<Eigen::MatrixXd> &training_cells,
        const std::vector<std::vector<int>> &training_species,
        const std::vector<Eigen::MatrixXd> &training_positions,
        const std::vector<Eigen::VectorXd> &training_labels,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &sparse_indices);

  void load_local_training_data(const std::vector<Eigen::MatrixXd> &training_cells,
        const std::vector<std::vector<int>> &training_species,
        const std::vector<Eigen::MatrixXd> &training_positions,
        const std::vector<Eigen::VectorXd> &training_labels,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices);

  void compute_matrices(const std::vector<Eigen::VectorXd> &training_labels,
        std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices);

  Eigen::MatrixXd varmap_coeffs; // for debugging. TODO: remove this line 
};

#endif
