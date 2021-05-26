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
#include <matrix.h>

class ParallelSGP : public SparseGP {
public:
  // Training and sparse points.
  std::vector<ClusterDescriptor> global_sparse_descriptors;
  std::vector<std::vector<ClusterDescriptor>> local_sparse_descriptors;
  std::vector<std::vector<std::vector<int>>> global_sparse_indices;

  // Parallel parameters
  int u_size, u_size_single_kernel, u_size_per_proc; 
  int f_size, f_size_single_kernel, f_size_per_proc;
  int nmin_struc, nmax_struc, nmin_envs, nmax_envs;
  std::vector<Eigen::VectorXi> n_struc_clusters_by_type;
  int global_n_labels;

  // Constructors.
  ParallelSGP();
  ParallelSGP(std::vector<Kernel *> kernels, double energy_noise,
           double force_noise, double stress_noise);

//  std::vector<ClusterDescriptor>
//  initialize_sparse_descriptors(const Structure &structure, std::vector<ClusterDescriptor> sparse_desc);
  void initialize_local_sparse_descriptors(const Structure &structure);
  void initialize_global_sparse_descriptors(const Structure &structure);

  void add_global_noise(int n_energy, int n_force, int n_stress); 
  Eigen::VectorXd global_noise_vector;

  void add_training_structure(const Structure &structure);
  
  Eigen::VectorXi sparse_indices_by_type(int n_types, std::vector<int> species, const std::vector<int> atoms);    
  void add_specific_environments(const Structure&, std::vector<int>);
  void add_local_specific_environments(const Structure &structure, const std::vector<int> atoms);
  void add_global_specific_environments(const Structure &structure, const std::vector<int> atoms);
  void predict_local_uncertainties(Structure &structure);

  void build(const std::vector<Structure> &training_strucs,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices,
        int n_types);
 
  void load_local_training_data(const std::vector<Structure> &training_strucs,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices,
        int n_types);

  void gather_sparse_descriptors(std::vector<int> n_clusters_by_type,
        const std::vector<Structure> &training_strucs,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices);

  void compute_matrices(const std::vector<Structure> &training_strucs);

  Eigen::MatrixXd varmap_coeffs; // for debugging. TODO: remove this line 
};

#endif
