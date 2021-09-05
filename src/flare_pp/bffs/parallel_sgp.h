#ifndef PARALLEL_SGP_H
#define PARALLEL_SGP_H

#include "descriptor.h"
#include "kernel.h"
#include "structure.h"
#include "sparse_gp.h"
#include <vector>
#include <nlohmann/json.hpp>
#include "json.h"

class ParallelSGP : public SparseGP {
public:
  // Training and sparse points.
  std::vector<ClusterDescriptor> global_sparse_descriptors;
  std::vector<std::vector<ClusterDescriptor>> local_sparse_descriptors;
  std::vector<std::vector<std::vector<int>>> global_sparse_indices;
  std::vector<std::vector<int>> local_label_indices;
  int local_label_size;

  // Parallel parameters
  int u_size, u_size_single_kernel, u_size_per_proc; 
  int f_size, f_size_single_kernel, f_size_per_proc;
  int nmin_struc, nmax_struc, nmin_envs, nmax_envs;
  std::vector<Eigen::VectorXi> n_struc_clusters_by_type;
  int global_n_labels;

  // Constructors.
  ParallelSGP();

  /**
   Basic Parallel Sparse GP constructor. This class inherits from SparseGP class and accept
   the same input parameters.

   @param kernels A list of Kernel objects, e.g. NormalizedInnerProduct, SquaredExponential.
        Note the number of kernels should be equal to the number of descriptor calculators.
   @param energy_noise Noise hyperparameter for total energy.
   @param force_noise Noise hyperparameter for atomic forces.
   @param stress_noise Noise hyperparameter for total stress.
   */
  ParallelSGP(std::vector<Kernel *> kernels, double energy_noise,
           double force_noise, double stress_noise);

//  std::vector<ClusterDescriptor>
//  initialize_sparse_descriptors(const Structure &structure, std::vector<ClusterDescriptor> sparse_desc);
  void initialize_local_sparse_descriptors(const Structure &structure);
  void initialize_global_sparse_descriptors(const Structure &structure);

  void add_global_noise(int n_energy, int n_force, int n_stress); 
  Eigen::VectorXd global_noise_vector, local_noise_vector;
  Eigen::MatrixXd local_labels;
  void add_training_structure(const Structure &structure);
  
  Eigen::VectorXi sparse_indices_by_type(int n_types, std::vector<int> species, const std::vector<int> atoms);    
  void add_specific_environments(const Structure&, std::vector<int>);
  void add_local_specific_environments(const Structure &structure, const std::vector<int> atoms);
  void add_global_specific_environments(const Structure &structure, const std::vector<int> atoms);
  void predict_local_uncertainties(Structure &structure);

  /**
   Method for constructing SGP model from training dataset.  

   @param training_strucs A list of Structure objects
   @param cutoff The cutoff for SGP
   @param descriptor_calculators A list of Descriptor objects, e.g. B2, B3, ...
   @param trianing_sparse_indices A list of indices of sparse environments in each training structure
   @param n_types An integer to specify number of types. For B2 descriptor, n_type is equal to the
        number of species
   */
  void build(const std::vector<Structure> &training_strucs,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices,
        int n_types);

  /**
   Method for loading training data distributedly. Each process loads a portion of the whole training
   data set, and load the whole sparse set.

   @param training_strucs A list of Structure objects
   @param cutoff The cutoff for SGP
   @param descriptor_calculators A list of Descriptor objects, e.g. B2, B3, ...
   @param trianing_sparse_indices A list of indices of sparse environments in each training structure
   @param n_types An integer to specify number of types. For B2 descriptor, n_type is equal to the
        number of species
   */
  void load_local_training_data(const std::vector<Structure> &training_strucs,
        double cutoff, std::vector<Descriptor *> descriptor_calculators,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices,
        int n_types);

  void gather_sparse_descriptors(std::vector<int> n_clusters_by_type,
        const std::vector<Structure> &training_strucs,
        const std::vector<std::vector<std::vector<int>>> &training_sparse_indices);

  /**
   Method for computing kernel matrices and vectors
 
   @param training_strucs A list of Structure objects
   */
  void compute_matrices(const std::vector<Structure> &training_strucs);


  Eigen::MatrixXd varmap_coeffs; // for debugging. TODO: remove this line 
};

#endif