#ifndef BK_H
#define BK_H

#include "descriptor.h"
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "json.h"

class Structure;

class Bk : public Descriptor {
public:
  std::function<void(std::vector<double> &, std::vector<double> &, double, int,
                     std::vector<double>)>
      radial_pointer;
  std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>
      cutoff_pointer;
  std::string radial_basis, cutoff_function;
  std::vector<double> radial_hyps, cutoff_hyps;
  std::vector<int> descriptor_settings;
  Eigen::VectorXd coeffs;
  std::vector<std::vector<int>> nu;

  std::string descriptor_name = "Bk";

  /** Matrix of cutoff values, with element (i, j) corresponding to the cutoff
   * assigned to the species pair (i, j), where i is the central species
   * and j is the environment species.
   */
  Eigen::MatrixXd cutoffs;

  Bk();

  Bk(const std::string &radial_basis, const std::string &cutoff_function,
     const std::vector<double> &radial_hyps,
     const std::vector<double> &cutoff_hyps,
     const std::vector<int> &descriptor_settings);

  /**
   * Construct the Bk descriptor with distinct cutoffs for each pair of species.
   */
  Bk(const std::string &radial_basis, const std::string &cutoff_function,
     const std::vector<double> &radial_hyps,
     const std::vector<double> &cutoff_hyps,
     const std::vector<int> &descriptor_settings,
     const Eigen::MatrixXd &cutoffs);

  DescriptorValues compute_struc(Structure &structure);
  void write_to_file(std::ofstream &coeff_file, int coeff_size);
  nlohmann::json return_json();
};

void compute_Bk(Eigen::MatrixXd &Bk_vals, Eigen::MatrixXd &Bk_force_dervs,
                Eigen::VectorXd &Bk_norms, Eigen::VectorXd &Bk_force_dots,
                const Eigen::MatrixXcd &single_bond_vals,
                const Eigen::MatrixXcd &single_bond_force_dervs,
                const Eigen::VectorXi &unique_neighbor_count,
                const Eigen::VectorXi &cumulative_neighbor_count,
                const Eigen::VectorXi &descriptor_indices, 
                std::vector<std::vector<int>> nu, int nos, int K, int N,
                int lmax, const Eigen::VectorXd &coeffs);

/**
 * Compute single bond vector with different cutoffs assigned to different
 * pairs of elements.
 */
void complex_single_bond_multiple_cutoffs(
    Eigen::MatrixXd &single_bond_vals, Eigen::MatrixXd &force_dervs,
    Eigen::MatrixXd &neighbor_coordinates, Eigen::VectorXi &neighbor_count,
    Eigen::VectorXi &cumulative_neighbor_count,
    Eigen::VectorXi &neighbor_indices,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        radial_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    int nos, int N, int lmax, const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps, const Structure &structure,
    const Eigen::MatrixXd &cutoffs);

/**
 * TODO: remove this function.
 */
void complex_single_bond(
    Eigen::MatrixXcd &single_bond_vals, Eigen::MatrixXcd &force_dervs,
    Eigen::MatrixXd &neighbor_coordinates, Eigen::VectorXi &neighbor_count,
    Eigen::VectorXi &cumulative_neighbor_count,
    Eigen::VectorXi &neighbor_indices,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        radial_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    int nos, int N, int lmax, const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps, const Structure &structure);

void to_json(nlohmann::json& j, const Bk & p);
void from_json(const nlohmann::json& j, Bk & p);

#endif
