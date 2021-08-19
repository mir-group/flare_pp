#include "bk.h"
#include "cutoffs.h"
#include "descriptor.h"
#include "omp.h"
#include "radial.h"
#include "structure.h"
#include "coeffs.h"
#include "indices.h"
#include "y_grad.h"
#include <fstream> // File operations
#include <iomanip> // setprecision
#include <iostream>

Bk ::Bk() {}

Bk ::Bk(const std::string &radial_basis, const std::string &cutoff_function,
        const std::vector<double> &radial_hyps,
        const std::vector<double> &cutoff_hyps,
        const std::vector<int> &descriptor_settings) {

  this->radial_basis = radial_basis;
  this->cutoff_function = cutoff_function;
  this->radial_hyps = radial_hyps;
  this->cutoff_hyps = cutoff_hyps;
  this->descriptor_settings = descriptor_settings; // nos, K, nmax, lmax

  // check if lmax = 0 when K = 1
  if (this->descriptor_settings[1] == 1 && this->descriptor_settings[3] != 0) {
    std::cout << "Warning: change lmax to 0 because K = 1" << std::endl;
    this->descriptor_settings[3] = 0;
  }

  nu = compute_indices(descriptor_settings); 
  std::cout << "nu size: " << nu.size() << std::endl;
  coeffs = compute_coeffs(descriptor_settings[1], descriptor_settings[3]);

  set_radial_basis(radial_basis, this->radial_pointer);
  set_cutoff(cutoff_function, this->cutoff_pointer);

  // Create cutoff matrix.
  int n_species = descriptor_settings[0];
  double cutoff_val = radial_hyps[1];
  cutoffs = Eigen::MatrixXd::Constant(n_species, n_species, cutoff_val);
}

Bk ::Bk(const std::string &radial_basis, const std::string &cutoff_function,
        const std::vector<double> &radial_hyps,
        const std::vector<double> &cutoff_hyps,
        const std::vector<int> &descriptor_settings,
        const Eigen::MatrixXd &cutoffs) {

  this->radial_basis = radial_basis;
  this->cutoff_function = cutoff_function;
  this->radial_hyps = radial_hyps;
  this->cutoff_hyps = cutoff_hyps;
  this->descriptor_settings = descriptor_settings; // nos, K, nmax, lmax

  // check if lmax = 0 when K = 1
  if (this->descriptor_settings[1] == 1 && this->descriptor_settings[3] != 0) {
    std::cout << "Warning: change lmax to 0 because K = 1" << std::endl;
    this->descriptor_settings[3] = 0;
  }

  nu = compute_indices(descriptor_settings); 
  std::cout << "nu size: " << nu.size() << std::endl;
  coeffs = compute_coeffs(descriptor_settings[1], descriptor_settings[3]);

  set_radial_basis(radial_basis, this->radial_pointer);
  set_cutoff(cutoff_function, this->cutoff_pointer);

  // Assign cutoff matrix.
  this->cutoffs = cutoffs;
}

void Bk ::write_to_file(std::ofstream &coeff_file, int coeff_size) {
  int n_species = descriptor_settings[0];
  int K = descriptor_settings[1];
  int n_max = descriptor_settings[2];
  int l_max = descriptor_settings[3];

  coeff_file << "\n" << "B" << K << "\n";

  // Report radial basis set.
  coeff_file << radial_basis << "\n";

  // Record number of species, nmax, lmax, and the cutoff.
  double cutoff = radial_hyps[1];

  coeff_file << n_species << " " << K << " " << n_max << " " << l_max << " ";
  coeff_file << coeff_size << "\n";
  coeff_file << cutoff_function << "\n";

  // Report cutoffs to 2 decimal places.
  coeff_file << std::fixed << std::setprecision(2);
  for (int i = 0; i < n_species; i ++){
    for (int j = 0; j < n_species; j ++){
      coeff_file << cutoffs(i, j) << " ";
    }
  }
  coeff_file << "\n";
}

DescriptorValues Bk ::compute_struc(Structure &structure) {

  // Initialize descriptor values.
  DescriptorValues desc = DescriptorValues();

  // Compute single bond values.
  Eigen::MatrixXcd single_bond_vals, force_dervs;
  Eigen::MatrixXd neighbor_coords;
  Eigen::VectorXi unique_neighbor_count, cumulative_neighbor_count,
      descriptor_indices;

  int nos = descriptor_settings[0];
  int K = descriptor_settings[1];
  int N = descriptor_settings[2];
  int lmax = descriptor_settings[3];

  complex_single_bond(single_bond_vals, force_dervs, neighbor_coords,
                      unique_neighbor_count, cumulative_neighbor_count,
                      descriptor_indices, radial_pointer, cutoff_pointer, nos,
                      N, lmax, radial_hyps, cutoff_hyps, structure, cutoffs);

  // Compute descriptor values.
  Eigen::MatrixXd Bk_vals, Bk_force_dervs;
  Eigen::VectorXd Bk_norms, Bk_force_dots;

  compute_Bk(Bk_vals, Bk_force_dervs, Bk_norms, Bk_force_dots, single_bond_vals,
             force_dervs, unique_neighbor_count, cumulative_neighbor_count,
             descriptor_indices, nu, nos, K, N, lmax, coeffs);

  // Gather species information.
  int noa = structure.noa;
  Eigen::VectorXi species_count = Eigen::VectorXi::Zero(nos);
  Eigen::VectorXi neighbor_count = Eigen::VectorXi::Zero(nos);
  for (int i = 0; i < noa; i++) {
    int s = structure.species[i];
    int n_neigh = unique_neighbor_count(i);
    species_count(s)++;
    neighbor_count(s) += n_neigh;
  }

  // Initialize arrays.
  int n_d = Bk_vals.cols();
  desc.n_descriptors = n_d;
  desc.n_types = nos;
  desc.n_atoms = noa;
  desc.volume = structure.volume;
  desc.cumulative_type_count.push_back(0);
  for (int s = 0; s < nos; s++) {
    int n_s = species_count(s);
    int n_neigh = neighbor_count(s);

    // Record species and neighbor count.
    desc.n_clusters_by_type.push_back(n_s);
    desc.cumulative_type_count.push_back(desc.cumulative_type_count[s] + n_s);
    desc.n_clusters += n_s;
    desc.n_neighbors_by_type.push_back(n_neigh);

    desc.descriptors.push_back(Eigen::MatrixXd::Zero(n_s, n_d));
    desc.descriptor_force_dervs.push_back(
        Eigen::MatrixXd::Zero(n_neigh * 3, n_d));
    desc.neighbor_coordinates.push_back(Eigen::MatrixXd::Zero(n_neigh, 3));

    desc.cutoff_values.push_back(Eigen::VectorXd::Ones(n_s));
    desc.cutoff_dervs.push_back(Eigen::VectorXd::Zero(n_neigh * 3));
    desc.descriptor_norms.push_back(Eigen::VectorXd::Zero(n_s));
    desc.descriptor_force_dots.push_back(Eigen::VectorXd::Zero(n_neigh * 3));

    desc.neighbor_counts.push_back(Eigen::VectorXi::Zero(n_s));
    desc.cumulative_neighbor_counts.push_back(Eigen::VectorXi::Zero(n_s));
    desc.atom_indices.push_back(Eigen::VectorXi::Zero(n_s));
    desc.neighbor_indices.push_back(Eigen::VectorXi::Zero(n_neigh));
  }

  // Assign to structure.
  Eigen::VectorXi species_counter = Eigen::VectorXi::Zero(nos);
  Eigen::VectorXi neighbor_counter = Eigen::VectorXi::Zero(nos);
  for (int i = 0; i < noa; i++) {
    int s = structure.species[i];
    int s_count = species_counter(s);
    int n_neigh = unique_neighbor_count(i);
    int n_count = neighbor_counter(s);
    int cum_neigh = cumulative_neighbor_count(i);

    desc.descriptors[s].row(s_count) = Bk_vals.row(i);
    desc.descriptor_force_dervs[s].block(n_count * 3, 0, n_neigh * 3, n_d) =
        Bk_force_dervs.block(cum_neigh * 3, 0, n_neigh * 3, n_d);
    desc.neighbor_coordinates[s].block(n_count, 0, n_neigh, 3) =
        neighbor_coords.block(cum_neigh, 0, n_neigh, 3);

    desc.descriptor_norms[s](s_count) = Bk_norms(i);
    desc.descriptor_force_dots[s].segment(n_count * 3, n_neigh * 3) =
        Bk_force_dots.segment(cum_neigh * 3, n_neigh * 3);

    desc.neighbor_counts[s](s_count) = n_neigh;
    desc.cumulative_neighbor_counts[s](s_count) = n_count;
    desc.atom_indices[s](s_count) = i;
    desc.neighbor_indices[s].segment(n_count, n_neigh) =
        descriptor_indices.segment(cum_neigh, n_neigh);

    species_counter(s)++;
    neighbor_counter(s) += n_neigh;
  }

  return desc;
}

void compute_Bk(Eigen::MatrixXd &Bk_vals, Eigen::MatrixXd &Bk_force_dervs,
                Eigen::VectorXd &Bk_norms, Eigen::VectorXd &Bk_force_dots,
                const Eigen::MatrixXcd &single_bond_vals,
                const Eigen::MatrixXcd &single_bond_force_dervs,
                const Eigen::VectorXi &unique_neighbor_count,
                const Eigen::VectorXi &cumulative_neighbor_count,
                const Eigen::VectorXi &descriptor_indices, 
                std::vector<std::vector<int>> nu, int nos, int K, int N,
                int lmax, const Eigen::VectorXd &coeffs) {

  int n_atoms = single_bond_vals.rows();
  int n_neighbors = cumulative_neighbor_count(n_atoms);

  // The value of last counter is the number of descriptors
  std::vector<int> last_index = nu[nu.size()-1];
  int n_d = last_index[last_index.size()-1] + 1; 

  // Initialize arrays.
  Bk_vals = Eigen::MatrixXd::Zero(n_atoms, n_d);
  Bk_force_dervs = Eigen::MatrixXd::Zero(n_neighbors * 3, n_d);
  Bk_norms = Eigen::VectorXd::Zero(n_atoms);
  Bk_force_dots = Eigen::VectorXd::Zero(n_neighbors * 3);

#pragma omp parallel for
  for (int atom = 0; atom < n_atoms; atom++) {
    int n_atom_neighbors = unique_neighbor_count(atom);
    int force_start = cumulative_neighbor_count(atom) * 3;
    for (int i = 0; i < nu.size(); i++) {
      std::vector<int> nu_list = nu[i];
      std::vector<int> single_bond_index = std::vector<int>(nu_list.end() - 2 - K, nu_list.end() - 2); // Get n1_l, n2_l, n3_l, etc.
      // Forward
      std::complex<double> A_fwd = 1;
      Eigen::VectorXcd dA = Eigen::VectorXcd::Ones(K);
      for (int t = 0; t < K - 1; t++) {
        A_fwd *= single_bond_vals(atom, single_bond_index[t]);
        dA(t + 1) *= A_fwd;
      }
      // Backward
      std::complex<double> A_bwd = 1;
      for (int t = K - 1; t > 0; t--) {
        A_bwd *= single_bond_vals(atom, single_bond_index[t]);
        dA(t - 1) *= A_bwd;
      }
      std::complex<double> A = A_fwd * single_bond_vals(atom, single_bond_index[K - 1]);

      int counter = nu_list[nu_list.size() - 1];
      int m_index = nu_list[nu_list.size() - 2];
      Bk_vals(atom, counter) += real(coeffs(m_index) * A); 

      // Store force derivatives.
      for (int n = 0; n < n_atom_neighbors; n++) {
        for (int comp = 0; comp < 3; comp++) {
          int ind = force_start + n * 3 + comp;
          std::complex<double> dA_dr = 1;
          for (int t = 0; t < K; t++) {
            dA_dr += dA(t) * single_bond_force_dervs(ind, single_bond_index[t]);
          }
          Bk_force_dervs(ind, counter) +=
              real(coeffs(m_index) * dA_dr);
        }
      }
    }
    // Compute descriptor norm and force dot products.
    Bk_norms(atom) = sqrt(Bk_vals.row(atom).dot(Bk_vals.row(atom)));
    Bk_force_dots.segment(force_start, n_atom_neighbors * 3) =
        Bk_force_dervs.block(force_start, 0, n_atom_neighbors * 3, n_d) *
        Bk_vals.row(atom).transpose();
  }
}

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
    const std::vector<double> &cutoff_hyps, const Structure &structure,
    const Eigen::MatrixXd &cutoffs) {

  int n_atoms = structure.noa;
  int n_neighbors = structure.n_neighbors;

  // Count atoms inside the descriptor cutoff.
  neighbor_count = Eigen::VectorXi::Zero(n_atoms);
  Eigen::VectorXi store_neighbors = Eigen::VectorXi::Zero(n_neighbors);
#pragma omp parallel for
  for (int i = 0; i < n_atoms; i++) {
    int i_neighbors = structure.neighbor_count(i);
    int rel_index = structure.cumulative_neighbor_count(i);
    int central_species = structure.species[i];
    for (int j = 0; j < i_neighbors; j++) {
      int current_count = neighbor_count(i);
      int neigh_index = rel_index + j;
      int neighbor_species = structure.neighbor_species(neigh_index);
      double rcut = cutoffs(central_species, neighbor_species);
      double r = structure.relative_positions(neigh_index, 0);
      // Check that atom is within descriptor cutoff.
      if (r <= rcut) {
        int struc_index = structure.structure_indices(neigh_index);
        // Update neighbor list.
        store_neighbors(rel_index + current_count) = struc_index;
        neighbor_count(i)++;
      }
    }
  }

  // Count cumulative number of unique neighbors.
  cumulative_neighbor_count = Eigen::VectorXi::Zero(n_atoms + 1);
  for (int i = 1; i < n_atoms + 1; i++) {
    cumulative_neighbor_count(i) +=
        cumulative_neighbor_count(i - 1) + neighbor_count(i - 1);
  }

  // Record neighbor indices.
  int bond_neighbors = cumulative_neighbor_count(n_atoms);
  neighbor_indices = Eigen::VectorXi::Zero(bond_neighbors);
#pragma omp parallel for
  for (int i = 0; i < n_atoms; i++) {
    int i_neighbors = neighbor_count(i);
    int ind1 = cumulative_neighbor_count(i);
    int ind2 = structure.cumulative_neighbor_count(i);
    for (int j = 0; j < i_neighbors; j++) {
      neighbor_indices(ind1 + j) = store_neighbors(ind2 + j);
    }
  }

  // Initialize single bond arrays.
  int number_of_harmonics = (lmax + 1) * (lmax + 1);
  int no_bond_vals = N * number_of_harmonics;
  int single_bond_size = no_bond_vals * nos;

  single_bond_vals = Eigen::MatrixXcd::Zero(n_atoms, single_bond_size);
  force_dervs = Eigen::MatrixXcd::Zero(bond_neighbors * 3, single_bond_size);
  neighbor_coordinates = Eigen::MatrixXd::Zero(bond_neighbors, 3);

#pragma omp parallel for
  for (int i = 0; i < n_atoms; i++) {
    int i_neighbors = structure.neighbor_count(i);
    int rel_index = structure.cumulative_neighbor_count(i);
    int neighbor_index = cumulative_neighbor_count(i);
    int central_species = structure.species[i];

    // Initialize radial hyperparameters.
    std::vector<double> new_radial_hyps = radial_hyps;

    // Initialize radial and spherical harmonic vectors.
    std::vector<double> g = std::vector<double>(N, 0);
    std::vector<double> gx = std::vector<double>(N, 0);
    std::vector<double> gy = std::vector<double>(N, 0);
    std::vector<double> gz = std::vector<double>(N, 0);

    Eigen::VectorXcd h, hx, hy, hz;

    double x, y, z, r, g_val, gx_val, gy_val, gz_val;
    std::complex<double> bond, bond_x, bond_y, bond_z, h_val;
    int s, neigh_index, descriptor_counter, unique_ind;
    for (int j = 0; j < i_neighbors; j++) {
      neigh_index = rel_index + j;
      int neighbor_species = structure.neighbor_species(neigh_index);
      double rcut = cutoffs(central_species, neighbor_species);
      r = structure.relative_positions(neigh_index, 0);
      if (r > rcut)
        continue; // Skip if outside cutoff.
      x = structure.relative_positions(neigh_index, 1);
      y = structure.relative_positions(neigh_index, 2);
      z = structure.relative_positions(neigh_index, 3);
      s = structure.neighbor_species(neigh_index);

      // Reset the endpoint of the radial basis set.
      new_radial_hyps[1] = rcut;

      // Store neighbor coordinates.
      neighbor_coordinates(neighbor_index, 0) = x;
      neighbor_coordinates(neighbor_index, 1) = y;
      neighbor_coordinates(neighbor_index, 2) = z;

      // Compute radial basis values and spherical harmonics.
      calculate_radial(g, gx, gy, gz, radial_function, cutoff_function, x, y, z,
                       r, rcut, N, new_radial_hyps, cutoff_hyps);
      get_complex_Y(h, hx, hy, hz, x, y, z, lmax);

      // Store the products and their derivatives.
      descriptor_counter = s * no_bond_vals;

      for (int radial_counter = 0; radial_counter < N; radial_counter++) {
        // Retrieve radial values.
        g_val = g[radial_counter];
        gx_val = gx[radial_counter];
        gy_val = gy[radial_counter];
        gz_val = gz[radial_counter];

        for (int angular_counter = 0; angular_counter < number_of_harmonics;
             angular_counter++) {

          // Compute single bond value.
          h_val = h(angular_counter);
          bond = g_val * h_val;

          // Calculate derivatives with the product rule.
          bond_x = gx_val * h_val + g_val * hx(angular_counter);
          bond_y = gy_val * h_val + g_val * hy(angular_counter);
          bond_z = gz_val * h_val + g_val * hz(angular_counter);

          // Update single bond arrays.
          single_bond_vals(i, descriptor_counter) += bond;

          force_dervs(neighbor_index * 3, descriptor_counter) += bond_x;
          force_dervs(neighbor_index * 3 + 1, descriptor_counter) += bond_y;
          force_dervs(neighbor_index * 3 + 2, descriptor_counter) += bond_z;

          descriptor_counter++;
        }
      }
      neighbor_index++;
    }
  }
}

void to_json(nlohmann::json& j, const Bk & p){
  j = nlohmann::json{
    {"radial_basis", p.radial_basis},
    {"cutoff_function", p.cutoff_function},
    {"radial_hyps", p.radial_hyps},
    {"cutoff_hyps", p.cutoff_hyps},
    {"descriptor_settings", p.descriptor_settings},
    {"cutoffs", p.cutoffs},
    {"descriptor_name", p.descriptor_name}
  };
}

void from_json(const nlohmann::json& j, Bk & p){
  p = Bk(
    j.at("radial_basis"),
    j.at("cutoff_function"),
    j.at("radial_hyps"),
    j.at("radial_hyps"),
    j.at("descriptor_settings"),
    j.at("cutoffs")
  );
}

nlohmann::json Bk ::return_json(){
  nlohmann::json j;
  to_json(j, *this);
  return j;
}