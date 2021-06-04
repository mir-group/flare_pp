#include <chrono>
#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include "omp.h"

#include "utils.h"
#include "b2.h"
#include "sparse_gp.h"
#include "structure.h"
#include "normalized_dot_product.h"

int main(int argc, char* argv[]) {
  // Default setting of descriptors
  int N = 8;
  int L = 4;
  double cutoff = 4.0;
  std::string radial_string = "chebyshev";
  std::string cutoff_string = "quadratic";

  // Species setting
  int n_species = 1;
  bool input_species = false;
  std::map<std::string, int> species_map; 

  // Default setting of kernels
  double sigma = 7.0;
  int power = 2;

  // Default setting of SGP
  double sigma_e = 0.1;
  double sigma_f = 0.1;
  double sigma_s = 0.001;
  double Kuu_jitter = 1e-8;

  // Default input file
  std::string filename = "dft_data.xyz";
  std::string coefname = "par_beta.txt";
  std::string contributor = "Me";

  // Read input file
  std::ifstream file("input.flare");
  std::vector<std::string> v;
  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      v = utils::split(line, ' ');
      if (v[0] == std::string("cutoff")) {
        cutoff = std::stod(v[1]);
      } else if (v[0] == std::string("nmax")) {
        N = std::stoi(v[1]);
      } else if (v[0] == std::string("lmax")) {
        L = std::stoi(v[1]);
      } else if (v[0] == std::string("radial_func")) {
        radial_string = v[1];
      } else if (v[0] == std::string("cutoff_func")) {
        cutoff_string = v[1];
      } else if (v[0] == std::string("species")) {
        for (int s = 1; s < v.size(); s++) {
          species_map[v[s]] = s - 1;
        }
        n_species = v.size() - 1;
        input_species = true;
      } else if (v[0] == std::string("sigma")) {
        sigma = std::stod(v[1]);
      } else if (v[0] == std::string("power")) {
        power = std::stoi(v[1]);
      } else if (v[0] == std::string("sigma_e")) {
        sigma_e = std::stod(v[1]);
      } else if (v[0] == std::string("sigma_f")) {
        sigma_f = std::stod(v[1]);
      } else if (v[0] == std::string("sigma_s")) {
        sigma_s = std::stod(v[1]);
      } else if (v[0] == std::string("Kuu_jitter")) {
        Kuu_jitter = std::stod(v[1]);
      } else if (v[0] == std::string("data_file")) {
        filename = v[1];
      } else if (v[0] == std::string("coef_file")) {
        coefname = v[1];
      } else if (v[0] == std::string("contributor")) {
        contributor = v[1];
      }
    }
  }

  if (!input_species) throw;

  std::vector<double> radial_hyps{0, cutoff};
  std::vector<double> cutoff_hyps;

  std::vector<int> descriptor_settings{n_species, N, L};

  std::vector<Descriptor *> dc;
  B2 ps(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
        descriptor_settings);
  dc.push_back(&ps);

  NormalizedDotProduct kernel_norm = NormalizedDotProduct(sigma, power);
  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel_norm);

  std::cout << "Kuu_jitter=" << Kuu_jitter << std::endl;
  SparseGP sparse_gp(kernels, sigma_e, sigma_f, sigma_s);
  sparse_gp.Kuu_jitter = Kuu_jitter;

  // Read data from .xyz file
  std::vector<Structure> struc_list;
  std::vector<std::vector<std::vector<int>>> sparse_indices;
  std::tie(struc_list, sparse_indices) = utils::read_xyz(filename, species_map);

  // Build parallel sgp
  int n_types = n_species;
  for (int t = 0; t < struc_list.size(); t++) {
    std::cout << "Adding structure " << t << std::endl;
    Structure struc(struc_list[t].cell, struc_list[t].species, 
            struc_list[t].positions, cutoff, dc);

    struc.energy = struc_list[t].energy;
    struc.forces = struc_list[t].forces;
    struc.stresses = struc_list[t].stresses;
 
    sparse_gp.add_training_structure(struc);
    sparse_gp.add_specific_environments(struc, sparse_indices[0][t]);
  }
  std::cout << "Finish adding. Start building matrices" << std::endl;
  sparse_gp.update_matrices_QR();
  std::cout << "Sparse GP is built!" << std::endl;

  sparse_gp.write_mapping_coefficients(coefname, contributor, 0);
  std::cout << "Mapping coefficients are written" << std::endl;

}
