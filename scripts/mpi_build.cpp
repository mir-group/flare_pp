#include <chrono>
#include <iostream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <Eigen/Dense>
#include <blacs.h>
#include "omp.h"
#include "mpi.h"

#include "utils.h"
#include "b2.h"
#include "parallel_sgp.h"
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
  std::vector<double> single_atom_energy;

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
      } else if (v[0] == std::string("single_atom_energy")) {
        for (int s = 1; s < v.size(); s++) {
          single_atom_energy.push_back(std::stod(v[s]));
        }
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

  if (!input_species) throw;    // Check the species are input
  if (single_atom_energy.size() == 0) {     // Check single atom energies are input
    for (int s = 0; s < n_species; s++) {   // Otherwise set to 0
      single_atom_energy.push_back(0.0);
    }
  } else {                                  // Single atoms energies should correspond
    assert(single_atom_energy.size() == n_species); // to species list
  }

  // Set descriptors for SGP
  std::vector<double> radial_hyps{0, cutoff};
  std::vector<double> cutoff_hyps;

  std::vector<int> descriptor_settings{n_species, N, L};

  std::vector<Descriptor *> dc;
  B2 ps(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
        descriptor_settings);
  dc.push_back(&ps);

  // Set kernels for SGP
  NormalizedDotProduct kernel_norm = NormalizedDotProduct(sigma, power);
  std::vector<Kernel *> kernels;
  kernels.push_back(&kernel_norm);

  std::cout << "Kuu_jitter=" << Kuu_jitter << std::endl;
  ParallelSGP parallel_sgp = ParallelSGP(kernels, sigma_e, sigma_f, sigma_s);
  parallel_sgp.Kuu_jitter = Kuu_jitter;

  // Read data from .xyz file
  std::vector<Structure> struc_list;
  std::vector<std::vector<std::vector<int>>> sparse_indices;
  std::tie(struc_list, sparse_indices) = utils::read_xyz(filename, species_map);

  // Build parallel sgp
  int n_types = n_species;
  parallel_sgp.build(struc_list, cutoff, dc, sparse_indices, n_types);
  std::cout << "Parallel_sgp is built!" << std::endl;

  if (blacs::mpirank == 0) {
    parallel_sgp.write_mapping_coefficients(coefname, contributor, {0});
    std::cout << "Mapping coefficients are written" << std::endl;

    // validate the Kuu is symmetric
    for (int r = 0; r < parallel_sgp.Kuu.rows(); r++) {
      for (int c = r; c < parallel_sgp.Kuu.cols(); c++) {
        double dKuu = parallel_sgp.Kuu(r, c) - parallel_sgp.Kuu(c, r);
        if (std::abs(dKuu) > 1e-6) {
          std::cout << r << " " << c << " " << dKuu << " " << parallel_sgp.Kuu(r, c) << std::endl;
        }
      }
    }

    std::cout << "start L_inv" << std::endl;
    for (int r = 0; r < parallel_sgp.L_inv.rows(); r++) {
      for (int c = 0; c <= r; c++) {
        std::cout << r << " " << c << " " << parallel_sgp.L_inv(r, c) << std::endl;
      }
    }
    std::cout << "end L_inv" << std::endl;

    std::cout << "start R_inv" << std::endl;
    for (int r = 0; r < parallel_sgp.R_inv.rows(); r++) {
      for (int c = r; c < parallel_sgp.R_inv.cols(); c++) {
        std::cout << r << " " << c << " " << parallel_sgp.R_inv(r, c) << std::endl;
      }
    }
    std::cout << "end R_inv" << std::endl;

    std::cout << "start R" << std::endl;
    for (int r = 0; r < parallel_sgp.R.rows(); r++) {
      for (int c = r; c < parallel_sgp.R.cols(); c++) {
        std::cout << r << " " << c << " " << std::setprecision (17) << parallel_sgp.R(r, c) << std::endl;
      }
    }
    std::cout << "end R" << std::endl;


    std::cout << "Start Q_b" << std::endl;
    for (int r = 0; r < parallel_sgp.Q_b.size(); r++) {
      std::cout << std::setprecision (17) << parallel_sgp.Q_b(r) << std::endl;
    }
    std::cout << "End Q_b" << std::endl;

    std::cout << "Start alpha" << std::endl;
    for (int r = 0; r < parallel_sgp.alpha.size(); r++) {
      std::cout << std::setprecision (17) << parallel_sgp.alpha(r) << std::endl;
    }
    std::cout << "End alpha" << std::endl;
  }

}
