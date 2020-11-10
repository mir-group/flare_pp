#include "normalized_dot_product.h"
#include "compact_structure.h"
#include "descriptor.h"
#include "four_body.h"
#include "three_body.h"
#include "three_body_wide.h"
#include "two_body.h"
#include "dot_product_kernel.h"
#include "squared_exponential.h"
#include "local_environment.h"
#include "b2.h"
#include "structure.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdlib.h>

class CompactStructureTest : public ::testing::Test {
public:
  int n_atoms = 10;
  int n_species = 2;
  Eigen::MatrixXd cell, cell_2;
  std::vector<int> species, species_2;
  Eigen::MatrixXd positions, positions_2;
  B2_Calculator desc1;
  B2 ps;
  std::vector<DescriptorCalculator *> descriptor_calculators;
  std::vector<CompactDescriptor *> dc;
  CompactStructure test_struc, test_struc_2;
  StructureDescriptor struc2;
  DescriptorValues struc_desc;

  double cell_size = 10;
  double cutoff = cell_size / 2;
  int N = 3;
  int L = 3;
  std::string radial_string = "chebyshev";
  std::string cutoff_string = "cosine";
  std::vector<double> radial_hyps{0, cutoff};
  std::vector<double> cutoff_hyps;
  std::vector<int> descriptor_settings{n_species, N, L};
  int descriptor_index = 0;
  std::vector<double> many_body_cutoffs{cutoff};

  double sigma = 2.0;
  double ls = 0.9;
  int power = 2;
  NormalizedDotProduct kernel_3;
  DotProductKernel kernel_2;
  SquaredExponential kernel;

  CompactStructureTest() {
    // Make positions.
    cell = Eigen::MatrixXd::Identity(3, 3) * cell_size;
    cell_2 = Eigen::MatrixXd::Identity(3, 3) * cell_size;
    positions = Eigen::MatrixXd::Random(n_atoms, 3) * cell_size / 2;
    positions_2 = Eigen::MatrixXd::Random(n_atoms, 3) * cell_size / 2;

    // Make random species.
    for (int i = 0; i < n_atoms; i++) {
      species.push_back(rand() % n_species);
      species_2.push_back(rand() % n_species);
    }

    desc1 = B2_Calculator(radial_string, cutoff_string, radial_hyps,
                          cutoff_hyps, descriptor_settings, descriptor_index);
    ps = B2(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
            descriptor_settings);

    descriptor_calculators.push_back(&desc1);
    dc.push_back(&ps);

    test_struc = CompactStructure(cell, species, positions, cutoff, dc);
    test_struc_2 = CompactStructure(cell_2, species_2, positions_2, cutoff, dc);
    struc2 = StructureDescriptor(cell, species, positions, cutoff,
                                 many_body_cutoffs, descriptor_calculators);
    
    struc_desc = test_struc.descriptors[0];

    kernel = SquaredExponential(sigma, ls);
    kernel_2 = DotProductKernel(sigma, power, 0);
    kernel_3 = NormalizedDotProduct(sigma, power);
  }
};