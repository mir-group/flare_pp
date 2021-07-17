#include "bk.h"
#include "b2.h"
#include "b3.h"
#include "descriptor.h"
#include "test_structure.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <cmath>
#include <list>
#include <iostream>

// Test different types
template <typename T>
class DescTest : public StructureTest {
public:
  using List = std::list<T>;
};

using DescTypes = ::testing::Types<B2, B3>;
TYPED_TEST_SUITE(DescTest, DescTypes);

TYPED_TEST(DescTest, TestBk) {
  // Set up B1/2/3 descriptors
  std::vector<int> descriptor_settings_1{this->n_species, this->N, this->L};
  std::vector<Descriptor *> descriptors1;

  TypeParam desc1(this->radial_string, this->cutoff_string, this->radial_hyps, 
                  this->cutoff_hyps, descriptor_settings_1);
  descriptors1.push_back(&desc1);

  // Set up Bk descriptors
  int K = desc1.K;
  std::vector<int> descriptor_settings_2{this->n_species, K, this->N, this->L};
  Bk desc2(this->radial_string, this->cutoff_string, this->radial_hyps, 
           this->cutoff_hyps, descriptor_settings_2);
  std::vector<Descriptor *> descriptors2;
  descriptors2.push_back(&desc2);
 
  Structure struc1 = Structure(this->cell, this->species, this->positions, this->cutoff, descriptors1);
  Structure struc2 = Structure(this->cell, this->species, this->positions, this->cutoff, descriptors2);
  
  // Check the descriptor dimensions
  std::vector<int> last_index = desc2.nu[desc2.nu.size()-1];
  int n_d = last_index[last_index.size()-1] + 1; // the size of list nu
  int n_d1 = struc1.descriptors[0].n_descriptors;
  int n_d2 = struc2.descriptors[0].n_descriptors;
  EXPECT_EQ(n_d, n_d1);
  EXPECT_EQ(n_d1, n_d2);
  
  // Check that Bk and B3 give the same descriptors.
  double d1, d2;
  int nu_ind;
  for (int i = 0; i < struc1.descriptors.size(); i++) {
    for (int j = 0; j < struc1.descriptors[i].descriptors.size(); j++) {
      for (int k = 0; k < struc1.descriptors[i].descriptors[j].rows(); k++) {
        for (int l = 0; l < struc1.descriptors[i].descriptors[j].cols(); l++) {
          d1 = struc1.descriptors[i].descriptors[j](k, l);
          d2 = struc2.descriptors[i].descriptors[j](k, l);
          EXPECT_NEAR(d1, d2, 1e-8);
        }
      }
    }
  }

}

TEST_F(StructureTest, RotationTest) {

  // Choose arbitrary rotation angles.
  double xrot = 1.28;
  double yrot = -3.21;
  double zrot = 0.42;

  // Define rotation matrices.
  Eigen::MatrixXd Rx{3, 3}, Ry{3, 3}, Rz{3, 3}, R{3, 3};
  Rx << 1, 0, 0, 0, cos(xrot), -sin(xrot), 0, sin(xrot), cos(xrot);
  Ry << cos(yrot), 0, sin(yrot), 0, 1, 0, -sin(yrot), 0, cos(yrot);
  Rz << cos(zrot), -sin(zrot), 0, sin(zrot), cos(zrot), 0, 0, 0, 1;
  R = Rx * Ry * Rz;

  Eigen::MatrixXd rotated_pos = positions * R.transpose();
  Eigen::MatrixXd rotated_cell = cell * R.transpose();

  // Define descriptors.
  //descriptor_settings[2] = 2;
  int lmax = 5;
  int K = 2;
  int nos = n_species;

  std::vector<int> descriptor_settings{n_species, K, N, lmax};
  //std::vector<int> descriptor_settings{n_species, N, lmax};
  Bk desc(radial_string, cutoff_string, radial_hyps, cutoff_hyps,
          descriptor_settings);

  std::vector<Descriptor *> descriptors;
  descriptors.push_back(&desc);

  Structure struc1 = Structure(cell, species, positions, cutoff, descriptors);
  Structure struc2 =
      Structure(rotated_cell, species, rotated_pos, cutoff, descriptors);

  // Check that B1 is rotationally invariant.
  double d1, d2;

  std::cout << "n_descriptors=" << struc1.descriptors[0].n_descriptors << std::endl;
  for (int n = 0; n < struc1.descriptors[0].n_descriptors; n++) {
    d1 = struc1.descriptors[0].descriptors[0](0, n);
    d2 = struc2.descriptors[0].descriptors[0](0, n);
    EXPECT_NEAR(d1, d2, 1e-10);
  }
}

  // TEST_F(DescriptorTest, SingleBond) {
  //   // Check that B1 descriptors match the corresponding elements of the
  //   // single bond vector.
  //   double d1, d2, diff;
  //   double tol = 1e-10;

  //   desc1.compute(env1);
  //   desc2.compute(env2);

  //   for (int n = 0; n < no_desc; n++) {
  //     d1 = desc1.descriptor_vals(n);
  //     d2 = desc1.single_bond_vals(n);
  //     diff = d1 - d2;
  //     EXPECT_LE(abs(diff), tol);
  //   }
// }

// TEST_F(DescriptorTest, CentTest) {
//   double finite_diff, exact, diff;
//   double tolerance = 1e-5;

//   desc1.compute(env1);
//   desc2.compute(env2);

//   // Perturb the coordinates of the central atom.
//   for (int m = 0; m < 3; m++) {
//     positions_3 = positions_1;
//     positions_3(0, m) += delta;
//     struc3 = Structure(cell, species, positions_3);
//     env3 = LocalEnvironment(struc3, 0, rcut);
//     env3.many_body_cutoffs = many_body_cutoffs;
//     env3.compute_indices();
//     desc3.compute(env3);

//     // Check derivatives.
//     for (int n = 0; n < no_desc; n++) {
//       finite_diff =
//           (desc3.descriptor_vals(n) - desc1.descriptor_vals(n)) / delta;
//       exact = desc1.descriptor_force_dervs(m, n);
//       diff = abs(finite_diff - exact);
//       EXPECT_LE(diff, tolerance);
//     }
//   }

//   int lmax = 8;
//   desc4.compute(env1);
//   desc5.compute(env2);
//   no_desc = desc4.descriptor_vals.rows();

//   // Perturb the coordinates of the central atom.
//   for (int m = 0; m < 3; m++) {
//     positions_3 = positions_1;
//     positions_3(0, m) += delta;
//     struc3 = Structure(cell, species, positions_3);
//     env3 = LocalEnvironment(struc3, 0, rcut);
//     env3.many_body_cutoffs = many_body_cutoffs;
//     env3.compute_indices();
//     desc6.compute(env3);

//     // Check derivatives.
//     for (int n = 0; n < no_desc; n++) {
//       finite_diff =
//           (desc6.descriptor_vals(n) - desc4.descriptor_vals(n)) / delta;
//       exact = desc4.descriptor_force_dervs(m, n);
//       diff = abs(finite_diff - exact);
//       EXPECT_LE(diff, tolerance);
//     }
//   }
// }

// TEST_F(DescriptorTest, EnvTest) {
//   double finite_diff, exact, diff;
//   double tolerance = 1e-5;

//   desc1.compute(env1);
//   desc2.compute(env2);

//   // Perturb the coordinates of the environment atoms.
//   for (int p = 1; p < noa; p++) {
//     for (int m = 0; m < 3; m++) {
//       positions_3 = positions_1;
//       positions_3(p, m) += delta;
//       struc3 = Structure(cell, species, positions_3);
//       env3 = LocalEnvironment(struc3, 0, rcut);
//       env3.many_body_cutoffs = many_body_cutoffs;
//       env3.compute_indices();
//       desc3.compute(env3);

//       // Check derivatives.
//       for (int n = 0; n < no_desc; n++) {
//         finite_diff =
//             (desc3.descriptor_vals(n) - desc1.descriptor_vals(n)) / delta;
//         exact = desc1.descriptor_force_dervs(p * 3 + m, n);
//         diff = abs(finite_diff - exact);
//         EXPECT_LE(diff, tolerance);
//       }
//     }
//   }

//   int lmax = 8;
//   desc4.compute(env1);
//   desc5.compute(env2);
//   no_desc = desc1.descriptor_vals.rows();

//   // Perturb the coordinates of the environment atoms.
//   for (int p = 1; p < noa; p++) {
//     for (int m = 0; m < 3; m++) {
//       positions_3 = positions_1;
//       positions_3(p, m) += delta;
//       struc3 = Structure(cell, species, positions_3);
//       env3 = LocalEnvironment(struc3, 0, rcut);
//       env3.many_body_cutoffs = many_body_cutoffs;
//       env3.compute_indices();
//       desc6.compute(env3);

//       // Check derivatives.
//       for (int n = 0; n < no_desc; n++) {
//         finite_diff =
//             (desc6.descriptor_vals(n) - desc5.descriptor_vals(n)) / delta;
//         exact = desc4.descriptor_force_dervs(p * 3 + m, n);
//         diff = abs(finite_diff - exact);
//         EXPECT_LE(diff, tolerance);
//       }
//     }
//   }
// }

// TEST_F(DescriptorTest, StressTest) {
//   int stress_ind = 0;
//   double finite_diff, exact, diff;
//   double tolerance = 1e-5;

//   desc1.compute(env1);
//   desc2.compute(env2);

//   // Test all 6 independent strains (xx, xy, xz, yy, yz, zz).
//   for (int m = 0; m < 3; m++) {
//     for (int n = m; n < 3; n++) {
//       cell_2 = cell;
//       positions_2 = positions_1;

//       // Perform strain.
//       cell_2(0, m) += cell(0, n) * delta;
//       cell_2(1, m) += cell(1, n) * delta;
//       cell_2(2, m) += cell(2, n) * delta;
//       for (int k = 0; k < noa; k++) {
//         positions_2(k, m) += positions_1(k, n) * delta;
//       }

//       struc2 = Structure(cell_2, species, positions_2);
//       env2 = LocalEnvironment(struc2, 0, rcut);
//       env2.many_body_cutoffs = many_body_cutoffs;
//       env2.compute_indices();
//       desc2.compute(env2);

//       // Check stress derivatives.
//       for (int p = 0; p < no_desc; p++) {
//         finite_diff =
//             (desc2.descriptor_vals(p) - desc1.descriptor_vals(p)) / delta;
//         exact = desc1.descriptor_stress_dervs(stress_ind, p);
//         diff = abs(finite_diff - exact);
//         EXPECT_LE(diff, tolerance);
//       }

//       stress_ind++;
//     }
//   }

//   int lmax = 8;
//   desc4.compute(env1);
//   desc5.compute(env2);
//   no_desc = desc1.descriptor_vals.rows();
//   stress_ind = 0;

//   // Test all 6 independent strains (xx, xy, xz, yy, yz, zz).
//   for (int m = 0; m < 3; m++) {
//     for (int n = m; n < 3; n++) {
//       cell_2 = cell;
//       positions_2 = positions_1;

//       // Perform strain.
//       cell_2(0, m) += cell(0, n) * delta;
//       cell_2(1, m) += cell(1, n) * delta;
//       cell_2(2, m) += cell(2, n) * delta;
//       for (int k = 0; k < noa; k++) {
//         positions_2(k, m) += positions_1(k, n) * delta;
//       }

//       struc2 = Structure(cell_2, species, positions_2);
//       env2 = LocalEnvironment(struc2, 0, rcut);
//       env2.many_body_cutoffs = many_body_cutoffs;
//       env2.compute_indices();
//       desc5.compute(env2);

//       // Check stress derivatives.
//       for (int p = 0; p < no_desc; p++) {
//         finite_diff =
//             (desc5.descriptor_vals(p) - desc4.descriptor_vals(p)) / delta;
//         exact = desc4.descriptor_stress_dervs(stress_ind, p);
//         diff = abs(finite_diff - exact);
//         EXPECT_LE(diff, tolerance);
//       }

//       stress_ind++;
//     }
//   }
// }
