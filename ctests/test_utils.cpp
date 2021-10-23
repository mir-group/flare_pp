#include "utils.h"
#include "structure.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

class UtilsTest : public ::testing::Test {
protected:
  std::vector<Structure> struc_list;
  std::vector<std::vector<std::vector<int>>> sparse_indices;
  std::string filename = std::string("dft_data.xyz");
  std::map<std::string, int> species_map = {{"H", 0,}, {"He", 1,}}; 
};

TEST_F(UtilsTest, XYZTest) {
  std::tie(struc_list, sparse_indices) = utils::read_xyz(filename, species_map);
  std::cout << "Done read" << std::endl;

  // Test if species are read correctly
  EXPECT_EQ(struc_list[0].species[0], 1);
  EXPECT_EQ(struc_list[1].species[2], 0);
  EXPECT_EQ(struc_list[2].species[4], 1);
  std::cout << "species matches" << std::endl;

  // Test lattice
  Eigen::MatrixXd cell = Eigen::MatrixXd::Identity(3, 3);
  for (int s = 0; s < struc_list.size(); s++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        EXPECT_NEAR(struc_list[s].cell(i, j), cell(i, j), 1e-8);
      }
    }
  }
  std::cout << "cell matches" << std::endl;

  // Test if positions are read correctly
  EXPECT_NEAR(struc_list[0].positions(1, 0), 0.69338569, 1e-8);
  EXPECT_NEAR(struc_list[1].positions(2, 2), 0.01626457, 1e-8);
  EXPECT_NEAR(struc_list[2].positions(3, 1), 0.87700387, 1e-8);
  std::cout << "pos matches" << std::endl;

  // Test if energy are read correctly
  EXPECT_NEAR(struc_list[0].energy(0), 0.9133510291442271, 1e-8);
  EXPECT_NEAR(struc_list[1].energy(0), 0.5845339958967781, 1e-8);
  EXPECT_NEAR(struc_list[2].energy(0), 0.2679636480618098, 1e-8);
  std::cout << "energy matches" << std::endl;

  // Test if forces are read correctly
  EXPECT_NEAR(struc_list[0].forces(1 * 3 + 0), 0.92320467, 1e-8);
  EXPECT_NEAR(struc_list[1].forces(2 * 3 + 2), 0.02892780, 1e-8);
  EXPECT_NEAR(struc_list[2].forces(3 * 3 + 1), 0.56835792, 1e-8);
  std::cout << "forces matches" << std::endl;

  // Test if stress are read correctly
  // the struc stress are the xx, xy, xz, yy, yz, zz of xyz file
  EXPECT_NEAR(struc_list[0].stresses(2), -0.38721540888657746, 1e-8);
  EXPECT_NEAR(struc_list[1].stresses(3), -0.04982809557310608, 1e-8);
  EXPECT_NEAR(struc_list[2].stresses(5), -0.00887571986324354, 1e-8);
  std::cout << "stress matches" << std::endl;

  // Test if sparse indices are read correctly
  EXPECT_EQ(sparse_indices[0][0].size(), 0);
  EXPECT_EQ(sparse_indices[0][1].size(), 1);
  EXPECT_EQ(sparse_indices[0][2].size(), 3);
  EXPECT_EQ(sparse_indices[0][1][0], 2);
  EXPECT_EQ(sparse_indices[0][2][2], 4);
  std::cout << "sparse ind matches" << std::endl;
}
