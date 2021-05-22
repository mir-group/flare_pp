#include "utils.h"
#include "structure.h"
#include "gtest/gtest.h"
#include <Eigen/Dense>
#include <cmath>
#include <iostream>

class UtilsTest : public ::testing::Test {
protected:
  std::tuple<std::vector<Structure>> struc_list;
  std::vector<std::vector<int>> sparse_indices;
  std::string filename = std::string("dft_data.xyz");
};

TEST_F(UtilsTest, XYZTest) {
  std::tie(struc_list, sparse_indices) = utils::read_xyz(filename);
}
