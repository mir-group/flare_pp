#include "utils.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iterator>

#define MAXLINE 1024

void utils::grab(FILE *fptr, int n, double *list) {
  char *ptr;
  char line[MAXLINE];

  int i = 0;
  while (i < n) {
    fgets(line, MAXLINE, fptr);
    ptr = strtok(line, " \t\n\r\f");
    list[i++] = atof(ptr);
    while ((ptr = strtok(NULL, " \t\n\r\f")))
      list[i++] = atof(ptr);
  }
}

//void utils::read_xyz(char *filename) {
//  char line[MAXLINE];
//  FILE *fptr;
//  fptr = fopen(filename.c_str(), "r");
//}
//

template <typename Out>
void split(const std::string &s, char delim, Out result) {
  std::istringstream iss(s);
  std::string item;
  while (std::getline(iss, item, delim)) {
    *result++ = item;
  }
}

std::vector<std::string> split(const std::string &s, char delim) {
  /* Convert a line of string into a list
   * Similar to the python method str.split()
   */
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

int utils::read_xyz(char *filename) {
  std::ifstream file(filename);
  int n_atoms;
  Eigen::MatrixXd cell, positions;
  Eigen::VectorXd labels;
  std::vector<int> species;

  std::vector<std::string> values;
  std::vector<Eigen::MatrixXd> cell_list, position_list;
  std::vector<Eigen::VectorXd> label_list;
  std::vector<std::vector<int>> species_list;
  std::vector<std::vector<int>> sparse_ind_list;

  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      values = split(line, " ");
      if (values.size() == 1) {
        // the 1st line of a block, the number of atoms in a frame
        n_atoms = std::stoi(values[0]);
        cell = Eigen::MatrixXd::Zero(3, 3);
        positions = Eigen::MatrixXd::Zero(n_atoms, 3);
        labels = Eigen::VectorXd::Zero(1 + 3 * n_atoms + 6);
        species = Eigen::VectorXd::Zeros(n_atoms);
        std::vector<int> sparse_inds;
      } else if (values.size() >= 16) {
        // the 2nd line of a block, including cell, energy, stress, sparse indices
      } else if (values.size() == 7) {
        // the rest n_atoms lines of a block, with format "symbol x y z fx fy fz"
      } else {
        // raise error
      }
      printf("%s", line.c_str());

    }
    file.close();
  }
  return 0;
}
