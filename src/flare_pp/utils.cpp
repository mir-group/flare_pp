#include "utils.h"
#include "structure.h"
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

std::tuple<std::vector<Structure>, std::vector<std::vector<int>>> utils::read_xyz(char *filename) {

  std::ifstream file(filename);
  int n_atoms, atom_ind;
  Eigen::MatrixXd cell, positions;
  double energy;
  Eigen::VectorXd forces, stress;
  std::vector<int> species;

  std::vector<Structure> structure_list;
  std::vector<std::vector<int>> sparse_inds_list;
  std::vector<std::string> values;

  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      values = split(line, " ");
      if (values.size() == 1) {
        // the 1st line of a block, the number of atoms in a frame
        n_atoms = std::stoi(values[0]);
        cell = Eigen::MatrixXd::Zero(3, 3);
        positions = Eigen::MatrixXd::Zero(n_atoms, 3);
        forces = Eigen::VectorXd::Zero(n_atoms * 3);
        energy = 0;
        stress = Eigen::VectorXd::Zero(6);
        species = Eigen::VectorXd::Zeros(n_atoms);
        std::vector<int> sparse_inds;
        atom_ind = 0;
      } else if (values.size() >= 16) {
        // the 2nd line of a block, including cell (9), energy (1), stress (6), sparse indices
        cell(0, 0) = std::stod(values[0]);
        cell(0, 1) = std::stod(values[1]);
        cell(0, 2) = std::stod(values[2]);
        cell(1, 0) = std::stod(values[3]);
        cell(1, 1) = std::stod(values[4]);
        cell(1, 2) = std::stod(values[5]);
        cell(2, 0) = std::stod(values[6]);
        cell(2, 1) = std::stod(values[7]);
        cell(2, 2) = std::stod(values[8]);
        energy = std::stod(values[9]);
        stress(0) = std::stod(values[10]);
        stress(1) = std::stod(values[11]);
        stress(2) = std::stod(values[12]);
        stress(3) = std::stod(values[13]);
        stress(4) = std::stod(values[14]);
        stress(5) = std::stod(values[15]);
        for (int i = 16; i < values.size(); i++) {
          sparse_inds.push_back(std::stoi(values[i]));
        }
      } else if (values.size() == 7) {
        // the rest n_atoms lines of a block, with format "symbol x y z fx fy fz"
        species(atom_ind) = std::stoi(values[0]);
        positions(atom_ind, 0) = std::stod(values[1]);
        positions(atom_ind, 1) = std::stod(values[2]);
        positions(atom_ind, 2) = std::stod(values[3]);
        forces(3 * atom_ind + 0) = std::stod(values[4]);
        forces(3 * atom_ind + 1) = std::stod(values[5]);
        forces(3 * atom_ind + 2) = std::stod(values[6]);
        atom_ind += 1;
      } else {
        // raise error
        printf("Unknown line!!!");
      }

      Structure structure(cell, species, positions);
      structure.energy = energy;
      structure.forces = forces;
      structure.stresses = stress;
      structure_list.push_back(structure); 
      sparse_inds_list.push_back(sparse_inds);

    }
    file.close();
  }
  return std::make_tuple(structure_list, sparse_inds_list);
}
