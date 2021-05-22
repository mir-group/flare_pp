#include "structure.h"
#include <tuple>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>

#ifndef UTILS_H
#define UTILS_H

namespace utils {
//  void grab(FILE *fptr, int n, double *list);
  std::tuple<std::vector<Structure>, std::vector<std::vector<int>>> read_xyz(std::string filename);
  template <typename Out>
  void split(const std::string &s, char delim, Out result);
  std::vector<std::string> split(const std::string &s, char delim);
}

#endif
