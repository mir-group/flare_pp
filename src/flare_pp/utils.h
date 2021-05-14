#ifndef UTILS_H
#define UTILS_H

namespace utils {
  void grab(FILE *fptr, int n, double *list);
  std::tuple<std::vector<Structure>, std::vector<std::vector<int>>> utils::read_xyz(char *filename);
  template <typename Out>
  void split(const std::string &s, char delim, Out result);
  std::vector<std::string> split(const std::string &s, char delim);
}

#endif
