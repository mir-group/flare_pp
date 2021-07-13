#ifndef INDICES
#define INDICES
#include <vector>

// Indices of (n1,l1,m1), ..., (nK,lK,mK) for B term from A_nlm

std::vector<std::vector<int>> compute_indices(const std::vector<int> &descriptor_settings);

std::vector<std::vector<int>> K1(int n_radial, int lmax);
std::vector<std::vector<int>> K2(int n_radial, int lmax);
std::vector<std::vector<int>> K3(int n_radial, int lmax);
#endif
