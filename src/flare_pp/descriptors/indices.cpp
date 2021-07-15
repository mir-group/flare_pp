#include "indices.h"
#include <iostream>
#include <cmath>

std::vector<std::vector<int>> compute_indices(const std::vector<int> &descriptor_settings) {
  int nos = descriptor_settings[0];
  int K = descriptor_settings[1]; 
  int nmax = descriptor_settings[2];
  int lmax = descriptor_settings[3];

  int n_radial = nos * nmax;
  if (K == 2) { 
    return K2(n_radial, lmax);
  } else if (K == 3) {
    return K3(n_radial, lmax);
  } else {
    return K3(n_radial, lmax);
  }
}

std::vector<std::vector<int>> K2(int n_radial, int lmax) {
  int n1, n2, l, m, n1_l, n2_l;
  int n_harmonics = (lmax + 1) * (lmax + 1);
  std::vector<std::vector<int>> index_list;
  int counter = 0;
  for (int n1 = 0; n1 < n_radial; n1++) {
    for (int n2 = n1; n2 < n_radial; n2++) { // can be simplified?
      for (int l = 0; l < (lmax + 1); l++) {
        for (int m = 0; m < (2 * l + 1); m++) {
          n1_l = n1 * n_harmonics + (l * l + m);
          n2_l = n2 * n_harmonics + (l * l + (2 * l - m));
          int m_index = (m - l) % 2;
          index_list.push_back({n1, n2, l, m, n1_l, n2_l, m_index, counter});
        }
        counter++;
      }
    }
  }
  return index_list;
}

std::vector<std::vector<int>> K3(int n_radial, int lmax) {
  int n1, n2, n3, l1, l2, l3, m1, m2, m3, n1_l, n2_l, n3_l;
  int n_harmonics = (lmax + 1) * (lmax + 1);
  std::vector<std::vector<int>> index_list;
  int counter = 0;
  for (int n1 = 0; n1 < n_radial; n1++) {
    for (int n2 = n1; n2 < n_radial; n2++) {
      for (int n3 = n2; n3 < n_radial; n3++) {
        for (int l1 = 0; l1 < (lmax + 1); l1++) {
          int ind_1 = pow(lmax + 1, 4) * l1 * l1;
          for (int l2 = 0; l2 < (lmax + 1); l2++) {
            int ind_2 = ind_1 + pow(lmax + 1, 2) * l2 * l2 * (2 * l1 + 1);
            for (int l3 = 0; l3 < (lmax + 1); l3++) {
              if ((abs(l1 - l2) > l3) || (l3 > l1 + l2))
                continue;
              int ind_3 = ind_2 + l3 * l3 * (2 * l2 + 1) * (2 * l1 + 1);
              for (int m1 = 0; m1 < (2 * l1 + 1); m1++) {
                n1_l = n1 * n_harmonics + (l1 * l1 + m1);
                int ind_4 = ind_3 + m1 * (2 * l3 + 1) * (2 * l2 + 1);
                for (int m2 = 0; m2 < (2 * l2 + 1); m2++) {
                  n2_l = n2 * n_harmonics + (l2 * l2 + m2);
                  int ind_5 = ind_4 + m2 * (2 * l3 + 1);
                  for (int m3 = 0; m3 < (2 * l3 + 1); m3++) {
                    if (m1 + m2 + m3 - l1 - l2 - l3 != 0)
                      continue;
                    n3_l = n3 * n_harmonics + (l3 * l3 + m3);

                    int m_index = ind_5 + m3;
                    index_list.push_back({n1, n2, n3, l1, l2, l3, m1, m2, m3, n1_l, n2_l, n3_l, m_index, counter});
                  }
                }
              }
              counter++;
            }
          }
        }
      } 
    }
  }
  return index_list;
}
