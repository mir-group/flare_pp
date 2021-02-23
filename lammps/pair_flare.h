// Jonathan Vandermause
// Pair style based on pair_eam.h

#ifdef PAIR_CLASS

PairStyle(flare, PairFLARE)

#else

#ifndef LMP_PAIR_FLARE_H
#define LMP_PAIR_FLARE_H

#include "pair.h"
#include <Eigen/Dense>
#include <cstdio>
#include <vector>

namespace LAMMPS_NS {

class PairFLARE : public Pair {
public:
  PairFLARE(class LAMMPS *);
  virtual ~PairFLARE();
  virtual void compute(int, int);
  void settings(int, char **);
  virtual void coeff(int, char **);
  void init_style();
  double init_one(int, int);

protected:
  int num_kern, n_species, n_descriptors;
  int *n_max, *l_max, *beta_size;

  int *radial_code, *cutoff_code;
  std::vector<std::function<void(std::vector<double> &, std::vector<double> &, double, int,
                     std::vector<double>)>>
      basis_function;
  std::vector<std::function<void(std::vector<double> &, double, double,
                     std::vector<double>)>>
      cutoff_function;
  int *descriptor_code;

  double **radial_hyps, cutoff_hyps;

  double *cutoffs;
  double cutoff;
  double *beta;
  Eigen::MatrixXd beta_matrix;
  std::vector<std::vector<Eigen::MatrixXd>> beta_matrices;

  virtual void allocate();
  virtual void read_file(char *);
  void grab(FILE *, int, double *);
};

} // namespace LAMMPS_NS

#endif
#endif
