#ifndef LAMMPS_DESCRIPTOR_H
#define LAMMPS_DESCRIPTOR_H

#include <Eigen/Dense>
#include <functional>
#include <vector>


void complex_single_bond(
    double **x, int *type, int jnum, int n_inner, int i, double xtmp,
    double ytmp, double ztmp, int *jlist,
    std::function<void(std::vector<double> &, std::vector<double> &, double,
                       int, std::vector<double>)>
        basis_function,
    std::function<void(std::vector<double> &, double, double,
                       std::vector<double>)>
        cutoff_function,
    int n_species, int N, int lmax,
    const std::vector<double> &radial_hyps,
    const std::vector<double> &cutoff_hyps, Eigen::VectorXcd &single_bond_vals,
    Eigen::MatrixXcd &single_bond_env_dervs,
    const Eigen::MatrixXd &cutoff_matrix); 

void compute_Bk(Eigen::VectorXd &Bk_vals, Eigen::MatrixXd &Bk_env_dervs,
                double &norm_squared, Eigen::VectorXd &Bk_env_dot,
                const Eigen::VectorXcd &single_bond_vals,
                const Eigen::MatrixXcd &single_bond_env_dervs, 
                std::vector<std::vector<int>> nu,
                int n_species, int K, int N, int lmax, 
                const Eigen::VectorXd &coeffs,
                const Eigen::MatrixXd &beta_matrix,
                Eigen::VectorXcd &u, double *evdwl);

#endif
