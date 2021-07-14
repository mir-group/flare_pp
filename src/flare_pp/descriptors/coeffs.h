#ifndef COEFFS
#define COEFFS
#include <Eigen/Dense>

Eigen::VectorXd compute_coeffs(int K, int lmax);

Eigen::VectorXd coeffs_K2(int lmax);

// Wigner 3j coefficients generated for l = 0, 1, 2, 3 using
// sympy.physics.wigner.wigner_3j
Eigen::VectorXd coeffs_K3(int lmax);

#endif
