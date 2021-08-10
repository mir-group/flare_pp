#include "lammps_descriptor.h"
#include "radial.h"
#include "y_grad.h"
#include <cmath>
#include <iostream>

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
    const std::vector<double> &cutoff_hyps, Eigen::VectorXd &single_bond_vals,
    Eigen::MatrixXd &single_bond_env_dervs,
    const Eigen::MatrixXd &cutoff_matrix) {

  // Initialize basis vectors and spherical harmonics.
  std::vector<double> g = std::vector<double>(N, 0);
  std::vector<double> gx = std::vector<double>(N, 0);
  std::vector<double> gy = std::vector<double>(N, 0);
  std::vector<double> gz = std::vector<double>(N, 0);

  int n_harmonics = (lmax + 1) * (lmax + 1);
  std::vector<double> h = std::vector<double>(n_harmonics, 0);
  std::vector<double> hx = std::vector<double>(n_harmonics, 0);
  std::vector<double> hy = std::vector<double>(n_harmonics, 0);
  std::vector<double> hz = std::vector<double>(n_harmonics, 0);

  // Prepare LAMMPS variables.
  int central_species = type[i] - 1;
  double delx, dely, delz, rsq, r;
  double g_val, gx_val, gy_val, gz_val, h_val;
  std::complex<double> bond, bond_x, bond_y, bond_z;
  int j, s, descriptor_counter;

  // Initialize vectors.
  int n_radial = n_species * N;
  int n_bond = n_radial * n_harmonics;
  single_bond_vals = Eigen::VectorXd::Zero(n_bond);
  single_bond_env_dervs = Eigen::MatrixXd::Zero(n_inner * 3, n_bond);

  // Initialize radial hyperparameters.
  std::vector<double> new_radial_hyps = radial_hyps;

  // Loop over neighbors.
  int n_count = 0;
  for (int jj = 0; jj < jnum; jj++) {
    j = jlist[jj];

    delx = x[j][0] - xtmp;
    dely = x[j][1] - ytmp;
    delz = x[j][2] - ztmp;
    rsq = delx * delx + dely * dely + delz * delz;
    r = sqrt(rsq);

    // Retrieve the cutoff.
    int s = type[j] - 1;
    double cutoff = cutoff_matrix(central_species, s);
    double cutforcesq = cutoff * cutoff;

    if (rsq < cutforcesq) { // minus a small value to prevent numerial error
      // Reset endpoint of the radial basis set.
      new_radial_hyps[1] = cutoff;

      calculate_radial(g, gx, gy, gz, basis_function, cutoff_function, delx,
                       dely, delz, r, cutoff, N, new_radial_hyps, cutoff_hyps);
      get_complex_Y(h, hx, hy, hz, delx, dely, delz, lmax);

      // Store the products and their derivatives.
      descriptor_counter = s * N * n_harmonics;

      for (int radial_counter = 0; radial_counter < N; radial_counter++) {
        // Retrieve radial values.
        g_val = g[radial_counter];
        gx_val = gx[radial_counter];
        gy_val = gy[radial_counter];
        gz_val = gz[radial_counter];

        for (int angular_counter = 0; angular_counter < n_harmonics;
             angular_counter++) {

          h_val = h[angular_counter];
          bond = g_val * h_val;

          // Calculate derivatives with the product rule.
          bond_x = gx_val * h_val + g_val * hx[angular_counter];
          bond_y = gy_val * h_val + g_val * hy[angular_counter];
          bond_z = gz_val * h_val + g_val * hz[angular_counter];

          // Update single bond basis arrays.
          single_bond_vals(descriptor_counter) += bond;

          single_bond_env_dervs(n_count * 3, descriptor_counter) += bond_x;
          single_bond_env_dervs(n_count * 3 + 1, descriptor_counter) += bond_y;
          single_bond_env_dervs(n_count * 3 + 2, descriptor_counter) += bond_z;

          descriptor_counter++;
        }
      }
      n_count++;
    }
  }
}

void compute_Bk(Eigen::VectorXd &Bk_vals, Eigen::MatrixXd &Bk_force_dervs,
                double &norm_squared, Eigen::VectorXd &Bk_force_dots,
                const Eigen::VectorXcd &single_bond_vals,
                const Eigen::MatrixXcd &single_bond_force_dervs,
                const Eigen::VectorXi &descriptor_indices, 
                std::vector<std::vector<int>> nu, int nos, int K, int N,
                int lmax, const Eigen::VectorXd &coeffs) {

  int env_derv_size = single_bond_env_dervs.rows();
  int n_neighbors = env_derv_size / 3;

  // The value of last counter is the number of descriptors
  std::vector<int> last_index = nu[nu.size()-1];
  int n_d = last_index[last_index.size()-1] + 1; 

  // Initialize arrays.
  Bk_vals = Eigen::VectorXd::Zero(n_d);
  Bk_force_dervs = Eigen::MatrixXd::Zero(env_derv_size, n_d);
  Bk_force_dots = Eigen::VectorXd::Zero(env_derv_size);
  norm_squared = 0.0;

  for (int i = 0; i < nu.size(); i++) {
    std::vector<int> nu_list = nu[i];
    std::vector<int> single_bond_index = std::vector<int>(nu_list.end() - 2 - K, nu_list.end() - 2); // Get n1_l, n2_l, n3_l, etc.
    // Forward
    std::complex<double> A_fwd = 1;
    Eigen::VectorXcd dA = Eigen::VectorXcd::Ones(K);
    for (int t = 0; t < K - 1; t++) {
      A_fwd *= single_bond_vals(single_bond_index[t]);
      dA(t + 1) *= A_fwd;
    }
    // Backward
    std::complex<double> A_bwd = 1;
    for (int t = K - 1; t > 0; t--) {
      A_bwd *= single_bond_vals(single_bond_index[t]);
      dA(t - 1) *= A_bwd;
    }
    std::complex<double> A = A_fwd * single_bond_vals(single_bond_index[K - 1]);

    int counter = nu_list[nu_list.size() - 1];
    int m_index = nu_list[nu_list.size() - 2];
    Bk_vals(counter) += real(coeffs(m_index) * A); 

    // Store force derivatives.
    for (int n = 0; n < n_neighbors; n++) {
      for (int comp = 0; comp < 3; comp++) {
        int ind = n * 3 + comp;
        std::complex<double> dA_dr = 1;
        for (int t = 0; t < K; t++) {
          dA_dr += dA(t) * single_bond_force_dervs(ind, single_bond_index[t]);
        }
        Bk_force_dervs(ind, counter) +=
            real(coeffs(m_index) * dA_dr);
      }
    }
  }
  // Compute descriptor norm and force dot products.
  norm_squared = Bk_vals.dot(Bk_vals);
  Bk_force_dot = Bk_force_dervs * Bk_vals.transpose();
}
