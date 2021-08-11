#include "pair_flare.h"
#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// flare++ modules
#include "cutoffs.h"
#include "lammps_descriptor.h"
#include "radial.h"
#include "y_grad.h"
#include "indices.h"
#include "coeffs.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

PairFLARE::PairFLARE(LAMMPS *lmp) : Pair(lmp) {
  restartinfo = 0;
  manybody_flag = 1;

  beta = NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairFLARE::~PairFLARE() {
  if (copymode)
    return;


  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairFLARE::compute(int eflag, int vflag) {
  int i, j, ii, jj, inum, jnum, itype, jtype, n_inner, n_count;
  double evdwl, delx, dely, delz, xtmp, ytmp, ztmp, rsq;
  double *coeff;
  int *ilist, *jlist, *numneigh, **firstneigh;

  evdwl = 0.0;
  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  double empty_thresh = 1e-8;

  for (ii = 0; ii < inum; ii++) {
    i = list->ilist[ii];
    itype = type[i];
    jnum = numneigh[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    jlist = firstneigh[i];

    for (int kern = 0; kern < num_kern; kern++) {
      cutoff_matrix = cutoff_matrices[kern];

      // Count the atoms inside the cutoff. 
      // TODO: this might be duplicated when multiple kernels share the same cutoff
      n_inner = 0;
      for (int jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        int s = type[j] - 1;
        double cutoff_val = cutoff_matrix(itype-1, s);
        delx = x[j][0] - xtmp;
        dely = x[j][1] - ytmp;
        delz = x[j][2] - ztmp;
        rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < (cutoff_val * cutoff_val))
          n_inner++;
      }

      double norm_squared, val_1, val_2;
      Eigen::VectorXd single_bond_vals, vals, env_dot, beta_p, partial_forces;
      Eigen::MatrixXd single_bond_env_dervs, env_dervs;

      // Compute covariant descriptors.
      complex_single_bond(x, type, jnum, n_inner, i, xtmp, ytmp, ztmp, jlist,
                          basis_function[kern], cutoff_function[kern], cutoffs[kern], 
                          n_species, n_max[kern], l_max[kern], 
                          radial_hyps[kern], cutoff_hyps[kern], 
                          single_bond_vals, single_bond_env_dervs, cutoff_matrix);

      // Compute invariant descriptors.
      compute_Bk(vals, env_dervs, norm_squared, env_dot,
                 single_bond_vals, single_bond_env_dervs, nu[kern],
                 n_species, K[kern], n_max[kern], lmax[kern], coeffs[kern]);

      // Continue if the environment is empty.
      if (norm_squared < empty_thresh)
        continue;
  
      // Compute local energy and partial forces.
      beta_p = beta_matrices[kern][itype - 1] * vals;
      evdwl = vals.dot(beta_p) / norm_squared;
 
      partial_forces =
          2 * (- env_dervs * beta_p + evdwl * env_dot) / norm_squared;

      // Update energy, force and stress arrays.
      n_count = 0;
      for (int jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx * delx + dely * dely + delz * delz;
  
        if (rsq < (cutoff * cutoff)) {
          double fx = -partial_forces(n_count * 3);
          double fy = -partial_forces(n_count * 3 + 1);
          double fz = -partial_forces(n_count * 3 + 2);
          f[i][0] += fx;
          f[i][1] += fy;
          f[i][2] += fz;
          f[j][0] -= fx;
          f[j][1] -= fy;
          f[j][2] -= fz;
  
          if (vflag) {
            ev_tally_xyz(i, j, nlocal, newton_pair, 0.0, 0.0, fx, fy, fz, delx,
                         dely, delz);
          }
          n_count++;
        }
      }
      // Compute local energy.
      if (eflag)
        ev_tally_full(i, 2.0 * evdwl, 0.0, 0.0, 0.0, 0.0, 0.0);

    }

  }

  if (vflag_fdotr)
    virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairFLARE::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");

  // Set the diagonal of setflag to 1 (otherwise pair.cpp will throw an error)
  for (int i = 1; i <= n; i++)
    setflag[i][i] = 1;

  // Create cutsq array (used in pair.cpp)
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairFLARE::settings(int narg, char ** /*arg*/) {
  // "flare" should be the only word after "pair_style" in the input file.
  if (narg > 0)
    error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   read DYNAMO funcfl file
------------------------------------------------------------------------- */

void PairFLARE::coeff(int narg, char **arg) {
  if (!allocated)
    allocate();

  // Should be exactly 3 arguments following "pair_coeff" in the input file.
  if (narg != 3)
    error->all(FLERR, "Incorrect args for pair coefficients");

  // Ensure I,J args are "* *".
  if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
    error->all(FLERR, "Incorrect args for pair coefficients");

  read_file(arg[2]);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairFLARE::init_style() {
  // Require newton on.
  if (force->newton_pair == 0)
    error->all(FLERR, "Pair style requires newton pair on");

  // Request a full neighbor list.
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairFLARE::init_one(int i, int j) {
  // init_one is called for each i, j pair in pair.cpp after calling init_style.

  return cutoff;
}

/* ----------------------------------------------------------------------
   read potential values from a DYNAMO single element funcfl file
------------------------------------------------------------------------- */

void PairFLARE::read_file(char *filename) {
  int me = comm->me;
  char line[MAXLINE];
  FILE *fptr;

  // Check that the potential file can be opened.
  fptr = utils::open_potential(filename,lmp,nullptr);
  if (fptr == NULL) {
    char str[128];
    snprintf(str, 128, "Cannot open potential file %s", filename);
    error->one(FLERR, str);
  }

  int tmp, nwords;
  fgets(line, MAXLINE, fptr);
  fgets(line, MAXLINE, fptr);
  sscanf(line, "%i", &num_kern); // number of descriptors/kernels

  memory->create(radial_code, num_kern, "pair:radial_code");
  memory->create(cutoff_code, num_kern, "pair:cutoff_code");
  memory->create(K, num_kern, "pair:K");
  memory->create(n_max, num_kern, "pair:n_max");
  memory->create(l_max, num_kern, "pair:l_max");
  memory->create(beta_size, num_kern, "pair:beta_size");
  memory->create(cutoffs, num_kern, "pair:cutoffs");

  //MPI_Bcast(&num_kern, 1, MPI_INT, 0, world);
  for (int k = 0; k < num_kern; k++) {
    char desc_str[MAXLINE];
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%s", desc_str); // Descriptor name
//    if (!strcmp(desc_str, "B1")) {
//      descriptor_code[k] = 1;
//    } else if (!strcmp(desc_str, "B2")) {
//      descriptor_code[k] = 2;
//    } else {
//      char str[128]; 
//      snprintf(str, 128, "Descriptor %s is not supported\n.", desc_str);
//      error->all(FLERR, str);
//    }


    char radial_str[MAXLINE], cutoff_str[MAXLINE];
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%s", radial_str); // Radial basis set
    if (!strcmp(radial_str, "chebyshev")) {
      radial_code[k] = 1;
    } else {
      char str[128]; 
      snprintf(str, 128, "Radial function %s is not supported\n.", radial_str);
      error->all(FLERR, str);
    }
    
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%i %i %i %i %i", &n_species, &K[k], &n_max[k], &l_max[k], &beta_size[k]);
    
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%s", cutoff_str); // Cutoff function
    if (!strcmp(cutoff_str, "cosine")) {
      cutoff_code[k] = 1;
    } else if (!strcmp(cutoff_str, "quadratic")) {
      cutoff_code[k] = 2;
    } else {
      char str[128]; 
      snprintf(str, 128, "Cutoff function %s is not supported\n.", cutoff_str);
      error->all(FLERR, str);
    }

//    // Parse the cutoffs
//    fgets(line, MAXLINE, fptr);
//    sscanf(line, "%lg", &cutoffs[k]); // Cutoffs
//    cutoff = 0;
//    for (int i = 0; i < sizeof(cutoffs) / sizeof(cutoffs[0]); i++) { // Use max cut as cutoff
//      if (cutoffs[i] > cutoff) cutoff = cutoffs[i];
//    }

    // Parse the cutoffs.
    int n_cutoffs = n_species * n_species;
    memory->create(cutoffs, n_cutoffs, "pair:cutoffs");
    if (me == 0)
      grab(fptr, n_cutoffs, cutoffs);
    MPI_Bcast(cutoffs, n_cutoffs, MPI_DOUBLE, 0, world);
  
    // Fill in the cutoff matrix.
    cutoff = -1;
    cutoff_matrix = Eigen::MatrixXd::Zero(n_species, n_species);
    int cutoff_count = 0;
    for (int i = 0; i < n_species; i++){
      for (int j = 0; j < n_species; j++){
        double cutoff_val = cutoffs[cutoff_count];
        cutoff_matrix(i, j) = cutoff_val;
        if (cutoff_val > cutoff) cutoff = cutoff_val;
        cutoff_count ++;
      }
    }
    cutoff_matrices.push_back(cutoff_matrix);

    //MPI_Bcast(&n_species, 1, MPI_INT, 0, world);
    //MPI_Bcast(&cutoff, 1, MPI_DOUBLE, 0, world);
    //MPI_Bcast(n_max, num_kern, MPI_INT, 0, world);
    //MPI_Bcast(l_max, num_kern, MPI_INT, 0, world);
    //MPI_Bcast(beta_size, num_kern, MPI_INT, 0, world);
    //MPI_Bcast(cutoffs, num_kern, MPI_DOUBLE, 0, world);
    //MPI_Bcast(radial_code, num_kern, MPI_INT, 0, world);
    //MPI_Bcast(cutoff_code, num_kern, MPI_INT, 0, world);

    // Compute indices and coefficients
    std::vector<int> descriptor_settings = {n_species, K[k], n_max[k], l_max[k]};
    std::vector<std::vector<int>> nu_kern = compute_indices(descriptor_settings);
    Eigen::VectorXd coeffs_kern = compute_coeffs(K[k], l_max[k]);
    nu.push_back(nu_kern);
    coeffs.push_back(coeffs_kern);

    // Set number of descriptors.
    std::vector<int> last_index = nu_kern[nu_kern.size()-1];
    int n_descriptors = last_index[last_index.size()-1] + 1; 

    // Check the relationship between the power spectrum and beta.
    int beta_check = n_descriptors * (n_descriptors + 1) / 2;
    if (beta_check != beta_size[k])
      error->all(FLERR, "Beta size doesn't match the number of descriptors.");

    // Set the radial basis.
    if (radial_code[k] == 1){ 
      basis_function.push_back(chebyshev);
      std::vector<double> rh = {0, cutoffs[k]};
      radial_hyps.push_back(rh);
      std::vector<double> ch;
      cutoff_hyps.push_back(ch);
    }
  
    // Set the cutoff function.
    if (cutoff_code[k] == 1) 
      cutoff_function.push_back(cos_cutoff);
    else if (cutoff_code[k] == 2) 
      cutoff_function.push_back(quadratic_cutoff);

    // Parse the beta vectors.
    // TODO: check this memory creation
    memory->create(beta, beta_size[k] * n_species, "pair:beta");
    grab(fptr, beta_size[k] * n_species, beta);
    //MPI_Bcast(beta1, beta_size[k] * n_species, MPI_DOUBLE, 0, world);
    
    // Fill in the beta matrix.
    // TODO: Remove factor of 2 from beta.
    Eigen::MatrixXd beta_matrix;
    std::vector<Eigen::MatrixXd> beta_matrix_kern;
    int beta_count = 0;
    double beta_val;
    for (int s = 0; s < n_species; s++) {
      beta_matrix = Eigen::MatrixXd::Zero(n_descriptors, n_descriptors);
      for (int i = 0; i < n_descriptors; i++) {
        for (int j = i; j < n_descriptors; j++) {
          if (i == j)
            beta_matrix(i, j) = beta[beta_count];
          else if (i != j) {
            beta_val = beta[beta_count] / 2;
            beta_matrix(i, j) = beta_val;
            beta_matrix(j, i) = beta_val;
          }
          beta_count++;
        }
      }
      beta_matrix_kern.push_back(beta_matrix);
    }
    beta_matrices.push_back(beta_matrix_kern);

    // TODO: check this memory destroy
    memory->destroy(beta);

  }
}

/* ----------------------------------------------------------------------
   grab n values from file fp and put them in list
   values can be several to a line
   only called by proc 0
------------------------------------------------------------------------- */

void PairFLARE::grab(FILE *fptr, int n, double *list) {
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
