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

  memory->destroy(beta);

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

    // Count the atoms inside the cutoff.
    n_inner = 0;
    for (int jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;
      rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < (cutoff * cutoff))
        n_inner++;
    }

    for (int kern = 0; kern < num_kern; kern++) {
      double norm_squared, val_1, val_2;
      Eigen::VectorXd single_bond_vals, vals, env_dot, beta_p, partial_forces;
      Eigen::MatrixXd single_bond_env_dervs, env_dervs

      // Compute covariant descriptors.
      single_bond(x, type, jnum, n_inner, i, xtmp, ytmp, ztmp, jlist,
                  basis_function[kern], cutoff_function[kern], cutoffs[kern], n_species, n_max[kern],
                  l_max[kern], radial_hyps[kern], cutoff_hyps[kern], single_bond_vals,
                  single_bond_env_dervs);
  
      // Compute invariant descriptors.
      B2_descriptor(vals, env_dervs, norm_squared, env_dot,
                    single_bond_vals, single_bond_env_dervs, n_species, n_max[kern],
                    l_max[kern]);
  
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
    }

    // Compute local energy.
    if (eflag)
      ev_tally_full(i, 2.0 * evdwl, 0.0, 0.0, 0.0, 0.0, 0.0);
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
  char line[MAXLINE]
  FILE *fptr;

  // Check that the potential file can be opened.
  if (me == 0) {
    fptr = utils::open_potential(filename,lmp,nullptr);
    if (fptr == NULL) {
      char str[128];
      snprintf(str, 128, "Cannot open potential file %s", filename);
      error->one(FLERR, str);
    }
  }

  int tmp, nwords;
  int radial_str_total = 0;
  int cutoff_str_total = 0;
  if (me == 0) {
    fgets(line, MAXLINE, fptr);
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%i", &num_kern); // number of descriptors/kernels

    std::vector<char> radial_string[num_kern][MAXLINE], cutoff_string[num_kern][MAXLINE]; // need to check here
    //char radial_string[MAXLINE], cutoff_string[MAXLINE];
    std::vector<int> radial_string_length[num_kern], cutoff_string_length[num_kern];

    for (k = 0; k < num_kern; k++) {
      fgets(line, MAXLINE, fptr);
      sscanf(line, "%s", radial_string[k]); // Radial basis set
      radial_string_length[k] = strlen(radial_string[k]);
      radial_str_total += radial_string_length[k] + 1;
      
      fgets(line, MAXLINE, fptr);
      sscanf(line, "%i %i %i %i", &n_species, &n_max[k], &l_max[k], &beta_size[k]);
      
      fgets(line, MAXLINE, fptr);
      sscanf(line, "%s", cutoff_string[k]); // Cutoff function
      cutoff_string_length[k] = strlen(cutoff_string[k]);
      cutoff_str_total += cutoff_string_length[k] + 1;
      
      fgets(line, MAXLINE, fptr);
      sscanf(line, "%lg", &cutoffs[k]); // Cutoffs
      cutoff = max_element(std::begin(cutoffs), std::end(cutoffs)); // Use max cut as cutoff
    }
  }

  MPI_Bcasn(&num_kern, 1, MPI_INT, 0, world);
  MPI_Bcast(&n_species, 1, MPI_INT, 0, world);
  MPI_Bcast(&n_max, num_kern, MPI_INT, 0, world);
  MPI_Bcast(&l_max, num_kern, MPI_INT, 0, world);
  MPI_Bcast(&beta_size, num_kern, MPI_INT, 0, world);
  MPI_Bcast(&cutoff, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&cutoffs, num_kern, MPI_DOUBLE, 0, world);
  MPI_Bcast(&radial_string_length, num_kern, MPI_INT, 0, world);
  MPI_Bcast(&cutoff_string_length, num_kern, MPI_INT, 0, world);
  MPI_Bcast(radial_string, radial_str_total, MPI_CHAR, 0, world);
  MPI_Bcast(cutoff_string, cutoff_str_total, MPI_CHAR, 0, world);


  for (k = 0; k < num_kern; k++) {
    // Set number of descriptors.
    int n_radial = n_max[k] * n_species;
    n_descriptors = (n_radial * (n_radial + 1) / 2) * (l_max[k] + 1);
  
    // Check the relationship between the power spectrum and beta.
    int beta_check = n_descriptors * (n_descriptors + 1) / 2;
    if (beta_check != beta_size[k])
      error->all(FLERR, "Beta size doesn't match the number of descriptors.");
  
    // Set the radial basis.
    if (!strcmp(radial_string[k], "chebyshev")) {
      basis_function[k] = chebyshev;
      radial_hyps[k] = std::vector<double>{0, cutoffs[k]};
    } else {
      char str[128]; 
      snprintf(str, 128, "Radial function %s is not supported\n.", radial_string[k]);
      error->all(FLERR, str);
    }
  
    // Set the cutoff function.
    if (!strcmp(cutoff_string[k], "quadratic")) {
      cutoff_function[k] = quadratic_cutoff;
    } else if (!strcmp(cutoff_string[k], "cosine")) {
      cutoff_function[k] = cos_cutoff;
    } else {
      char str[128]; 
      snprintf(str, 128, "Cutoff function %s is not supported\n.", cutoff_string[k]);
      error->all(FLERR, str);
    }
 
    // Parse the beta vectors.
    memory->create(beta, beta_size * n_species, "pair:beta");
    if (me == 0)
      grab(fptr, beta_size * n_species, beta);
    MPI_Bcast(beta, beta_size * n_species, MPI_DOUBLE, 0, world);
  
    // Fill in the beta matrix.
    // TODO: Remove factor of 2 from beta.
    Eigen::MatrixXd beta_matrix;
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
      beta_matrices.push_back(beta_matrix);
    }
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
