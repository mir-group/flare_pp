#include "compute_flare_std_atom.h"
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

ComputeFlareStdAtom::ComputeFlareStdAtom(LAMMPS *lmp, int narg, char **arg) : 
  Compute(lmp, narg, arg),
  stds(nullptr)
{
  if (narg < 4) error->all(FLERR, "Illegal compute flare/std/atom command");

  peratom_flag = 1;
  size_peratom_cols = 0;
  timeflag = 1;
  comm_reverse = 1;

  // restartinfo = 0;
  // manybody_flag = 1;

  setflag = 0;
  cutsq = NULL;

  beta = NULL;
  coeff(narg, arg);

  nmax = 0;
  desc_derv = NULL;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

ComputeFlareStdAtom::~ComputeFlareStdAtom() {
  if (copymode)
    return;

//  if (allocated) {
//    memory->destroy(setflag);
//    memory->destroy(cutsq);
//  }

  memory->destroy(radial_code);
  memory->destroy(cutoff_code);
  memory->destroy(K);
  memory->destroy(n_max);
  memory->destroy(l_max);
  memory->destroy(beta_size);
  memory->destroy(cutoffs);

  memory->destroy(stds);
  memory->destroy(desc_derv);
}

/* ----------------------------------------------------------------------
   init specific to this compute command 
------------------------------------------------------------------------- */

void ComputeFlareStdAtom::init() {
  // Require newton on.
//  if (force->newton_pair == 0)
//    error->all(FLERR, "Compute command requires newton pair on");

  // Request a full neighbor list.
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;
}

void ComputeFlareStdAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}


/* ---------------------------------------------------------------------- */

void ComputeFlareStdAtom::compute_peratom() {
  if (atom->nmax > nmax) {
    memory->destroy(stds);
    nmax = atom->nmax;
    memory->create(stds,nmax,"flare/std/atom:stds");
    vector_atom = stds; // TODO: stds should be a matrix now
  }

  int i, j, ii, jj, inum, jnum, itype, jtype, n_inner, n_count;
  double delx, dely, delz, xtmp, ytmp, ztmp, rsq;
  int *ilist, *jlist, *numneigh, **firstneigh;

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nall = nlocal + atom->nghost;
  int newton_pair = force->newton_pair;
  int ntotal = nlocal;
  if (force->newton) ntotal += atom->nghost;

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  for (ii = 0; ii < ntotal; ii++) {
    stds[ii] = 0.0;
  }

  double empty_thresh = 1e-8;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
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
  
        delx = x[j][0] - xtmp;
        dely = x[j][1] - ytmp;
        delz = x[j][2] - ztmp;
        rsq = delx * delx + dely * dely + delz * delz;
        if (rsq < (cutoff * cutoff)) {
          n_inner++;
        }
  
      }

      double norm_squared, variance;
      Eigen::VectorXcd single_bond_vals, u;
      Eigen::VectorXd vals, env_dot, partial_forces, beta_p;
      Eigen::MatrixXcd single_bond_env_dervs;
      Eigen::MatrixXd env_dervs;

      // Compute covariant descriptors.
      // TODO: this function call is duplicated for multiple kernels
      complex_single_bond(x, type, jnum, n_inner, i, xtmp, ytmp, ztmp, jlist,
                          basis_function[kern], cutoff_function[kern], 
                          n_species, n_max[kern], l_max[kern], 
                          radial_hyps[kern], cutoff_hyps[kern], 
                          single_bond_vals, single_bond_env_dervs, cutoff_matrix);

      // Compute invariant descriptors.
      compute_Bk(vals, env_dervs, norm_squared, env_dot,
                 single_bond_vals, single_bond_env_dervs, nu[kern],
                 n_species, K[kern], n_max[kern], l_max[kern], coeffs[kern],
                 beta_matrices[kern][itype - 1], u, &variance);

      // Continue if the environment is empty.
      if (norm_squared < empty_thresh)
        continue;
  
      // Compute local energy and partial forces.
      // TODO: not needed if using "u"
      beta_p = beta_matrices[kern][itype - 1] * vals;
      stds[i] = pow(abs(vals.dot(beta_p)) / norm_squared, 0.5); // the numerator could be negative
    }
  }
}

/* ---------------------------------------------------------------------- */

int ComputeFlareStdAtom::pack_reverse_comm(int n, int first, double *buf)
{
    // TODO: add desc_derv to this
  int i,m,last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    for (int comp = 0; comp < 3; comp++) {
      buf[m++] = stds[i];
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeFlareStdAtom::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (int comp = 0; comp < 3; comp++) {
      stds[j] += buf[m++];
    }
  }

}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeFlareStdAtom::memory_usage()
{
  double bytes = nmax * 3 * (1 + n_descriptors) * sizeof(double);
  return bytes;
}



/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void ComputeFlareStdAtom::allocate() {
  allocated = 1;
//  int n = atom->ntypes;
//
//  memory->create(setflag, n + 1, n + 1, "compute:setflag");
//
//  // Set the diagonal of setflag to 1 (otherwise pair.cpp will throw an error)
//  for (int i = 1; i <= n; i++)
//    setflag[i][i] = 1;
//
//  // Create cutsq array (used in pair.cpp)
//  memory->create(cutsq, n + 1, n + 1, "compute:cutsq");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
   read DYNAMO funcfl file
------------------------------------------------------------------------- */

void ComputeFlareStdAtom::coeff(int narg, char **arg) {
  if (!allocated)
    allocate();

  // Should be exactly 3 arguments following "compute" in the input file.
  if (narg != 4)
    error->all(FLERR, "Incorrect args for compute coefficients");

  read_file(arg[3]);

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

//double ComputeFlareStdAtom::init_one(int i, int j) {
//  // init_one is called for each i, j pair in pair.cpp after calling init_style.
//  if (setflag[i][j] == 0)
//    error->all(FLERR, "All pair coeffs are not set");
//  return cutoff;
//}

/* ----------------------------------------------------------------------
   read potential values from a DYNAMO single element funcfl file
------------------------------------------------------------------------- */

void ComputeFlareStdAtom::read_file(char *filename) {
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

  memory->create(radial_code, num_kern, "compute:radial_code");
  memory->create(cutoff_code, num_kern, "compute:cutoff_code");
  memory->create(K, num_kern, "compute:K");
  memory->create(n_max, num_kern, "compute:n_max");
  memory->create(l_max, num_kern, "compute:l_max");
  memory->create(beta_size, num_kern, "compute:beta_size");

  for (int k = 0; k < num_kern; k++) {
    char desc_str[MAXLINE];
    fgets(line, MAXLINE, fptr);
    sscanf(line, "%s", desc_str); // Descriptor name

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

    // Parse the cutoffs.
    int n_cutoffs = n_species * n_species;
    memory->create(cutoffs, n_cutoffs, "compute:cutoffs");
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
      std::vector<double> rh = {0, cutoffs[0]}; // It does not matter what 
                                                // cutoff is used, will be 
                                                // modified to cutoff_matrix 
                                                // when computing descriptors 
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
    int n_size = n_species * n_descriptors;
    int beta_count = 0;
    double beta_val;
    for (int k = 0; k < n_species; k++) {
  //  for (int l = 0; l < n_species; l++) {
  
      beta_matrix = Eigen::MatrixXd::Zero(n_descriptors, n_descriptors);
      for (int i = 0; i < n_descriptors; i++) {
        for (int j = 0; j < n_descriptors; j++) {
          //beta_matrix(k * n_descriptors + i, l * n_descriptors + j) = beta[beta_count];
          beta_matrix(i, j) = beta[beta_count];
          beta_count++;
        }
      }
      beta_matrix_kern.push_back(beta_matrix);
  //  }
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

void ComputeFlareStdAtom::grab(FILE *fptr, int n, double *list) {
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
