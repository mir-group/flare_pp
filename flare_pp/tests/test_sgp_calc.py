import os
import numpy as np
import pytest
import sys

from ase.io import read
from flare import struc
from flare.lammps import lammps_calculator

from flare_pp.sparse_gp import SGP_Wrapper
from flare_pp.sparse_gp_calculator import SGP_Calculator
from flare_pp._C_flare import NormalizedDotProduct, B2, SparseGP, Structure


np.random.seed(10)

# Make random structure.
n_atoms = 4
cell = np.eye(3)
train_positions = np.random.rand(n_atoms, 3)
test_positions = np.random.rand(n_atoms, 3)
atom_types = [1, 2]
atom_masses = [2, 4]
species = [1, 2, 1, 2]
train_structure = struc.Structure(cell, species, train_positions)
test_structure = struc.Structure(cell, species, test_positions)

# Test update db
custom_range = [1, 2, 3]
energy = np.random.rand()
forces = np.random.rand(n_atoms, 3)
stress = np.random.rand(6)
np.savez(
    "random_data",
    train_pos=train_positions,
    test_pos=test_positions,
    energy=energy,
    forces=forces,
    stress=stress,
)

# Create sparse GP model.
sigma = 1.0
power = 2
kernel = NormalizedDotProduct(sigma, power)
cutoff_function = "quadratic"
cutoff = 1.5
many_body_cutoffs = [cutoff]
radial_basis = "chebyshev"
radial_hyps = [0.0, cutoff]
cutoff_hyps = []
settings = [len(atom_types), 4, 3] 
calc = B2(radial_basis, cutoff_function, radial_hyps, cutoff_hyps, settings)
sigma_e = 1.0
sigma_f = 1.0
sigma_s = 1.0
species_map = {1: 0, 2: 1}
max_iterations = 20

sgp_py = SGP_Wrapper(
    [kernel],
    [calc],
    cutoff,
    sigma_e,
    sigma_f,
    sigma_s,
    species_map,
    max_iterations=max_iterations,
)
sgp_calc = SGP_Calculator(sgp_py)

def test_update_db():
    """Check that the covariance matrices have the correct size after the
    sparse GP is updated."""

    sgp_calc.sgp_model.update_db(
        train_structure, forces, [3], energy, stress, mode="uncertain"
    )

    n_envs = len(custom_range)
    assert sgp_calc.sgp_model.sparse_gp.Kuu.shape[0] == n_envs
    assert sgp_calc.sgp_model.sparse_gp.Kuf.shape[1] == 1 + n_atoms * 3 + 6


def test_io():
    sgp_calc.write_model("sgp.json")
    new_calc = SGP_Calculator.from_file("sgp.json")

    assert len(sgp_calc) == len(new_calc)
    assert np.allclose(sgp_calc.sgp_model.sparse_gp.alpha, new_calc.sgp_model.sparse_gp.alpha)
