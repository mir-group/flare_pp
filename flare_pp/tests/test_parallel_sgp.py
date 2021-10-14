import os, sys, shutil, time
import numpy as np
import pytest

from ase.io import read, write
from ase.calculators import lammpsrun
from flare import struc
from flare.ase.atoms import FLARE_Atoms
from flare.lammps import lammps_calculator

from flare_pp.sparse_gp import SGP_Wrapper
from flare_pp.sparse_gp_calculator import SGP_Calculator
from flare_pp.parallel_sgp import ParSGP_Wrapper
from flare_pp._C_flare import NormalizedDotProduct, Bk, SparseGP, Structure
import flare_pp._C_flare as _C_flare
print(_C_flare.__file__)

import flare_pp

#from ctypes import CDLL, RTLD_GLOBAL
#CDLL("/n/sw/intel-cluster-studio-2019/mkl/lib/intel64/libmkl_rt.so", RTLD_GLOBAL)

np.random.seed(10)

# Make random structure.
n_atoms = 4
cell = np.eye(3)
train_positions = np.random.rand(n_atoms, 3)
# test_positions = np.random.rand(n_atoms, 3)
test_positions = train_positions
atom_types = [1, 2]
atom_masses = [2, 4]
species = np.random.randint(0, 2, 10) + 1
train_structure = struc.Structure(cell, species, train_positions)
test_structure = struc.Structure(cell, species, test_positions)

# Test update db
custom_range = [1, 2, 3]
energy = np.random.rand()
forces = np.random.rand(n_atoms, 3) * 10
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
kernel1 = NormalizedDotProduct(sigma, power)

sigma = 2.0
power = 2
kernel2 = NormalizedDotProduct(sigma, power)

sigma = 3.0
power = 2
kernel3 = NormalizedDotProduct(sigma, power)

kernel_list = [kernel1, kernel2] #, kernel3]

cutoff_function = "quadratic"
cutoff = 1.5
many_body_cutoffs = [cutoff]
radial_basis = "chebyshev"
radial_hyps = [0.0, cutoff]
cutoff_hyps = []

settings = [len(atom_types), 1, 4, 0]
calc1 = Bk(radial_basis, cutoff_function, radial_hyps, cutoff_hyps, settings)

settings = [len(atom_types), 2, 4, 3]
calc2 = Bk(radial_basis, cutoff_function, radial_hyps, cutoff_hyps, settings)

settings = [len(atom_types), 3, 2, 2]
calc3 = Bk(radial_basis, cutoff_function, radial_hyps, cutoff_hyps, settings)

calc_list = [calc1, calc2] #, calc3]

sigma_e = 0.1
sigma_f = 0.1
sigma_s = 0.1
species_map = {1: 0, 2: 1}
max_iterations = 20

sgp_py = ParSGP_Wrapper(
    kernel_list,
    calc_list,
    cutoff,
    sigma_e,
    sigma_f,
    sigma_s,
    species_map,
    max_iterations=max_iterations,
    variance_type="local",
)
sgp_calc = SGP_Calculator(sgp_py)

# Make random structure.
n_atoms = 10
cell = np.eye(3)

training_strucs = []
training_sparse_indices = [[] for i in range(len(kernel_list))]
for n in range(5): 
    train_positions = np.random.rand(n_atoms, 3)
    test_positions = train_positions
    species = np.random.randint(0, 2, n_atoms) + 1
    train_structure = struc.Structure(cell, species, train_positions)
    
    # Test update db
    energy = np.random.rand()
    forces = np.random.rand(n_atoms, 3) * 10
    stress = np.random.rand(6)
    train_structure.energy = energy
    train_structure.forces = forces
    train_structure.stress = stress
    
    training_strucs.append(train_structure)
    for k in range(len(kernel_list)):
        training_sparse_indices[k].append(np.random.randint(0, n_atoms, n_atoms // 2).tolist())


# TODO: calling `build` twice gets issue
sgp_py.build(training_strucs, training_sparse_indices)
#sgp_py.train()
from flare_pp.sparse_gp import compute_negative_likelihood_grad_stable
new_hyps = np.array(sgp_py.hyps) + 1

tic = time.time()
compute_negative_likelihood_grad_stable(new_hyps, sgp_py.sparse_gp, precomputed=False)
toc = time.time()
print("compute_negative_likelihood_grad_stable TIME:", toc - tic)

def test_update_db():
    """Check that the covariance matrices have the correct size after the
    sparse GP is updated."""

    training_strucs = [train_structure]
    training_sparse_indices = [[[0, 1, 2]]]

    sgp_py.build(training_strucs, training_sparse_indices)
#    sgp_py.update_db(
#        train_structure,
#        forces,
#        custom_range=[3, 3, 3],
#        energy=energy,
#        stress=stress,
#        mode="uncertain",
#    )

#    n_envs = len(custom_range)
#    assert sgp_py.sparse_gp.Kuu.shape[0] == sgp_py.sparse_gp.Kuf.shape[0]
    assert sgp_py.sparse_gp.Kuf.shape[1] == 1 + n_atoms * 3 + 6


#def test_train():
#    """Check that the hyperparameters and likelihood are updated when the
#    train method is called."""
#
#    hyps_init = tuple(sgp_py.hyps)
#    sgp_py.train()
#    hyps_post = tuple(sgp_py.hyps)
#
#    assert hyps_init != hyps_post
#    assert sgp_py.likelihood != 0.0
#
#
#def test_dict():
#    """
#    Check the method from_dict and as_dict
#    """
#
#    out_dict = sgp_py.as_dict()
#    assert len(sgp_py) == len(out_dict["training_structures"])
#    new_sgp, _ = SGP_Wrapper.from_dict(out_dict)
#    assert len(sgp_py) == len(new_sgp)
#    assert len(sgp_py.sparse_gp.kernels) == len(new_sgp.sparse_gp.kernels)
#    assert np.allclose(sgp_py.hyps, new_sgp.hyps)
#
#
#def test_coeff():
#    # Dump potential coefficient file
#    sgp_py.write_mapping_coefficients("lmp.flare", "A", [0, 1, 2])
#
#    # Dump uncertainty coefficient file
#    # here the new kernel needs to be returned, otherwise the kernel won't be found in the current module
#    new_kern = sgp_py.write_varmap_coefficients("beta_var.txt", "B", [0, 1, 2])
#
#    assert (
#        sgp_py.sparse_gp.sparse_indices[0] == sgp_py.sgp_var.sparse_indices[0]
#    ), "the sparse_gp and sgp_var don't have the same training data"
#
#    for s in range(len(atom_types)):
#        org_desc = sgp_py.sparse_gp.sparse_descriptors[0].descriptors[s]
#        new_desc = sgp_py.sgp_var.sparse_descriptors[0].descriptors[s]
#        if not np.allclose(org_desc, new_desc):  # the atomic order might change
#            assert np.allclose(org_desc.shape, new_desc.shape)
#            for i in range(org_desc.shape[0]):
#                flag = False
#                for j in range(
#                    new_desc.shape[0]
#                ):  # seek in new_desc for matching of org_desc
#                    if np.allclose(org_desc[i], new_desc[j]):
#                        flag = True
#                        break
#                assert flag, "the sparse_gp and sgp_var don't have the same descriptors"
#
#
#@pytest.mark.skipif(
#    not os.environ.get("lmp", False),
#    reason=(
#        "lmp not found "
#        "in environment: Please install LAMMPS "
#        "and set the $lmp env. "
#        "variable to point to the executatble."
#    ),
#)
#def test_lammps():
#    # create ASE calc
#    lmp_command = os.environ.get("lmp")
#    specorder = ["H", "He"]
#    pot_file = "lmp.flare"
#    params = {
#        "command": lmp_command,
#        "pair_style": "flare",
#        "pair_coeff": [f"* * {pot_file}"],
#    }
#    files = [pot_file]
#    lmp_calc = lammpsrun.LAMMPS(
#        label=f"tmp",
#        keep_tmp_files=True,
#        tmp_dir="./tmp/",
#        parameters=params,
#        files=files,
#        specorder=specorder,
#    )
#
#    test_atoms = test_structure.to_ase_atoms()
#    test_atoms.calc = lmp_calc
#    lmp_f = test_atoms.get_forces()
#    lmp_e = test_atoms.get_potential_energy()
#    lmp_s = test_atoms.get_stress()
#
#    print("GP predicting")
#    test_atoms.calc = None
#    test_atoms = FLARE_Atoms.from_ase_atoms(test_atoms)
#    test_atoms.calc = sgp_calc
#    sgp_f = test_atoms.get_forces()
#    sgp_e = test_atoms.get_potential_energy()
#    sgp_s = test_atoms.get_stress()
#
#    print("Energy")
#    print(lmp_e, sgp_e)
#    assert np.allclose(lmp_e, sgp_e)
#
#    print("Forces")
#    print(np.concatenate([lmp_f, sgp_f], axis=1))
#    assert np.allclose(lmp_f, sgp_f)
#
#    print("Stress")
#    print(lmp_s)
#    print(sgp_s)
#    assert np.allclose(lmp_s, sgp_s)
#
#
#@pytest.mark.skipif(
#    not os.environ.get("lmp", False),
#    reason=(
#        "lmp not found "
#        "in environment: Please install LAMMPS "
#        "and set the $lmp env. "
#        "variable to point to the executatble."
#    ),
#)
#def test_lammps_uncertainty():
#    # create ASE calc
#    lmp_command = os.environ.get("lmp")
#    specorder = ["H", "He"]
#    pot_file = "lmp.flare"
#    params = {
#        "command": lmp_command,
#        "pair_style": "flare",
#        "pair_coeff": [f"* * {pot_file}"],
#    }
#    files = [pot_file]
#    lmp_calc = lammpsrun.LAMMPS(
#        label=f"tmp",
#        keep_tmp_files=True,
#        tmp_dir="./tmp/",
#        parameters=params,
#        files=files,
#        specorder=specorder,
#    )
#
#    test_atoms = test_structure.to_ase_atoms()
#
#    # compute uncertainty 
#    in_lmp = """
#atom_style atomic 
#units metal
#boundary p p p 
#atom_modify sort 0 0.0 
#
#read_data data.lammps 
#
#### interactions
#pair_style flare 
#pair_coeff * * lmp.flare 
#mass 1 1.008000 
#mass 2 4.002602 
#
#### run
#fix fix_nve all nve
#compute unc all flare/std/atom beta_var.txt
#dump dump_all all custom 1 traj.lammps id type x y z vx vy vz fx fy fz c_unc[1] c_unc[2] c_unc[3] 
#thermo_style custom step temp press cpu pxx pyy pzz pxy pxz pyz ke pe etotal vol lx ly lz atoms
#thermo_modify flush yes format float %23.16g
#thermo 1
#run 0
#"""
#    os.chdir("tmp")
#    write("data.lammps", test_atoms, format="lammps-data")
#    with open("in.lammps", "w") as f:
#        f.write(in_lmp)
#    shutil.copyfile("../beta_var.txt", "./beta_var.txt")
#    os.system(f"{lmp_command} < in.lammps > log.lammps")
#    unc_atoms = read("traj.lammps", format="lammps-dump-text")
#    lmp_stds = [unc_atoms.get_array(f"c_unc[{i+1}]") / sgp_py.hyps[i] for i in range(len(calc_list))]
#    lmp_stds = np.squeeze(lmp_stds).T
#
#    # Test mapped variance (need to use sgp_var)
#    test_atoms.calc = None
#    test_atoms = FLARE_Atoms.from_ase_atoms(test_atoms)
#    test_atoms.calc = sgp_calc
#    test_atoms.calc.gp_model.sparse_gp = sgp_py.sgp_var
#    test_atoms.calc.reset()
#    sgp_stds = test_atoms.calc.get_uncertainties(test_atoms)
#    print(sgp_stds)
#    print(lmp_stds)
#    assert np.allclose(sgp_stds, lmp_stds)
