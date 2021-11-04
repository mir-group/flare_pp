import pytest
import numpy as np
import os
from copy import deepcopy
import json

from flare_pp._C_flare import SparseGP, NormalizedDotProduct, B2, Structure
from flare_pp.sparse_gp import SGP_Wrapper
from flare_pp.sparse_gp_calculator import SGP_Calculator

from flare.ase.atoms import FLARE_Atoms

from ase.io import read, write
from ase.calculators import lammpsrun
from ase.calculators.lj import LennardJones
from ase.build import bulk

from .get_sgp import get_random_atoms, get_empty_sgp, \
    get_updated_sgp, get_sgp_calc


def test_update_db():
    """Check that the covariance matrices have the correct size after the
    sparse GP is updated."""

    # Create a labelled structure.
    custom_range = [2, 2, 3]
    training_structure = get_random_atoms()
    training_structure.calc = LennardJones()
    forces = training_structure.get_forces()
    energy = training_structure.get_potential_energy()
    stress = training_structure.get_stress()

    # Update the SGP.
    sgp = get_empty_sgp()
    sgp.update_db(
        training_structure, 
        forces, 
        custom_range, 
        energy, 
        stress,
        mode="uncertain",
    )

    n_envs = len(custom_range)
    n_atoms = len(training_structure)
    assert sgp.sparse_gp.Kuu.shape[0] == np.sum(custom_range)
    assert sgp.sparse_gp.Kuf.shape[1] == 1 + n_atoms * 3 + 6


def test_train():
    """Check that the hyperparameters and likelihood are updated when the
    train method is called."""

    sgp = get_updated_sgp()
    hyps_init = tuple(sgp.hyps)
    sgp.train()
    hyps_post = tuple(sgp.hyps)

    assert hyps_init != hyps_post
    assert sgp.likelihood != 0.0


def test_dict():
    """
    Check the method from_dict and as_dict
    """

    sgp_wrapper = get_updated_sgp()
    out_dict = sgp_wrapper.as_dict()
    assert len(sgp_wrapper) == len(out_dict["training_structures"])
    new_sgp, _ = SGP_Wrapper.from_dict(out_dict)
    assert len(sgp_wrapper) == len(new_sgp)
    assert len(sgp_wrapper.sparse_gp.kernels) == len(new_sgp.sparse_gp.kernels)
    assert np.allclose(sgp_wrapper.hyps, new_sgp.hyps)


def test_dump():
    """
    Check the method from_file and write_model of SGP_Wrapper
    """

    sgp_wrapper = get_updated_sgp()
    sgp_wrapper.write_model("sgp.json")
    new_sgp, _ = SGP_Wrapper.from_file("sgp.json")
    os.remove("sgp.json")
    assert len(sgp_wrapper) == len(new_sgp)
    assert len(sgp_wrapper.sparse_gp.kernels) == len(new_sgp.sparse_gp.kernels)
    assert np.allclose(sgp_wrapper.hyps, new_sgp.hyps)


def test_calc():
    """
    Check the method from_file and write_model of SGP_Calculator
    """

    sgp_wrapper = get_updated_sgp()
    calc = SGP_Calculator(sgp_wrapper)
    calc.write_model("sgp_calc.json")
    new_calc, _ = SGP_Calculator.from_file("sgp_calc.json")
    os.remove("sgp_calc.json")
    assert len(calc.gp_model) == len(new_calc.gp_model)


def test_write_model():
    """Test that a reconstructed SGP calculator predicts the same forces
    as the original."""

    training_structure = get_random_atoms()
    sgp_calc = get_sgp_calc()

    # Predict on training structure.
    training_structure.calc = sgp_calc
    forces = training_structure.get_forces()

    # Write the SGP to JSON.
    sgp_name = "sgp_calc.json"
    sgp_calc.write_model(sgp_name)

    # Odd Pybind-related issue here that seems to be related to polymorphic
    # kernel pointers. Need to return the kernel list for SGP prediction to
    # work. Possibly related to:
    # https://stackoverflow.com/questions/49633990/polymorphism-and-pybind11
    sgp_calc_2, _ = SGP_Calculator.from_file(sgp_name)

    os.remove(sgp_name)

    # Compute forces with reconstructed SGP.
    training_structure.calc = sgp_calc_2
    forces_2 = training_structure.get_forces()

    # Check that they're the same.
    max_abs_diff = np.max(np.abs(forces - forces_2))
    assert max_abs_diff < 1e-8

def test_coeff():
    sgp_py = get_updated_sgp()

    # Dump potential coefficient file
    sgp_py.write_mapping_coefficients("lmp.flare", "A", [0, 1, 2])

    # Dump uncertainty coefficient file
    # here the new kernel needs to be returned, otherwise the kernel won't be found in the current module
    new_kern = sgp_py.write_varmap_coefficients("beta_var.txt", "B", [0, 1, 2])

    assert (
        sgp_py.sparse_gp.sparse_indices[0] == sgp_py.sgp_var.sparse_indices[0]
    ), "the sparse_gp and sgp_var don't have the same training data"

    for s in range(len(sgp_py.species_map)):
        org_desc = sgp_py.sparse_gp.sparse_descriptors[0].descriptors[s]
        new_desc = sgp_py.sgp_var.sparse_descriptors[0].descriptors[s]
        if not np.allclose(org_desc, new_desc):  # the atomic order might change
            assert np.allclose(org_desc.shape, new_desc.shape)
            for i in range(org_desc.shape[0]):
                flag = False
                for j in range(
                    new_desc.shape[0]
                ):  # seek in new_desc for matching of org_desc
                    if np.allclose(org_desc[i], new_desc[j]):
                        flag = True
                        break
                assert flag, "the sparse_gp and sgp_var don't have the same descriptors"


@pytest.mark.skipif(
    not os.environ.get("lmp", False),
    reason=(
        "lmp not found "
        "in environment: Please install LAMMPS "
        "and set the $lmp env. "
        "variable to point to the executatble."
    ),
)
def test_lammps():
    # create ASE calc
    lmp_command = os.environ.get("lmp")
    specorder = ["C", "O"]
    pot_file = "lmp.flare"
    params = {
        "command": lmp_command,
        "pair_style": "flare",
        "pair_coeff": [f"* * {pot_file}"],
    }
    files = [pot_file]
    lmp_calc = lammpsrun.LAMMPS(
        label=f"tmp",
        keep_tmp_files=True,
        tmp_dir="./tmp/",
        parameters=params,
        files=files,
        specorder=specorder,
    )

    test_atoms = get_random_atoms(a=2.0, sc_size=2, numbers=[6, 8], set_seed=12345)
    test_atoms.calc = lmp_calc
    lmp_f = test_atoms.get_forces()
    lmp_e = test_atoms.get_potential_energy()
    lmp_s = test_atoms.get_stress()

    print("GP predicting")
    test_atoms.calc = None
    test_atoms = FLARE_Atoms.from_ase_atoms(test_atoms)
    sgp_calc = get_sgp_calc()
    test_atoms.calc = sgp_calc
    sgp_f = test_atoms.get_forces()
    sgp_e = test_atoms.get_potential_energy()
    sgp_s = test_atoms.get_stress()

    print("Energy")
    print(lmp_e, sgp_e)
    assert np.allclose(lmp_e, sgp_e)

    print("Forces")
    print(np.concatenate([lmp_f, sgp_f], axis=1))
    assert np.allclose(lmp_f, sgp_f)

    print("Stress")
    print(lmp_s)
    print(sgp_s)
    assert np.allclose(lmp_s, sgp_s)


@pytest.mark.skipif(
    not os.environ.get("lmp", False),
    reason=(
        "lmp not found "
        "in environment: Please install LAMMPS "
        "and set the $lmp env. "
        "variable to point to the executatble."
    ),
)
def test_lammps_uncertainty():
    # create ASE calc
    lmp_command = os.environ.get("lmp")
    specorder = ["C", "O"]
    pot_file = "lmp.flare"
    params = {
        "command": lmp_command,
        "pair_style": "flare",
        "pair_coeff": [f"* * {pot_file}"],
    }
    files = [pot_file]
    lmp_calc = lammpsrun.LAMMPS(
        label=f"tmp",
        keep_tmp_files=True,
        tmp_dir="./tmp/",
        parameters=params,
        files=files,
        specorder=specorder,
    )

    test_atoms = get_random_atoms(a=2.0, sc_size=2, numbers=[6, 8], set_seed=54321)

    # compute uncertainty 
    in_lmp = """
atom_style atomic 
units metal
boundary p p p 
atom_modify sort 0 0.0 

read_data data.lammps 

### interactions
pair_style flare 
pair_coeff * * lmp.flare 
mass 1 1.008000 
mass 2 4.002602 

### run
fix fix_nve all nve
compute unc all flare/std/atom beta_var.txt
dump dump_all all custom 1 traj.lammps id type x y z vx vy vz fx fy fz c_unc[1] c_unc[2] c_unc[3] 
thermo_style custom step temp press cpu pxx pyy pzz pxy pxz pyz ke pe etotal vol lx ly lz atoms
thermo_modify flush yes format float %23.16g
thermo 1
run 0
"""
    os.chdir("tmp")
    write("data.lammps", test_atoms, format="lammps-data")
    with open("in.lammps", "w") as f:
        f.write(in_lmp)
    shutil.copyfile("../beta_var.txt", "./beta_var.txt")
    os.system(f"{lmp_command} < in.lammps > log.lammps")
    unc_atoms = read("traj.lammps", format="lammps-dump-text")
    sgp_py = get_updated_sgp()
    lmp_stds = [unc_atoms.get_array(f"c_unc[{i+1}]") / sgp_py.hyps[i] for i in range(len(calc_list))]
    lmp_stds = np.squeeze(lmp_stds).T

    # Test mapped variance (need to use sgp_var)
    test_atoms.calc = None
    test_atoms = FLARE_Atoms.from_ase_atoms(test_atoms)
    sgp_calc = get_sgp_calc()
    test_atoms.calc = sgp_calc
    test_atoms.calc.gp_model.sparse_gp = sgp_py.sgp_var
    test_atoms.calc.reset()
    sgp_stds = test_atoms.calc.get_uncertainties(test_atoms)
    print(sgp_stds)
    print(lmp_stds)
    assert np.allclose(sgp_stds, lmp_stds)
