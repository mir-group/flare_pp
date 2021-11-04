import numpy as np
from flare_pp._C_flare import SparseGP, NormalizedDotProduct, Bk, Structure
from flare_pp.sparse_gp import SGP_Wrapper
from flare_pp.sparse_gp_calculator import SGP_Calculator
from flare_pp.parallel_sgp import ParSGP_Wrapper
from flare.ase.atoms import FLARE_Atoms
from ase import Atoms
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.build import make_supercell


# Define kernels.
sigma = 1.0
power = 2
kernel1 = NormalizedDotProduct(sigma, power)

sigma = 2.0
power = 2
kernel2 = NormalizedDotProduct(sigma, power)

sigma = 3.0
power = 2
kernel3 = NormalizedDotProduct(sigma, power)

# Define calculators.
species_map = {6: 0, 8: 1}

cutoff = 5.0
cutoff_function = "quadratic"
radial_basis = "chebyshev"
radial_hyps = [0.0, cutoff]
cutoff_hyps = []

settings = [len(species_map), 1, 4, 0]
calc1 = Bk(radial_basis, cutoff_function, radial_hyps, cutoff_hyps, settings)

settings = [len(species_map), 2, 4, 3]
calc2 = Bk(radial_basis, cutoff_function, radial_hyps, cutoff_hyps, settings)

settings = [len(species_map), 3, 2, 2]
calc3 = Bk(radial_basis, cutoff_function, radial_hyps, cutoff_hyps, settings)

# Define remaining parameters for the SGP wrapper.
sigma_e = 0.001
sigma_f = 0.05
sigma_s = 0.006
single_atom_energies = {0: -5, 1: -6}
variance_type = "local"
max_iterations = 20
#opt_method = "L-BFGS-B"
opt_method = "BFGS"
bounds = [(None, None), (None, None), (None, None), (sigma_e, None), (None, None), (None, None)]


def get_random_atoms(a=2.0, sc_size=2, numbers=[6, 8],
                     set_seed: int = None):

    """Create a random structure."""

    if set_seed:
        np.random.seed(set_seed)

    cell = np.eye(3) * a
    positions = np.array([[0, 0, 0], [a/2, a/2, a/2]])
    unit_cell = Atoms(cell=cell, positions=positions, numbers=numbers,
                      pbc=True)
    multiplier = np.identity(3) * sc_size
    atoms = make_supercell(unit_cell, multiplier)
    atoms.positions += (2 * np.random.rand(len(atoms), 3) - 1) * 0.1
    # modify the symbols
    random_numbers = np.array(numbers)[np.random.choice(2, len(atoms), replace=True)].tolist()
    atoms.numbers = random_numbers
    flare_atoms = FLARE_Atoms.from_ase_atoms(atoms)

    calc = SinglePointCalculator(flare_atoms)
    calc.results["energy"] = np.random.rand()
    calc.results["forces"] = np.random.rand(len(atoms), 3)
    calc.results["stress"] = np.random.rand(6)
    flare_atoms.calc = calc
 
    return flare_atoms


def get_empty_sgp():
    empty_sgp = SGP_Wrapper(
        [kernel1, kernel2, kernel3], 
        [calc1, calc2, calc3], 
        cutoff, 
        sigma_e, 
        sigma_f, 
        sigma_s, 
        species_map,
        single_atom_energies=single_atom_energies, 
        variance_type=variance_type,
        opt_method=opt_method, 
        bounds=bounds, 
        max_iterations=max_iterations,
    )

    return empty_sgp

def get_updated_sgp():
    training_structure = get_random_atoms()
    training_structure.calc = LennardJones()

    forces = training_structure.get_forces()
    energy = training_structure.get_potential_energy()
    stress = training_structure.get_stress()

    sgp = get_empty_sgp()
    sgp.update_db(
        training_structure, 
        forces, 
        custom_range=[(1, 2, 3, 4, 5) for k in range(len(sgp.descriptor_calculators))],
        energy=energy, 
        stress=stress, 
        mode="specific",
    )

    return sgp


def get_sgp_calc():
    sgp = get_updated_sgp()
    sgp_calc = SGP_Calculator(sgp)

    return sgp_calc


def get_empty_parsgp():
    empty_sgp = ParSGP_Wrapper(
        [kernel1, kernel2, kernel3], 
        [calc1, calc2, calc3], 
        cutoff, 
        sigma_e, 
        sigma_f, 
        sigma_s, 
        species_map,
        single_atom_energies=single_atom_energies, 
        variance_type=variance_type,
        opt_method=opt_method, 
        bounds=bounds, 
        max_iterations=max_iterations,
    )

    return empty_sgp

def get_training_data():
    # Make random structure.
    sgp = get_empty_sgp()
    n_frames = 5
    training_strucs = []
    training_sparse_indices = [[] for i in range(len(sgp.descriptor_calculators))]
    for n in range(n_frames): 
        train_structure = get_random_atoms(a=2.0, sc_size=2, numbers=list(species_map.keys()))
        n_atoms = len(train_structure)
        training_strucs.append(train_structure)
        for k in range(len(sgp.descriptor_calculators)):
            training_sparse_indices[k].append(np.random.randint(0, n_atoms, n_atoms // 2).tolist())
    return training_strucs, training_sparse_indices
