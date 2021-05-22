import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


def add_sparse_indices_to_xyz(xyz_file_in, ind_file, xyz_file_out):
    """
    Example:
        sparse_indices.txt has lines with format:
        frame_number    ind1 ind2 ind3
    """
    frames = read(xyz_file_in, format="extxyz", index=":")
    indices = open(ind_file).readlines()
    assert len(frames) == len(indices)
    for i in range(len(frames)):
        sparse_ind = indices[i].split()[1:]
        sparse_ind = np.array([int(s) for s in sparse_ind])
        frames[i].info["sparse_indices"] = sparse_ind
    
    write(xyz_file_out, frames, format="extxyz")


def struc_list_to_xyz(struc_list, xyz_file_out, species_map, sparse_indices=None):
    """
    Args:
        struc_list: a list of Structure objects
    """
    if sparse_indices:
        assert len(sparse_indices) == len(struc_list)

    frames = []
    code2species = {v: k for k, v in species_map.items()} # an inverse map of species_map
    for i, struc in enumerate(struc_list): 
        species_number = [code2species[s] for s in struc.species]
        atoms = Atoms(
            number=species_number, 
            positions=struc.positions, 
            cell=struc.cell, 
            pbc=True
        )
    
        properties = ["forces", "energy", "stress"]
        results = {"forces": struc.forces, "energy": struc.energy, "stress": struc.stresses}
        calculator = SinglePointCalculator(atoms, **results)
        atoms.set_calculator(calculator)

        if sparse_indices:
            atoms.info["sparse_indices"] = sparse_indices[i]

        frames.append(atoms)

    write(xyz_file_out, frames, format="extxyz")

