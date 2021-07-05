import numpy as np
from ase.io import read, write
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


def add_sparse_indices_to_xyz(xyz_file_in, ind_file, xyz_file_out):
    """
    Suppose we have an .xyz file saving all the DFT frames from OTF training, 
    and a file saving the sparse indices of atoms from each frame in the .xyz
    file. For example, `sparse_indices.txt` has lines with format:
    ```
    frame_number    ind1 ind2 ind3
    ```
    Then this function combines the two files, such that the sparse indices
    are written to the .xyz frames.

    Args:
        xyz_file_in (str): the file name of the .xyz file, which saves the 
            DFT frames from OTF.
        ind_file (str): the file name of the sparse indices, whose number of 
            rows should be equal to the number of frames in `xyz_file_in`.
        xyz_file_out (str): the output file name. The output will be an .xyz file.
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
    Write a list of `Structure` objects (and their sparse indices) into an .xyz file

    Args:
        struc_list (list): a list of `Structure` objects.
        xyz_file_out (str): the output file name. The output will be an .xyz file.
        species_map (dict): the map from chemical number to coded species. For
            example, `species_map = {14: 0, 6: 1}`, means that in the structure list,
            the coded species 0 corresponds to silicon, and coded species 1 corresponds
            to carbon.
        sparse_indices (list): a list of arrays. Each array is the indices of sparse 
            atoms added to GP in the corresponding `Structure` object.
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

