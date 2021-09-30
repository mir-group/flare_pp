import json
import numpy as np
from scipy.optimize import minimize
from typing import List, Union, Tuple
import warnings
from flare import struc
from flare.ase.atoms import FLARE_Atoms
from flare.utils.element_coder import NumpyEncoder

from mpi4py import MPI

from flare_pp._C_flare import ParallelSGP
from flare_pp.sparse_gp import SGP_Wrapper
from flare_pp.utils import convert_to_flarepp_structure


class ParSGP_Wrapper(SGP_Wrapper):
    """Wrapper class used to make the C++ sparse GP object compatible with
    OTF. Methods and properties are designed to mirror the GP class."""

    def __init__(
        self,
        kernels: List,
        descriptor_calculators: List,
        cutoff: float,
        sigma_e: float,
        sigma_f: float,
        sigma_s: float,
        species_map: dict,
        variance_type: str = "SOR",
        single_atom_energies: dict = None,
        energy_training=True,
        force_training=True,
        stress_training=True,
        max_iterations=10,
        opt_method="BFGS",
        bounds=None,
    ):

        super().__init__(
            kernels=kernels,
            descriptor_calculators=descriptor_calculators,
            cutoff=cutoff,
            sigma_e=sigma_e,
            sigma_f=sigma_f,
            sigma_s=sigma_s,
            species_map=species_map,
            variance_type=variance_type,
            single_atom_energies=single_atom_energies,
            energy_training=energy_training,
            force_training=force_training,
            stress_training=stress_training,
            max_iterations=max_iterations,
            opt_method=opt_method,
            bounds=bounds,
        )
        self.sparse_gp = ParallelSGP(kernels, sigma_e, sigma_f, sigma_s)

    def build(
        self,
        training_strucs: Union[List[struc.Structure], List[FLARE_Atoms]],
        training_sparse_indices: List[List[List[int]]],
    ):
        # Check the shape of sparse_indices
        assert (
            len(training_sparse_indices[0][0]) >= 0
        ), """Sparse indices should be a list
                [[[atom1_of_kernel1_of_struc1, ...], 
                  [atom1_of_kernel2_of_struc1, ...]],
                 [[atom1_of_kernel1_of_struc2, ...], 
                  [atom1_of_kernel2_of_struc2, ...]]]"""

        # Convert flare Structure or FLARE_Atoms to flare_pp Structure
        struc_list = []
        for structure in training_strucs:
            energy = structure.energy
            forces = structure.forces

            # Convert stress order to xx, xy, xz, yy, yz, zz
            s = structure.stress
            stress = None
            if s is not None:
                if len(s) == 6:
                    stress = - s[[0, 5, 4, 1, 3, 2]]
                elif s.shape == (3, 3):
                    stress = - np.array([s[0, 0], s[0, 1], s[0, 2], s[1, 1], s[1, 2], s[2, 2]]) 

            structure_descriptor = convert_to_flarepp_structure(
                structure,
                self.species_map,
                energy,
                forces,
                stress,
                self.energy_training,
                self.force_training,
                self.stress_training,
                self.single_atom_energies,
                cutoff=None,
                descriptor_calculators=None,
            )

            struc_list.append(structure_descriptor)

        n_types = len(self.species_map)
        self.sparse_gp.build(
            struc_list, 
            self.cutoff, 
            self.descriptor_calculators, 
            training_sparse_indices,
            n_types,
        )
