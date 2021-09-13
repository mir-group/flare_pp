import json
import numpy as np
from flare_pp import _C_flare
from flare_pp._C_flare import ParallelSGP, Structure, NormalizedDotProduct
from scipy.optimize import minimize
from typing import List
import warnings
from flare import struc
from flare.ase.atoms import FLARE_Atoms
from flare.utils.element_coder import NumpyEncoder
from mpi4py import MPI


class ParSGP_Wrapper:
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

        self.sparse_gp = ParallelSGP(kernels, sigma_e, sigma_f, sigma_s)
        self.descriptor_calculators = descriptor_calculators
        self.cutoff = cutoff
        self.hyps_mask = None
        self.species_map = species_map
        self.variance_type = variance_type
        self.single_atom_energies = single_atom_energies
        self.energy_training = energy_training
        self.force_training = force_training
        self.stress_training = stress_training
        self.max_iterations = max_iterations
        self.opt_method = opt_method
        self.bounds = bounds

        # Make placeholder hyperparameter labels.
        self.hyp_labels = []
        for n in range(len(self.hyps)):
            self.hyp_labels.append("Hyp" + str(n))

        # prepare a new sGP for variance mapping
        self.sgp_var = None
        if isinstance(
            kernels[0], NormalizedDotProduct
        ):  # TODO: adapt this to multiple kernels
            if kernels[0].power == 1:
                self.sgp_var_flag = "self"
            else:
                self.sgp_var_flag = "new"
        else:
            warnings.warn(
                "kernels[0] should be NormalizedDotProduct for variance mapping"
            )
            self.sgp_var_flag = None

    @property
    def training_data(self):
        return self.sparse_gp.training_structures

    @property
    def hyps(self):
        return self.sparse_gp.hyperparameters

    @property
    def hyps_and_labels(self):
        return self.hyps, self.hyp_labels

    @property
    def likelihood(self):
        return self.sparse_gp.log_marginal_likelihood

    @property
    def likelihood_gradient(self):
        return self.sparse_gp.likelihood_gradient

    @property
    def force_noise(self):
        return self.sparse_gp.force_noise

    def __str__(self):
        gp_str = ""
        gp_str += f"Number of hyperparameters: {len(self.hyps)}\n"
        gp_str += f"Hyperparameter array: {str(self.hyps)}\n"

        if self.hyp_labels is None:
            # Put unlabeled hyperparameters on one line
            gp_str = gp_str[:-1]
            gp_str += str(self.hyps) + "\n"
        else:
            for hyp, label in zip(self.hyps, self.hyp_labels):
                gp_str += f"{label}: {hyp} \n"

        return gp_str

    def __len__(self):
        return len(self.training_data)

    def check_L_alpha(self):
        pass

    def write_model(self, name: str):
        """
        Write to .json file
        """
        if ".json" != name[-5:]:
            name += ".json"
        with open(name, "w") as f:
            json.dump(self.as_dict(), f, cls=NumpyEncoder)

    def as_dict(self):
        out_dict = {}
        for key in vars(self):
            if key not in ["sparse_gp", "sgp_var", "descriptor_calculators"]:
                out_dict[key] = getattr(self, key, None)

        # save descriptor_settings
        desc_calc = self.descriptor_calculators
        assert (len(desc_calc) == 1) and (isinstance(desc_calc[0], _C_flare.B2))
        b2_calc = desc_calc[0]
        b2_dict = {
            "type": "B2",
            "radial_basis": b2_calc.radial_basis,
            "cutoff_function": b2_calc.cutoff_function,
            "radial_hyps": b2_calc.radial_hyps,
            "cutoff_hyps": b2_calc.cutoff_hyps,
            "descriptor_settings": b2_calc.descriptor_settings,
        }
        out_dict["descriptor_calculators"] = [b2_dict]

        # save hyps
        out_dict["hyps"], out_dict["hyp_labels"] = self.hyps_and_labels

        # only save kernel type and hyps
        kernel_list = []
        for kern in self.sparse_gp.kernels:
            if isinstance(kern, NormalizedDotProduct):
                kernel_list.append(("NormalizedDotProduct", kern.sigma, kern.power))
            else:
                raise NotImplementedError
        out_dict["kernels"] = kernel_list

        out_dict["training_structures"] = []
        for s in range(len(self.training_data)):
            custom_range = self.sparse_gp.sparse_indices[0][s]
            struc_cpp = self.training_data[s]

            # invert mapping of species
            inv_species_map = {v: k for k, v in self.species_map.items()}
            species = [inv_species_map[s] for s in struc_cpp.species]

            # build training structure
            train_struc = struc.Structure(
                struc_cpp.cell,
                species,
                struc_cpp.positions,
            )
            train_struc.forces = struc_cpp.forces
            train_struc.stress = struc_cpp.stresses

            # Add back the single atom energies to dump the original energy
            single_atom_sum = 0
            if self.single_atom_energies is not None:
                for spec in struc_cpp.species:
                    single_atom_sum += self.single_atom_energies[spec]

            train_struc.energy = struc_cpp.energy + single_atom_sum

            out_dict["training_structures"].append(train_struc.as_dict())

        out_dict["sparse_indice"] = self.sparse_gp.sparse_indices
        return out_dict

    @staticmethod
    def from_dict(in_dict):
        """
        Need an initialized GP
        """
        # recover kernels from checkpoint
        kernel_list = in_dict["kernels"]
        kernels = []
        for k, kern in enumerate(kernel_list):
            if kern[0] != "NormalizedDotProduct":
                raise NotImplementedError
            assert kern[1] == in_dict["hyps"][k]
            kernels.append(NormalizedDotProduct(kern[1], kern[2]))
        
        # recover descriptors from checkpoint
        desc_calc = in_dict["descriptor_calculators"]
        assert len(desc_calc) == 1
        b2_dict = desc_calc[0]
        assert b2_dict["type"] == "B2"
        calc = _C_flare.B2(
            b2_dict["radial_basis"],
            b2_dict["cutoff_function"],
            b2_dict["radial_hyps"],
            b2_dict["cutoff_hyps"],
            b2_dict["descriptor_settings"],
        )

        # change the keys of single_atom_energies and species_map to int
        sae_dict = {int(k): v for k, v in in_dict["single_atom_energies"].items()}
        species_map = {int(k): v for k, v in in_dict["species_map"].items()}

        gp = ParSGP_Wrapper(
            kernels=kernels,
            descriptor_calculators=[calc],
            cutoff=in_dict["cutoff"],
            sigma_e=in_dict["hyps"][-3],
            sigma_f=in_dict["hyps"][-2],
            sigma_s=in_dict["hyps"][-1],
            species_map=species_map,
            variance_type=in_dict["variance_type"],
            single_atom_energies=sae_dict,
            energy_training=in_dict["energy_training"],
            force_training=in_dict["force_training"],
            stress_training=in_dict["stress_training"],
            max_iterations=in_dict["max_iterations"],
            opt_method=in_dict["opt_method"],
            bounds=in_dict["bounds"],
        )

        # update db
        training_data = in_dict["training_structures"]
        for s in range(len(training_data)):
            custom_range = in_dict["sparse_indice"][0][s]
            train_struc = struc.Structure.from_dict(training_data[s])

            if len(train_struc.energy) > 0:
                energy = train_struc.energy[0]
            else:
                energy = None

            gp.update_db(
                train_struc,
                train_struc.forces,
                custom_range=custom_range,
                energy=energy,
                stress=train_struc.stress,
                mode="specific",
                sgp=None,
                update_qr=False,
            )

        gp.sparse_gp.update_matrices_QR()
        return gp, kernels

    @staticmethod
    def from_file(filename: str):
        with open(filename, "r") as f:
            in_dict = json.loads(f.readline())
        return ParSGP_Wrapper.from_dict(in_dict)

    def build(
        self,
        training_cells,
        training_species,
        training_positions,
        training_labels,
        training_sparse_indices, 
    ):
        self.sparse_gp.build(
            training_cells,
            training_species,
            training_positions,
            training_labels,
            self.cutoff,
            self.descriptor_calculators,
            training_sparse_indices, 
        )

    def write_mapping_coefficients(self, filename, contributor, kernel_idx):
        self.sparse_gp.write_mapping_coefficients(filename, contributor, kernel_idx)

    def write_varmap_coefficients(self, filename, contributor, kernel_idx):
        old_kernels = self.sparse_gp.kernels
        assert (len(old_kernels) == 1) and (
            kernel_idx == 0
        ), "Not support multiple kernels"

        if self.sgp_var_flag == "new":
            # change to power 1 kernel
            power = 1
            new_kernels = [NormalizedDotProduct(old_kernels[0].sigma, power)]

            self.sgp_var = ParallelSGP(
                new_kernels,
                self.sparse_gp.energy_noise,
                self.sparse_gp.force_noise,
                self.sparse_gp.stress_noise,
            )

            # add training data
            sparse_indices = self.sparse_gp.sparse_indices
            assert len(sparse_indices) == len(old_kernels)
            assert len(sparse_indices[0]) == len(self.training_data)

            for s in range(len(self.training_data)):
                custom_range = sparse_indices[0][s]
                struc_cpp = self.training_data[s]

                if len(struc_cpp.energy) > 0:
                    energy = struc_cpp.energy[0]
                else:
                    energy = None

                self.update_db(
                    struc_cpp,
                    struc_cpp.forces,
                    custom_range=custom_range,
                    energy=energy,
                    stress=struc_cpp.stresses,
                    mode="specific",
                    sgp=self.sgp_var,
                    update_qr=False,
                )

            # write var map coefficient file
            self.sgp_var.update_matrices_QR()
            self.sgp_var.write_varmap_coefficients(filename, contributor, kernel_idx)
            return new_kernels

        elif self.sgp_var_flag == "self":
            self.sparse_gp.write_varmap_coefficients(filename, contributor, kernel_idx)
            self.sgp_var = self.sparse_gp
            return old_kernels

