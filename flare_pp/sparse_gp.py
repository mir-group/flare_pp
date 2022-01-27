import json
from time import time
import numpy as np
from flare_pp import _C_flare
from flare_pp._C_flare import SparseGP, Structure, NormalizedDotProduct
from scipy.optimize import minimize
from typing import List
import warnings
from flare import struc
from flare.ase.atoms import FLARE_Atoms
from flare.utils.element_coder import NumpyEncoder


class SGP_Wrapper:
    """
    Wrapper class used to make the C++ sparse GP object compatible with
    OTF. Methods and properties are designed to mirror the GP class.
    __init__ constructor to initialize the sparse GP object. 
    Args: 
        kernels (list): selection which determines an environment's similarity to others. Typically just [NormalizedDotProduct(sigma, power)].
        descriptor_calculators (list): can choose between B1, B2, and B3 ACE descriptors, or a combination of them (e.g., B1 + B2 + B3).
        cutoff (float): cutoff (in angstroms) to define the atomic environment. Can be optimized on a system-basis (using grid-test) to increase log-likelihood of GP.
        sigma_e (float): initial guess for energy-noise hyperparamter in GP function. Typically set to 1 meV/atom (e.g., 0.001 * n_atoms, units of eV).
        sigma_f (float): initial guess for force-noise hyperparamter in GP function. This value is typically system dependent, but should range from 10 meV to 1 eV/A.
        sigma_s (float): initial guess for stress-noise hyperparamter in GP function. 
        species_map (dict): dictionary which maps atoms to new indices in GP model. Requires atomic number of each species (e.g., system containing Au and H yields {79: 0, 1: 1}).
        variance_type (str): selection that determines what type of variance is used in GP.
        single_atom_energies (dict): dictionary to assign approximate values for single atom energies. Typically helps in training GP, as energy noise hyperparameter can blow-up otherwise.
        energy_training (bool): set to `True` for GP to train on energy labels. Usually helps for improving both energy and force predictions.
        force_training (bool): set to `True` for GP to train on force labels. Most important choice (as one typically has more force labels than energy or stress).
        stress_training (bool): set to `True` for GP to train on stress labels. Typically system dependent.
        max_iterations (int): integer to determine length of hyperparameter optimization. Typically set to value between 50 and 100, otherwise hyperparameters do not fully converge.
        opt_method (str): flag for determinining minimization method employed during hyperparameter optimization.
        bounds (dict?): option to constrain the range during hyperparameter optimization. Can limit certain values exploding due to data scarcity.
    """

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
        max_iterations=100,
        opt_method="BFGS",
        bounds=None,
    ):
        self.sparse_gp = SparseGP(kernels, sigma_e, sigma_f, sigma_s)
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
                self.sgp_var = self.sparse_gp
            else:
                self.sgp_var_flag = "new"
        else:
            warnings.warn(
                "kernels[0] should be NormalizedDotProduct for variance mapping"
            )
            self.sgp_var_flag = None

    @property
    def training_data(self):
        """
        Point the property `training_data` to Sparse GP c++ `training_structures`.
        """
        return self.sparse_gp.training_structures

    @property
    def hyps(self):
        """
        Point the property `hyps` to Sparse GP c++ `hyperparameters`.
        """
        return self.sparse_gp.hyperparameters

    @property
    def hyps_and_labels(self):
        """
        Point the property `hyps_and_labels` to Sparse GP c++ `hyps` and `hyp_labels`.
        """
        return self.hyps, self.hyp_labels

    @property
    def likelihood(self):
        """
        Point the property `likelihood` to Sparse GP c++ `log_marginal_likelihood`.
        """
        return self.sparse_gp.log_marginal_likelihood

    @property
    def likelihood_gradient(self):
        """
        Point the property `likelihood_gradient` to Sparse GP c++ `likelihood_gradient`.
        """
        return self.sparse_gp.likelihood_gradient

    @property
    def force_noise(self):
        """
        Point the property `force_noise` to Sparse GP c++ `force_noise`.
        """
        return self.sparse_gp.force_noise

    def __str__(self):
        """
        Define GP string containing number and array of hyperparameters.
        """
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
        """
        Define function to return length of training data
        """
        return len(self.training_data)

    def check_L_alpha(self):
        pass

    def write_model(self, name: str):
        """
        Write model to .json file
        Args:
            name (str): Name of model to be written (as .json file)
        """
        if ".json" != name[-5:]:
            name += ".json"
        with open(name, "w") as f:
            json.dump(self.as_dict(), f, cls=NumpyEncoder)

    def as_dict(self):
        """
        Converts the class object into a dictionary
        """
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

        kernel_list = []
        if getattr(self, "sgp_var_kernels", None):
            for kern in self.sgp_var_kernels:
                if isinstance(kern, NormalizedDotProduct):
                    kernel_list.append(("NormalizedDotProduct", kern.sigma, kern.power))
                else:
                    raise NotImplementedError
            out_dict["sgp_var_kernels"] = kernel_list

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
        Builds class object from dictionary. Need an initialized GP.
        Args:
            in_dict (dict): requires initialized GP in dictionary format 
        """
        # Recover kernel from checkpoint.
        kernel_list = in_dict["kernels"]
        assert len(kernel_list) == 1
        kernel_hyps = kernel_list[0]
        assert kernel_hyps[0] == "NormalizedDotProduct"
        sigma = float(kernel_hyps[1])
        power = int(kernel_hyps[2])
        kernel = NormalizedDotProduct(sigma, power)
        kernels = [kernel]

        # Recover descriptor from checkpoint.
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
        if in_dict["single_atom_energies"] is not None:
             sae_dict = {int(k): v for k, v in in_dict["single_atom_energies"].items()}
        else:
             sae_dict = None
        species_map = {int(k): v for k, v in in_dict["species_map"].items()}

        gp = SGP_Wrapper(
            kernels=[kernel],
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
        """
        Read from file and construct dictionary
        Args:
            filename (str): filename to read and return `from_dict` object
        """
        with open(filename, "r") as f:
            in_dict = json.loads(f.readline())
        return SGP_Wrapper.from_dict(in_dict)

    def update_db(
        self,
        structure: Structure,
        forces,
        custom_range=(),
        energy: float = None,
        stress: "ndarray" = None,
        mode: str = "specific",
        sgp: SparseGP = None,  # for creating sgp_var
        update_qr=True,
        atom_indices=[-1],
    ):
        """
        Update database, define number of sparse environments to add per frame, and determine further processing (e.g., kernel matrices)
        Args:
            structure (Structure): flare structure class
            forces (class): forces 
            custom_range (tuple): number of sparse environments to be included from each frame
            energy (float): energy from structure object, default = `None`
            stress (str): stress from structure object, default = `None`
            mode (str): mode for defining atomic environments
            sgp (bool): set to `True` to produce Sparse GP (with number of atomic envs from custom_range to be added), or `False` to add all atomic envs from each frame, default = `None`
            update_qr (bool): set to `True` to build kernel matrices from updated training set, compute and store the matrices needed for future prediction, default = `None`
            atom_indices (list): keep track of atom indices for atoms added to sparse, or full, set (depending on custom_range)
        """

        # Convert coded species to 0, 1, 2, etc.
        if isinstance(structure, (struc.Structure, FLARE_Atoms)):
            coded_species = []
            for spec in structure.coded_species:
                coded_species.append(self.species_map[spec])
        elif isinstance(structure, Structure):
            coded_species = structure.species
        else:
            raise Exception

        # Convert flare structure to structure descriptor.
        structure_descriptor = Structure(
            structure.cell,
            coded_species,
            structure.positions,
            self.cutoff,
            self.descriptor_calculators,
        )

        # Add labels to structure descriptor.
        if (energy is not None) and (self.energy_training):
            # Sum up single atom energies.
            single_atom_sum = 0
            if self.single_atom_energies is not None:
                for spec in coded_species:
                    single_atom_sum += self.single_atom_energies[spec]

            # Correct the energy label and assign to structure.
            corrected_energy = energy - single_atom_sum
            structure_descriptor.energy = np.array([[corrected_energy]])

        if (forces is not None) and (self.force_training):
            structure_descriptor.forces = forces.reshape(-1)

        if (stress is not None) and (self.stress_training):
            structure_descriptor.stresses = stress

        # Update the sparse GP.
        if sgp is None:
            sgp = self.sparse_gp

        sgp.add_training_structure(structure_descriptor, atom_indices)
        if mode == "all":
            if not custom_range:
                sgp.add_all_environments(structure_descriptor)
            else:
                raise Exception("Set mode='specific' for a user-defined custom_range")
        elif mode == "uncertain":
            if len(custom_range) == 1:  # custom_range gives n_added
                n_added = custom_range
                sgp.add_uncertain_environments(structure_descriptor, n_added)
            else:
                raise Exception(
                    "The custom_range should be set as [n_added] if mode='uncertain'"
                )
        elif mode == "specific":
            if not custom_range:
                warnings.warn(
                    "The mode='specific' but no custom_range is given, will not add sparse envs"
                )
            else:
                sgp.add_specific_environments(structure_descriptor, custom_range)
        elif mode == "random":
            if len(custom_range) == 1:  # custom_range gives n_added
                n_added = custom_range
                sgp.add_random_environments(structure_descriptor, n_added)
            else:
                raise Exception(
                    "The custom_range should be set as [n_added] if mode='random'"
                )
        else:
            raise NotImplementedError

        if update_qr:
            sgp.update_matrices_QR()

    def set_L_alpha(self):
        # Taken care of in the update_db method.
        pass

    def train(self, logger_name=None):
        """
        optimize hyperparameters given current database of atomic environments
        Args:
            logger_name: default = `None` (not accessed??)
        """
        optimize_hyperparameters(
            self.sparse_gp,
            max_iterations=self.max_iterations,
            method=self.opt_method,
            bounds=self.bounds,
        )

    def write_mapping_coefficients(self, filename, contributor, kernel_idx):
        """
        write mapping coefficient for sparse GP
        Args:
            filename (str): string for file name assignment
            contributor (str): string to assign contributor name (creator)
            kernel_idx (str?): kernel index
        """
        self.sparse_gp.write_mapping_coefficients(filename, contributor, kernel_idx)

    def write_varmap_coefficients(self, filename, contributor, kernel_idx):
        """
        write mapping coefficient for sparse GP
        Args:
            filename (str): string for file name assignment
            contributor (str): string to assign contributor name (creator)
            kernel_idx (str?): kernel index    
        """
        old_kernels = self.sparse_gp.kernels
        assert (len(old_kernels) == 1) and (
            kernel_idx == 0
        ), "Not support multiple kernels"
        assert isinstance(old_kernels[0], NormalizedDotProduct)

        power = 1
        new_kernels = [NormalizedDotProduct(old_kernels[0].sigma, power)]

        # Build a power=1 SGP from scratch
        if self.sgp_var is None:
            print("Build a new SGP with power = 1")
            self.sgp_var, new_kernels = self.duplicate(new_kernels=new_kernels)
            self.sgp_var_kernels = new_kernels

        # Check hyperparameters and training data, if not match, construct a new SGP
        assert len(self.sgp_var.kernels) == 1
        assert np.allclose(self.sgp_var.kernels[0].power, 1.0)
        is_same_hyps = np.allclose(
            self.sgp_var.hyperparameters, self.sparse_gp.hyperparameters
        )
        n_sgp = len(self.training_data)
        n_sgp_var = len(self.sgp_var.training_structures)
        is_same_data = n_sgp == n_sgp_var 

        # Add new data if sparse_gp has more data than sgp_var
        if not is_same_data:
            n_add = n_sgp - n_sgp_var
            assert n_add > 0, "sgp_var has more training data than sgp"
            print("Training data not match, adding", n_add, "structures")
            for s in range(n_add):
                custom_range = self.sparse_gp.sparse_indices[0][s + n_sgp_var]
                struc_cpp = self.training_data[s + n_sgp_var]

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

            self.sgp_var.update_matrices_QR()

        if not is_same_hyps:
            print("Hyps not match, set hyperparameters")
            self.sgp_var.set_hyperparameters(self.sparse_gp.hyperparameters)
            self.sgp_var_kernels = self.sgp_var.kernels

        new_kernels = self.sgp_var.kernels
        print("Map with current sgp_var")

        self.sgp_var.write_varmap_coefficients(filename, contributor, kernel_idx)

        return new_kernels

    def duplicate(self, new_hyps=None, new_kernels=None, new_powers=None):
        # TODO: change to __copy__ method
        # TODO: add compatibility with other kernels
        if new_hyps is None:
            hyps = self.sparse_gp.hyperparameters
        else:
            hyps = new_hyps

        if new_kernels is None:
            assert len(hyps) == len(self.sparse_gp.kernels) + 3
            kernels = []
            for k, kern in enumerate(self.sparse_gp.kernels):
                assert isinstance(kern, NormalizedDotProduct)
                if new_powers is not None:
                    power = new_powers[k]
                else:
                    power = kern.power
                kernels.append(NormalizedDotProduct(hyps[k], power))
        else:
            kernels = new_kernels

        n_kern = len(kernels)
        new_gp = SparseGP(kernels, hyps[n_kern], hyps[n_kern + 1], hyps[n_kern + 2])

        # add training data
        sparse_indices = self.sparse_gp.sparse_indices
        assert len(sparse_indices) == len(kernels)
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
                sgp=new_gp,
                update_qr=False,
            )

        new_gp.update_matrices_QR()
        return new_gp, kernels


def compute_negative_likelihood(hyperparameters, sparse_gp,
                                print_vals=False):
    """
    Compute the negative log likelihood and gradient with respect to the
    hyperparameters.
    Args:
        hyperparameters (list): hyperparameters
        sparse_gp (class): sparse GP class object
        print_vals (bool): print values of hyperparameters and negative likelihood
    """
    assert len(hyperparameters) == len(sparse_gp.hyperparameters)

    sparse_gp.set_hyperparameters(hyperparameters)
    sparse_gp.compute_likelihood()
    negative_likelihood = -sparse_gp.log_marginal_likelihood

    if print_vals:
        print_hyps(hyperparameters, negative_likelihood)

    return negative_likelihood


def compute_negative_likelihood_grad(hyperparameters, sparse_gp,
                                     print_vals=False):
    """
    Compute the negative log likelihood and gradient with respect to the
    hyperparameters.
    Args:
        hyperparameters (list): hyperparameters
        sparse_gp (class): sparse GP class object
        print_vals (bool): print values of hyperparameters and negative likelihood
    """
    assert len(hyperparameters) == len(sparse_gp.hyperparameters)

    negative_likelihood = \
        -sparse_gp.compute_likelihood_gradient(hyperparameters)
    negative_likelihood_gradient = -sparse_gp.likelihood_gradient

    if print_vals:
        print_hyps_and_grad(
            hyperparameters, negative_likelihood_gradient, negative_likelihood
            )

    return negative_likelihood, negative_likelihood_gradient

def compute_negative_likelihood_grad_stable(hyperparameters, sparse_gp, precomputed=False):
    """
    Compute the negative log likelihood and gradient with respect to the
    hyperparameters.
    Args:
        hyperparameters (list): hyperparameters
        sparse_gp (SparseGP): Flare sparse GP object
        precomputed (bool): do not compute likelihood gradient
    """
    assert len(hyperparameters) == len(sparse_gp.hyperparameters)

    sparse_gp.set_hyperparameters(hyperparameters)

    negative_likelihood = -sparse_gp.compute_likelihood_gradient_stable(precomputed)
    negative_likelihood_gradient = -sparse_gp.likelihood_gradient

    print_hyps_and_grad(
        hyperparameters, negative_likelihood_gradient, negative_likelihood
    )

    return negative_likelihood, negative_likelihood_gradient


def print_hyps(hyperparameters, neglike):
    """
    Print hyperparamters and likelihood.
    Args:
        hyperparameters (list): hyperparameter list
        neglike (float): negative likelihood
    """
    print("Hyperparameters:")
    print(hyperparameters)
    print("Likelihood:")
    print(-neglike)
    print("\n")


def print_hyps_and_grad(hyperparameters, neglike_grad, neglike):
    """
    Print hyperparamters, likelihood gradient, and likelihood.
    Args:
        hyperparameters (list): hyperparameter list
        neglike_grad (float): negative likelihood gradient
        neglike (float): negative likelihood
    """
    print("Hyperparameters:")
    print(hyperparameters)
    print("Likelihood gradient:")
    print(-neglike_grad)
    print("Likelihood:")
    print(-neglike)
    print("\n")


def optimize_hyperparameters(
    sparse_gp,
    display_results=False,
    gradient_tolerance=1e-4,
    max_iterations=100,
    bounds=None,
    method="BFGS",
):
    """
    Optimize the hyperparameters of the sparse GP model. Methods provided below.
    Args:
        sparse_gp (SparseGP): Flare sparse GP object
        display_results (bool): set to `True` to display hyperparameter optimization steps
        gradient_tolerance (float): convergence threshold for hyperparameter gradient
        max_iterations (int): maximum number of hyperparameter optimization steps to attempt if convergence tolerance is not met
        bounds (dict): hyperparameter bounds for optimization
        method (str): optimization method to be employed
    """

    initial_guess = sparse_gp.hyperparameters
    precompute = True
    for kern in sparse_gp.kernels:
        if not isinstance(kern, NormalizedDotProduct):
            precompute = False
            break
    if precompute:
        tic = time()
        print("Precomputing KnK for hyps optimization")
        sparse_gp.precompute_KnK()
        print("Done precomputing. Time:", time() - tic)
        arguments = (sparse_gp, precompute)
    else:
        arguments = (sparse_gp, precompute)

    if method == "BFGS":
        optimization_result = minimize(
            compute_negative_likelihood_grad_stable,
            initial_guess,
            arguments,
            method="BFGS",
            jac=True,
            options={
                "disp": display_results,
                "gtol": gradient_tolerance,
                "maxiter": max_iterations,
            },
        )

        # Assign likelihood gradient.
        sparse_gp.likelihood_gradient = -optimization_result.jac

    elif method == "L-BFGS-B":
        optimization_result = minimize(
            compute_negative_likelihood_grad_stable,
            initial_guess,
            arguments,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "disp": display_results,
                "gtol": gradient_tolerance,
                "maxiter": max_iterations,
            },
        )

        # Assign likelihood gradient.
        sparse_gp.likelihood_gradient = -optimization_result.jac

    elif method == "nelder-mead":
        optimization_result = minimize(
            compute_negative_likelihood,
            initial_guess,
            arguments,
            method="nelder-mead",
            options={
                "disp": display_results,
                "maxiter": max_iterations,
            },
        )

    # Set the hyperparameters to the optimal value.
    sparse_gp.set_hyperparameters(optimization_result.x)
    sparse_gp.log_marginal_likelihood = -optimization_result.fun

    return optimization_result


if __name__ == "__main__":
    pass
