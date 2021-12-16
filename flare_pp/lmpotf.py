from lammps import lammps
import ase
import os
import numpy as np
import sys
from typing import Union, Optional, Callable, Any, List
from flare_pp._C_flare import Structure, SparseGP
from flare_pp.sparse_gp import optimize_hyperparameters
import logging


def transform_stress(stress: List[List[float]]) -> List[List[float]]:
    return -np.array(
        [
            stress[0, 0],
            stress[0, 1],
            stress[0, 2],
            stress[1, 1],
            stress[1, 2],
            stress[2, 2],
        ]
    )


class LMPOTF:
    """
    Module for performing On-The-Fly (OTF) training, also known as active learning,
    entirely within LAMMPS.
    """

    def __init__(
        self,
        sparse_gp: SparseGP,
        descriptors: List,
        rcut: float,
        type2number: Union[int, List[int]],
        dftcalc: object,
        energy_correction: Optional[float] = None,
        dft_call_threshold: float = 0.005,
        dft_add_threshold: float = 0.0025,
        std_xyz_fname: Optional[Callable[[int], str]] = None,
        model_fname: str = "otf.flare",
        hyperparameter_optimization: Callable[
            ["LMPOTF", object, int], bool
        ] = lambda lmpotf, lmp, step: False,
        opt_bounds: Optional[List[int]] = None,
        opt_method: Optional[str] = "L-BFGS-B",
        opt_iterations: Optional[int] = 50,
        post_dft_callback: Callable[["LMPOTF", int], None] = lambda lmpotf, step: 0,
        wandb: object = None,
        log_fname: str = "otf.log",
    ) -> object:
        """

        Parameters
        ----------
        sparse_gp
            The :cpp:class:`SparseGP` object to train.
        descriptors
            A list of descriptor objects, or a single descriptor (most common), e.g. :cpp:class:`B2`.
        rcut
            The interaction cut-off radius.
        type2number
            The atomic numbers of all LAMMPS types.
        dftcalc
            An ASE calculator, e.g. Espresso.
        energy_correction
            Correction to the DFT potential energy. If not provided, it will be set to the potential energy
            of the initial configuration.
            This ensures that the energies will be centered around 0, as assumed by the GP, and is critical for good accuracy.
        dft_call_threshold
            Uncertainty threshold for whether to call DFT.
        dft_add_threshold
            Uncertainty threshold for whether to add an atom to the training set.
        std_xyz_fname
            Function for the name of the file in which to save ASE Atoms with per-atom uncertainties as charges.
            Takes as input this LMPOTF object and the current step.
        model_fname
            Name of the saved model, must correspond to `pair_coeff`.
        hyperparameter_optimization
            Boolean function that determines whether to run hyperparameter optimization, as a function of this LMPOTF
            object, the LAMMPS instance and the current step.
        opt_bounds
            Bounds for the hyperparameter optimization.
        opt_method
            Algorithm for the hyperparameter optimization.
        opt_iterations
            Max number of iterations for the hyperparameter optimization.
        post_dft_callback
            A function that is called after every DFT call. Receives this LMPOTF object and the current step.
        wandb
            The wandb object, which should already be initialized.
        log_fname
            An output file to which logging info is written.
        """
        self.sparse_gp = sparse_gp
        self.descriptors = np.atleast_1d(descriptors)
        self.rcut = rcut

        self.type2number = np.atleast_1d(type2number)
        self.ntypes = len(self.type2number)

        self.dftcalc = dftcalc
        self.energy_correction = energy_correction
        self.dft_call_threshold = dft_call_threshold
        self.dft_add_threshold = dft_add_threshold
        self.post_dft_callback = post_dft_callback

        self.dft_calls = 0
        self.last_dft_call = -100
        self.std_xyz_fname = std_xyz_fname

        self.model_fname = model_fname

        self.hyperparameter_optimization = hyperparameter_optimization
        self.opt_bounds = opt_bounds
        self.opt_method = opt_method
        self.opt_iterations = opt_iterations

        self.wandb = wandb

        logging.basicConfig(filename=log_fname, level=logging.DEBUG)
        self.logger = logging.getLogger("lmpotf")

    def save(self, fname):
        self.sparse_gp.write_mapping_coefficients(fname, "LMPOTF", 0)

    def step(self, lmpptr, evflag=0):
        """
        Function called by LAMMPS at every step.
        This is the function that must be called by `fix python/invoke`.

        Parameters
        ----------
        lmpptr : ptr
            Pointer to running LAMMPS instance.
        evflag : int
            evflag given by LAMMPS, ignored.
        """

        try:
            lmp = lammps(ptr=lmpptr)
            natoms = lmp.get_natoms()
            x = lmp.gather_atoms("x", 1, 3)
            x = np.ctypeslib.as_array(x, shape=(natoms, 3)).reshape(natoms, 3)
            step = int(lmp.get_thermo("step"))

            boxlo, boxhi, xy, yz, xz, _, _ = lmp.extract_box()
            cell = np.diag(np.array(boxhi) - np.array(boxlo))
            cell[1, 0] = xy
            cell[2, 0] = xz
            cell[2, 1] = yz

            types = lmp.gather_atoms("type", 0, 1)
            types = np.ctypeslib.as_array(types, shape=(natoms))

            structure = Structure(cell, types - 1, x, self.rcut, self.descriptors)

            if self.dft_calls == 0:
                # Call DFT on the initial structure and add it to the training set

                self.logger.info("Initial step, calling DFT")

                pe, F = self.run_dft(cell, x, types, step, structure)
                self.sparse_gp.add_training_structure(structure)
                self.sparse_gp.add_random_environments(structure, [8])
                self.sparse_gp.update_matrices_QR()
                self.save(self.model_fname)

            else:
                self.logger.info(f"Step {step}")

                # Predict uncertainties, check if any are above the threshold
                sigma = self.sparse_gp.hyperparameters[0]
                self.sparse_gp.predict_local_uncertainties(structure)
                variances = structure.local_uncertainties[0]
                stds = np.sqrt(np.abs(variances)) / sigma

                if self.std_xyz_fname is not None:
                    frame = ase.Atoms(
                        positions=x,
                        numbers=self.type2number[types - 1],
                        cell=cell,
                        pbc=True,
                    )
                    frame.set_array("charges", stds)
                    ase.io.write(self.std_xyz_fname(step), frame, format="extxyz")

                wandb_log = {"max_uncertainty": np.amax(stds)}
                self.logger.info(f"Max uncertainty: {np.amax(stds)}")

                if np.any(stds > self.dft_call_threshold):
                    # Predict the GPs forces and energy to compute errors
                    self.sparse_gp.predict_DTC(structure)
                    predE = structure.mean_efs[0]
                    predF = structure.mean_efs[1:-6].reshape((-1, 3))
                    predS = structure.mean_efs[-6:]
                    Fstd = np.sqrt(np.abs(structure.variance_efs[1:-6])).reshape(
                        (-1, 3)
                    )
                    Estd = np.sqrt(np.abs(structure.variance_efs[0]))
                    Sstd = np.sqrt(np.abs(structure.variance_efs[-6:]))

                    # Get true forces and energy
                    pe, F = self.run_dft(cell, x, types, step, structure)

                    # Add atoms above threshold to training set
                    atoms_to_be_added = np.arange(natoms)[stds > self.dft_add_threshold]
                    self.sparse_gp.add_training_structure(structure)
                    self.sparse_gp.add_specific_environments(
                        structure, atoms_to_be_added
                    )
                    self.sparse_gp.update_matrices_QR()

                    # Save coefficient file
                    self.save(self.model_fname)

                    # Reload model in the LAMMPS pair style
                    lmp.command(f"pair_coeff * * {self.model_fname}")

                    # If requested, optimize hyperparameters
                    if self.hyperparameter_optimization(self, lmp, step):
                        optimize_hyperparameters(
                            self.sparse_gp,
                            bounds=self.opt_bounds,
                            method=self.opt_method,
                            max_iterations=self.opt_iterations,
                        )

                    # Log to Weights and Biases
                    wandb_log["Fmae"] = np.mean(np.abs(F - predF))
                    wandb_log["Emae"] = np.abs(pe - predE) / natoms
                    wandb_log["n_added"] = len(atoms_to_be_added)
                    for qty in "n_added", "Fmae", "Emae":
                        self.logger.info(f"{qty}: {wandb_log[qty]}")
                if self.wandb is not None:
                    wandb_log["uncertainties"] = self.wandb.Histogram(stds)
                    # x_std = np.hstack((x, stds.reshape(-1, 1)))
                    # wandb_log["uncertainties_3d"] = self.wandb.Object3D(x_std)
                    self.wandb.log(wandb_log, step=step)
        except Exception as err:
            self.logger.exception("LMPOTF ERROR")
            raise err

    def run_dft(self, cell, x, types, step, structure):
        atomic_numbers = self.type2number[types - 1]
        frame = ase.Atoms(
            positions=x,
            numbers=atomic_numbers,
            cell=cell,
            calculator=self.dftcalc,
            pbc=True,
        )
        pe = frame.get_potential_energy()
        if self.energy_correction is None:
            self.energy_correction = pe
        pe -= self.energy_correction

        F = frame.get_forces()
        stress = frame.get_stress(voigt=False)

        ase.io.write(self.std_xyz_fname(step), frame, format="extxyz")

        structure.forces = F.reshape(-1)
        structure.energy = np.array([pe])
        # structure.stresses = transform_stress(stress)

        self.dft_calls += 1
        self.last_dft_call = step

        self.post_dft_callback(self, step)

        return pe, F
