from flare import struc, env, otf_parser
from flare.lammps import lammps_calculator
import datetime
import numpy as np

# define cell, positions, and species
cell = 100 * np.eye(3)
separation = 4.0
positions = np.array([[0, 0, 0], [separation, 0, 0]])
atom_types = [1]
atom_masses = [108]
atom_species = [1, 1]
structure = struc.Structure(cell, atom_species, positions)

# set up input and data files
data_file_name = 'tmp.data'
lammps_location = 'beta.txt'
style_string = 'flare'
coeff_string = '* * {}'.format(lammps_location)
lammps_executable = '$lmp'
dump_file_name = 'tmp.dump'
input_file_name = 'tmp.in'
output_file_name = 'tmp.out'

# write data file
data_text = lammps_calculator.lammps_dat(structure, atom_types,
                                         atom_masses, atom_species)
lammps_calculator.write_text(data_file_name, data_text)

# write input file
input_text = \
    lammps_calculator.generic_lammps_input(data_file_name, style_string,
                                           coeff_string, dump_file_name)
lammps_calculator.write_text(input_file_name, input_text)

# run lammps
lammps_calculator.run_lammps(lammps_executable, input_file_name,
                             output_file_name)