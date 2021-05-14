import numpy as np
from ase.io import read


def convert_xyz(file_in, file_out, species_map):
    frames = read(file_in, index=":")
    with open(file_out, "w") as f:
        for frame in frames:
            n_atoms = len(frame)
            f.write(f"{n_atoms}\n")
            cell = frame.cell
            energy = frame.get_potential_energy()
            forces = frame.get_forces()
            stress = -frame.get_stress()[[0, 5, 4, 1, 3, 2]]
            sparse_inds = frame.get_info("sparse_indices")
            f.write(
                f"{cell[0, 0]} {cell[0, 1]} {cell[0, 2]} {cell[1, 0]} {cell[1, 1]} {cell[1, 2]} {cell[2, 0]} {cell[2, 1]} {cell[2, 2]} {energy} {stress[0]} {stress[1]} {stress[2]} {stress[3]} {stress[4]} {stress[5]}\n"
            )
            for i in range(n_atoms):
                coded_symbol = species_map[frame.symbol[i]]
                pos = frame.positions[i]
                f.write(f"{coded_symbol} {pos[0]} {pos[1]} {pos[2]} {forces[i, 0]} {forces[i, 1]} {forces[i, 2]}\n")

if __name__ == "__main__":
    convert_xyz("dft.xyz", "data.xyz", {"Si": 0, "C": 1})
