# Arthur Hagopian <arthur.hagopian@umontpellier.fr>, version 01/19/2024

# INPUT FILE NEEDED : "trajectories.xyz"
# Should contain trajectories of the whole simulation (concatenated file)

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from pathlib import Path
from scipy.constants import Avogadro
import sys
from MDAnalysis.lib.distances import calc_angles
import math
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import math

# DEFINITIONS
type_surface_atoms = "Pt"
num_surface_atoms = 20
x_min = 0
y_min = 0
z_min = 0
x_max = 13.859251
y_max = 9.602000
z_max = 38.316067
thickness_water = 27
start_analysis = int(5000 / 5)
stop_analysis = int(15000 / 5)
num_bins = 100

# Define universe
u = mda.Universe("trajectories_wrapped.xyz")
# Define atoms
o_atoms = u.select_atoms("name O")
h_atoms = u.select_atoms("name H")
# Define electrode surface atoms
electrode_surface_atoms = u.select_atoms(f"name {type_surface_atoms}")[
    -num_surface_atoms:
]


# Initialize variables
num_timesteps = len(u.trajectory[start_analysis:stop_analysis])
angles_bissectrice_counts = np.zeros(num_bins)

# Iterate over each timestep : define electrode surface position
electrode_surface_atoms_avg_per_frame = []
for ts in u.trajectory[start_analysis:stop_analysis]:

    electrode_surface_atoms_positions = electrode_surface_atoms.positions[
        :, 2
    ]  # Z coordinates
    electrode_surface_atoms_avg_per_frame.append(
        sum(electrode_surface_atoms_positions) / len(electrode_surface_atoms_positions)
    )


electrode_surface_atoms_avg = sum(electrode_surface_atoms_avg_per_frame) / len(
    electrode_surface_atoms_avg_per_frame
)
print(f"Electrode surface position : {electrode_surface_atoms_avg}")

bin_edges = np.linspace(0, 14, num_bins + 1)


# Iterate over each timestep
o_atoms_counts = np.zeros(num_bins)
dipoles_per_bin = np.zeros(num_bins)
counter = []
for ts in u.trajectory[start_analysis:stop_analysis]:

    o_atom_zs = []
    dipoles = []
    dic_counter = {}

    # Collect atom position
    o_atoms_positions = o_atoms.positions
    h_atoms_positions = h_atoms.positions

    # Define water molecules
    water_molecules = []
    distance_from_electrode = []
    o_indice = 0
    for o in o_atoms:
        # Find o atoms
        o_atom_z = o.position[2]
        if o_atom_z <= electrode_surface_atoms_avg + thickness_water / 2:
            distance_from_electrode.append(o_atom_z - electrode_surface_atoms_avg)
        else:
            distance_from_electrode.append(
                electrode_surface_atoms_avg + thickness_water - o_atom_z
            )
        # Find water molecules
        water_molecule = []
        h1_indice = o_indice * 2
        h2_indice = h1_indice + 1
        water_molecule.append(o_indice)
        water_molecule.append(h1_indice)
        water_molecule.append(h2_indice)
        water_molecules.append(water_molecule)
        # Search for bissectrice
        o_x = o_atoms_positions[water_molecule[0]][0]
        o_y = o_atoms_positions[water_molecule[0]][1]
        o_z = o_atoms_positions[water_molecule[0]][2]
        h1_x = h_atoms_positions[water_molecule[1]][0]
        h1_y = h_atoms_positions[water_molecule[1]][1]
        h1_z = h_atoms_positions[water_molecule[1]][2]
        h2_x = h_atoms_positions[water_molecule[2]][0]
        h2_y = h_atoms_positions[water_molecule[2]][1]
        h2_z = h_atoms_positions[water_molecule[2]][2]
        dx = np.abs(h1_x - h2_x)
        dy = np.abs(h1_y - h2_y)
        dz = np.abs(h1_z - h2_z)
        # PBC
        if dx > x_max * 0.5:
            h2_x = h_atoms_positions[water_molecule[2]][0] - x_max
        if dy > y_max * 0.5:
            h2_y = h_atoms_positions[water_molecule[2]][1] - y_max
        # if (dz > z_max * 0.5):
        #     h2_z = h_atoms_positions[water_molecule[2]][2] - z_max
        bissectrice_point = np.array(
            [(h1_x + h2_x) / 2, (h1_y + h2_y) / 2, (h1_z + h2_z) / 2]
        )
        if o_atom_z <= electrode_surface_atoms_avg + thickness_water / 2:
            z_point_bottle = np.array([o_x, o_y, o_z + 1])
            angle_bissectrice_bottle = calc_angles(
                bissectrice_point, o_atoms_positions[water_molecule[0]], z_point_bottle
            )
            dipole = math.cos(angle_bissectrice_bottle)
            dipoles.append(dipole)
        else:
            z_point_top = np.array([o_x, o_y, o_z - 1])
            angle_bissectrice_top = calc_angles(
                bissectrice_point, o_atoms_positions[water_molecule[0]], z_point_top
            )
            dipole = math.cos(angle_bissectrice_top)
            dipoles.append(dipole)

        o_indice += 1

    # Bin the different angles
    o_atoms_bin_indices = np.searchsorted(
        bin_edges, distance_from_electrode
    )  # Associate positions to bin indices
    o_atoms_counts += np.bincount(
        o_atoms_bin_indices, minlength=num_bins
    )  # Add 1 every time a bin is occupied
    # print(o_atoms_counts)
    i = 0
    while i < len(o_atoms_bin_indices):
        if o_atoms_bin_indices[i] in counter:
            dipoles_per_bin[o_atoms_bin_indices[i]] += dipoles[i]
        else:
            dipoles_per_bin[o_atoms_bin_indices[i]] = dipoles[i]
        counter.append(o_atoms_bin_indices[i])
        i += 1

# Calculate the concentration per bin
num_h2o_per_ang3 = (
    55.5 * 1e-27 * Avogadro
)  # Molarity of water = 55.5 mol.L-1 && 1 ang^3 = 1e-27 L
num_particles_per_mol_per_ang3 = Avogadro * 1e-27
volume_bin = (
    (bin_edges[1] - bin_edges[0]) * (x_max - x_min) * (y_max - y_min)
)  # In ang^3
num_h2o_per_bin = volume_bin * num_h2o_per_ang3

# Averaging
avg_o_atoms_dipoles = (
    dipoles_per_bin
    / num_timesteps
    / 2
    / volume_bin
    / num_particles_per_mol_per_ang3
    * 18.01528
    / 1000
)

avg_o_atoms_counts = o_atoms_counts / num_timesteps / 2
o_atoms_concentration = (
    (avg_o_atoms_counts / volume_bin) / num_particles_per_mol_per_ang3 * 18.01528 / 1000
)

# Define x-axis
bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

step_ad_H2O_counts = 0
terrace_ad_H2O_counts = 0
# adsorbed water in terrace and step region
for i in range(len(bin_centers)):
    if bin_centers[i] <= 3.05:
        terrace_ad_H2O_counts += avg_o_atoms_counts[i]

print(f"adsorbed water counts at step sites : {step_ad_H2O_counts}")
print(f"adsorbed water counts at terrace sites : {terrace_ad_H2O_counts}")

# i = 0
# for o in o_atoms_concentration:
#     weight = o
#     position = i
#     if position in [int(elt) for elt in list(dic_z_dipole.keys())]:
#         o_atoms_concentration[i] = dic_z_dipole[position]
#     else:
#         o_atoms_concentration[i] = 0
#     i += 1

### PRINT IN water_density.dat ###
o_concent_bulk = []
bulk_water_width = 14
water_thickness = 27
# water_bulk_region_min = water_region_middle - water_width/2
# water_bulk_region_max = water_region_middle + water_width/2
file_out = open("water_density.dat", "w")
i = 0
while i < len(o_atoms_concentration):
    file_out.write(
        "%7.5f" % (bin_centers[i])
        + "    "
        + "%7.5f" % (o_atoms_concentration[i])
        + "\n"
    )
    if (
        water_thickness / 2
        > bin_centers[i]
        > water_thickness / 2 - bulk_water_width / 2
    ):
        o_concent_bulk.append(o_atoms_concentration[i])
    i += 1
ave_o_concent_bulk = sum(o_concent_bulk) / len(o_concent_bulk)
file_out.close()
print("... Water density data printed in file : water_density.dat\n")
print(f"water bulk concentration : {ave_o_concent_bulk}")

file_out = open("water_dipole.dat", "w")
i = 0
while i < len(avg_o_atoms_dipoles):
    file_out.write(
        "%7.5f" % (bin_centers[i]) + "    " + "%7.5f" % (avg_o_atoms_dipoles[i]) + "\n"
    )
    i += 1
file_out.close()
print("... Water dipole data printed in file : water_dipole.dat\n")
