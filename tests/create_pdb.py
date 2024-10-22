"""
Create topology file from an extended xyz trajectory file
"""

from ase.io import read
import MDAnalysis as mda
from MDAnalysis import transformations as trans
from WatAnalysis.topology import find_water_residues, find_layer_residues

xyz_files = [
    "C:/Users/kamlbtde/LocalFiles/lorentz_data/jinwen/lucas.xyz",
    "C:/Users/kamlbtde/LocalFiles/lorentz_data/arthur/lucas.xyz",
    "C:/Users/kamlbtde/LocalFiles/lorentz_data/melander/lucas.xyz",
    "C:/Users/kamlbtde/LocalFiles/lorentz_data/jiaxin/jiaxin.xyz",
]
num_per_layer_list = [20, 20, 36, 36]
surface_idx_list = [[2, 3], [3], [4], [1, 2]]

for xyz, num, surf_idx in zip(xyz_files, num_per_layer_list, surface_idx_list):
    atoms = read(xyz, format="extxyz", index="-1")

    u = mda.Universe(
        xyz, transformations=trans.boxdimensions.set_dimensions(atoms.cell.cellpar())
    )

    find_water_residues(u, oh_cutoff=2)
    find_layer_residues(u, num_per_layer=num, surface_idx=surf_idx)

    with mda.Writer(xyz[:-4] + ".pdb", n_atoms=u.atoms.n_atoms) as writer:
        sorted_atoms = u.atoms.residues.atoms  # Group atoms by residues
        # This is crucial otherwise H,H,O do not end up in the same residue
        # upon loading
        writer.write(sorted_atoms)
