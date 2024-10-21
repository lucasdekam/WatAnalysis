"""
Analysis class(es) for interfacial water based on MDAnalysis.analysis.base
"""

import numpy as np
from ase.geometry import cellpar_to_cell
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core.universe import Universe
from MDAnalysis.core.groups import AtomGroup

from .physics import water_cos_theta, water_density
from .utils import bin_edges_to_grid


def _get_h2o_positions(
    water_ag: AtomGroup,
    hydrogen_ag: AtomGroup,
    oxygen_ag: AtomGroup,
):
    """
    For a collection of water molecules, get the positions of O atoms and
    (separately) the positions of the first and second H atom.

    Requires that the water AtomGroup is partitioned in residues.
    """
    assert (
        len(water_ag.residues) > 1
    ), "Atom group must contain one residue for each water molecule"

    # Initialize lists to store atom positions
    o_atoms, h1_atoms, h2_atoms = [], [], []

    # Iterate over water residues and gather O and H atom positions
    for res in water_ag.residues:
        atoms = res.atoms
        h_atoms = atoms.intersection(hydrogen_ag)

        if len(h_atoms) == 2:
            o_atom = atoms.intersection(oxygen_ag)
            o_atoms.append(o_atom.positions[0])
            h1_atoms.append(h_atoms.positions[0])
            h2_atoms.append(h_atoms.positions[1])

    # Convert the lists into numpy arrays
    o_pos = np.array(o_atoms)
    h1_pos = np.array(h1_atoms)
    h2_pos = np.array(h2_atoms)
    return o_pos, h1_pos, h2_pos


class WaterProfiles(AnalysisBase):
    """
    Parameters
    ----------
    universe : MDAnalysis.core.universe.Universe
        A Universe containing the trajectory
    surface_sel: str = "resname SURFACE"
        Selection string for surface Residues.
    oxygen_sel : str = "name O"
        Selection string for oxygen Atoms
    hydrogen_sel : str = "name H"
        Selection string for hydrogen Atoms
    water_sel : str = "resname WAT"
        Selection string for water molecules
    axis : int
        Axis perpendicular to the surface(s).
    verbose : bool = False
        Turn on more logging and debugging
    """

    def __init__(
        self,
        universe: Universe,
        surface_sel: str = "resname SURFACE",
        water_sel: str = "resname WAT",
        oxygen_sel: str = "name O",
        hydrogen_sel: str = "name H",
        axis: int = 2,
        verbose: bool = False,
    ):
        self.universe = universe
        trajectory = self.universe.trajectory
        super().__init__(trajectory, verbose=verbose)
        self.n_frames = len(trajectory)

        self.axis = axis
        self.ave_axis = np.delete(np.arange(3), self.axis)

        # Selection of atom groups (TODO: check if it is okay to do this in __init__)
        self.water_ag = self.universe.select_atoms(water_sel)
        self.oxygen_ag = self.universe.select_atoms(oxygen_sel)
        self.hydrogen_ag = self.universe.select_atoms(hydrogen_sel)
        self.n_water = len(self.water_ag.residues)

        self.surf_res = self.universe.select_atoms(surface_sel).residues

    def _prepare(self):
        self.z_water = np.zeros((self.n_frames, self.n_water))
        self.geo_dipole_water = np.zeros((self.n_frames, self.n_water))
        self.z_surf = np.zeros((self.n_frames, len(self.surf_res)))

    def _single_frame(self):
        """
        Compute surface position, water density and cos theta for a single frame
        """
        # Surface position
        z_surf = [np.mean(res.atoms.positions[:, self.axis]) for res in self.surf_res]
        np.copyto(self.z_surf[self._frame_index], z_surf)

        # O, H, H positions
        o_pos, h1_pos, h2_pos = _get_h2o_positions(
            self.water_ag,
            self.hydrogen_ag,
            self.oxygen_ag,
        )

        # Water density
        np.copyto(self.z_water[self._frame_index], o_pos[:, self.axis])

        # Water cos theta
        cos_theta = water_cos_theta(
            o_pos, h1_pos, h2_pos, axis=self.axis, box=self._ts.dimensions
        )
        np.copyto(self.geo_dipole_water[self._frame_index], cos_theta)

    def _conclude(self):
        # Surface position
        self.results["z_surf"] = self.z_surf

        # Surface area
        cell = cellpar_to_cell(self.universe.dimensions)
        cross_area = np.linalg.norm(
            np.cross(cell[self.ave_axis[0]], cell[self.ave_axis[1]])
        )

        z_max = self.universe.dimensions[self.axis]

        # water density
        counts, bin_edges = np.histogram(
            self.z_water.flatten(),
            bins=int(z_max / 0.1),
            range=(0, z_max),
        )
        n_water = counts / self.n_frames
        grid_volume = np.diff(bin_edges) * cross_area
        rho = water_density(n_water, grid_volume)
        self.results["rho_water"] = [bin_edges_to_grid(bin_edges), rho]

        # water orientation
        counts, bin_edges = np.histogram(
            self.z_water.flatten(),
            bins=int(z_max / 0.1),
            range=(0, z_max),
            weights=self.geo_dipole_water.flatten(),
        )
        self.results["geo_dipole_water"] = [
            bin_edges_to_grid(bin_edges),
            counts / self.n_frames,
        ]
