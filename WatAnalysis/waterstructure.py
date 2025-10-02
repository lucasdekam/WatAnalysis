# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Functionality for computing time-averaged water structure properties from
molecular dynamics trajectories of water at interfaces
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from MDAnalysis.lib.distances import minimize_vectors

from . import utils


def calc_water_dipoles(
    h_positions: np.ndarray,
    o_positions: np.ndarray,
    water_dict: Dict[int, List[int]],
    box: np.ndarray,
    mic: bool = True,
) -> np.ndarray:
    """
    Calculate dipole moments for water molecules.

    Parameters
    ----------
    h_positions : np.ndarray
        Positions of hydrogen atoms.
    o_positions : np.ndarray
        Positions of oxygen atoms.
    water_dict : Dict[int, List[int]]
        Dictionary mapping oxygen atom indices to two bonded hydrogen atom indices.
    box : np.ndarray
        Simulation cell defining periodic boundaries.

    Returns
    -------
    np.ndarray
        Array of dipole vectors for each oxygen atom. Entries are NaN for non-water oxygen atoms.
    """
    o_indices = np.array([k for k in water_dict.keys()])
    h1_indices = np.array([v[0] for v in water_dict.values()])
    h2_indices = np.array([v[1] for v in water_dict.values()])

    oh1_vectors = h_positions[h1_indices] - o_positions
    oh2_vectors = h_positions[h2_indices] - o_positions

    if mic:
        oh1_vectors = minimize_vectors(oh1_vectors, box)
        oh2_vectors = minimize_vectors(oh2_vectors, box)

    dipoles = np.ones(o_positions.shape) * np.nan
    dipoles[o_indices, :] = oh1_vectors + oh2_vectors
    return dipoles


def calc_density_profile(
    z_surf: Tuple[float, float],
    z_water: np.ndarray,
    cross_area: float,
    n_frames: int,
    dz: float = 0.1,
    sym: bool = False,
    mol_mass: float = 18.015,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the density profile of water along the z-axis.

    Parameters
    ----------
    z_surf : Tuple[float, float]
        The positions of the left and right surfaces.
    z_water : np.ndarray
        The z-coordinates of water molecules.
    cross_area : float
        The cross-sectional area perpendicular to the z-axis.
    n_frames : int
        Number of trajectory frames from which water molecules were counted.
    dz : float, optional
        The bin width for the histogram (default is 0.1).
    sym : bool, optional
        If True, symmetrize the density profile (default is False).
    mol_mass : float
        Molecular mass of the atoms in g/mol

    Returns
    -------
    z : np.ndarray
        The spatial coordinates along the z-axis.
    rho : np.ndarray
        The density values corresponding to the z-coordinates.
    """
    # Make histogram
    counts, bin_edges = np.histogram(
        z_water.flatten(),
        bins=int((z_surf[1] - z_surf[0]) / dz),
        range=z_surf,
    )

    # Spatial coordinates
    z = utils.bin_edges_to_grid(bin_edges)

    # Density values
    n_water = counts / n_frames
    grid_volume = np.diff(bin_edges) * cross_area
    rho = utils.calc_density(n_water, grid_volume, mol_mass)
    if sym:
        rho = (rho[::-1] + rho) / 2
    return z, rho


def calc_orientation_profile(
    z_surf: Tuple[float, float],
    z_water: np.ndarray,
    cos_theta: np.ndarray,
    cross_area: float,
    dz: float,
    sym: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the orientation profile of water molecules.

    Parameters
    ----------
    z_surf : Tuple[float, float]
        The positions of the left and right surface.
    z_water : np.ndarray
        The z-coordinates of water molecules.
    cos_theta : np.ndarray
        The cosine of the angle between the water molecule dipole and the z-axis.
    cross_area : float
        The cross-sectional area of the system.
    dz : float
        The bin width in the z-direction.
    sym : bool, optional
        If True, symmetrize the orientation profile (default is False).

    Returns
    -------
    z : np.ndarray
        The z-coordinates of the grid points.
    rho_cos_theta : np.ndarray
        The orientation profile of water molecules.
    """
    # Only include non-nan dipole values (only O with 2 H)
    valid = ~np.isnan(cos_theta.flatten())

    counts, bin_edges = np.histogram(
        z_water.flatten()[valid],
        bins=int((z_surf[1] - z_surf[0]) / dz),
        range=z_surf,
        weights=cos_theta.flatten()[valid],
    )

    z = utils.bin_edges_to_grid(bin_edges)
    n_water = counts / z_water.shape[0]
    grid_volume = np.diff(bin_edges) * cross_area
    rho_cos_theta = utils.calc_water_density(n_water, grid_volume)

    if sym:
        rho_cos_theta = (rho_cos_theta - rho_cos_theta[::-1]) / 2
    return z, rho_cos_theta


def calc_costheta_profile(
    z_surf: Tuple[float, float],
    z_water: np.ndarray,
    cos_theta: np.ndarray,
    dz: float,
    sym: bool = False,
):
    """
    Calculate the profile of the average cosine of the angle theta as a function of z-coordinate.

    Parameters
    ----------
    z_surf : Tuple[float, float]
        The positions of the left and right surface.
    z_water : np.ndarray
        The z-coordinates of the water molecules.
    cos_theta : np.ndarray
        The cosine of the angle theta for each water molecule.
    dz : float
        The bin width for z-coordinates.
    sym : bool, optional
        If True, symmetrize the profile by averaging with its reverse (default is False).

    Returns
    -------
    z : np.ndarray
        The midpoints of the bins along the z-axis.
    avg_cos_theta : np.ndarray
        The average cosine of the angle theta in each bin.
    """
    # Only include non-nan dipole values (only O with 2 H)
    valid = ~np.isnan(cos_theta.flatten())

    bin_edges = np.linspace(z_surf[0], z_surf[1], int((z_surf[1] - z_surf[0]) / dz) + 1)
    z = utils.bin_edges_to_grid(bin_edges)

    # Digitize z-coordinates to find which bin each value belongs to
    bin_indices = np.digitize(z_water.flatten()[valid], bin_edges)

    # Compute average cos(theta) in each bin
    avg_cos_theta = np.array(
        [
            (
                cos_theta.flatten()[valid][bin_indices == i].mean()
                if np.any(bin_indices == i)
                else np.nan
            )
            for i in range(1, len(bin_edges))
        ]
    )

    if sym:
        avg_cos_theta = (avg_cos_theta - avg_cos_theta[::-1]) / 2
    return z, avg_cos_theta


def count_water_in_region(
    z1: np.ndarray,
    z2: np.ndarray,
    z_water: np.ndarray,
    interval: Tuple[float, float],
    mask: Optional[np.ndarray] = None,
):
    """
    Count the number of water molecules in a specified region.

    Parameters
    ----------
    z1 : np.ndarray
        Array of z-coordinates for the lower surface.
    z2 : np.ndarray
        Array of z-coordinates for the upper surface.
    z_water : np.ndarray
        Array of z-coordinates for water molecules.
    mask : np.ndarray
        Boolean array for values of z_water to include.
    interval : Tuple[float, float]
        Tuple specifying the interval for the region.

    Returns
    -------
    n_water : np.ndarray
        Array containing the count of water molecules in the specified region.
    """
    if mask is None:
        mask = np.ones(z_water.shape, dtype=bool)

    mask_lo, mask_hi = utils.get_region_masks(z_water, z1, z2, interval)
    mask_lo = mask_lo & mask
    mask_hi = mask_hi & mask

    n_water = np.count_nonzero(mask_lo, axis=1) + np.count_nonzero(mask_hi, axis=1)
    return n_water


def calc_angular_distribution(
    mask_lo: np.ndarray,
    mask_hi: np.ndarray,
    cos_theta: np.ndarray,
    n_bins: int = 90,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the angular distribution of water molecules between two surfaces.

    Parameters
    ----------
    mask_lo : np.ndarray
        Boolean array masking the values of cos_theta in the relevant region near the lower surface
    mask_hi : np.ndarray
        Boolean array masking the values of cos_theta in the relevant region near the upper surface
    cos_theta : np.ndarray
        Array of cosine of the angles between water molecules and the surface normal.
    n_bins : int, optional
        Number of bins for the histogram (default is 90).

    Returns
    -------
    grid : np.ndarray
        Grid of bin centers.
    angle_distribution : np.ndarray
        Normalized histogram of angular distribution.
    """
    lower_surface_angles = np.arccos(cos_theta[mask_lo].flatten()) / np.pi * 180
    upper_surface_angles = np.arccos(-cos_theta[mask_hi].flatten()) / np.pi * 180

    combined_angles = np.concatenate([lower_surface_angles, upper_surface_angles])
    angle_distribution, bin_edges = np.histogram(
        combined_angles,
        bins=n_bins,
        range=(0.0, 180.0),
        density=True,
    )
    grid = utils.bin_edges_to_grid(bin_edges)
    return grid, angle_distribution
