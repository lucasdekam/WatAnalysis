# SPDX-License-Identifier: LGPL-3.0-or-later
"""
Functionality for computing time-averaged water structure properties from
molecular dynamics trajectories of water at interfaces
"""

from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import scipy.constants as C
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
    dz: float = 0.1,
    sym: bool = False,
    mol_mass: float = 18.015,
    n_blocks: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the density profile of water along the z-axis using block averaging.

    Parameters
    ----------
    z_surf : Tuple[float, float]
        The positions of the left and right surfaces.
    z_water : np.ndarray
        The z-coordinates of water molecules in the shape (n_frames, n_atoms).
    cross_area : float
        The cross-sectional area perpendicular to the z-axis.
    dz : float, optional
        The bin width for the histogram (default is 0.1).
    sym : bool, optional
        If True, symmetrize the density profile (default is False).
    mol_mass : float
        Molecular mass of the atoms in g/mol
    n_blocks : int, optional
        Number of blocks for averaging (default is 1).

    Returns
    -------
    z : np.ndarray
        The spatial coordinates along the z-axis.
    rho_mean : np.ndarray
        The mean density values corresponding to the z-coordinates.
    rho_stderr : np.ndarray
        The standard error of the density values.
    """
    n_frames = z_water.shape[0]
    block_size = n_frames // n_blocks
    n_bins = int((z_surf[1] - z_surf[0]) / dz)

    rho_blocks = []

    # Calculate density for each block
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size if i < n_blocks - 1 else n_frames
        z_block = z_water[start_idx:end_idx]

        counts, bin_edges = np.histogram(
            z_block,
            bins=n_bins,
            range=z_surf,
        )

        n_water = counts / (end_idx - start_idx)
        grid_volume = np.diff(bin_edges) * cross_area
        rho = utils.calc_density(n_water, grid_volume, mol_mass)

        if sym:
            rho = (rho[::-1] + rho) / 2

        rho_blocks.append(rho)

    z = utils.bin_edges_to_grid(bin_edges)
    rho_blocks = np.array(rho_blocks)
    rho_mean = rho_blocks.mean(axis=0)
    rho_stderr = np.zeros(rho_mean.shape)
    if n_blocks > 1:
        rho_stderr = rho_blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)

    return z, rho_mean, rho_stderr


def calc_density_profile_reweighting(
    z: np.ndarray,
    z_surf: Tuple[float, float],
    cross_area: float,
    e0: np.ndarray,
    enew: np.ndarray,
    dz: float = 0.1,
    sym: bool = False,
    mol_mass: float = 18.015,
    method: Literal["cea", "boltzmann"] = "cea",
    temperature: float = 300.0,
):
    """
    Calculate density profile using either direct Boltzmann or Cumulant
    Expansion Approximation (CEA) reweighting methods.

    Parameters
    ----------
    z : np.ndarray
        Atomic z-coordinates with shape (n_frames, n_atoms).
    z_surf : Tuple[float, float]
        Range (min, max) for histogram binning in z-direction.
    cross_area : float
        Cross-sectional area for density calculation.
    e0 : np.ndarray
        Energies from original simulation with shape (n_frames,).
    enew : np.ndarray
        Energies from potential by which reweighting is done with shape (n_frames,).
    dz : float, optional
        Bin width for histogram in z-direction. Default is 0.1.
    sym : bool, optional
        If True, symmetrize the density profile. Default is False.
    mol_mass : float, optional
        Molecular mass in g/mol. Default is 18.015 (water).
    method : Literal["cea", "boltzmann"], optional
        Reweighting method: "cea" for Cumulant Expansion Approximation or
        "boltzmann" for Boltzmann reweighting. Default is "cea".
    temperature : float, optional
        Temperature in Kelvin. Default is 300.0.

    Returns
    -------
    grid : np.ndarray
        Grid points corresponding to bin centers with shape (n_bins,).
    rho_reweighted : np.ndarray
        Reweighted density profile with shape (n_bins,). Units: g/cm^3
        or mol/cm^3 if mol_mass=1.
    """
    assert e0.shape == enew.shape, "E0 and E_new should have the same shape"
    assert len(e0) == z.shape[0], "len(E0) should be equal to z.shape[0]"

    beta = C.elementary_charge / (C.Boltzmann * temperature)

    # --- 1. unweighted histogram a_t(b)
    n_bins = int((z_surf[1] - z_surf[0]) / dz)

    # counts per bin per frame
    all_counts = []
    for z_frame in z:  # shape (n_atoms,)
        counts, bin_edges = np.histogram(z_frame, bins=n_bins, range=z_surf)
        all_counts.append(counts)
    a_tb = np.array(all_counts)  # shape (n_frames, n_bins) --> indices (t, b)
    grid = utils.bin_edges_to_grid(bin_edges)

    # --- 2. averages over frames
    a_mean = a_tb.mean(axis=0)  # unweighted histogram <a_b>_E0; shape (n_bins,)

    # --- 3. prepare reweighting
    delta_e = enew - e0

    if method == "boltzmann":
        weights = np.exp(-beta * delta_e)  # shape (n_frames,)
        weights /= np.mean(weights)
        n_reweighted = np.mean(a_tb * weights[:, None], axis=0)
    elif method == "cea":
        de_mean = delta_e.mean()  # <(E-E0)>_E0, scalar
        a_de_mean = (a_tb * delta_e[:, None]).mean(
            axis=0
        )  # <a_b(E-E0)>_E0; shape (n_bins,)
        n_reweighted = a_mean - beta * (a_de_mean - a_mean * de_mean)
    else:
        raise ValueError("method must be 'boltzmann' or 'cea'.")

    grid_volume = np.diff(bin_edges) * cross_area
    rho_reweighted = utils.calc_density(n_reweighted, grid_volume, mol_mass)
    if sym:
        rho_reweighted = (rho_reweighted[::-1] + rho_reweighted) / 2

    return grid, rho_reweighted


def calc_orientation_profile(
    z_surf: Tuple[float, float],
    z_water: np.ndarray,
    cos_theta: np.ndarray,
    cross_area: float,
    dz: float,
    sym: bool = False,
    n_blocks: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the orientation profile of water molecules using block averaging.

    Parameters
    ----------
    z_surf : Tuple[float, float]
        The positions of the left and right surface.
    z_water : np.ndarray
        The z-coordinates of water molecules with shape (n_frames, n_atoms).
    cos_theta : np.ndarray
        The cosine of the angle between the water molecule dipole and the z-axis; shape (n_frames, n_atoms)
    cross_area : float
        The cross-sectional area of the system.
    dz : float
        The bin width in the z-direction.
    sym : bool, optional
        If True, symmetrize the orientation profile (default is False).
    n_blocks : int, optional
        Number of blocks for averaging (default is 1).

    Returns
    -------
    z : np.ndarray
        The z-coordinates of the grid points.
    rho_cos_theta_mean : np.ndarray
        The mean orientation profile of water molecules.
    rho_cos_theta_stderr : np.ndarray
        The standard error of the orientation profile.
    """
    n_frames = z_water.shape[0]
    block_size = n_frames // n_blocks
    n_bins = int((z_surf[1] - z_surf[0]) / dz)

    rho_blocks = []

    # Calculate orientation profile for each block
    for i in range(n_blocks):
        start_idx = i * block_size
        end_idx = (i + 1) * block_size if i < n_blocks - 1 else n_frames
        z_block = z_water[start_idx:end_idx]
        cos_theta_block = cos_theta[start_idx:end_idx]

        # Only include non-nan dipole values
        valid = ~np.isnan(cos_theta_block.flatten())

        counts, bin_edges = np.histogram(
            z_block.flatten()[valid],
            bins=n_bins,
            range=z_surf,
            weights=cos_theta_block.flatten()[valid],
        )

        n_water = counts / (end_idx - start_idx)
        grid_volume = np.diff(bin_edges) * cross_area
        rho_cos_theta = utils.calc_water_density(n_water, grid_volume)

        if sym:
            rho_cos_theta = (rho_cos_theta - rho_cos_theta[::-1]) / 2

        rho_blocks.append(rho_cos_theta)

    z = utils.bin_edges_to_grid(bin_edges)
    rho_blocks = np.array(rho_blocks)
    rho_cos_theta_mean = rho_blocks.mean(axis=0)
    rho_cos_theta_stderr = np.zeros(rho_cos_theta_mean.shape)
    if n_blocks > 1:
        rho_cos_theta_stderr = rho_blocks.std(axis=0, ddof=1) / np.sqrt(n_blocks)

    return z, rho_cos_theta_mean, rho_cos_theta_stderr


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
