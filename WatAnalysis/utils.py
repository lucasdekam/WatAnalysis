# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
from ase import Atoms
from ase.geometry import get_layers
from MDAnalysis.lib.distances import distance_array
from scipy import constants


def bin_edges_to_grid(bin_edges: np.ndarray):
    """
    Convert bin edges to grid points at bin centers.

    Parameters
    ----------
    bin_edges : np.ndarray
        Array of bin edges.

    Returns
    -------
    np.ndarray
        Array of grid points at bin centers.
    """
    return bin_edges[:-1] + np.diff(bin_edges) / 2


def calc_density(n, v, mol_mass: float):
    """
    calculate density (g/cm^3) from the number of particles

    Parameters
    ----------
    n : int or array
        number of particles
    v : float or array
        volume
    mol_mass : float
        mole mass in g/mol
    """
    rho = (n / constants.Avogadro * mol_mass) / (
        v * (constants.angstrom / constants.centi) ** 3
    )
    return rho


def calc_water_density(n, v):
    """
    Calculate the density of water from the number of particles and volume.

    Parameters
    ----------
    n : int or array
        number of particles
    v : float or array
        volume

    Returns
    -------
    float or array
        Density of water in g/cm^3
    """
    return calc_density(n, v, 18.015)


def identify_water_molecules(
    h_positions: np.ndarray,
    o_positions: np.ndarray,
    box: np.ndarray,
    oh_cutoff: float,
    ignore_warnings: bool = False,
) -> Dict[int, List[int]]:
    """
    Identify water and hydronium species by assigning each H to its nearest O.

    Returns
    -------
    Dict[int, List[int]]
        Dictionary mapping oxygen atom indices to lists of bonded hydrogen atom indices.
        e.g., {O_idx: [H1, H2]} for water, {O_idx: [H1, H2, H3]} for hydronium.
    """
    # Initialize dictionary for all Oxygen indices
    species_dict = {i: [] for i in range(len(o_positions))}

    # Calculate all-to-all distances considering Periodic Boundary Conditions (PBC)
    # Assuming distance_array handles PBC as in your original snippet
    all_distances = np.zeros((h_positions.shape[0], o_positions.shape[0]))
    distance_array(h_positions, o_positions, result=all_distances, box=box)

    for h_idx, distances in enumerate(all_distances):
        # Find the index of the closest Oxygen
        o_idx = np.argmin(distances)
        min_dist = distances[o_idx]

        # Check if the closest O is within a reasonable bonding cutoff
        if min_dist <= oh_cutoff:
            species_dict[o_idx].append(h_idx)
        else:
            if not ignore_warnings:
                warnings.warn(
                    f"Hydrogen {h_idx} is orphaned (nearest O is {min_dist:.2f} A away)."
                )

    if not ignore_warnings:
        not_water = {k: v for k, v in species_dict.items() if len(v) != 2}
        for k, v in not_water.items():
            warnings.warn(
                f"Oxygen {k} has {len(v)} hydrogens near it (not two). Ignoring."
            )
        species_dict = {k: v for k, v in species_dict.items() if len(v) == 2}

    return species_dict


def mic_1d(x: np.ndarray, box_length: float, ref: float = 0.0) -> np.ndarray:
    """
    Apply the minimum image convention to a 1D coordinate in a periodic cell.

    Parameters
    ----------
    x : np.ndarray
        Coordinates to be wrapped.
    box_length : float
        Length of the periodic cell.
    ref : float
        Reference coordinate around which the coordinates are wrapped.

    Returns
    -------
    np.ndarray
        Wrapped coordinates within the first principal cell centered around
        the reference coordinate.
    """
    _x = x - ref
    _x = _x - np.round(_x / box_length) * box_length
    return _x + ref


def exponential_moving_average(data, alpha=0.1):
    """Exponential moving average"""
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema


def get_region_masks(
    z_coords: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    interval: Tuple[Optional[float], Optional[float]],
):
    """
    Generate region masks based on z-coordinates and specified intervals.

    Parameters
    ----------
    z_coords : np.ndarray
        Array of z-coordinates.
    z1 : np.ndarray
        Array of z-coordinates of lower surface.
    z2 : np.ndarray
        Array of z-coordinates of upper surface.
    interval : Tuple[Optional[float], Optional[float]],
        Interval range to create masks.
        If None, the mask will be True for all z-coordinates.

    Returns
    -------
    mask1 : np.ndarray
        Boolean mask where z_coords are within the interval range of z1.
    mask2 : np.ndarray
        Boolean mask where z_coords are within the interval range of z2.
    """
    mask_lo = np.ones_like(z_coords, dtype=bool)
    mask_hi = np.ones_like(z_coords, dtype=bool)
    # add mask for lower bounds
    if interval[0] is not None:
        mask_lo = mask_lo & (z_coords > (z1[:, np.newaxis] + interval[0]))
        mask_hi = mask_hi & (z_coords < (z2[:, np.newaxis] - interval[0]))
    # add mask for upper bounds
    if interval[1] is not None:
        mask_lo = mask_lo & (z_coords <= (z1[:, np.newaxis] + interval[1]))
        mask_hi = mask_hi & (z_coords >= (z2[:, np.newaxis] - interval[1]))
    # mask_lo = (z_coords > (z1[:, np.newaxis] + interval[0])) & (
    #     z_coords <= (z1[:, np.newaxis] + interval[1])
    # )
    # mask_hi = (z_coords < (z2[:, np.newaxis] - interval[0])) & (
    #     z_coords >= (z2[:, np.newaxis] - interval[1])
    # )
    return mask_lo, mask_hi


def guess_surface_indices(
    atoms: Atoms,
    element: str = "Pt",
    tolerance: float = 1.4,
) -> Tuple[List, List]:
    """
    Guess indices of surface atoms on both sides of a slab with surface normals along the
    z-direction.

    Inside the function, the slab is translated until it does not cross the surface boundaries.
    In this way the surfaces with normal vectors pointing up and down are identified in the
    same way consistently.

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object representing the periodic cell structure containing the slab
    element : str, optional
        Chemical symbol of the element to analyze for surface atoms (default is "Pt")
    tolerance : float, optional
        Distance tolerance parameter for layer identification (default is 1.4, ~1/2 the layer
        spacing of Pt in a slab)

    Returns
    -------
    List[List, List]
        A tuple containing two lists:
        - First list contains indices of atoms on the surface with up-pointing normal vector
        - Second list contains indices of atoms on the surface with down-pointing normal vector
    """
    atoms = atoms.copy()
    atoms.wrap()

    def _crosses_z_boundary(slab: Atoms):
        coords_z = slab.get_positions()[:, 2]
        z_diff = coords_z.max() - coords_z.min()
        z_diff_mic = mic_1d(z_diff, box_length=slab.cell[2][2])
        return ~np.isclose(z_diff, z_diff_mic, atol=1e-5, rtol=0)

    while _crosses_z_boundary(atoms[atoms.symbols == element]):
        atoms.translate([0, 0, 1])
        atoms.wrap()

    pt_indices = np.flatnonzero(atoms.symbols == element)
    tags, _ = get_layers(
        atoms[atoms.symbols == element], miller=(0, 0, 1), tolerance=tolerance
    )
    layer_tags = np.unique(tags)

    surf_up = pt_indices[tags == layer_tags[-1]].tolist()
    surf_dw = pt_indices[tags == layer_tags[0]].tolist()
    return surf_up, surf_dw


def calc_pbc_r_max(a, b):
    # calculate area based on cross product
    area = np.cross(a, b)
    area = np.linalg.norm(area)
    base_length = np.linalg.norm(a)
    height = area / base_length
    r_max = min(height, base_length) / 2
    return r_max
