import numpy as np

from MDAnalysis.lib.distances import minimize_vectors
from scipy import constants


def water_cos_theta(
    o_pos: np.ndarray,
    h1_pos: np.ndarray,
    h2_pos: np.ndarray,
    axis: int,
    box: np.ndarray,
):
    """
    Calculate cos theta for a collection of water molecules, given the
    positions of the individual atoms
    """
    # Compute bond vector
    bond_vec_1 = minimize_vectors(h1_pos - o_pos, box=box)
    bond_vec_2 = minimize_vectors(h2_pos - o_pos, box=box)
    dipole_vector = bond_vec_1 + bond_vec_2

    # Project dipole vector on main axis
    cos_theta = (dipole_vector[:, axis]) / np.linalg.norm(dipole_vector, axis=-1)
    return cos_theta


def density(n, v, mol_mass: float):
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


def water_density(n, v):
    return density(n, v, 18.015)
