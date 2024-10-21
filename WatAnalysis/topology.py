import numpy as np
from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.core.universe import Universe
from MDAnalysis.exceptions import NoDataError


def find_water_residues(
    u: Universe,
    oh_cutoff: float,
    water_resname: str = "WAT",
    oxygen_sel: str = "name O",
    hydrogen_sel: str = "name H",
) -> None:
    """
    Define water residues with the following algorithm.
        1. Appoint each oxygen atom to a new residue.
        2. Appoint each hydrogen atom to the residue of the
           closest oxygen atom.
    This algorithm does not (yet) take into account spontaneous water ionization.

    Parameters
    ----------
    u: MDAnalysis.core.universe.Universe
        Universe object containing water molecules; this object will be modified
        to have residues
    oh_cutoff: float
        Maximum cutoff distance for intramolecular O-H bonds, in Angstrom
    """
    # Add the resname topology attribute
    try:
        u.residues.resnames
    except NoDataError:
        u.add_TopologyAttr("resname")

    # Select hydrogen and oxygen atom groups
    hydrogen_atoms = u.select_atoms(hydrogen_sel)
    oxygen_atoms = u.select_atoms(oxygen_sel)

    # Create a new residue for each oxygen atom
    for o in oxygen_atoms:
        resid = np.max(u.residues.resids) + 1
        res = u.add_Residue(
            resid=resid,
            resname=water_resname,
            resnum=resid,
        )
        o.residue = res

    # Appoint each H to the residue of the closest O
    for h in hydrogen_atoms:
        pairs, distances = capped_distance(
            h.position,
            oxygen_atoms,
            max_cutoff=oh_cutoff,
            box=u.dimensions,
            return_distances=True,
        )

        if len(pairs) > 0:
            smallest_dist_pair = pairs[np.argsort(distances)[0]]
            closest_o = oxygen_atoms[smallest_dist_pair[1]]  # pair: [h_index, o_index]
            h.residue = closest_o.residue


def find_layer_residues(
    u: Universe,
    num_per_layer: int,
    metal_sel: str = "name Pt",
    resname_metal: str = "METAL",
    surface_idx: tuple[int] | None = None,
    resname_surface: str = "SURFACE",
):
    """
    Define different residues for all metal layers in the simulation box. Layers are
    defined in the z-direction. This procedure makes most sense for layers in the x-y
    plane, and might not work as intended for stepped surfaces.
    Optionally, the surface layers are defined by the argument surface_idx; their
    residues are named "SURFACE".
    All metal residues, including the surfaces, are assigned to a new Segment with segid
    <resname_metal>.

    Parameters
    ----------
    u: MDAnalysis.core.universe.Universe
        Universe object containing water molecules; this object will be modified
        to have residues
    num_per_layer: int
        Number of metal atoms per layer, e.g., 20 for a 5x4 surface
    """
    try:
        u.residues.resnames
    except NoDataError:
        u.add_TopologyAttr("resname")

    metal_atoms = u.select_atoms(metal_sel)
    z_coords = metal_atoms.positions[:, 2]
    sorted_ids = np.argsort(z_coords).reshape(-1, num_per_layer)

    seg = u.add_Segment(segid=resname_metal)

    for i, layer_ids in enumerate(sorted_ids):
        name = resname_metal
        if i in np.atleast_1d(surface_idx):
            name = resname_surface

        resid = np.max(u.residues.resids) + 1
        res = u.add_Residue(
            segment=seg,
            resid=resid,
            resname=name,
            resnum=resid,
        )
        layer_atoms = metal_atoms[layer_ids]
        layer_atoms.residues = [res] * len(layer_atoms)
