import glob
from pathlib import Path
from multiprocessing import Pool, cpu_count

import click
import numpy as np
from ase.io import read
import MDAnalysis as mda
from MDAnalysis.transformations import boxdimensions

from WatAnalysis.utils import guess_surface_indices
from WatAnalysis.analysis import WaterAnalysis


def process_trajectory(
    traj_file: str,
    interface: tuple[float, float],
    n_blocks: int,
    dipole_params: tuple[int, int, int],
    survival_params: tuple[int, int, int],
    ion_species: str | None,
    output: str,
):
    """Run density + orientation + dynamics workflow for one trajectory."""

    traj = Path(traj_file)
    subdir = traj.parent
    data_file = subdir / "system.data"

    if not data_file.exists():
        click.echo(f"[WARNING] No system.data in {subdir}, skipping.")
        return

    click.echo(f"[INFO] Processing {traj}")

    atoms = read(data_file, format="lammps-data")

    # Map element symbols to LAMMPS type index
    symbol_map = {symbol: i + 1 for i, symbol in enumerate(np.unique(atoms.symbols))}

    # Surface detection
    surf_ids = guess_surface_indices(atoms=atoms, element="Au", tolerance=1.4)

    # Build MDAnalysis universe
    u = mda.Universe(
        str(data_file),
        str(traj),
        topology_format="DATA",
        format="XTC",
        transformations=[boxdimensions.set_dimensions(atoms.cell.cellpar())],
        atom_style="id type x y z",
    )

    # Species selection
    species_sels = []
    if ion_species is not None:
        if ion_species not in symbol_map:
            click.echo(
                f"[WARNING] Species '{ion_species}' not found, skipping ion density."
            )
        else:
            species_sels.append(f"type {symbol_map[ion_species]}")

    # Water analysis object
    task = WaterAnalysis(
        u,
        surf_ids=surf_ids,
        species_sels=species_sels,
        oxygen_sel=f"type {symbol_map['O']}",
        hydrogen_sel=f"type {symbol_map['H']}",
        verbose=False,
    )

    task.run()

    results = {}

    # Profiles
    results["density"] = task.density_profile(n_blocks=n_blocks)
    results["orientation"] = task.orientation_profile(n_blocks=n_blocks)
    results["costheta"] = task.costheta_profile()

    # Region counts
    results["water_count"] = task.count_in_region(interface)

    # Angular distributions
    results["angular_distribution"] = task.angular_distribution(interface)

    # Dipole autocorrelation
    tau_dip, c_dip = task.dipole_autocorrelation(
        max_tau=dipole_params[0],
        delta_tau=dipole_params[1],
        interval=interface,
        step=dipole_params[2],
    )
    results["dipole_autocorrelation"] = (tau_dip, c_dip)

    # Survival probability
    tau_sp, sp = task.survival_probability(
        max_tau=survival_params[0],
        delta_tau=survival_params[1],
        interval=interface,
        step=survival_params[2],
    )
    results["survival_probability"] = (tau_sp, sp)

    # Ion density profile
    if species_sels:
        results[f"density_{ion_species}"] = task.species_density_profile(
            species_sels[0], n_blocks=n_blocks
        )

    # Total dipole
    results["total_dipole"] = task.total_dipole()

    # Save
    outfile = subdir / output
    np.savez(outfile, **results)
    click.echo(f"[INFO] Saved results to {outfile}")


@click.command()
@click.option(
    "--pattern",
    required=True,
    help="Glob for trajectory files (e.g. './*/pos_traj.xtc').",
)
@click.option(
    "--nprocs",
    default=1,
    type=int,
    show_default=True,
    help="Number of parallel processes.",
)
@click.option(
    "--interface",
    nargs=2,
    type=float,
    required=True,
    help="Interface bounds (lower upper), z relative to surface.",
)
@click.option(
    "--n-blocks",
    default=10,
    type=int,
    show_default=True,
    help="Number of blocks for block averaging.",
)
@click.option(
    "--dipole-max-tau",
    default=250,
    type=int,
    show_default=True,
    help="Maximum correlation time for dipole autocorrelation, in number of timesteps.",
)
@click.option(
    "--dipole-delta-tau",
    default=1,
    type=int,
    show_default=True,
    help="Resolution of the dipole autocorrelation curve, in number of timesteps.",
)
@click.option(
    "--dipole-step",
    default=250,
    type=int,
    show_default=True,
    help="Step size (number of timesteps) between correlation time origins. If equal to max-tau, there is "
    "no overlap between time windows.",
)
@click.option(
    "--survival-max-tau",
    default=500,
    type=int,
    show_default=True,
    help="Maximum correlation time for survival probability, in number of timesteps.",
)
@click.option(
    "--survival-delta-tau",
    default=5,
    type=int,
    show_default=True,
    help="Resolution of the survival probability curve, in number of timesteps.",
)
@click.option(
    "--survival-step",
    default=500,
    type=int,
    show_default=True,
    help="Step size (number of timesteps) between correlation time origins. If equal to max-tau, there is "
    "no overlap between time windows.",
)
@click.option(
    "--ion",
    default=None,
    type=str,
    help="Optional ionic species (e.g. Na).",
    show_default=True,
)
@click.option(
    "--output",
    default="watanalysis_results.npz",
    type=str,
    help="Output NPZ filename",
    show_default=True,
)
def main(
    pattern,
    nprocs,
    interface,
    n_blocks,
    dipole_max_tau,
    dipole_delta_tau,
    dipole_step,
    survival_max_tau,
    survival_delta_tau,
    survival_step,
    ion,
    output,
):
    """
    WatAnalysis CLI

    Run water structure and dynamics analysis on folders containing:
    - system.data
    - xtc trajectory
    """

    traj_files = sorted(glob.glob(pattern))

    if len(traj_files) == 0:
        click.echo("[ERROR] No trajectory files found.")
        return

    click.echo(f"[INFO] Found {len(traj_files)} trajectory files")

    dipole_params = (dipole_max_tau, dipole_delta_tau, dipole_step)
    survival_params = (survival_max_tau, survival_delta_tau, survival_step)

    if nprocs > 1:
        nprocs = min(nprocs, cpu_count(), len(traj_files))
        click.echo(f"[INFO] Running in parallel with {nprocs} processes")

        with Pool(processes=nprocs) as pool:
            pool.starmap(
                process_trajectory,
                [
                    (
                        tf,
                        interface,
                        n_blocks,
                        dipole_params,
                        survival_params,
                        ion,
                        output,
                    )
                    for tf in traj_files
                ],
            )
    else:
        for tf in traj_files:
            process_trajectory(
                tf,
                interface,
                n_blocks,
                dipole_params,
                survival_params,
                ion,
            )


if __name__ == "__main__":
    main()
