import numpy as np
import time
from ase import io
import MDAnalysis as mda
from MDAnalysis import transformations as trans

from zjxpack.postprocess.metal import ECMetal

from WatAnalysis.parallel import parallel_exec
from WatAnalysis.dielectric import ParallelInverseDielectricConstant as PIDC
from WatAnalysis.dielectric import InverseDielectricConstant as IDC

dim = [11.246, 11.246, 35.94, 90, 90, 90]

atoms = io.read("/data/jxzhu/2022_leiden/02.nnp_validation/input/coord.xyz")
atoms.set_cell(dim)
atoms = ECMetal(atoms, metal_type="Pt", surf_atom_num=16)
surf_ids = atoms.get_surf_idx()
# print(surf_ids)

# load trajectory
u = mda.Universe(
    "/data/jxzhu/2022_leiden/02.nnp_validation/input/interface.psf",
    "/data/jxzhu/2022_leiden/02.nnp_validation/input/raw_aimd.xyz")
transform = trans.boxdimensions.set_dimensions(dim)
u.trajectory.add_transformations(transform)

start = time.time()
task = PIDC(
    universe=u,
    bins=np.arange(10),
    axis="z",
    temperature=330,
    make_whole=False,
    verbose=True,
    surf_ids=surf_ids,
    c_ag="name O",
    select_all=True,
)
parallel_exec(task.run, 0, 30000, 1, 4)
np.save("parallel_inveps.npy", task.results["inveps"])
fmt = "\nParallel Work Completed! Used Time: {:.3f} seconds"
print(fmt.format(time.time() - start))

start = time.time()
task = IDC(
    universe=u,
    bins=np.arange(10),
    axis="z",
    temperature=330,
    make_whole=False,
    verbose=True,
    surf_ids=surf_ids,
    c_ag="name O",
    select_all=True,
)
task.run()
np.save("serial_inveps.npy", task.results.inveps)
fmt = "\nSerial Work Completed! Used Time: {:.3f} seconds"
print(fmt.format(time.time() - start))