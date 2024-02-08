# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import os
import sys
sys.path.append(os.getcwd())
import h5py
import numpy as np
import torch
from joblib import Parallel, delayed
from phi.flow import (  # SoftGeometryMask,; Sphere,; batch,; tensor,
    Box,
    CenteredGrid,
    Noise,
    StaggeredGrid,
    advect,
    diffuse,
    extrapolation,
    fluid,
)
from phi.math import reshaped_native
from phi.math import seed as phi_seed
from tqdm import tqdm
import time
from pdearena import utils

from pde import PDEConfig

logger = logging.getLogger(__name__)


idx = 0
s = 314101025
pde_nx, pde_ny, pde_Lx, pde_Ly, pde_dt = 128, 128, 32., 32., 1.5
pde_nt = 56
pde_skip_nt = 8
pde_buoyancy_y = 0.5
pde_nu = 0.01

# def genfunc(idx, s):
phi_seed(idx + s)
smoke = abs(
    CenteredGrid(
        Noise(scale=11.0, smoothness=6.0),
        extrapolation.BOUNDARY,
        x=pde_nx,
        y=pde_ny,
        bounds=Box[0 : pde_Lx, 0 : pde_Ly],
    )
)  # sampled at cell centers

velocity = StaggeredGrid(
    0, extrapolation.ZERO, x=pde_nx, y=pde_ny, bounds=Box[0 : pde_Lx, 0 : pde_Ly]
)  # sampled in staggered form at face centers

fluid_field_ = []
velocity_ = []
for i in range(0, pde_nt + pde_skip_nt):
    t0 = time.time()
    smoke = advect.semi_lagrangian(smoke, velocity, pde_dt)
    buoyancy_force = (smoke * (0, pde_buoyancy_y)).at(velocity)  # resamples smoke to velocity sample points
    velocity = advect.semi_lagrangian(velocity, velocity, pde_dt) + pde_dt * buoyancy_force
    velocity = diffuse.explicit(velocity, pde_nu, pde_dt)
    velocity, _ = fluid.make_incompressible(velocity)
    fluid_field_.append(reshaped_native(smoke.values, groups=("x", "y", "vector"), to_numpy=True))
    velocity_.append(
        reshaped_native(
            velocity.staggered_tensor(),
            groups=("x", "y", "vector"),
            to_numpy=True,
        )
    )
    t_elapsed = time.time() - t0
    print('iteration {} of {}, elapsed time {:.4}s'.format(
        i,pde_nt + pde_skip_nt,t_elapsed
    ))

fluid_field_ = np.asarray(fluid_field_[pde_skip_nt :]).squeeze()
# velocity has the shape [nt, nx+1, ny+2, 2]
velocity_corrected_ = np.asarray(velocity_[pde_skip_nt :]).squeeze()[:, :-1, :-1, :]
# return fluid_field_[:: pde_sample_rate, ...], velocity_corrected_[:: pde_sample_rate, ...]
import matplotlib.pyplot as plt
for i in range(fluid_field_.shape[0]):
    plt.imshow(fluid_field_[i,...])

plt.imshow(velocity_corrected_[-1,:,:,0])

'''


with utils.Timer() as gentime:
    rngs = np.random.randint(np.iinfo(np.int32).max, size=num_samples)
    fluid_field, velocity_corrected = zip(
        *Parallel(n_jobs=n_parallel)(delayed(genfunc)(idx, rngs[idx]) for idx in tqdm(range(num_samples)))
    )
with utils.Timer() as gentime:
    fluid_field, velocity_corrected = genfunc(0, rngs[0]) 

logger.info(f"Took {gentime.dt:.3f} seconds")

with utils.Timer() as writetime:
    for idx in range(num_samples):
        # fmt: off
        # Saving the trajectories
        h5f_u[idx : (idx + 1), ...] = fluid_field[idx]
        h5f_vx[idx : (idx + 1), ...] = velocity_corrected[idx][..., 0]
        h5f_vy[idx : (idx + 1), ...] = velocity_corrected[idx][..., 1]
        # fmt:on
        xcoord[idx : (idx + 1), ...] = np.asarray([np.linspace(0, pde_Lx, pde_nx)])
        dx[idx : (idx + 1)] = pde_dx
        ycoord[idx : (idx + 1), ...] = np.asarray([np.linspace(0, pde_Ly, pde_ny)])
        dy[idx : (idx + 1)] = pde_dy
        tcoord[idx : (idx + 1), ...] = np.asarray([np.linspace(pde_tmin, pde_tmax, pde_trajlen)])
        dt[idx : (idx + 1)] = pde_dt * pde_sample_rate
        buo_y[idx : (idx + 1)] = pde_buoyancy_y

logger.info(f"Took {writetime.dt:.3f} seconds writing to disk")

print()
print("Data saved")
print()
print()
h5f.close()
'''




