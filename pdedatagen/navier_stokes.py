# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import os

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

from pdearena import utils

from .pde import PDEConfig

logger = logging.getLogger(__name__)


def generate_trajectories_smoke(
    pde: PDEConfig,
    mode: str,
    num_samples: int,
    batch_size: int,
    device: torch.device = torch.device("cpu"),
    dirname: str = "data",
    n_parallel: int = 1,
    seed: int = 42,
) -> None:
    """
    Generate data trajectories for smoke inflow in bounded domain
    Args:
        pde (PDEConfig): pde at hand [NS2D]
        mode (str): [train, valid, test]
        num_samples (int): how many trajectories do we create
        batch_size (int): batch size
        device: device (cpu/gpu)
    Returns:
        None
    """

    pde_string = str(pde)
    logger.info(f"Equation: {pde_string}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Number of samples: {num_samples}")

    save_name = os.path.join(dirname, "_".join([pde_string, mode, str(seed), f"{pde.buoyancy_y:.5f}"]))
    if mode == "train":
        save_name = save_name + "_" + str(num_samples)
    h5f = h5py.File("".join([save_name, ".h5"]), "a")
    dataset = h5f.create_group(mode)

    tcoord, xcoord, ycoord, dt, dx, dy = {}, {}, {}, {}, {}, {}
    h5f_u, h5f_vx, h5f_vy = {}, {}, {}

    nt, nx, ny = pde.grid_size[0], pde.grid_size[1], pde.grid_size[2]
    # The scalar field u, the components of the vector field vx, vy,
    # the coordinations (tcoord, xcoord, ycoord) and dt, dx, dt are saved
    h5f_u = dataset.create_dataset("u", (num_samples, nt, nx, ny), dtype=float)
    h5f_vx = dataset.create_dataset("vx", (num_samples, nt, nx, ny), dtype=float)
    h5f_vy = dataset.create_dataset("vy", (num_samples, nt, nx, ny), dtype=float)
    tcoord = dataset.create_dataset("t", (num_samples, nt), dtype=float)
    dt = dataset.create_dataset("dt", (num_samples,), dtype=float)
    xcoord = dataset.create_dataset("x", (num_samples, nx), dtype=float)
    dx = dataset.create_dataset("dx", (num_samples,), dtype=float)
    ycoord = dataset.create_dataset("y", (num_samples, ny), dtype=float)
    dy = dataset.create_dataset("dy", (num_samples,), dtype=float)
    buo_y = dataset.create_dataset("buo_y", (num_samples,), dtype=float)

    def genfunc(idx, s):
        phi_seed(idx + s)
        smoke = abs(
            CenteredGrid(
                Noise(scale=11.0, smoothness=6.0),
                extrapolation.BOUNDARY,
                x=pde.nx,
                y=pde.ny,
                bounds=Box[0 : pde.Lx, 0 : pde.Ly],
            )
        )  # sampled at cell centers
        velocity = StaggeredGrid(
            0, extrapolation.ZERO, x=pde.nx, y=pde.ny, bounds=Box[0 : pde.Lx, 0 : pde.Ly]
        )  # sampled in staggered form at face centers
        fluid_field_ = []
        velocity_ = []
        for i in range(0, pde.nt + pde.skip_nt):
            smoke = advect.semi_lagrangian(smoke, velocity, pde.dt)
            buoyancy_force = (smoke * (0, pde.buoyancy_y)).at(velocity)  # resamples smoke to velocity sample points
            velocity = advect.semi_lagrangian(velocity, velocity, pde.dt) + pde.dt * buoyancy_force
            velocity = diffuse.explicit(velocity, pde.nu, pde.dt)
            velocity, _ = fluid.make_incompressible(velocity)
            fluid_field_.append(reshaped_native(smoke.values, groups=("x", "y", "vector"), to_numpy=True))
            velocity_.append(
                reshaped_native(
                    velocity.staggered_tensor(),
                    groups=("x", "y", "vector"),
                    to_numpy=True,
                )
            )

        fluid_field_ = np.asarray(fluid_field_[pde.skip_nt :]).squeeze()
        # velocity has the shape [nt, nx+1, ny+2, 2]
        velocity_corrected_ = np.asarray(velocity_[pde.skip_nt :]).squeeze()[:, :-1, :-1, :]
        return fluid_field_[:: pde.sample_rate, ...], velocity_corrected_[:: pde.sample_rate, ...]

    with utils.Timer() as gentime:
        rngs = np.random.randint(np.iinfo(np.int32).max, size=num_samples)
        fluid_field, velocity_corrected = zip(
            *Parallel(n_jobs=n_parallel)(delayed(genfunc)(idx, rngs[idx]) for idx in tqdm(range(num_samples)))
        )
    with utils.Timer() as gentime:
        rngs = np.random.randint(np.iinfo(np.int32).max, size=num_samples)
        fluid_field, velocity_corrected = genfunc(0, rngs[0]) 

    ###
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    data = fluid_field
    Lx, Ly = pde.Lx, pde.Ly

    def contour_gif(
            data,
            animation_name = 'animation.mp4',
            Lx = 1,
            Ly = 1,
            ):
        global anim
        x,y = np.meshgrid(
            np.linspace(0,Lx,data.shape[1]),
            np.linspace(0,Lx,data.shape[2])
        )
        Nt = data.shape[0]

        fig = plt.figure()
        ax = plt.axes(xlim=(0, Lx), ylim=(0, Ly), xlabel='x', ylabel='y')
        cvals = np.linspace(0,data.max(),50)      # set contour values 
        cont = plt.contourf(x, y, data[0,:,:], cvals)    # first image on screen
        plt.colorbar()

        # animation function
        def animate(i):
            global cont
            z = data[i,:,:]
            for c in cont.collections:
                c.remove()  # removes only the contours, leaves the rest intact
            cont = plt.contourf(x, y, z, cvals)
            plt.title('t = %i:  %.2f' % (i,z[5,5]))
            return cont

        anim = FuncAnimation(fig, animate, frames=Nt, repeat=False)
        anim.save(animation_name, writer=FFMpegWriter())
        return anim

    anim = contour_gif(data)

    ###
    import matplotlib.pyplot as plt
    for i in range(fluid_field.shape[0]):
        plt.imshow(fluid_field[i,...])
        plt.show()
    plt.imshow(fluid_field[-1,:,:])

    fig,ax = plt.subplots()
    def animate(i):
        ax.clear()
        plot =  ax.imshow(fluid_field[i,:,:])
        return plot

    ani = FuncAnimation(fig, animate, interval=40, blit=True, repeat=True, frames=100)    
    ani.save("TLI.gif", dpi=300, writer=PillowWriter(fps=25))

    import imageio
    images = []
    for i in range(fluid_field.shape[0]):
        images.append(fluid_field[i,...])
    imageio.mimsave('movie.gif', images)

    logger.info(f"Took {gentime.dt:.3f} seconds")

    with utils.Timer() as writetime:
        for idx in range(num_samples):
            # fmt: off
            # Saving the trajectories
            h5f_u[idx : (idx + 1), ...] = fluid_field[idx]
            h5f_vx[idx : (idx + 1), ...] = velocity_corrected[idx][..., 0]
            h5f_vy[idx : (idx + 1), ...] = velocity_corrected[idx][..., 1]
            # fmt:on
            xcoord[idx : (idx + 1), ...] = np.asarray([np.linspace(0, pde.Lx, pde.nx)])
            dx[idx : (idx + 1)] = pde.dx
            ycoord[idx : (idx + 1), ...] = np.asarray([np.linspace(0, pde.Ly, pde.ny)])
            dy[idx : (idx + 1)] = pde.dy
            tcoord[idx : (idx + 1), ...] = np.asarray([np.linspace(pde.tmin, pde.tmax, pde.trajlen)])
            dt[idx : (idx + 1)] = pde.dt * pde.sample_rate
            buo_y[idx : (idx + 1)] = pde.buoyancy_y

    logger.info(f"Took {writetime.dt:.3f} seconds writing to disk")

    print()
    print("Data saved")
    print()
    print()
    h5f.close()
