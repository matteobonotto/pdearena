import numpy as np 
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import matplotlib.pyplot as plt

def contour_gif(
        data,
        animation_name = None,
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
        z = data[i,:,:]
        # for c in cont.collections:
        #     c.remove()  # removes only the contours, leaves the rest intact
        cont = plt.contourf(x, y, z, cvals)
        plt.title('t = %i' % (i))
        return cont

    anim = FuncAnimation(fig, animate, frames=Nt, repeat=True)
    if animation_name is not None:
        anim.save(animation_name, writer=FFMpegWriter())
    return anim
