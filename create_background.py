import sys

import numpy as np
import matplotlib.pyplot as plt
import netCDF4


if __name__ == "__main__":
    with netCDF4.Dataset(sys.argv[1], "r") as f:
        x, y = (np.array(f.variables[k]).T for k in ("xu", "yu"))
        u = np.array(f.variables["u"][-1,-1, ...]).T
    x = x % 360
    x[x > 180] -= 360
    i = np.argsort(x)
    x = x[i]
    u = u[i,:]
    u[u < -1e10] = np.nan

    fig = plt.figure(frameon=False, figsize=((x.max() - x.min()) / (y.max() - y.min()), 1), dpi=2000)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_aspect("auto")
    ax.set_axis_off()
    ax.set_xlim((-180,180))
    ax.set_ylim((-90,90))
    fig.add_axes(ax)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mask = np.isnan(u)
    ax.contourf(xx, yy, mask, [0., 0.5, 1.], colors=[".9", "coral"])
    ax.contour(xx, yy, mask, [0.5], colors=".15", linewidths=0.1)
    fig.savefig("bg.png")
