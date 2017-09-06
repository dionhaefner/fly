import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import netCDF4

import fly


def parse_cli():
    defaults = vars(fly.Fly)
    parser = argparse.ArgumentParser(description="Animate Veros output using fly",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # input settings
    parser.add_argument("NCFILE", help="netCDF4 file containing data")
    parser.add_argument("--time", type=int, required=False, default=-1)
    parser.add_argument("--depth", type=int, required=False, default=-1)
    parser.add_argument("--flow-variables", nargs=2, required=False, default=None)
    parser.add_argument("--shading", choices=("none", "variable", "absvalue"), default="absvalue")
    parser.add_argument("--shading-variable", required=False, default=None)

    # output settings
    parser.add_argument("--num-gridlines", type=int, nargs=2, default=defaults["num_gridlines"])
    parser.add_argument("--colormap", required=False, default=defaults["colormap"])
    parser.add_argument("--background-image", required=False, default=defaults["background_image"])
    parser.add_argument("--bgcolor", nargs=4, type=float, required=False, default=defaults["bgcolor"])
    parser.add_argument("--resolution", nargs=2, type=int, required=False, default=defaults["resolution"])
    parser.add_argument("--num-segments", type=int, required=False, default=defaults["num_segments"])
    parser.add_argument("--markers", required=False, default=defaults["markers"])
    parser.add_argument("--rotate", action="store_true")
    parser.add_argument("--record", action="store_true")
    return vars(parser.parse_args())


def shiftdata(lon, arr):
    lon = lon % 360
    lon[lon > 180] -= 360
    ix = np.argmin(np.abs(lon), axis=0)
    assert len(set(ix)) == 1
    ix = ix[0]
    slon = np.vstack((lon[ix:,:], lon[:ix,:] + 360., lon[ix:ix+1,:] + 360.))
    sarr = np.vstack((arr[ix:,:], arr[:ix,:], arr[ix:ix+1,:]))
    return slon, sarr


def read_markers():
    labels, latitudes, longitudes = [], [], []
    with open(data.get(self.marker_file), 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            label = row[1]
            latitude = np.pi / 2 + float(row[2]) * np.pi / 180.
            longitude = np.pi  + float(row[3]) * np.pi / 180.
            labels.append(label)
            latitudes.append(latitude)
            longitudes.append(longitude)
    return labels, latitudes, longitudes


def interpolate(oldcoords, arr, newcoords):
    x, y = np.meshgrid(*oldcoords, indexing="ij")
    lon, lat = newcoords
    invalid = arr.mask
    arr[invalid] = np.nan
    return scipy.interpolate.griddata((x.flatten(), y.flatten()), arr.flatten(), (lon, lat),
                                      method="nearest", fill_value=0.)


if __name__ == "__main__":
    args = parse_cli()
    lat, lon = np.mgrid[-90:90:500j,-180:180:500j]

    with netCDF4.Dataset(args.pop("NCFILE"), "r") as f:
        x, y = (f.variables[k][...] for k in ("xt", "yt"))
        x = x % 360
        x[x > 180] -= 360
        flow_variables = args.pop("flow_variables")
        if flow_variables:
            vector_field = np.array([interpolate((x,y), f.variables[k][args["time"], args["depth"], :, :].T, (lon, lat)) for k in flow_variables])
            vector_field[np.isnan(vector_field)] = 0.
        else:
            vector_field = None

        shading = args.pop("shading")
        if shading == "absvalue":
            if not flow_variables:
                raise ValueError("must give flow_variables when using absvalue shading")
            scalar_field = np.sqrt(vector_field[0]**2 + vector_field[1]**2)
        elif shading == "variable":
            scalar_field = interpolate((x,y), f.variables[args.pop("shading_variable")][args["time"], args["depth"], :, :].T, (lon, lat))
            scalar_field[np.isnan(scalar_field)] = 0.
            plt.imshow(scalar_field)
            plt.show()
        else:
            scalar_field = None


    fly = fly.Fly(scalar_field, vector_field)
    for setting, value in args.items():
        setattr(fly, setting, value)
    fly.setup()
    fly.run()
