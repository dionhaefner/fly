import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pygrib

def read_wind_data(path):
    grbs = pygrib.open(path)
    grbs.seek(0)
    grb = grbs.readline()
    lat, lon = grb.latlons()
    U = grb.values
    grb = grbs.readline()
    V = grb.values
    grbs.close()
    return lon, lat, U, V

proj = ccrs.PlateCarree()
ax = plt.axes(projection=proj)
ax.coastlines()
ax.stock_img()
every = 20
x, y, u, v = read_wind_data("wind_data.grib")
print(x)
ax.quiver(x[::every,::every],y[::every,::every],u[::every,::every],v[::every,::every])
plt.savefig("quiver-map.png")

plt.figure()
every = 20
y, x = np.indices(u.shape)
x = np.flipud(x)
y = np.flipud(y)
plt.quiver(x[::every,::every],y[::every,::every],u[::every,::every],v[::every,::every])
plt.savefig("quiver-naive.png")
