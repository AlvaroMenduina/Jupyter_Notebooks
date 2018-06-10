import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs

MU_EARTH = 3.986e14           # Gravitational Parameter: Earth [m^3/s^2]
R_EARTH = 6371000             # Earth's radius [m]
OMEGA_EARTH = 7.2921159e-5    # Earth's angular speed [rad/s]
N_points = 200

def solve_kepler(M, e):
    """
    Solve Kepler's equation in terms of the Eccentric Anomaly E
            M = E - e sin(E)
    It uses Newton's method
    -----
    Returns
    True Anomaly (Theta)
    Based on the equation
    tan(theta/2) = np.sqrt(1+e)/np.sqrt(1-e) * tan(E/2)
    """
    E0 = M
    err = 1.
    k = 0

    while np.linalg.norm(err) > 1e-6:
        E0 = E0 - (M - E0 + e*np.sin(E0))/(-1. + e*np.cos(E0))
        k += 1
        err = M - E0 + e*np.sin(E0)
        print(k)

    a = np.tan(E0/2)
    b = np.sqrt(1 - e) / np.sqrt(1 + e)
    theta = 2 * np.arctan2(a, b)
    return theta

def orbital_elements(h, e, i, raan, omega):
    """
    Construct the orbital elements
    """
    a = h * 1000 + R_EARTH
    ecc = e
    inc = np.deg2rad(i)
    RAAN = np.deg2rad(raan)
    omega = np.deg2rad(omega)

    T = 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH)
    print('Orbital period: %.1f min' % (T / 60))

    return [a, ecc, inc, RAAN, omega]

def compute_track(elements, theta, periods=1, rotation=False):

    a = elements[0]
    RAAN = elements[3]
    inc, omega = elements[2], elements[4]
    T = 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH)

    lat = np.arcsin(np.sin(inc) * np.sin(omega + theta))
    lon = RAAN + np.arctan2(np.tan(omega + theta), (np.cos(inc)) ** (-1))

    for i in range(periods*N_points - 1):
        l_before, l_after = lon[i], lon[i + 1]
        if np.abs(l_before - l_after) > np.pi / 2:
            lon[i + 1:] += np.pi

    new_lon = np.rad2deg(lon)
    new_lat = np.rad2deg(lat)

    # Effect of Earth's rotation
    t = np.linspace(0, periods*T, periods*N_points)
    extra_omega = np.rad2deg(OMEGA_EARTH * t)
    if rotation:
        new_lon -= extra_omega

    return new_lon, new_lat

theta = np.linspace(0, 2*np.pi, N_points)

""" Effect of Earth's rotation """
orbit_1 = orbital_elements(h=(26600e3-R_EARTH)/1e3, e=0.0, i=62.8, raan=0, omega=270)
lon1, lat1 = compute_track(orbit_1, theta, rotation=True)
# lon2, lat2 = compute_track(orbit_1, theta, rotation=True)
lon2, lat2 = compute_track(orbit_1, theta_new, rotation=True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1,
                     projection=ccrs.PlateCarree())
# ax.scatter(lon1, lat1, color='black', s=5, label="No rotation")
ax.scatter(lon2, lat2, color='red', s=5, label="Earth's Rotation")
plt.legend()
ax.stock_img()
ax.gridlines()
ax.coastlines()
plt.title("Effect of Earth's rotation on the Ground Track")

""" Several periods """
per = 3
orbit_0 = orbital_elements(h=1000, e=0.0, i=60, raan=0, omega=0)
theta0 = np.linspace(0, 2*np.pi*per, per*N_points)
lon0, lat0 = compute_track(orbit_1, theta0, periods=per, rotation=True)
colors = cm.Greys(np.linspace(0, 1, len(theta0)))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1,
                     projection=ccrs.PlateCarree())
ax.scatter(lon0, lat0, color=colors, s=5)
ax.stock_img()
ax.gridlines()
ax.coastlines()
plt.title("%d Periods" %(per))

""" Effect of Inclination """
i2, i3 = 45, 60
orbit_2 = orbital_elements(h=600, e=0.0, i=i2, raan=0, omega=0)
orbit_3 = orbital_elements(h=600, e=0.0, i=i3, raan=0, omega=0)
lon2, lat2 = compute_track(orbit_2, theta, rotation=True)
lon3, lat3 = compute_track(orbit_3, theta, rotation=True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1,
                     projection=ccrs.PlateCarree())
ax.scatter(lon2, lat2, color='black', s=5, label=i2)
ax.scatter(lon3, lat3, color='red', s=5, label=i3)
plt.legend(title='Inclination [deg]')
ax.stock_img()
ax.gridlines()
ax.coastlines()
plt.title("Effect of orbit inclination")

""" Effect of RAAN """
raan1, raan2 = 0, 90
orbit_4 = orbital_elements(h=600, e=0.0, i=45, raan=raan1, omega=0)
orbit_5 = orbital_elements(h=600, e=0.0, i=45, raan=raan2, omega=0)
lon4, lat4 = compute_track(orbit_4, theta, rotation=True)
lon5, lat5 = compute_track(orbit_5, theta, rotation=True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1,
                     projection=ccrs.PlateCarree())
ax.scatter(lon4, lat5, color='black', s=5, label=raan1)
ax.scatter(lon5, lat5, color='red', s=5, label=raan2)
plt.legend(title='RAAN [deg]')
ax.stock_img()
ax.gridlines()
ax.coastlines()
plt.title("Effect of Right Ascension of Ascending Node")

plt.show()