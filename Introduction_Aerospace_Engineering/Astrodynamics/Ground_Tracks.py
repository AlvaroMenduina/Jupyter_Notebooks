import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs

MU_EARTH = 3.986e14  # Gravitational Parameter: Earth [m^3/s^2]
R_EARTH = 6371000  # Earth's radius [m]
OMEGA_EARTH = 7.2921159e-5  # Earth's angular speed [rad/s]
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
        E0 = E0 - (M - E0 + e * np.sin(E0)) / (-1. + e * np.cos(E0))
        k += 1
        err = M - E0 + e * np.sin(E0)
        print(k)

    a = np.tan(E0 / 2)
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

def compute_track(elements, periods=1, rotation=False):
    """
    Compute the Ground Track of a Satellite given the Orbital Elements
    ----------
    Options:
    Period: how many orbital periods you want to compute
    Rotation: if True it accounts for Earth's rotation
    """

    a, e = elements[0], elements[1]
    RAAN = elements[3]
    inc, omega = elements[2], elements[4]
    T = 2 * np.pi * np.sqrt(a ** 3 / MU_EARTH)

    # Construct the Mean Anomaly
    M = np.linspace(0, 2*np.pi*periods, int(N_points*periods))
    # Solve Kepler's equation
    theta = solve_kepler(M, e)
    for i in range(M.shape[0] - 1):
        before, after = theta[i], theta[i+1]
        if np.abs(before - after) > np.pi / 2:
            theta[i + 1:] += 2*np.pi

    # Compute Latitude and Longitude
    lat = np.arcsin(np.sin(inc) * np.sin(omega + theta))
    lon = RAAN + np.arctan2(np.tan(omega + theta), (np.cos(inc)) ** (-1))

    # Fix Pi jumps in the Longitude
    for i in range(M.shape[0] - 1):
        l_before, l_after = lon[i], lon[i + 1]
        if np.abs(l_before - l_after) > np.pi / 2:
            lon[i + 1:] += np.pi

    new_lon = np.rad2deg(lon)
    new_lat = np.rad2deg(lat)

    # Effect of Earth's rotation
    t = np.linspace(0, periods*T, int(periods*N_points))
    extra_omega = np.rad2deg(OMEGA_EARTH * t)
    if rotation:
        new_lon -= extra_omega

    return new_lon, new_lat

if __name__ == "__main__":

    # ================================================================================== #
    #                               Influence of Earth's rotation                        #
    # ================================================================================== #

    leo = orbital_elements(h=2500, e=0.0, i=45, raan=0, omega=0)
    lon_no_rot, lat_no_rot = compute_track(leo, periods=1, rotation=False)
    lon_rot, lat_rot = compute_track(leo, periods=1, rotation=True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.PlateCarree())
    ax.scatter(lon_no_rot, lat_no_rot, color='black', s=5, label="No rotation")
    ax.scatter(lon_rot, lat_rot, color='red', s=5, label="Earth's Rotation")
    plt.legend()
    ax.stock_img()
    ax.gridlines()
    ax.coastlines()
    plt.title("Effect of Earth's rotation on the Ground Track")
    plt.show()

    # ================================================================================== #
    #                                  Several orbital periods                           #
    # ================================================================================== #

    per = 3
    leo_per = orbital_elements(h=1000, e=0.0, i=45, raan=0, omega=0)
    lon_per, lat_per = compute_track(leo, periods=per, rotation=True)
    colors = cm.Greys(np.linspace(0, 1, N_points*per))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.PlateCarree())
    ax.scatter(lon_per, lat_per, color=colors, s=5)
    ax.stock_img()
    ax.gridlines()
    ax.coastlines()
    plt.title("%d Periods" % (per))

    # ================================================================================== #
    #                                  Effect of Inclination                             #
    # ================================================================================== #

    i1, i2 = 45, 60
    inc1 = orbital_elements(h=600, e=0.0, i=i1, raan=0, omega=0)
    inc2 = orbital_elements(h=600, e=0.0, i=i2, raan=0, omega=0)
    lon_in1, lat_in1 = compute_track(inc1, periods=1, rotation=True)
    lon_in2, lat_in2 = compute_track(inc2, periods=1, rotation=True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.PlateCarree())
    ax.scatter(lon_in1, lat_in1, color='black', s=5, label=i1)
    ax.scatter(lon_in2, lat_in2, color='red', s=5, label=i2)
    plt.legend(title='Inclination [deg]')
    ax.stock_img()
    ax.gridlines()
    ax.coastlines()
    plt.title("Effect of Orbit Inclination")

    # ================================================================================== #
    #                                   Effect of RAAN                                   #
    # ================================================================================== #

    raan1, raan2 = 0, 90
    r1 = orbital_elements(h=600, e=0.0, i=45, raan=raan1, omega=0)
    r2 = orbital_elements(h=600, e=0.0, i=45, raan=raan2, omega=0)
    lon_r1, lat_r1 = compute_track(r1, periods=1, rotation=True)
    lon_r2, lat_r2 = compute_track(r2, periods=1, rotation=True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1,
                         projection=ccrs.PlateCarree())
    ax.scatter(lon_r1, lat_r1, color='black', s=5, label=raan1)
    ax.scatter(lon_r2, lat_r2, color='red', s=5, label=raan2)
    plt.legend(title='RAAN [deg]')
    ax.stock_img()
    ax.gridlines()
    ax.coastlines()
    plt.title("Effect of Right Ascension of the Ascending Node")

    # ================================================================================== #
    #                                    Geosynchronous                                  #
    # ================================================================================== #

    T_earth = 2 * np.pi / OMEGA_EARTH
    a_synch = (MU_EARTH * (T_earth/2/np.pi)**2)**(1/3)
    h_synch = (a_synch - R_EARTH)/1e3

    e_g1, i_g1, omega_g1 = 0.2, 45, 0
    label_g1 = 'e=%.1f, i=%.1f, omega=%.1f' %(e_g1, i_g1, omega_g1)
    geo_synch_1 = orbital_elements(h=h_synch, e=e_g1, i=i_g1, raan=0, omega=omega_g1)
    lon_synch_1, lat_synch_1 = compute_track(geo_synch_1, periods=1, rotation=True)

    e_g2, i_g2, omega_g2 = 0.2, 45, 45
    label_g2 = 'e=%.1f, i=%.1f, omega=%.1f' %(e_g2, i_g2, omega_g2)
    geo_synch_2 = orbital_elements(h=h_synch, e=e_g2, i=i_g2, raan=0, omega=omega_g2)
    lon_synch_2, lat_synch_2 = compute_track(geo_synch_2, periods=1, rotation=True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.scatter(lon_synch_1, lat_synch_1, color='black', s=5, label=label_g1)
    ax.scatter(lon_synch_2, lat_synch_2, color='red', s=5, label=label_g2)
    plt.legend(title='Orbital parameters')
    ax.stock_img()
    ax.gridlines()
    ax.coastlines()
    plt.title("Geosynchronous orbit")

    # ================================================================================== #
    #                                    Molniya Orbit                                   #
    # ================================================================================== #

    h_mol, e_mol, i_mol, raan_mol, omega_mol = (26600e3-R_EARTH)/1e3, 0.74, 62.8, 75, 45
    mol_label = "h=%.1f [km], e=%.1f, i=%.1f [deg], RAAN=%.1f [deg], omega=%.1f [deg]" %(h_mol, e_mol, i_mol, raan_mol, omega_mol)
    molniya = orbital_elements(h=h_mol, e=e_mol, i=i_mol, raan=raan_mol, omega=omega_mol)
    lon_mol, lat_mol = compute_track(molniya, periods=2, rotation=True)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.scatter(lon_mol, lat_mol, color='black', s=5)
    ax.stock_img()
    ax.gridlines()
    ax.coastlines()
    plt.title("Molniya orbit: " + mol_label)

    plt.show()