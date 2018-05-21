import numpy as np
import matplotlib.pyplot as plt

""" ASTRO CONSTANTS """
R_SUN = 695508          # Solar radius [km]
M_SUN = 2e30            # Solar mass [kg]
R_JUP = 69911           # Jupiter radius [km]
M_JUP = 1.89e27         # Juputer mass [kg]
AU = 149597870700       # AU in [m]

""" PARAMETERS """
# Host Star: 51 Pegasi
M_s = 1.11              # Solar Masses
R_s = 1.237             # Solar Radii

# Exoplanet: 51 Peg b
M_p = 0.472             # Jupiter Masses
R_p = 1.9               # Jupiter Radii
a = 0.0527              # Semi-major axis [AU]
e = 0.0052
P = 4.230785            # Orbital period [days]

# Radial velocity
K = 55.94               # Semi-amplitude [m/s]

# Observations
N_transit = 1000         # Data points per transit plot
N_radial = 15           # Data points in Radial Velocity
eps_a = 1e-3
eps_b = 1e-3

if __name__ == "__main__":

    # Ratio of Planet size / Star size
    RpRs = R_p * R_JUP / (R_s * R_SUN)
    F_transit = 1 - RpRs**2

    # Transit duration
    theta = R_SUN * 1e3 / (a * AU) / np.pi
    # theta = 45. / 360 #FIXME
    t_transit = theta * P

    # Ideal Fluxes transit
    times = np.linspace(0., P, N_transit)
    raw_flux = np.ones_like(times)
    t_ingress = P/2. - t_transit/2
    t_egress = P/2. + t_transit/2
    for i, t in enumerate(times):
        if (t_ingress < t < t_egress):
            raw_flux[i] = F_transit

    # Noisy
    noisy_flux = raw_flux * np.random.uniform(low=(1-eps_a), high=(1+eps_a), size=N_transit)
    noisy_flux += np.random.uniform(low=(-eps_b), high=(eps_a), size=N_transit)

    tt = np.linspace(0., 3*P, 3*N_transit)
    ff = np.concatenate([noisy_flux, noisy_flux, noisy_flux])

    plt.figure()
    plt.scatter(times, noisy_flux, s=3)
    plt.ylabel('Flux ratio')
    plt.xlabel('Days')

    plt.figure()
    plt.scatter(tt, ff, s=3)
    plt.ylabel('Flux ratio')
    plt.xlabel('Days')

    # Save Transit data
    data = np.concatenate((times[:, np.newaxis], noisy_flux[:, np.newaxis]), axis=-1)
    np.savetxt('transit.txt', data)

    # RV measurements
    times_rv = 2*np.pi * np.linspace(0, P, 100)
    rv = K * np.sin(times_rv)

    rv_rand = np.random.choice(100, size=N_radial)

    plt.figure()
    plt.plot(times_rv, rv, linestyle='--')
    plt.errorbar(times_rv[rv_rand], rv[rv_rand],
                 yerr=5, xerr=1, fmt='o', ecolor='black')
    plt.ylabel('RV')

    plt.show()


