import numpy as np
from numpy.random import RandomState
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import zern_core as zern


def compute_rms(phase_array):
    mean = np.mean(phase_array)
    rms = np.sqrt(1. / N ** 2 * np.sum((phase_array - mean) ** 2))

    return rms


def solve_triangular(Z):
    """
    Solves the equation of Triangular Numbers T_n
    T_n = n (n + 1) / 2
    for T_n = N_zern
    """

    n = int(0.5 * (-1 + np.sqrt(1 + 8 * Z)))

    return n


def decay_factor(n, ratio=100):
    """
    Computes how much you need to rescale the coefficients
    at each Zernike row such that after "n" levels your magnitude
    has been reduced by a "ratio"
    """

    log_10_alfa = -np.log10(ratio) / (n - 1)
    alfa = 10 ** (log_10_alfa)

    return alfa


def generate_decay_vector(n, decay_alfa):
    """
    Create a vector of length Z containing
    the required scaling
    """
    vec = [1]
    for i in range(1, n):
        new_vec = [decay_alfa ** i for _ in range(i + 1)]
        vec.extend(new_vec)
    return np.array(vec)


if __name__ == "__main__":

    # Parameters
    N =1024
    N_zern = 100
    randgen = RandomState(1234)

    coef = randgen.normal(size=N_zern)
    print('First 10 Zernike coefficients')
    print(coef[:10])

    # --------------------------------------------------------------------
    """ Plot the Wavefront in Polar coordinates """

    rho_1 = np.linspace(0.0, 1.0, N, endpoint=True)
    theta_1 = np.linspace(0.0, 2 * np.pi, N)

    rr, tt = np.meshgrid(rho_1, theta_1)
    rho, theta = rr.flatten(), tt.flatten()

    z = zern.ZernikeNaive(mask=np.ones((N, N)))
    phase = z(coef=coef, rho=rho, theta=theta, normalize_noll=False, mode='Jacobi', print_option=None)
    phase_map = phase.reshape((N, N))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(phase_map, origin='lower', cmap='viridis', extent=[0, 1, 0, 2 * np.pi])
    ax.set_aspect('auto')
    plt.xlabel(r'Radius $\rho$')
    plt.ylabel(r'Angle $\theta$')
    # plt.title('Wavefront map in Polar Coordinates')

    # --------------------------------------------------------------------
    """ Plot the Wavefront in Cartesian coordinates """

    xx = rr * np.cos(tt)
    yy = rr * np.sin(tt)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = ax.contourf(xx, yy, phase_map, 100, cmap='seismic')
    ax.set_aspect('equal')
    # plt.title('Wavefront map in Cartesian Coordinates')
    plt.xlabel(r'$X$')
    plt.ylabel(r'$Y$')
    # plt.colorbar(ax1)

    # --------------------------------------------------------------------
    """ Compute radial PSD for the WAVEFRONT """

    profile = []
    for i in np.arange(0, N, 50):
        fft_r = 1. / N * fft(phase_map[i, :])
        PSD = fftshift((np.abs(fft_r)) ** 2)
        PSD_sym = PSD[N // 2 + 1:]
        profile.append(PSD_sym)

    nn = PSD_sym.shape[0]
    f = np.linspace(1, N // 2, N // 2)
    f2 = f ** (-2)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    for i in range(len(profile)):
        ax.plot(profile[i], color='red')

    ax.plot(f, f2, color='black', linestyle='--', label=r'$f^{-2}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([1, N // 2])
    # ax.set_ylim([1e-9, 1])
    ax.set_xlabel('Spatial Frequency')
    ax.legend()
    # plt.title('Radial PSD')
    plt.show()

    # --------------------------------------------------------------------
    """ Evolution of the RMS  """

    H = z.model_matrix
    Z = H.shape[-1]
    model_matrix = H.reshape((N, N, Z))

    mean = np.mean(phase_map)
    total_rms = np.sqrt(1. / N ** 2 * np.sum((phase_map - mean) ** 2))

    print('Initial RMS of the Wavefront: %.3f [waves]' % total_rms)

    RMS = []
    maps = []
    remaining = np.zeros((N, N))
    max_phase = np.max(phase_map)
    min_phase = np.min(phase_map)

    for i in range(N_zern):
        copy_phase = phase_map.copy()
        coef_zern = coef[:i + 1]
        matrix = model_matrix[:, :, :i + 1]
        phase_to_extract = np.dot(matrix, coef_zern)
        remaining = copy_phase - phase_to_extract
        RMS.append(compute_rms(remaining)/total_rms)

        if (np.mod(i, 20) == 0):
            maps.append(remaining)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(RMS)
    ax.set_xlim([0, N_zern])
    ax.set_ylim([0., 1.])
    # plt.title('Evolution of RMS as we remove Zernikes')
    plt.xlabel('Zernike polynomial')
    plt.ylabel(r'$\frac{RMS}{RMS_0}$')

    # --------------------------------------------------------------------
    """ Impact of decaying Weights """

    n = solve_triangular(Z)
    alfa = decay_factor(n, 100)
    v = generate_decay_vector(n, alfa)
    print(v)

    rescaled_coef = v[:N_zern] * coef
    # Rescale them so that they have comparable PV to the original map
    norm_unscaled = np.linalg.norm(coef)
    norm_rescaled = np.linalg.norm(rescaled_coef)
    rescaled_coef *= (norm_unscaled / norm_rescaled)

    new_phase = np.dot(model_matrix[:, :, :N_zern], rescaled_coef)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax1 = ax.contourf(xx, yy, new_phase, 100, cmap='hot')
    ax.set_aspect('equal')
    plt.title('Wavefront map in Cartesian (Decaying Coeff)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(ax1)

    fft_resc = 1. / N * fft(new_phase[N // 2, :])
    PSD_resc = fftshift((np.abs(fft_resc)) ** 2)

    PSD_resc = PSD_resc[N // 2 + 1:]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.loglog(PSD_sym, label='No Decay')
    ax.loglog(PSD_resc, label='Decay')
    ax.set_xlabel('Spatial Frequency')
    ax.set_xlim([1, N // 2])
    ax.set_ylim([1e-6, 1])
    plt.legend()
    plt.title('Radial PSD')
    plt.show()

    mean = np.mean(new_phase)
    new_rms = np.sqrt(1. / N ** 2 * np.sum((new_phase - mean) ** 2))

    RMS_decay = []
    remaining = np.zeros((N, N))
    max_phase = np.max(new_phase)
    min_phase = np.min(new_phase)

    for i in range(N_zern):
        copy_phase = new_phase.copy()
        coef_zern = rescaled_coef[:i + 1]
        matrix = model_matrix[:, :, :i + 1]
        phase_to_extract = np.dot(matrix, coef_zern)
        remaining = copy_phase - phase_to_extract
        RMS_decay.append(compute_rms(remaining) / new_rms)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(RMS, label='No decay')
    ax.plot(RMS_decay, label='Decay')
    ax.set_xlim([0, N_zern])
    ax.set_ylim([0., 1.])
    # plt.title('Evolution of RMS as we remove Zernikes')
    plt.xlabel('Zernike polynomial')
    plt.ylabel(r'$\frac{RMS}{RMS_0}$')
    plt.legend()
    plt.show()


    # --------------------------------------------------------------------
    """ PSD of the PSF """

    from numpy.fft import fft2, fftshift


    def compute_PSF(pupil_mask, phase_map, wavelength):
        phase = 2 * np.pi * 1j * phase_map / wavelength
        pupil_function = pupil_mask * np.exp(phase)
        electric_field = fftshift(fft2(pupil_function))
        PSF = (np.abs(electric_field)) ** 2
        return PSF


    N_pix = 1024
    D = 39.  # ELT diameter: 39 [meters]
    wave = 1.5e-6  # Wavelength: 1.5 [microns]
    l_u0 = 5. * D  # Place-holder for the physical length l_u

    u_min, u_max = -l_u0 / 2., l_u0 / 2.
    u = np.linspace(u_min, u_max, N_pix)
    v = u
    uu, vv = np.meshgrid(u, v)

    # Pupil mask
    pupil_mask = uu ** 2 + vv ** 2 <= (D / 2.) ** 2

    ideal_psf = compute_PSF(pupil_mask, pupil_mask, wave)
    PEAK = ideal_psf.max()
    ideal_psf /= PEAK

    PIX = 15
    min_pix, max_pix = N_pix // 2 - PIX, N_pix // 2 + PIX + 1
    pixels = np.linspace(-PIX, PIX, 2 * PIX + 1).astype(int)

    # Show a zoom of the PSF array
    psf_zoom = ideal_psf[min_pix:max_pix, min_pix:max_pix]
    plt.figure()
    plt.imshow(psf_zoom, extent=(-PIX, PIX, -PIX, PIX))
    plt.colorbar()
    plt.xlabel('Pixels')
    plt.ylabel('Pixels')
    plt.title('Normalized PSF')

    # Plot a 1D profile of the PSF
    psf_1D = ideal_psf[N_pix // 2, min_pix:max_pix]
    plt.figure()
    plt.plot(pixels, psf_1D)
    plt.scatter(pixels, psf_1D, s=20, color='Black')
    plt.xlabel('Pixels')
    plt.title('1D cross-section of the PSF')

    # PSD of the PSF
    fft_r = 1. / N * fft(ideal_psf[N // 2, :])
    ideal_PSD = fftshift((np.abs(fft_r)) ** 1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogx(ideal_PSD[N // 2:] / ideal_PSD[N // 2])
    ax.set_xlabel('Spatial Frequency')
    plt.title('Radial PSD')

    factor = 1.5
    psf_phase = phase_map / phase_map.max() * wave * factor

    aberrated_psf = compute_PSF(pupil_mask, psf_phase, wave)
    aberrated_psf /= PEAK
    print('Peak PSF', aberrated_psf.max())

    psf_zoom = aberrated_psf[min_pix:max_pix, min_pix:max_pix]
    plt.figure()
    plt.imshow(psf_zoom, extent=(-PIX, PIX, -PIX, PIX))
    plt.colorbar()
    plt.xlabel('Pixels')
    plt.ylabel('Pixels')
    plt.title('Normalized PSF')

    # Find where the PSF peak lies
    ij = np.argwhere(aberrated_psf.max() == aberrated_psf)
    ic, jc = ij[:, 0][0], ij[:, 1][0]


    def PSD_formula(array):
        fft_r = 1. / N * fft(array)
        return fftshift((np.abs(fft_r)) ** 1)


    def compute_PSD(PSF, centers):
        ic, jc = centers
        right = PSD_formula(PSF[ic, :])
        up = PSD_formula(PSF[:, jc])

        m = ideal_PSD[N // 2]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(ideal_PSD[N // 2:]/m, color='black', label='Perfect PSF')
        ax.plot(right[N // 2:]/m, label='Horizontal')
        ax.plot(up[N // 2:]/m, label='Vertical')
        ax.set_xlabel('Spatial Frequency')
        ax.set_ylim([0., 1.])
        ax.set_xlim([1, N // 2])
        ax.legend()
        plt.title('Modulation Transfer Function (MTF)')
        plt.savefig('a')


    compute_PSD(aberrated_psf, (ic, jc))

    plt.show()


    plt.show()