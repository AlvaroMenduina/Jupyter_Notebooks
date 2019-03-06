"""

Analysis of Fourier transforms of Zernike polynomials for the
PSF differential features


"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from numpy.fft import fft2, fftshift
import zern_core as zern

N_pix = 1024                                        # Number of pixels in array
N_zern = 15                                         # Number of Zernike terms to consider

rho = 0.15                                          # Aperture Radius
D = 2*rho                                           # Diameter
rho_zern = 1.*rho                                   # Zernike normalization radius

pix = 50                                            # Pixels for the zoomed arrays
minPix = (N_pix - pix)//2
maxPix = (N_pix + pix)//2

# Pupil and Image space sampling
Luv = 3.5                                           # Physical Length of Pupil Array
Lxy = N_pix * D / Luv                               # Length of Image Array in [Lambda / D] units
l = pix * 2 * rho / Luv                             # Length of Zoomed Image Array in [Lambda / D] units
pix_scale = 2 * rho / Luv                           # Scale of pixels in the Image Array [Lambda / D] units
size = np.linspace(-Lxy / 2, Lxy / 2, N_pix,
                   endpoint=True)

cmap = 'viridis'

def reshape_model_matrix(matrix, mask):
    # Reshape the Zernike model matrix from flattened to square
    H = [zern.invert_mask(matrix[:,i], mask) for i in range(matrix.shape[-1])]
    H = np.array(H)
    H_new = np.moveaxis(H, 0, -1)
    return H_new

if __name__ == "__main__":

    x = np.linspace(-Luv/2, Luv/2, N_pix, endpoint=True)

    xx, yy = np.meshgrid(x, x)
    r = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(xx, yy)

    pupil_mask = xx ** 2 + yy ** 2 <= rho ** 2
    r, theta = r[pupil_mask], theta[pupil_mask]

    # Initialize the ZernikeNaive to get access to the Model Matrix
    np.random.seed(123)
    coef = np.random.normal(size=N_zern)
    z = zern.ZernikeNaive(mask=pupil_mask)
    _phase = z(coef=coef, rho=r/rho_zern, theta=theta, normalize_noll=False, mode='Jacobi', print_option='Silent')

    # ==================================================================================================================
    """ Model matrix for Zernike polynomials """

    # H contains the Zernike Polynomials
    H = reshape_model_matrix(z.model_matrix, pupil_mask)
    N_zern = H.shape[-1]
    print('Using %d Zernikes' %N_zern)

    # To check the Zernikes are properly computed
    a_zernike = H[:,:, 1]
    plt.figure()
    plt.imshow(a_zernike, extent=(-Luv/2, Luv/2, -Luv/2, Luv/2))
    plt.xlim([-rho, rho])
    plt.ylim([-rho, rho])
    plt.colorbar()

    # ==================================================================================================================
    #                                           1st Order Taylor Expansion
    # ==================================================================================================================

    """ (1) Pupil Aperture terms """
    # Formula: 1j (F_{k} F^*_{\Pi} - F^*_{k} F_{\Pi})

    # Calculate the ideal PSF to normalize everything by its peak
    PSF0 = fftshift(fft2(H[:,:,0]*np.exp(1j*H[:,:,0]), norm='ortho'))
    PSF0 = (np.abs(PSF0))**2
    PEAK = PSF0.max()

    # Aperture component F_{\Pi}
    F_Pi = fftshift(fft2(H[:,:,0], norm='ortho'))
    F_Pi_conj = np.conj(F_Pi)

    # Aberration component
    for k in np.arange(1, 10):
        F_k = fftshift(fft2(H[:,:,k], norm='ortho'))
        F_k_conj = np.conj(F_k)

        F_PiK = 1j * (F_k * F_Pi_conj - F_k_conj * F_Pi)

        print(np.max(np.imag(F_PiK)))    # sanity check
        F_PiK0 = np.real(F_PiK)

        # Airy Disk
        airy = Circle((pix_scale/2,-pix_scale/2),1.22,
                      fill=None, linestyle='--')
        fig, ax = plt.subplots(1)
        im = ax.imshow(F_PiK0 / PEAK *100, cmap=cmap, extent=(-Lxy/2, Lxy/2, -Lxy/2, Lxy/2))
        plt.colorbar(im)
        ax.add_patch(airy)
        plt.xlim([-l/2, l/2])
        plt.ylim([-l/2, l/2])
        plt.xlabel(r'$[\lambda/D]$')
        plt.ylabel(r'$[\lambda/D]$')

        if np.abs(F_PiK0.max()) < 1e-6:
            im.set_clim(-10, 10)

        plt.title(r'$i (F_{%d}F^*_{\Pi} - F^*_{%d}F_{\Pi})$' %(k,k))
        # plt.savefig('F_Pi_%d' %k)

    # ==================================================================================================================
    """ (2) Zernike terms F_ZZ """
    """ F_jj terms """

    for k in np.arange(1, 10):
        F_k = fftshift(fft2(1j*H[:,:,k], norm='ortho'))
        F_k_conj = np.conj(F_k)
        F_ZZ = F_k * F_k_conj
        F_ZZ0 = np.real(F_ZZ)

        airy = Circle((pix_scale/2,-pix_scale/2),1.22,
                      fill=None, linestyle='--', color='white')
        fig, ax = plt.subplots(1)
        im = ax.imshow(F_ZZ0 / PEAK *100, cmap=cmap, extent=(-Lxy/2, Lxy/2, -Lxy/2, Lxy/2))
        plt.colorbar(im)
        ax.add_patch(airy)
        plt.xlim([-l/2, l/2])
        plt.ylim([-l/2, l/2])
        plt.xlabel(r'$[\lambda/D]$')
        plt.ylabel(r'$[\lambda/D]$')

        plt.title(r'$F_{%d}F^*_{%d}$' %(k,k))
        # plt.savefig('F%d%d' %(k,k))

    """ F_jk terms """

    j = 3
    for k in np.arange(1, 15):
        F_j = fftshift(fft2(H[:,:,j], norm='ortho'))
        F_j_conj = np.conj(F_j)
        F_k = fftshift(fft2(H[:,:,k], norm='ortho'))
        F_k_conj = np.conj(F_k)

        a = F_k * F_j_conj
        b = F_j * F_k_conj
        F_jk = a + b
        F_jk0 = np.real(F_jk)
        m = F_jk0.max()

        airy = Circle((pix_scale/2,-pix_scale/2),1.22,
                      fill=None, linestyle='--', color='black')

        fig, ax = plt.subplots(1)
        im = ax.imshow(F_jk0 / PEAK *100, cmap=cmap, extent=(-Lxy/2, Lxy/2, -Lxy/2, Lxy/2))
        plt.colorbar(im)
        ax.add_patch(airy)
        plt.xlim([-l/2, l/2])
        plt.ylim([-l/2, l/2])
        plt.xlabel(r'$[\lambda/D]$')
        plt.ylabel(r'$[\lambda/D]$')
        plt.title(r'$F_{%d}F^*_{%d} + F^*_{%d}F_{%d}$' %(j,k, j, k))

        if m < 1e-6:
            im.set_clim(-10, 10)

        # plt.savefig('F%d%d' %(j,k))

    # ==================================================================================================================
    #                                       Accuracy of the Taylor expansion
    # ==================================================================================================================

    # Use coma as the aberration
    coma = H[:,:,7]

    # Limits of aberration intensity (log_10(a))
    e_min = -4
    e_max = 0
    epsilons = np.logspace(e_min, e_max, 100)
    errs_1, errs_2 = [], []
    for eps in epsilons:
        wavefront = eps*coma
        pupil = pupil_mask * np.exp(1j*wavefront)
        PSF_true = np.abs((fftshift(fft2(pupil, norm='ortho')))**2)
        Peak = PSF_true.max()

        # plt.figure()
        # plt.imshow(PSF_true)
        # plt.xlim([minPix, maxPix])
        # plt.ylim([minPix, maxPix])
        # plt.colorbar()
        # plt.title('True PSF with Coma ($\epsilon=%.2f$)' %eps)

        # 1 + i\Phi
        linear_expansion = pupil_mask * (np.ones_like(pupil_mask) + 1j*wavefront)
        PSF_linear = np.abs((fftshift(fft2(linear_expansion, norm='ortho'))) ** 2)

        residual_linear = np.abs(PSF_true - PSF_linear) / Peak * 100
        m_err1 = np.max(residual_linear)
        errs_1.append(m_err1)

        # 1 + i\Phi - \Phi^2 / 2
        quadratic_expansion = pupil_mask * (np.ones_like(pupil_mask) + 1j * wavefront - (wavefront) ** 2 / 2)
        PSF_quadratic = np.abs((fftshift(fft2(quadratic_expansion, norm='ortho'))) ** 2)

        residual_quadratic = np.abs(PSF_true - PSF_quadratic) / Peak * 100
        m_err2 = np.max(residual_quadratic)
        errs_2.append(m_err2)

    plt.figure()
    plt.loglog(epsilons, errs_1, label=r'$1 + i\Phi$')
    plt.loglog(epsilons, errs_2, label=r'$1 + i\Phi - \Phi^2/2$')
    plt.xlabel(r'Aberration coefficient $a_j$ [waves]')
    plt.ylabel(r'PSF error [per cent]')
    plt.xlim([1e-4, 1])
    plt.ylim([1e-14, 1e2])
    plt.legend(title='Taylor expansion')
    plt.grid(True)

    # Compute the slopes to check the order
    slope_1 = (np.log10(errs_1[-1]) - np.log10(errs_1[0])) / (e_max - e_min)
    slope_2 = (np.log10(errs_2[-1]) - np.log10(errs_2[0])) / (e_max - e_min)


    # ==================================================================================================================
    #                                           2nd Order Taylor expansion
    # ==================================================================================================================
    """ Quadratic Aperture terms """
    # Formula: -0.5(F_{\Pi} F^*_{\Phi2} + F^*_{Pi}F_{\Phi2}

    j = 1
    for j in np.arange(1, 10):

        phase = H[:, :, j]
        F_j = fftshift(fft2(phase**2, norm='ortho'))
        F_j_conj = np.conj(F_j)
        F_Pi = fftshift(fft2(H[:, :, 0], norm='ortho'))
        F_phi = -0.5*np.real(F_Pi*F_j_conj + np.conj(F_Pi)*F_j)


        airy = Circle((pix_scale / 2, -pix_scale / 2), 1.22,
                      fill=None, linestyle='--', color='white')
        fig, ax = plt.subplots(1)
        im = ax.imshow((-F_phi)/ PEAK * 100, cmap='viridis', extent=(-Lxy / 2, Lxy / 2, -Lxy / 2, Lxy / 2))
        ax.add_patch(airy)
        plt.colorbar(im)
        plt.xlim([-l/2, l/2])
        plt.ylim([-l/2, l/2])
        plt.xlabel(r'$[\lambda/D]$')
        plt.ylabel(r'$[\lambda/D]$')
        plt.title(r'$-F_{\Pi \Phi^2}$(%d)' %(j))
        plt.savefig('F_Pi_Phi2_%d' %(j))

    """ The Third Orders """

    for j in np.arange(1, 10):
        phase = H[:, :, j]
        F_j = fftshift(fft2(phase, norm='ortho'))
        F_j_conj = np.conj(F_j)


        F_j2 = fftshift(fft2(phase**2, norm='ortho'))
        F_j2_conj = np.conj(F_j2)

        F_phi1 = 0.5 * (F_j_conj * F_j2)
        F_phi2 = 0.5 * (F_j * F_j2_conj)
        F_phi = 1j* (F_phi1 - F_phi2)
        print(np.max(np.abs(np.imag(F_phi))))

        airy = Circle((pix_scale / 2, -pix_scale / 2), 1.22,
                      fill=None, linestyle='--', color='black')

        fig, ax = plt.subplots(1)
        im = ax.imshow(np.real(F_phi)/ PEAK * 100, cmap='viridis',extent=(-Lxy / 2, Lxy / 2, -Lxy / 2, Lxy / 2))
        ax.add_patch(airy)
        plt.colorbar(im)
        plt.xlim([-l / 2, l / 2])
        plt.ylim([-l / 2, l / 2])
        plt.xlabel(r'$[\lambda/D]$')
        plt.ylabel(r'$[\lambda/D]$')
        plt.title(r'$i/2 (F^*_{\Phi}F_{\Phi^2} - F_{\Phi}F^*_{\Phi^2})$(%d)' %(j))
        if np.max(np.real(F_phi)) < 1e-4:
            im.set_clim(-2.5, 2.5)
        # plt.savefig('F%d' % (j))


    # Cross Terms (only when at least 2 aberrations)
    j = 1
    for k in np.arange(1, 15):
        phase_j = H[:, :, j]
        F_j = fftshift(fft2(phase_j, norm='ortho'))
        F_j_conj = np.conj(F_j)
        phase_k = H[:, :, k]
        F_k = fftshift(fft2(phase_k, norm='ortho'))
        F_k_conj = np.conj(F_k)

        F_2j = fftshift(fft2(phase_j ** 2, norm='ortho'))
        F_2j_conj = np.conj(F_2j)
        F_2k = fftshift(fft2(phase_k ** 2, norm='ortho'))
        F_2k_conj = np.conj(F_2k)

        plus = F_j_conj*F_2k + F_k_conj*F_2j
        minus = F_j*F_2k_conj + F_k*F_2j_conj

        F = np.real(1j/2*(plus - minus))

        airy = Circle((pix_scale / 2, -pix_scale / 2), 1.22,
                      fill=None, linestyle='--', color='black')
        fig, ax = plt.subplots(1)
        im = ax.imshow(F/ PEAK * 100, cmap='viridis', extent=(-Lxy / 2, Lxy / 2, -Lxy / 2, Lxy / 2))
        ax.add_patch(airy)
        plt.colorbar(im)
        plt.xlim([-l/2, l/2])
        plt.ylim([-l/2, l/2])
        plt.xlabel(r'$[\lambda/D]$')
        plt.ylabel(r'$[\lambda/D]$')
        if np.max(F) < 1e-4:
            im.set_clim(-2.5, 2.5)

        plt.title('%d %d' %(j,k))
        plt.savefig('%d %d' %(j,k))


    # ==================================================================================================================
    #                                     Analysis of Bessel and trigonometric factors
    # ==================================================================================================================

    """ Bessel factors """
    from scipy.special import jv as bessel
    def J(n, x):
        f = (bessel(n+1, 2*np.pi*x) / x)**2
        return f

    x = np.linspace(0., 2.5, 1000)

    plt.figure()
    for n in np.arange(1, 5):
        plt.plot(x, J(n, x), label=n)

    plt.xlim([0, x.max()])
    plt.legend(title=r'Zernike order $n$')
    plt.xlabel(r'$x$')
    plt.title(r'$J_{n+1}^2(2\pi x)/x^2$')
    plt.show()


    # Parity of Bessel functions
    plt.figure()
    x0 = 5
    x = np.linspace(-x0, x0, 1000)
    styles = ['--', ':', '-.']
    for i, nu in enumerate([0, 1, 2, 3, 4, 5]):
        y = bessel(nu, x)
        if nu%2 == 0:
            color = 'blue'
        else:
            color = 'red'
        plt.plot(x, y, label=nu, linestyle=styles[i//2], color=color)

    plt.legend(title=r'Bessel order $\nu$')
    plt.grid(True)
    plt.xlabel('x')
    plt.xlim([-x0, x0])
    plt.title(r'$J_{\nu}(x)$')

    # ==================================================================================================================

    """ Trigonometric factors """

    phi = np.linspace(0, 2*np.pi, 1000)

    plt.figure()
    for m in np.arange(1, 3):
        plt.plot(phi, (np.cos(m*phi))**2, label=m)

    plt.xlim([0, 2*np.pi])
    # plt.ylim([0, 1])
    plt.legend(title=r'Zernike order $m$')
    plt.xlabel(r'Angle $\theta$')
    # plt.title(r'$\frac{J_{n+1}^2(2\pi k)}{k^2}$')
    plt.show()

    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(phi, (np.sin(1*phi))**2, color='red')
    axarr[0, 0].set_title(r'$m=-1$')
    axarr[0, 1].plot(phi, (np.sin(2*phi))**2, color='red')
    axarr[0, 1].set_title(r'$m=-2$')
    axarr[1, 0].plot(phi, (np.sin(3*phi))**2, color='red')
    axarr[1, 0].set_title(r'$m=-3$')
    axarr[1, 0].set_xlabel(r'Angle $\theta$')
    axarr[1, 1].plot(phi, (np.sin(4*phi))**2, color='red')
    axarr[1, 1].set_title(r'$m=-4$')
    axarr[1, 1].set_xlabel(r'Angle $\theta$')
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    f, axarr = plt.subplots(2, 2)
    axarr[0, 0].plot(phi, np.sin(1*phi)*np.cos(1*phi), color='red')
    axarr[0, 0].set_title(r'$\sin{\theta}\cdot \cos{\theta}$')
    axarr[0, 1].plot(phi, np.sin(3*phi)*np.cos(1*phi), color='red')
    axarr[0, 1].set_title(r'$\sin{3\theta}\cdot \cos{\theta}$')
    axarr[1, 0].plot(phi, np.sin(3*phi)*np.cos(3*phi), color='red')
    axarr[1, 0].set_title(r'$\sin{3\theta}\cdot \cos{3\theta}$')
    axarr[1, 0].set_xlabel(r'Angle $\theta$')
    axarr[1, 1].plot(phi, (np.sin(4*phi))**2, color='red')
    axarr[1, 1].set_title(r'$m=-4$')
    axarr[1, 1].set_xlabel(r'Angle $\theta$')
    # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    # ------------------------------------------------------------------------------------------------------------------






