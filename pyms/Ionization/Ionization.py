import sys

import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.integrate as integrate

# from py_multislice import wavev, relativistic_mass_correction, interaction_constant


def colorize(z, ccc=None, max=None, min=None, gamma=1):
    from colorsys import hls_to_rgb

    # Get shape of array
    n, m = z.shape
    # Create RGB array
    c = np.zeros((n, m, 3))
    # Set infinite values to be constant color
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))

    # A is the color (Hue)
    A = (np.angle(z[idx])) / (2 * np.pi)
    A = (A) % 1.0

    # B is the lightness
    B = np.ones_like(A) * 0.5

    # Calculate min and max of array or take user provided values
    if min is None:
        min_ = (np.abs(z) ** gamma).min()
    else:
        min_ = min
    if max is None:
        max_ = (np.abs(z) ** gamma).max()
    else:
        max_ = np.abs(max) ** gamma
    if ccc is None:
        range = max_ - min_
        if range < 1e-10:
            C = np.ones(z.shape)[idx] * 0.49
        else:
            C = ((np.abs(z[idx]) - min_) / range) ** gamma * 0.5

    else:
        C = ccc
    # C = np.ones_like(B)*0.5

    c[idx] = [hls_to_rgb(a, cc, b) for a, b, cc in zip(A, B, C)]
    return c


class orbital:
    def __init__(self, Z: int, config: str, n: int, l: int, epsilon=0):
        """Initialize the orbital class and return an orbital 
           object.
           Parameters:
           Z:       Atomic number
           config:  String describing configuration of atom ie:
                    carbon (C): config = '1s2 2s2 2p2' 
           n:       Principal quantum number of orbital, for 
                    continuum wavefunctions n=0
           l:       Orbital angular momentum quantum number of 
                    orbital
           epsilon: Optional, energy of continuum wavefunction"""

        # Load arguments into orbital object
        self.Z = Z
        self.config = config
        self.n = n
        self.l = l
        if self.n == 0:
            self.epsilon = epsilon

        # Use pfac (Python flexible atomic code) interface to
        # communicate with underlying fac code

        # Get atom
        pfac.fac.SetAtom(ATOMICSYMBOL[Z])
        if n == 0:
            configstring = pfac.fac.ATOMICSYMBOL[Z] + "ex"
        else:
            configstring = pfac.fac.ATOMICSYMBOL[Z] + "bound"
        # Set up configuration
        pfac.fac.Config(configstring, config)
        # Optimize atomic energy levels
        pfac.fac.ConfigEnergy(0)
        # Optimize radial wave functions
        pfac.fac.OptimizeRadial(configstring)
        # Optimize energy levels
        pfac.fac.ConfigEnergy(1)

        # Orbital title
        if n > 0:
            # Bound wave function case
            angmom = ["s", "p", "d", "f"][l]
            # Title in the format "Ag 1s", "O 2s" etc..
            self.title = "{0} {1}{2}".format(pfac.fac.ATOMICSYMBOL[Z], n, angmom)
        else:
            # Continuum wave function case
            # Title in the format "Ag e = 10 eV l'=2" etc..
            self.title = "{0} \\varepsilon = {1} l' = {2}".format(
                pfac.fac.ATOMICSYMBOL[Z], epsilon, l
            )

        # Calculate relativstic quantum number from
        # non-relativistic input
        kappa = -1 - l

        # Output desired wave function from table
        pfac.fac.WaveFuncTable("orbital.txt", n, kappa, epsilon)

        # Clear table
        # ClearOrbitalTable ()
        pfac.fac.Reinit(config=1)

        with open("orbital.txt", "r") as content_file:
            content = content_file.read()
        import re

        self.ilast = int(re.search("ilast\\s+=\\s+([0-9]+)", content).group(1))
        self.energy = float(re.search("energy\\s+=\\s+([^\\n]+)\\n", content).group(1))
        # Load information into table
        table = np.loadtxt("orbital.txt", skiprows=15)

        # Load radial grid (in atomic units)
        self.r = table[:, 1]

        # Load large component of wave function
        self.wfn_table = table[: self.ilast, 4]

        from scipy.interpolate import interp1d

        # If continuum wave function load phase-amplitude solution
        #
        if self.n == 0:
            self.__amplitude = interp1d(
                table[self.ilast - 1 :, 1], table[self.ilast - 1 :, 2], fill_value=0
            )
            self.__phase = interp1d(
                table[self.ilast - 1 :, 1], table[self.ilast - 1 :, 3], fill_value=0
            )

        # For bound wave functions we simply interpolate the
        # tabulated values of a0 the wavefunction
        # *2
        self.__wfn = interp1d(
            table[: self.ilast, 1], table[: self.ilast, 4], kind="cubic", fill_value=0
        )

    def wfn(self, r):
        """Evaluate radial wavefunction on grid r from tabulated values"""

        is_arr = isinstance(r, np.ndarray)

        if is_arr:
            r_ = r
        else:
            r_ = np.asarray([r])

        # Initialize output array
        wvfn = np.zeros(r_.shape, dtype=np.float)

        # Region I and II refer to the two solution regions used in the
        # Flexible Atomic Code for continuum wave functions. Region I
        # (close to the nucleus) is where the radial Dirac equation is
        # solved with a numerical integration using the Numerov algorithm.
        # In Region II, a phase-amplitude solution is used.

        # For bound wave functions, or for r in region I for
        # a continuum wave function we simply interpolate the
        # tabulated values of the wavefunction
        mask = np.logical_and(self.r[0] <= r_, r_ < self.r[self.ilast - 1])
        wvfn[mask] = self.__wfn(r_[mask])

        # For bound atomic wave functions our work here is done...
        if self.n > 0:
            return wvfn

        # For a continuum wave function inbetween region I and II
        # interpolate between the regions
        mask = np.logical_and(
            r_ >= self.r[self.ilast - 1], r_ <= self.r[self.ilast + 1]
        )
        if np.any(mask):
            r1 = self.r[self.ilast - 1]
            r2 = self.r[self.ilast + 1]
            # Phase amplitude
            PA = self.__amplitude(r2) * np.sin(self.__phase(r2))
            # Tabulated
            TB = self.__wfn(r1)
            wvfn[mask] = (PA - TB) / (r2 - r1) * (r_[mask] - r1) + TB

        # For a continuum wave function and r in region II
        # interpolate amplitude and phase
        # wvfn[:] = 0.0
        mask = r_ > self.r[self.ilast + 1]
        wvfn[mask] = self.__amplitude(r_[mask]) * np.sin(self.__phase(r_[mask]))
        if is_arr:
            return wvfn
        else:
            return wvfn[0]

    def plot(self, grid=np.linspace(0.0, 2.0, num=50), show=True, ylim=None):
        """Plot wavefunction at positions given by grid r"""

        fig, ax = plt.subplots(figsize=(4, 4))
        wavefunction = self.wfn(grid)
        ax.plot(grid, wavefunction)
        ax.set_title(self.title)
        if ylim is None:
            ylim_ = [1.2 * np.amin(wavefunction), 1.2 * np.amax(wavefunction)]
        else:
            ylim_ = ylim
        ax.set_ylim(ylim_)
        ax.set_xlim([np.amin(grid), np.amax(grid)])
        ax.set_xlabel("r (a.u.)")
        ax.set_ylabel("$P_{nl}(r)$")
        if show:
            plt.show(block=True)
        return fig


def transition_potential(
    orb1, orb2, gridshape, gridsize, ml, mlprime, keV, bandwidth_limiting=2.0 / 3
):
    """Evaluate an inelastic transition potential for excitation of an
       electron from orbital 1 to orbital 2 on grid with shape gridshape
       and real space dimensions in Angstrom given by gridsize"""

    # Bohr radius in Angstrom
    a0 = 5.29177210903e-1

    # Calculate energy loss
    deltaE = orb1.energy - orb2.energy
    k0 = wavev(keV * 1e3)
    kn = wavev(keV * 1e3 + deltaE)

    # Minimum momentum transfer at this energy loss
    qz = k0 - kn

    # Get grid dimensions in reciprocal space in units of inverse Bohr radii
    # (to match atomic wave function output from the Flexible Atomic Code)
    qspace = [gridshape[i] / gridsize[i] for i in range(2)]
    deltaq = [1 / gridsize[i] for i in range(2)]
    qgrid = [
        np.fft.fftfreq(gridshape[1]) * qspace[0],
        np.fft.fftfreq(gridshape[0]) * qspace[1],
    ]

    # Transverse momentum transfer
    qt = np.sqrt(qgrid[0][:, np.newaxis] ** 2 + qgrid[1][np.newaxis, :] ** 2)
    # Amplitude of momentum transfer at each gridpoint
    qabs = np.sqrt(qt ** 2 + qz ** 2)
    # Polar angle of momentum transfer
    qtheta = np.pi - np.arctan(qt / qz)

    # Azimuth angle of momentum transfer
    qphi = np.arctan2(qgrid[1][np.newaxis, :], qgrid[0][:, np.newaxis])

    # Maximum coordinate at which transition potential will be evaluated
    qmax = np.amax(qabs) * bandwidth_limiting

    # Initialize output array
    Hn0 = np.zeros(gridshape, dtype=np.complex)

    # Get spherical Bessel functions, spherical harmonics and Wigner 3j symbols
    from scipy.special import spherical_jn, sph_harm
    from sympy.physics.wigner import wigner_3j

    # Get interpolation function from scipy
    from scipy.interpolate import interp1d

    # Get angular momentum quantum numbers for both states
    l = int(orb1.l)
    lprime = int(orb2.l)

    # Check that ml and mlprime, the projection quantum numbers for the bound
    # and free states, are sensible
    if np.abs(ml) > l:
        return Hn0
    if np.abs(mlprime) > lprime:
        return Hn0

    def ovlap(q, lprimeprime):
        """Overlap integral of orbital wave functions weighted
           by spherical bessel function"""

        # Function currently written assuming at least one of the
        # orbitals is bound
        # Find maximum radial coordinate
        rmax = 0
        if orb1.n > 0:
            rmax = orb1.r[orb1.ilast - 1]
        if orb2.n > 0:
            rmax = max(rmax, orb2.r[orb2.ilast - 1])

        # The following ensures that q can be passed as a single value or
        # as an array
        is_arr = isinstance(q, np.ndarray)
        if is_arr:
            q_ = q
        else:
            q_ = np.asarray([q])

        # Initialize output array
        jq = np.zeros_like(q_)

        for iQ, Q in enumerate(np.ravel(q_)):
            # Redefine kernel for this value of q, factor of a0 converts q from
            # units of inverse Angstrom to inverse Bohr radii,
            overlap_kernel = (
                lambda x: orb1.wfn(x)
                * spherical_jn(lprimeprime, 2 * np.pi * Q * a0 * x)
                * orb2.wfn(x)
            )
            jq[iQ] = integrate.quad(overlap_kernel, 0, rmax)[0]

        return jq

    # Mesh to calculate overlap integrals on and then interpolate
    # from
    qmesh = np.arange(0, qmax * 1.05, min(deltaq))

    # Only evaluate transition potential within the multislice
    # bandwidth limit
    qmask = qabs <= qmax

    # The triangle inequality for the Wigner 3j symbols mean that result is
    # only non-zero for certain values of lprimeprime:
    # |l-lprime|<=lprimeprime<=|l+lprime|
    lprimeprimes = np.arange(np.abs(l - lprime), np.abs(l + lprime) + 1, dtype=np.int)
    if lprimeprimes.shape[0] < 1:
        return None

    for lprimeprime in lprimeprimes:
        jq = None
        # Set of projection quantum numbers
        mlprimeprimes = np.arange(-lprimeprime, lprimeprime + 1, dtype=np.int)

        # Non mlprimeprime dependent part of prefactor from Eq (13) from
        # Dwyer Ultramicroscopy 104 (2005) 141-151
        prefactor1 = (
            np.sqrt(4 * np.pi)
            * ((-1j) ** lprimeprime)
            * np.sqrt((2 * lprime + 1) * (2 * lprimeprime + 1) * (2 * l + 1))
        )
        for mlprimeprime in mlprimeprimes:
            if ml - mlprime - mlprimeprime != 0:
                continue
            # Evaluate Eq (14) from Dwyer Ultramicroscopy 104 (2005) 141-151
            prefactor2 = (
                (-1.0) ** (mlprime + mlprimeprime)
                * np.float(wigner_3j(lprime, lprimeprime, l, 0, 0, 0))
                * np.float(
                    wigner_3j(lprime, lprimeprime, l, -mlprime, -mlprimeprime, ml)
                )
            )

            if np.abs(prefactor2) > 1e-12:
                # Set up interpolation of overlap integral function in Eq (13)
                # from Dwyer Ultramicroscopy 104 (2005) 141-151
                # Checking if None ensures that jq is only evaluated if actually
                # needed for each lprimeprime
                if jq is None:
                    jq = interp1d(qmesh, ovlap(qmesh, lprimeprime))(qabs[qmask])

                Ylm = sph_harm(mlprimeprime, lprimeprime, qphi[qmask], qtheta[qmask])

                # Evaluate Eq (13) from Dwyer Ultramicroscopy 104 (2005) 141-151
                Hn0[qmask] += prefactor1 * prefactor2 * jq * Ylm
    # Vaccuum permitivity in Coulomb/VA
    eps0 = 8.8541878128e-22
    # Electron charge in Coulomb
    qe = 1.60217662e-19

    # Relativistic electron mass correction
    gamma = relativistic_mass_correction(keV * 1e3)

    sigma = interaction_constant(keV * 1e3 + deltaE, units="rad/VA")

    # Need to multiply by area of k-space pixel (1/gridsize) and multiply by
    # #pixels to get correct units from inverse Fourier transform
    Hn0 *= np.prod(gridshape) ** 1.5 / np.prod(gridsize)

    # Return result of Eq. (10) from Dwyer Ultramicroscopy 104 (2005) 141-151
    # in real space
    return (
        qe / 4 / np.pi ** 2 / eps0 * sigma * np.fft.ifft2(Hn0 / qabs ** 2) * np.sqrt(2)
    )
    # return 2*np.pi**1.5/a0**3*np.fft.ifft2(Hn0/qabs**2,norm='ortho')
    # return 2*np.pi**1.5/a0**2*const*np.fft.ifft2(Hn0/qabs**2,norm='ortho')


if __name__ == "__main__":
    for lprime in [0, 1, 2]:
        Sifree = orbital(14, "1s1 2s2 2p6 3s2 3p2", 0, lprime, epsilon=1e5)
        Sifree.plot(np.linspace(0, 1))
    sys.exit()

    keV = 300

    # GaK = orbital(31,'1s1 2s2 2p6 3s2 3p6 4s2 3d10 4p1',0,0,epsilon=1e3)
    # GaK.plot(np.linspace(0,1,num=200))
    # sys.exit()

    FeL = orbital(26, "1s2 2s2 2p6 3s2 3p6 4s2 3d6", 2, 1)

    FeFree = orbital(26, "1s2 2s2 2p6 3s2 3p6 4s2 3d6", 0, 2, epsilon=1)
    rdims = [2 * 3.905, 2 * 3.905]
    grid = [256, 256]
    Hn0 = np.fft.fftshift(transition_potential(FeL, FeFree, grid, rdims, 0, 1, keV))
    Hn0_inten = np.abs(Hn0) ** 2
    print(np.amax(Hn0_inten), np.amin(Hn0_inten), np.mean(Hn0_inten))
    fig, ax = plt.subplots()
    ax.imshow(Hn0_inten)
    plt.show(block=True)
    sys.exit()

    LaN45 = orbital(57, "1s2 2s2 2p6 3s2 3p6 3d10 4s2 4p6 4d10 5s2 5p6 5d1 6s2", 4, 2)
    # LaN45.plot(np.linspace(0,5,num=200))

    print(integrate.quad(lambda x: LaN45.wfn(x) ** 2, 0, 5)[0])
    LaFree = orbital(
        57, "1s2 2s2 2p6 3s2 3p6 4s2 3d10 4p6 5s2 4d9 5p6 6s2 5d1", 0, 3, epsilon=1
    )
    # LaFree.plot(np.linspace(0,50,num=200))
    rdims = [2 * 11.76, 2 * 11.76]
    grid = [128, 128]
    Hn0s = np.zeros((4, *grid), dtype=np.complex)
    mls = 0
    # lprimes

    for rdims in ([40, 40], [50, 50])[:1]:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        Hn0 = np.fft.fftshift(
            transition_potential(LaN45, LaFree, grid, rdims, 0, 0, keV)
        )
        ax[1, 1].imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(Hn0)))))
        Hn0 *= -1j  # interaction_constant(keV*1e3,units = 'rad/VA')
        ax[0, 0].imshow(np.abs(Hn0) ** 2)
        ax[0, 1].imshow(colorize(Hn0))

        ax[1, 0].plot(
            np.linspace(-rdims[0] / 2, rdims[0] / 2, num=grid[0]),
            np.mean(np.abs(Hn0) ** 2, axis=0),
        )
        # ax[1,0].set_ylim([0,0.005])
        ax[1, 0].set_xlim([-5.88, 5.88])
        print(np.amax(np.abs(Hn0) ** 2), np.amin(np.abs(Hn0) ** 2))
        plt.show(block=True)
    sys.exit()
    Sifree = orbital(14, "1s1 2s2 2p6 3s2 3p2", 0, 1, epsilon=10)
    # Ofree.plot()
    Si1s = orbital(14, "1s2 2s2 2p6 3s2 3p2", 1, 0)
    # O1s.plot()
    keV = 100

    from mpl_toolkits import mplot3d

    # plt.rc('text', usetex=True)
    fig = plt.figure(figsize=(4, 12))
    funcs = [np.real, np.imag, np.real]
    for iqnum, qnums in enumerate([(0, 0), (1, 0), (1, 1)]):
        lprime, mlprime = qnums
        Sifree = orbital(14, "1s1 2s2 2p6 3s2 3p2", 0, lprime, epsilon=10)
        Hn0 = transition_potential(Si1s, Sifree, [128, 128], [2, 2], 0, mlprime, keV)
        Hn0 = np.fft.fftshift(Hn0)

        ax = fig.add_subplot(311 + iqnum, projection="3d")
        grid = np.linspace(-1.0, 1.0, num=128)
        # colors = plt.get_cmap('hsv')((np.angle(Hn0)+np.pi)/(2*np.pi))
        ax.plot_surface(
            grid[:, np.newaxis], grid[np.newaxis, :], funcs[iqnum](Hn0)
        )  # ,
        # facecolors = colors)
        ax.set_xlabel(r"x")
        ax.set_ylabel(r"y")
        ax.set_title("l'={0}, m'={1}".format(lprime, mlprime))

    fig.savefig("Dwyer_reproduction.pdf")
    plt.show(block=True)

    sys.exit()
