import sys

import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.integrate as integrate
import torch
import tqdm
from .Probe import wavev, relativistic_mass_correction
from .crystal import interaction_constant
from .utils.numpy_utils import fourier_shift

# from py_multislice import wavev, relativistic_mass_correction, interaction_constant

# def _v(m1, m2, hue):
#     hue = hue % 1.0
#     if hue < ONE_SIXTH:
#         return m1 + (m2-m1)*hue*6.0
#     if hue < 0.5:
#         return m2
#     if hue < TWO_THIRD:
#         return m1 + (m2-m1)*(TWO_THIRD-hue)*6.0
#     return m1

# def hls_to_rgb(h, l, s):
#     """Modified version of the function provided by colorsys
#     https://github.com/python/cpython/blob/2.7/Lib/colorsys.py
#     to be explicity array oriented"""
#     result = np.zeros((*h.shape,3),dtype=np.float32)
#     mask = s == 0.0
#     result[mask,:] = l[mask]

#     mask = np.logical_not(mask)

#     if l <= 0.5:
#         m2 = l * (1.0+s)
#     else:
#         m2 = l+s-(l*s)
#     m1 = 2.0*l - m2
#     return (_v(m1, m2, h+ONE_THIRD), _v(m1, m2, h), _v(m1, m2, h-ONE_THIRD))

def colorize(z):
    from colorsys import hls_to_rgb
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.3)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    return c

# def colorize(z, ccc=None, max=None, min=None, gamma=1):
#     from colorsys import hls_to_rgb

#     # Get shape of array
#     n, m = z.shape
#     # Create RGB array
#     c = np.zeros((n, m, 3))
#     # Set infinite values to be constant color
#     c[np.isinf(z)] = (1.0, 1.0, 1.0)
#     c[np.isnan(z)] = (0.5, 0.5, 0.5)

#     idx = ~(np.isinf(z) + np.isnan(z))

#     # A is the color (Hue)
#     A = (np.angle(z[idx])) / (2 * np.pi)
#     A = (A) % 1.0

#     # B is the lightness
#     B = np.ones_like(A) * 0.5

#     # Calculate min and max of array or take user provided values
#     if min is None:
#         min_ = (np.abs(z) ** gamma).min()
#     else:
#         min_ = min
#     if max is None:
#         max_ = (np.abs(z) ** gamma).max()
#     else:
#         max_ = np.abs(max) ** gamma
#     if ccc is None:
#         range = max_ - min_
#         if range < 1e-10:
#             C = np.ones(z.shape)[idx] * 0.49
#         else:
#             C = ((np.abs(z[idx]) - min_) / range) ** gamma * 0.5

#     else:
#         C = ccc
#     # C = np.ones_like(B)*0.5

#     c[idx] = [hls_to_rgb(a, cc, b) for a, b, cc in zip(A, B, C)]
#     return c`


class orbital:
    def __init__(self, Z: int, config: str, n: int, l: int, epsilon=1):
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
            assert epsilon > 0, "Energy of continuum electron must be > 0"
            self.epsilon = epsilon

        # Use pfac (Python flexible atomic code) interface to
        # communicate with underlying fac code
        import pfac.fac

        # Get atom
        pfac.fac.SetAtom(pfac.fac.ATOMICSYMBOL[Z])
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

        if self.n == 0:
            # If continuum wave function load phase-amplitude solution
            self.__amplitude = interp1d(
                table[self.ilast - 1 :, 1], table[self.ilast - 1 :, 2], fill_value=0
            )
            self.__phase = interp1d(
                table[self.ilast - 1 :, 1], table[self.ilast - 1 :, 3], fill_value=0
            )

            # If continuum wave function also change normalization units from
            # 1/sqrt(k) in atomic units to units of 1/sqrt(Angstrom eV)
            # Hartree atomic energy unit in eV
            Eh = 27.211386245988
            # Fine structure constant
            alpha = 7.2973525693e-3
            # Convert energy to Hartree units
            eH = epsilon / Eh
            # wavenumber in atomic units
            ke = np.sqrt(2 * eH * (1 + alpha ** 2 * eH / 2))
            # Normalization
            norm = 1 / np.sqrt(ke)

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
    orb1,
    orb2,
    gridshape,
    gridsize,
    ml,
    mlprime,
    eV,
    bandwidth_limiting=2.0 / 3,
    qspace=False,
):
    """Evaluate an inelastic transition potential for excitation of an
       electron from orbital 1 to orbital 2 on grid with shape gridshape
       and real space dimensions in Angstrom given by gridsize"""

    # Bohr radius in Angstrom
    a0 = 5.29177210903e-1

    # Calculate energy loss
    deltaE = orb1.energy - orb2.energy
    # Calculate wave number in inverse Angstrom of incident and scattered
    # electrons
    k0 = wavev(eV)
    kn = wavev(eV + deltaE)

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
    if bandwidth_limiting is not None:
        qmax = np.amax(qabs) * bandwidth_limiting
    else:
        qmax = np.amax(qabs)

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
    # gamma = relativistic_mass_correction(eV)

    sigma = interaction_constant(eV + deltaE, units="rad/VA")

    # Need to multiply by area of k-space pixel (1/gridsize) and multiply by
    # pixels to get correct units from inverse Fourier transform
    Hn0 *= np.prod(gridshape) ** 1.5 / np.prod(gridsize)

    # Apply constants
    Hn0 = qe / 4 / np.pi ** 2 / eps0 * sigma * Hn0 / qabs ** 2 * np.sqrt(2)

    # Return result of Eq. (10) from Dwyer Ultramicroscopy 104 (2005) 141-151
    # in real space
    if not qspace:
        Hn0 = np.fft.ifft2(Hn0)

    return Hn0
    # return 2*np.pi**1.5/a0**3*np.fft.ifft2(Hn0/qabs**2,norm='ortho')
    # return 2*np.pi**1.5/a0**2*const*np.fft.ifft2(Hn0/qabs**2,norm='ortho')


def transition_potential_multislice(
    probes,
    nslices,
    subslices,
    propagators,
    transmission_functions,
    ionization_potentials,
    ionization_sites,
    tiling=[1, 1],
    device_type=None,
    seed=None,
    return_numpy=True,
    qspace_in=False,
    posn=None,
    image_CTF=None,
    threshhold=1e-4,
    showProgress=True,
):
    from .py_multislice import multislice
    from .utils.torch_utils import (
        amplitude,
        complex_mul,
        modulus_square,
        ensure_torch_array,
        fourier_shift_torch,
    )

    # Number of subslices
    nsubslices = len(subslices)

    # Get grid shape
    gridshape = np.asarray(transmission_functions.size()[-3:-1])

    # Total number of slices in multislice
    niterations = nslices * nsubslices

    if device_type is None:
        device = transmission_functions.device

    # Ensure pytorch arrays
    transmission_functions = ensure_torch_array(transmission_functions, device=device)
    dtype = transmission_functions.dtype
    ionization_potentials = ensure_torch_array(
        ionization_potentials, dtype=dtype, device=device
    )

    if image_CTF is not None:
        image_CTF = ensure_torch_array(image_CTF, dtype=dtype, device=device)
    propagators = ensure_torch_array(propagators, dtype=dtype, device=device)
    probes = ensure_torch_array(probes, dtype=dtype, device=device)

    if threshhold is not None:
        trigger = np.zeros((ionization_potentials.size(0),))
        for i, ionization_potential in enumerate(ionization_potentials):
            trigger[i] = (
                threshhold * modulus_square(ionization_potential) / np.prod(gridshape)
            )

    # Ionization potentials must be in reciprocal space
    ionization_potentials = torch.fft(ionization_potentials, signal_ndim=2)

    # Diffraction pattern
    if image_CTF is None:
        output = torch.zeros(*probes.size()[:-1], device=device, dtype=dtype)
    else:
        output = torch.zeros(*image_CTF.size()[:-1], device=device, dtype=dtype)

    # If Fourier space probes are passed, inverse Fourier transform them
    if qspace_in:
        probes = torch.ifft(probes, signal_ndim=2)

    # Loop over slices of specimens
    for i in tqdm.tqdm(range(niterations), desc="Slice", disable=not showProgress):

        subslice = i % nsubslices

        # Find inelastic transitions within this slice
        zmin = 0 if subslice == 0 else subslices[subslice - 1]
        atoms_in_slice = np.nonzero(
            (ionization_sites[:, 2] % 1.0 >= zmin)
            & (ionization_sites[:, 2] % 1.0 < subslices[subslice])
        )

        # Loop over inelastic transitions within the slice
        for atom in tqdm.tqdm(
            atoms_in_slice[0], desc="Transitions in slice", disable=not showProgress
        ):

            for j, ionization_potential in enumerate(ionization_potentials):

                # Calculate inelastically scattered wave for ionization transition
                # potential shifted to position of slice
                posn = (
                    torch.from_numpy(ionization_sites[atom, :2] * gridshape)
                    .type(dtype)
                    .to(device)
                )
                psi_n = complex_mul(
                    fourier_shift_torch(ionization_potential, posn, qspace_in=True),
                    probes,
                )

                # Only propagate this wave to the exit surface if it is deemed
                # to contribute significantly (above a user-determined threshhold)
                # to the image. Pass threshhold = None to disable this feature
                if threshhold is not None:
                    if modulus_square(psi_n) < trigger[j]:
                        continue

                # Propagate to exit surface
                psi_n = multislice(
                    psi_n,
                    np.arange(i, niterations),
                    propagators,
                    transmission_functions,
                    tiling=tiling,
                    qspace_out=True,
                    subslicing=True,
                    return_numpy=False,
                )

                # Perform imaging if requested, otherwise just accumulate diffraction
                # pattern
                if image_CTF is None:
                    output += amplitude(psi_n)
                else:
                    output += amplitude(
                        torch.ifft(complex_mul(psi_n, image_CTF), signal_ndim=2)
                    )

        # Propagate probe one slice
        probes = multislice(
            probes,
            [i + 1],
            propagators,
            transmission_functions,
            return_numpy=False,
            output_to_bandwidth_limit=False,
        )

    if return_numpy:
        return output.cpu().numpy()
    return output


def make_transition_potentials(
    gridshape,
    rsize,
    eV,
    Z,
    epsilon,
    boundQuantumNumbers,
    boundConfiguration,
    freeQuantumNumbers,
    freeConfiguration,
    qspace=False,
):

    boundOrbital = orbital(Z, boundConfiguration, *boundQuantumNumbers[:2])

    nstates = len(freeQuantumNumbers)
    Hn0 = np.zeros((nstates, *gridshape), dtype=np.complex)

    for istate, Qnumbers in enumerate(freeQuantumNumbers):
        freeOrbital = orbital(Z, freeConfiguration, 0, Qnumbers[0], epsilon=epsilon)

        # Calculate transition on same grid as multislice
        Hn0[istate] = transition_potential(
            boundOrbital,
            freeOrbital,
            gridshape,
            rsize,
            boundQuantumNumbers[2],
            Qnumbers[1],
            eV,
            bandwidth_limiting=None,
            qspace=qspace,
        )

    return Hn0


def tile_out_ionization_image(image, tiling):
    """For an ionization based image """
    tiled_image = np.zeros_like(image)
    for y in range(tiling[0]):
        for x in range(tiling[1]):
            tiled_image += fourier_shift(
                image, [y / tiling[0], x / tiling[1]], pixel_units=False
            )
    return tiled_image


if __name__ == "__main__":
    pass
