"""
Functions for calculating ionization cross sections.

For ionization based TEM methods such as electron energy-loss spectroscopy (EELS)
or energy-filtered TEM (EFTEM).
"""
import numpy as np
import matplotlib.pyplot as plt
import re
import scipy.integrate as integrate
import torch
import tqdm
from .Probe import wavev, relativistic_mass_correction
from .utils.numpy_utils import fourier_shift

# List of letters for each orbital, used to convert between orbital angular
# momentum quantum number ell and letter
orbitals = ["s", "p", "d", "f"]

# Bohr radius in Angstrom
a0 = 5.29177210903e-1

# Rydberg energy in electron volts
Ry = 13.605693122994


def get_q_numbers_for_transition(ell, order=1):
    """
    Calculate set of quantum numbers for excited states.

    For ionization from bound quantum number l, calculate all excited
    state quantum numbers ml, lprime, and mlprime needed to calculate all
    atomic transitions.

    Parameters
    ----------
    ell : int
        Target orbital angular momentum quantum number
    order : int, optional
        Largest change in orbital angular momentum quantum number, order = 1
        gives all dipole terms, order = 2 gives all quadropole terms etc.
    """
    # Get projection quantum numbers
    mls = np.arange(-ell, ell + 1)
    qnumbers = []
    minlprime = max(ell - order, 0)
    for lprime in np.arange(minlprime, ell + order + 1):
        for mlprime in np.arange(-lprime, lprime + 1):
            for ml in mls:
                qnumbers.append([lprime, mlprime, ml])
    return qnumbers


def get_transitions(Z, n, ell, epsilon, eV, gridshape, gridsize, order=1, contr=0.95):
    """
    Calculate all transitions for a particular target orbital.

    Parameters
    ----------
    Z : int
        Target atomic number
    n : int
        Target orbital principle quantum number
    ell : int
        Target orbital angular momentum quantum number
    epsilon : Optional, float
        Energy of continuum wavefunction, ie energy above ionization threshhold
    eV : float
        Probe energy in electron volts
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    gridsize : (2,) array_like
        The real space size of the simulation grid in Angstrom
    Keyword arguments
    -----------------
    order : int
        Largest change in orbital angular momentum quantum number, order = 1
        gives all dipole terms, order = 2 gives all quadropole terms etc.
    contr : float
        Threshhold for rejection of ionization transition potential, eg. if
        contr == 0.95 an individual transition is rejected if it would
        contribute less than 5 % to the total signal
    """
    # Get orbital configuration in bound state
    orbital_configuration = full_orbital_filling(Z)

    # Free configuration is the bound orbital with one less electron, find this
    # orbital in the string and parse its current filling
    target_orbital_string = str(n) + orbitals[ell]
    current_filling = int(
        re.search(target_orbital_string + "([0-9]+)", orbital_configuration).group(1)
    )

    # Subtract one electron to get the new filling
    new_filling = current_filling - 1

    # Update the orbital configuration string to create the new orbital filling
    new_orbital_string = target_orbital_string + str(new_filling)
    target_orbital_string = target_orbital_string + str(current_filling)
    excited_configuration = orbital_configuration.replace(
        target_orbital_string, new_orbital_string
    )

    # Now generate the bound_orbital object using pfac
    bound_orbital = orbital(Z, orbital_configuration, n, ell)

    qnumberset = get_q_numbers_for_transition(bound_orbital.ell, order)

    transition_potentials = []

    # Loop over all excited states of interest
    for qnumbers in tqdm.tqdm(qnumberset, desc="Calculating transition potentials"):
        lprime, mlprime, ml = qnumbers

        # Generate orbital for excited state using pfac
        excited_state = orbital(
            bound_orbital.Z, excited_configuration, 0, lprime, epsilon
        )

        # Calculate transition potential for this escited state
        Hn0 = transition_potential(
            bound_orbital, excited_state, gridshape, gridsize, ml, mlprime, eV
        )

        transition_potentials.append(Hn0)

    tot = np.sum(np.square(np.abs(transition_potentials)))

    # Reject orbitals which fall below the user-supplied threshhold
    return np.stack(
        [
            Hn0
            for Hn0 in transition_potentials
            if np.sum(np.abs(Hn0) ** 2) / tot > 1 - contr
        ]
    )


class orbital:
    """
    A class for storing the results of a fac atomic structure calculation.

    When initialized this class will calculate the wave function of a bound
    electron using the flexible atomic code (fac) atomic structure code and
    store the necessary information about the radial electron wave function.
    """

    def __init__(self, Z: int, config: str, n: int, ell: int, epsilon=1):
        """
        Initialize the orbital class and return an orbital object.

        Parameters
        ----------
        Z : int
            Atomic number
        config : str
            String describing configuration of atom ie:
            carbon (C): config = '1s2 2s2 2p2'
        n : int
            Principal quantum number of orbital, for continuum wavefunctions
            pass n=0
        ell : int
            Orbital angular momentum quantum number of orbital
        epsilon : Optional, float
            Energy of continuum wavefunction in eV (only matters if n == 0)
        """
        # Load arguments into orbital object
        self.Z = Z
        self.config = config
        self.n = n
        self.ell = ell
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
            angmom = ["s", "p", "d", "f"][ell]
            # Title in the format "Ag 1s", "O 2s" etc..
            self.title = "{0} {1}{2}".format(pfac.fac.ATOMICSYMBOL[Z], n, angmom)
        else:
            # Continuum wave function case
            # Title in the format "Ag e = 10 eV l'=2" etc..
            self.title = "{0} \\varepsilon = {1} l' = {2}".format(
                pfac.fac.ATOMICSYMBOL[Z], epsilon, ell
            )

        # Calculate relativstic quantum number from
        # non-relativistic input
        kappa = -1 - ell

        # Output desired wave function from table
        pfac.fac.WaveFuncTable("orbital.txt", n, kappa, epsilon)

        # Clear table
        # ClearOrbitalTable ()
        pfac.fac.Reinit(config=1)

        with open("orbital.txt", "r") as content_file:
            content = content_file.read()

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
            # Normalization used in flexible atomic code
            facnorm = 1 / np.sqrt(ke)
            # Desired normalization from Manson 1972
            norm = 1 / np.sqrt(np.pi * epsilon / Ry)
            # norm = 1

            # If continuum wave function load phase-amplitude solution
            self.__amplitude = interp1d(
                table[self.ilast - 1 :, 1],
                table[self.ilast - 1 :, 2] / facnorm * norm,
                fill_value=0,
            )
            self.__phase = interp1d(
                table[self.ilast - 1 :, 1], table[self.ilast - 1 :, 3], fill_value=0
            )
            self.wfn_table *= norm / facnorm

        # For bound wave functions we simply interpolate the
        # tabulated values of a0 the wavefunction
        # *2
        self.__wfn = interp1d(
            table[: self.ilast, 1], table[: self.ilast, 4], kind="cubic", fill_value=0
        )

    def __call__(self, r):
        """Evaluate radial wavefunction on grid r from tabulated values."""
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

    def plot(self, grid=None, show=True, ylim=None):
        """Plot wavefunction at positions given by grid r in Bohr radii."""
        fig, ax = plt.subplots(figsize=(4, 4))
        if grid is None:
            rmax = max(2.0, self.r[self.ilast - 1])
            grid = np.linspace(0.0, rmax, num=50)

        wavefunction = self(grid)
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
    """
    Calculate an ionization transition potential.

    Evaluate an inelastic transition potential for excitation of an electron
    from orbital orb1 to orbital orb2 on grid with shape gridshape and real space
    dimensions in Angstrom given by gridsize

    Parameters
    ----------
    orb1 : class pyms.Ionization.orbital
        The bound state orbital object
    orb1 : class pyms.Ionization.orbital
        The excited state orbital object
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    gridsize : (2,) array_like
        The real space size of the simulation grid in Angstrom
    ml : int
        The angular angular momentum projection quantum number of the bound
        state
    mlprime : int
        The angular angular momentum projection quantum number of the excited
        state
    eV : float
        The energy of the probe electron

    Keyword arguments
    -----------------
    bandwidth_limiting : float
        The band-width limiting as a fraction of the grid of the excitation
    qspace : bool
        If qspace = True return the ionization transition potential in reciprocal
        space
    """
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
    qq = [gridshape[i] / gridsize[i] for i in range(2)]
    deltaq = [1 / gridsize[i] for i in range(2)]
    qgrid = [np.fft.fftfreq(gridshape[1]) * qq[0], np.fft.fftfreq(gridshape[0]) * qq[1]]

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
    ell = int(orb1.ell)
    lprime = int(orb2.ell)

    # Check that ml and mlprime, the projection quantum numbers for the bound
    # and free states, are sensible
    #  see http://mathworld.wolfram.com/Wigner3j-Symbol.html)
    if np.abs(ml) > ell:
        return Hn0
    if np.abs(mlprime) > lprime:
        return Hn0

    def ovlap(q, lprimeprime):
        """Overlap jn weighted integral of orbital wave functions."""
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
            grid = 2 * np.pi * Q * a0
            overlap_kernel = (
                lambda x: orb1(x) * spherical_jn(lprimeprime, grid * x) * orb2(x)
            )
            jq[iQ] = integrate.quad(overlap_kernel, 0, rmax)[0]

        # Bound wave function was in units of 1/sqrt(bohr-radii) and excited
        # wave function was in units of 1/sqrt(bohr-radii Rydbergs) integration
        # was performed in borh-radii units so result is 1/sqrt(Rydbergs)
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
    lprimeprimes = np.arange(
        np.abs(ell - lprime), np.abs(ell + lprime) + 1, dtype=np.int
    )
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
            * np.sqrt((2 * lprime + 1) * (2 * lprimeprime + 1) * (2 * ell + 1))
        )
        for mlprimeprime in mlprimeprimes:
            # Check second selection rule
            # (http://mathworld.wolfram.com/Wigner3j-Symbol.html)
            if ml - mlprime - mlprimeprime != 0:
                continue
            # Evaluate Eq (14) from Dwyer Ultramicroscopy 104 (2005) 141-151
            prefactor2 = (
                (-1.0) ** (mlprime + mlprimeprime)
                * np.float(wigner_3j(lprime, lprimeprime, ell, 0, 0, 0))
                * np.float(
                    wigner_3j(lprime, lprimeprime, ell, -mlprime, -mlprimeprime, ml)
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

    # Need to multiply by area of k-space pixel (1/gridsize) and multiply by
    # pixels to get correct units from inverse Fourier transform (since an
    # inverse Fourier transform implicitly divides by gridshape)
    Hn0 *= np.prod(gridshape) / np.prod(gridsize)

    # Multiply by orbital filling
    # filling = 4*ell+2

    # Apply constants:
    # sqrt(Rdyberg) to convert to 1/sqrt(eV) units
    # 1 / (2 pi**2 a0 kn) as as per paper
    # Relativistic mass correction to go from a0 to relativistically corrected a0*
    # divide by q**2
    Hn0 *= (
        np.sqrt(2)
        * relativistic_mass_correction(eV)
        / (2 * a0 * np.pi ** 2 * np.sqrt(Ry) * kn * qabs ** 2)
    )

    # Return result of Eq. (10) from Dwyer Ultramicroscopy 104 (2005) 141-151
    # in real space
    if not qspace:
        Hn0 = np.fft.ifft2(Hn0)

    return Hn0


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
    qspace_out=True,
    posn=None,
    image_CTF=None,
    threshhold=1e-4,
    showProgress=True,
):
    """
    Perform a multislice calculation with a transition potential for ionization.

    Parameters
    ----------
    probes : (n,Y,X) complex array_like
        Electron wave functions for a set of input probes
    nslices : int, array_like
        The number of slices (iterations) to perform multislice over
    propagators : (N,Y,X,2) torch.array
        Fresnel free space operators required for the multislice algorithm
        used to propagate the scattering matrix
    transmission_functions : (N,Y,X,2)
        The transmission functions describing the electron's interaction
        with the specimen for the multislice algorithm
    ionization_potentials :

    qspace_out : bool
        Does nothing, purely there to match the calling signature of the STEM
        function.
    """
    from .py_multislice import multislice
    from .utils.torch_utils import (
        amplitude,
        complex_mul,
        modulus_square,
        ensure_torch_array,
        fourier_shift_torch,
        get_device,
    )

    device = get_device(device_type)
    # Number of subslices
    nsubslices = len(subslices)

    # Get grid shape
    gridshape = np.asarray(transmission_functions.size()[-3:-1])

    # Total number of slices in multislice
    niterations = nslices * nsubslices

    # Ensure pytorch arrays
    transmission_functions = ensure_torch_array(transmission_functions)

    dtype = transmission_functions.dtype
    ionization_potentials = ensure_torch_array(
        ionization_potentials, dtype=dtype, device=device
    )

    if image_CTF is not None:
        image_CTF = ensure_torch_array(image_CTF, dtype=dtype, device=device)
    propagators = ensure_torch_array(propagators, dtype=dtype, device=device)
    probes = ensure_torch_array(probes, dtype=dtype, device=device)

    # If Fourier space probes are passed, inverse Fourier transform them
    if qspace_in:
        probes = torch.ifft(probes, signal_ndim=2)

    # Calculate threshholds below which an ionization will not be included in
    # the simulation.
    if threshhold is not None:
        trigger = np.zeros((ionization_potentials.size(0),))
        for i, ionization_potential in enumerate(ionization_potentials):
            print(modulus_square(ionization_potential))
            trigger[i] = threshhold * modulus_square(ionization_potential)

    # Ionization potentials must be in reciprocal space
    ionization_potentials = torch.fft(ionization_potentials, signal_ndim=2)

    # Output array
    from .utils.torch_utils import size_of_bandwidth_limited_array

    nprobes = probes.size(0)
    gridout = size_of_bandwidth_limited_array(probes.size()[-3:-1])
    output = torch.zeros(nprobes, *gridout, device=device, dtype=dtype)

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
                print(
                    modulus_square(psi_n),
                    modulus_square(
                        fourier_shift_torch(ionization_potential, posn, qspace_in=True)
                    ),
                    trigger[j],
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


def tile_out_ionization_image(image, tiling):
    """
    Tile out a ionization based image.

    To save time, only ionizations in a single repeat unit cell are simulated.
    This routine tiles out the result from this unit cell to all other unit
    cells
    """
    tiled_image = np.zeros_like(image)
    for y in range(tiling[0]):
        for x in range(tiling[1]):
            tiled_image += fourier_shift(
                image, [y / tiling[0], x / tiling[1]], pixel_units=False
            )
    return tiled_image


def valence_orbitals(Z):
    """Return the valence orbital filling for an atom with atomic number Z."""
    if Z < 3:
        return "1s" + str(Z)
    elif Z < 5:
        return "2s" + str(Z - 2)
    elif Z < 11:
        return "2s2 2p" + str(Z - 4)
    elif Z < 13:
        return "3s" + str(Z - 10)
    elif Z < 19:
        return "3s2 3p" + str(Z - 12)
    elif Z < 21:
        return "4s" + str(Z - 18)
    elif Z == 24:
        return "3d5 4s1"
    elif Z == 29:
        return "3d10 4s1"
    elif Z < 31:
        return "3d" + str(Z - 20) + " 4s2"
    elif Z < 37:
        return "3d10 4s2 4p" + str(Z - 30)
    elif Z < 39:
        return "5s" + str(Z - 36)
    elif Z == 41:
        return "4d4 5s1"
    elif Z == 42:
        return "4d5 5s1"
    elif Z == 44:
        return "4d7 5s1"
    elif Z == 45:
        return "4d8 5s1"
    elif Z == 46:
        return "4d10"
    elif Z == 47:
        return "4d10 5s1"
    elif Z < 49:
        return "4d" + str(Z - 38) + " 5s2"
    elif Z < 55:
        return "4d10 5s2 5p" + str(Z - 48)
    elif Z < 57:
        return "6s" + str(Z - 54)
    elif Z == 57:
        return "5d1 6s2"
    elif Z == 58:
        return "4f1 5d1 6s2"
    elif Z == 64:
        return "4f7 5d1 6s2"
    elif Z < 71:
        return "4f" + str(Z - 56) + " 6s2"
    # Filling for Pt and Au
    elif any([Z == 78, Z == 79]):
        return "4f14 5d" + str(Z - 69) + " 6s1"
    elif Z < 81:
        return "4f14 5d" + str(Z - 70) + " 6s2"
    elif Z < 87:
        return "4f14 5d10 6s2 6p" + str(Z - 80)
    elif Z < 89:
        return "7s" + str(Z - 86)
    # Filling for Ac and Th
    elif Z in [89, 90]:
        return "7s2 6d" + str(Z - 88)
    # Filling for Pa, U, Np and Cm
    elif Z in [91, 92, 93, 96]:
        return "7s2 5f" + str(Z - 89) + " 6d1"
    elif Z < 103:
        return "7s2 5f" + str(Z - 88)
    else:
        return None


def noble_gas_filling(Z):
    """Return the noble gas filling for an atom with atomic number Z."""
    if Z < 2:
        return ""
    orb = "1s2 "
    if Z < 10:
        return orb
    orb += "2s2 2p6 "
    if Z < 18:
        return orb
    orb += "3s2 3p6 "
    if Z < 36:
        return orb
    orb += "4s2 3d10 4p6 "
    if Z < 54:
        return orb
    orb += "5s2 4d10 5p6 "
    if Z < 71:
        return orb
    return None


def full_orbital_filling(Z):
    """Return the full orbital filling for an atom with atomic number Z."""
    return noble_gas_filling(Z) + valence_orbitals(Z)


if __name__ == "__main__":
    pass
