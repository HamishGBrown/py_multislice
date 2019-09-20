import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from re import split, match
from os.path import splitext
from .atomic_scattering_params import e_scattering_factors, atomic_symbol
from .utils.torch_utils import (
    sinc,
    roll_n,
    cx_from_numpy,
    cx_to_numpy,
    complex_mul,
    torch_c_exp,
    fourier_shift_array,
    amplitude
)


def Xray_scattering_factor(Z, gsq, units="A"):
    # Bohr radius in Angstrom
    a0 = 0.529177
    # gsq = g**2
    return Z - 2 * np.pi ** 2 * a0 * gsq * electron_scattering_factor(
        Z, gsq, units=units
    )


def electron_scattering_factor(Z, gsq, units="VA"):
    ai = e_scattering_factors[Z - 1, 0:10:2]
    bi = e_scattering_factors[Z - 1, 1:10:2]

    # Planck's constant in kg Angstrom/s
    h = 6.62607004e-24
    # Electron rest mass in kg
    me = 9.10938356e-31
    # Electron charge in Coulomb
    qe = 1.60217662e-19

    fe = np.zeros_like(gsq)

    for i in range(5):
        fe += ai[i] * (2 + bi[i] * gsq) / (1 + bi[i] * gsq) ** 2

    # Result can be returned in units of Volt Angstrom ('VA') or Angstrom ('A')
    if units == "VA":
        return h ** 2 / (2 * np.pi * me * qe) * fe
    elif units == "A":
        return fe


def interaction_constant(E, units="rad/VA"):
    """Calculates the interaction constant, sigma, to convert electrostatic
    potential (in V Angstrom) to radians. Units of this constant are rad/(V
    Angstrom).  See Eq. (2.5) in Kirkland's Advanced Computing in electron
    microscopy """
    # Planck's constant in kg Angstrom /s
    h = 6.62607004e-24
    # Electron rest mass in kg
    me = 9.10938356e-31
    # Electron charge in Coulomb
    qe = 1.60217662e-19
    # Electron wave number (reciprocal of wavelength) in Angstrom
    k0 = wavev(E)
    # Relativistic electron mass correction
    gamma = relativistic_mass_correction(E)
    if units == "rad/VA":
        return 2 * np.pi * gamma * me * qe / k0 / h / h
    elif units == "rad/A":
        return gamma / k0


class crystal:
    # Elements in a crystal object:
    # unitcell - An array containing the side lengths of the orthorhombic unit cell
    # atomtypes - A string array containing the symbols of atomic elements in the cell
    # natoms - Total number of atoms of each element within the cell
    # atoms - An array of dimensions total number of atoms by 6 which for each atom
    #        contains the fractional cooordinates within the unit cell for each atom
    #        in the first three entries, the atomic number in the fourth entry,
    #        the atomic occupancy (not yet implemented in the multislice) in the
    #        fifth entry and mean squared atomic displacement in the sixth entry
    #
    def __init__(self, fnam, temperature_factor_units="urms"):
        """Initializes a crystal object by reading in a *.p1 file, which is
        outputted by the vesta software:

        K. Momma and F. Izumi, "VESTA 3 for three-dimensional visualization of
        crystal, volumetric and morphology data," J. Appl. Crystallogr., 44,
        1272-1276 (2011).

        or a prismatic style *.xyz file
        """
        f = open(fnam, "r")

        ext = splitext(fnam)[1]

        # Read title
        self.Title = f.readline().strip()

        if ext == ".p1":

            # I have no idea what the second line in the p1 file format means
            # so ignore it
            f.readline()

            # Get unit cell vector - WARNING assume an orthorhombic unit cell
            self.unitcell = np.diag(np.loadtxt(f, max_rows=3, dtype=np.float))

            # Get the atomic symbol of each element
            self.atomtypes = np.loadtxt(f, max_rows=1, dtype=str, ndmin=1)

            # Get atomic number from lookup table
            Zs = [
                atomic_symbol.index(self.atomtypes[i].strip())
                for i in range(self.atomtypes.shape[0])
            ]

            # Get the number of atoms of each type
            self.natoms = np.loadtxt(f, max_rows=1, dtype=int, ndmin=1)

            # Skip empty line
            f.readline()

            # Total number of atoms
            totnatoms = np.sum(self.natoms)
            # Intialize array containing atomic information
            self.atoms = np.zeros((totnatoms, 6))

            for i in range(totnatoms):
                atominfo = split(r"\s+", f.readline().strip())[:6]
                self.atoms[i, :3] = np.asarray(atominfo[:3], dtype=np.float)
                self.atoms[i, 3] = atomic_symbol.index(
                    match("([A-Za-z]+)", atominfo[3]).group(0)
                )
                self.atoms[i, 4:6] = np.asarray(atominfo[4:6], dtype=np.float)
        elif ext == ".xyz":
            # Read in unit cell dimensions
            self.unitcell = np.asarray(
                [float(x) for x in split(r"\s+", f.readline().strip())[:3]]
            )

            atoms = []
            for line in f:

                # Look for end of file marker
                if line.strip() == "-1":
                    break
                # Otherwise parse line
                atoms.append(
                    np.array([float(x) for x in split(r"\s+", line.strip())[:6]])
                )

            # Now stack all atoms into numpy array
            atoms = np.stack(atoms, axis=0)

            # Rearrange columns of numpy array to match standard
            totnatoms = atoms.shape[0]
            self.atoms = np.zeros((totnatoms, 6))
            self.atoms[:, :3] = atoms[:, 1:4]
            self.atoms[:, 3] = atoms[:, 0]
            self.atoms[:, 4:6] = atoms[:, 4:6]

            # Convert atomic positions to fractional coordinates
            self.atoms[:, :3] /= self.unitcell[:3][np.newaxis, :]

        else:
            print("File extension: {0} not recognized".format(ext))
            return None
        # If temperature factors are given as B then convert to urms
        if temperature_factor_units == "B":
            self.atoms[:, 4:6] /= 8 * np.pi ** 2

    def quickplot(self, atomscale=0.01, cmap=plt.get_cmap("Dark2")):
        """Makes a quick 3D scatter plot of the crystal"""
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = cmap(self.atoms[:, 3] / np.amax(self.atoms[:, 3]))
        sizes = self.atoms[:, 3] ** (4) * atomscale
        ax.scatter(*[self.atoms[:, i] for i in range(3)], c=colors, s=sizes)
        plt.show(block=True)

    def make_transmission_functions(
        self,
        pixels,
        eV,
        subslices=[1.0],
        tiling=[1, 1],
        fe=None,
        displacements=True,
        fftout=True,
        device=None,
    ):
        """Make the transmission functions for this crystal, which are the
           exponential of the specimen potential scaled by the interaction
           constant for electrons, sigma."""

        # Make the specimen electrostatic potential
        T = self.make_potential(
            pixels, subslices, tiling, fe=fe, displacements=displacements, device=device
        )

        # Now take the complex exponential of the electrostatic potential
        # scaled by the electron interaction constant
        T = torch.fft(torch_c_exp(interaction_constant(eV) * T), signal_ndim=2)

        # Band-width limit the transmission function, see Earl Kirkland's book 
        # for an discussion of why this is necessary
        for i in range(T.shape[0]):
            T[i, ...] = bandwidth_limit_array(T[i, ...])

        if fftout:
            return torch.ifft(T, signal_ndim=2)
        return T

    def calculate_scattering_factors(self, pixels, tiling=[1, 1]):
        """Calculates the electron scattering factors on a reciprocal space
           grid of pixel size pixels assuming a unit cell tiling given by 
           tiling"""
        # Get real space and pixel dimensions of the array as numpy arrays
        rsize = np.asarray(self.unitcell[:2]) * np.asarray(tiling[:2])
        psize = np.asarray(pixels)

        # Get reciprocal space array
        g = q_space_array(psize, rsize)
        gsq = np.square(g[0]) + np.square(g[1])

        # Get a list of unique atomic elements
        elements = list(set(np.asarray(self.atoms[:, 3], dtype=np.int)))

        # Initialise scattering factor array
        fe = np.zeros((len(elements), *pixels), dtype=np.float32)

        # Loop over unique elements
        for ielement, element in enumerate(elements):
            fe[ielement, :, :] = electron_scattering_factor(element, gsq)

        return fe

    def make_potential(
        self,
        pixels,
        subslices=[1.0],
        tiling=[1, 1],
        bandwidth_limit=True,
        displacements=True,
        fe=None,
        device=None,
    ):
        """"Calculates the projected electrostatic potential for a 
            crystal on a pixel grid with dimensions specified by array 
            pixels. Subslicing the unit cell is achieved by passing an 
            array subslices that contains as its entries the depths at 
            which each subslice should be terminated in units of 
            fractional coordinates. Tiling of the unit cell (often
            necessary to make a sufficiently large simulation grid to 
            fit the probe) is achieved by passing the tiling factors in 
            the array tiling.
            """

        # Initialize device cuda if available, CPU if no cuda is available
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                device = torch.device("cpu")

        tiling_ = np.asarray(tiling[:2])
        gsize = np.asarray(self.unitcell[:2]) * tiling_
        psize = np.asarray(pixels)

        pixperA = np.asarray(pixels) / np.asarray(self.unitcell[:2]) / tiling_

        # Get a list of unique atomic elements
        elements = list(set(np.asarray(self.atoms[:, 3], dtype=np.int)))

        # Get number of unique atomic elements
        nelements = len(elements)
        nsubslices = len(subslices)

        # FDES method
        # Intialize potential array
        P = torch.zeros(
            np.prod([nelements, nsubslices, *pixels, 2]),
            device=device,
            dtype=torch.float,
        )

        # Construct a map of which atom corresponds to which slice
        islice = np.zeros((self.atoms.shape[0]), dtype=np.int)
        slice_stride = np.prod(pixels) * 2
        if nsubslices > 1:
            # Finds which slice atom can be found in
            # WARNING Assumes that the slices list ends with 1.0 and is in
            # ascending order
            for i in range(nsubslices):
                zmin = 0 if i == 0 else subslices[i - 1]
                atoms_in_slice = (self.atoms[:, 2] % 1.0 >= zmin) & (
                    self.atoms[:, 2] % 1.0 < subslices[i]
                )
                islice[atoms_in_slice] = i * slice_stride
            islice = torch.from_numpy(islice).type(torch.long).to(device)
        else:
            islice = 0
        # Make map a pytorch Tensor

        # Construct a map of which atom corresponds to which element
        element_stride = nsubslices * slice_stride
        ielement = (
            torch.tensor(
                [
                    elements.index(int(self.atoms[iatom, 3]))
                    for iatom in range(self.atoms.shape[0])
                ],
                dtype=torch.long,
                device=device,
            )
            * element_stride
        )

        if displacements:
            urms = torch.tensor(
                np.sqrt(self.atoms[:, 5])[:, np.newaxis] * pixperA[np.newaxis, :],
                dtype=P.dtype,
                device=device,
            ).view(self.atoms.shape[0], 2)

        # FDES algorithm implemented using the pytorch scatter_add function,
        # which takes a list of numbers and adds them to a corresponding list
        # of coordinates
        for tile in range(tiling[0] * tiling[1]):
            # For these atomic coordinates (in fractional coordinates) convert
            # to pixel coordinates
            posn = (
                (
                    self.atoms[:, :2]
                    + np.asarray([tile % tiling[0], tile // tiling[0]])[np.newaxis, :]
                )
                / tiling_
                * psize
            )
            posn = torch.from_numpy(posn).to(device).type(P.dtype)

            if displacements:
                # Add displacement sampled from normal distribution to account
                # for atomic thermal motion
                disp = (
                    torch.randn(self.atoms.shape[0], 2, dtype=P.dtype, device=device)
                    * urms
                )

                # print(disp)
                posn[:, :2] += disp

            yc = (
                torch.remainder(torch.ceil(posn[:, 0]).type(torch.long), pixels[0])
                * pixels[1]
                * 2
            )
            yf = (
                torch.remainder(torch.floor(posn[:, 0]).type(torch.long), pixels[0])
                * pixels[1]
                * 2
            )
            xc = torch.remainder(torch.ceil(posn[:, 1]).type(torch.long), pixels[1]) * 2
            xf = (
                torch.remainder(torch.floor(posn[:, 1]).type(torch.long), pixels[1]) * 2
            )

            yh = torch.remainder(posn[:, 0], 1.0)
            yl = 1.0 - yh
            xh = torch.remainder(posn[:, 1], 1.0)
            xl = 1.0 - xh

            # Each pixel is set to the overlap of a shifted rectangle in that pixel
            P.scatter_add_(0, ielement + islice + yc + xc, yh * xh)
            P.scatter_add_(0, ielement + islice + yc + xf, yh * xl)
            P.scatter_add_(0, ielement + islice + yf + xc, yl * xh)
            P.scatter_add_(0, ielement + islice + yf + xf, yl * xl)

        # Now view potential as a 4D array for next bit
        P = P.view(nelements, nsubslices, *pixels, 2)

        # FFT potential to reciprocal space
        P = torch.fft(P, signal_ndim=2)

        # Make sinc functions with appropriate singleton dimensions for pytorch
        # broadcasting /gridsize[0]*pixels[0] /gridsize[1]*pixels[1]
        sincy = (
            sinc(torch.from_numpy(np.fft.fftfreq(pixels[0])))
            .view([1, 1, pixels[0], 1, 1])
            .to(device)
            .type(P.dtype)
        )
        sincx = (
            sinc(torch.from_numpy(np.fft.fftfreq(pixels[1])))
            .view([1, 1, 1, pixels[1], 1])
            .to(device)
            .type(P.dtype)
        )
        # #Divide by sinc functions
        P /= sincy
        P /= sincx

        # Option to precalculate scattering factors and pass to program which
        # saves computation for
        if fe is None:
            fe_ = self.calculate_scattering_factors(pixels, tiling)
        else:
            fe_ = fe

        # Convolve with electron scattering factors using Fourier
        # convolution theorem
        # P = torch.ones(nelements,1,*pixels,2,device=device)
        P *= torch.from_numpy(fe_).view(nelements, 1, *pixels, 1).to(device)

        # Add atoms together
        P = (
            torch.sum(P, dim=0)
            / np.prod(self.unitcell[:2])
            / np.prod(tiling)
            * np.prod(pixels)
        )

        # Only return real part
        return torch.ifft(P, signal_ndim=2)[..., 0]


def q_space_array(pixels, gridsize):
    """Returns the appropriately scaled 2D reciprocal space array for pixel size
    given by pixels (#y pixels, #x pixels) and real space size given by gridsize
    (y size, x size)"""
    return np.meshgrid(
        *[np.fft.fftfreq(pixels[i], d=gridsize[i] / pixels[i]) for i in [1, 0]]
    )


def aberration(q, lam, df=0, cs=0, c5=0):
    """calculates the aberration function chi as a function of
    reciprocal space extent q for an electron with wavelength lam.

    Parameters
    ----------
    q : number
        reciprocal space extent (Inverse angstroms).
    lam : number
        wavelength of electron (Inverse angstroms).
    df : number
        Probe defocus (angstroms).
    cs : number
        Probe spherical aberration (mm).
    c5 : number
        Probe c5 coefficient (mm)."""
    p = lam * q
    chi = df * np.square(p) / 2.0 + cs * 1e7 * np.power(p, 4) / 4.0
    chi += c5 * 1e7 * np.power(p, 6) / 6.0
    return 2 * np.pi * chi / lam


def construct_illum(
    pix_dim,
    real_dim,
    eV,
    app,
    beam_tilt=[0, 0],
    aperture_shift=[0, 0],
    df=0,
    cs=0,
    c5=0,
    q=None,
    app_units="mrad",
    qspace=False,
    normalized=True,
):
    """Makes a probe wave function with pixel dimensions given in pix_dim
    and real_dimensions given by real_dim
    ---------
    pix_dim --- The pixel size of the grid
    real_dim --- The size of the grid in Angstrom
    keV --- The energy of the probe electrons in keVnews
    app --- The apperture in units specified by app_units
    df --- Probe defocus in A, a negative value indicate overfocus
    cs --- The 3rd order spherical aberration coefficient
    c5 --- The 5rd order spherical aberration coefficient
    app_units --- The units of the aperture size (A^-1 or mrad)"""

    npiy, npix = pix_dim[:2]
    y, x = real_dim[:2]
    if q == None:
        q = q_space_array(pix_dim, real_dim[:2])
    k = wavev(eV)

    if app_units == "mrad":
        app_ = np.tan(app / 1000.0) * k
    else:
        app_ = app

    probe = np.zeros(pix_dim, dtype=np.complex)

    qarray1 = np.sqrt(np.square(q[0] - beam_tilt[0]) + np.square(q[1] - beam_tilt[1]))
    qarray2 = np.sqrt(
        np.square(q[0] - beam_tilt[0] - aperture_shift[0])
        + np.square(q[1] - beam_tilt[1] - aperture_shift[1])
    )
    probe[qarray2 < app_] = np.exp(
        -1j * aberration(qarray1[qarray2 < app_], 1.0 / k, df, cs, c5)
    )
    if normalized:
        probe /= np.sqrt(np.sum(np.square(np.abs(probe))))

    # Return real or diffraction space probe depending on user preference
    if qspace:
        return probe
    else:
        return np.fft.ifft2(probe, norm="ortho")


def wavev(E):
    """Calculates the relativistically corrected wavenumber k0 (reciprocal of
    the wavelength) for an electron of energy eV. See Eq. (2.5) in Kirkland's
    Advanced Computing in electron microscopy"""
    # Planck's constant times speed of light in eV Angstrom
    hc = 1.23984193e4
    # Electron rest mass in eV
    m0c2 = 5.109989461e5
    return np.sqrt(E * (E + 2 * m0c2)) / hc


def relativistic_mass_correction(E):
    """Gives the relativistic mass correction, m/m0 or gamma, for an electron
    with kinetic energy given by E in eV. Eq. (2.2) in Kirkland's Advanced
     Computing in electron microscopy"""
    # Electron rest mass in eV
    m0c2 = 5.109989461e5
    return (m0c2 + E) / m0c2


def bandwidth_limit_array(array, limit=2 / 3):
    """Band-width limit an array to fraction of its maximum given by limit"""
    if isinstance(array, np.ndarray):
        pixelsize = array.shape[:2]
        array[
            (
                np.square(np.fft.fftfreq(pixelsize[0]))[:, np.newaxis]
                + np.square(np.fft.fftfreq(pixelsize[1]))[np.newaxis, :]
            )
            * (2 / limit) ** 2
            > 1
        ] = 0
    else:
        pixelsize = array.size()[:2]
        array[
            (
                torch.from_numpy(np.fft.fftfreq(pixelsize[0]) ** 2).view(
                    pixelsize[0], 1
                )
                + torch.from_numpy(np.fft.fftfreq(pixelsize[1]) ** 2).view(
                    1, pixelsize[1]
                )
            )
            * (2 / limit) ** 2
            > 1
        ] = 0

    return array


def make_propagators(pixelsize, gridsize, eV, subslices):
    # We will use the construct_illum function to generate the propagator, the
    # aperture of this propagator will supply the bandwidth limit of our simulation
    # it must be 2/3rds of our pixel gridsize
    app = np.amax(np.asarray(pixelsize) / np.asarray(gridsize[:2]) / 2)

    # Intitialize array
    prop = np.zeros((len(subslices), *pixelsize), dtype=np.complex)
    for islice, slice in enumerate(subslices):
        if islice == 0:
            deltaz = slice * gridsize[2]
        else:
            deltaz = (slice - subslices[islice - 1]) * gridsize[2]

        # Calculate propagator
        prop[islice, :, :] = bandwidth_limit_array(
            construct_illum(
                pixelsize,
                gridsize[:2],
                eV,
                app,
                df=deltaz,
                app_units="invA",
                qspace=True,
                normalized=False,
            )
        )

    return prop


def multislice(
    probes,
    propagators,
    transmission_functions,
    nslices,
    tiling=None,
    device_type=None,
    seed=None,
):
    """For a given probe or set of probes, propagators, and transmission 
        functions perform the multislice algorithm for nslices iterations."""

    # Seed random number generator, if None then the system clock will
    # be used as a seed
    r = np.random.RandomState(seed)

    # Initialize device cuda if available, CPU if no cuda is available
    if device_type is None and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_type is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device_type)

    # Since pytorch doesn't have a complex data type we need to add an extra
    # dimension of size 2 to each tensor that will store real and imaginary
    # components.
    if not isinstance(transmission_functions, torch.Tensor):
        T = cx_from_numpy(transmission_functions, device=device)
    else:
        T = transmission_functions
    if not isinstance(propagators, torch.Tensor):
        P = cx_from_numpy(propagators, device=device)
    else:
        P = propagators
    if not isinstance(probes, torch.Tensor):
        psi = cx_from_numpy(probes, device=device)
    else:
        psi = probes

    nT, nsubslices, nopiy, nopix = T.size()[:4]

    for slice in range(nslices):
        for subslice in range(nsubslices):
            # Pick random phase grating
            it = r.randint(0, nT)

            # Transmit and forward Fourier transform

            if tiling is None or (tiling[0] == 1 & tiling[1] == 1):
                psi = torch.fft(complex_mul(T[it, subslice, ...], psi), signal_ndim=2)
                # If the transmission function is from a tiled unit cell then
                # there is the option of randomly shifting it around to
                # generate "more" transmission functions
            elif nopiy % tiling[0] == 0 and nopix % tiling[1] == 0:
                # Shift an integer number of pixels in y
                T_ = roll_n(
                    T[it, subslice, ...],
                    0,
                    r.randint(0, tiling[0]) * (nopiy // tiling[0]),
                )

                # Shift an integer number of pixels in x
                T_ = roll_n(T_, 1, r.randint(1, tiling[1]) * (nopix // tiling[1]))

                # Perform transmission operation
                psi = torch.fft(complex_mul(T_, psi), signal_ndim=2)
            else:
                # Case of a non-integer shifting of the unit cell
                yshift = r.randint(0, tiling[0]) * (nopiy / tiling[0])
                xshift = r.randint(0, tiling[1]) * (nopix / tiling[1])
                shift = torch.tensor([yshift, xshift])

                # Generate an array to perform Fourier shift of transmission
                # function
                FFT_shift_array = fourier_shift_array([nopiy, nopix], shift)

                # Apply Fourier shift theorem for sub-pixel shift
                T_ = torch.ifft(
                    complex_mul(
                        FFT_shift_array, torch.fft(T[it, subslice, ...], signal_ndim=2)
                    ),
                    signal_ndim=2,
                )

                # Perform transmission operation
                psi = torch.fft(complex_mul(T_, psi), signal_ndim=2)

            # Propagate and inverse Fourier transform
            psi = torch.ifft(complex_mul(psi, P[subslice, ...]), signal_ndim=2)

    return cx_to_numpy(psi)

def STEM(rsize,probe,
    propagators,
    transmission_functions,
    nslices,eV,alpha,batch_size = 1,detectors = None,FourD_STEM=False,
    scan_posn=None,device = torch.device('cpu')
    tiling = [1,1],
    device_type=None,
    seed=None):

    from .Probe import nyquist_sampling

    gridshape = propagators.shape[-2:]

    # Generate scan positions if not supplied
    if scan_posn is None:

        # Calculate field of view of scan
        FOV = np.asarray(rsize)/np.asarray(tiling)

        # Calculate number of scan positions in STEM scan
        nscan = nyquist_sampling(FOV, eV=eV, alpha=alpha)

        #Get scan position in pixel coordinates
        scan_posn = []
        # Y scan coordinates
        scan_posn.append(np.arange(0,gridshape[0],step = gridshape[0]/nscan[0]))
        # X scan coordinates
        scan_posn.append(np.arange(0,gridshape[1],step = gridshape[1]/nscan[1]))

    # Total number of scan positions
    nscantot = scan_posn[0].shape[0]*scan_posn[1].shape[1]

    # Assume real space probe is passed in so perform Fourier transform in
    # anticipation of application of Fourier shift theorem
    probe_ = torch.fft(cx_from_numpy(probe,device=device),signal_ndim=2)

    # Work out whether to perform conventional STEM or not
    conventional_STEM = not detectors is None

    if conventional_STEM : 
        # Initialize array in which to store resulting STEM images
        STEM_image = np.zeros((detectors.shape[0],nscantot))
        
        # Also move detectors to pytorch if necessary
        if not isinstance(detectors, torch.Tensor):
            D = torch.from_numpy(detectors, device=device)
        else:
            D = detectors

    # Initialize array in which to store resulting 4D-STEM datacube
    if FourD_STEM : datacube = np.zeros((nscantot,*gridshape))
    

    # This algorithm allows for "batches" of probe to be sent through the 
    # multislice algorithm to achieve some speed up at the cost of storing more
    # probes in memory
    
    if (seed is None and batch_size>1) :
        # If no seed passed to random number generator then make one to pass to
        # the multislice algorithm. This ensure that each probe sees the same
        # frozen phonon configuration if we are doing batched multislice 
        # calculations
        seed = np.randint(0,2**32-1)

    for i in tqdm(range(int(np.ceil(nscantot/batch_size)))):

        # Make shifted probes
        scan_index = np.arange(i*batch_size,(i+1)*batch_size,dtype=np.int)
        # y scan is fast(est changing) scan direction
        yscan = scan_posn[0][scan_index%nscan[0]]
        # x scan is slow(est changing) scan direction
        xscan = scan_posn[1][scan_index//nscan[0]]

        # Shift probes using Fourier shift theorem, prepare shift operators
        # and store them in the array that the probes will eventually inhabit
        posn = torch.cat([torch.from_numpy(x).to(device) for x in [yscan,xscan]]
                         dim = 1)
        probes = fourier_shift_array(gridshape, posn, device=device)

        # Apply shift to original probe
        probes = torch.ifft(complex_mul(probe_.view(1,*probe_.size()),probes))

        #Perform multislice
        probes = amplitude(multislice(probes,propagators,transmission_functions,
                            nslices,tiling,device,seed))
        
        #Calculate STEM images 
        if conventional_STEM: 
            STEM_images[...,scan_index] = torch.sum(D.view(1,*D.size())*
                                            probes.view(batch_size,1,*gridshape),
                                            (-2,-1))
        #Store datacube
        if FourD_STEM : datacube[scan_index,...] = probes.cpu().numpy()

    if conventional_STEM and FourD_STEM: 
        return STEM_images.reshape(ndet,*nscan),Four_D_STEM.reshape(*nscan,)
    if FourD_STEM: return Four_D_STEM.reshape(*nscan,)
    if conventional_STEM: return STEM_images.reshape(ndet,*nscan)

def unit_cell_shift(array, axis, shift, tiles):
    """For an array consisting of a number of repeat units given by tiles
       shift than array an integer number of unit cells"""

    intshift = array.size(axis) // tiles
    integer_divisible = intshift == 0

    indices = torch.remainder(torch.arange(array.shape[-3 + axis]) - shift)
    if axis == 0:
        return array[indices, :, :]
    if axis == 1:
        return array[:, indices, :]


if __name__ == "__main__":
    sample = crystal("1000048.p1")
    sample.quickplot()
    sys.exit()
    from matplotlib import rc

    rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})

    # sample = crystal('1005012.p1')
    sample = crystal("SrTiO3.p1")
    # sample.atoms[:,5] = 0.05
    # subslices  = [0.132,0.196,0.236,0.362,0.4999,0.6378,0.7233,0.8,0.867,1.0]
    subslices = [1.0]
    gridsize = np.zeros((3))
    gridsize[:3] = sample.unitcell[:3]
    nT = 4
    tiling = [16, 16]
    pixsize = [512, 512]
    # pixsize = [128,128]
    eV = 300e3
    app = 24.0

    print("Setting up calculation")
    T = torch.zeros(nT, len(subslices), *pixsize, 2, dtype=torch.float)
    fe = sample.calculate_scattering_factors(pixsize, tiling)
    for i in range(nT):

        T[i, :, :, :] = sample.make_transmission_functions(
            pixsize, eV, subslices, tiling, fe=fe, displacements=True
        )

    fig = plt.figure(figsize=(2 * 4, 3 * 4))

    ax = fig.add_subplot(321)
    ax.imshow(np.angle(cx_to_numpy(T[1, 0, :, :])))
    ax = fig.add_subplot(322)
    ax.imshow(np.angle(cx_to_numpy(T[1, 0, :, :])))
    # for i in range(1): Image.fromarray(np.abs(np.fft.fft2(cx_to_numpy(T[1,i,:,:])))).save('T_{0}.tif'.format(i))
    # plt.show()
    # sys.exit()
    gridsize[:2] = gridsize[:2] * np.asarray(tiling)
    P = make_propagators(pixsize, gridsize, eV, subslices)
    ax = fig.add_subplot(323)
    ax.imshow(np.fft.fftshift(np.imag(P[0, :, :])))
    ax = fig.add_subplot(324)
    ax.imshow(np.fft.fftshift(np.real(P[0, :, :])))
    probe = construct_illum(pixsize, gridsize[:2], eV, app)
    # probe =  np.ones(pixsize,dtype=np.complex)/np.sqrt(np.prod(pixsize))
    # print(np.sum(np.square(np.abs(probe))))
    ax = fig.add_subplot(325)
    # ax.imshow(colorize(probe))
    ax.imshow(np.square(np.abs(probe)))
    # print(np.argmax(np.square(np.abs(probe))))
    # sys.exit()
    # print(np.sum(np.square(np.abs(probe))))
    # plt.show()
    # print('Performing multislice')
    exit_wave = np.zeros(pixsize)
    from tqdm import tqdm
    from PIL import Image

    nfph = 50
    t = 100
    # Image.fromarray(exit_wave).save('cbed.tiff')
    for i in tqdm(range(nfph)):

        exit_wave += (
            np.abs(
                np.fft.fft2(
                    multislice(
                        probe, P, T, gridsize, int(np.ceil(t / 3.905)), tiling=tiling
                    )
                )
            )
            ** 2
            / nfph
        )
        # fig,ax = plt.subplots()
        # ax.imshow(exit_wave)
        # plt.show()
    Image.fromarray(np.fft.fftshift(exit_wave) / np.prod(pixsize)).save("cbed.tiff")
    ax = fig.add_subplot(326)
    # ax.imshow(np.fft.fftshift(np.square(np.abs((np.fft.fft2(exit_wave,norm='ortho'))))))
    plt.show()
