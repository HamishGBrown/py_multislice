import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from re import split, match
from os.path import splitext
from .atomic_scattering_params import e_scattering_factors, atomic_symbol
from .crystal import crystal
from .utils.numpy_utils import (
    bandwidth_limit_array,
    q_space_array,
    fourier_interpolate_2d,
    crop,
)
from .utils.torch_utils import (
    sinc,
    roll_n,
    cx_from_numpy,
    cx_to_numpy,
    complex_matmul,
    complex_mul,
    torch_c_exp,
    fourier_shift_array,
    amplitude,
    get_device,
    ensure_torch_array
)


def make_propagators(pixelsize, gridsize, eV, subslices=[1.0]):
    """Make the Fresnel freespace propagators for a multislice simulation.

    Keyword arguments:
    pixelsize -- Pixel dimensions of the 2D grid
    gridsize  -- Size of the grid in real space (first two dimensions) and
                 thickness of the object in multislice (third dimension)
    eV        -- Probe energy in electron volts
    subslices -- A one dimensional array-like object containing the depths
                 (in fractional coordinates) at which the object will be
                 subsliced. The last entry should always be 1.0. For example,
                 to slice the object into four equal sized slices pass
                 [0.25,0.5,0.75,1.0]
    """
    from .Probe import make_contrast_transfer_function

    # We will use the make_contrast_transfer_function function to generate the propagator, the
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
            make_contrast_transfer_function(
                pixelsize, gridsize[:2], eV, app, df=deltaz, app_units="invA"
            )
        )

    return prop


def multislice_cupy(
    probes,
    propagators,
    transmission_functions,
    nslices,
    tiling=None,
    device_type="GPU",
    seed=None,
    return_numpy=True,
):
    """For a given probe or set of probes, propagators, and transmission 
        functions perform the multislice algorithm for nslices iterations."""

    # Seed random number generator, if None then the system clock will
    # be used as a seed
    r = np.random.RandomState(seed)

    # If GPU calculations are requested then import the cupy library
    # otherwise import numpy (useful feature of numpy and cupy having very
    # the same function names)
    if device_type == "GPU":
        import cupy as cp
    else:
        import numpy as cp

    # Since pytorch doesn't have a complex data type we need to add an extra
    # dimension of size 2 to each tensor that will store real and imaginary
    # components.
    T = cp.asarray(transmission_functions)
    P = cp.asarray(propagators)
    psi = cp.asarray(probes)

    nT, nsubslices, nopiy, nopix = T.shape()[:4]

    for slice in range(nslices):
        for subslice in range(nsubslices):
            # Pick random phase grating
            it = r.randint(0, nT)

            # Transmit and forward Fourier transform

            if tiling is None or (tiling[0] == 1 & tiling[1] == 1):
                psi = cp.fft.fft2(T[it, subslice, ...] * psi)
                # If the transmission function is from a tiled unit cell then
                # there is the option of randomly shifting it around to
                # generate "more" transmission functions
            elif nopiy % tiling[0] == 0 and nopix % tiling[1] == 0:
                # Shift an integer number of pixels in y and x
                T_ = cp.roll(
                    T[it, subslice, ...],
                    (
                        r.randint(0, tiling[0]) * (nopiy // tiling[0]),
                        r.randint(1, tiling[1]) * (nopix // tiling[1]),
                    ),
                    axis=(-2, -1),
                )

                # Perform transmission operation
                psi = cp.fft.fft2(T_ * psi)
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

    if return_numpy:
        return cx_to_numpy(psi)
    return psi




def multislice(
    probes,
    nslices,
    propagators,
    transmission_functions,
    tiling=None,
    device_type=None,
    seed=None,
    return_numpy=True,
    qspace_in=False,
    qspace_out=False,
    posn=None,
    subslicing=False,
):
    """For a given probe or set of probes, propagators, and transmission 
        functions perform the multislice algorithm for nslices iterations."""

    # Seed random number generator, if None then np.random.RandomState will
    # use the system clock as a seed
    r = np.random.RandomState(seed)

    # Initialize device cuda if available, CPU if no cuda is available
    device = get_device(device_type)

    # Since pytorch doesn't have a complex data type we need to add an extra
    # dimension of size 2 to each tensor that will store real and imaginary
    # components.
    T = ensure_torch_array(transmission_functions, device=device)
    P = ensure_torch_array(propagators, dtype=T.dtype, device=device)
    psi = ensure_torch_array(probes, dtype=T.dtype, device=device)

    nT, nsubslices, nopiy, nopix = T.size()[:4]

    # Probe needs to start multislice algorithm in real space (warning
    # this is not a normalized fft)
    if qspace_in:
        psi = torch.ifft(psi, signal_ndim=2)  # ,normalized=True)

    if isinstance(nslices, int):
        # Number of iterations of the multslice algorithm
        niterations = nslices if subslicing else nslices * nsubslices
        slices = np.arange(niterations)
    else:
        slices = nslices

    for i,islice in enumerate(slices):
        subslice = islice % nsubslices

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
                T[it, subslice, ...], 0, r.randint(0, tiling[0]) * (nopiy // tiling[0])
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
            FFT_shift_array = fourier_shift_array(
                [nopiy, nopix], shift, dtype=T.dtype, device=T.device
            )

            # Apply Fourier shift theorem for sub-pixel shift
            T_ = torch.ifft(
                complex_mul(
                    FFT_shift_array, torch.fft(T[it, subslice, ...], signal_ndim=2)
                ),
                signal_ndim=2,
            )

            # Perform transmission operation
            psi = torch.fft(complex_mul(T_, psi), signal_ndim=2)

        # Propagate
        psi = complex_mul(psi, P[subslice, ...])
        
        # This logic statement allows for the probe to be returned
        # in reciprocal space
        if not (qspace_out and i == len(slices)-1):
            psi = torch.ifft(psi, signal_ndim=2)

    if return_numpy:
        return cx_to_numpy(psi)
    return psi


def make_detector(gridshape, rsize, eV, betamax, betamin=0, units="mrad"):
    """Make a STEM detector with acceptance angle between betamin and betamax"""

    from .Probe import wavev

    # Get reciprocal space array
    q = q_space_array(gridshape, rsize)

    # If units are mrad convert qspace array from inverse Angstrom to mrad
    if units == "mrad":
        q /= wavev(eV) / 1000

    # Calculate modulus square of reciprocal space array
    qsq = np.square(q[0]) + np.square(q[1])

    # Make detector
    detector = np.logical_and(qsq < betamax ** 2, qsq >= betamin ** 2)

    # Convert logical to integer
    return np.where(detector, 1, 0)


def generate_STEM_raster(gridshape, FOV, eV, alpha, tiling=[1, 1]):
    """For field of view FOV and a focused electron probe of energy eV in
    electron volts and probe forming aperture alpha generate the STEM scan
    raster positions in units of pixels. Option of scanning only a single unit
    cell if tiling = 1 is passed"""

    # Calculate number of scan positions in STEM scan
    from .Probe import nyquist_sampling

    nscan = nyquist_sampling(FOV / np.asarray(tiling), eV=eV, alpha=alpha)

    # Generate Y and X scan coordinates
    return [
        np.arange(0, gridshape[i] / tiling[i], step=gridshape[i] / nscan[i] / tiling[i])
        for i in range(2)
    ]


def multislice_STEM(
    rsize,
    probe,
    propagators,
    transmission_functions,
    nslices,
    eV,
    alpha,
    batch_size=1,
    detectors=None,
    FourD_STEM=False,
    datacube=None,
    scan_posn=None,
    dtype=None,
    device=None,
    tiling=[1, 1],
    seed=None,
    showProgress=True,
):
    """Perform a STEM simulation using only the multislice algorithm"""
    if dtype is None:
        dtype = transmission_functions.dtype
    if device is None:
        device = transmission_functions.device

    method = multislice
    args = (propagators, transmission_functions, tiling, device, seed)
    kwargs = {"return_numpy": False, "qspace_in": True, "qspace_out": True}

    return STEM(
        rsize,
        probe,
        method,
        nslices,
        eV,
        alpha,
        batch_size=batch_size,
        detectors=detectors,
        FourD_STEM=FourD_STEM,
        datacube=datacube,
        scan_posn=scan_posn,
        device=device,
        tiling=tiling,
        seed=seed,
        showProgress=showProgress,
        method_args=args,
        method_kwargs=kwargs,
    )


def STEM(
    rsize,
    probe,
    method,
    nslices,
    eV,
    alpha,
    S=None,
    batch_size=1,
    detectors=None,
    FourD_STEM=False,
    datacube=None,
    scan_posn=None,
    dtype=torch.float32,
    device=torch.device("cpu"),
    tiling=[1, 1],
    seed=None,
    showProgress=True,
    method_args=(),
    method_kwargs={},
):
    """Perform a STEM image simulation.

    Keyword arguments:
    rsize       -- The real space size of the grid in Angstroms
    probe       -- The probe that will be rastered over the object
    method      -- A function that takes a probe and propagates it to the exit
                   surface of the specimen
    nslices     -- The number of slices to perform multislice over
    eV          -- Accelerating voltage of the probe, needed to work out probe
                   sampling requirements
    alpha       -- The convergence angle of the probe in mrad, needed to work 
                   out probe sampling requirements
    S           -- A scattering matrix object to perform STEM simulations with
                   using the PRISM algorithm (optional)
    batch_size  -- Number of probes to perform multislice on simultaneously
    detectors   -- Diffraction plane detectors to perform conventional STEM
                    imaging with
    fourD_STEM  -- Pass fourD_STEM = True to perform 4D-STEM simulations. To
                    save disk space a tuple containing pixel size and 
                    diffraction space extent of the datacube can be passed in.
                    For example ([64,64],[1.2,1.2]) will output diffraction 
                    patterns measuring 64 x 64 pixels and 1.2 x 1.2 inverse
                    Angstroms.
    scan_posn   -- Tuple containing arrays of y and x scan positions, overrides
                    internal calculations of STEM sampling
    device      -- torch.device object which will determine which device (CPU
                    or GPU) the calculations will run on
    tiling      -- Tiling of the simulation object on the grid
    seed        -- Seed for the random number generator for frozen phonon 
                    configurations
    showProgress-- Pass showProgress=False to disable progress bar.
    """

    # Get number of thicknesses in the series
    nthick = len(nslices)

    # Get shape of grid
    gridshape = probe.shape[-2:]

    # Datatype (precision) is inferred from transmission functions
    # dtype = transmission_functions.dtype

    # Device (CPU or GPU) is also inferred from transmission functions
    # if device is None:
    #     device = transmission_functions.device

    # Generate scan positions if not supplied
    if scan_posn is None:
        scan_posn = generate_STEM_raster(gridshape, rsize[:2], eV, alpha, tiling)

    # Number of scan positions
    nscan = [x.size for x in scan_posn]
    nscantot = np.prod(nscan)

    # Assume real space probe is passed in so perform Fourier transform in
    # anticipation of application of Fourier shift theorem
    probe_ = torch.fft(
        ensure_torch_array(probe, dtype=dtype, device=device), signal_ndim=2
    )

    # Work out whether to perform conventional STEM or not
    conventional_STEM = not detectors is None

    if conventional_STEM:
        # Get number of detectors
        ndet = detectors.shape[0]

        # Initialize array in which to store resulting STEM images
        STEM_image = np.zeros((ndet, nthick, nscantot))

        # Also move detectors to pytorch if necessary
        D = ensure_torch_array(detectors, device=device, dtype=dtype)

    # Initialize array in which to store resulting 4D-STEM datacube if required
    return_datacube = False
    if FourD_STEM:

        # Check whether a resampling directive has been given
        if isinstance(FourD_STEM, (list, tuple)):
            # Get output grid and diffraction space size of that grid from tuple
            gridout = FourD_STEM[0]
            sizeout = FourD_STEM[1]

            #
            diff_pat_crop = np.round(
                np.asarray(sizeout) * np.asarray(rsize[:2])
            ).astype(np.int)

            # Define resampling function
            resize = lambda array: fourier_interpolate_2d(
                crop(array, diff_pat_crop), gridout
            )
        else:
            # If no resampling then the output size is just the simulation
            # grid size
            gridout = gridshape
            # Define a do nothing function
            resize = lambda array: array

        # Check whether a datacube is already provided or not
        if datacube is None:
            datacube = np.zeros((nthick, nscantot, *gridout))
            return_datacube = True
        else:
            # If datacube is provided, flatten the scan dimensions of the array
            # the reshape function `should` provide a new view of the object,
            # not a copy of the whole array
            datacube = datacube.reshape((nthick, nscantot, *gridout))

    # This algorithm allows for "batches" of probe to be sent through the
    # multislice algorithm to achieve some speed up at the cost of storing more
    # probes in memory

    if seed is None and batch_size > 1:
        # If no seed passed to random number generator then make one to pass to
        # the multislice algorithm. This ensure that each probe sees the same
        # frozen phonon configuration if we are doing batched multislice
        # calculations
        seed = np.random.randint(0, 2 ** 32 - 1)

    for i in tqdm(range(int(np.ceil(nscantot / batch_size))), disable=not showProgress):

        # Make shifted probes
        scan_index = np.arange(
            i * batch_size, min((i + 1) * batch_size, nscantot), dtype=np.int
        )

        K = scan_index.shape[0]
        # x scan is fast(est changing) scan direction
        xscan = scan_posn[1][scan_index % nscan[1]]
        # y scan is slow(est changing) scan direction
        yscan = scan_posn[0][scan_index // nscan[1]]

        # Shift probes using Fourier shift theorem, prepare shift operators
        # and store them in the array that the probes will eventually inhabit.

        # Array of scan positions must be of size batch_size x 2
        posn = torch.cat(
            [
                torch.from_numpy(x).type(dtype).to(device).view(K, 1)
                for x in [yscan, xscan]
            ],
            dim=1,
        )

        # The shift operator array array will be of size batch_size x Y x X
        probes = fourier_shift_array(gridshape, posn, dtype=dtype, device=device)

        # Apply shift to original probe
        probes = complex_mul(probe_.view(1, *probe_.size()), probes)

        # Thickness series6
        #  - need to take the difference between sequential thickness variations
        for it, t in enumerate(np.diff(nslices, prepend=0)):

            # Evaluate exit surface wave function from input probes
            probes = method(probes, t, *method_args, posn=posn.cpu().numpy(), **method_kwargs)

            # Calculate amplitude of probes
            amp = amplitude(probes)
            if S is None:
                amp /= np.prod(probes.size()[-3:-1])

            # Calculate STEM images
            if conventional_STEM:
                # broadcast detector and probe arrays to
                # ndet x batch_size x Y x X and reduce final two dimensions

                STEM_image[:ndet, it, scan_index] = (
                    torch.sum(
                        D.view(ndet, 1, *gridshape) * amp.view(1, K, *gridshape),
                        (-2, -1),
                    )
                    .cpu()
                    .numpy()
                )
            # Store datacube
            if FourD_STEM:
                datacube[it, scan_index, ...] += resize(
                    np.fft.fftshift(amp.cpu().numpy(), axes=(-1, -2))
                )

    # Unflatten 4D-STEM datacube scan dimensions
    if FourD_STEM:
        datacube = datacube.reshape(nthick, *nscan, *gridout)

    if conventional_STEM and return_datacube:
        return STEM_image.reshape(ndet, nthick, *nscan), datacube
    if return_datacube:
        return datacube
    if conventional_STEM:
        return STEM_image.reshape(ndet, nthick, *nscan)


def unit_cell_shift(array, axis, shift, tiles):
    """For an array consisting of a number of repeat units given by tiles
       shift than array an integer number of unit cells"""

    intshift = array.size(axis) // tiles

    indices = torch.remainder(torch.arange(array.shape[-3 + axis]) - shift)
    if axis == 0:
        return array[indices, :, :]
    if axis == 1:
        return array[:, indices, :]


def max_grid_resolution(gridshape, rsize, bandwidthlimit=2 / 3, eV=None):
    """For a given grid pixel size and real space size return maximum resolution permitted
       by the multislice grid. If the probe accelerating voltage is passed in as eV 
       resolution will be given in units of mrad, otherwise resolution will be given in units
       of inverse Angstrom."""
    max_res = min([gridshape[x] / rsize[x] / 2 * bandwidthlimit for x in range(2)])
    if eV is None:
        return max_res
    from .Probe import wavev

    return max_res / wavev(eV) * 1e3


class scattering_matrix:
    def __init__(
        self,
        rsize,
        propagators,
        transmission_functions,
        nslices,
        eV,
        alpha,
        batch_size=1,
        device=None,
        PRISM_factor=[1, 1],
        tiling=[1, 1],
        device_type=None,
        seed=None,
        showProgress=True,
        bandwidth_limit=2 / 3,
    ):
        """Make a scattering matrix for dynamical scattering calculations using 
        the PRISM algorithm"""

        from .Probe import wavev, plane_wave_illumination

        # Get size of grid
        gridshape = np.shape(propagators)[1:3]

        # Datatype (precision) is inferred from transmission functions
        dtype = transmission_functions.dtype

        # Device (CPU or GPU) is also inferred from transmission functions
        if device is None:
            device = transmission_functions.device

        # Get alpha in units of inverse Angstrom
        self.alpha_ = wavev(eV) * alpha * 1e-3

        # Make a list of beams in the scattering matrix
        q = q_space_array(gridshape, rsize)
        self.beams = np.argwhere(
            np.logical_and(
                np.logical_and(
                    np.less_equal(q[0] ** 2 + q[1] ** 2, self.alpha_),
                    (
                        np.mod(
                            np.fft.fftfreq(gridshape[0], d=1 / gridshape[0]).astype(
                                np.int
                            ),
                            PRISM_factor[0],
                        )
                        == 0
                    )[:, np.newaxis],
                ),
                (
                    np.mod(
                        np.fft.fftfreq(gridshape[1], d=1 / gridshape[1]).astype(np.int),
                        PRISM_factor[1],
                    )
                    == 0
                )[np.newaxis, :],
            )
        )
        nbeams = self.beams.shape[0]

        # We will only store output of the scattering matrix up to the band
        # width limit of the calculation, since this is a circular band-width
        # limit on a square grid we have to get somewhat fancy and store a mapping
        # of the pixels within the bandwidth limit to a one-dimensional vector
        self.bw_mapping = np.argwhere(
            (np.fft.fftfreq(gridshape[0]) ** 2)[:, np.newaxis]
            + (np.fft.fftfreq(gridshape[1]) ** 2)[np.newaxis, :]
            < (bandwidth_limit / 2) ** 2
        )
        self.nbout = self.bw_mapping.shape[0]

        # Initialize scattering matrix
        self.S = torch.zeros(nbeams, self.nbout, 2, dtype=dtype, device=device)

        if seed is None:
            # If no seed passed to random number generator then make one to pass to
            # the multislice algorithm. This ensure that each column in the scattering
            # matrix sees the same frozen phonon configuration
            seed = np.random.randint(0, 2 ** 32 - 1)

        # Initialize probe wave function array
        psi = torch.zeros(batch_size, *gridshape, 2, dtype=dtype, device=device)

        for i in tqdm(
            range(int(np.ceil(nbeams / batch_size))), disable=not showProgress
        ):
            #
            for j in range(batch_size):
                jj = j + batch_size * i
                psi[j, ...] = cx_from_numpy(
                    plane_wave_illumination(
                        gridshape,
                        rsize[:2],
                        tilt=self.beams[jj, :],
                        tilt_units="pixels",
                    )
                )

            psi = multislice(
                psi,
                nslices,
                propagators,
                transmission_functions,
                tiling,
                device,
                seed,
                return_numpy=False,
                qspace_out=True,
            )

            self.S[i * batch_size : (i + 1) * batch_size, ...] = psi[
                :, self.bw_mapping[:, 0], self.bw_mapping[:, 1], :
            ]

        self.gridshape = gridshape
        self.PRISM_factor = PRISM_factor
        self.rsize = rsize
        self.tiling = tiling
        self.eV = eV

    def __call__(self, probes, nslices, posn=None):
        """Evaluate the the Smatrix probe matrix multiplication for a number
        of probes in windows centred about window_posn nslices does nothing
        and is only included to match the function call signature for the
        STEM routine"""

        # Distance function to calculate windows about scan positions
        dist = lambda x, X: np.abs((np.arange(X) - x + X // 2) % X - X // 2)

        # TODO Implement PRISM thickness series
        # Perform matrix multiplication of scattering matrix with
        # probe vector

        probe_vec = complex_matmul(
            probes[:, self.beams[:, 0], self.beams[:, 1]], self.S
        )

        # Now reshape output from vectors to square arrays
        probes[...] = 0
        probes[:, self.bw_mapping[:, 0], self.bw_mapping[:, 1], :] = probe_vec

        # Apply PRISM cropping in real space if appropriate
        if np.any([self.PRISM_factor[i] > 1 for i in [0, 1]]):
            # Transform probe to real space
            probes = torch.ifft(probes, signal_ndim=2)

            for k in range(probes.size(0)):

                # Calculate windows in vertical and horizontal directions
                window = [
                    (
                        dist(posn[k][i], self.gridshape[i])
                        > self.gridshape[i] / self.PRISM_factor[i] / 2
                    )
                    for i in range(2)
                ]

                # Set array values outside windows to zero
                probes[k, window[0]] = 0
                probes[k, :, window[1]] = 0

            # Transform probe back to Fourier space
            probes = torch.fft(probes, signal_ndim=2)
        return probes

    def Smatrix_STEM(
        self,
        probe,
        alpha,
        batch_size=1,
        detectors=None,
        datacube=None,
        showProgress=True,
        FourD_STEM=False,
        scan_posn=None,
    ):
        """Using the scattering matrix simulate a STEM experiment"""
        dtype = self.S.dtype
        device = self.S.device

        method = self

        return STEM(
            self.rsize,
            probe,
            method,
            [1],
            self.eV,
            alpha,
            batch_size=batch_size,
            detectors=detectors,
            FourD_STEM=FourD_STEM,
            datacube=datacube,
            scan_posn=scan_posn,
            device=device,
            tiling=self.tiling,
            showProgress=showProgress,
        )

    def plot(self, show=True):
        """Make a montage plot of the scattering matrix"""
        from .utils import colorize

        nopiy, nopix = self.gridshape

        yind = (
            (self.beams[:, 0] + np.amax(self.beams[self.beams[:, 0] < nopiy // 2, 0]))
            % nopiy
            // self.PRISM_factor[0]
        )
        xind = (
            (self.beams[:, 1] + np.amax(self.beams[self.beams[:, 1] < nopix // 2, 1]))
            % nopix
            // self.PRISM_factor[1]
        )

        nrows = max(yind) + 1
        ncols = max(xind) + 1

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

        for i in range(nrows * ncols):
            ax[i % nrows, i // nrows].set_axis_off()

        for ibeam in range(self.beams.shape[0]):
            S = torch.zeros((nopiy, nopix, 2), dtype=self.S.dtype, device=self.S.device)
            S[self.bw_mapping[:, 0], self.bw_mapping[:, 1], :] = self.S[ibeam, ...]

            S = torch.ifft(S, signal_ndim=2).cpu()
            ax[yind[ibeam], xind[ibeam]].imshow(colorize(cx_to_numpy(S)))

        if show:
            plt.show(block=True)
        return fig
