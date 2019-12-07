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
    ensure_torch_array,
    crop_to_bandwidth_limit_torch,
    size_of_bandwidth_limited_array,
    fourier_interpolate_2d_torch,
)


def make_propagators(pixelsize, gridsize, eV, subslices=[1.0], tilt=[0, 0]):
    """Make the Fresnel freespace propagators for a multislice simulation.

    Arguments:
    pixelsize -- Pixel dimensions of the 2D grid
    gridsize  -- Size of the grid in real space (first two dimensions) and
                 thickness of the object in multislice (third dimension)
    eV        -- Probe energy in electron volts
    subslices -- A one dimensional array-like object containing the depths
                 (in fractional coordinates) at which the object will be
                 subsliced. The last entry should always be 1.0. For example,
                 to slice the object into four equal sized slices pass
                 [0.25,0.5,0.75,1.0]
    tilt      -- Allows the user to simulate a (small < 50 mrad) tilt of the
                 specimen, units in mrad.
    """
    from .Probe import make_contrast_transfer_function, wavev

    # We will use the make_contrast_transfer_function function to generate the propagator, the
    # aperture of this propagator will supply the bandwidth limit of our simulation
    # it must be 2/3rds of our pixel gridsize
    app = np.amax(np.asarray(pixelsize) / np.asarray(gridsize[:2]) / 2)

    # Shift of optic axis to take
    optic_axis = np.asarray(tilt) * 1e-3 * wavev(eV)

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
                pixelsize,
                gridsize[:2],
                eV,
                app,
                df=deltaz,
                app_units="invA",
                optic_axis=optic_axis,
            )
        )

    return prop


def generate_slice_indices(nslices, nsubslices, subslicing=False):
    """Generate the slice indices for the multislice routine"""
    from collections import Sequence

    if isinstance(nslices, Sequence) or isinstance(nslices, np.ndarray):
        # If a list is passed, continue as before
        return nslices
    else:
        # If an integer is passed generate a list of slices to iterate through
        niterations = nslices if subslicing else nslices * nsubslices
        return np.arange(niterations)


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
    output_to_bandwidth_limit=True,
    reverse=False,
    transpose=False,
):
    """For a given probe or set of probes, propagators, and transmission 
        functions perform the multislice algorithm for nslices iterations."""

    # If a single integer is passed to the routine then Seed random number generator,
    # , if None then np.random.RandomState will use the system clock as a seed
    seed_provided = not (seed is None)
    if not seed_provided:
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

    slices = generate_slice_indices(nslices, nsubslices, subslicing)

    for i, islice in enumerate(slices):

        # If an array-like object is passed then this will be used to uniquely
        # and reproducibly seed each iteration of the multislice algorithm
        if seed_provided:
            r = np.random.RandomState(seed[islice])

        subslice = islice % nsubslices

        # Pick random phase grating
        it = r.randint(0, nT)

        # To save memory in the case of equally sliced sample, there is the option
        # of only using one propagator, this statement catches this case.
        if P.dim()<3:
            P_ = P
        else:
            P_ = P[subslice]

        # If the transmission function is from a tiled unit cell then
        # there is the option of randomly shifting it around to
        # generate "more" transmission functions
        if tiling is None or (tiling[0] == 1 & tiling[1] == 1):
            T_ = T[it, subslice, ...]
        elif nopiy % tiling[0] == 0 and nopix % tiling[1] == 0:
            # Shift an integer number of pixels in y
            T_ = roll_n(
                T[it, subslice, ...], 0, r.randint(0, tiling[0]) * (nopiy // tiling[0])
            )

            # Shift an integer number of pixels in x
            T_ = roll_n(T_, 1, r.randint(1, tiling[1]) * (nopix // tiling[1]))
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

        # Perform multislice iteration
        if transpose or reverse:
            # Reverse multislice complex conjugates the transmission and propagation
            # Both reverse and transpose multislice reverse the order of the transmission
            # and conjugation operations
            psi = complex_mul(
                torch.ifft(
                    complex_mul(torch.fft(psi, signal_ndim=2), P_, reverse),
                    signal_ndim=2,
                ),
                T_,
                reverse,
            )
        else:
            psi = complex_mul(torch.fft(complex_mul(psi, T_), signal_ndim=2), P_)

        if i == len(slices) - 1 and output_to_bandwidth_limit:
            if transpose or reverse:
                psi = torch.fft(psi, signal_ndim=2)
            # The probe can be cropped to the bandwidth limit, this removes
            # superfluous array entries in reciprocal space that are zero
            # Since the next inverse FFT will apply a factor equal to the
            # square root number of pixels we have to adjust the values
            # of the array to compensate
            psi /= torch.sqrt(
                torch.prod(torch.tensor(psi.size()[-3:-1], dtype=T.dtype))
            )
            psi = crop_to_bandwidth_limit_torch(psi)
            psi *= torch.sqrt(
                torch.prod(torch.tensor(psi.size()[-3:-1], dtype=T.dtype))
            )
            if (transpose or reverse) and not qspace_out:
                psi = torch.ifft(psi, signal_ndim=2)

        # This logic statement allows for the probe to be returned
        # in reciprocal space
        # TODO the not (transpose or reverse) is a quick fix needs to be better
        # thought out long term for predictable behaviour
        if not np.all([qspace_out, i == len(slices) - 1]) and (
            not (transpose or reverse)
        ):
            psi = torch.ifft(psi, signal_ndim=2)

    if (transpose or reverse) and qspace_out:
        psi = torch.fft(psi, signal_ndim=2)

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


def generate_STEM_raster(gridshape, rsize, eV, alpha, tiling=[1, 1],ROI = [0.0,0.0,1.0,1.0]):
    """For a grid of pixel size given by gridshape and real space size rsize
    return probe positions for nyquist sampled STEM raster
    
    Arguments:
    gridshape -- Pixel dimensions of the 2D grid
    rsize     -- Size of the grid in real space in units of Angstroms
    eV        -- Probe energy in electron volts
    alpha     -- Probe forming aperture semi-angle in mrad

    Keyword arguements
    tiling    -- Tiling of grid for a repeat unit cell, STEM raster will only
                 scan a single unit cell
    ROI       -- Fraction of the unit cell to be scanned this array should
                 contain [y0,x0,y1,x1] where [x0,y0] and [x1,y1] are the bottom
                 left and top right coordinates of the region of interest (ROI)
                 expressed as a fraction of the total grid (or unit cell)
                 
    """

    # Calculate number of scan positions in STEM scan
    from .Probe import nyquist_sampling

    # Field of view in Angstrom
    FOV = np.asarray([rsize[0]*(ROI[2]-ROI[0]),rsize[1]*(ROI[3]-ROI[1])])

    # Number of scan coordinates in each dimension
    nscan = nyquist_sampling(FOV / np.asarray(tiling), eV=eV, alpha=alpha)

    # Generate Y and X scan coordinates
    return [
        np.arange(ROI[0+i]*gridshape[i]/tiling[i], ROI[2+i]*gridshape[i] / tiling[i], step=np.diff(ROI[i::2])[0]*gridshape[i] / nscan[i] / tiling[i])
        / gridshape[i]
        for i in range(2)
    ]


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

    Arguments:
    rsize       -- The real space size of the grid in Angstroms
    probe       -- The probe that will be rastered over the object
    method      -- A function that takes a probe and propagates it to the exit
                   surface of the specimen
    nslices     -- The number of slices to perform multislice over
    eV          -- Accelerating voltage of the probe, needed to work out probe
                   sampling requirements
    alpha       -- The convergence angle of the probe in mrad, needed to work 
                   out probe sampling requirements
    
    Keyword arguments
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
    from .utils.torch_utils import detect

    # Get number of thicknesses in the series
    nthick = len(nslices)

    # Get shape of grid
    gridshape = probe.shape[-2:]

    # Generate scan positions in units of fractions of the grid if not supplied
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
            gridout = size_of_bandwidth_limited_array(gridshape)
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
        probes = fourier_shift_array(
            gridshape, posn, dtype=dtype, device=device, units="fractional"
        )

        # Apply shift to original probe
        probes = complex_mul(probe_.view(1, *probe_.size()), probes)

        # Thickness series6
        #  - need to take the difference between sequential thickness variations
        for it, t in enumerate(np.diff(nslices, prepend=0)):

            # Evaluate exit surface wave function from input probes
            probes = method(probes, t, *method_args, posn=posn, **method_kwargs)

            # Calculate amplitude of probes
            amp = amplitude(probes)

            # Calculate STEM images
            if conventional_STEM:
                # broadcast detector and probe arrays to
                # ndet x batch_size x Y x X and reduce final two dimensions

                STEM_image[:ndet, it, scan_index] = detect(D, amp).cpu().numpy()

            # Store datacube
            if FourD_STEM:
                datacube[it, scan_index] += resize(
                    np.fft.fftshift(amp.cpu().numpy(), axes=(-1, -2))
                )

    # Unflatten 4D-STEM datacube scan dimensions
    if FourD_STEM:
        datacube = datacube.reshape(nthick, *nscan, *gridout)

    if conventional_STEM and return_datacube:
        return np.squeeze(STEM_image.reshape(ndet, nthick, *nscan)), np.squeeze(datacube)
    if return_datacube:
        return np.squeeze(datacube)
    if conventional_STEM:
        return np.squeeze(STEM_image.reshape(ndet, nthick, *nscan))


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
        nslice,
        eV,
        alpha,
        GPU_streaming=False,
        batch_size=1,
        device=None,
        PRISM_factor=[1, 1],
        tiling=[1, 1],
        device_type=None,
        seed=None,
        showProgress=True,
        bandwidth_limit=2 / 3,
        Fourier_space_output=False,
        subslicing=False,
        transposed=False,
    ):
        """Make a scattering matrix for dynamical scattering calculations using 
        the PRISM algorithm"""

        from .Probe import wavev

        # Get size of grid
        gridshape = np.shape(propagators)[1:3]

        # Datatype (precision) is inferred from transmission functions
        self.dtype = transmission_functions.dtype

        # Device (CPU or GPU) is also inferred from transmission functions
        self.device = device
        if self.device is None:
            self.device = transmission_functions.device

        # Get alpha in units of inverse Angstrom
        self.alpha_ = wavev(eV) * alpha * 1e-3

        # Make a list of beams in the scattering matrix
        q = q_space_array(gridshape, rsize)
        self.PRISM_factor = PRISM_factor
        self.beams = np.argwhere(
            np.logical_and(
                np.logical_and(
                    np.less_equal(q[0] ** 2 + q[1] ** 2, self.alpha_),
                    (
                        np.mod(
                            np.fft.fftfreq(gridshape[0], d=1 / gridshape[0]).astype(
                                np.int
                            ),
                            self.PRISM_factor[0],
                        )
                        == 0
                    )[:, np.newaxis],
                ),
                (
                    np.mod(
                        np.fft.fftfreq(gridshape[1], d=1 / gridshape[1]).astype(np.int),
                        self.PRISM_factor[1],
                    )
                    == 0
                )[np.newaxis, :],
            )
        )

        self.nbeams = self.beams.shape[0]

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

        self.gridshape = gridshape
        self.PRISM_factor = PRISM_factor
        self.doPRISM = np.any([self.PRISM_factor[i] > 1 for i in [0, 1]])
        self.rsize = rsize
        self.eV = eV
        self.Fourier_space_output = Fourier_space_output
        self.tiling = tiling
        self.nsubslices = propagators.shape[0]
        slices = generate_slice_indices(
            nslice, propagators.shape[0], subslicing=subslicing
        )
        self.GPU_streaming = GPU_streaming
        self.transposed = transposed

        self.seed = seed
        if self.seed is None:
            # If no seed passed to random number generator then make one to pass to
            # the multislice algorithm. This ensure that each column in the scattering
            # matrix sees the same frozen phonon configuration
            self.seed = np.random.randint(0, 2 ** 32 - 1, size=len(slices),dtype=np.uint32)

        # This switch tells the propagate function to initialize the Smatrix
        # to plane waves
        self.initialized = False
        # Propagate wave functions of scattering matrix
        self.current_slice = 0
        self.Propagate(
            nslice, propagators, transmission_functions, subslicing=subslicing
        )

    def Propagate(
        self,
        nslice,
        propagators,
        transmission_functions,
        subslicing=False,
        batch_size=3,
        showProgress=True,
        transpose=False,
    ):
        """Propagate a scattering matrix to slice nslice of the specimen
        Arguments:
        nslice      -- The slice in the specimen to propagate the scattering
                       matrix to
        propagators -- Fresnel free space operators required for the multislice
                       algorithm used to propagate the scattering matrix
        transmission_functions -- The transmission functions describing the
                        electron's interaction with the specimen for the 
                        multislice algorithm
        
        Keyword arguments
        batch_size  -- The multislice algorithm can be performed on multiple
                       scattering matrix columns at once to parrallelize 
                       computation
        subslicing  -- Pass subslicing=True to access propagation to sub-slices
                       of the unit cell, in this case nslices is taken to be
                       in units of subslices to propagate rather than unit-cells
                       (i.e. nslices = 3 will propagate 1.5 unit cells for 
                       a subslicing of 2 subslices per unit cell)
        showProgress -- Pass showProgress = False to disable live progress 
                        readout
        transpose   -- Make a "transposed" scattering matrix - see Brown et al.
                       (2019) Physical Review Research paper for a discussion
                       of this 
        """
        from .utils.torch_utils import (
            cx_from_numpy,
            crop_to_bandwidth_limit_torch,
            size_of_bandwidth_limited_array,
        )
        from .Probe import plane_wave_illumination

        # Initialize scattering matrix if necessary
        if not self.initialized:
            if self.Fourier_space_output:
                self.S = torch.zeros(
                    self.nbeams, self.nbout, 2, dtype=self.dtype, device=self.device
                )
            else:

                self.stored_gridshape = size_of_bandwidth_limited_array(
                    transmission_functions.size()[-3:-1]
                )

                self.S = torch.zeros(
                    self.nbeams,
                    *self.stored_gridshape,
                    2,
                    dtype=self.dtype,
                    device=self.device
                )
            for ibeam in range(self.nbeams):
                # Initialize S-matrix to plane-waves
                psi = cx_from_numpy(
                    plane_wave_illumination(
                        self.gridshape,
                        self.rsize[:2],
                        tilt=self.beams[ibeam, :],
                        tilt_units="pixels",
                        qspace=True,
                    )
                )

                # Adjust intensity for correct normalization of S matrix rows
                # taking into account the PRISM factor that needs to be applied
                # when the Smatrix is evaluated (only 1/product(PRISM_factor)
                # beams are taken and only 1/product(PRISM_factor) intensity
                # is cropped out in real space)
                psi *= torch.prod(torch.tensor(self.PRISM_factor, dtype=self.dtype))

                if self.Fourier_space_output:
                    self.S[ibeam, ...] = psi[
                        self.bw_mapping[:, 0], self.bw_mapping[:, 1], :
                    ]
                else:
                    self.S[ibeam, ...] = crop_to_bandwidth_limit_torch(psi)
            if not self.Fourier_space_output:
                self.S = torch.ifft(self.S, signal_ndim=2, normalized=True)
            self.initialized = True

        # Make nslice_ which always accounts of subslices of the cyrstal structure
        if subslicing:
            nslice_ = nslice
        else:
            nslice_ = nslice * self.nsubslices

        # Work out direction of propagation through specimen
        direction = np.sign(nslice_ - self.current_slice)
        if direction == 0:
            direction = 1

        if nslice_ > len(self.seed):
            # Add new seeds to determine random translations for frozen-phonon
            # multislice (required for reversability of multislice) if required
            self.seed = np.concatenate(
                [
                    self.seed,
                    np.random.randint(0, 2 ** 32 - 1, size=nslice_ - len(self.seed)),
                ]
            )

        # Now generate list of slices that the multislice algorithm will run through
        slices = np.arange(self.current_slice, nslice_, direction)
        if direction < 0:
            slices += direction

        # For a transposed scattering matrix the order of the slices
        # in multislice should be reversed
        if self.transposed:
            slices = slices[::-1]

        # Initialize array that will be used as input to the multislice routine
        psi = torch.zeros(
            batch_size, *self.gridshape, 2, dtype=self.dtype, device=self.device
        )

        # If streaming of Smatrix columns to the GPU is being used, ensure
        # that propagators and transmission functions for the multislice are
        # already on the GPU
        if self.GPU_streaming:
            propagators = ensure_torch_array(propagators).cuda()
            transmission_functions = ensure_torch_array(transmission_functions).cuda()

        self.current_slice = nslice_
        if len(slices) < 1:
            return

        # Loop over the different plane wave components (or columns) of the
        # scattering matrix
        for i in tqdm(
            range(int(np.ceil(self.nbeams / batch_size))), disable=not showProgress
        ):
            psi[...] = 0.0
            beams = np.arange(
                i * batch_size, min((i + 1) * batch_size, self.nbeams), dtype=np.int
            )

            if self.Fourier_space_output:
                # Expand S-matrix input to full grid for multislice propagation
                psi[:, self.bw_mapping[:, 0], self.bw_mapping[:, 1], :] = self.S[beams]
            else:
                # Fourier interpolate stored real space S-matrix column onto
                # multislice grid
                psi = fourier_interpolate_2d_torch(self.S[beams], self.gridshape, False)

            if self.GPU_streaming:
                psi = ensure_torch_array(psi, dtype=self.dtype).to("cuda")

            output = multislice(
                psi[: beams.shape[0]],
                slices,
                propagators,
                transmission_functions,
                self.tiling,
                self.device,
                self.seed,
                return_numpy=False,
                qspace_out=self.Fourier_space_output,
                transpose=self.transposed,
                reverse=direction < 0,
            )

            if self.GPU_streaming:
                output = output.to(self.device)

            if self.Fourier_space_output:
                # Take into account the sqrt(# of pixels) that would otherwise have been
                # applied by the inverse Fourier transform in the multislice routine
                output /= torch.sqrt(
                    torch.prod(torch.tensor(self.gridshape, dtype=self.dtype))
                )

                self.S[beams, ...] = output[
                    :, self.bw_mapping[:, 0], self.bw_mapping[:, 1], :
                ]
            else:
                self.S[beams, ...] = output

    def __call__(self, probes, nslices, posn=None):
        """Evaluate the the Smatrix probe matrix multiplication for a number
        of probes in windows centred about window_posn. The variable nslices 
        does nothing and is only included to match the function call signature 
        for the STEM routine"""
        from .utils.torch_utils import crop_window_to_flattened_indices_torch
        from copy import deepcopy

        # Distance function to calculate windows about scan positions
        dist = lambda x, X: np.abs((np.arange(X) - x + X // 2) % X - X // 2)

        if self.doPRISM:
            crop_ = [
                torch.arange(
                    -self.gridshape[i] // (2 * self.PRISM_factor[i]),
                    self.gridshape[i] // (2 * self.PRISM_factor[i]),
                )
                for i in range(2)
            ]

        if self.Fourier_space_output:
            # A note on normalization: an individual probe enters the STEM routine
            # with sum_squared intensity of 1, but the STEM routine applies an
            # FFT so the sum_squared intensity is now equal to # pixels
            # For a correct matrix multiplication we must now divide by sqrt(# pixels)
            probe_vec = complex_matmul(
                probes[:, self.beams[:, 0], self.beams[:, 1]], self.S
            ) / np.sqrt(np.prod(self.gridshape))

            # Now reshape output from vectors to square arrays
            probes[...] = 0
            probes[:, self.bw_mapping[:, 0], self.bw_mapping[:, 1], :] = probe_vec

            # Apply PRISM cropping in real space if appropriate
            if self.doPRISM:
                shape = probes.size()

                probes = torch.ifft(probes, signal_ndim=2).flatten(-3, -2)
                for k in range(probes.size(0)):

                    # Calculate windows in vertical and horizontal directions
                    window = crop_window_to_flattened_indices_torch(
                        [
                            (crop_[i] + posn[k, i] * self.gridshape[i])
                            % self.gridshape[i]
                            for i in range(2)
                        ],
                        self.gridshape,
                    )
                    probe = deepcopy(probes[k])
                    probes[k] = 0
                    probes[k, window, :] = probe[window, :]

                probes = probes.reshape(shape)

                # Transform probe back to Fourier space
                return torch.fft(probes, signal_ndim=2)
        else:
            # Flatten the array dimensions
            flattened_shape = [self.S.size(-3) * self.S.size(-2), 2]
            output = torch.zeros(
                probes.size(0), *flattened_shape, dtype=self.dtype, device=self.device
            )

            # For evaluating the probes in real space we only want to perform the matrix
            # multiplication and summation within the real space PRISM cropping region
            for k in range(probes.size(0)):
                window = crop_window_to_flattened_indices_torch(
                    [
                        (crop_[i] + posn[k, i] * self.stored_gridshape[i])
                        % self.stored_gridshape[i]
                        for i in range(2)
                    ],
                    self.stored_gridshape,
                )

                output[k, window] = torch.sum(
                    complex_mul(
                        probes[k, self.beams[:, 0], self.beams[:, 1]].view(
                            self.nbeams, 1, 2
                        ),
                        self.S.view(self.nbeams, *flattened_shape)[:, window],
                    ),
                    dim=0,
                ) / np.sqrt(np.prod(self.gridshape))
            output = output.reshape(probes.size(0), *self.S.size()[-3:])

            return torch.fft(output, signal_ndim=2, normalized=True)

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
        # The method used to propagate an electron wave to the exit surface
        # is the __call__ method for the scattering matrix object
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
            device=self.device,
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
