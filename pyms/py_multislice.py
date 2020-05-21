"""Module containing functions for core multislice and PRISM algorithms."""
import numpy as np
import torch
from tqdm import tqdm
from .Probe import wavev, focused_probe
from .utils.numpy_utils import (
    bandwidth_limit_array,
    q_space_array,
    fourier_interpolate_2d,
    crop,
)
from .utils.torch_utils import (
    roll_n,
    cx_from_numpy,
    cx_to_numpy,
    complex_matmul,
    complex_mul,
    fourier_shift_array,
    amplitude,
    get_device,
    ensure_torch_array,
    crop_to_bandwidth_limit_torch,
    size_of_bandwidth_limited_array,
    fourier_interpolate_2d_torch,
    crop_window_to_flattened_indices_torch,
    crop_torch,
)


def make_propagators(
    gridshape,
    gridsize,
    eV,
    subslices=[1.0],
    tilt=[0, 0],
    tilt_units="mrad",
    bandwidth_limit=2 / 3,
):
    """
    Make the Fresnel freespace propagators for a multislice simulation.

    Parameters
    ----------
    gridshape : (2,) array_like
        Pixel dimensions of the 2D grid
    gridsize : (3,) array_like
        Size of the grid in real space (first two dimensions) and thickness of
        the object (third dimension)
    eV : float
        Probe energy in electron volts

    Keyword arguments
    -----------------
    subslices : array_like, optional
        A one dimensional array-like object containing the depths (in fractional
        coordinates) at which the object will be subsliced. The last entry
        should always be 1.0. For example, to slice the object into four equal
        sized slices pass [0.25,0.5,0.75,1.0]
    tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) tilt of the specimen,
        by shearing the propagator. Units given by input variable tilt_units.
    tilt_units : string, optional
        Units of specimen tilt, can be 'mrad','pixels' or 'invA'
    """
    from .Probe import make_contrast_transfer_function

    # We will use the make_contrast_transfer_function function to generate
    # the propagator, the aperture of this propagator will go out to the maximum
    # possible and the function bandwidth_limit_array will provide the band
    # width_limiting
    gridmax = np.asarray(gridshape) / np.asarray(gridsize[:2]) / 2
    app = np.hypot(*gridmax)

    # Intitialize array
    prop = np.zeros((len(subslices), *gridshape), dtype=np.complex)
    for islice, slice in enumerate(subslices):
        if islice == 0:
            deltaz = slice * gridsize[2]
        else:
            deltaz = (slice - subslices[islice - 1]) * gridsize[2]

        # Calculate propagator
        prop[islice, :, :] = bandwidth_limit_array(
            make_contrast_transfer_function(
                gridshape,
                gridsize[:2],
                eV,
                app,
                df=deltaz,
                app_units="invA",
                optic_axis=tilt,
                tilt_units=tilt_units,
            ),
            limit=bandwidth_limit,
        )

    return prop


def generate_slice_indices(nslices, nsubslices, subslicing=False):
    """Generate the slice indices for the multislice routine."""
    from collections.abc import Sequence

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
    """
    Multislice algorithm for scattering of an electron probe.

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

    Keyword arguments
    -----------------
    tiling : (2,) array_like
        Tiling of a repeat unit cell on simulation grid.
    device_type : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    seed : int, optional
        Seed for the random number generator for frozen phonon configurations
    return_numpy : bool, optional
        Calculations are performed on pytorch tensors for speed, however numpy
        arrays are more convenient for processing. This input allows the
        user to control how the output is returned
    qspace_in : bool, optional
        Should be set to True if the input wavefunction is in momentum (q) space
        and False otherwise
    qspace_out : bool, optional
        Should be set to True if the output wavefunction is desired in momentum
        (q) space and False otherwise
    posn :
        Does nothing, included to match calling signature for STEM function
    subslicing : bool, optional
        Pass subslicing=True to access propagation to sub-slices of the
        unit cell, in this case nslices is taken to be in units of subslices
        to propagate rather than unit cells (i.e. nslices = 3 will propagate
        1.5 unit cells for a subslicing of 2 subslices per unit cell)
    output_to_bandwidth_limit : bool, optional
        Bandwidth-limiting of the arrays is used in multislice to stop
        wrap-around error in reciprocal space, therefore the output of the
        multislice algorithm will be zero beyond some point in reciprocal space
        if this is set to True then these array entries will be cropped out.
        This does have the effect of the output of the function being on a
        different sized grid to the input.
    reverse : bool, optional
        Run inverse multislice (for back propagation of a wavefunction)
    transpose : bool, optional
        Reverse the order of the multislice operations, ie. apply propagator
        first and then transmission function
    """
    # If a single integer is passed to the routine then Seed random number generator,
    # , if None then np.random.RandomState will use the system clock as a seed
    seed_provided = not (seed is None)
    if not seed_provided:
        r = np.random.RandomState()
        np.random.seed(seed)

    # Initialize device cuda if available, CPU if no cuda is available
    device = get_device(device_type)

    # Since pytorch doesn't have a complex data type we need to add an extra
    # dimension of size 2 to each tensor that will store real and imaginary
    # components.
    T = ensure_torch_array(transmission_functions, device=device)
    P = ensure_torch_array(propagators, dtype=T.dtype, device=device)
    psi = ensure_torch_array(probes, dtype=T.dtype, device=device)

    nT, nsubslices, nopiy, nopix = T.size()[:4]

    # Probe needs to start multislice algorithm in real space
    if qspace_in:
        psi = torch.ifft(psi, signal_ndim=2)

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
        if P.dim() < 4:
            P_ = P
        else:
            P_ = P[subslice]

        # If the transmission function is from a tiled unit cell then
        # there is the option of randomly shifting it around to
        # generate "more" transmission functions
        if tiling is None or (tiling[0] == 1 & tiling[1] == 1):
            T_ = T[it, subslice]
        elif nopiy % tiling[0] == 0 and nopix % tiling[1] == 0:
            # Shift an integer number of pixels in y
            T_ = roll_n(
                T[it, subslice], 0, r.randint(0, tiling[0]) * (nopiy // tiling[0])
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
                complex_mul(FFT_shift_array, torch.fft(T[it, subslice], signal_ndim=2)),
                signal_ndim=2,
            )

        # Perform multislice iteration
        if transpose or reverse:
            # Reverse multislice complex conjugates the transmission and
            # propagation. Both reverse and transpose multislice reverse
            # the order of the transmission and conjugation operations
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


def STEM_phase_contrast_transfer_function(probe, detector):
    """
    Calculate the STEM phase contrast transfer function.

    For a thin and weakly scattering sample convolution with the STEM contrast
    transfer function gives a good approximate for STEM image contrast.

    Parameters
    ----------
    probe : complex, (Y,X) array_like
        The STEM probe in reciprocal space
    detector : real, (Y,X) array_like
        The STEM detector

    Returns
    -------
    PCTF : complex (Y,X) np.ndarray
        The phase contrast transfer function
    """
    from .utils import convolve

    norm = np.sum(np.square(np.abs(probe)))
    # Use two ffts to perform reflection k -> -k
    PCTF = (
        convolve(
            probe,
            np.fft.fft2(
                np.fft.fft2(np.conj(probe) * detector, norm="ortho"), norm="ortho"
            ),
        )
        / norm
    )
    PCTF -= np.conj(np.fft.fft2(np.fft.fft2(PCTF, norm="ortho"), norm="ortho"))

    return -2 * np.imag(PCTF)


# TODO make detectors binary for use with numpy and pytorch sum routines
def make_detector(gridshape, rsize, eV, betamax, betamin=0, units="mrad"):
    """
    Make a STEM detector with acceptance angle between betamin and betamax.

    Parameters
    ----------
    gridshape : (2,) array_like
        Pixel dimensions of the 2D grid
    rsize :  (2,) array_like
        Size of the grid in real space in units of Angstroms
    eV : float
        Probe energy in electron volts
    betamax : float
        Detector outer acceptance semi-angle

    Keyword arguments
    -----------------
    betamin : float, optional
        Detector inner acceptance semi-angle
    units : float, optional
        Units of betamin and betamax, mrad or invA are both acceptable
    """
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


def nyquist_sampling(rsize=None, resolution_limit=None, eV=None, alpha=None):
    """
    Calculate nyquist sampling (typically for minimum sampling of a STEM probe).

    If array size in units of length is passed then return how many probe
    positions are required otherwise just return the sampling. Alternatively
    pass probe accelerating voltage (eV) in electron-volts and probe forming
    aperture (alpha) in mrad and the resolution limit in inverse length will be
    calculated for you.
    """
    if eV is None and alpha is None:
        step_size = 1 / (4 * resolution_limit)
    elif resolution_limit is None:
        step_size = 1 / (4 * wavev(eV) * alpha * 1e-3)
    else:
        return None

    if rsize is None:
        return step_size
    else:
        return np.ceil(rsize / step_size).astype(np.int)


def generate_STEM_raster(
    rsize,
    eV,
    alpha,
    tiling=[1, 1],
    ROI=[0.0, 0.0, 1.0, 1.0],
    gridshape=[1, 1],
    invA=False,
):
    """
    Return the probe positions for a nyquist-sampled STEM raster.

    For a real space size rsize return probe positions in units of fraction of
    the array for nyquist sampled STEM raster

    Parameters
    ----------
    rsize :  (2,) array_like
        Size of the grid in real space in units of Angstroms
    eV : float
        Probe energy in electron volts
    alpha : float
        Probe forming aperture semi-angle in mrad or inverse Angstorm
        (if invA == True)

    Keyword arguments
    -----------------
    gridshape : (2,) array_like, optional
        Pixel dimensions of the 2D grid, by default [1,1] so probe positions will
        be returned as a fraction of the array size
    tiling : (2,) array_like
        Tiling of a repeat unit cell on simulation grid, if provided STEM raster
        will only scan a single unit cell.
    ROI : (4,) array_like
        Fraction of the unit cell to be scanned. Should contain [y0,x0,y1,x1]
        where [x0,y0] and [x1,y1] are the bottom left and top right coordinates
        of the region of interest (ROI) expressed as a fraction of the total
        grid (or unit cell).
    invA : bool
        If True, alpha is taken to be in units of inverse Angstrom, not mrad.
        This also means that the value of eV no longer matters
    """
    # Field of view in Angstrom
    FOV = np.asarray([rsize[0] * (ROI[2] - ROI[0]), rsize[1] * (ROI[3] - ROI[1])])

    if invA:
        # Number of scan coordinates in each dimension
        nscan = nyquist_sampling(FOV / np.asarray(tiling), resolution_limit=alpha)
    else:
        # Number of scan coordinates in each dimension
        nscan = nyquist_sampling(FOV / np.asarray(tiling), eV=eV, alpha=alpha)

    # Generate Y and X scan coordinates
    yy, xx = [
        np.arange(
            ROI[0 + i] * gridshape[i] / tiling[i],
            ROI[2 + i] * gridshape[i] / tiling[i],
            step=np.diff(ROI[i::2])[0] * gridshape[i] / nscan[i] / tiling[i],
        )
        / gridshape[i]
        for i in range(2)
    ]

    return np.stack(np.broadcast_arrays(yy[:, None], xx[None, :]), axis=2)


def workout_4DSTEM_datacube_DP_size(FourD_STEM, rsize, gridshape):
    """
    Calculate 4D-STEM datacube diffraction pattern gridsize and resampling function.

    Parameters
    ----------
    fourD_STEM : bool or array_like
        Pass fourD_STEM = True gives 4D STEM output with native simulation grid
        sampling. Alternatively, to save disk space a tuple containing pixel
        size and diffraction space extent of the datacube can be passed in. For
        example ([64,64],[1.2,1.2]) will output diffraction patterns measuring
        64 x 64 pixels and 1.2 x 1.2 inverse Angstroms.
    rsize : (2,) array_like
        Real space size of simulation grid
    gridshape : (2,) array_like
        Pixel size of gridshape

    Returns
    -------
    gridout : (2,) array_like
        Pixel size of the diffraciton pattern output
    resize : function
        A function that takes diffraction patterns from the simulation and
        resamples and crops them to the requested size.
    """
    # Check whether a resampling directive has been given
    if isinstance(FourD_STEM, (list, tuple)):
        gridout = FourD_STEM[0]

        if len(FourD_STEM) > 1:
            # Get output grid and diffraction space size of that grid from tuple
            Ksize = FourD_STEM[1]

            #
            diff_pat_crop = np.round(np.asarray(Ksize) * np.asarray(rsize[:2])).astype(
                np.int
            )

            # Define resampling function to crop and interpolate
            # diffraction patterns
            def resize(array):
                cropped = crop(np.fft.fftshift(array, axes=(-1, -2)), diff_pat_crop)
                return fourier_interpolate_2d(cropped, gridout, norm="conserve_L1")

        else:
            # The size in inverse Angstrom of the grid
            Ksize = np.asarray(gridout) / np.asarray(rsize)

            # Define resampling function to just crop diffraction
            # patterns
            def resize(array):
                return crop(np.fft.fftshift(array, axes=(-1, -2)), gridout)

    else:
        # If no resampling then the output size is just the simulation
        # grid size
        gridout = size_of_bandwidth_limited_array(gridshape)

        # The size in inverse Angstrom of the grid
        Ksize = np.asarray(gridout) / np.asarray(rsize)

        # Define a resampling function that does nothing
        def resize(array):
            return np.fft.fftshift(array, axes=(-1, -2))

    return gridout, resize, Ksize


def STEM(
    rsize,
    probe,
    method,
    nslices,
    eV,
    alpha,
    batch_size=1,
    detectors=None,
    FourD_STEM=False,
    datacube=None,
    scan_posn=None,
    dtype=torch.float32,
    device=None,
    tiling=[1, 1],
    seed=None,
    showProgress=True,
    method_args=(),
    method_kwargs={},
    STEM_image=None,
):
    """
    Perform a scanning transmission electron microscopy (STEM) image simulation.

    Will return an array containing conventional STEM images and/or a 4D-STEM
    datacube depending on inputs

    Parameters
    ----------
    rsize : (2,) array_like
        The real space size of the grid in Angstroms
    probe : (Y,X) array_like
        The probe that will be rastered over the object
    method : function
        A function that takes a probe and propagates it to the exit surface of
        the specimen
    nslices : int, array_like
        The number of slices to perform multislice over
    eV : float
        Accelerating voltage of the probe, needed to work out probe sampling
        requirements
    alpha : float
        The convergence angle of the probe in mrad, needed to work out probe
        sampling requirements

    Keyword arguments
    -----------------
    batch_size : int, optional
        The multislice algorithm can be performed on multiple scattering matrix
        columns at once to parrallelize computation, this number is set by
        batch_size.
    detectors : (Ndet, Y, X) array_like, optional
        Diffraction plane detectors to perform conventional STEM imaging. If
        None is passed then no conventional STEM images will be returned.
    fourD_STEM : bool or array_like, optional
        Pass fourD_STEM = True to perform 4D-STEM simulations. To save disk
        space a tuple containing pixel size and diffraction space extent of the
        datacube can be passed in. For example ([64,64],[1.2,1.2]) will output
        diffraction patterns measuring 64 x 64 pixels and 1.2 x 1.2 inverse
        Angstroms.
    datacube :  (Ny, Nx, Y, X) array_like, optional
        datacube for 4D-STEM output, if None is passed (default) this will be
        initialized in the function. If a datacube is passed then the result
        will be added by the STEM routine (useful for multiple frozen phonon
        iterations)
    scan_posn :  (...,2) array_like, optional
        Array containing the STEM scan positions in fractional coordinates.
        If provided scan_posn.shape[:-1] will give the shape of the STEM image.
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit floating point
    device : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    tiling : (2,) array_like, optional
        Tiling of a repeat unit cell on simulation grid, STEM raster will only
        scan a single unit cell.
    seed : int, optional
        Seed for the random number generator for frozen phonon configurations
    showProgress : bool, optional
        Pass showProgress=False to disable progress bar.
    method_args : list, optional
        Arguments for the method function used to propagate probes to the exit
        surface
    method_kwargs : Dict, optional
        Keyword arguments for the method function used to propagate probes to
        the exit surface
    STEM_image : (Ndet,Ny,Nx) array_like, optional
        Array that will contain the conventional STEM images, if not passed
        will be initialized within the function. If it is passed then the result
        will be accumulated within the function, which is useful for multiple
        frozen phonon iterations.
    """
    from .utils.torch_utils import detect

    # Get number of thicknesses in the series
    nthick = len(nslices)

    if device is None:
        device = get_device(device)

    # Get shape of grid
    if torch.is_tensor(probe):
        gridshape = probe.shape[-3:-1]
    else:
        gridshape = probe.shape[-2:]

    # Generate scan positions in units of pixels if not supplied
    if scan_posn is None:
        scan_posn = generate_STEM_raster(rsize[:2], eV, alpha, tiling)

    # Number of scan positions
    scan_shape = scan_posn.shape[:-1]
    nscantot = np.prod(scan_shape)
    scan_posn = scan_posn.reshape((nscantot, 2))

    # Ensure scan_posn is a pytorch tensor with same device and datatype as other
    # arrays
    scan_posn = torch.as_tensor(scan_posn).to(device).type(dtype)

    # Assume real space probe is passed in so perform Fourier transform in
    # anticipation of application of Fourier shift theorem
    probe_ = ensure_torch_array(probe, dtype=dtype, device=device)
    probe_ = torch.fft(probe_, signal_ndim=2)

    # Work out whether to perform conventional STEM or not
    conventional_STEM = detectors is not None

    if conventional_STEM:
        # Get number of detectors
        ndet = detectors.shape[0]

        # Initialize array in which to store resulting STEM images
        if STEM_image is None:
            STEM_image = np.zeros((ndet, nthick, nscantot))
        else:
            STEM_image = STEM_image.reshape((ndet, nthick, nscantot))

        # Also move detectors to pytorch if necessary
        D = ensure_torch_array(detectors, device=device, dtype=dtype)
    else:
        STEM_image = None

    # Initialize array in which to store resulting 4D-STEM datacube if required
    if FourD_STEM:

        # Get diffraction pattern gridsize in pixels from input and function
        # to resample the simulation output to store in the datacube
        gridout, resize, _ = workout_4DSTEM_datacube_DP_size(
            FourD_STEM, rsize, gridshape
        )

        # Check whether a datacube is already provided or not
        if datacube is None:
            datacube = np.zeros((nthick, nscantot, *gridout))
        # else:
        # If datacube is provided, flatten the scan dimensions of the array
        # the reshape function `should` provide a new view of the object,
        # not a copy of the whole array

        # If we have a list or tuple of datacubes then we need to reshape
        # them one by one (ie a list of hdf5 datacubes)
        # if isinstance(datacube,(list,tuple)):
        #     for i,dcube in enumerate(datacube):
        #         datacube[i] = dcube[:].reshape((nscantot,*dcube.shape[-2:]))
        # else:
        #     # Otherwise we can reshape all the datacubes as a contiguous array
        #     datacube = datacube.reshape((nthick, nscantot, *datacube.shape[-2:]))

    # This algorithm allows for "batches" of probe to be sent through the
    # multislice algorithm to achieve some speed up at the cost of storing more
    # probes in memory

    if seed is None and batch_size > 1:
        # If no seed passed to random number generator then make one to pass to
        # the multislice algorithm. This ensure that each probe sees the same
        # frozen phonon configuration if we are doing batched multislice
        # calculations
        seed = np.random.randint(0, 2 ** 31 - 1)

    for i in tqdm(
        range(int(np.ceil(nscantot / batch_size))),
        disable=not showProgress,
        desc="Probe positions",
    ):

        # Make shifted probes
        scan_index = np.arange(
            i * batch_size, min((i + 1) * batch_size, nscantot), dtype=np.int
        )

        # The shift operator array array will be of size batch_size x Y x X
        probes = fourier_shift_array(
            gridshape,
            torch.as_tensor(scan_posn[scan_index]),
            dtype=dtype,
            device=device,
            units="fractional",
        )

        # Apply shift to original probe
        probes = complex_mul(probe_.view(1, *probe_.size()), probes)

        # Thickness series
        #  - need to take the difference between sequential thickness variations
        for it, t in enumerate(np.diff(nslices, prepend=0)):

            # Evaluate exit surface wave function from input probes
            probes = method(
                probes, t, *method_args, posn=scan_posn[scan_index], **method_kwargs
            )

            # Calculate amplitude of probes
            amp = amplitude(probes)

            # Correct normalization in Fourier space
            amp /= np.prod(probes.size()[-3:-1])

            # Calculate STEM images
            if conventional_STEM:
                # broadcast detector and probe arrays to
                # ndet x batch_size x Y x X and reduce final two dimensions
                STEM_image[:ndet, it, scan_index] += detect(D, amp).cpu().numpy()

            # Store datacube
            if FourD_STEM:
                DPS = resize(amp.cpu().numpy())
                ind = np.unravel_index(scan_index, scan_shape)
                for idp, DP in enumerate(DPS):
                    datacube[it][ind[0][idp], ind[1][idp]] += DP

    # Unflatten 4D-STEM datacube scan dimensions, use numpy squeeze to
    # remove superfluous dimensions (ones with length 1)
    # if FourD_STEM:
    # datacube = np.squeeze(
    #     datacube.reshape(nthick, *scan_shape, *datacube.shape[-2:])
    # )

    if conventional_STEM:
        STEM_image = np.squeeze(STEM_image.reshape(ndet, nthick, *scan_shape))
    return {"STEM images": STEM_image, "datacube": datacube}


def unit_cell_shift(array, axis, shift, tiles):
    """
    Shift an array an integer number of unit cell.

    For an array consisting of a number of repeat units given by tiles
    shift than array an integer number of unit cells.
    """
    indices = torch.remainder(torch.arange(array.shape[-3 + axis]) - shift)
    if axis == 0:
        return array[indices, :, :]
    if axis == 1:
        return array[:, indices, :]


def max_grid_resolution(gridshape, rsize, bandwidthlimit=2 / 3, eV=None):
    """
    For a given pixel sampling, return maximum multislice grid resolution.

    For a given grid pixel size (gridshape) and real space size (rsize) return
    maximum resolution permitted by the multislice grid. If the probe
    accelerating voltage is passed in as eV resolution will be given in units
    of mrad, otherwise resolution will be given in units of inverse Angstrom.
    """
    max_res = min([gridshape[x] / rsize[x] / 2 * bandwidthlimit for x in range(2)])
    if eV is None:
        return max_res

    return max_res / wavev(eV) * 1e3


class scattering_matrix:
    """Scattering matrix object for calculations using the PRISM algorithm."""

    def __init__(
        self,
        rsize,
        propagators,
        transmission_functions,
        nslice,
        eV,
        alpha,
        GPU_streaming=False,
        batch_size=30,
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
        stored_gridshape=None,
    ):
        """Initialize a scattering matrix.

        Parameters
        ----------
        rsize : (2,) array_like
            Real space size of the simulation grid in Angstrom
        propagators : (N,Y,X,2) torch.array
            Fresnel free space operators required for the multislice algorithm
            used to propagate the scattering matrix
        transmission_functions : (N,Y,X,2)
            The transmission functions describing the electron's interaction
            with the specimen for the multislice algorithm
        nslice : int
            The number of slices of the specimen to propagate the scattering
            matrix to
        eV : float
            Electron probe energy in electron-volts
        alpha : float
            Maximum input angle for the scattering matrix, should match the
            probe forming aperture used in experiment

        Keyword arguments
        -----------------
        GPU_streaming : bool, optional
            If True, the scattering matrix will be stored off GPU RAM and
            streamed to GPU RAM as necessary, does nothing if the calculation
            is CPU only
        batch_size : int, optional
            The multislice algorithm can be performed on multiple scattering
            matrix columns at once to parrallelize computation, this number is
            set by batch_size.
        device : torch.device, optional
            torch.device object which will determine which device (CPU or GPU)
            the calculations will run on. By default this will be determined
            by what device the transmission functions are stored on.
        PRISM_factor : int (2,) array_like
            The PRISM "interpolation factor" this is the amount by which the
            scattering matrices are cropped in real space to speed up
            calculations see Ophus, Colin. "A fast image simulation algorithm
            for scanning transmission electron microscopy." Advanced structural
            and chemical imaging 3.1 (2017): 13 for details on this.
        seed : int32, optional
            A seed to control seeding of the frozen phonon approximation
        showProgress : bool, optional
            Pass showProgress = False to disable live progress readout
        bandwidth_limit : float, optional
            Band-width limiting of the transmission function and propagators to
            prevent wrap-around error in the multislice algorithm, 2/3 by
            default
        Fourier_space_output : bool, optional
            If True the scattering matrix output will be stored in reciprocal
            space, default is False
        subslicing : bool, optional
            Pass subslicing=True to access propagation to sub-slices of the
            unit cell, in this case nslices is taken to be in units of subslices
            to propagate rather than unit cells (i.e. nslices = 3 will propagate
            1.5 unit cells for a subslicing of 2 subslices per unit cell)
        transposed : bool, optional
            Make a "transposed" scattering matrix - see Brown et al. (2019)
            Physical Review Research paper for a discussion of this and its
            applications
        stored_gridshape : (2,) array_like
            Size of the stored grid, can be chosen to be smaller than the
            multislice grid to speed up computation of a smaller diffraction
            space view than that implied by the multislice at no cost to
            computational accuracy.
        """
        # Get size of grid
        gridshape = transmission_functions.size()[-3:-1]

        # Datatype (precision) is inferred from transmission functions
        self.dtype = transmission_functions.dtype

        # Device (CPU or GPU) is also inferred from transmission functions
        self.device = device
        if GPU_streaming:
            self.device = torch.device("cpu")
        elif self.device is None:
            self.device = transmission_functions.device

        # Get alpha in units of inverse Angstrom
        self.alpha_ = wavev(eV) * alpha * 1e-3

        self.PRISM_factor = PRISM_factor

        # Make a list of beams in the scattering matrix
        # Take beams inside the aperture and every nth beam where n is the
        # PRISM "interpolation" factor
        q = q_space_array(gridshape, rsize)
        inside_aperture = np.less_equal(q[0] ** 2 + q[1] ** 2, self.alpha_ ** 2)
        mody, modx = [
            np.mod(np.fft.fftfreq(x, 1 / x).astype(np.int), p) == 0
            for x, p in zip(gridshape, self.PRISM_factor)
        ]
        self.beams = np.nonzero(
            np.logical_and(
                np.logical_and(inside_aperture, mody[:, None]), modx[None, :]
            )
        )
        self.beams = [(x + y // 2) % y - y // 2 for x, y in zip(self.beams, gridshape)]

        self.nbeams = len(self.beams[0])

        # For a scattering matrix stored in real space there is the option
        # of storing it on a much smaller pixel grid than the grid used for
        # multislice. This is handy when, for example, a large grid
        # is required for a converged multislice calculation but only
        # the bright-field region of diffraction (small angle region)
        # is of interest. Be careful using this in conjunction with
        # multiple calls of the propagation method for the scattering matrix,
        # as information outside the angular range of the stored grid is lost.
        self.crop_output = not (stored_gridshape is None)
        if self.crop_output:
            self.stored_gridshape = stored_gridshape

            # We will only store output of the scattering matrix up to the band
            # width limit of the calculation, since this is a circular band-width
            # limit on a square grid we have to get somewhat fancy and store a mapping
            # of the pixels within the bandwidth limit to a one-dimensional vector
            self.bw_mapping = np.argwhere(
                np.logical_and(
                    (
                        np.abs(np.fft.fftfreq(gridshape[0], d=1 / gridshape[0]))
                        < self.stored_gridshape[0] // 2
                    )[:, np.newaxis],
                    (
                        np.abs(np.fft.fftfreq(gridshape[1], d=1 / gridshape[1]))
                        < self.stored_gridshape[1] // 2
                    )[np.newaxis, :],
                )
            )

        else:
            self.stored_gridshape = size_of_bandwidth_limited_array(
                transmission_functions.size()[-3:-1]
            )

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

        self.gridshape, self.rsize, self.eV = [np.asarray(gridshape), rsize, eV]
        self.bw_mapping = (
            self.bw_mapping + self.gridshape // 2
        ) % self.gridshape - self.gridshape // 2
        self.PRISM_factor, self.tiling = [PRISM_factor, tiling]
        self.doPRISM = np.any([self.PRISM_factor[i] > 1 for i in [0, 1]])
        self.Fourier_space_output = Fourier_space_output
        self.nsubslices = transmission_functions.size()[1]
        slices = generate_slice_indices(nslice, self.nsubslices, subslicing=subslicing)
        self.GPU_streaming = GPU_streaming
        self.transposed = transposed

        self.seed = seed
        if self.seed is None:
            # If no seed passed to random number generator then make one to pass to
            # the multislice algorithm. This ensure that each column in the scattering
            # matrix sees the same frozen phonon configuration
            self.seed = np.random.randint(
                0, 2 ** 31 - 1, size=len(slices), dtype=np.uint32
            )

        # This switch tells the propagate function to initialize the Smatrix
        # to plane waves
        self.initialized = False
        # Propagate wave functions of scattering matrix
        self.current_slice = 0
        self.show_Progress = showProgress
        self.Propagate(
            nslice,
            propagators,
            transmission_functions,
            subslicing=subslicing,
            showProgress=self.show_Progress,
            batch_size=batch_size,
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
        """
        Propagate a scattering matrix to slice nslice of the specimen.

        Parameters
        ----------
        nslice : int
            The slice in the specimen to propagate the scattering matrix to
        propagators : (N,Y,X,2) torch.array
            Fresnel free space operators required for the multislice algorithm
            used to propagate the scattering matrix
        transmission_functions : (N,Y,X,2)
            The transmission functions describing the electron's interaction
            with the specimen for the multislice algorithm

        Keyword arguments
        -----------------
        batch_size : int, optional
            The multislice algorithm can be performed on multiple scattering
            matrix columns at once to parrallelize computation, this number is
            set by batch_size.
        subslicing : bool, optional
            Pass subslicing=True to access propagation to sub-slices of the
            unit cell, in this case nslices is taken to be in units of subslices
            to propagate rather than unit cells (i.e. nslices = 3 will propagate
            1.5 unit cells for a subslicing of 2 subslices per unit cell)
        showProgress : bool, optional
            Pass showProgress = False to disable live progress readout
        transpose : bool, optional
            Make a "transposed" scattering matrix - see Brown et al. (2019)
            Physical Review Research paper for a discussion of this and its
            applications
        """
        from .Probe import plane_wave_illumination

        # Initialize scattering matrix if necessary
        if not self.initialized:
            if self.Fourier_space_output:
                self.S = torch.zeros(
                    self.nbeams, self.nbout, 2, dtype=self.dtype, device=self.device
                )
            else:
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
                        tilt=[self.beams[0][ibeam], self.beams[1][ibeam]],
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
                    self.S[ibeam] = psi[self.bw_mapping[:, 0], self.bw_mapping[:, 1], :]
                else:
                    self.S[ibeam] = fourier_interpolate_2d_torch(
                        psi, self.stored_gridshape, qspace_in=True, correct_norm=False
                    )
            self.initialized = True

        # Make nslice_ which always accounts of subslices of the cyrstal structure
        if subslicing:
            nslice_ = nslice
        else:
            nslice_ = nslice * self.nsubslices

        # Work out direction of propagation through specimen
        if nslice_ != self.current_slice:
            direction = np.sign(nslice_ - self.current_slice)
        else:
            direction = 1

        if direction == 0:
            direction = 1

        if nslice_ > len(self.seed):
            # Add new seeds to determine random translations for frozen-phonon
            # multislice (required for reversability of multislice) if required
            self.seed = np.concatenate(
                [
                    self.seed,
                    np.random.randint(0, 2 ** 31 - 1, size=nslice_ - len(self.seed)),
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
            range(int(np.ceil(self.nbeams / batch_size))),
            disable=not showProgress,
            desc="Calculating S-matrix",
        ):
            psi[...] = 0.0
            beams = np.arange(
                i * batch_size, min((i + 1) * batch_size, self.nbeams), dtype=np.int
            )

            if self.Fourier_space_output:
                # Expand S-matrix input to full grid for multislice propagation
                psi[
                    : beams.shape[0], self.bw_mapping[:, 0], self.bw_mapping[:, 1], :
                ] = self.S[beams]
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
                qspace_in=self.Fourier_space_output,
                qspace_out=self.Fourier_space_output,
                transpose=self.transposed,
                output_to_bandwidth_limit=False,
                reverse=direction < 0,
            )

            if self.GPU_streaming:
                output = output.to(self.device)

            if self.Fourier_space_output:

                self.S[beams] = output[
                    :, self.bw_mapping[:, 0], self.bw_mapping[:, 1], :
                ] * np.sqrt(np.prod(self.stored_gridshape) / np.prod(self.gridshape))
            else:
                output = fourier_interpolate_2d_torch(
                    output, self.stored_gridshape, correct_norm=False
                )
                self.S[beams] = output

    def PRISM_crop_window(self, win=None, device=None):
        """Calculate 2D array indices of STEM crop window."""
        device = get_device(device)
        if win is None:
            win = self.PRISM_factor

        crop_ = [
            torch.arange(
                -self.stored_gridshape[i] // (2 * win[i]),
                self.stored_gridshape[i] // (2 * win[i]),
                device=device,
            )
            for i in range(2)
        ]
        return crop_

    def __call__(self, probes, nslices, posn=None, Smat=None, scan_transform=None):
        """
        Calculate exit-surface waves function using the scattering matrix.

        Parameters
        ----------
        probes : (N,Y,X,2) torch.array
            Input wave functions to calculate exit surface wave functions from
            must be in Diffraction space
        nslices :
            Does nothing, only there to match call signature for STEM routine
        posn : array_like (N,2)
            Positions of
        S : array_like (Nbeams,Y,X,2)
            Scattering matrix object

        Returns
        -------
        output : (N,Y,X,2) torch.array
            Exit surface wave functions
        """
        from copy import deepcopy

        if Smat is None:
            Smat = self.S
        Sshape = Smat.shape

        device = Smat.device
        crop_ = self.PRISM_crop_window(device=device)
        # Ensure posn and probes are pytorch arrays
        probes = ensure_torch_array(probes, dtype=self.dtype, device=device)

        # Ensure probes tensors correspond to the shape N x Y x X x 2
        # If they have the shape Y x X x 2 then reshape to 1 x Y x X x 2
        if probes.ndim < 4:
            probes = probes.view(1, *probes.shape)

        # Get number of probes
        nprobes = probes.shape[0]

        if posn is None:
            posn = torch.zeros(nprobes, 2, device=device, dtype=self.dtype)
        else:
            posn = torch.as_tensor(posn, device=device, dtype=self.dtype).view(
                (nprobes, 2)
            )

        if scan_transform is not None:
            posn = scan_transform(posn)

        # TODO decide whether to remove the Fourier_space_output option
        if self.Fourier_space_output:

            # A note on normalization: an individual probe enters the STEM routine
            # with sum_squared intensity of 1, but the STEM routine applies an
            # FFT so the sum_squared intensity is now equal to # pixels
            # For a correct matrix multiplication we must now divide by sqrt(# pixels)
            probe_vec = complex_matmul(
                probes[:, self.beams[0], self.beams[1]], Smat
            ) / np.sqrt(np.prod(self.gridshape))

            # Now reshape output from vectors to square arrays
            probes = torch.zeros(
                nprobes, *self.stored_gridshape, 2, dtype=self.dtype, device=self.device
            )
            probes[:, self.bw_mapping[:, 0], self.bw_mapping[:, 1], :] = probe_vec

            # Apply PRISM cropping in real space if appropriate
            if self.doPRISM:
                shape = probes.size()

                probes = torch.ifft(probes, signal_ndim=2).flatten(-3, -2)
                for k in range(nprobes):

                    # Calculate windows in vertical and horizontal directions
                    window = crop_window_to_flattened_indices_torch(
                        [
                            (crop_[i] + posn[k, i] * self.stored_gridshape[i])
                            % self.stored_gridshape[i]
                            for i in range(2)
                        ],
                        self.stored_gridshape,
                    )
                    probe = deepcopy(probes[k])
                    probes[k] = 0
                    probes[k, window, :] = probe[window, :]

                probes = probes.reshape(shape)

                # Transform probe back to Fourier space
                return torch.fft(probes, signal_ndim=2)

            return probes
        else:
            # Flatten the array dimensions
            flattened_shape = [Smat.size(-3) * Smat.size(-2), 2]
            output = torch.zeros(
                probes.size(0), *flattened_shape, dtype=self.dtype, device=Smat.device
            )

            # For evaluating the probes in real space we only want to perform the matrix
            # multiplication and summation within the real space PRISM cropping region
            for k in range(probes.size(0)):

                window = crop_window_to_flattened_indices_torch(
                    [
                        (
                            crop_[i]
                            + (posn[k, i] * Sshape[-3 + i]).type(torch.LongTensor)
                        )
                        % Sshape[-3 + i]
                        for i in range(2)
                    ],
                    Sshape[-3:-1],
                )

                output[k, window] += complex_matmul(
                    probes[k, self.beams[0], self.beams[1]],
                    Smat.view(self.nbeams, *flattened_shape)[:, window],
                ) / np.sqrt(np.prod(probes.size()[-3:-1]))

            output = crop_torch(
                output.reshape(probes.size(0), *Smat.size()[-3:]), self.stored_gridshape
            )

            return torch.fft(output, signal_ndim=2)

    def STEM_with_GPU_streaming(
        self,
        detectors=None,
        FourD_STEM=None,
        datacube=None,
        STEM_image=None,
        nstreams=None,
        df=0,
        aberrations=[],
        ROI=[0.0, 0.0, 1.0, 1.0],
        device=None,
        scan_posns=None,
        showProgress=True,
    ):
        """
        Perform STEM with scattering matrix streamed between RAM and GPU memory.

        This allows much larger fields of view to be calculated with relatively
        modest graphics card memory. The STEM raster is segmented into spatially
        close clusters and the probe positions in these clusters are processed
        sequentially, with the relevant part of the scattering matrix streamed
        from CPU to GPU memory.

        Keyword arguement
        ----------
        detectors : (Ndet, Y, X) array_like, optional
            Diffraction plane detectors to perform conventional STEM imaging. If
            None is passed then no conventional STEM images will be returned.
        fourD_STEM : bool or array_like, optional
            Pass fourD_STEM = True to perform 4D-STEM simulations. To save disk
            space a tuple containing pixel size and diffraction space extent of the
            datacube can be passed in. For example ([64,64],[1.2,1.2]) will output
            diffraction patterns measuring 64 x 64 pixels and 1.2 x 1.2 inverse
            Angstroms.
        datacube :  (Ny, Nx, Y, X) array_like, optional
            datacube for 4D-STEM output, if None is passed (default) this will be
            initialized in the function. If a datacube is passed then the result
            will be added by the STEM routine (useful for multiple frozen phonon
            iterations)
        STEM_image : (Ndet,Ny,Nx) array_like, optional
            Array that will contain the conventional STEM images, if not passed
            will be initialized within the function. If it is passed then the result
            will be accumulated within the function, which is useful for multiple
            frozen phonon iterations.
        nstreams : int, optional
            Number of streams (seperate transfers from CPU to GPU memory). If
            None this will just be set to the product of the PRISM interpolation
            factor
        df : float, optional
            Defocus in Angstrom
        aberrations : list, optional
            A list containing a set of the class aberration, pass an empty list for
            an unaberrated contrast transfer function.
        ROI : (4,) array_like
            Fraction of the unit cell to be scanned. Should contain [y0,x0,y1,x1]
            where [x0,y0] and [x1,y1] are the bottom left and top right coordinates
            of the region of interest (ROI) expressed as a fraction of the total
            grid (or unit cell).
        device : torch.device, optional
            torch.device object which will determine which device (CPU or GPU) the
            calculations will run on.
        scan_posn :  (...,2) array_like, optional
            Array containing the STEM scan positions in fractional coordinates.
            If provided scan_posn.shape[:-1] will give the shape of the STEM
            image.
        showProgress : bool, optional
            Pass showProgress=False to disable progress bar.
        """
        device = get_device(device)

        # Get indices of PRISM cropping window
        crop_ = [x.cpu().numpy() for x in self.PRISM_crop_window()]

        # Make the STEM probe
        probe = focused_probe(
            self.gridshape,
            self.rsize[:2],
            self.eV,
            self.alpha_,
            df=df,
            aberrations=aberrations,
            app_units="invA",
        )
        probe = cx_from_numpy(probe, device=device, dtype=self.dtype)

        # Make scan positions if none already provided
        if scan_posns is None:
            scan_posns = generate_STEM_raster(
                self.rsize, self.eV, self.alpha_, tiling=self.tiling, ROI=ROI, invA=True
            )
        # Get scan (and STEM image) array shape and total number of scan positions
        scan_shape = scan_posns.shape[:-1]
        nscan = np.product(scan_shape)

        # Flatten scan positions to simplify iteration later on.
        scan_posns = scan_posns.reshape((nscan, 2))

        # Calculate default 4D-STEM diffraction pattern sampling
        if FourD_STEM is True:
            GS = self.stored_gridshape
            FourD_STEM = [GS, GS / self.rsize[:2]]

        # Allocate diffraction pattern and STEM images if not already provided
        if FourD_STEM:
            gridout = workout_4DSTEM_datacube_DP_size(
                FourD_STEM, self.rsize, self.gridshape
            )[0]
        if (datacube is None) and FourD_STEM:
            datacube = np.zeros((*scan_shape, *gridout))
        if not FourD_STEM:
            datacube = None

        # If detectors are provided then we are doing conventional STEM
        doConventionalSTEM = detectors is not None

        # Initialize STEM images if not provided
        if doConventionalSTEM:
            ndet = detectors.shape[0]
            if STEM_image is None:
                STEM_image = np.zeros((ndet, nscan))
        else:
            STEM_image = None

        if nstreams is None:
            # If the number of seperate streams is not suggested by the
            # user, make this equal to the product of the PRISM factor
            nstreams = int(np.product(self.PRISM_factor))

        # Divide up the scan positions into clusters based on Euclidean
        # distance
        from sklearn.cluster import Birch

        model = Birch(threshold=0.01, n_clusters=nstreams)
        yhat = model.fit_predict(scan_posns)
        clusters = np.unique(yhat)

        # Now do STEM with each of the scan position clusters, streaming
        # only the necessary bits of the scattering matrix to the GPU.
        Datacube_segment = None
        STEM_image_segment = None
        FlatS = self.S.reshape((self.nbeams, np.prod(self.stored_gridshape), 2))

        # Loop over probe positions clusters. This would be a good candidate
        # for multi-GPU work.
        for cluster in tqdm(
            clusters, desc="Probe position clusters", disable=not showProgress
        ):
            # Get map of probe positions in cluster
            points = np.nonzero(yhat == cluster)[0]
            npoints = len(points)

            # Get segments of images to update
            if doConventionalSTEM:
                STEM_image_segment = STEM_image[:, points]
            if FourD_STEM:
                Datacube_segment = np.zeros((1, npoints, 1, *gridout))
            pix_posn = scan_posns[points] * np.asarray(self.stored_gridshape)

            # Work out bounds of the rectangular region of the scattering
            # matrix to stream to the GPU
            ymin, ymax = [
                int(np.floor(np.amin(pix_posn[:, 0]) + crop_[0][0])),
                int(np.ceil(np.amax(pix_posn[:, 0]) + crop_[0][-1])),
            ]
            xmin, xmax = [
                int(np.floor(np.amin(pix_posn[:, 1]) + crop_[1][0])),
                int(np.ceil(np.amax(pix_posn[:, 1]) + crop_[1][-1])),
            ]
            size = np.asarray([(ymax - ymin), (xmax - xmin)])

            # Get indices of region of scattering matrix to stream to GPU
            window = [np.arange(a, b) for a, b in zip([ymin, xmin], [ymax, xmax])]
            indices = crop_window_to_flattened_indices_torch(
                window, self.stored_gridshape
            )

            # Get segment of the scattering matrix to stream to GPU
            segmentshape = [len(x) for x in window]
            SegmentS = FlatS[:, indices, :].reshape((self.nbeams, *segmentshape, 2))

            # Define a function that will map probe positions for the global
            # scattering matrix to their correct place on the smaller scattering
            # matrix streamed to the GPU.
            gshape = torch.as_tensor(self.stored_gridshape).to(device).type(self.dtype)
            Origin = torch.as_tensor([ymin, xmin]).to(device).type(self.dtype)
            segment_size = torch.as_tensor(size).to(device).type(self.dtype)

            def scan_transform(posn):
                return (posn * gshape - Origin) / segment_size

            # Keyword arguments to be passed to the __call__ function by the
            # STEM routine
            kwargs = {"Smat": SegmentS.to(device), "scan_transform": scan_transform}

            # Calculate STEM images
            STEM(
                self.rsize,
                probe,
                self.__call__,
                [1],
                self.eV,
                self.alpha_,
                detectors=detectors,
                FourD_STEM=FourD_STEM,
                datacube=Datacube_segment,
                scan_posn=scan_posns[points].reshape((npoints, 1, 2)),
                STEM_image=STEM_image_segment,
                method_kwargs=kwargs,
                showProgress=False,
                device=device,
            )

            if doConventionalSTEM:
                STEM_image[:, points] += STEM_image_segment
            if FourD_STEM:
                for point, Dp in zip(points, Datacube_segment[0]):
                    y, x = np.unravel_index(point, scan_shape)
                    datacube[y, x] += Dp[0]
        # plt.show(block=True)

        # Unflatten 4D-STEM datacube scan dimensions, use numpy squeeze to
        # remove superfluous dimensions (ones with length 1)
        # if FourD_STEM:
        #     datacube = datacube.reshape(*scan_shape, *datacube.shape[-2:])

        if doConventionalSTEM:
            STEM_image = np.squeeze(STEM_image.reshape(ndet, *scan_shape))

        # Return STEM images and datacube as a dictionary. If either of these
        # objects were not calculated the dictionary will contain None for those
        # entries.
        return {"STEM images": STEM_image, "datacube": datacube}
