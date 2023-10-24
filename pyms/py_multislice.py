"""Module containing functions for core multislice and PRISM algorithms."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from .Probe import wavev, focused_probe
from .utils.numpy_utils import (
    ensure_array,
    bandwidth_limit_array,
    q_space_array,
    fourier_interpolate,
    crop,
)
from .utils.torch_utils import (
    complex_to_real_dtype_torch,
    iscomplex,
    # roll_n,
    fourier_shift_array,
    amplitude,
    get_device,
    ensure_torch_array,
    crop_to_bandwidth_limit_torch,
    size_of_bandwidth_limited_array,
    fourier_interpolate_torch,
    crop_window_to_flattened_indices_torch,
    crop_window_to_periodic_indices,
    crop_torch,
    fourier_shift_array_1d,
)


def tqdm_handler(showProgress):
    """Handle showProgress boolean or string input for the tqdm progress bar."""
    if isinstance(showProgress, str):
        if showProgress.lower() == "notebook":
            from tqdm import tqdm_notebook as tqdm
        tdisable = False
    elif isinstance(showProgress, bool):
        tdisable = not showProgress
        from tqdm import tqdm
    return tdisable, tqdm


def thickness_to_slices(
    thicknesses, slice_thickness, subslicing=False, subslices=[1.0]
):
    """Convert thickness in Angstroms to number of multislice slices."""
    from .__init__ import _int

    t = np.asarray(ensure_array(thicknesses))
    if subslicing:
        # Work out how many slice of the structure is closest to the desired
        # output thicknesses
        m = len(subslices)
        nslices = (t // slice_thickness).astype(_int) * m
        from scipy.spatial.distance import cdist

        # Work out which subslices of the structure
        remainder = (t % slice_thickness) / slice_thickness
        n = len(remainder)
        dist = cdist(
            remainder.reshape((n, 1)),
            np.concatenate(([0], subslices[:-1])).reshape((m, 1)),
        )
        z = [0] + (nslices + np.asarray([i for i in np.argmin(dist, axis=1)])).tolist()
        return [np.arange(z[i], z[i + 1]) for i in range(len(z) - 1)]
    else:

        return np.ceil(t / slice_thickness).astype(_int)


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
    Returns
    -------
    P : (n,Y,X)
        Fresnel free-space progators, the first dimension will be of size
        len(`gridsize`)
    """
    from .Probe import make_contrast_transfer_function

    # We will use the make_contrast_transfer_function function to generate
    # the propagator, the aperture of this propagator will go out to the maximum
    # possible and the function bandwidth_limit_array will provide the band
    # width_limiting
    gridmax = np.asarray(gridshape) / np.asarray(gridsize[:2]) / 2
    app = np.hypot(*gridmax)

    # Intitialize array
    prop = np.zeros((len(subslices), *gridshape), dtype=complex)
    for islice, s_ in enumerate(subslices):
        if islice == 0:
            deltaz = s_ * gridsize[2]
        else:
            deltaz = (s_ - subslices[islice - 1]) * gridsize[2]

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
    tiling=[1, 1],
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
    Veff=None,
):
    """
    Multislice algorithm for scattering of an electron probe.

    Parameters
    ----------
    probes : (n,Y,X) or (Y,X) complex array_like
        Electron probe wave function(s)
    nslices : int, array_like
        The number of slices (iterations) to perform multislice over, if an
    propagators : (Z,Y,X) or (Y,X) complex array_like
        Fresnel free space operators required for the multislice algorithm
        used to propagate the scattering matrix
    transmission_functions : (Z,nT,Y,X) complex array_like
        The transmission functions describing the electron's interaction
        with the specimen for the multislice algorithm
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
        and False otherwise (this is the default)
    qspace_out : bool, optional
        Should be set to True if the output wavefunction is desired in momentum
        (q) space and False otherwise (this is the default)
    posn : None, optional
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
    Veff: (nelements,nsubslices,Y,X) complex torch.tensor, optional
        Effective scattering potential (μ's) for EELS and/or EDX

    Returns
    -------
    psi : (Y,X) or (n,Y,X) complex torch.tensor or np.ndarray
        Exit surface wave functions as a pytorch tensor or numpy array (default)
        depending on whether return_numpy is True or False. If the input `probes`
        is two dimensional then n = 1
    """
    # Dimensions for FFT
    d_ = (-2, -1)

    # Initialize device cuda if available, CPU if no cuda is available
    device = get_device(device_type)

    if Veff is not None:
        crosssec=probes[1]
        probes=probes[0]
        nsignals = crosssec.shape[0]
        batch_size = crosssec.shape[1]
        value = torch.zeros((batch_size,), dtype=torch.cfloat, device=device)

    # If a single integer is passed to the routine then Seed random number generator,
    # , if None then np.random.RandomState will use the system clock as a seed
    seed_provided = not (seed is None)
    if not seed_provided:
        r = np.random.RandomState()
        np.random.seed(seed)

    # Since pytorch doesn't have a complex data type we need to add an extra
    # dimension of size 2 to each tensor that will store real and imaginary
    # components.
    T = ensure_torch_array(transmission_functions, device=device)
    P = ensure_torch_array(propagators, dtype=T.dtype, device=device)
    psi = ensure_torch_array(probes, dtype=T.dtype, device=device)
    if Veff is not None:
        V = ensure_torch_array(Veff, dtype=T.dtype, device=device)
    # sys.exit()

    nT, nsubslices, nopiy, nopix = T.shape[:4]

    # Probe needs to start multislice algorithm in real space
    if qspace_in:
        psi = torch.fft.ifftn(psi, dim=d_)

    slices = generate_slice_indices(nslices, nsubslices, subslicing)
    
    def conjugateornot(array, conjugate):
        if conjugate:
            return torch.conj(array)
        else:
            return array

    for i, islice in enumerate(slices):

        # If an array-like object is passed then this will be used to uniquely
        # and reproducibly seed each iteration of the multislice algorithm
        if seed_provided:
            r = np.random.RandomState(seed[islice])

        subslice = islice % nsubslices
        
        # Calculate contribution to cross-sections (done before multislice iteration
        # while psi to be in real space; consistent with transmit before propagate;
        # whether using frozen phonon or absortpive, assume Veff includes Debye-Waller
        #  factors, so no need for shifts)
        if Veff is not None:
            modpsisq = psi * torch.conj(psi)
            for isignal in range(nsignals):
                value[0:batch_size] = torch.sum(torch.fft.fftn(modpsisq, dim=d_) * Veff[isignal,subslice,:,:], dim=d_)
                crosssec[isignal,0:batch_size] += value.cpu().numpy().real


        # Pick random phase grating
        it = r.randint(0, nT)

        # To save memory in the case of equally sliced sample, there is the option
        # of only using one propagator, this statement catches this case.
        if P.dim() < 3:
            P_ = P
        else:
            P_ = P[subslice]

        # If the transmission function is from a tiled unit cell then
        # there is the option of randomly shifting it around to
        # generate "more" psuedo-random transmission functions
        def shift(array, tiling, dim):
            if tiling == 1:
                # No shift
                return array

            pix = array.shape[dim]
            s = r.randint(0, tiling) * (pix / tiling)
            if pix % tiling == 0:
                # Shift by an integer number of pixels
                return torch.roll(array, int(s), dim)
            else:
                # Sub-integer pixel shift
                v = [1, 1]
                v[dim] = pix
                FFT_shift_array = fourier_shift_array_1d(
                    pix, s, dtype=torch.float, device=array.device
                ).view(v)
                return torch.fft.ifft(
                    FFT_shift_array * torch.fft.fft(array, dim=dim), dim=dim
                )

        T_ = shift(shift(T[it, subslice], tiling[0], 0), tiling[1], 1)
        # Perform multislice iteration
        if transpose or reverse:
            # Reverse multislice complex conjugates the transmission and
            # propagation. Both reverse and transpose multislice reverse
            # the order of the transmission and conjugation operations
            # probe should start in real space and finish this iteration in
            # real space
            psi = (
                torch.fft.ifftn(
                    torch.fft.fftn(psi, dim=d_) * conjugateornot(P_, reverse),
                    dim=d_,
                )
                * conjugateornot(T_, reverse)
            )

        else:
            # Standard multislice iteration - probe should start in real space
            # and finish this iteration in reciprocal space
            psi = torch.fft.fftn(psi * T_, dim=d_) * P_

        # The probe can be cropped to the bandwidth limit, this removes
        # superfluous array entries in reciprocal space that are zero
        # Since the next inverse FFT will apply a factor equal to the
        # square root number of pixels we have to adjust the values
        # of the array to compensate
        if i == len(slices) - 1:
            lim = 2 / 3 if output_to_bandwidth_limit else 1
            psi = crop_to_bandwidth_limit_torch(
                psi,
                qspace_in=not (transpose or reverse),
                qspace_out=qspace_out,
                limit=lim,
                norm="conserve_norm",
            )
        elif not (transpose or reverse):
            # Inverse Fourier transform back to real space for next iteration
            psi = torch.fft.ifftn(psi, dim=d_)

    if len(slices) < 1 and qspace_out:
        psi = torch.fft.fftn(psi, dim=d_)

    if Veff is not None:
        if return_numpy:
            return (psi.cpu().numpy(),crosssec)
        return (psi,crosssec)
    else:
        if return_numpy:
            return psi.cpu().numpy()
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
    PCTF : (Y,X) np.ndarray
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
    betamin : float, optional
        Detector inner acceptance semi-angle
    units : float, optional
        Units of betamin and betamax, mrad or invA are both acceptable
    Returns
    -------
    D : (ndet,Y,X) array_like
        The detector functions
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
        return np.ceil(rsize / step_size).astype(int)


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
    Returns
    -------
    probe_posns : (nY,nX,2) np.ndarray
        The probe positions in fractions of the array if gridshape is [1,1] and
        in pixel units if gridshape is the size of the pixel array.
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
        )[: nscan[i]]
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
                int
            )

            # Define resampling function to crop and interpolate
            # diffraction patterns
            def resize(array):
                cropped = crop(np.fft.fftshift(array, axes=(-1, -2)), diff_pat_crop)
                return fourier_interpolate(cropped, gridout, norm="conserve_L1")

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
            return crop(np.fft.fftshift(array, axes=(-1, -2)), gridout)

    return gridout, resize, Ksize


def second_moment(array):
    """Calculate the second moment of 2D array as a fraction of array size."""
    grids = [np.fft.fftfreq(x) for x in array.shape]
    mass = np.sum(array)
    first_moment = [
        np.sum(x) / mass for x in [grids[0][:, None] * array, grids[1][None, :] * array]
    ]

    y2 = ((grids[0] - first_moment[0] + 0.5) % 1.0 - 0.5) ** 2
    x2 = ((grids[1] - first_moment[1] + 0.5) % 1.0 - 0.5) ** 2
    grid = y2[:, None] + x2[None, :]

    return np.sqrt(np.sum(grid * array) / mass)


def generate_probe_spread_plot(
    gridshape,
    structure,
    eV,
    app,
    thickness,
    subslices=[1],
    tiling=[1, 1],
    showcrossection=True,
    df=0,
    probe_posn=[0, 0],
    show=True,
    device=None,
    P=None,
    T=None,
    nslices=None,
    nT = 1,
    showProgress = True,
):
    """
    Generate probe spread plot to assist with selection of appropriate multislice grid.

    A multislice calculation assumes periodic boundary conditions. To avoid
    artefacts associated with this the pixel grid must be chosen to have
    sufficient size so that the probe does not artificially interfere with
    itself through the periodic boundary (wrap around error). The grid sampling
    must also be sufficient that electrons scattered to high angles are not
    scattered beyond the band-width limit of the array.

    The probe spread plot helps identify whenever these two events are happening.
    If the probe intensity drops below 0.95 (as a fraction of initial intensity)
    then the grid is not sampled finely enough, the pixel size of the array
    (gridshape) needs to increased for finer sampling of the specimen potential.
    If the probe spread exceeds 0.2 (as a fraction of the array) then too much
    of the probe is spreading to the edges of the array, the real space size
    of the array (usually controlled by the tiling of the unit cell) needs to
    be increased.

    Parameters
    ----------
    gridshape : (2,) array_like
        Pixel dimensions of the 2D grid
    structure : pyms.structure_routines.structure
        The structure of interest
    eV : float
        Probe energy in electron volts
    app : float
        Probe-forming aperture in mrad
    thickness : float
        The maximum thickness of the simulation object in Angstrom
    subslices : array_like, optional
        A one dimensional array-like object containing the depths (in fractional
        coordinates) at which the object will be subsliced. The last entry
        should always be 1.0. For example, to slice the object into four equal
        sized slices pass [0.25,0.5,0.75,1.0]
    tiling : (2,) array_like, optional
        Tiling of a repeat unit cell on simulation grid
    showcrossection : bool
        Pass True to plot the projected cross section of the probe to inspect
        the spread.
    df : float
        Probe defocus in Angstrom
    probe_posn : array_like, optional
        Probe position as a fraction of the unit-cell
    P : (n,Y,X) array_like, optional
        Precomputed Fresnel free-space propagators
    T : (n,Y,X) array_like
        Precomputed transmission functions
    nT : boolean, optional
        Specify frozen phonon (True, 1) or absorptive potential (False, 0)
    showProgress : str or bool, optional
        Pass False to disable progress readout, pass 'notebook' to get correct
        progress bar behaviour inside a jupyter notebook
    Returns
    -------
    fig : matplotlib.figure object
        The figure on which the probe spread is plotted
    """
    tdisable, tqdm = tqdm_handler(showProgress)

    # Calculate multislice propagator and transmission functions
    from .Premixed_routines import multislice_precursor

    if P is None or T is None:
        P, T = multislice_precursor(
            structure,
            gridshape,
            eV,
            subslices=subslices,
            tiling=tiling,
            device=device,
            nT=nT,
            showProgress=showProgress,
        )
        

    # Calculate focused STEM probe
    probe = focused_probe(
        gridshape, structure.unitcell[:2] * np.asarray(tiling), eV, app, df=df
    )
    pos = np.asarray(probe_posn) / np.asarray(tiling)
    from .utils import fourier_shift, fourier_interpolate

    probe = fourier_shift(probe, pos, pixel_units=False)

    ncols = 1 + showcrossection
    fig, ax = plt.subplots(ncols=ncols, figsize=(ncols * 4, 4), squeeze=False)
    # Total number of slices (not including subslicing of structure)
    if nslices is None:
        nslices = int(np.ceil(thickness / structure.unitcell[2]))
    # Total number of slices (including subslicing of structure)
    maxslices = nslices * len(subslices)

    variances = np.zeros(maxslices)
    intensity = np.zeros(maxslices)

    crossection = np.zeros((maxslices, gridshape[0]))

    # Array must be shifted to center probe position
    shift = (pos * np.asarray(gridshape)).astype(int)

    for i in tqdm(range(maxslices), disable = tdisable, desc = "Running multislice"):
        probe = multislice(
            probe,
            [i % len(subslices)],
            P,
            T,
            tiling=tiling,
            subslicing=True,
            output_to_bandwidth_limit=False,
            device_type=device,
        )
        
        mod = np.roll(np.abs(probe) ** 2, shift=-shift, axis=(-2, -1))
        # Record probe intensity and spread
        intensity[i] = np.sum(mod)
        variances[i] = second_moment(mod)
        if showcrossection:
            crossection[i] = np.sum(mod, axis=-1)

    thicknesses = structure.unitcell[2] * (
        np.broadcast_to(np.arange(nslices)[:, None], (nslices, len(subslices))).ravel()
        + np.tile(subslices, nslices)
    )
    ax[0, 0].set_xlim([0, thicknesses[-1]])
    ax[0, 0].set_ylim([0, 1.1])

    ax[0, 0].set_ylabel(
        "$\\sqrt{\\int \\Psi^2 dx}$", color="red"
    )  # we already handled the x-label with ax1
    ax[0, 0].set_xlabel(r"Depth of propagation ($\AA$)")
    ax[0, 0].tick_params(axis="y", labelcolor="red")
    ax[0, 0].set_title("Probe intensity and spread")

    ax2 = ax[0, 0].twinx()  # instantiate a second axes that shares the same x-axis

    ax2.plot(thicknesses, variances, "b-")
    ax2.tick_params(axis="y", labelcolor="b")
    ax2.plot([0, thickness], [0.2, 0.2], "b--")
    ax2.set_ylim([0, 0.5])
    ax2.set_ylabel("$\\sqrt{\\int \\Psi^2 x^2 dx}$", color="blue")
    ax[0, 0].plot(thicknesses, intensity, "r-")
    ax[0, 0].plot([0, thicknesses[-1]], [0.95, 0.95], "r--")
    nz, ny = crossection.shape
    if showcrossection:
        ax[0, 1].imshow(
            np.fft.fftshift(np.sqrt(crossection), axes=1),
            extent=[0, gridshape[0], thickness, 0],
            cmap=plt.get_cmap("gnuplot"),
        )
        ax[0, 1].set_ylabel(r"Depth of propagation ($\AA$)")
        ax[0, 1].set_title("Probe depth cross-section")
    fig.tight_layout()
    if show:
        plt.show(block=True)
    return fig


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
    PACBED=False,
    scan_posn=None,
    dtype=torch.float32,
    device=None,
    tiling=[1, 1],
    seed=None,
    showProgress=True,
    method_args=(),
    method_kwargs={},
    STEM_image=None,
    Veff=None,
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
    batch_size : int, optional
        The multislice algorithm can be performed on multiple probes columns
        at once to parrallelize computation, the number of parrallel computations
        is set by batch_size.
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
    PACBED : bool or array_like, optional
        If True the STEM function will calculate a position averaged convergent
        electron diffraction (PABCED) pattern by averaging the diffraction space.
        Passing an array will specify the size to crop the PACBED pattern to.
    scan_posn :  (...,2) array_like, optional
        Array containing the STEM scan positions in fractional coordinates.
        If provided scan_posn.shape[:-1] will give the shape of the STEM image.
        result over all scan positions
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit floating point
    device : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    tiling : (2,) array_like, optional
        Tiling of a repeat unit cell on simulation grid, STEM raster will only
        scan a single unit cell.
    seed : array_like or int, optional
        Seed for the random number generator for frozen phonon configurations
    showProgress : str or bool, optional
        Pass False to disable progress readout, pass 'notebook' to get correct
        progress bar behaviour inside a jupyter notebook
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
    Veff: (nelements,nsubslices,Y,X) complex torch.tensor, optional
        Effective scattering potential (μ's) for EELS and/or EDX

    Returns
    -------
    Result : dict
        A dictionary with the keys "STEM images", "datacube" and "PACBED" which
        contain the conventional STEM images, the 4D-STEM datacube and the PACBED
        pattern respectively. If any of these simulations where not performed
        then the relevant entry will just contain None
    """
    from .utils.torch_utils import detect

    tdisable, tqdm = tqdm_handler(showProgress)
    rdtype = complex_to_real_dtype_torch(dtype)

    # Get number of thicknesses in the series
    nthick = len(nslices)

    if isinstance(nslices[0], (int, np.integer)):
        nslices_ = np.diff(nslices, prepend=0)
    else:
        nslices_ = nslices

    if device is None:
        device = get_device(device)

    # Get shape of grid
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
    scan_posn = torch.as_tensor(scan_posn).to(device).type(rdtype)

    # Assume real space probe is passed in so perform Fourier transform in
    # anticipation of application of Fourier shift theorem
    probe_ = ensure_torch_array(probe, device=device)
    probe_ = torch.fft.fftn(probe_, dim=[-2, -1])

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
            datacube = np.zeros((nthick, *scan_shape, *gridout))

    if PACBED:
        # Get PACBED pattern gridsize in pixels from input and function
        # to resample the simulation output to store in the datacube
        gridout2, resize2, _ = workout_4DSTEM_datacube_DP_size(
            PACBED, rsize, gridshape
        )
        PACBED_pattern = torch.zeros((nthick, *gridshape), device=device)
    else:
        PACBED_pattern = None

    if Veff is not None:
        nsignals = Veff.shape[0]
        crosssec_images = np.zeros((nsignals,nthick,nscantot))
    else:
        crosssec_images = None

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
        disable=tdisable,
        desc="Probe positions",
    ):

        # Make shifted probes
        scan_index = np.arange(
            i * batch_size, min((i + 1) * batch_size, nscantot), dtype=int
        )

        # The shift operator array array will be of size batch_size x Y x X
        probes = fourier_shift_array(
            gridshape,
            scan_posn[scan_index],
            dtype=dtype,
            device=device,
            units="fractional",
        )

        # Apply shift to original probe
        probes = probe_.view(1, *probe_.size()) * probes

        if Veff is not None:
            crosssec = np.zeros((nsignals,len(scan_index)))

        # Thickness series
        for it, t in enumerate(nslices_):
            if Veff is not None:
                probes = (probes,crosssec)

            # Evaluate exit surface wave function from input probes
            probes = method(
                probes, t, *method_args, posn=scan_posn[scan_index], **method_kwargs, Veff=Veff
            )

            if Veff is not None:
                crosssec=probes[1]
                probes=probes[0]
                crosssec_images[:,it,scan_index]=crosssec

            # Calculate amplitude of probes, a real output is assumed to be the
            # amplitude of the exit surface wave function. Also correct
            # normalization to be in units of fractional intensity
            cmplxout = iscomplex(probes)
            if cmplxout:
                amp = amplitude(probes) / np.prod(probes.size()[-2:])
            else:
                amp = probes / np.prod(probes.size()[-2:])

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

            if PACBED:
                PACBED_pattern[it] += torch.sum(amp, axis=0) / nscantot

            # In some cases the amplitude will be returned by the function
            # in this case multiple thickness values should not be used! This
            # break command helps prevent misuse.
            if not cmplxout:
                break

    if Veff is not None:
        STEM_crosssection_images = crosssec_images.reshape(nsignals,nthick,*scan_shape)
    else:
        STEM_crosssection_images = None

    if conventional_STEM:
        STEM_image = np.squeeze(STEM_image.reshape(ndet, nthick, *scan_shape))
    if PACBED:
        PACBED_pattern = resize2(PACBED_pattern.cpu().numpy())
    return {"STEM images": STEM_image, "datacube": datacube, "PACBED": PACBED_pattern, "STEM crosssection images": STEM_crosssection_images}


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
        """
        Initialize with a set of propagators and transmission functions.

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
        showProgress : str or bool, optional
            Pass False to disable progress readout, pass 'notebook' to get correct
            progress bar behaviour inside a jupyter notebook
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
        gridshape = transmission_functions.shape[-2:]

        # Datatype (precision) is inferred from transmission functions
        self.dtype = transmission_functions.dtype
        self.rdtype = complex_to_real_dtype_torch(self.dtype)

        # Device (CPU or GPU) is also inferred from transmission functions
        self.device = device
        if GPU_streaming:
            self.device = torch.device("cuda")
        elif self.device is None:
            self.device = transmission_functions.device

        # Get alpha in units of inverse Angstrom
        self.alpha_ = wavev(eV) * alpha * 1e-3

        self.PRISM_factor = PRISM_factor
        self.doPRISM = np.any(np.asarray(PRISM_factor) > 1)

        # Make a list of beams in the scattering matrix
        # Take beams inside the aperture and every nth beam where n is the
        # PRISM "interpolation" factor
        q = q_space_array(gridshape, rsize)
        inside_aperture = np.less_equal(q[0] ** 2 + q[1] ** 2, self.alpha_ ** 2)
        mody, modx = [
            np.mod(np.fft.fftfreq(x, 1 / x).astype(int), p) == 0
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
                transmission_functions.shape[-2:]
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
        self.nsubslices = transmission_functions.shape[1]
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
        batch_size : int, optional
            The multislice algorithm can be performed on multiple scattering
            matrix columns at once to parrallelize computation, this number is
            set by batch_size.
        subslicing : bool, optional
            Pass subslicing=True to access propagation to sub-slices of the
            unit cell, in this case nslices is taken to be in units of subslices
            to propagate rather than unit cells (i.e. nslices = 3 will propagate
            1.5 unit cells for a subslicing of 2 subslices per unit cell)
        showProgress : str or bool, optional
            Pass False to disable progress readout, pass 'notebook' to get correct
            progress bar behaviour inside a jupyter notebook
        transpose : bool, optional
            Make a "transposed" scattering matrix - see Brown et al. (2019)
            Physical Review Research paper for a discussion of this and its
            applications
        """
        tdisable, tqdm = tqdm_handler(showProgress)
        from .Probe import plane_wave_illumination

        # Initialize scattering matrix if necessary
        if not self.initialized:

            if self.Fourier_space_output:
                self.S = torch.zeros(
                    self.nbeams, self.nbout, dtype=self.dtype, device=self.device
                )
            else:
                self.S = torch.zeros(
                    self.nbeams,
                    *self.stored_gridshape,
                    dtype=self.dtype,
                    device=self.device
                )
            for ibeam in range(self.nbeams):
                # Initialize S-matrix to plane-waves
                psi = torch.from_numpy(
                    plane_wave_illumination(
                        self.gridshape,
                        self.rsize[:2],
                        self.eV,
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
                    self.S[ibeam] = psi[self.bw_mapping[:, 0], self.bw_mapping[:, 1]]
                else:
                    self.S[ibeam] = fourier_interpolate_torch(
                        psi,
                        self.stored_gridshape,
                        qspace_in=True,
                        qspace_out=False,
                        norm="conserve_norm",
                    )
            self.initialized = True

        # Make nslice_ which always accounts of subslices of the structure
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

        # If streaming of Smatrix columns to the GPU is being used, ensure
        # that propagators and transmission functions for the multislice are
        # already on the GPU
        if self.GPU_streaming:
            propagators = ensure_torch_array(propagators, dtype=self.dtype).cuda()
            transmission_functions = ensure_torch_array(
                transmission_functions, dtype=self.dtype
            ).cuda()

        self.current_slice = nslice_
        if len(slices) < 1:
            return

        # Loop over the different plane wave components (or columns) of the
        # scattering matrix
        for i in tqdm(
            range(int(np.ceil(self.nbeams / batch_size))),
            disable=tdisable,
            desc="Calculating S-matrix",
        ):

            beams = np.arange(
                i * batch_size, min((i + 1) * batch_size, self.nbeams), dtype=int
            )
            if self.Fourier_space_output:
                # Initialize array that will be used as input to the multislice routine
                psi = torch.zeros(
                    batch_size, *self.gridshape, dtype=self.dtype, device=self.device
                )
                # Expand S-matrix input to full grid for multislice propagation
                psi[
                    : beams.shape[0], self.bw_mapping[:, 0], self.bw_mapping[:, 1]
                ] = self.S[beams]
            else:
                # Fourier interpolate stored real space S-matrix column onto
                # multislice grid
                psi = fourier_interpolate_torch(
                    self.S[beams], self.gridshape, norm="conserve_norm"
                )

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
                    :, self.bw_mapping[:, 0], self.bw_mapping[:, 1]
                ] * np.sqrt(np.prod(self.stored_gridshape) / np.prod(self.gridshape))
            else:
                output = fourier_interpolate_torch(
                    output, self.stored_gridshape, norm="conserve_norm"
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
        Sshape = [int(x) for x in Smat.shape]

        device = Smat.device
        crop_ = self.PRISM_crop_window(device=device)
        # Ensure posn and probes are pytorch arrays
        probes = ensure_torch_array(probes, dtype=self.dtype, device=device)

        # Ensure probes tensors correspond to the shape N x Y x X
        # If they have the shape Y x X then reshape to 1 x Y x X
        if probes.ndim < 3:
            probes = probes.view(1, *probes.shape)

        # Get number of probes
        nprobes = probes.shape[0]

        if posn is None:
            posn = torch.zeros(nprobes, 2, device=device, dtype=float)
        else:
            if isinstance(posn, list):
                posn = np.array(posn)
        posn = torch.as_tensor(posn, device=device, dtype=float).view((nprobes, 2))

        if scan_transform is not None:
            posn = scan_transform(posn)

        # TODO decide whether to remove the Fourier_space_output option
        if self.Fourier_space_output:

            # A note on normalization: an individual probe enters the STEM routine
            # with sum_squared intensity of 1, but the STEM routine applies an
            # FFT so the sum_squared intensity is now equal to # pixels
            # For a correct matrix multiplication we must now divide by sqrt(# pixels)
            probe_vec = torch.matmul(
                probes[:, self.beams[0], self.beams[1]], Smat
            ) / np.sqrt(np.prod(self.gridshape))

            # Now reshape output from vectors to square arrays
            probes = torch.zeros(
                nprobes, *self.stored_gridshape, dtype=self.dtype, device=self.device
            )
            probes[:, self.bw_mapping[:, 0], self.bw_mapping[:, 1]] = probe_vec

            # Apply PRISM cropping in real space if appropriate
            if self.doPRISM:
                shape = probes.size()

                probes = torch.fft.ifftn(probes, dim=[-2, -1]).flatten(-3, -2)
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
                return torch.fft.fftn(probes, dim=[-2, -1])

            return probes
        else:
            # Flatten the array dimensions
            Smatshape = Smat.shape
            flattened_shape = [Smatshape[0], Smatshape[-2] * Smatshape[-1]]
            N = probes.shape[0]
            output = torch.zeros(
                N, *Smat.shape[-2:], dtype=self.dtype, device=Smat.device
            )

            # For evaluating the probes in real space we only want to perform the matrix
            # multiplication and summation within the real space PRISM cropping region
            # Stride is pixel size of real space PRISM cropping region
            stride = [x // y for x, y in zip(self.stored_gridshape, self.PRISM_factor)]
            halfstride = [x // 2 for x in stride]

            for probe, pos, out in zip(probes, posn, output):

                if self.doPRISM:

                    start = [
                        int(torch.round(pos[i] * Sshape[-2 + i])) - halfstride[i]
                        for i in range(2)
                    ]
                    windows = crop_window_to_periodic_indices(
                        [start[0], stride[0], start[1], stride[1]], Sshape[-2:]
                    )

                    for wind in windows:
                        # Narrow returns a cropped window the the input tensor
                        outview, sview = [
                            x.narrow(-2, wind[0][0], wind[0][1]).narrow(
                                -1, wind[1][0], wind[1][1]
                            )
                            for x in [out, Smat]
                        ]

                        p = probe[self.beams[0], self.beams[1]].view(self.nbeams, 1, 1)
                        outview += torch.sum(p * sview, axis=0)
                else:
                    output += torch.matmul(
                        probe[self.beams[0], self.beams[1]], Smat.view(flattened_shape)
                    ).view(Smatshape[1:])

            output /= np.sqrt(np.prod(probes.size()[-2:]))
            output = crop_torch(
                output.reshape(probes.size(0), *(Smat.size()[-2:])),
                self.stored_gridshape,
            )

            return torch.fft.fftn(output, dim=[-2, -1])

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

        Parameters
        ----------
        self : scattering_matrix
            The scattering matrix object.
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
        showProgress : str or bool, optional
            Pass False to disable progress readout, pass 'notebook' to get correct
            progress bar behaviour inside a jupyter notebook
        """
        device = get_device(device)
        tdisable, tqdm = tqdm_handler(showProgress)

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
        probe = torch.from_numpy(probe).to(device).type(self.dtype)

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
                STEM_image = STEM_image.reshape((ndet, nscan))
        else:
            STEM_image = None

        if nstreams is None:
            # If the number of seperate streams is not suggested by the
            # user, make this equal to the product of the PRISM factor
            nstreams = int(np.product(self.PRISM_factor))

        # Divide up the scan positions into clusters based on Euclidean
        # distance
        from sklearn.cluster import Birch

        if nscan > 1:
            model = Birch(threshold=0.01, n_clusters=nstreams)
            yhat = model.fit_predict(scan_posns)
            clusters = np.unique(yhat)
        else:
            yhat, clusters = [[0], [0]]

        # Now do STEM with each of the scan position clusters, streaming
        # only the necessary bits of the scattering matrix to the GPU.
        Datacube_segment = None
        STEM_image_segment = None
        FlatS = self.S.reshape((self.nbeams, np.prod(self.stored_gridshape)))

        # Loop over probe positions clusters. This would be a good candidate
        # for multi-GPU work.
        for cluster in tqdm(clusters, desc="Probe position clusters", disable=tdisable):
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
            SegmentS = FlatS[:, indices].reshape((self.nbeams, *segmentshape))

            # Define a function that will map probe positions for the global
            # scattering matrix to their correct place on the smaller scattering
            # matrix streamed to the GPU.
            gshape = torch.as_tensor(self.stored_gridshape).to(device).type(torch.int)
            Origin = torch.as_tensor([ymin, xmin]).to(device).type(self.rdtype)
            segment_size = torch.as_tensor(size).to(device).type(self.rdtype)

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


def phase_from_com(com, reg=1e-10, rsize=[1, 1]):
    """
    Integrate 4D-STEM centre of mass (DPC) measurements to calculate object phase.

    Assumes a three dimensional array com, with the final two dimensions
    corresponding to the image and the first dimension of the array corresponding
    to the y and x centre of mass respectively.
    """
    # Get shape of arrays
    ny, nx = com.shape[-2:]
    s = (ny, nx) # Shape of the real output to the inverse FFT.
    # s = None

    d = np.asarray(rsize) / np.asarray([ny, nx])
    # Calculate Fourier coordinates for array
    ky = np.fft.fftfreq(ny, d=d[0])
    kx = np.fft.rfftfreq(nx, d=d[1])

    # Calculate numerator and denominator expressions for solution of
    # phase from centre of mass measurements
    numerator = ky[:, None] * np.fft.rfft2(com[0]) + kx[None, :] * np.fft.rfft2(com[1])
    denominator = 1j * ((kx ** 2)[None, :] + (ky ** 2)[:, None]) + reg

    # Avoid a divide by zero for the origin of the Fourier coordinates
    numerator[0, 0] = 0
    denominator[0, 0] = 1

    # Return real part of the inverse Fourier transform
    return np.fft.irfft2(numerator / denominator, s=s)
