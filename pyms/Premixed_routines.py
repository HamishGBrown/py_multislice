"""
Premixed routines for simulation of some standard TEM techniques.

The py_multislice library provides users with the ability to put their
own simulations together using the "building blocks" of py_multislice.
For standard simulation types the premixed_routines.py functions
allows users to set up simulations faster and easier.
"""

import numpy as np
import matplotlib.pyplot as plt  # Added for debugging purposes
import torch
from .py_multislice import (
    make_propagators,
    tqdm_handler,
    scattering_matrix,
    generate_STEM_raster,
    make_detector,
    multislice,
    STEM,
    workout_4DSTEM_datacube_DP_size,
    nyquist_sampling,
    phase_from_com,
    thickness_to_slices,
)
from .utils.torch_utils import (
    amplitude,
    get_device,
    crop_to_bandwidth_limit_torch,
    size_of_bandwidth_limited_array,
    torch_dtype_to_numpy,
    real_to_complex_dtype_torch,
    complex_to_real_dtype_torch,
    ensure_torch_array,
    fourier_shift_array,
)
from .utils.numpy_utils import (
    fourier_shift,
    crop_to_bandwidth_limit,
    ensure_array,
    q_space_array,
    crop,
    single_or_max,
)
from .utils.output import initialize_h5_datacube_object
from .Ionization import (
    get_transitions,
    tile_out_ionization_image,
    transition_potential_multislice,
)
from .Probe import (
    focused_probe,
    make_contrast_transfer_function,
    plane_wave_illumination,
    convert_tilt_angles,
)


def window_indices(center, windowsize, gridshape):
    """Generate array indices for a sub-region cropped out of a larger grid."""
    window = []
    for i, wind in enumerate(windowsize):
        indices = np.arange(-wind // 2, wind // 2, dtype=int) + wind % 2
        indices += int(np.floor(center[i] * gridshape[i]))
        indices = np.mod(indices, gridshape[i])
        window.append(indices)

    return (window[0][:, None] * gridshape[0] + window[1][None, :]).ravel()


# Pixel sizes that are amenable to FFTs from https://blake.bcm.edu/emanwiki/EMAN2/BoxSize
optimalpixels = [
    64,
    72,
    84,
    96,
    100,
    104,
    112,
    120,
    128,
    132,
    140,
    168,
    180,
    192,
    196,
    208,
    216,
    220,
    224,
    240,
    256,
    260,
    288,
    300,
    320,
    352,
    360,
    384,
    416,
    440,
    448,
    480,
    512,
    540,
    560,
    576,
    588,
    600,
    630,
    640,
    648,
    672,
    686,
    700,
    720,
    750,
    756,
    768,
    784,
    800,
    810,
    840,
    864,
    882,
    896,
    900,
    960,
    972,
    980,
    1000,
    1008,
    1024,
    1050,
    1080,
    1120,
    1134,
    1152,
    1176,
    1200,
    1250,
    1260,
    1280,
    1296,
    1344,
    1350,
    1372,
    1400,
    1440,
    1458,
    1470,
    1500,
    1512,
    1536,
    1568,
    1600,
    1620,
    1680,
    1728,
    1750,
    1764,
    1792,
    1800,
    1890,
    1920,
    1944,
    1960,
    2000,
    2016,
    2048,
    2058,
    2100,
    2160,
    2240,
    2250,
    2268,
    2304,
    2352,
    2400,
    2430,
    2450,
    2500,
    2520,
    2560,
    2592,
    2646,
    2688,
    2700,
    2744,
    2800,
    2880,
    2916,
    2940,
    3000,
    3024,
    3072,
    3136,
    3150,
    3200,
    3240,
    3360,
    3402,
    3430,
    3456,
    3500,
    3528,
    3584,
    3600,
    3750,
    3780,
    3840,
    3888,
    3920,
    4000,
    4032,
    4050,
    4096,
]


def autosampling_STEM(
    structure, thickness, app, eV, qmax=None, precision=1, generate_plot=False
):
    """
    Automatically calculate grid pixels and tiling for multislice simulations in Scanning Transmission Electron Microscopy (STEM).

    This function determines the optimal grid sampling and tiling for multislice STEM simulations by considering the probe spread (variance)
    and total intensity criteria. It allows for an optional plot generation to visualize the convergence of intensity and probe spread metrics.

    Parameters:
    ----------
    structure : object
        The structure object containing atomic positions and unit cell dimensions.
    thickness : float
        The thickness of the sample in Angstroms.
    app : float
        The probe aperture in milliradians.
    eV : float
        The electron energy in electronvolts (eV).
    qmax : float, optional
        The maximum scattering angle in mrad. Default is None. When provided, it overrides the default kmax calculation.
    precision : int, optional
        The precision level for grid sampling. Default is 1. Values can be 0 (rough), 1 (medium), or 2 (precise).
    generate_plot : bool, optional
        If True, generates a plot showing the convergence of intensity and probe spread metrics. Default is False.

    Returns:
    -------
    gridshape : list
        The optimal grid shape in pixels for the multislice simulation.
    tiling : list
        The optimal tiling for the multislice simulation.

    Notes:
    -----
    - The function uses the Jenks Natural Breaks clustering algorithm to determine subslices of the structure.
    - `generate_probe_spread_plot` is a utility function used to calculate probe intensity and variance metrics.
    - If the variance and intensity criteria are not met, the function adjusts the kmax value and retries the calculations.
    - When `generate_plot` is True, the function plots the convergence of intensity and probe spread metrics.

    Example:
    -------
    >>> gridshape, tiling = autosampling_STEM(structure, thickness=50, app=30, eV=200, qmax=0.2, precision=1, generate_plot=True)
    """

    # Calculate grid sampling in inverse Angstroms
    # This is equivalent to the grid sampling in pix/Angstrom
    kmax = 7.5 + precision * 2.5

    # Option for manual override of qmax, eg. when a high angle
    # annular dark field (HAADF) detector is used.
    if qmax is not None:
        from .Probe import wavev

        kmax = max(kmax, wavev(eV) * np.atan(qmax))

    # Adjust kmax to account for bandwidth limiting
    kmax *= 1.5

    # Probe spread (variance) and total intensity criteria
    variancecriterion = 0.2 - 0.05 * precision
    intensitycriterion = [0.95, 0.97, 0.99][precision]

    # Ratio of the sides of the unit cell
    aspectratio = structure.unitcell[0] / structure.unitcell[1]

    from .py_multislice import generate_probe_spread_plot

    subslices = structure.autoslice()
    variances = []
    intensities = []
    pixels = []
    tilings = []
    for i in range(1, 10):
        tiling = np.asarray([i, max(1, int(np.round(i * aspectratio)))])
        gridshape = [
            optimalpixels[(np.abs(optimalpixels - t * u * kmax)).argmin()]
            for t, u in zip(tiling, structure.unitcell[:2])
        ]

        inten, var = generate_probe_spread_plot(
            gridshape,
            structure,
            eV,
            app,
            single_or_max(thickness),
            subslices=subslices,
            tiling=tiling,
            just_metrics=True,
            showProgress=False,
        )

        intensities.append(min(inten))
        variances.append(max(var))
        pixels.append(gridshape[0])
        tilings.append(tiling)

        if max(var) < variancecriterion and min(inten) > intensitycriterion:
            break
        elif min(inten) < intensitycriterion:
            kmax += 1

    if generate_plot:
        # Plot convergence of intensity and probe spread (variance) metrics
        fig, ax = plt.subplots()

        ax.set_xlim([0, 1.2 * pixels[-1]])
        ax.set_ylim([0, 1.1])

        ax.set_ylabel(
            "$\\sqrt{\\int \\Psi^2 dx}$", color="red"
        )  # we already handled the x-label with ax1
        ax.set_xlabel(r"Pixels")
        ax.tick_params(axis="y", labelcolor="red")
        ax.set_title("Probe intensity and spread")

        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.plot(pixels, variances, "bo")
        ax2.tick_params(axis="y", labelcolor="b")
        for criterion, label in zip([0.1, 0.15, 0.2], ["precise", "medium", "rough"]):
            ax2.plot([0, pixels[-1] * 1.2], [criterion] * 2, "b--")
            ax2.text(np.mean(ax.get_xlim()), criterion, label)
        ax2.set_ylim([0, 0.5])
        ax2.set_ylabel("$\\sqrt{\\int \\Psi^2 x^2 dx}$", color="blue")
        ax.plot(pixels, intensities, "ro")
        for p, v, t in zip(pixels, variances, tilings):
            ax2.annotate("[{0},{1}]".format(*t), [p, v])
        ax.plot([0, pixels[-1] * 1.2], [0.95, 0.95], "r--")
        plt.show(block=True)

    return gridshape, tiling


def autosampling_TEM(structure, precision=1):

    """
    Automatically calculate grid pixels and tiling for Transmission Electron Microscopy (TEM) simulations.

    This function determines the optimal grid sampling and tiling for TEM simulations by considering the
    precision level. It calculates the required grid shape and tiling based on the unit cell dimensions of the structure.

    Parameters:
    ----------
    structure : object
        The structure object containing atomic positions and unit cell dimensions.
    precision : int, optional
        The precision level for grid sampling. Default is 1. Values can be 0 (rough), 1 (medium), or 2 (precise).

    Returns:
    -------
    gridshape : list of int
        The optimal grid shape in pixels for the TEM simulation.
    tiling : list of int
        The optimal tiling for the TEM simulation.

    Notes:
    -----
    - The function calculates the grid sampling in inverse Angstroms, which is equivalent to the grid sampling in pixels per Angstrom.
    - The `kmax` value is adjusted to account for bandwidth limiting.
    - The minimum inverse Angstrom per pixel diffraction space sampling is determined based on the precision level.
    - The real space extent is determined by the diffraction space sampling.
    - The grid shape is determined by the tiling, maximum resolution, and structure size.

    Example:
    -------
    >>> structure = pyms.structure.fromfile('STO.xtl')
    >>> gridshape, tiling = autosampling_TEM(structure, precision=1)
    """

    # Calculate grid sampling in inverse Angstroms
    # This is equivalent to the grid sampling in pix/Angstrom
    kmax = 7.5 + precision * 2.5
    # Adjust kmax to account for bandwidth limiting
    kmax *= 1.5

    # Ratio of the sides of the unit cell
    # aspectratio = structure.unitcell[0]/structure.unitcell[1]

    # Minimum inverse Angstrom per pixel diffraction space sampling
    mininvAperpix = [0.1, 0.05, 0.01][precision]

    # Diffraction space sampling determines real space extent
    tiling = [int(np.ceil(1 / (x * mininvAperpix))) for x in structure.unitcell[:2]]

    # Tiling, max resolution and structure size determines gridshape
    gridshape = [
        optimalpixels[(np.abs(optimalpixels - t * u * kmax)).argmin()]
        for t, u in zip(tiling, structure.unitcell[:2])
    ]

    return gridshape, tiling


def STEM_PRISM(
    structure,
    gridshape,
    eV,
    app,
    thicknesses,
    subslices=None,
    tiling=[1, 1],
    PRISM_factor=[1, 1],
    df=0,
    nfph=1,
    aberrations=[],
    batch_size=5,
    FourD_STEM=False,
    DPC=False,
    h5_filename=None,
    datacube=None,
    scan_posn=None,
    dtype=torch.float32,
    device_type=None,
    showProgress=True,
    detector_ranges=None,
    D=None,
    stored_gridshape=None,
    ROI=[0.0, 0.0, 1.0, 1.0],
    S=None,
    nT=5,
    GPU_streaming=True,
    Calculate_potential_on_CPU=False,
    P=None,
    T=None,
    displacements=True,
):
    """
    Perform a STEM simulation using the PRISM algorithm.

    The PRISM algorithm saves time by calculating the scattering matrix operator
    and reusing this to calculate each probe in the STEM raster. See Ophus,
    Colin. "A fast image simulation algorithm for scanning transmission
    electron microscopy." Advanced structural and chemical imaging 3.1 (2017):
    13 for details on this.

    Parameters
    ----------
    structure : pyms.structure_routines.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts
    app : float
        Objective aperture in mrad
    thicknesses : float or array_like
        Thickness of the calculation
    subslices : array_like, optional
        A one dimensional array-like object containing the depths (in fractional
        coordinates) at which the object will be subsliced. The last entry
        should always be 1.0. For example, to slice the object into four equal
        sized slices pass [0.25,0.5,0.75,1.0]
    tiling : (2,) array_like, optional
        Tiling of a repeat unit cell on simulation grid
    PRISM_factor : int (2,) array_like, optional
        The PRISM "interpolation factor" this is the amount by which the
        scattering matrices are cropped in real space to speed up
        calculations see Ophus, Colin. "A fast image simulation algorithm
        for scanning transmission electron microscopy." Advanced structural
        and chemical imaging 3.1 (2017): 13 for details on this.
    df : float or (ndf,) array_like
        Probe defocus, can be an array like object to perform a focal series. In
        the latter case it would be expected that h5_filename is a list of
        equal length to df
    aberrations : list, optional
        A list containing a set of the class aberration, pass an empty list for
        an unaberrated probe.
    batch_size : int, optional
        The multislice algorithm can be performed on multiple scattering matrix
        columns at once to parallelize computation, this number is set by
        batch_size.
    fourD_STEM : bool or array_like, optional
        Pass fourD_STEM = True to perform 4D-STEM simulations. To save disk
        space a tuple containing pixel size and diffraction space extent of the
        datacube can be passed in. For example ([64,64],[1.2,1.2]) will output
        diffraction patterns measuring 64 x 64 pixels and 1.2 x 1.2 inverse
        Angstroms.
    h5_filename : string or (ndf,) array_like of strings
        FourD-STEM images can be streamed directly to a hdf5 file output,
        avoiding the need for them to be stored in intermediate memory. To take
        advantage of this the user can provide a list of filenames to output
        these result to.
    datacube :  (ndf, Ny, Nx, Y, X) array_like, optional
        datacube for 4D-STEM output, if None is passed (default) this will be
        initialized in the function. If a datacube is passed then the result
        will be added to by the STEM routine (useful for multiple frozen phonon
        iterations)
    scan_posn :  (ny,nx,2) array_like, optional
        An array containing y and x scan positions in the final dimension for
        each scan position in the STEM raster, if provided overrides internal
        calculations of STEM sampling
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit (single precision)
        floating point
    device_type : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    showProgress : str or bool, optional
        Pass False to disable progress readout, pass 'notebook' to get correct
        progress bar behaviour inside a jupyter notebook
    stored_gridshape : int, array_like
        The size of the stored array in the scattering matrix.
    detector_ranges : array_like, optional
        The acceptance angles of each of the stem detectors should be stored as
        a list of lists: ie [[0,10],[10,20]] will make two detectors spanning
        0 to 10 mrad and 10 to 20 mrad.
    D : (ndet x Y x X) array_like, optional
        A precomputed set of STEM detectors, overrides detector_ranges.
    ROI : (4,) array_like
        Fraction of the unit cell to be scanned. Should contain [y0,x0,y1,x1]
        where [x0,y0] and [x1,y1] are the bottom left and top right coordinates
        of the region of interest (ROI) expressed as a fraction of the total
        grid (or unit cell).
    S : (nbeams x Y x X)
        A precalculated scattering matrix (useful for calculating defocus series)
        if this is provided then nfph will be ignored since a new scattering
        matrix should be calculated for each frozen phonon iteration
    nT : int, optional
        Number of independent multislice transmission functions generated and
        then selected from in the frozen phonon algorithm
    GPU_streaming : bool
        Choose whether to use GPU streaming or not for the scattering matrix,
        providing a scattering matrix to the routine will override whatever is
        selected for this option
    Calculate_potential_on_CPU : bool
        Calculate the potential, which can sometimes be the most memory intensive
        part of the calculation on the CPU and then stream to the GPU
    P : (n,Y,X) array_like, optional
        Precomputed Fresnel free-space propagators
    T : (n,Y,X) array_like
        Precomputed transmission functions
    Returns
    -------
    result : dict
        A dictionary containing numpy arrays with the requested simulations.
        Possible keys include "STEM images" (conventional STEM images from a
        monolithic detector), "datacube" (the 4D-STEM datacube) and "DPC" (the
        differential phase contrast reconstruction of the electrostatic potential)
    """
    tdisable, tqdm = tqdm_handler(showProgress)

    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Perform autoslicing by default
    if subslices is None:
        subslices = structure.autoslice()

    # Calculate real space grid size
    real_dim = structure.unitcell[:2] * np.asarray(tiling)

    # Check if a scattering matrix is provided
    S_provided = S is not None
    # If S is provided then nfph will be ignored since a new scattering
    # matrix should be calculated for each frozen phonon iteration
    if S_provided:
        nfph_ = 1
    else:
        nfph_ = nfph

    Fourier_space_output = False
    # Fourier_space_output = np.all([x == 1 for x in PRISM_factor])

    # Work out the shape of the scan raster
    scanshape = generate_STEM_raster(real_dim, eV, app, tiling=tiling, ROI=ROI).shape[
        :-1
    ]
    ndf = len(ensure_array(df))

    # Make STEM detectors
    maxq = 0
    STEM_images = [None for i in range(len(ensure_array(df)))]
    doconvSTEM = (detector_ranges is not None) or DPC
    if doconvSTEM:
        if detector_ranges is not None:
            if D is None:
                D = np.stack(
                    [
                        make_detector(gridshape, real_dim, eV, drange[1], drange[0])
                        for drange in detector_ranges
                    ]
                )
                # Work out maximum angular value needed in calculation
                maxq = max(
                    maxq,
                    np.amax(
                        convert_tilt_angles(
                            detector_ranges, "mrad", real_dim, eV, True
                        ),
                        axis=0,
                    )[1],
                )
            else:
                maxq = max(D.shape[-2:] / real_dim[-2:])
        # Prepare DPC detectors if necessary
        if DPC:
            DPC_det = np.stack(q_space_array(gridshape, real_dim), axis=0)
            if D is not None:
                D = np.concatenate([D, DPC_det], axis=0)
            else:
                D = DPC_det
        ndet = D.shape[0]
        STEM_images = np.zeros(
            (ndf, ndet, *scanshape), dtype=torch_dtype_to_numpy(dtype)
        )
    else:
        D = None

    # If no instructions are passed regarding size of the 4D-STEM
    # datacube size then we will have to base this on the output size
    # of the scattering matrix
    h5file = None
    if FourD_STEM and isinstance(FourD_STEM, (list, tuple)):
        if len(FourD_STEM) > 1:
            maxq = max(maxq, max(FourD_STEM[1]))
        else:
            maxq = max(maxq, max(FourD_STEM[0]) / real_dim[np.argmax(FourD_STEM[0])])
    else:
        from .py_multislice import max_grid_resolution

        maxq = max(maxq, max_grid_resolution(gridshape, real_dim))

    if FourD_STEM:
        DPshape, _, Ksize = workout_4DSTEM_datacube_DP_size(
            FourD_STEM, real_dim, gridshape
        )
        if h5_filename is None:
            if datacube is None:
                datacubes = np.zeros(
                    (ndf,) + scanshape + tuple(DPshape),
                    dtype=torch_dtype_to_numpy(dtype),
                )
            else:
                datacubes = datacube
        else:

            Rpix = nyquist_sampling(eV=eV, alpha=app)

            datacubes = []
            h5files = []
            for h5_file in ensure_array(h5_filename):
                # Make a datacube for each thickness of interest
                datacube, h5file = initialize_h5_datacube_object(
                    scanshape + tuple(DPshape),
                    h5_file,
                    dtype=torch_dtype_to_numpy(dtype),
                    Rpix=Rpix,
                    diffsize=Ksize,
                    eV=eV,
                    alpha=app,
                    sample=structure.Title,
                )
                datacubes.append(datacube)
                h5files.append(h5file)
    else:
        datacubes = [None for i in range(len(ensure_array(df)))]
    # Calculations can be expedited by only storing the scattering matrix out
    # to the maximum scattering angle of interest here we work out if the
    # calculation would benefit from this
    maxpix = np.minimum(
        (maxq * np.asarray(real_dim)).astype(int),
        size_of_bandwidth_limited_array(gridshape),
    )

    PandTnotprovided = (P is None) and (T is None)

    for i in tqdm(
        range(nfph_), desc="Frozen phonon iteration", disable=tdisable or nfph_ < 2
    ):
        # Option to provide a precomputed scattering matrix (S), however if
        # none is given then we make our own

        if not S_provided:
            # Make propagators and transmission functions for multslice
            torch.cuda.empty_cache()
            if PandTnotprovided:
                if Calculate_potential_on_CPU:
                    potdevice = torch.device("cpu")
                else:
                    potdevice = device

                P, T = multislice_precursor(
                    structure,
                    gridshape,
                    eV,
                    subslices=subslices,
                    tiling=tiling,
                    nT=nT,
                    device=potdevice,
                    showProgress=showProgress,
                    displacements=displacements,
                )
                if Calculate_potential_on_CPU:
                    T = T.to(device)

            # Convert thicknesses into number of slices for multislice
            nslices = np.ceil(thicknesses / structure.unitcell[2]).astype(int)

            S = scattering_matrix(
                real_dim[:2],
                P,
                T,
                nslices,
                eV,
                app,
                PRISM_factor=PRISM_factor,
                tiling=tiling,
                batch_size=batch_size,
                device=device,
                stored_gridshape=maxpix,
                Fourier_space_output=Fourier_space_output,
                GPU_streaming=GPU_streaming,
                showProgress=showProgress,
            )
            del P
            del T
            torch.cuda.empty_cache()
        for idf, df_ in enumerate(ensure_array(df)):
            if S.GPU_streaming:
                result = S.STEM_with_GPU_streaming(
                    detectors=D,
                    FourD_STEM=FourD_STEM,
                    datacube=datacubes[idf],
                    STEM_image=STEM_images[idf],
                    df=df_,
                    aberrations=aberrations,
                    scan_posns=scan_posn,
                    showProgress=showProgress,
                )
            else:
                # Make the STEM probe
                probe = focused_probe(
                    gridshape, real_dim[:2], eV, app, df=df_, aberrations=aberrations
                )
                result = STEM(
                    S.rsize,
                    probe,
                    S,
                    [1],
                    S.eV,
                    app,
                    batch_size=batch_size,
                    detectors=D,
                    FourD_STEM=FourD_STEM,
                    datacube=[datacubes[idf]],
                    scan_posn=scan_posn,
                    device=S.device,
                    tiling=S.tiling,
                    showProgress=showProgress,
                    STEM_image=STEM_images[idf],
                )
            if FourD_STEM:
                datacubes[idf] = result["datacube"][0][:] / nfph_
            if doconvSTEM:
                STEM_images[idf] = result["STEM images"] / nfph_
    result = {"STEM images": np.squeeze(STEM_images), "datacube": np.squeeze(datacubes)}
    # Perform DPC reconstructions (requires py4DSTEM)
    if DPC:
        result["DPC"] = phase_from_com(
            STEM_images[:, -2:] / nfph_, rsize=structure.unitcell
        )

    # Close all hdf5 files
    if h5file is not None:
        h5file.close()

    return result


def PACBED(
    structure,
    eV,
    app,
    thicknesses,
    nfph=10,
    gridshape=None,
    subslices=None,
    tiling=None,
    device=None,
    nT=5,
    subslicing=False,
    P=None,
    T=None,
    dtype=torch.float32,
    showProgress=True,
    precision=1,
):
    """
    Calculate position averaged convergent electron diffraction (PACBED) patterns.

    This is a convenience function, PACBEDs can also be calculated in the
    `STEM_multislice` function

    Parameters
    ----------
    structure : pyms.structure_routines.structure
        The structure of interest
    eV : float
        Probe energy in electron volts
    app : float
        Objective aperture in mrad
    thicknesses : float or array_like
        Thickness of the calculation
    gridshape : (2,) array_like, optional
        Pixel size of the simulation grid, if None is passed (default) then the
        autosampling routine is called
    subslices : array_like, optional
        A one dimensional array-like object containing the depths (in fractional
        coordinates) at which the object will be subsliced. The last entry
        should always be 1.0. For example, to slice the object into four equal
        sized slices pass [0.25,0.5,0.75,1.0]. If None is passed then the autoslicing
        routine is called
    nfph : int, optional
        Number of iterations of the frozen phonon algorithm (10 by default, which
        is a good rule of thumb for convergence)
    tiling : (2,) array_like, optional
        Tiling of a repeat unit cell on simulation grid
    device : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    nT : int, optional
        Number of independent multislice transmission functions generated and
        then selected from in the frozen phonon algorithm
    subslicing : bool,optional
        Set to True to allow output at fractions of the unit cell thickness
    P : (nslices,Y,X) array_like, optional
        Precomputed Fresnel free-space propagators
    T : (nT,nslices,Y,X) array_like
        Precomputed transmission functions
    showProgress : str or bool, optional
        Pass False to disable progress readout, pass 'notebook' to get correct
        progress bar behaviour inside a jupyter notebook
    precision : int, optional
        The precision level for grid sampling. Default is 1. Values can be 0 (rough), 1 (medium), or 2 (precise).
    Returns
    -------
    result : (len(nslices),Y,X) np.ndarray
        The requested PACBED simulation
    """
    # Perform routine to work out sampling for PACBED
    if gridshape is None or tiling is None:
        gridshape, tiling = autosampling_STEM(
            structure, thicknesses, app, eV, precision=precision
        )

    # Sampling of PACBED is half that required for a STEM image
    scan_posn = generate_STEM_raster(
        structure.unitcell[:2] * np.asarray(tiling) / 2, eV, app, tiling=tiling
    )

    # Perform autoslicing by default
    if subslices is None:
        subslices = structure.autoslice()

    result = STEM_multislice(
        structure,
        gridshape,
        eV,
        app,
        thicknesses,
        subslices,
        nfph=nfph,
        PACBED=True,
        scan_posn=scan_posn,
        tiling=tiling,
        device_type=device,
        subslicing=subslicing,
        nT=nT,
        T=T,
        P=P,
        showProgress=showProgress,
        dtype=dtype,
    )["PACBED"]

    return result


def STEM_multislice(
    structure,
    eV,
    app,
    thicknesses,
    gridshape=None,
    subslices=None,
    df=0,
    nfph=25,
    aberrations=[],
    batch_size=5,
    FourD_STEM=False,
    PACBED=False,
    h5_filename=None,
    STEM_images=None,
    DPC=False,
    scan_posn=None,
    dtype=torch.float32,
    device_type=None,
    beam_tilt=[0, 0],
    specimen_tilt=[0, 0],
    tilt_units="mrad",
    ROI=[0.0, 0.0, 1.0, 1.0],
    tiling=None,
    seed=None,
    showProgress=True,
    detector_ranges=None,
    D=None,
    fractional_occupancy=True,
    displacements=True,
    subslicing=False,
    nT=5,
    P=None,
    T=None,
    signal_list=None,
    precision=1,
):
    """
    Perform a STEM simulation using only the multislice algorithm.

    Parameters
    ----------
    structure : pyms.structure_routines.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts
    app : float
        Objective aperture in mrad
    thicknesses : float or array_like
        Thickness of the calculation
    gridshape : (2,) array_like, optional
        Pixel size of the simulation grid, if None is passed (default) then the
        autosampling routine is called
    subslices : array_like, optional
        A one dimensional array-like object containing the depths (in fractional
        coordinates) at which the object will be subsliced. The last entry
        should always be 1.0. For example, to slice the object into four equal
        sized slices pass [0.25,0.5,0.75,1.0]. If None is passed then the autoslicing
        routine is called
    df : float or array_like, optional
        Probe defocus, convention is that a negative defocus has the probe
        focused "into" the specimen
    nfph : int, optional
        Number of iterations of the frozen phonon algorithm (25 by default, which
        is a good rule of thumb for convergence). Set to 1 for absorptive
        calculations.
    aberrations : list, optional
        A list of of probe aberrations of class pyms.Probe.aberration, pass an
        empty list for an aberration free probe (the defocus aberration is also
        controlled by the df parameter)
    batch_size : int, optional
        The multislice algorithm can be performed on multiple probes columns
        at once to parallelize computation, the number of parallel computations
        is set by batch_size.
    fourD_STEM : bool or array_like, optional
        Pass fourD_STEM = True to perform 4D-STEM simulations. To save disk
        space a tuple containing pixel size and diffraction space extent of the
        datacube can be passed in. For example ([64,64],[1.2,1.2]) will output
        diffraction patterns cropped and downsampled/interpolated to measure
        64 x 64 pixels and 1.2 x 1.2 inverse Angstroms.
    PACBED : bool or array_like, optional
        Pass PACBED = True to calculate a position averaged convergent beam
        electron diffraction pattern (PACBED). Passing an array will specify the
        size to crop the PACBED pattern to.
    h5_filename : string or array_like of strings
        FourD-STEM images can be streamed directly to a hdf5 file output,
        avoiding the need for them to be stored in intermediate memory. To take
        advantage of this the user can provide a list of filenames to output
        these result to.
    STEM_images : (ndet, nthick, ndf, ny, nx) array_like
        An array containing the STEM images. If not provided but detector_ranges
        is then this will be initialized in the function.
    DPC : bool
        Set to True to simulate centre-of-mass differential phase contrast (DPC)
        STEM imaging. The centre of mass images will be added to the STEM_images
        output and the differential phase contrast reconstructions from these
        images will be outputted seperately in the results dictionary
    scan_posn :  (ny,nx,2) array_like, optional
        An array containing y and x scan positions in the final dimension for
        each scan position in the STEM raster, if provided overrides internal
        calculations of STEM sampling
    device_type : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    ROI : (4,) array_like
        Fraction of the unit cell to be scanned. Should contain [y0,x0,y1,x1]
        where [x0,y0] and [x1,y1] are the bottom left and top right coordinates
        of the region of interest (ROI) expressed as a fraction of the total
        grid (or unit cell).
    tiling : (2,) array_like, optional
        Tiling of a repeat unit cell on simulation grid
    seed : int
        Seed for random number generator for generating transmission functions
        and frozen phonon passes. Useful for testing purposes
    nT : int, optional
        Number of independent multislice transmission functions generated and
        then selected from in the frozen phonon algorithm. Set to 0 or -1 for
        absorptive calculation.
    contr : float, optional
        A threshhold for inclusion of ionization transitions within the
        calculation, if contr = 1.0 all ionization transitions will be included
        in the simulation otherwise only the transitions that make up a
        fraction equal to contr of the total ionization transitions will be
        included
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit floating point
    beam_tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) beam tilt, To maintain
        periodicity of the wave function at the boundaries this tilt is rounded
        to the nearest pixel value.
    specimen_tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) tilt of the specimen,
        by shearing the propagator. Units given by input variable tilt_units.
    tilt_units : string, optional
        Units of specimen and beam tilt, can be 'mrad','pixels' or 'invA'
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
    aberrations : list, optional
        A list containing a set of the class aberration, pass an empty list for
        an unaberrated probe.
    showProgress : str or bool, optional
        Pass False to disable progress readout, pass 'notebook' to get correct
        progress bar behaviour inside a jupyter notebook
    detector_ranges : array_like, optional
        The acceptance angles of each of the stem detectors should be stored as
        a list of lists: ie [[0,10],[10,20]] will make two detectors spanning
        0 to 10 mrad and 10 to 20 mrad.
    D : (ndet x Y x X) array_like, optional
        A precomputed set of STEM detectors, overrides detector_ranges.
    fractional_occupancy: bool
        Set to false to disable fracitonal occupancy of an atomic site by two
        different atomic species.
    subslicing : bool,optional
        Set to True to allow output at fractions of the unit cell thickness
    nT : int, optional
        Number of independent multislice transmission functions generated and
        then selected from in the frozen phonon algorithm
    P : (nslices,Y,X) array_like, optional
        Precomputed Fresnel free-space propagators
    T : (nT,nslices,Y,X) array_like
        Precomputed transmission functions
    signal_list : dictionary
        Transitions for which the cross-sections will be calculated. Of form:
        [{"signal":"<EELS/EDX>","Z":<atomic number>,"shell":"<1s,2s,2p>",
          "E0":<accelerating voltage>,"DeltaE":<window above threshold, in eV>,},...]
    precision : int, optional
        The precision level for grid sampling. Default is 1. Values can be 0 (rough), 1 (medium), or 2 (precise).

    Returns
    -------
    result : dict
        Can contain up to three entries with keys 'STEM images' which is the
        conventional STEM images simulated, 'datacube' which is a 4D-STEM
        datacube, 'PACBED' which is the position averaged convergent beam
        electron diffraction pattern and 'DPC' which are the reconstructions
        of the specimen phase from the centre-of-mass STEM images.
    """
    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    cdtype = real_to_complex_dtype_torch(dtype)

    tdisable, tqdm = tqdm_handler(showProgress)

    # Perform routine to work out sampling
    if gridshape is None or tiling is None:
        gridshape, tiling = autosampling_STEM(
            structure, thicknesses, app, eV, precision=precision
        )

    real_dim = structure.unitcell[:2] * np.asarray(tiling)

    # Perform autoslicing by default
    if subslices is None:
        subslices = structure.autoslice()

    # Ensure defoci are in proper array
    df = ensure_array(df)

    # Convert thicknesses into number of slices for multislice
    nslices = thickness_to_slices(
        thicknesses, structure.unitcell[2], subslicing, subslices
    )

    # Make STEM detectors
    if detector_ranges is not None:
        D = np.stack(
            [
                make_detector(gridshape, real_dim, eV, drange[1], drange[0])
                for drange in detector_ranges
            ]
        )

    if DPC:
        DPC_det = np.stack(q_space_array(gridshape, real_dim), axis=0)
        if D is not None:
            D = np.concatenate([D, DPC_det], axis=0)
        else:
            D = DPC_det

    method = multislice

    # Output only to multislice bandwidth limit if only one thickness is
    # considered, otherwise the full array must be kept for subsequent
    # iterations of the multislice algorithm
    output_to_bandwidth_limit = len(nslices) < 1

    kwargs = {
        "return_numpy": False,
        "qspace_in": True,
        "qspace_out": True,
        "output_to_bandwidth_limit": output_to_bandwidth_limit,
        "subslicing": subslicing,
    }

    # Initialize 4D-STEM datacube object
    DPshape, _, Ksize = workout_4DSTEM_datacube_DP_size(FourD_STEM, real_dim, gridshape)
    justPACBED = np.all(
        [
            PACBED,
            not FourD_STEM,
            not STEM_images,
            not DPC,
            detector_ranges is None,
            D is None,
        ]
    )
    if scan_posn is None:
        if justPACBED:
            # Sampling of PACBED is half that required for a STEM image
            scan_posn = generate_STEM_raster(
                structure.unitcell[:2] * np.asarray(tiling) / 2,
                eV,
                app,
                tiling=tiling,
            )
        else:
            scan_posn = generate_STEM_raster(real_dim, eV, app, tiling=tiling, ROI=ROI)
    scanshape = scan_posn.shape[:-1]
    if (detector_ranges is None) and (not DPC):
        D = np.zeros([1, *scanshape])

    # datacube should be an object with shape (ndf,nt,Ny,Nx,ny,nx)
    # this is currently not implemented for h5 files
    if h5_filename is None:
        # 4D-STEM Datacubes will be stored in the following list as they
        # are generated by the STEM routine
        datacubes = [
            None for i in range(len(ensure_array(df)))
        ]  # Modified to enable defocus series
    else:
        # Make a datacube for each thickness and defocus of interest
        nt = len(nslices)
        datacubes = []
        files = []
        for i in range(nt):
            datacube, f = initialize_h5_datacube_object(
                scanshape + tuple(DPshape),
                ensure_array(h5_filename)[i],
                dtype=torch_dtype_to_numpy(dtype),
                Rpix=nyquist_sampling(eV=eV, alpha=app),
                diffsize=Ksize,
                eV=eV,
                alpha=app,
                sample=structure.Title,
            )
            files.append(f)
            datacubes.append(datacube)

    STEM_images = np.zeros((len(df), D.shape[0], len(nslices), *scanshape))
    STEM_crosssection_images = np.zeros((len(nslices), *scanshape))

    if PACBED:
        DPshape2, _, Ksize = workout_4DSTEM_datacube_DP_size(
            PACBED, real_dim, gridshape
        )
        PACBED_pattern = np.zeros((len(nslices), len(df), *DPshape2))

    if signal_list is not None:
        # Calculate effective scattering potentials for cross-section calculations
        # Note: these include Debye-Waller factor smearing whether or not the calculation
        # is frozen phonon. See Findlay et al., Ultramicroscopy 104 (2005) 126. https://doi.org/10.1016/j.ultramic.2005.03.004
        Veff = structure.make_effective_scattering_potential(
            signal_list,
            gridshape,
            eV,
            subslices,
            tiling,
            device=device,
            dtype=dtype,
            fractional_occupancy=True,
        )
    else:
        Veff = None

    for i in tqdm(range(nfph), desc="Frozen phonon iteration", disable=tdisable):

        # Make propagators and transmission functions for multslice
        if P is None and T is None:
            P, T = multislice_precursor(
                structure,
                gridshape,
                eV,
                subslices=subslices,
                tiling=tiling,
                dtype=dtype,
                nT=nT,
                device=device,
                showProgress=showProgress,
                displacements=displacements,
                specimen_tilt=specimen_tilt,
                tilt_units=tilt_units,
                seed=seed,
            )

        # Put new transmission functions and propagators into arguments
        args = (P, T, tiling, device, seed)

        # Loop over defocus
        for idf, df_ in enumerate(
            tqdm(ensure_array(df), desc="Defocus series", disable=tdisable)
        ):
            # Make the STEM probe
            probe = focused_probe(
                gridshape,
                real_dim[:2],
                eV,
                app,
                df=df_,
                aberrations=aberrations,
                beam_tilt=beam_tilt,
                tilt_units=tilt_units,
            )

            result = STEM(
                real_dim,
                probe,
                method,
                nslices,
                eV,
                app,
                batch_size=batch_size,
                detectors=D,
                FourD_STEM=FourD_STEM,
                PACBED=PACBED,
                scan_posn=scan_posn,
                device=device,
                tiling=tiling,
                seed=seed,
                showProgress=showProgress,
                method_args=args,
                method_kwargs=kwargs,
                datacube=datacubes[idf],
                STEM_image=STEM_images[idf],
                Veff=Veff,
            )
            datacubes[idf] = result["datacube"]
            if result["STEM images"] is not None:
                STEM_images[idf] = result["STEM images"]
            if result["PACBED"] is not None:
                PACBED_pattern[:, idf] += result["PACBED"]
        if result["STEM crosssection images"] is not None:
            STEM_crosssection_images = result["STEM crosssection images"]

    # Normalize 4D-STEM datacubes by number of frozen phonon iterations
    if (datacubes is not None) and (FourD_STEM):
        if isinstance(datacubes, (list, tuple)):
            for idf in range(len(datacubes)):
                for it in range(len(datacubes[idf])):
                    datacubes[idf][it][:] = datacubes[idf][it][:] / nfph
        else:
            datacubes /= nfph

    if result["STEM images"] is not None:
        STEM_images = STEM_images / nfph

    if result["PACBED"] is not None:
        PACBED_pattern /= nfph

    if DPC:
        DPC_images = phase_from_com(
            np.squeeze(STEM_images.transpose(1, 2, 0, 3, 4))[-2:],
            rsize=structure.unitcell[:2],
        )  # Reshape STEM_images array so the DPC image array has the same ordering of dimensions as that of the other simulation results

    # Close all hdf5 files
    if h5_filename is not None:
        for f in files:
            f.close()

    if (detector_ranges is not None) and (not DPC):
        result["STEM images"] = np.squeeze(
            STEM_images.transpose(1, 2, 0, 3, 4)
        )  # Swap axes so output is of the form (Detector, thickness, defocus, ny, nx)
    elif (detector_ranges is not None) and (DPC):
        result["DPC"] = np.squeeze(STEM_images.transpose(1, 2, 0, 3, 4))[-2:]
        result["STEM images"] = np.squeeze(STEM_images.transpose(1, 2, 0, 3, 4)[:-2])
    elif (detector_ranges is None) and (DPC):
        result["DPC"] = np.squeeze(
            STEM_images.transpose(1, 2, 0, 3, 4)
        )  # Swap axes so output is of the form (Detector, thickness, defocus, ny, nx)
        del result["STEM images"]
    else:
        del result["STEM images"]

    if FourD_STEM:
        result["datacube"] = np.squeeze(np.array(datacubes).transpose(1, 0, 2, 3, 4, 5))
    else:
        del result["datacube"]

    # result = {"STEM images": np.squeeze(STEM_images), "datacube": datacubes, "STEM crosssection images": STEM_crosssection_images}

    if PACBED:
        result["PACBED"] = np.squeeze(PACBED_pattern)
    else:
        del result["PACBED"]

    if DPC:
        result["DPC"] = np.append(
            result["DPC"], np.expand_dims(DPC_images, axis=0), axis=0
        )

    if result["STEM crosssection images"] is not None:
        result["STEM crosssection images"] = np.squeeze(STEM_crosssection_images)
    else:
        del result["STEM crosssection images"]

    return result


def multislice_precursor(
    structure,
    gridshape,
    eV,
    subslices=None,
    tiling=[1, 1],
    specimen_tilt=[0, 0],
    tilt_units="mrad",
    nT=5,
    device=get_device(None),
    dtype=torch.float32,
    showProgress=True,
    displacements=True,
    fractional_occupancy=True,
    seed=None,
    band_width_limiting=[2 / 3, 2 / 3],
):
    """
    Make transmission functions and propagators for the multislice algorithm.

    Parameters
    ---------
    structure : pyms.structure_routines.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts
    subslices : array_like, optional
        A one dimensional array-like object containing the depths (in fractional
        coordinates) at which the object will be subsliced. The last entry
        should always be 1.0. For example, to slice the object into four equal
        sized slices pass [0.25,0.5,0.75,1.0]
    tiling : (2,) array_like, optional
        Tiling of a repeat unit cell on simulation grid
    specimen_tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) tilt of the specimen,
        by shearing the propagator. Units given by input variable tilt_units.
    tilt_units : string, optional
        Units of specimen tilt, can be 'mrad','pixels' or 'invA'
    nT : int, optional
        Number of independent multislice transmission functions generated and
        then selected from in the frozen phonon algorithm
        If nt<=0, calculates transmission functions for thermally-smeared elastic
        potential plus TDS absorptive potential
    device : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit floating point
    showProgress : str or bool, optional
        Pass False to disable progress readout, pass 'notebook' to get correct
        progress bar behaviour inside a jupyter notebook
    displacements : bool, optional
        Pass False to disable thermal vibration of the atoms
    fractional_occupancy : bool, optional
        Pass False to fractional occupancy of atomic sites
    bandwidth_limiting : (2,) array_like, optional
        The bandwidth limiting scheme. The propagator and transmission functions
        will be bandwidth limited up to the fraction given by the first and
        second entries of band_width_limiting respectively

    Returns
    -------
    P : complex np.ndarray (len(subslices),Y,X) or (Y,X)
        The Fresnel free-space propagators for multislice. If the slicing of the
        specimen is even (ie. np.diff(subslices) are all the same) then a single
        propagator will be returned otherwise different propagators for each slice
        are returned
    T : torch.Tensor (nT,len(subslices),Y,X,2)
        The specimen transmission functions for multislice
    """
    tdisable, tqdm = tqdm_handler(showProgress)

    # Perform autoslicing by default
    if subslices is None:
        subslices = structure.autoslice()

    # Calculate grid size in Angstrom
    rsize = np.zeros(3)
    rsize[:3] = structure.unitcell[:3]
    rsize[:2] *= np.asarray(tiling)

    # If slices are basically equidistant, we can use the same propagator
    if np.std(np.diff(subslices, prepend=0)) < 1e-4:
        P = make_propagators(
            gridshape,
            rsize,
            eV,
            subslices[:1],
            tilt=specimen_tilt,
            tilt_units=tilt_units,
            bandwidth_limit=band_width_limiting[0],
        )[0]
    else:
        P = make_propagators(
            gridshape,
            rsize,
            eV,
            subslices,
            tilt=specimen_tilt,
            tilt_units=tilt_units,
            bandwidth_limit=band_width_limiting[0],
        )

    if (
        nT >= 1
    ):  # If at least one independent multislice transmission functions sought, assume frozen phonon calculation
        T = torch.zeros(nT, len(subslices), *gridshape, device=device, dtype=dtype)
        T = torch.complex(*(2 * [T]))

        for i in tqdm(range(nT), desc="Making projected potentials", disable=tdisable):
            T[i] = structure.make_transmission_functions(
                gridshape,
                eV,
                subslices=subslices,
                tiling=tiling,
                device=device,
                dtype=dtype,
                displacements=displacements,
                fractional_occupancy=fractional_occupancy,
                seed=seed,
                bandwidth_limit=band_width_limiting[1],
            )

    else:  # Otherwise, assume absorptive calculation sought
        print(
            "Note: will use thermally-smeared elastic potential plus TDS absorptive potential"
        )
        T = torch.zeros(
            1, len(subslices), *gridshape, device=device, dtype=dtype
        )  # Hardcodes for single transmission function
        T = torch.complex(*(2 * [T]))

        T[0] = structure.make_transmission_functions_absorptive(
            gridshape,
            eV,
            subslices=subslices,
            tiling=tiling,
            device=device,
            dtype=dtype,
            displacements=False,
            fractional_occupancy=fractional_occupancy,
            seed=seed,
            bandwidth_limit=band_width_limiting[1],
            showProgress=showProgress,
        )

    return P, T


def STEM_EELS_multislice(
    structure,
    gridshape,
    eV,
    app,
    detector_ranges,
    thicknesses,
    Ztarget,
    n,
    ell,
    epsilon,
    df=0,
    subslices=None,
    device_type=None,
    tiling=[1, 1],
    nfph=5,
    nT=5,
    contr=0.95,
    dtype=torch.float32,
    showProgress=True,
    ionization_cutoff=1e-4,
    beam_tilt=[0, 0],
    specimen_tilt=[0, 0],
    tilt_units="mrad",
    aberrations=[],
    batch_size=1,
    subslicing=False,
    Hn0=None,
    P=None,
    T=None,
):
    """
    Perform a STEM-EELS simulation using only the multislice algorithm.

    Parameters
    ----------
    structure : pyms.structure_routines.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts
    app : float
        Objective aperture in mrad
    detector_ranges : array_like
        The maximum acceptance angles of each of the spectrometer apertures
        should be stored as in an array: ie [10,20] will make two detectors
        spanning 0 to 10 mrad and 0 to 20 mrad.
    thicknesses : float or array_like
        Thickness of the calculation
    Ztarget : int
        Atomic number of the ionization targets
    n : int
        Principal atomic number of the bound transition of interest (1 for K
        shell, 2 for L shell etc.)
    ell : int
        orbital angular momentum quantum number of hte bound transition of
        interest
    epsilon : float
        Energy above ionization threshhold energy
    df : float, optional
        Probe defocus
    subslices : array_like, optional
        A one dimensional array-like object containing the depths (in fractional
        coordinates) at which the object will be subsliced. The last entry
        should always be 1.0. For example, to slice the object into four equal
        sized slices pass [0.25,0.5,0.75,1.0]
    device_type : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    tiling : (2,) array_like, optional
        Tiling of a repeat unit cell on simulation grid
    nT : int, optional
        Number of independent multislice transmission functions generated and
        then selected from in the frozen phonon algorithm
    nfph : int, optional
        Number of frozen phonon iterations.
    contr : float, optional
        A threshhold for inclusion of ionization transitions within the
        calculation, if contr = 1.0 all ionization transitions will be included
        in the simulation otherwise only the transitions that make up a
        fraction equal to contr of the total ionization transitions will be
        included
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit floating point
    ionization_cutoff : float
        Threshhold below which the contribution of certain ionizations will be
        ignored.
    beam_tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) beam tilt, To maintain
        periodicity of the wave function at the boundaries this tilt is rounded
        to the nearest pixel value.
    specimen_tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) tilt of the specimen,
        by shearing the propagator. Units given by input variable tilt_units.
    tilt_units : string, optional
        Units of specimen and beam tilt, can be 'mrad','pixels' or 'invA'
    aberrations : list, optional
        A list containing a set of the class aberration, pass an empty list for
        an unaberrated probe.
    showProgress : str or bool, optional
        Pass False to disable progress readout, pass 'notebook' to get correct
        progress bar behaviour inside a jupyter notebook
    subslicing : bool,optional
        Set to True to allow output at fractions of the unit cell thickness
    Hn0 : (n,y,x) np.complex, optional
        Precalculated Hn0s ionization transition potentials, if not provided
        these will be calculated.
    P : (nslices,y,x) array_like, optional
        Precalculated multislice propagators, default is None which will mean
        that these are calculated within the routine.
    T : (nT,nslices,y,x) array_like, optional
        Precalculated multislice transmission functions, default is None which
        will mean that these are calculated within the routine.
    """
    tdisable, tqdm = tqdm_handler(showProgress)

    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Perform autoslicing by default
    if subslices is None:
        subslices = structure.autoslice()

    # Get gridsize in Angstrom
    rsize = structure.unitcell[:2] * np.asarray(tiling)

    # Convert thicknesses to multislice slices
    nslices = thickness_to_slices(
        thicknesses, structure.unitcell[2], subslicing=subslicing, subslices=subslices
    )

    # Get the coordinates of the target atoms in a unit cell
    tiled_structure = structure.tile(*tiling)
    ionization_sites = tiled_structure.atoms[tiled_structure.atoms[:, 3] == Ztarget][
        :, :3
    ]
    # Make probe
    probe = focused_probe(
        gridshape,
        rsize,
        eV,
        app,
        df=df,
        beam_tilt=beam_tilt,
        tilt_units=tilt_units,
        aberrations=aberrations,
    )
    # Generate ionization transition potentials for atom of interest
    if Hn0 is None:
        Hn0 = get_transitions(
            Ztarget, n, ell, epsilon, eV, gridshape, rsize, order=1, contr=contr
        )

    # Make EELS aperture masks
    D = np.stack(
        [
            make_detector(gridshape, rsize, eV, d, 0)
            for d in ensure_array(detector_ranges)
        ]
    )

    # The method pyms.Ionization.transition_potential_multislice will be passed
    # to the STEM routine, make a list of the function arguments and a
    # dictionary of the keyword arguments to also pass to the STEM routine

    kwargs = {
        "tiling": tiling,
        "device_type": device,
        "return_numpy": False,
        "qspace_in": True,
        "threshhold": ionization_cutoff,
        "showProgress": showProgress,
        "tqposition": 1,
    }

    probe_posn = generate_STEM_raster(rsize, eV, app, tiling=tiling)
    # STEM_images array shape should be (ndet, Y, X)
    STEM_images = np.zeros((D.shape[0], *probe_posn.shape[:2]))

    for _ in tqdm(
        range(nfph), desc="Frozen phonon interations:", disable=tdisable, position=0
    ):
        # Make propagators and transmission functions for multislice
        if P is None or T is None:
            P, T = multislice_precursor(
                structure,
                gridshape,
                eV,
                subslices=subslices,
                tiling=tiling,
                nT=nT,
                device=device,
                showProgress=showProgress,
                specimen_tilt=specimen_tilt,
                tilt_units=tilt_units,
            )
        args = (subslices, P, T, Hn0, ionization_sites)
        # Call STEM routine
        STEM_images += STEM(
            rsize,
            probe,
            transition_potential_multislice,
            nslices,
            eV,
            app,
            batch_size=1,
            scan_posn=probe_posn,
            detectors=D,
            device=device,
            tiling=tiling,
            showProgress=showProgress,
            method_args=args,
            method_kwargs=kwargs,
        )["STEM images"][:, 0, ...]
    return STEM_images / nfph


def CBED(
    structure,
    eV,
    app,
    thicknesses,
    gridshape=None,
    subslices=None,
    device_type=None,
    dtype=torch.float32,
    tiling=None,
    nT=5,
    nfph=25,
    showProgress=True,
    beam_tilt=[0, 0],
    specimen_tilt=[0, 0],
    df=0,
    tilt_units="mrad",
    aberrations=[],
    probe_posn=None,
    subslicing=False,
    nslices=None,
    precision=1,
    P=None,
    T=None,
    seed=None,
):
    """
    Perform a convergent-beam electron diffraction (CBED) simulation.

    This is the diffraction pattern formed by a focused probe. This can also
    be used to simulate electron diffraction patterns from a plane wave source
    if a small enough aperture (app) is selected

    Parameters
    ----------
    structure : pyms.structure_routines.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts
    app : float
        Image-forming lens aperture in mrad, pass None for aperture-less imaging
    thicknesses : float or array_like
        Thickness of the object in the calculation, will be rounded off to the
        nearest unit cell.
    subslices : array_like, optional
        A one dimensional array-like object containing the depths (in fractional
        coordinates) at which the object will be subsliced. The last entry
        should always be 1.0. For example, to slice the object into four equal
        sized slices pass [0.25,0.5,0.75,1.0]
    device_type : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit floating point
    tiling : (2,) array_like, optional
        Tiling of a repeat unit cell on simulation grid
    nT : int, optional
        Number of independent multislice transmission functions generated and
        then selected from in the frozen phonon algorithm
    nfph : int, optional
        Number of iterations of the frozen phonon algorithm (25 by default)
    showProgress : str or bool, optional
        Pass False to disable progress readout, pass 'notebook' to get correct
        progress bar behaviour inside a jupyter notebook
    beam_tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) beam tilt, To maintain
        periodicity of the wave function at the boundaries this tilt is rounded
        to the nearest pixel value.
    specimen_tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) tilt of the specimen,
        by shearing the propagator. Units given by input variable tilt_units.
    df : float, optional
        Probe defocus in Angstrom
    tilt_units : string, optional
        Units of specimen and beam tilt, can be 'mrad','pixels' or 'invA'
    aberrations : list, optional
        A list containing a set of the class aberration, pass an empty list for
        an unaberrated probe.
    probe_posn : array_like, optional
        Probe position as a fraction of the unit-cell, by default the probe will
        be in the upper-left corner of the simulation grid. This is the [0,0]
        coordinate.
    subslicing : bool,optional
        Set to True to allow output at fractions of the unit cell thickness
    nslices : int or array_like
        Maximum slice number or set of slices to perform multislice algorithm
        over.
    precision : int, optional
        The precision level for grid sampling. Default is 1. Values can be 0 (rough), 1 (medium), or 2 (precise).
    P : (n,Y,X) array_like, optional
        Precomputed Fresnel free-space propagators
    T : (n,Y,X) array_like
        Precomputed transmission functions
    seed : int
        Seed for random number generator for generating transmission functions
        and frozen phonon passes. Useful for testing purposes
    Returns
    -------
    CBED : (len(thicknesses,Y,X)) array_like
        The requested convergent beam electron diffraction patterns
    scale : (2,) float
        Scale of the CBED array in inverse Angstrom/pixel
    """
    tdisable, tqdm = tqdm_handler(showProgress)
    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Perform routine to work out sampling for STEM
    if gridshape is None or tiling is None:
        gridshape, tiling = autosampling_STEM(
            structure, thicknesses, app, eV, precision=precision
        )

    # Perform autoslicing by default
    if subslices is None:
        subslices = structure.autoslice()

    # Make propagators and transmission functions for multislice
    if P is None and T is None:
        P, T = multislice_precursor(
            structure,
            gridshape,
            eV,
            subslices=subslices,
            tiling=tiling,
            dtype=dtype,
            nT=nT,
            device=device,
            showProgress=showProgress,
            specimen_tilt=specimen_tilt,
            tilt_units=tilt_units,
            seed=seed,
        )

    t_ = np.asarray(ensure_array(thicknesses))

    output = np.zeros((t_.shape[0], *size_of_bandwidth_limited_array(gridshape)))

    if nslices is None:
        # Convert thicknesses into number of slices for multislice
        nslices = thickness_to_slices(
            thicknesses, structure.unitcell[2], subslicing, subslices
        )

    # Iteration over frozen phonon configurations
    for _ in tqdm(range(nfph), desc="Frozen phonon iteration", disable=tdisable):
        # Make probe
        probe = focused_probe(
            gridshape,
            structure.unitcell[:2] * np.asarray(tiling),
            eV,
            app,
            qspace=True,
            df=df,
            aberrations=aberrations,
            beam_tilt=beam_tilt,
            tilt_units=tilt_units,
        )

        if not (probe_posn is None):
            probe = fourier_shift(
                probe,
                np.asarray(probe_posn) / np.asarray(tiling),
                pixel_units=False,
                qspacein=True,
                qspaceout=True,
            )

        # Run multislice iterating over different thickness outputs
        for it, t in enumerate(np.diff(nslices, prepend=0)):
            probe = multislice(
                probe,
                t,
                P,
                T,
                tiling=tiling,
                output_to_bandwidth_limit=False,
                qspace_in=True,
                qspace_out=True,
                device_type=device,
                seed=seed,
            )

            output[it] += (
                np.abs(
                    crop(
                        np.fft.fftshift(probe),
                        size_of_bandwidth_limited_array(gridshape),
                    )
                )
                ** 2
            )

    # Divide output by # of pixels to compensate for Fourier transform
    return output / np.prod(gridshape) / nfph, 1 / (
        np.asarray(tiling) * structure.unitcell[:2]
    )


def HRTEM(
    structure,
    eV,
    app,
    thicknesses,
    gridshape=None,
    subslices=None,
    df=0,
    device_type=None,
    dtype=torch.float32,
    tiling=None,
    nT=5,
    nfph=25,
    showProgress=True,
    beam_tilt=[0, 0],
    specimen_tilt=[0, 0],
    tilt_units="mrad",
    aberrations=[],
    subslicing=False,
    P=None,
    T=None,
    precision=1,
):
    """
    Perform a high-resolution transmission electron microscopy (HRTEM) simulation.

    This assumes plane wave illumination and a post-specimen image forming lens

    Parameters
    ----------
    structure : pyms.structure_routines.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts
    app : float
        Image-forming lens aperture in mrad, pass None for aperture-less imaging
    thicknesses : float or array_like
        Thickness of the object in the calculation, will be rounded off to the
        nearest unit cell.
    subslices : array_like, optional
        A one dimensional array-like object containing the depths (in fractional
        coordinates) at which the object will be subsliced. The last entry
        should always be 1.0. For example, to slice the object into four equal
        sized slices pass [0.25,0.5,0.75,1.0]
    df : float or array_like
        Probe defocus or defocii (if array)
    device_type : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit floating point
    tiling : (2,) array_like, optional
        Tiling of a repeat unit cell on simulation grid
    nT : int, optional
        Number of independent multislice transmission functions generated and
        then selected from in the frozen phonon algorithm
    nfph : int, optional
        Number of iterations of the frozen phonon algorithm (25 by default)
    showProgress : str or bool, optional
        Pass False to disable progress readout, pass 'notebook' to get correct
        progress bar behaviour inside a jupyter notebook
    beam_tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) beam tilt, To maintain
        periodicity of the wave function at the boundaries this tilt is rounded
        to the nearest pixel value.
    specimen_tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) tilt of the specimen,
        by shearing the propagator. Units given by input variable tilt_units.
    tilt_units : string, optional
        Units of specimen and beam tilt, can be 'mrad','pixels' or 'invA'
    aberrations : list, optional
        A list containing a set of the class aberration, pass an empty list for
        an unaberrated contrast transfer function.
    subslicing : bool,optional
        Set to True to allow output at fractions of the unit cell thickness
    P : (n,Y,X) array_like, optional
        Precomputed Fresnel free-space propagators
    T : (n,Y,X) array_like, optional
        Precomputed transmission functions
    Returns
    -------
    result : (len(df), len(nslices), Y,X) array_like
        Resulting HRTEM images, result will be cropped to the reciprocal space
        bandwidth limit and in general the Y and X array size will be 2/3 of
        the requested array
    """

    if tiling is None or gridshape is None:
        gridshape, tiling = autosampling_TEM(structure, precision)

    tdisable, tqdm = tqdm_handler(showProgress)
    cdtype = real_to_complex_dtype_torch(dtype)

    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Perform autoslicing by default
    if subslices is None:
        subslices = structure.autoslice()

    # Make propagators and transmission functions for multslice
    if P is None and T is None:
        P, T = multislice_precursor(
            structure,
            gridshape,
            eV,
            subslices=subslices,
            tiling=tiling,
            dtype=dtype,
            nT=nT,
            device=device,
            showProgress=showProgress,
        )

    bw_limit_size = size_of_bandwidth_limited_array(gridshape)

    t_ = ensure_array(thicknesses)
    output = np.zeros((len(t_), *bw_limit_size))

    # Convert thicknesses into number of slices for multislice
    nslices = thickness_to_slices(t_, structure.unitcell[2], subslicing, subslices)

    rsize = np.asarray(structure.unitcell[:2]) * np.asarray(tiling)

    # Option for focal series, just pass an array of defocii, this bit of
    # code will set up the lens transfer functions for each case
    defocii = ensure_array(df)

    ctf = (
        torch.from_numpy(
            np.stack(
                [
                    make_contrast_transfer_function(
                        bw_limit_size, rsize, eV, app, df=df_, aberrations=aberrations
                    )
                    for df_ in defocii
                ]
            )
        )
        .type(cdtype)
        .to(device)
    )
    output = np.zeros((len(defocii), len(nslices), *bw_limit_size))

    # Iteration over frozen phonon configurations
    for _ in tqdm(range(nfph), desc="Frozen phonon iteration", disable=tdisable):
        probe = plane_wave_illumination(
            gridshape, rsize, eV, beam_tilt, tilt_units, qspace=True
        )
        # Run multislice iterating over different thickness outputs
        for it, t in enumerate(np.diff(nslices, prepend=0)):
            probe = multislice(
                probe,
                t,
                P,
                T,
                tiling=tiling,
                output_to_bandwidth_limit=False,
                qspace_in=True,
                qspace_out=True,
                return_numpy=False,
                device_type=device,
            )
            output[:, it, ...] += (
                amplitude(
                    torch.fft.ifftn(
                        ctf * crop_to_bandwidth_limit_torch(probe), dim=(-2, -1)
                    )
                )
                .cpu()
                .numpy()
            )

    # Calculate Angstrom per pixel (scale)
    Aperpix = structure.unitcell[:2] * np.asarray(tiling) / output.shape[-2:]

    # Divide output by # of pixels to compensate for Fourier transform
    return np.squeeze(output) / nfph, Aperpix


def EFTEM(
    structure,
    gridshape,
    eV,
    app,
    thicknesses,
    Ztarget,
    n,
    ell,
    epsilon,
    df=0,
    subslices=None,
    device_type=None,
    tiling=[1, 1],
    nT=5,
    contr=0.95,
    dtype=torch.float32,
    beam_tilt=[0, 0],
    specimen_tilt=[0, 0],
    tilt_units="mrad",
    aberrations=[],
    showProgress=True,
    ionization_cutoff=None,
    Hn0=None,
    nfph=5,
    P=None,
    T=None,
):
    """
    Perform an elemental mapping energy-filtered TEM (EFTEM) simulation.

    Parameters
    ----------
    structure : pyms.structure_routines.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts
    app : float
        Objective aperture in mrad
    thicknesses : float or array_like
        Thickness of the calculation
    Ztarget : int
        Atomic number of the ionization targets
    n : int
        Principal atomic number of the bound transition of interest (1 for K
        shell, 2 for L shell etc.)
    ell : int
        orbital angular momentum quantum number of hte bound transition of
        interest
    epsilon : float
        Energy above ionization threshhold energy
    df : float or array_like, optional
        Probe defocus, 0 by default.
    subslices : array_like, optional
        A one dimensional array-like object containing the depths (in fractional
        coordinates) at which the object will be subsliced. The last entry
        should always be 1.0. For example, to slice the object into four equal
        sized slices pass [0.25,0.5,0.75,1.0]
    device_type : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    tiling : (2,) array_like, optional
        Tiling of a repeat unit cell on simulation grid
    nT : int, optional
        Number of independent multislice transmission functions generated and
        then selected from in the frozen phonon algorithm
    contr : float, optional
        A threshhold for inclusion of ionization transitions within the
        calculation, if contr = 1.0 all ionization transitions will be included
        in the simulation otherwise only the transitions that make up a
        fraction equal to contr of the total ionization transitions will be
        included
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit floating point
    beam_tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) beam tilt, To maintain
        periodicity of the wave function at the boundaries this tilt is rounded
        to the nearest pixel value.
    specimen_tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) tilt of the specimen,
        by shearing the propagator. Units given by input variable tilt_units.
    tilt_units : string, optional
        Units of specimen and beam tilt, can be 'mrad','pixels' or 'invA'
    aberrations : list, optional
        A list containing a set of the class aberration, pass an empty list for
        an unaberrated probe.
    showProgress : str or bool, optional
        Pass False to disable progress readout, pass 'notebook' to get correct
        progress bar behaviour inside a jupyter notebook
    ionization_cutoff : float
        Threshhold below which the contribution of certain ionizations will be
        ignored.
    Hn0 : (n,y,x) np.complex, optional
        Precalculated Hn0s ionization transition potentials, if not provided
        these will be calculated.
    P : (nslices,y,x) array_like, optional
        Precalculated multislice propagators, default is None which will mean
        that these are calculated within the routine.
    T : (nT,nslices,y,x) array_like, optional
        Precalculated multislice transmission functions, default is None which
        will mean that these are calculated within the routine.
    Returns
    -------
    result : (len(df),Y,X) array_like
        The EFTEM images, if an array of defocus values is provided then a
        defocus series will be provided.
    """
    tdisable, tqdm = tqdm_handler(showProgress)

    # Calculate grid size in Angstrom
    rsize = np.zeros(3)
    rsize[:3] = structure.unitcell[:3]
    rsize[:2] *= np.asarray(tiling)

    # Perform autoslicing by default
    if subslices is None:
        subslices = structure.autoslice()

    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Force defocus to be an array
    defocii = ensure_array(df)

    # Construct our (plane wave) illuminating probe
    probe = plane_wave_illumination(gridshape, rsize, eV, beam_tilt, tilt_units)

    # Calculate the size of the grid after band-width limiting
    bw_limit_size = size_of_bandwidth_limited_array(gridshape)

    # Calculate lens contrast transfer functions
    ctfs = (
        torch.from_numpy(
            np.stack(
                [
                    make_contrast_transfer_function(
                        bw_limit_size, rsize, eV, app, df=df_, aberrations=aberrations
                    )
                    for df_ in defocii
                ],
                axis=0,
            )
        )
        .to(real_to_complex_dtype_torch(dtype))
        .to(device)
    )

    # Convert thicknesses to number of unit cells
    nslices = np.ceil(thicknesses / structure.unitcell[2]).astype(int)

    # Get the coordinates of the target atoms in a unit cell
    mask = structure.atoms[:, 3] == Ztarget

    # Adjust fractional coordinates for tiling of unit cell
    coords = structure.atoms[mask][:, :3] / np.asarray(tiling + [1])

    # nstates = len(freeQuantumNumbers)
    if Hn0 is None:
        Hn0 = get_transitions(
            Ztarget, n, ell, epsilon, eV, gridshape, rsize, order=1, contr=0.99
        )

    result = np.zeros(ctfs.shape, dtype=torch_dtype_to_numpy(dtype))

    # Perform multislice simulation and return the tiled out result
    for _ in tqdm(
        range(nfph), desc="Frozen phonon iterations:", disable=tdisable, position=0
    ):
        # Calculate propagator and transmission functions for multislice
        if P is None or T is None:
            P, T = multislice_precursor(
                structure,
                gridshape,
                eV,
                subslices,
                tiling,
                specimen_tilt,
                tilt_units,
                nT,
                device,
                dtype,
                showProgress,
            )

        img = transition_potential_multislice(
            probe,
            nslices,
            subslices,
            P,
            T,
            Hn0,
            coords,
            image_CTF=ctfs,
            tiling=[1, 1],
            device_type=device,
            seed=None,
            return_numpy=True,
            qspace_in=False,
            threshhold=ionization_cutoff,
            showProgress=showProgress,
            tqposition=1,
        )
        result += tile_out_ionization_image(
            img,
            tiling,
        )
    return result


def STEM_EELS_PRISM(
    structure,
    gridshape,
    eV,
    app,
    det,
    thicknesses,
    Ztarget,
    n,
    ell,
    epsilon,
    nfph=5,
    aberrations=[],
    df=0,
    Hn0_crop=None,
    subslices=None,
    device_type=None,
    tiling=[1, 1],
    nT=5,
    PRISM_factor=[1, 1],
    contr=0.95,
    dtype=torch.float32,
    Hn0s=None,
    showProgress=True,
    do_reverse_multislice=True,
):
    """
    Perform a STEM EELS simulation using the PRISM algorithm.

    See Brown, Hamish G., Jim Ciston, and Colin Ophus. "Linear-scaling
    algorithm for rapid computation of inelastic transitions in the presence of
    multiple electron scattering." Physical Review Research 1.3 (2019): 033186
    for details. Two scattering matrices are used to propagate the probe from
    entrance surface to plane of ionization and then to exit surface and this
    generally economizes on the total number of multislice iterations necessary.

    Parameters
    ----------
    structure : pyms.structure_routines.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts
    app : float
        Probe-forming aperture in mrad
    det : float
        EELS aperture acceptance angle in mrad
    thicknesses : float
        Thickness of the calculation
    Ztarget : int
        Atomic number of the ionization targets
    n : int
        Principal atomic number of the bound transition of interest (1 for K
        shell, 2 for L shell etc.)
    ell : int
        orbital angular momentum quantum number of hte bound transition of
        interest
    epsilon : float
        Energy above ionization threshhold energy
    nfph : int,optional
        Number of frozen phonon iterations default is 5
    df : float,optional
        Probe defocus
    Hn0_crop : (2,) int array_like, optional
        Cropping of the electron transition potential to speed up calculation
    subslices : array_like, optional
        A one dimensional array-like object containing the depths (in fractional
        coordinates) at which the object will be subsliced. The last entry
        should always be 1.0. For example, to slice the object into four equal
        sized slices pass [0.25,0.5,0.75,1.0]
    device_type : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    tiling : (2,) array_like, optional
        Tiling of a repeat unit cell on simulation grid
    nT : int, optional
        Number of independent multislice transmission functions generated and
        then selected from in the frozen phonon algorithm
    PRISM_factor : int (2,) array_like, optional
        The PRISM "interpolation factor" this is the amount by which the
        scattering matrices are cropped in real space to speed up
        calculations see Ophus, Colin. "A fast image simulation algorithm
        for scanning transmission electron microscopy." Advanced structural
        and chemical imaging 3.1 (2017): 13 for details on this.
    contr : float, optional
        A threshhold for inclusion of ionization transitions within the
        calculation, if contr = 1.0 all ionization transitions will be included
        in the simulation otherwise only the transitions that make up a
        fraction equal to contr of the total ionization transitions will be
        included
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit floating point
    Hn0s : (n,y,x) np.complex, optional
        Precalculated Hn0s ionization transition potentials, if not provided
        these will be calculated.
    showProgress : str or bool, optional
        Pass False to disable progress readout, pass 'notebook' to get correct
        progress bar behaviour inside a jupyter notebook
    do_reverse_multislice : bool, optional
        Set to False (True is default) to turn off the reverse multislice
        optimization

    Returns
    -------
    EELS_image : (nY,nX) np.ndarray
        The calculated STEM EELS image
    """
    tdisable, tqdm = tqdm_handler(showProgress)
    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Perform autoslicing by default
    if subslices is None:
        subslices = structure.autoslice()

    # Calculate grid size in Angstrom
    rsize = np.zeros(3)
    rsize[:3] = structure.unitcell[:3]
    rsize[:2] *= np.asarray(tiling)

    # Generate scan positions in pixels
    scan = generate_STEM_raster(rsize[:2], eV, app, tiling=tiling, gridshape=gridshape)
    scan_shape = scan.shape[:2]
    nprobe_posn = np.prod(scan_shape)

    # Get the coordinates of the target atoms in a unit cell
    import copy

    tiled_structure = copy.deepcopy(structure).tile(*tiling)
    coords = tiled_structure.atoms[tiled_structure.atoms[:, 3] == Ztarget][:, :3]
    nslices = np.ceil(thicknesses / structure.unitcell[2]).astype(int)

    # Make ionization transition potentials
    if Hn0s is None:
        Hn0s = get_transitions(
            Ztarget,
            n,
            ell,
            epsilon,
            eV,
            size_of_bandwidth_limited_array(gridshape),
            rsize,
            order=1,
            contr=0.99,
        )

    # Make probe wavefunction vectors for scan
    # Get kspace grid in units of inverse pixels
    ky, kx = [
        np.fft.fftfreq(gridshape[-2 + i], d=1 / gridshape[-2 + i]) for i in range(2)
    ]

    # Electron wavelength
    from .Probe import wavev, chi

    lam = 1 / wavev(eV)
    scan_array = None

    # Initialize Image
    EELS_image = torch.zeros(nprobe_posn, dtype=dtype, device=device)

    # Loop over frozen phonons

    for ifph in tqdm(
        range(nfph), disable=tdisable, position=0, desc="Frozen phonon iteration"
    ):

        P, T = multislice_precursor(
            structure,
            gridshape,
            eV,
            subslices=subslices,
            tiling=tiling,
            device=device,
            dtype=dtype,
            showProgress=False,
        )

        # Scattering matrix 1 propagates probe from surface of specimen to slice of
        # interest
        S1 = scattering_matrix(
            rsize,
            P,
            T,
            0,
            eV,
            app,
            batch_size=5,
            subslicing=True,
            PRISM_factor=PRISM_factor,
            showProgress=False,
        )

        # Scattering matrix 2 propagates probe from slice of ionization to exit surface
        S2 = scattering_matrix(
            rsize,
            P,
            T,
            nslices * len(subslices),
            eV,
            det,
            batch_size=5,
            subslicing=True,
            transposed=True,
            PRISM_factor=PRISM_factor,
            showProgress=False,
        )
        # Link the slices and seeds of both scattering matrices
        S1.seed = S2.seed

        # On first iteration generate illumination vector and work out transition
        # potential cropping
        if ifph == 0:
            scan_array = np.zeros((nprobe_posn, S1.S.shape[0]), dtype=complex)

            # TODO test aberrations and defocii
            negtwopii = -2 * np.pi * 1j
            for i in range(S1.nbeams):
                ky_ = ky[S1.beams[0][i]]
                kx_ = kx[S1.beams[1][i]]
                k = np.hypot(ky_ / rsize[0], kx_ / rsize[1])
                kphi = np.arctan2(ky_, kx_)
                scan_array[:, i] = np.exp(
                    negtwopii
                    * (ky_ * scan[..., 0].ravel() + kx_ * scan[..., 1].ravel())
                    - 1j * chi(k, kphi, lam, df, aberrations)
                ).ravel()
            # Normalize scan_array and convert to torch array
            scan_array *= 1 / np.sqrt(np.sum(scan_array.shape[1]))
            scan_array = torch.from_numpy(scan_array).to(device)

            if Hn0_crop is None:
                Hn0_crop = [S1.stored_gridshape[i] // PRISM_factor[i] for i in range(2)]
            else:
                Hn0_crop = [
                    min(H, S // P)
                    for H, S, P in zip(Hn0_crop, S1.stored_gridshape, PRISM_factor)
                ]

            # We achieve cropping of the Hn0 with some abuse of the fourier
            # interpolation routine
            from .utils import fourier_interpolate

            Hn0s = np.fft.fft2(
                fourier_interpolate(Hn0s, Hn0_crop, qspace_in=True, qspace_out=True)
            )

        total_slices = nslices * len(subslices)
        for islice in tqdm(
            range(total_slices), desc="Slice", position=1, disable=tdisable
        ):

            # Propagate scattering matrices to this slice
            if islice > 0:
                S1.Propagate(
                    islice, P, T, subslicing=True, batch_size=5, showProgress=False
                )
                if do_reverse_multislice:
                    S2.Propagate(
                        total_slices - islice,
                        P,
                        T,
                        subslicing=True,
                        batch_size=5,
                        showProgress=False,
                    )
                else:
                    S2 = scattering_matrix(
                        rsize,
                        P,
                        T,
                        total_slices - islice,
                        eV,
                        det,
                        batch_size=5,
                        subslicing=True,
                        transposed=True,
                        PRISM_factor=PRISM_factor,
                        showProgress=False,
                        seed=S1.seed,
                    )

            # Flatten indices of second scattering matrix in preparation for
            # indexing
            S2.S = S2.S.reshape(S2.S.shape[0], np.prod(S2.stored_gridshape))

            # Work out which subslice of the crystal unit cell we are in
            subslice = islice % S1.nsubslices

            # Get list of atoms within this slice
            atomsinslice = coords[
                np.logical_and(
                    coords[:, 2] >= subslice / S1.nsubslices,
                    coords[:, 2] < (subslice + 1) / S1.nsubslices,
                ),
                :2,
            ]

            # Iterate over atoms in this slice
            for atom in tqdm(
                atomsinslice, "Transitions in slice", disable=tdisable, position=2
            ):

                windex = torch.from_numpy(
                    window_indices(atom, Hn0_crop, S1.stored_gridshape)
                ).to(device)

                for Hn0 in Hn0s:
                    # Initialize matrix describing this transition event
                    SHn0 = torch.zeros(
                        S2.S.shape[0], S1.S.shape[0], dtype=S1.S.dtype, device=device
                    )

                    # Sub-pixel shift of Hn0
                    posn = atom * np.asarray(S1.stored_gridshape)
                    destination = np.remainder(posn, 1.0)
                    Hn0_ = np.fft.fftshift(
                        fourier_shift(Hn0, destination, qspacein=True)
                    ).ravel()

                    # Convert Hn0 to pytorch Tensor
                    Hn0_ = torch.from_numpy(Hn0_).to(S1.S.dtype).to(device)

                    for i, S1component in enumerate(S1.S):
                        # Multiplication of component of first scattering matrix
                        # (takes probe it depth of ionization) with the transition
                        # potential
                        Hn0S1 = Hn0_ * S1component.flatten()[windex]

                        # Matrix multiplication with second scattering matrix (takes
                        # scattered electrons to EELS detector)
                        SHn0[:, i] = S2.S[:, windex] @ Hn0S1

                    # Build a mask such that only probe positions within a PRISM
                    # cropping region about the transition are evaluated
                    scan_mask = np.logical_and(
                        (
                            np.abs((scan[..., 0].ravel() - atom[0] + 0.5) % 1.0 - 0.5)
                            <= 1 / PRISM_factor[0] / 2
                        ),
                        (
                            np.abs((scan[..., 1].ravel() - atom[1] + 0.5) % 1.0 - 0.5)
                            <= 1 / PRISM_factor[1] / 2
                        ),
                    ).ravel()

                    EELS_image[scan_mask] += torch.sum(
                        amplitude(
                            SHn0
                            @ scan_array[scan_mask, :].to(SHn0.dtype).transpose(0, 1)
                        ),
                        axis=0,
                    )

            # Reshape scattering matrix S2 for propagation
            S2.S = S2.S.reshape((S2.S.shape[0], *S2.stored_gridshape))

    # Move EELS_image to cpu and numpy and then reshape to rectangular grid
    return EELS_image.cpu().numpy().reshape(scan_shape) / nfph
