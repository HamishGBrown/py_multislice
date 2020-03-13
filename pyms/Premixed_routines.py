"""
Premixed routines for simulation of some standard TEM techniques.

The py_multislice library provides users with the ability to put their
own simulations together using the "building blocks" of py_multislice.
For standard simulation types the premixed_routines.py functions
allows users to set up simulations faster and easier.
"""
import numpy as np
import torch
import tqdm
from .py_multislice import (
    make_propagators,
    scattering_matrix,
    generate_STEM_raster,
    make_detector,
    multislice,
    STEM,
)
from .utils.torch_utils import (
    cx_from_numpy,
    amplitude,
    complex_matmul,
    complex_mul,
    get_device,
    crop_to_bandwidth_limit_torch,
    size_of_bandwidth_limited_array,
)
from .utils.numpy_utils import (
    fourier_shift,
    crop,
    crop_to_bandwidth_limit,
    ensure_array,
)

from .Ionization import get_transitions, tile_out_ionization_image
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
        indices = np.arange(-wind // 2, wind // 2, dtype=np.int) + wind % 2
        indices += int(round(center[i] * gridshape[i]))
        indices = np.mod(indices, gridshape[i])
        window.append(indices)

    return (window[0][:, None] * gridshape[0] + window[1][None, :]).ravel()


def STEM_PRISM(
    structure,
    gridshape,
    eV,
    app,
    thicknesses,
    subslices=[1.0],
    tiling=[1, 1],
    PRISM_factor=[1, 1],
    df=0,
    nfph=1,
    aberrations=[],
    batch_size=5,
    FourD_STEM=False,
    datacube=None,
    scan_posn=None,
    dtype=torch.float32,
    device_type=None,
    showProgress=True,
    detector_ranges=None,
    stored_gridshape=None,
    S=None,
    nT=5,
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
    structure : pyms.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts
    app : float
        Objective aperture in mrad
    thicknesses : float or array_like
        Thickness of the calculation

    Keyword arguments
    -----------------
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
    df : float
        Probe defocus
    aberrations : list, optional
        A list containing a set of the class aberration, pass an empty list for
        an unaberrated probe.
    batch_size : int, optional
        The multislice algorithm can be performed on multiple scattering matrix
        columns at once to parrallelize computation, this number is set by
        batch_size.
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
    scan_posn :  tuple, optional
        Tuple containing arrays of y and x scan positions, overrides internal
        calculations of STEM sampling
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit floating point
    device_type : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    showProgress : bool, optional
        Pass False to disable progress readout
    stored_gridshape : int, array_like
        The size of the stored array in the scattering matrix
    detector_ranges : array_like, optional
        The acceptance angles of each of the stem detectors should be stored as
        a list of lists: ie [[0,10],[10,20]] will make two detectors spanning
        0 to 10 mrad and 10 to 20 mrad.
    D : (ndet x Y x X) array_like, optional
        A precomputed set of STEM detectors, overrides detector_ranges.
    S : (nbeams x Y x X)
        A precalculated scattering matrix (useful for calculating defocus series)
        if this is provided then nfph will be ignored since a new scattering
        matrix should be calculated for each frozen phonon iteration
    nT : int, optional
        Number of transmission functions generated to and then selected from
        in the frozen phonon algorithm
    """
    # Choose GPU if available and CPU if not
    device = get_device(device_type)

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

    # Make the STEM probe
    probe = focused_probe(
        gridshape, real_dim[:2], eV, app, df=df, aberrations=aberrations
    )

    Fourier_space_output = np.all([x == 1 for x in PRISM_factor])

    # Make STEM detectors
    maxq = 0
    if detector_ranges is None:
        D = None
    else:
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
                convert_tilt_angles(detector_ranges, "mrad", real_dim, eV, True), axis=0
            )[1],
        )

    # If no instructions are passed regarding size of the 4D-STEM
    # datacube size then we will have to base this on the output size
    # of the scattering matrix
    if FourD_STEM and isinstance(FourD_STEM, (list, tuple)):
        if len(FourD_STEM) > 1:
            maxq = max(maxq, max(FourD_STEM[1]))
        else:
            maxq = max(maxq, max(FourD_STEM[0]) / real_dim[np.argmax(FourD_STEM[0])])
    else:
        from .py_multislice import max_grid_resolution

        maxq = max(maxq, max_grid_resolution(gridshape, real_dim))

    # Calculations can be expedited by only storing the scattering matrix out
    # to the maximum scattering angle of interest here we work out if the
    # calculation would benefit from this
    maxpix = np.minimum(
        (maxq * np.asarray(real_dim)).astype(np.int),
        size_of_bandwidth_limited_array(gridshape),
    )

    for i in tqdm.tqdm(
        range(nfph_),
        desc="Frozen phonon iteration",
        disable=not showProgress or nfph_ < 2,
    ):
        # Option to provide a precomputed scattering matrix (S), however if
        # none is given then we make our own
        if not S_provided:
            print(structure, gridshape, eV, subslices, tiling, nT, device)
            # Make propagators and transmission functions for multslice
            P, T = multislice_precursor(
                structure,
                gridshape,
                eV,
                subslices=subslices,
                tiling=tiling,
                nT=nT,
                device=device,
                showProgress=showProgress,
            )

            # Convert thicknesses into number of slices for multislice
            nslices = np.ceil(thicknesses / structure.unitcell[2]).astype(np.int)

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
            datacube=datacube,
            scan_posn=scan_posn,
            device=S.device,
            tiling=S.tiling,
            showProgress=showProgress,
        )

        # Retrieve results from STEM routine
        if (D is not None) and FourD_STEM:
            STEM_images, datacube = result
        elif D is not None:
            STEM_images = result[0]
        elif FourD_STEM:
            datacube = result[0]

    return [i / nfph for i in [STEM_images, datacube] if i is not None]


def STEM_multislice(
    structure,
    gridshape,
    eV,
    app,
    thicknesses,
    subslices=[1.0],
    df=0,
    nfph=25,
    aberrations=[],
    batch_size=5,
    FourD_STEM=False,
    STEM_images=None,
    scan_posn=None,
    dtype=torch.float32,
    device_type=None,
    tiling=[1, 1],
    seed=None,
    showProgress=True,
    detector_ranges=None,
    D=None,
    fractional_occupancy=True,
    nT=5,
):
    """
    Perform a STEM simulation using only the multislice algorithm.

    Parameters
    ----------
    structure : pyms.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts
    app : float
        Objective aperture in mrad
    thicknesses : float or array_like
        Thickness of the calculation

    Keyword arguments
    -----------------
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
        Number of transmission functions generated to and then selected from
        in the frozen phonon algorithm
    contr : float, optional
        A threshhold for inclusion of ionization transitions within the
        calculation, if contr = 1.0 all ionization transitions will be inlcuded
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
    showProgress : bool, optional
        Pass False to disable progress readout
    detector_ranges : array_like, optional
        The acceptance angles of each of the stem detectors should be stored as
        a list of lists: ie [[0,10],[10,20]] will make two detectors spanning
        0 to 10 mrad and 10 to 20 mrad.
    D : (ndet x Y x X) array_like, optional
        A precomputed set of STEM detectors, overrides detector_ranges.
    """
    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Make the STEM probe
    real_dim = structure.unitcell[:2] * np.asarray(tiling)
    probe = focused_probe(
        gridshape, real_dim[:2], eV, app, df=df, aberrations=aberrations
    )

    # Convert thicknesses into number of slices for multislice
    nslices = np.ceil(thicknesses / structure.unitcell[2]).astype(np.int)

    # Make STEM detectors
    if detector_ranges is not None:
        D = np.stack(
            [
                make_detector(gridshape, real_dim, eV, drange[1], drange[0])
                for drange in detector_ranges
            ]
        )

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
    }

    datacube = None

    for i in tqdm.tqdm(
        range(nfph), desc="Frozen phonon iteration", disable=not showProgress
    ):

        # Make propagators and transmission functions for multslice
        P, T = multislice_precursor(
            structure,
            gridshape,
            eV,
            subslices=subslices,
            tiling=tiling,
            nT=nT,
            device=device,
            showProgress=False,
            fractional_occupancy=fractional_occupancy,
        )

        # Put new transmission functions and propagators into arguments
        args = (P, T, tiling, device, seed)

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
            scan_posn=scan_posn,
            device=device,
            tiling=tiling,
            seed=seed,
            showProgress=showProgress,
            method_args=args,
            method_kwargs=kwargs,
            datacube=datacube,
            STEM_image=STEM_images,
        )

        # Retrieve results from STEM routine
        if (D is not None) and FourD_STEM:
            STEM_images, datacube = result
        elif D is not None:
            STEM_images = result[0]
        elif FourD_STEM:
            datacube = result[0]

    return [i / nfph for i in [STEM_images, datacube] if i is not None]


def multislice_precursor(
    sample,
    gridshape,
    eV,
    subslices=[1.0],
    tiling=[1, 1],
    specimen_tilt=[0, 0],
    tilt_units="mrad",
    nT=5,
    device=get_device(None),
    dtype=torch.float32,
    showProgress=True,
    displacements=True,
    fractional_occupancy=True,
):
    """
    Make transmission functions and propagators for the multislice algorithm.

    Parameters
    ---------
    structure : pyms.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts

    Keyword arguments
    -----------------
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
        Number of transmission functions generated to and then selected from
        in the frozen phonon algorithm
    device : torch.device, optional
        torch.device object which will determine which device (CPU or GPU) the
        calculations will run on
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit floating point
    showProgress : bool, optional
        Pass False to disable progress readout
    displacements : bool, optional
        Pass False to disable thermal vibration of the atoms
    fractional_occupancy : bool, optional
        Pass False to fractional occupancy of atomic sites
    """
    # Calculate grid size in Angstrom
    rsize = np.zeros(3)
    rsize[:3] = sample.unitcell[:3]
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
        )[0]
    else:
        P = make_propagators(
            gridshape, rsize, eV, subslices, tilt=specimen_tilt, tilt_units=tilt_units
        )

    T = torch.zeros(nT, len(subslices), *gridshape, 2, device=device, dtype=dtype)

    for i in tqdm.tqdm(
        range(nT), desc="Making projected potentials", disable=not showProgress
    ):
        T[i] = sample.make_transmission_functions(
            gridshape,
            eV,
            subslices=subslices,
            tiling=tiling,
            device=device,
            dtype=dtype,
            displacements=displacements,
            fractional_occupancy=fractional_occupancy,
        )

    return P, T


def STEM_EELS_multislice(
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
    df=0,
    subslices=[1.0],
    device_type=None,
    tiling=[1, 1],
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
):
    """
    Perform a STEM-EELS simulation using only the multislice algorithm.

    Parameters
    ----------
    structure : pyms.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts
    app : float
        Objective aperture in mrad
    det : float
        EELS aperture acceptance angle in mrad
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

    Keyword arguments
    -----------------
    df : float
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
        Number of transmission functions generated to and then selected from
        in the frozen phonon algorithm
    contr : float, optional
        A threshhold for inclusion of ionization transitions within the
        calculation, if contr = 1.0 all ionization transitions will be inlcuded
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
    showProgress : bool, optional
        Pass False to disable progress readout
    """
    from .Ionization import transition_potential_multislice

    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Get gridsize in Angstrom
    rsize = structure.unitcell[:2] * np.asarray(tiling)

    # Convert thicknesses to multislice slices
    nslices = np.ceil(ensure_array(thicknesses) / structure.unitcell[2]).astype(np.int)

    # Make propagators and transmission functions for multslice
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
        beam_tilt=beam_tilt,
        tilt_units=tilt_units,
        aberrations=aberrations,
    )
    # Generate ionization transition potentials for atom of interest
    Hn0 = get_transitions(
        Ztarget, n, ell, epsilon, eV, gridshape, rsize, order=1, contr=contr
    )

    # Make EELS aperture masks
    detector_ranges = ensure_array(det)
    D = np.stack(
        [make_detector(gridshape, rsize, eV, drange, 0) for drange in detector_ranges]
    )

    # The method pyms.Ionization.transition_potential_multislice will be passed
    # to the STEM routine, make a list of the function arguments and a
    # dictionary of the keyword arguments to also pass to the STEM routine
    args = (subslices, P, T, Hn0, ionization_sites)
    kwargs = {
        "tiling": tiling,
        "device_type": device,
        "return_numpy": False,
        "qspace_in": True,
        "threshhold": ionization_cutoff,
        "showProgress": showProgress,
    }

    # Call STEM routine
    return STEM(
        rsize,
        probe,
        transition_potential_multislice,
        nslices,
        eV,
        app,
        batch_size=1,
        detectors=D,
        device=device,
        tiling=tiling,
        showProgress=showProgress,
        method_args=args,
        method_kwargs=kwargs,
    )[0]


def CBED(
    structure,
    gridshape,
    eV,
    app,
    thicknesses,
    subslices=[1.0],
    device_type=None,
    dtype=torch.float32,
    tiling=[1, 1],
    nT=5,
    nfph=25,
    showProgress=True,
    beam_tilt=[0, 0],
    specimen_tilt=[0, 0],
    df=0,
    tilt_units="mrad",
    aberrations=[],
    probe_posn=None,
):
    """
    Perform a convergent-beam electron diffraction (CBED) simulation.

    This is the diffraction pattern formed by a focused probe. This can also
    be used to simulate electron diffraction patterns from a plane wave source
    if a small enough aperture (app) is selected

    Parameters
    ----------
    structure : pyms.structure
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

    Keyword arguments
    -----------------
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
        Number of transmission functions generated to and then selected from
        in the frozen phonon algorithm
    nfph : int, optional
        Number of iterations of the frozen phonon algorithm (25 by default)
    showProgress : bool, optional
        Pass False to disable progress readout
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
    probe_posn : array_like, optional
        Probe position as a fraction of the unit-cell, by default the probe will
        be in the upper-left corner of the simulation grid. This is the [0,0]
        coordinate.
    """
    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Make propagators and transmission functions for multslice
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

    t_ = ensure_array(thicknesses)

    output = np.zeros((t_.shape[0], *size_of_bandwidth_limited_array(gridshape)))

    nslices = np.asarray(np.ceil(t_ / structure.unitcell[2]), dtype=np.int)

    # Iteration over frozen phonon configurations
    for ifph in tqdm.tqdm(
        range(nfph), desc="Frozen phonon iteration", disable=not showProgress
    ):
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
            )
            output[it] += np.abs(np.fft.fftshift(crop_to_bandwidth_limit(probe))) ** 2

    # Divide output by # of pixels to compensate for Fourier transform
    return output / np.prod(gridshape)


def HRTEM(
    structure,
    gridshape,
    eV,
    app,
    thicknesses,
    subslices=[1.0],
    df=0,
    device_type=None,
    dtype=torch.float32,
    tiling=[1, 1],
    nT=5,
    nfph=25,
    showProgress=True,
    beam_tilt=[0, 0],
    specimen_tilt=[0, 0],
    tilt_units="mrad",
    aberrations=[],
    P=None,
    T=None,
):
    """
    Perform a high-resolution transmission electron microscopy (HRTEM) simulation.

    This assumes plane wave illumination and a post-specimen image forming lens

    Parameters
    ----------
    structure : pyms.structure
        The structure of interest
    gridshape : (2,) array_like
        Pixel size of the simulation grid
    eV : float
        Probe energy in electron volts
    app : float
        Image-forming lens aperture in mrad, pass None for aperture-less imaging
    thicknesses : float or array_like
    det : float
        EELS aperture acceptance angle in mrad
    thicknesses : float or array_like
        Thickness of the object in the calculation, will be rounded off to the
        nearest unit cell.

    Keyword arguments
    -----------------
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
        Number of transmission functions generated to and then selected from
        in the frozen phonon algorithm
    nfph : int, optional
        Number of iterations of the frozen phonon algorithm (25 by default)
    showProgress : bool, optional
        Pass False to disable progress readout
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
    P : (n,Y,X) array_like, optional
        Precomputed Fresnel free-space propagators
    T : (n,Y,X) array_like
        Precomputed transmission functions
    """
    # Choose GPU if available and CPU if not
    device = get_device(device_type)

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

    nslices = np.asarray(np.ceil(t_ / structure.unitcell[2]), dtype=np.int)

    rsize = np.asarray(structure.unitcell[:2]) * np.asarray(tiling)

    # Option for focal series, just pass an array of defocii, this bit of
    # code will set up the lens transfer functions for each case
    defocii = ensure_array(df)

    ctf = cx_from_numpy(
        np.stack(
            [
                make_contrast_transfer_function(
                    bw_limit_size, rsize, eV, app, df=df_, aberrations=aberrations
                )
                for df_ in defocii
            ]
        ),
        dtype=dtype,
        device=device,
    )
    output = np.zeros((len(defocii), len(nslices), *bw_limit_size))

    # Iteration over frozen phonon configurations
    for ifph in tqdm.tqdm(
        range(nfph), desc="Frozen phonon iteration", disable=not showProgress
    ):
        probe = plane_wave_illumination(
            gridshape, rsize, beam_tilt, eV, tilt_units, qspace=True
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
                    torch.ifft(
                        complex_mul(ctf, crop_to_bandwidth_limit_torch(probe)),
                        signal_ndim=2,
                    )
                )
                .cpu()
                .numpy()
            )

    # Divide output by # of pixels to compensate for Fourier transform
    return np.squeeze(output) / nfph


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
    subslices=[1.0],
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
    ionization_cutoff=1e-4,
):
    """
    Perform an elemental mapping energy-filtered TEM (EFTEM) simulation.

    Parameters
    ----------
    structure : pyms.structure
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

    Keyword arguments
    -----------------
    df : float
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
        Number of transmission functions generated to and then selected from
        in the frozen phonon algorithm
    contr : float, optional
        A threshhold for inclusion of ionization transitions within the
        calculation, if contr = 1.0 all ionization transitions will be inlcuded
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
    showProgress : bool, optional
        Pass False to disable progress readout
    ionization_cutoff : float
        Threshhold below which the contribution of certain ionizations will be
        ignored.
    """
    from .Ionization import transition_potential_multislice

    # Calculate grid size in Angstrom
    rsize = np.zeros(3)
    rsize[:3] = structure.unitcell[:3]
    rsize[:2] *= np.asarray(tiling)

    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Force defocus to be an array
    defocii = ensure_array(df)

    # Construct our (plane wave) illuminating probe
    probe = plane_wave_illumination(gridshape, rsize, beam_tilt, eV, tilt_units)

    # Calculate the size of the grid after band-width limiting
    bw_limit_size = size_of_bandwidth_limited_array(gridshape)

    # Calculate lens contrast transfer functions
    ctfs = cx_from_numpy(
        np.stack(
            [
                make_contrast_transfer_function(
                    bw_limit_size, rsize, eV, app, df=df_, aberrations=aberrations
                )
                for df_ in defocii
            ],
            axis=0,
        ),
        dtype=dtype,
        device=device,
    )

    # Calculate propagator and transmission functions for multislice
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

    # Convert thicknesses to number of unit cells
    nslices = np.ceil(thicknesses / structure.unitcell[2]).astype(np.int)

    # Get the coordinates of the target atoms in a unit cell
    mask = structure.atoms[:, 3] == Ztarget

    # Adjust fractional coordinates for tiling of unit cell
    coords = structure.atoms[mask][:, :3] / np.asarray(tiling + [1])

    # nstates = len(freeQuantumNumbers)
    Hn0 = get_transitions(
        Ztarget, n, ell, epsilon, eV, gridshape, rsize, order=1, contr=0.99
    )

    # Perform multislice simulation and return the tiled out result
    return tile_out_ionization_image(
        transition_potential_multislice(
            probe,
            nslices,
            subslices,
            P,
            T,
            Hn0,
            coords,
            ctfs=ctfs,
            tiling=[1, 1],
            device_type=device,
            seed=None,
            return_numpy=True,
            qspace_in=False,
            threshhold=ionization_cutoff,
            showProgress=showProgress,
        ),
        tiling,
    )


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
    # TODO implement defocus, this is not yet included
    df=0,
    Hn0_crop=None,
    subslices=[1.0],
    device_type=None,
    tiling=[1, 1],
    nT=5,
    PRISM_factor=[1, 1],
    contr=0.95,
    dtype=torch.float32,
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
    structure : pyms.structure
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

    Keyword arguments
    -----------------
    df : float
        Probe defocus
    Hn0_crop : (2,) int array_like
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
        Number of transmission functions generated to and then selected from
        in the frozen phonon algorithm
    PRISM_factor : int (2,) array_like, optional
        The PRISM "interpolation factor" this is the amount by which the
        scattering matrices are cropped in real space to speed up
        calculations see Ophus, Colin. "A fast image simulation algorithm
        for scanning transmission electron microscopy." Advanced structural
        and chemical imaging 3.1 (2017): 13 for details on this.
    contr : float, optional
        A threshhold for inclusion of ionization transitions within the
        calculation, if contr = 1.0 all ionization transitions will be inlcuded
        in the simulation otherwise only the transitions that make up a
        fraction equal to contr of the total ionization transitions will be
        included
    dtype : torch.dtype, optional
        Datatype of the simulation arrays, by default 32-bit floating point
    """
    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Calculate grid size in Angstrom
    rsize = np.zeros(3)
    rsize[:3] = structure.unitcell[:3]
    rsize[:2] *= np.asarray(tiling)

    # TODO enable a thickness series
    nslices = np.ceil(thicknesses / structure.unitcell[2]).astype(np.int)
    P = make_propagators(gridshape, rsize, eV, subslices)
    T = torch.zeros(nT, len(subslices), *gridshape, 2, device=device)

    # Make the transmission functions for multislice
    for i in range(nT):
        T[i] = structure.make_transmission_functions(gridshape, eV, subslices, tiling)

    # Get the coordinates of the target atoms in a unit cell
    tiled_structure = structure.tile(*tiling)
    coords = tiled_structure.atoms[tiled_structure.atoms[:, 3] == Ztarget][:, :3]

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
        dtype=dtype,
    )

    # Scattering matrix 2 propagates probe from slice of ionization to exit surface
    S2 = scattering_matrix(
        rsize,
        P,
        T,
        nslices * len(subslices),
        eV,
        app,
        batch_size=5,
        subslicing=True,
        transposed=True,
        PRISM_factor=PRISM_factor,
        dtype=dtype,
    )
    # Link the slices and seeds of both scattering matrices
    S1.seed = S2.seed

    # Make ionization transition potentials
    Hn0 = get_transitions(
        Ztarget, n, ell, epsilon, eV, S1.stored_gridshape, rsize, order=1, contr=0.99
    )
    nstates = Hn0.shape[0]

    if Hn0_crop is None:
        Hn0_crop = [S1.stored_gridshape[i] for i in range(2)]
    else:
        Hn0_crop = [min(Hn0_crop[i], S1.stored_gridshape[i]) for i in range(2)]
        Hn0 = np.fft.fft2(
            np.fft.ifftshift(
                crop(np.fft.fftshift(Hn0, axes=[-2, -1]), Hn0_crop), axes=[-2, -1]
            )
        )

    # Make probe wavefunction vectors for scan
    # Get kspace grid in units of inverse pixels
    ky, kx = [
        np.fft.fftfreq(gridshape[-2 + i], d=1 / gridshape[-2 + i]) for i in range(2)
    ]

    # Generate scan positions in pixels
    scan = generate_STEM_raster(S1.S.shape[-2:], rsize[:2], eV, app)
    nprobe_posn = len(scan[0]) * len(scan[1])

    scan_array = np.zeros((nprobe_posn, S1.S.shape[0]), dtype=np.complex)

    # TODO implement aberrations and defocii
    for i in range(S1.nbeams):
        scan_array[:, i] = (
            np.exp(-2 * np.pi * 1j * ky[S1.beams[i, 0]] * scan[0])[:, None]
            * np.exp(-2 * np.pi * 1j * kx[S1.beams[i, 1]] * scan[1])[None, :]
        ).ravel()

    scan_array = cx_from_numpy(scan_array, dtype=S1.dtype, device=device)

    # Initialize Image
    EELS_image = torch.zeros(len(scan[0]) * len(scan[1]), dtype=S1.dtype, device=device)

    total_slices = nslices * len(subslices)
    for islice in tqdm.tqdm(range(total_slices), desc="Slice"):

        # Propagate scattering matrices to this slice
        if islice > 0:
            S1.Propagate(
                islice, P, T, subslicing=True, batch_size=5, showProgress=False
            )
            S2.Propagate(
                total_slices - islice,
                P,
                T,
                subslicing=True,
                batch_size=5,
                showProgress=False,
            )

        # Flatten indices of second scattering matrix in preparation for
        # indexing
        S2.S = S2.S.reshape(S2.S.shape[0], np.product(S2.stored_gridshape), 2)

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
        for atom in tqdm.tqdm(atomsinslice, "Transitions in slice"):

            windex = torch.from_numpy(
                window_indices(atom, Hn0_crop, S1.stored_gridshape)
            )

            for i in range(nstates):
                # Initialize matrix describing this transition event
                SHn0 = torch.zeros(
                    S1.S.shape[0], S2.S.shape[0], 2, dtype=S1.S.dtype, device=device
                )

                # Sub-pixel shift of Hn0
                posn = atom * np.asarray(gridshape)
                destination = np.remainder(posn, 1.0)
                Hn0_ = np.fft.fftshift(
                    fourier_shift(Hn0[i], destination, qspacein=True)
                ).ravel()

                # Convert Hn0 to pytorch Tensor
                Hn0_ = cx_from_numpy(Hn0_, dtype=S1.S.dtype, device=device)

                for i, S1component in enumerate(S1.S):
                    # Multiplication of component of first scattering matrix
                    # (takes probe it depth of ionization) with the transition
                    # potential
                    Hn0S1 = complex_mul(
                        Hn0_, S1component.flatten(end_dim=-2)[windex, :]
                    )

                    # Matrix multiplication with second scattering matrix (takes
                    # scattered electrons to EELS detector)
                    SHn0[i] = complex_matmul(S2.S[:, windex], Hn0S1)

                # Build a mask such that only probe positions within a PRISM
                # cropping region about the probe are evaluated
                scan_mask = np.logical_and(
                    (
                        np.abs((atom[0] - scan[0] + 0.5) % 1.0 - 0.5)
                        <= 1 / PRISM_factor[0] / 2
                    )[:, None],
                    (
                        np.abs((atom[1] - scan[1] + 0.5) % 1.0 - 0.5)
                        <= 1 / PRISM_factor[1] / 2
                    )[None, :],
                ).ravel()

                EELS_image[scan_mask] += torch.sum(
                    amplitude(complex_matmul(scan_array[scan_mask], SHn0)), axis=1
                )

        # Reshape scattering matrix S2 for propagation
        S2.S = S2.S.reshape((S2.S.shape[0], *S2.stored_gridshape, 2))

    # Move EELS_image to cpu and numpy and then reshape to rectangular grid
    return EELS_image.cpu().numpy().reshape(len(scan[0]), len(scan[1]))
