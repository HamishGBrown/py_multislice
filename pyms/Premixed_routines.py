# The py_multislice library provides users with the ability to put their
# own simulations together using the "building blocks" of py_multislice.
# For standard simulation types the premixed_routines.py functions
# allows users to set up simulations faster and easier.

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
    size_of_bandwidth_limited_array,
    crop_to_bandwidth_limit_torch,
)
from .utils.numpy_utils import fourier_shift, crop, crop_to_bandwidth_limit

# from .Ionization import make_transition_potentials
from .Probe import focused_probe, make_contrast_transfer_function


def window_indices(center, windowsize, gridshape):
    """Makes indices for a cropped window centered at center with size given
    by windowsize on a grid"""
    window = []
    for i, wind in enumerate(windowsize):
        indices = np.arange(-wind // 2, wind // 2, dtype=np.int) + wind % 2
        indices += int(round(center[i] * gridshape[i]))
        indices = np.mod(indices, gridshape[i])
        window.append(indices)

    return (window[0][:, None] * gridshape[0] + window[1][None, :]).ravel()


def STEM_PRISM(
    pix_dim,
    sample,
    thicknesses,
    eV,
    alpha,
    subslices=[1.0],
    df=0,
    aberrations=[],
    batch_size=5,
    FourD_STEM=False,
    datacube=None,
    scan_posn=None,
    dtype=torch.float32,
    device_type=None,
    tiling=[1, 1],
    PRISM_factor=[1, 1],
    seed=None,
    showProgress=True,
    detector_ranges=None,
    stored_gridshape=None,
    S=None,
    nT=5,
):
    """Perform a STEM simulation using only the PRISM algorithm"""

    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Calculate real space grid size
    real_dim = sample.unitcell[:2] * np.asarray(tiling)

    # Make the STEM probe
    probe = focused_probe(
        pix_dim, real_dim[:2], eV, alpha, df=df, aberrations=aberrations
    )

    Fourier_space_output = np.all([x == 1 for x in PRISM_factor])

    # Option to provide a precomputed scattering matrix (S), however if
    # none is given then we make our own
    if S is None:
        # Make propagators and transmission functions for multslice
        P, T = multislice_precursor(
            sample,
            pix_dim,
            eV,
            subslices=subslices,
            tiling=tiling,
            nT=nT,
            device=device,
            showProgress=showProgress,
        )

        # Convert thicknesses into number of slices for multislice
        nslices = np.ceil(thicknesses / sample.unitcell[2]).astype(np.int)

        S = scattering_matrix(
            real_dim[:2],
            P,
            T,
            nslices,
            eV,
            alpha,
            tiling=tiling,
            batch_size=batch_size,
            device=device,
            stored_gridshape=stored_gridshape,
            Fourier_space_output=Fourier_space_output,
        )

    # Make STEM detectors
    if detector_ranges is None:
        D = None
    else:
        D = np.stack(
            [
                make_detector(pix_dim, real_dim, eV, drange[1], drange[0])
                for drange in detector_ranges
            ]
        )

    # If no instructions are passed regarding size of the 4D-STEM
    # datacube size then we will have to base this on the output size
    # of the scattering matrix
    if not isinstance(FourD_STEM, (list, tuple)):
        # Generate scan positions in units of fractions of the grid if not supplied
        if scan_posn is None:
            scan_posn = generate_STEM_raster(pix_dim, real_dim[:2], eV, alpha, tiling)

        # Number of scan positions
        nscan = [x.size for x in scan_posn]

        if datacube is None:
            datacube = np.zeros([*nscan, *S.stored_gridshape], dtype=np.float32)

    return STEM(
        S.rsize,
        probe,
        S,
        [1],
        S.eV,
        alpha,
        batch_size=batch_size,
        detectors=D,
        FourD_STEM=FourD_STEM,
        datacube=datacube,
        scan_posn=scan_posn,
        device=S.device,
        tiling=S.tiling,
        showProgress=showProgress,
    )


def STEM_multislice(
    pix_dim,
    sample,
    thicknesses,
    eV,
    alpha,
    subslices=[1.0],
    df=0,
    nfph=25,
    aberrations=[],
    batch_size=5,
    FourD_STEM=False,
    scan_posn=None,
    dtype=torch.float32,
    device_type=None,
    tiling=[1, 1],
    seed=None,
    showProgress=True,
    detector_ranges=None,
    D = None,
    fractional_occupancy=True,
    nT=5,
):
    """Perform a STEM simulation using only the multislice algorithm"""

    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Make the STEM probe
    real_dim = sample.unitcell[:2] * np.asarray(tiling)
    probe = focused_probe(
        pix_dim, real_dim[:2], eV, alpha, df=df, aberrations=aberrations
    )

    # Convert thicknesses into number of slices for multislice
    nslices = np.ceil(thicknesses / sample.unitcell[2]).astype(np.int)

    # Make STEM detectors
    if detector_ranges is not None:
        D = np.stack(
            [
                make_detector(pix_dim, real_dim, eV, drange[1], drange[0])
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

    STEM_images = None
    datacube = None

    for i in tqdm.tqdm(
        range(nfph),
        desc="Frozen phonon iteration",
        disable=not showProgress,
    ):

        # Make propagators and transmission functions for multslice
        P, T = multislice_precursor(
            sample,
            pix_dim,
            eV,
            subslices=subslices,
            tiling=tiling,
            nT=nT,
            device=device,
            showProgress=showProgress,
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
            alpha,
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

    return [i/nfph for i in [STEM_images, datacube] if i is not None]


def multislice_precursor(
    sample,
    gridshape,
    eV,
    subslices=[1.0],
    tiling=[1, 1],
    nT=5,
    device=get_device(None),
    dtype=torch.float32,
    showProgress=True,
    displacements=True,
    fractional_occupancy=True,
):
    """Make transmission functions and propagators for multislice"""
    # Calculate grid size in Angstrom
    rsize = np.zeros(3)
    rsize[:3] = sample.unitcell[:3]
    rsize[:2] *= np.asarray(tiling)

    # If slices are basically equidistant, we can use the same propagator
    if np.std(np.diff(subslices, prepend=0)) < 1e-4:
        P = make_propagators(gridshape, rsize, eV, subslices[:1])[0]
    else:
        P = make_propagators(gridshape, rsize, eV, subslices)

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
    gridshape,
    sample,
    ionization_potentials,
    ionization_sites,
    thicknesses,
    eV,
    alpha,
    subslices=[1.0],
    batch_size=1,
    detectors=None,
    FourD_STEM=False,
    datacube=None,
    scan_posn=None,
    dtype=None,
    device_type=None,
    tiling=[1, 1],
    seed=None,
    showProgress=True,
    threshhold=1e-4,
    nT=5,
):
    """Perform a STEM-EELS simulation using only the multislice algorithm"""
    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Make propagators and transmission functions for multslice
    P, T = multislice_precursor(
        sample,
        gridshape,
        eV,
        subslices=subslices,
        tiling=tiling,
        nT=nT,
        device_type=device,
        showProgress=showProgress,
    )

    from . import transition_potential_multislice, tile_out_ionization_image

    rsize = sample.unitcell[:2] * np.asarray(tiling)

    # Make probe
    probe = focused_probe(
        gridshape, sample.unitcell[:2] * np.asarray(tiling), eV, alpha, qspace=True
    )

    # Convert thicknesses to multislice slices
    nslices = np.asarray(np.ceil(thicknesses / sample.unitcell[2]), dtype=np.int)

    method = transition_potential_multislice
    args = (
        nslices,
        subslices,
        P,
        T,
        ionization_potentials,
        ionization_sites,
        tiling,
        device,
        seed,
    )
    kwargs = {
        "return_numpy": False,
        "qspace_in": True,
        "qspace_out": True,
        "threshhold": threshhold,
        "showProgress": showProgress,
    }

    EELS_image = STEM(
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

    return tile_out_ionization_image(EELS_image, tiling)


def CBED(
    crystal,
    gridshape,
    eV,
    app,
    thicknesses,
    subslices=[1.0],
    device_type=None,
    tiling=[1, 1],
    nT=5,
    nfph=25,
    showProgress=True,
    probe_posn=None,
):

    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Make propagators and transmission functions for multslice
    P, T = multislice_precursor(
        crystal,
        gridshape,
        eV,
        subslices=subslices,
        tiling=tiling,
        nT=nT,
        device=device,
        showProgress=showProgress,
    )

    output = np.zeros(
        (thicknesses.shape[0], *size_of_bandwidth_limited_array(gridshape))
    )

    nslices = np.asarray(np.ceil(thicknesses / crystal.unitcell[2]), dtype=np.int)

    # Iteration over frozen phonon configurations
    for ifph in tqdm.tqdm(
        range(nfph), desc="Frozen phonon iteration", disable=not showProgress
    ):
        # Make probe
        probe = focused_probe(
            gridshape, crystal.unitcell[:2] * np.asarray(tiling), eV, app, qspace=True
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
            output[it, ...] += (
                np.abs(np.fft.fftshift(crop_to_bandwidth_limit(probe))) ** 2
            )

    # Divide output by # of pixels to compensate for Fourier transform
    return output / np.prod(gridshape)


def HRTEM(
    crystal,
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
    tilt=[0, 0],
    tilt_units="mrad",
    aberrations=[],
):
    from .Probe import plane_wave_illumination

    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Make propagators and transmission functions for multslice
    P, T = multislice_precursor(
        crystal,
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
    output = np.zeros((thicknesses.shape[0], *bw_limit_size))

    nslices = np.asarray(np.ceil(thicknesses / crystal.unitcell[2]), dtype=np.int)

    rsize = np.asarray(crystal.unitcell[:2]) * np.asarray(tiling)

    # Option for focal series, just pass an array of defocii, this bit of
    # code will set up the lens transfer functions for each case
    if np.isscalar(df):
        ctf = crop_to_bandwidth_limit_torch(
            cx_from_numpy(
                make_contrast_transfer_function(
                    gridshape, rsize, eV, app, df=df, aberrations=aberrations
                ),
                dtype=dtype,
                device=device,
            )
        )
        output = np.zeros((1, len(nslices), *bw_limit_size))
    else:
        ctf = torch.zeros(len(df), *bw_limit_size, 2, dtype=dtype, device=device)
        for idf, df_ in enumerate(df):
            ctf[idf] = crop_to_bandwidth_limit_torch(
                cx_from_numpy(
                    make_contrast_transfer_function(
                        gridshape, rsize, eV, app, df=df_, aberrations=aberrations
                    ),
                    dtype=dtype,
                    device=device,
                )
            )
        output = np.zeros((len(df), len(nslices), *bw_limit_size))

    # Iteration over frozen phonon configurations
    for ifph in tqdm.tqdm(
        range(nfph), desc="Frozen phonon iteration", disable=not showProgress
    ):
        probe = plane_wave_illumination(
            gridshape, rsize, tilt, eV, tilt_units, qspace=True
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
    return np.squeeze(output)


def STEM_EELS_PRISM(
    crystal,
    gridshape,
    eV,
    app,
    det,
    thicknesses,
    Ztarget,
    boundConfiguration,
    boundQuantumNumbers,
    freeConfiguration,
    freeQuantumNumbers,
    epsilon,
    Hn0_crop=None,
    subslices=[1.0],
    device_type=None,
    tiling=[1, 1],
    nT=5,
    PRISM_factor=[1, 1],
):

    # Choose GPU if available and CPU if not
    device = get_device(device_type)

    # Calculate grid size in Angstrom
    rsize = np.zeros(3)
    rsize[:3] = crystal.unitcell[:3]
    rsize[:2] *= np.asarray(tiling)

    # TODO enable a thickness series
    nslices = np.ceil(thicknesses / crystal.unitcell[2]).astype(np.int)
    P = make_propagators(gridshape, rsize, eV, subslices)
    T = torch.zeros(nT, len(subslices), *gridshape, 2, device=device)

    # Make the transmission functions for multislice
    for i in range(nT):
        T[i] = crystal.make_transmission_functions(gridshape, eV, subslices, tiling)

    # Get the coordinates of the target atoms in a unit cell
    mask = crystal.atoms[:, 3] == Ztarget

    # Adjust fractional coordinates for tiling of unit cell
    coords = crystal.atoms[mask][:, :3] / np.asarray(tiling + [1])

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
    )
    # Link the slices and seeds of both scattering matrices
    S1.seed = S2.seed

    from .Ionization import make_transition_potentials, tile_out_ionization_image

    nstates = len(freeQuantumNumbers)
    Hn0 = make_transition_potentials(
        gridshape,
        rsize,
        eV,
        Ztarget,
        epsilon,
        boundQuantumNumbers,
        boundConfiguration,
        freeQuantumNumbers,
        freeConfiguration,
    )

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
    EELS_image = EELS_image.cpu().numpy().reshape(len(scan[0]), len(scan[1]))
    return tile_out_ionization_image(EELS_image, tiling)
