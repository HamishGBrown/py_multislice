"""Utility functions for outputting data to file."""
import os
import numpy as np
import matplotlib.pyplot as plt
import png


def complex_to_png(arrayin, fnam):
    """Output a complex array as a hsv colormap in .png format."""
    from .numpy_utils import colorize

    # Convert complex to RGB colormap and then output array to png
    RGB_to_PNG(colorize(arrayin), fnam)


def array_to_RGB(arrayin, cmap=plt.get_cmap("viridis")):
    """Convert an array to RGB using a supplied colormap."""
    from .numpy_utils import renormalize

    return (cmap(renormalize(arrayin))[..., :3] * 256).astype(np.uint8)


def RGB_to_PNG(RGB_array, fnam):
    """Output an RGB array [shape (n,m,3)] as a .png file."""
    # Get array shape
    n, m = RGB_array.shape[:2]

    # Replace filename ending with .png
    fnam_out = os.path.splitext(fnam)[0] + ".png"
    png.fromarray(RGB_array.reshape((n, m * 3)), mode="RGB").save(fnam_out)


def save_array_as_png(array, fnam, cmap=plt.get_cmap("viridis")):
    """Output a numpy array as a .png file."""
    # Convert numpy array to RGB and then output to .png file
    RGB_to_PNG(array_to_RGB(array, cmap), fnam)


def datacube_to_py4DSTEM_viewable(
    datacube,
    filename,
    Rpix=None,
    diffsize=None,
    eV=None,
    alpha=None,
    comments="STEM datacube simulated using the py-multislice package",
    sample="",
):
    """Write a 4D-STEM datacube to a py4DSTEM hdf5 file including metadata."""
    import h5py

    f = h5py.File(os.path.splitext(filename)[0] + ".h5", "w")
    f.attrs["version_major"] = 0
    f.attrs["version_minor"] = 3
    grp = f.create_group("/4DSTEM_experiment/data/diffractionslices")
    grp = f.create_group("/4DSTEM_experiment/data/realslices")
    grp = f.create_group("/4DSTEM_experiment/data/pointlists")
    grp = f.create_group("/4DSTEM_experiment/data/pointlistarrays")
    grp = f.create_group("/4DSTEM_experiment/data/datacubes")
    grp.attrs["emd_group_type"] = 1
    grp.create_dataset(
        "datacube_0/datacube", shape=datacube.shape, data=datacube, dtype=datacube.dtype
    )

    f.create_group("4D-STEM_data/metadata")
    f.create_group("4D-STEM_data/metadata/original/shortlist")
    f.create_group("4D-STEM_data/metadata/original/all")
    f.create_group("4D-STEM_data/metadata/user")
    f.create_group("4D-STEM_data/metadata/processing")
    calibration = f.create_group("4D-STEM_data/metadata/calibration")
    if Rpix is not None:
        # WARNING - A probe step size is not expected, especially if
        # using the py_multislice generate_STEM_raster function, but
        # a user should be mindful of the possibility, especially if
        # generating their own scan positions and know that the py4DSTEM
        # data structure does not make allowance for it.
        calibration.attrs.create("R_pix_size", [Rpix])
        calibration.attrs.create("R_pix_units", ["angstrom"])
    if diffsize is not None:
        # WARNING - In simulation diffraction pattern samplings that
        # differ in the x and y direction are common if a rectangular
        # system is being studied and this is not typical in a well
        # calibrated electron microscope experiment. Therefore the
        # current py4DSTEM data stucture does not make allowances for
        # non-square diffraction pattern samplings and neither does this
        # routine
        Kpix = diffsize[-1] / datacube.shape[-1]
        calibration.attrs.create("K_pix_size", [Kpix])
        calibration.attrs.create("K_pix_units", ["angstrom^-1"])
    # A simulated 4D-STEM dataset should not have any rotation between
    # scan
    calibration.attrs.create("R_to_K_rotation_degrees", ["0"])
    if eV is not None:
        calibration.attrs.create("accelerating_voltage", [eV])
    if alpha is not None:
        calibration.attrs.create("convergence_semiangle_mrad", [alpha])
    com = f.create_group("4D-STEM_data/metadata/comments")
    com.attrs.create("Note", [comments])
    f.close()


def tiff_stack_out(array, tag):
    """Output a multi-dimensional array as a set of tif files."""
    from PIL import Image

    direc = os.path.dirname(tag)
    if not os.path.exists(direc):
        os.mkdir(direc)

    shape = array.shape
    ntiffs = np.prod(shape[:-2])
    array_ = array.reshape((ntiffs, *shape[-2:]))

    for i, img in enumerate(array_):
        Image.fromarray(img).save(os.path.splitext(tag.format(i))[0] + ".tif")
