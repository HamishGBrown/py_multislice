"""Utility functions for outputting data to file."""
import os
from matplotlib import pyplot as plt
import numpy as np

# import matplotlib.pyplot as plt
import png
import h5py
from PIL import Image
import pyms

from .. import _float, _int, _uint


def stack_to_animated_gif(
    arrayin,
    fnam,
    cmap=None,
    vmin=None,
    vmax=None,
    optimize=False,
    duration=100,
    loop=0,
):
    """
    Write a numpy array to an animated gif.

    Parameters
    ----------
    arrayin : float or int, array_like (...,Y,X)
        The array to convert to an animated gif, the leading dimensions of the
        array will be the different frames of the output gif
    fnam : string
        Output filename of the gif. The filename ending will be removed and
        .gif added
    cmap : matplotlib.cmap function, optional
        A function that takes an input from 0 to 1 and converts it to a (3,)
        RGB colour
    vmin : float, optional
        `vmin` and `vmax` define the data range that the colormap covers. By
        default, the colormap covers the complete value range of the supplied
        data.
    vmax : float,optional
        See `vmin` description
    optimize : bool, optional
        Tells the Python image library whether to compress the gif output or not
    duration : int, optional
        Duration of each frame in milliseconds
    loop : int, optional
        Number of times to loop the gif, 0 means infinite loop and -1 means that
        gif animation is played a single time
    """
    if cmap is None:
        cmap = plt.get_cmap("viridis")

    # Flatten dimensions leading up to the final two dimensions
    shapein = arrayin.shape
    nimgs = np.prod(shapein[:-2])
    array_ = arrayin.reshape((nimgs, *shapein[-2:]))

    # Replace filename ending with .gif
    fnam_out = os.path.splitext(fnam)[0] + ".gif"

    # Get max and min for colormap scaling
    if vmin is None:
        vmin = arrayin.min()
    if vmax is None:
        vmax = arrayin.max()

    # Convert image to correct format
    rgbstack = [Image.fromarray(array_to_RGB(x, cmap, vmin, vmax)) for x in array_]

    # save frames as individual gifs (work around since saving a stack using PIL
    # leads to unusual results)
    for i, frame in enumerate(rgbstack):
        frame.save("{0}.gif".format(i))

    # Write individual frames to animated gifs
    Image.open("0.gif").save(
        fnam_out,
        save_all=True,
        append_images=[
            Image.open("{0}.gif".format(i)) for i in range(1, len(rgbstack))
        ],
        optimize=optimize,
        duration=duration,
        loop=loop,
    )

    # Clean up individual gifs
    for i in range(len(rgbstack)):
        os.remove("{0}.gif".format(i))


def complex_to_png(arrayin, fnam):
    """Output a complex array as a hsv colormap in .png format."""
    from .numpy_utils import colorize

    # Convert complex to RGB colormap and then output array to png
    RGB_to_PNG(colorize(arrayin), fnam)


def array_to_RGB(arrayin, cmap=plt.get_cmap("viridis"), vmin=None, vmax=None):
    """Convert an array to RGB using a supplied colormap."""
    from .numpy_utils import renormalize

    kwargs = {"oldmin": vmin, "oldmax": vmax}
    return (cmap(renormalize(arrayin, **kwargs))[..., :3] * 256).astype(np.uint8)


def RGB_to_PNG(RGB_array, fnam):
    """Output an RGB array [shape (n,m,3)] as a .png file."""
    # Get array shape
    n, m = RGB_array.shape[:2]

    # Replace filename ending with .png
    fnam_out = os.path.splitext(fnam)[0] + ".png"
    png.fromarray(RGB_array.reshape((n, m * 3)), mode="RGB").save(fnam_out)


def save_array_as_png(array, fnam, cmap=plt.get_cmap("viridis"), vmin=None, vmax=None):
    """Output a numpy array as a .png file."""
    # Convert numpy array to RGB and then output to .png file
    RGB_to_PNG(array_to_RGB(array, cmap, vmin=vmin, vmax=vmax), fnam)


def initialize_h5_datacube_object(
    datacube_shape,
    filename,
    dtype=_float,
    Rpix=None,
    diffsize=None,
    eV=None,
    alpha=None,
    comments="STEM datacube simulated using the py-multislice package",
    sample="",
):
    """
    Initialize a py4DSTEM compatible hdf5 file to write a 4D-STEM datacube to.

    Returns
    -------
    dcube : h5py.dataset object
        The Datacube to write to, shape and datatype will be specified by inputs
        datacube_shape and dtype.
    f : h5py.File object
        The hdf5 file object, close when writing is finished
    """
    f = h5py.File(os.path.splitext(filename)[0] + ".h5", "w")
    f.attrs["version_major"] = 0
    f.attrs["version_minor"] = 3
    grp = f.create_group("/4DSTEM_experiment/data/diffractionslices")
    grp = f.create_group("/4DSTEM_experiment/data/realslices")
    grp = f.create_group("/4DSTEM_experiment/data/pointlists")
    grp = f.create_group("/4DSTEM_experiment/data/pointlistarrays")
    grp = f.create_group("/4DSTEM_experiment/data/datacubes")
    dcube = grp.create_dataset("datacube_0/datacube", shape=datacube_shape, dtype=dtype)
    grp.attrs["emd_group_type"] = 1
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
        Kpix = diffsize[-1] / datacube_shape[-1]
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
    return dcube, f


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
    dcube, f = initialize_h5_datacube_object(
        datacube.shape,
        filename,
        dtype=datacube.dtype,
        Rpix=Rpix,
        diffsize=diffsize,
        eV=eV,
        alpha=alpha,
        comments=comments,
        sample=sample,
    )

    dcube[:] = datacube[:]
    f.close()


def tiff_stack_out(array, tag):
    """
    Output a multi-dimensional array as a set of tif files in a directory.

    This directory can then be dropped into the FIJI (Image J) program to view
    as a stack.

    Parameters
    ----------
    array : (...,Y,X) np.ndarray
        Array to be written to tiff stack.
    tag : str
        A string that describes the output format of the files should be of the
        form 'directory/to/output/files/to/file_{0}' where {0} will be replaced
        with the index of the image within the stack.
    """
    from PIL import Image

    direc = os.path.dirname(tag)
    if not os.path.exists(direc):
        os.mkdir(direc)

    shape = array.shape
    ntiffs = np.prod(shape[:-2])
    array_ = array.reshape((ntiffs, *shape[-2:]))

    for i, img in enumerate(array_):
        Image.fromarray(img).save(os.path.splitext(tag.format(i))[0] + ".tif")
