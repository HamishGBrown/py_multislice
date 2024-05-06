"""Utility functions for working with the numpy library."""
import numpy as np
import torch
import copy


def ensure_array(input):
    """Force a (potentially scalar) input to be an array."""
    if hasattr(input, "__len__") and not isinstance(input, str):
        return input
    else:
        return np.asarray([input])


def is_array_like(input):
    return hasattr(input, "__len__")


def broadcast_from_unmeshed(coords):
    """
    For an unmeshed set of coordinates broadcast to a meshed ND array.

    Examples
    --------
    >>> broadcast_from_unmeshed([np.arange(5),np.arange(6)])
    [array([[0, 0, 0, 0, 0, 0],
       [1, 1, 1, 1, 1, 1],
       [2, 2, 2, 2, 2, 2],
       [3, 3, 3, 3, 3, 3],
       [4, 4, 4, 4, 4, 4]]), array([[0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5],
       [0, 1, 2, 3, 4, 5]])]
    """

    N = len(coords)
    pixels = [a.shape[0] for a in coords]

    # Broadcasting patterns
    R = np.ones((N, N), dtype=np.int16) + np.diag(pixels) - np.eye(N, dtype=np.int16)

    # Broadcast unmeshed grids
    return [np.broadcast_to(a.reshape(rr), pixels) for a, rr in zip(coords, R)]


def r_space_array(pixels, gridsize, meshed=True):
    """
    Return the appropriately scaled ND real space coordinates.

    Parameters
    -----------
    pixels : (N,) array_like
        Pixels in each dimension of a ND array
    gridsize : (N,) array_like
        Dimensions of the array in real space units
    meshed : bool, optional
        Option to output dense meshed grid (True) or output unbroadcasted
        arrays (False)
    """
    # N is the dimensionality of grid
    N = len(pixels)

    # Calculate unmeshed grids
    rspace = [np.fft.fftfreq(pixels[i], d=1 / gridsize[i]) for i in range(N)]

    # At this point we can return the arrays without broadcasting
    if meshed:
        return broadcast_from_unmeshed(rspace)
    else:
        return rspace


def q_space_array(pixels, gridsize, meshed=True):
    """
    Return the appropriately scaled 2D reciprocal space coordinates.

    Parameters
    -----------
    pixels : (N,) array_like
        Pixels in each dimension of a ND array
    gridsize : (N,) array_like
        Dimensions of the array in real space units
    meshed : bool, optional
        Option to output dense meshed grid (True) or output unbroadcasted
        arrays (False)

    Parameters
    -----------
    pixels : (N,) array_like
        Pixels in each dimension of a 2D array
    gridsize : (N,) array_like
        Dimensions of the array in real space units
    """
    # N is the dimensionality of grid
    N = len(pixels)

    qspace = [np.fft.fftfreq(pixels[i], d=gridsize[i] / pixels[i]) for i in range(N)]
    # At this point we can return the arrays without broadcasting
    if meshed:
        return broadcast_from_unmeshed(qspace)
    else:
        return qspace


def crop_window_to_flattened_indices(indices, shape):
    """Map  y and x indices describing a cropping window to a flattened 1d array."""
    return (
        indices[-1][None, :] % shape[-1]
        + (indices[-2][:, None] % shape[-2]) * shape[-1]
    ).ravel()


def crop_to_bandwidth_limit(
    array, limit=2 / 3, norm="conserve_norm", qspace_in=True, qspace_out=True
):
    """
    Crop an array to its bandwidth limit (ie remove superfluous array entries).

    assumes that input array is in Fourier space with zeroth Fourier component
    in upper-left corner
    """
    # New shape of final dimensions
    newshape = tuple([round(array.shape[-2 + i] * limit) for i in range(2)])

    return fourier_interpolate(
        array, newshape, norm=norm, qspace_in=qspace_in, qspace_out=qspace_out
    )


def bandwidth_limit_array(arrayin, limit=2 / 3, qspace_in=True, qspace_out=True):
    """
    Band-width limit an array in Fourier space.

    Band width limiting of the propagator and transmission functions is necessary
    in multislice to prevent "aliasing", wrapping round of high-angle scattering
    due the periodic boundary conditions implicit in the multislice algorithm.
    See sec. 6.8 of "Kirkland's Advanced Computing in Electron Microscopy" for
    more detail.

    Parameters
    ----------
    arrayin : array_like (...,Ny,Nx)
        Array to be bandwidth limited.
    limit : float
        Bandwidth limit as a fraction of the maximum reciprocal space frequency
        of the array.
    qspace_in : bool, optional
        Set to True if the input array is in reciprocal space (default),
        False if not
    qspace_out : bool, optional
        Set to True for reciprocal space output (default), False for real-space
        output.
    Returns
    -------
    array : array_like (...,Ny,Nx)
        The bandwidth limit of the array
    """
    # Transform array to real space if necessary
    if qspace_in:
        array = copy.deepcopy(arrayin)
    else:
        array = np.fft.fft2(arrayin)

    # Case where band-width limiting has been turned off
    if limit is not None:
        if is_array_like(limit):
            lmt = limit[-2:]
        else:
            lmt = limit * np.ones(2)

        shp = array.shape

        mask = (
            np.square(np.fft.fftfreq(shp[-2]) * 2 / lmt[0])[:, np.newaxis]
            + np.square(np.fft.fftfreq(shp[-1]) * 2 / lmt[1])[np.newaxis, :]
        ) >= 1
        import matplotlib.pyplot as plt

        if len(shp) > 2:
            array = array.reshape((np.prod(shp[:-2]), *shp[-2:]))
            for i in range(array.shape[0]):
                array[i][mask] = 0
            array = array.reshape(shp)
        else:
            array[mask] = 0

    if qspace_out:
        return array
    else:
        return np.fft.ifft2(array)


def Fourier_interpolation_masks(npiyin, npixin, npiyout, npixout):
    """Calculate a mask of array entries to be included in Fourier interpolation."""
    # Construct input and output fft grids
    qyin, qxin, qyout, qxout = [
        (np.fft.fftfreq(x, 1 / x)).astype(np.int32)
        for x in [npiyin, npixin, npiyout, npixout]
    ]

    # Get maximum and minimum common reciprocal space coordinates
    minqy, maxqy = [
        max(np.amin(qyin), np.amin(qyout)),
        min(np.amax(qyin), np.amax(qyout)),
    ]
    minqx, maxqx = [
        max(np.amin(qxin), np.amin(qxout)),
        min(np.amax(qxin), np.amax(qxout)),
    ]

    # Make 2d grids
    qqxout, qqyout = np.meshgrid(qxout, qyout)
    qqxin, qqyin = np.meshgrid(qxin, qyin)

    # Make a masks of common Fourier components for input and output arrays
    maskin = np.logical_and(
        np.logical_and(qqxin <= maxqx, qqxin >= minqx),
        np.logical_and(qqyin <= maxqy, qqyin >= minqy),
    )

    maskout = np.logical_and(
        np.logical_and(qqxout <= maxqx, qqxout >= minqx),
        np.logical_and(qqyout <= maxqy, qqyout >= minqy),
    )

    return maskin, maskout


def renormalize(array, oldmin=None, oldmax=None, newmax=1.0, newmin=0.0):
    """Rescales the array such that its maximum is newmax and its minimum is newmin."""
    if oldmin is not None:
        min_ = oldmin
    else:
        min_ = array.min()

    if oldmax is not None:
        max_ = oldmax
    else:
        max_ = array.max()

    return (
        np.clip((array - min_) / (max_ - min_), 0.0, 1.0) * (newmax - newmin) + newmin
    )


def convolve(array1, array2, axes=None):
    """
    Fourier convolution of two arrays over specified axes.

    array2 is broadcast to match array1 so axes refers to the dimensions of
    array1
    """
    # input and output shape
    s = array1.shape
    # Broadcast array2 to match array1
    a2 = np.broadcast_to(array2, s)
    # Axes of transformation
    a = axes
    if a is not None:
        s = [s[i] for i in a]
    if np.iscomplexobj(array1) or np.iscomplexobj(a2):
        return np.fft.ifftn(np.fft.fftn(array1, s, a) * np.fft.fftn(a2, s, a), s, a)
    else:
        return np.fft.irfftn(np.fft.rfftn(array1, s, a) * np.fft.rfftn(a2, s, a), s, a)


def colorize(z, saturation=0.8, minlightness=0.0, maxlightness=0.5):
    """
    Map a complex number to the hsl scale and output in RGB format.

    Parameters
    ----------
    z : complex, array_like
        Complex array to be plotted using hsl
    Saturation : float, optional
        (Uniform) saturation value of the hsl colormap
    minlightness, maxlightness : float, optional
        The amplitude of the complex array z will be mapped to the lightness of
        the output hsl map. These keyword arguments allow control over the range
        of lightness values in the map
    """
    from colorsys import hls_to_rgb

    # Get phase an amplitude of complex array
    r = np.abs(z)
    arg = np.angle(z)

    # Calculate hue, lightness and saturation
    h = arg / (2 * np.pi)
    ell = renormalize(r, newmin=minlightness, newmax=maxlightness)
    s = saturation

    # Convert HLS format to RGB format
    c = np.vectorize(hls_to_rgb)(h, ell, s)  # --> tuple
    # Convert to numpy array
    c = np.array(c)  # -->
    # Array has shape (3,n,m), but we need (n,m,3) for output, range needs to be
    # from 0 to 256
    c = (c.transpose(1,2,0) * 256).astype(np.uint8)
    return c


def fourier_crop(ain, shapeout):
    """ "
    Crop or pad a Fourier shifted array
    """

    def crop1d(array, s, d):
        # Number of dimensions of array
        N = len(array.shape)
        # Size of array that will be transferred to new grid
        s_ = min(array.shape[d], s)
        # Indices of grid region to transfer to new grid
        ind1 = (
            (np.s_[:],) * ((N + d) % N)
            + (np.s_[: s_ // 2 + s_ % 2],)
            + (np.s_[:],) * (N - (N + d) % N - 1)
        )
        ind2 = (
            (np.s_[:],) * ((N + d) % N)
            + (np.s_[-s_ // 2 + s_ % 2 :],)
            + (np.s_[:],) * (N - (N + d) % N - 1)
        )
        if s > array.shape[d]:
            xtra = list(array.shape)
            xtra[d] = s - array.shape[d]
            return np.concatenate(
                [array[ind1], np.zeros(xtra, dtype=array.dtype), array[ind2]], axis=d
            )
        else:
            return np.concatenate([array[ind1], array[ind2]], axis=d)

    array = copy.deepcopy(ain)
    for i, s in enumerate(shapeout):
        array = crop1d(array, s, i - len(shapeout))
    return array


def fourier_interpolate(
    ain, shapeout, norm="conserve_val", N=None, qspace_in=False, qspace_out=False
):
    """
    Perfom fourier interpolation on array ain so that its shape matches shapeout.

    Arguments
    ---------
    ain : (...,Ni,..,Ny,Nx) array_like
        Input numpy array, interpolation will be applied to the n trailing
        dimensions where n is the length of shapeout.
    shapeout : int (n,) , array_like
        Desired shape of output array
    norm : str, optional  {'conserve_val','conserve_norm','conserve_L1'}
        Normalization of output. If 'conserve_val' then array values are preserved
        if 'conserve_norm' L2 norm is conserved under interpolation and if
        'conserve_L1' L1 norm is conserved under interpolation
    N : int, optional
        Number of (trailing) dimensions to Fourier interpolate. By default take
        the length of shapeout
    qspace_in : bool, optional
        Set to True if the input array is in reciprocal space, False if not (default).
        Be careful with setting this to True for a non-complex array.
    qspace_out : bool, optional
        Set to True for reciprocal space output, False for real-space output (default).
    """
    # Import required FFT functions
    from numpy.fft import fftn

    if N is None:
        N = len(shapeout)

    # Make input complex
    aout = np.zeros(np.shape(ain)[:-N] + tuple(shapeout), dtype=complex)

    # Get input dimensions
    shapein = np.shape(ain)[-N:]

    # axes to Fourier transform
    axes = np.arange(-N, 0)

    if qspace_in:
        a = np.asarray(ain, dtype=complex)
    else:
        a = fftn(np.asarray(ain, dtype=complex), axes=axes)

    aout = fourier_crop(a, shapeout)
    # aout = np.fft.fftshift(crop(np.fft.fftshift(a,axes=axes),shapeout),axes=axes)

    # Fourier transform result with appropriate normalization
    if norm == "conserve_val":
        aout *= np.prod(shapeout) / np.prod(np.shape(ain)[-N:])
    elif norm == "conserve_norm":
        aout *= np.sqrt(np.prod(shapeout) / np.prod(np.shape(ain)[-N:]))

    if not qspace_out:
        aout = np.fft.ifftn(aout, axes=axes)

    # Return correct array data type
    if not np.iscomplexobj(ain):
        return np.real(aout)
    else:
        return aout


def oned_shift(N, shift, pixel_units=True):
    """
    Construct a one-dimensional shift array.

    Parameters
    ----------
    N     -- Number of pixels in the shift array
    shift -- Amount of shift to be achieved (default units of pixels)

    Keyword arguments
    ----------
    pixel_units -- Pass True if shift is to be units of pixels, False for
                   fraction of the array
    """
    # Create the Fourier space pixel coordinates of the shift array
    shiftarray = (np.arange(N) + N // 2) % N - N // 2

    # Conversion necessary if the shift is in units of pixels, and not fractions
    # of the array
    if pixel_units:
        shiftarray = shiftarray / N

    # The shift array is given mathematically as e^(-2pi i k Delta x) and this
    # is what is returned.
    return np.exp(-2 * np.pi * 1j * shiftarray * shift)


def fourier_shift(arrayin, shift, qspacein=False, qspaceout=False, pixel_units=True):
    """
    Shifts a 2d array using the Fourier shift theorem.

    Parameters
    ----------
    arrayin -- Array to be Fourier shifted
    shift   -- Shift in units of pixels (pass pixel_units = False for shift
               to be in units of fraction of the array size)

    Keyword arguments
    ----------
    qspacein    -- Pass True if arrayin is in Fourier space
    qspaceout   -- Pass True for Fourier space output, False (default) for
                   real space output
    pixel_units -- Pass True if shift is to be units of pixels, False for
                   fraction of the array
    """
    # Construct shift array
    shifty, shiftx = [
        oned_shift(arrayin.shape[-2 + i], shift[i], pixel_units) for i in range(2)
    ]

    # Now Fourier transform array and apply shift
    real = not np.iscomplexobj(arrayin)

    if real:
        array = np.asarray(arrayin, dtype=complex)
    else:
        array = arrayin

    if not qspacein:
        array = np.fft.fft2(array)

    array = shiftx[np.newaxis, :] * shifty[:, np.newaxis] * array

    if not qspaceout:
        array = np.fft.ifft2(array)

    if real:
        return np.real(array)
    else:
        return array


def add_noise(arrayin, Total_counts):
    """
    Add Poisson counting noise to simulated data.

    Parameters
    ----------
    arrayin : array_like
        Array giving the fraction of Total_counts that is expected at each pixel
        in the array.
    Total_counts : float
        Total number of electron counts expected over the array.
    """
    return np.random.poisson(arrayin * Total_counts)


def crop(arrayin, shapeout):
    """
    Crop the last n dimensions of arrayin to grid size shapeout.

    For entries of shapeout which are larger than the shape of the input array,
    perform zero-padding.

    Parameters
    ----------
    arrayin : (...,N1,N2,...Nn) array_like
        Array to be cropped or zero-padded.
    shapeout : (n,) array_like
        Desired output shape of the final n dimensions of arrayin
    """
    # Number of dimensions in input array
    ndim = arrayin.ndim

    # Trailing dimensions to be cropped/padded
    n = len(shapeout)

    # Number of dimensions not covered by shapeout (ie not to be cropped)
    nUntouched = ndim - n

    # Shape of output array
    shapeout_ = arrayin.shape[:nUntouched] + tuple(shapeout)

    arrayout = np.zeros(shapeout_, dtype=arrayin.dtype)

    oldshape = arrayin.shape[-n:]
    newshape = shapeout[-n:]

    def indices(y, y_):
        if y > y_:
            # Crop in y dimension
            y1, y2 = [(y - y_) // 2 + (y - y_) % 2, (y + y_) // 2 + (y - y_) % 2]
            y1_, y2_ = [0, y_]
        else:
            # Zero pad in y dimension
            y1, y2 = [0, y]
            y1_, y2_ = [(y_ - y) // 2 + (y - y_) % 2, (y + y_) // 2 + (y - y_) % 2]
        return np.index_exp[y1:y2][0], np.index_exp[y1_:y2_][0]

    ind = [indices(x, x_) for x, x_ in zip(oldshape, newshape)]
    inind, outind = map(tuple, zip(*ind))
    arrayout[nUntouched * np.index_exp[:] + outind] = arrayin[
        nUntouched * np.index_exp[:] + inind
    ]
    return arrayout


def Gaussian(sigma, gridshape, rsize, theta=0, normalized=True):
    r"""
    Calculate a normalized 2D Gaussian function.

    Notes
    -----
    Functional form
    .. math:: 1 / \sqrt { 2 \pi \sigma }  e^{ - ( x^2 + y^2 ) / 2 / \sigma^2 }

    Parameters
    ----------
    sigma : float or (2,) array_like
        The standard deviation of the Gaussian function, if an array is provided
        then the first two entries will give the y and x standard deviation of
        the Gaussian.
    gridshape : (2,) array_like
        Number of pixels in the grid.
    rsize : (2,) array_like
        Size of the grid in units of length
    theta : float, optional
        Angle of the two dimensional Gaussian function.
    """
    if isinstance(sigma, (list, tuple, np.ndarray)):
        sigmay, sigmax = sigma[:2]
    else:
        sigmax = sigma
        sigmay = sigma
    grid = r_space_array(gridshape, rsize)
    a = np.cos(theta) ** 2 / (2 * sigmax ** 2) + np.sin(theta) ** 2 / (2 * sigmay ** 2)
    b = -np.sin(2 * theta) / (4 * sigmax ** 2) + np.sin(2 * theta) / (4 * sigmay ** 2)
    c = np.sin(theta) ** 2 / (2 * sigmax ** 2) + np.cos(theta) ** 2 / (2 * sigmay ** 2)
    gaussian = np.exp(
        -(a * grid[1] ** 2 + 2 * b * grid[0] * grid[1] + c * grid[0] ** 2)
    )
    if normalized:
        gaussian /= np.sum(gaussian)
    return gaussian
