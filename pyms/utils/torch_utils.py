"""A set of utility functions for working with pytorch tensors."""
import torch
import numpy as np
from itertools import product
import copy
from .numpy_utils import is_array_like, q_space_array


re = np.s_[..., 0]
im = np.s_[..., 1]


def complex_dtype_to_real(dtype):
    a = torch.ones(1, dtype=dtype)
    if torch.is_complex(a):
        return a.real.dtype
    else:
        return dtype


# def numpy_real_dtype_to_complex(dtype):
#     a = np.ones(1,dtype=dtype)


def iscomplex(a: torch.Tensor):
    """Return True if a is complex, False otherwise."""
    return torch.is_complex(a)


def check_complex(A):
    """Raise a RuntimeWarning if tensor A is not complex."""
    for a in A:
        if not iscomplex(a):
            raise RuntimeWarning(
                "taking complex_mul of non-complex tensor! a.shape " + str(a.shape)
            )


def get_device(device_type=None):
    """Initialize device cuda if available, CPU if no cuda is available."""
    if device_type is None and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_type is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device_type)
    return device


def sinc(x):
    """Calculate the sinc function ie. sin(pi x)/(pi x)."""
    y = torch.where(torch.abs(x) < 1.0e-20, torch.tensor([1.0e-20], dtype=x.dtype), x)
    return torch.sin(np.pi * y) / np.pi / y


def ensure_torch_array(array, dtype=None, device=None):
    """
    Ensure that the input array is a pytorch tensor.

    Converts to a pytorch array if input is a numpy array and do nothing if the
    input is a pytorch tensor
    """
    from .. import (
        layered_structure_propagators,
        layered_structure_transmission_function,
    )

    if device is None:
        dv = get_device(device)

    if isinstance(array, torch.Tensor):
        return array.to(device)
    elif isinstance(array, layered_structure_transmission_function):
        for i in range(len(array.Ts)):
            array.Ts[i] = array.Ts[i].to(device)
        return array
    elif isinstance(array, layered_structure_propagators):
        for i in range(len(array.Ps)):
            array.Ps[i] = array.Ps[i].to(device)
        return array
    else:
        arrayout = torch.from_numpy(np.asarray(array)).to(device)
        if dtype is None:
            return arrayout
        else:
            return arrayout.type(dtype)


def amplitude(r):
    """
    Calculate the amplitude of a complex tensor.

    If the tensor is not complex then calculate square.
    """
    if torch.is_complex(r):
        return torch.abs(r) ** 2
    else:
        return r * r


def torch_dtype_to_numpy(dtype):
    """Convert a torch datatype to a numpy datatype."""
    scratch_array = torch.zeros(1, dtype=dtype)
    return scratch_array.cpu().numpy().dtype


def numpy_dtype_to_torch(dtype):
    """Convert a numpy datatype to a torch datatype."""
    return torch.from_numpy(np.zeros(1, dtype=dtype)).dtype


def real_to_complex_dtype_torch(dtype):
    return (torch.ones(1, dtype=dtype) * 1j).dtype


def complex_to_real_dtype_torch(dtype):
    return torch.abs(torch.zeros(1, dtype=dtype)).dtype


def fourier_shift_array_1d(
    y, posn, dtype=torch.float, device=torch.device("cpu"), units="pixels"
):
    """Apply Fourier shift theorem for sub-pixel shift to a 1 dimensional array."""
    if units == "pixels":
        n = 1
    else:
        n = y
    d_ = complex_dtype_to_real(dtype)
    if is_array_like(posn):

        return torch.exp(
            -2
            * np.pi
            * 1j
            * n
            * torch.fft.fftfreq(y, dtype=d_).to(device).view(1, y)
            * posn.view(len(posn), 1)
        )
    else:
        return torch.exp(
            -2 * np.pi * 1j * n * torch.fft.fftfreq(y, dtype=d_).to(device) * posn
        )


def fourier_shift_torch(
    array,
    posn,
    dtype=torch.float32,
    device=torch.device("cpu"),
    qspace_in=False,
    qspace_out=False,
    units="pixels",
):
    """
    Apply Fourier shift theorem for sub-pixel shifts to array.

    Parameters
    -----------
    array : torch.tensor (...,Y,X,2)
        Complex array to be Fourier shifted
    posn : torch.tensor (K x 2) or (2,)
        Shift(s) to be applied
    """

    if not qspace_in:
        array = torch.fft.fftn(array, dim=(-2, -1))

    array = array * fourier_shift_array(
        array.shape[-2:],
        posn,
        dtype=array.dtype,
        device=array.device,
        units=units,
    )

    if qspace_out:
        return array

    return torch.fft.ifftn(array, dim=[-2, -1])


def fourier_shift_array(
    size, posn, dtype=torch.float, device=torch.device("cpu"), units="pixels"
):
    """
    Create Fourier shift theorem array to (pixel) position given by list posn.

    Parameters
    ----------
    size : array_like
        size of the array (Y,X)
    posn : array_like
        can be a K x 2 array to give a K x Y x X shift arrays
    posn
    """
    # Get number of dimensions
    p_ = torch.as_tensor(posn)
    nn = len(p_.shape)

    # Get size of array
    y, x = size

    if nn == 1:
        # Make y ramp exp(-2pi i ky y)
        yramp = fourier_shift_array_1d(
            y, p_[0].item(), units=units, dtype=dtype, device=device
        )

        # Make y ramp exp(-2pi i kx x)
        xramp = fourier_shift_array_1d(
            x, p_[1].item(), units=units, dtype=dtype, device=device
        )

        # Multiply both arrays together, view statements for
        # appropriate broadcasting to 2D
        return yramp.view(y, 1) * xramp.view(1, x)
    else:
        K = p_.shape[0]
        # Make y ramp exp(-2pi i ky y)
        yramp, xramp = [
            fourier_shift_array_1d(xx, pos, units=units, dtype=dtype, device=device)
            for xx, pos in zip([y, x], p_.T)
        ]

        # Multiply both arrays together, view statements for
        # appropriate broadcasting to 2D
        return yramp.view(K, y, 1) * xramp.view(K, 1, x)


def crop_window_to_periodic_indices(win, shape):
    """
    Create indices for a rectangular subset of a larger array.

    If indices exceed the size of the larger array then these indices will wrap
    around to the other side of the grid providing two or more rectangular
    subsets of the larger array. Designed to be used in conjunction with
    the torch.narrow function to choose subsets of the square array to evaluate
    the PRISM algorithm on.

    Assumes that the requested window is smaller than the array size

    Parameters
    ----------
    win : (4,) array_like
        contains (y0,y,x0,x) the lower y index and y length and lower x index
        and x length
    shape : (2,) array_like
        Shape of the larger array

    Examples
    --------
    >>>> crop_window_to_periodic_indices([2,2,1,3],[5,5])
    (([2,2],[1,3]),)
    >>>> crop_window_to_periodic_indices([-1,3,1,3],[5,5])
    (([4,1],[1,3]),([0,2],[1,3]))
    >>>> crop_window_to_periodic_indices([4,4,1,3],[5,5])
    (([4,1],[1,3]),([0,3],[1,3]))
    >>>> list(crop_window_to_periodic_indices([4,4,3,3],[5,5]))
    (([4,1],[3,2]),([0,3],[3,2]),([4,1],[0,1]),([0,3],[0,1]))
    """

    def oneDindices(start, step, bound):
        if start + step > bound - 1:
            return [start, bound - start], [0, start + step - bound]
        elif start < 0:
            return [start % bound, bound - start % bound], [0, (start + step) % bound]
        else:
            return [[start, step]]

    y = oneDindices(*win[:2], shape[0])
    x = oneDindices(*win[2:], shape[1])

    return tuple(product(y, x))


def crop_window_to_flattened_indices_torch(indices: torch.Tensor, shape: list):
    """
    Create (flattened) indices for a rectangular subset of a larger array.

    Useful, for example for scattering matrix calculations where only a rectangular
    subset of the array is used in the PRISM interpolation routine

    Array indices exceeding the bounds of the array are wrapped to be consistent
    with periodic boundary conditions.

    Parameters
    ----------
    indices : torch.Tensor
        The centers of each of the cropping windows
    shape : array_like
        Size of the cropping windows

    Examples
    --------
    >>> indices = torch.as_tensor([[2,3,4],[1,2,3]])
    >>> gridshape = [4,4]
    >>> win = [3,3]
    >>> grid = torch.zeros(gridshape,dtype=torch.Long)
    tensor([[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]])
    >>> grid = grid.flatten()
    >>> ind = pyms.utils.crop_window_to_flattened_indices_torch(indices,gridshape)
    >>> grid[ind] = 1
    >>> grid.view(gridshape)
    tensor([[0, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 1]])
    """
    xind = torch.as_tensor(indices[-1]).view(1, len(indices[-1])) % shape[-1]
    yind = torch.as_tensor(indices[-2]).view(len(indices[-2]), 1) % shape[-2]
    return (xind + yind * shape[-1]).flatten().type(torch.LongTensor)


def bandwidth_limit_array_torch(
    arrayin, limit=2 / 3, qspace_in=True, qspace_out=True, soft=None, rfft=False
):
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
    soft : None or float, optional
        Apply a soft bandwidth limit, pass None (default) for a hardbandwidth limit,
        pass a floating point number for soft bandwidth limit where the number is the
        characteristic length of the error function edge of the bandwidth limit
    rfft: bool, optional
        Assume real space fft coordinates
    Returns
    -------
    array : array_like (...,Ny,Nx)
        The bandwidth limit of the array
    """
    # Transform array to Fourier space if necessary
    if qspace_in:
        array = copy.deepcopy(arrayin)
    else:
        array = torch.fft.fftn(arrayin, dim=[-2, -1])

    hardbandwidthlimit = soft is None

    # Case where band-width limiting has been turned off
    if limit is not None:
        if is_array_like(limit):
            lmt = limit[-2:]
        else:
            lmt = limit * np.ones(2)

        shp = array.shape

        q = [torch.square(torch.fft.fftfreq(s) * 2 / l) for s, l in zip(shp[-2:], lmt)]
        if rfft:
            q[1] = torch.square(torch.fft.rfftfreq((shp[-1] - 1) * 2) * 2 / lmt[-1])
        qgrid = q[0][:, np.newaxis] + q[1][np.newaxis, :]

        if hardbandwidthlimit:
            # Mask is a boolean array
            mask = qgrid >= 1

            if len(shp) > 2:
                array = array.reshape((np.prod(shp[:-2]), *shp[-2:]))
                for i in range(array.shape[0]):
                    array[i][mask] = 0
                array = array.reshape(shp)
            else:
                array[mask] = 0
        else:
            # Mask is a floating point array
            from scipy.special import erf

            grid = np.clip(-(np.sqrt(qgrid) - 1), 0, None) / soft
            mask = torch.as_tensor(erf(grid), dtype=torch.float32, device=array.device)
            array *= torch.as_tensor(mask, dtype=torch.float32, device=array.device)

    if qspace_out:
        return array
    else:
        return torch.fft.ifft2(array)


def crop_to_bandwidth_limit_torch(
    array: torch.Tensor,
    limit=2 / 3,
    qspace_in=True,
    qspace_out=True,
    norm="conserve_L2",
):
    """Crop an array to its bandwidth limit (remove superfluous array entries)."""

    # Get array shape, taking into account final dimension of size 2 if the array
    # is complex
    gridshape = array.shape[-2:]

    # New shape of final dimensions
    newshape = tuple([int(round(gridshape[i] * limit)) for i in range(2)])

    return fourier_interpolate_torch(
        array, newshape, norm=norm, qspace_in=qspace_in, qspace_out=qspace_out
    )


def size_of_bandwidth_limited_array(shape):
    """Get the size of an array after band-width limiting."""
    return list(crop_to_bandwidth_limit_torch(torch.zeros(*shape)).size())


def detect(detector, diffraction_pattern):
    """
    Apply a detector to a diffraction pattern.

    Calculates the signal in a diffraction pattern detector even if the size
    of the diffraction pattern and the detector are mismatched, assumes that
    the zeroth coordinate in reciprocal space is in the top-left hand corner
    of the array.
    """
    minsize = min(detector.size()[-2:], diffraction_pattern.size()[-2:])

    wind = [
        torch.fft.fftfreq(
            minsize[i], d=1 / minsize[i], dtype=torch.float, device=detector.device
        ).type(torch.int)
        for i in [0, 1]
    ]
    Dwind = crop_window_to_flattened_indices_torch(wind, detector.size())
    DPwind = crop_window_to_flattened_indices_torch(wind, diffraction_pattern.size())
    return torch.sum(
        detector.flatten(-2, -1)[:, None, Dwind]
        * diffraction_pattern.flatten(-2, -1)[None, :, DPwind],
        dim=-1,
    )


def fourier_interpolate_torch(
    ain, shapeout, norm="conserve_val", N=None, qspace_in=False, qspace_out=False
):
    """
    Fourier interpolation of array ain to shape shapeout.

    If shapeout is smaller than ain.shape then Fourier downsampling is
    performed

    Parameters
    ----------
    ain : (...,Nn,..,Ny,Nx) torch.tensor
        Input array
    shapeout : (n,) array_like
        Shape of output array
    norm : str, optional  {'conserve_val','conserve_norm','conserve_L1'}
        Normalization of output. If 'conserve_val' then array values are preserved
        if 'conserve_norm' L2 norm is conserved under interpolation and if
        'conserve_L1' L1 norm is conserved under interpolation
    N : int, optional
        Number of (trailing) dimensions to Fourier interpolate. By default take
        the length of shapeout
    qspace_in : bool, optional
        If True expect a Fourier space input, otherwise (default) expect a
        real space input
    qspace_out : bool, optional
        If True return a Fourier space output, otherwise (default) return in
        real space
    """
    dtype = ain.dtype

    if N is None:
        N = len(shapeout)

    inputComplex = iscomplex(ain)

    # Get input dimensions
    shapein = ain.size()[-N:]

    # axes to Fourier transform
    axes = np.arange(-N, 0).tolist()

    # Now transfer over Fourier coefficients from input to output array
    if inputComplex:
        ain_ = ain
    else:
        ain_ = torch.complex(
            ain, torch.zeros(ain.shape, dtype=ain.dtype, device=ain.device)
        )

    if not qspace_in:
        ain_ = torch.fft.fftn(ain_, dim=axes)

    aout = torch.fft.ifftshift(
        crop_torch(torch.fft.fftshift(ain_, dim=axes), shapeout), dim=axes
    )

    # Fourier transform result with appropriate normalization
    if norm == "conserve_val":
        aout *= np.prod(shapeout) / np.prod(np.shape(ain)[-N:])
    elif norm == "conserve_norm":
        aout *= np.sqrt(np.prod(shapeout) / np.prod(np.shape(ain)[-N:]))

    if not qspace_out:
        aout = torch.fft.ifftn(aout, dim=axes)

    # Return correct array data type
    if inputComplex:
        return aout
    return torch.real(aout)


def crop_torch(arrayin, shapeout):
    """
    Crop the last two dimensions of arrayin to grid size shapeout.

    For entries of shapeout which are larger than the shape of the input array,
    perform zero-padding.
    """
    C = iscomplex(arrayin)

    # Number of dimensions in input array
    ndim = arrayin.ndim

    # Trailing dimensions to be cropped/padded
    n = len(shapeout)

    # Number of dimensions not covered by shapeout (ie not to be cropped)
    nUntouched = ndim - n
    # Shape of output array
    shapeout_ = arrayin.shape[:nUntouched] + tuple(shapeout)

    arrayout = torch.zeros(shapeout_, dtype=arrayin.dtype, device=arrayin.device)

    oldshape = arrayin.shape[-n:]
    newshape = shapeout[-n:]

    def indices(y, y_):
        """Get slice objects for cropping or padding a 1D array from size y to size y_"""
        if y > y_:
            # Crop in y dimension
            y1, y2 = [(y - y_) // 2 + (y - y_) % 2, (y + y_) // 2 + (y - y_) % 2]
            y1_, y2_ = [0, y_]
        else:
            # Zero pad in y dimension
            y1, y2 = [0, y]
            y1_, y2_ = [(y_ - y) // 2 + (y - y_) % 2, (y + y_) // 2 + (y - y_) % 2]
        return slice(y1, y2), slice(y1_, y2_)

    ind = [indices(x, x_) for x, x_ in zip(oldshape, newshape)]
    inind, outind = map(tuple, zip(*ind))
    arrayout[nUntouched * (Ellipsis,) + outind] = arrayin[
        nUntouched * (Ellipsis,) + inind
    ]
    return arrayout
