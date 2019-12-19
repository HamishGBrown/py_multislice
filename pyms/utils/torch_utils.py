import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

re = np.s_[..., 0]
im = np.s_[..., 1]


def iscomplex(a: torch.Tensor):
    return a.shape[-1] == 2


def check_complex(A):
    for a in A:
        if not iscomplex(a):
            raise RuntimeWarning(
                "taking complex_mul of non-complex tensor! a.shape " + str(a.shape)
            )


def to_complex(real, imag=None):
    if imag is None:
        return torch.stack(
            [real, torch.zeros(real.size(), dtype=real.dtype, device=real.device)], -1
        )
    else:
        return torch.stack([real, imag], -1)


def get_device(device_type=None):
    """ Initialize device cuda if available, CPU if no cuda is available"""
    if device_type is None and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_type is None:
        device = torch.device("cpu")
    else:
        device = torch.device(device_type)
    return device


def complex_matmul(a: torch.Tensor, b: torch.Tensor, conjugate=False) -> torch.Tensor:
    """Complex matrix multiplication of tensors a and b. Pass conjugate = True
    to conjugate tensor b in the multiplication."""
    check_complex([a, b])
    are = a[re]
    aim = a[im]
    bre = b[re]
    bim = b[im]
    if conjugate:
        real = are @ bre + aim @ bim
        imag = -are @ bim + aim @ bre
    else:
        real = are @ bre - aim @ bim
        imag = are @ bim + aim @ bre

    return torch.stack([real, imag], -1)


def complex_mul(a: torch.Tensor, b: torch.Tensor, conjugate=False) -> torch.Tensor:
    """Complex array multiplication of tensors a and b. Pass conjugate = True
    to conjugate tensor b in the multiplication."""
    check_complex([a, b])
    are = a[re]
    aim = a[im]
    bre = b[re]
    bim = b[im]
    if conjugate:
        real = are * bre + aim * bim
        imag = -are * bim + aim * bre
    else:
        real = are * bre - aim * bim
        imag = are * bim + aim * bre

    return torch.stack([real, imag], -1)


def colorize(z, ccc=None, max=None, min=None):
    from colorsys import hls_to_rgb

    # Get shape of array
    n, m = z.shape

    # Intialize RGB array
    c = np.zeros((n, m, 3))

    # Set infinite and nan values to constant
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    # Map phase to float between 0 and 1 and store in array A
    A = ((np.angle(z[idx])) / (2 * np.pi)) % 1.0

    B = np.ones_like(A)
    if min is None:
        min_ = np.abs(z).min()
    else:
        min_ = min
    if max is None:
        max_ = np.abs(z).max()
    else:
        max_ = np.abs(max)

    if ccc is None:
        C = (np.abs(z[idx]) - min_) / (max_ - min_) * 0.5
    else:
        C = ccc
    # C = np.ones_like(B)*0.5
    c[idx] = [hls_to_rgb(a, cc, b) for a, b, cc in zip(A, B, C)]
    return c


# def pad_2d(array, padding,value=0):
#     """Pad a 2d array with value"""
#     #Calculate size of output array
#     outsize = array.size()
#     ndim = len(array.size())
#     for i in ndim:
#         outsize[i] += padding[i][0]+padding[i][1]

#     #initialize output padded array
#     if value == 0 :
#         out = torch.zeros(*outsize,dtype=array.dtype,device=array.device)
#     else:
#         out = torch.ones(*outsize,dtype=array.dtype,device=array.device)*value

#     for i in ndim:
#         out[padding[i][0]:padding[i][1]]


def torch_c_exp(angle):
    """Calculate exp(1j*angle)"""
    if angle.size()[-1] != 2:
        # Case of a real exponent
        result = torch.zeros(*angle.shape, 2, dtype=angle.dtype, device=angle.device)
        result[re] = torch.cos(angle)
        result[im] = torch.sin(angle)
    else:
        # Case of a complex valued exponent
        exp = torch.exp(-angle[im])
        result[re] = exp * torch.cos(angle[re])
        result[im] = exp * torch.sin(angle[re])
    return result


def torch_plot(array):

    if array.size()[-1] == 2:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

        axes[0].imshow(
            np.angle(array[..., 0].cpu().numpy() + 1j * array[..., 1].cpu().numpy())
        )
        axes[0].set_title("Phase")
        axes[1].imshow(
            np.square(array[..., 0].cpu().numpy())
            + np.square(array[..., 1].cpu().numpy())
        )
        axes[1].set_title("Amplitude")
        axes[2].imshow(
            colorize(array[..., 0].cpu().numpy() + 1j * array[..., 1].cpu().numpy())
        )
        axes[2].set_title("Colorized")
    else:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.imshow(array.cpu().numpy())
        ax.set_axis_off()

    plt.show()


def sinc(x):
    """sinc function return sin(pi x)/(pi x)"""
    y = torch.where(torch.abs(x) < 1.0e-20, torch.tensor([1.0e-20], dtype=x.dtype), x)
    return torch.sin(np.pi * y) / np.pi / y


def ensure_torch_array(array, dtype=torch.float, device=torch.device("cpu")):
    """Takes an array and converts it to a pytorch array if it is numpy
    and does nothing if the array is already a numpy array"""

    if not isinstance(array, torch.Tensor):
        if np.iscomplexobj(array):
            return cx_from_numpy(array, dtype=dtype, device=device)
        else:
            return torch.from_numpy(array).type(dtype).to(device)
    else:
        return array


def torch_output(array, fnam, add1=True):
    """Outputs a pytorch tensor as a 32-bit tiff for quick inspection"""
    from numpy import abs, angle

    numparray = array[:, :, 0].cpu().numpy() + 1j * array[:, :, 1].cpu().numpy()
    if add1:
        Image.fromarray(abs(1 + numparray)).save(fnam + "_amp.tif")
        Image.fromarray(angle(1 + numparray)).save(fnam + "_phse.tif")
    else:
        print(abs(numparray).dtype)
        Image.fromarray(abs(numparray)).save(fnam + "_amp.tif")
        Image.fromarray(angle(numparray)).save(fnam + "_phse.tif")


def modulus_square(r):
    return torch.sum(amplitude(r))


def amplitude(r):
    """Calculate the amplitude of a complex tensor (final dimension is 2) otherwise
    do nothing"""
    if r.size(-1) == 2:
        return r[..., 0] * r[..., 0] + r[..., 1] * r[..., 1]
    else:
        return r


def roll_n(X, axis, n):
    f_idx = tuple(
        slice(None, None, None) if i != axis % X.dim() else slice(0, n, None)
        for i in range(X.dim())
    )
    b_idx = tuple(
        slice(None, None, None) if i != axis % X.dim() else slice(n, None, None)
        for i in range(X.dim())
    )
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)


def cx_from_numpy(x: np.array, dtype=torch.float, device=None) -> torch.Tensor:
    if "complex" in str(x.dtype):
        out = torch.zeros(x.shape + (2,), device=device)
        out[re] = torch.from_numpy(x.real)
        out[im] = torch.from_numpy(x.imag)
    else:
        if x.shape[-1] != 2:
            out = torch.zeros(x.shape + (2,))
            out[re] = torch.from_numpy(x.real)
        else:
            out = torch.zeros(x.shape + (2,))
            out[re] = torch.from_numpy(x[re])
            out[re] = torch.from_numpy(x[im])
    return out.type(dtype)


def cx_to_numpy(x: torch.Tensor) -> np.ndarray:

    check_complex(x)

    return x[re].cpu().numpy() + 1j * x[im].cpu().numpy()


def batch_fftshift2d_real(x, axes=None):
    """Apply an FFTshift operation the last two dimensions of the torch tensor
       x which is a real valued array"""

    # If no instructions are given regarding axes apply
    # an FFT shift to all axes
    if axes is None:
        axes = [x for x in range(len(x.size()))]

    # Loop over axes
    for dim in axes:

        # Shift axes by number of dimensions divided by 2, add 1 for odd axes
        n_shift = x.size(dim) // 2 + x.size(dim) % 2

        # Apply circular shift to each axes
        x = roll_n(x, axis=dim, n=n_shift)

    # Return array
    return x


def batch_fftshift2d(x, axes=None):
    """Apply an FFTshift operation the last two dimensions of the torch tensor
       x"""
    # real and imaginary components of the tensor assumed to be stored in the
    # final dimension of array x, torch.unbind maps these to seperate arrays
    # real and imag, see https://pytorch.org/docs/stable/torch.html#torch.unbind
    real, imag = torch.unbind(x, -1)

    # If no instructions are given regarding axes to apply FFT shift apply
    # an FFT shift to all axes
    if axes is None:
        axes = [x for x in range(len(real.size()))]

    # Loop over axes
    for dim in axes:

        # Shift axes by number of dimensions divided by 2, add 1 for odd axes
        n_shift = real.size(dim) // 2 + real.size(dim) % 2

        # Apply circular shift to each axes
        [real, imag] = [roll_n(x, axis=dim, n=n_shift) for x in [real, imag]]

    # Return array with real and imaginary components spliced back together
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def batch_ifftshift2d(x, axes=None):
    # Split tensor into real and imaginary components
    real, imag = torch.unbind(x, -1)

    # If axes is None then fftshift all dimensions
    if axes is None:
        axes = torch.arange(len(real.size()) - 1, -1, -1)

    # Loop over axes
    for dim in axes:
        real = roll_n(real, axis=dim, n=real.size(dim) // 2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim) // 2)

    # Splice back real and imaginary components
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)


def scatter_add_patches(
    input: torch.Tensor, out: torch.Tensor, axes, positions, patch_size, index=None
) -> torch.Tensor:
    """
    Scatter_adds K patches of size :patch_size: at axes [ax1, ax2] into the output tensor. The patches are added at
    positions :positions:. Additionally, several dimensions can be summed, specified by reduce_dims.
    :param input:   K x M1 x M2 at least 3-dimensional tensor of patches,
    :param out: at least two-dimensional tensor that is the scatter_add target
    :param axes: (2,) axes at which to scatter the input
    :param positions: K x 2 LongTensor
    :param patch_size: (2,) LongTensor
    :param reduce_dims: (N,) LongTensor
    :return: out, the target of scatter_add_
    """
    if index is None:
        index = get_scatter_gather_indices(positions, patch_size, out, axes)

    out.view(-1).scatter_add_(0, index.view(-1), input.view(-1))
    return out


def get_scatter_gather_indices(r, s, input, axes, edge_behaviour="periodic"):
    """Get indicies for the scatter or gather functions, where r is a set of
    indices with dimensions K x 2, s is the size of the patch that will be
    gathered of scattered to, input is the array that will be gathered from
    or scattered to and axes are the indices of the dimensions that the
    patches will come from."""
    # Get number of patches
    K = r.shape[0]

    # Create tensors with indices of the patches (0,1,2,...,patchsize[i]), use
    # view to adjust to the correct dimensionality, then expand these tensors to
    # two dimensional arrays. The function "expand" returns a new view of the
    # tensor with singleton dimensions expanded to a larger size, ie it performs
    # a similar role to the meshgrid function in numpy.
    # index0 and index1 are shape s[0] x s[1]
    start = -s[0] // 2 + s[0] % 2
    stop = s[0] // 2 + s[0] % 2
    index0 = (
        torch.arange(start, stop, device=input.device, dtype=torch.long)
        .view(s[0], 1)
        .expand(s[0], s[1])
    )
    start = -s[1] // 2 + s[1] % 2
    stop = s[1] // 2 + s[1] % 2
    index1 = (
        torch.arange(start, stop, device=input.device, dtype=torch.long)
        .view(1, s[1])
        .expand(s[0], s[1])
    )

    # Add the positions of the patches (r) to each of these indices (index0 and
    # index1), view expands the dimensionality of r to allow for proper
    # broadcasting.
    # index is shape K x s[0] x s[1]
    if edge_behaviour == "periodic":
        index = input.stride(axes[0]) * (
            torch.remainder(index0 + r[:, 0].view(K, 1, 1), input.shape[axes[0]])
        ) + input.stride(axes[1]) * (
            torch.remainder(index1 + r[:, 1].view(K, 1, 1), input.shape[axes[1]])
        )
    else:
        index = input.stride(axes[0]) * (
            torch.clamp(index0 + r[:, 0].view(K, 1, 1), 0, input.shape[axes[0]])
        ) + input.stride(axes[1]) * (
            torch.clamp(index1 + r[:, 1].view(K, 1, 1), 0, input.shape[axes[1]])
        )

    # Now add the offsets for the higher dimensions of the array, torch.arange
    # produces an array between 0 and the stride of the final axis and the view
    # method maps this to a shape that can be broadcast across the whole index
    # array
    higher_dim_offsets = torch.arange(input.stride(axes[1]), device=input.device).view(
        1, 1, 1, input.stride(axes[1])
    )

    # Reshape the index array and broadcast it across the stride of the final
    # dimension
    # index is shape K x s[0] x s[1] x stride(axes[1])
    index = index.view(index.shape[0], index.shape[1], index.shape[2], 1).expand(
        (index.shape[0], index.shape[1], index.shape[2], input.stride(axes[1]))
    )

    # Add the offsets for the higher dimension to hte array
    index = index + higher_dim_offsets

    # condense all dimensions x_0 ... x_a leading up to the those designated
    # in the input axes
    dim0size = (
        torch.prod(torch.Tensor([input.shape[: axes[0]]])).int().item()
        if axes[0] > 0
        else 1
    )

    view = [dim0size]

    # Add all of those dimensions after those specified by axes (do not condense)
    for d in input.shape[axes[0] :]:
        view.append(d)
    y = input.view(torch.Size(view)).squeeze()
    # index is shape K x prod(size(input[:axes[0]])) x s[0] x s[1] x stride(axes[1])
    index = index.view(
        index.shape[0], 1, index.shape[1], index.shape[2], index.shape[3]
    ).expand((index.shape[0], dim0size, index.shape[1], index.shape[2], index.shape[3]))

    # Add the offset for dimensions leading up to axis[0]
    lower_dim_offset = torch.arange(dim0size, device=input.device) * y.stride(0)
    lower_dim_offset = lower_dim_offset.view(1, dim0size, 1, 1, 1).long()
    index = index + lower_dim_offset

    # print(f"new index shape: {index.shape}")
    return index.contiguous().view(-1)


def gather_patches(input, axes, positions, patch_size, index=None) -> torch.Tensor:
    """
    Gathers K patches of size :patch_size: at axes [ax1, ax2] of the input tensor. The patches are collected started at
    K positions pos.
    if :input: is an n-dimensional tensor with size (x_0, x_1, x_2, ..., x_a, x_ax1, x_ax2, x_b, ..., x_{n-1})
    then :out: is an n-dimensional tensor with size  (K, x_0, x_1, x_2, ..., x_a, patch_size[0], patch_size[1], x_3, ..., x_{n-1})
    :param input: at least two-dimensional tensor
    :param axes: axes at which to gather the patches
    :param positions: K x 2 LongTensor
    :param patch_size: (2,) LongTensor
    :param out: n-dimensional tensor with size  (K, x_0, x_1, x_2, ..., x_a, patch_size[0], patch_size[1], x_3, ..., x_{n-1})
    :return:
    """

    r = positions
    s = patch_size
    K = positions.shape[0]

    # Call the get index function
    if index is None:
        index = get_scatter_gather_indices(r, s, input, axes).contiguous().view(-1)

    # condense all dimensions x_0 ... x_a leading up to the those designated
    # in the input axes
    dim0size = (
        torch.prod(torch.Tensor([input.shape[: axes[0]]])).int().item()
        if axes[0] > 0
        else 1
    )
    view = [dim0size]

    # Add all of those dimensions after those specified by axes (do not condense)
    for d in input.shape[axes[0] :]:
        view.append(d)
    y = input.view(torch.Size(view)).squeeze()

    # Select the relevant indices
    out = torch.index_select(y.view(-1), 0, index)

    # Now reconstruct the output array into the right shape

    # out_view will be shape of the output tensor. This will be constructed over
    # the next few lines, starting with the number of positions, K.
    out_view = (K,)
    # Progessively add the indices of the input, starting with the indices
    # leading up to the start of the axes specified in the input variable axes
    for ax in input.shape[: axes[0]]:
        out_view += (ax,)

    # Add the size of the patch to the output shape
    # The function .item() turns a tensor into a number single value
    out_view += (patch_size[0].item(),)
    out_view += (patch_size[1].item(),)

    # Progessively add the indices of the input, finishing with the indices
    # after those axes specified in the input variable axes
    for ax in input.shape[axes[1] + 1 :]:
        out_view += (ax,)

    # Return out with the shape built up over the last few lines
    out = out.view(out_view)
    return out


def torch_bin(array, factor=2):
    y, x = array.size()[-2:]
    biny, binx = [y // factor, x // factor]
    yy, xx = [biny * factor, binx * factor]
    result = torch.zeros(
        *array.size()[:2], biny, binx, device=array.device, dtype=array.dtype
    )

    for iy in range(factor):
        for ix in range(factor):
            result[..., :, :] += array[
                ..., 0 + iy : yy + iy : factor, 0 + ix : xx + ix : factor
            ]
    return result


def align(arr1, arr2):
    """Align array arr1 to arr2 using Fourier correlation."""
    # Correlate both arrays
    corr = correlate(arr2, arr1)

    # Now get index of maximum value of correlation
    ind = torch.argmax(corr)
    y, x = [ind // arr1.stride(-2), ind % arr1.stride(-2)]

    return roll_n(roll_n(arr1, -2, y), -1, x)


def correlate(arr1, arr2):
    """Find the cross correlation of two arrays using the Fourier transform. 
    Assumes that both arrays are periodic so can be circularly shifted."""

    complex_input = arr1.size(-1) == 2 and arr2.size(-1) == 2

    if complex_input:
        result = complex_mul(*[torch.fft(x, 2) for x in [arr2, arr1]], conjugate=True)
        return torch.ifft2(result, 2)
    else:
        # Make arrays "complex" by ading zero valued imaginary component to final index
        a_ = torch.cat(
            [x.view(*arr1.size(), 1) for x in [arr1, torch.zeros_like(arr1)]], -1
        )
        b_ = torch.cat(
            [x.view(*arr2.size(), 1) for x in [arr2, torch.zeros_like(arr2)]], -1
        )

        # Multiply arr1 by conjugate of arr2 in Fourier space
        # TODO make use of the real Fourier transforms (torch.rfft) to make this much
        # more efficient
        result = complex_mul(*[torch.fft(x, 2) for x in [b_, a_]], conjugate=True)

        # Return real part of invere Fourier transformed result
        return torch.ifft(result, 2)[..., 0]


def fftfreq(n, dtype=torch.float, device=torch.device("cpu")):
    """Same as numpy.fft.fftfreq(n)*n """
    return (torch.arange(n, dtype=dtype, device=device) + n // 2) % n - n // 2


def fourier_shift_array_1d(
    y, posn, dtype=torch.float, device=torch.device("cpu"), units="pixels"
):
    ramp = torch.empty(y, 2, dtype=dtype, device=device)
    ky = 2 * np.pi * fftfreq(y) * posn
    if units == "pixels":
        ky /= y
    ramp[..., 0] = torch.cos(ky)
    ramp[..., 1] = -torch.sin(ky)
    return ramp


def fourier_shift_torch(
    array,
    posn,
    dtype=torch.float,
    device=torch.device("cpu"),
    qspace_in=False,
    qspace_out=False,
    units="pixels",
):
    """Apply Fourier shift theorem to array. Calls Fourier_shift_array to generate
    the shift array"""

    if not qspace_in:
        array = torch.fft.fft2(array, signal_ndim=2)

    array = complex_mul(
        array,
        fourier_shift_array(
            array.size()[-3:-1],
            posn,
            dtype=array.dtype,
            device=array.device,
            units=units,
        ),
    )

    if qspace_out:
        return array

    return torch.ifft(array, signal_ndim=2)


def fourier_shift_array(
    size, posn, dtype=torch.float, device=torch.device("cpu"), units="pixels"
):
    """Fourier shift theorem array for array shape size, 
       to (pixel) position given by list posn.
       
       posn can be a K x 2 array to give a K x Y x X 
       array of Fourier shift arrays"""
    # Get number of dimensions
    nn = len(posn.size())

    # Get size of array
    y, x = size

    if nn == 1:
        # Make y ramp exp(-2pi i ky y)
        yramp = fourier_shift_array_1d(
            y, posn[0], units=units, dtype=dtype, device=device
        )

        # Make y ramp exp(-2pi i kx x)
        xramp = fourier_shift_array_1d(
            x, posn[1], units=units, dtype=dtype, device=device
        )

        # Multiply both arrays together, view statements for
        # appropriate broadcasting to 2D
        return complex_mul(yramp.view(y, 1, 2), xramp.view(1, x, 2))
    else:
        K = posn.size(0)
        # Make y ramp exp(-2pi i ky y)
        yramp = torch.empty(K, y, 2, dtype=dtype, device=device)
        ky = (
            2
            * np.pi
            * fftfreq(y, dtype=dtype, device=device).view(1, y)
            * posn[:, 0].view(K, 1)
        )
        if units == "pixels":
            ky /= y
        yramp[..., 0] = torch.cos(ky)
        yramp[..., 1] = -torch.sin(ky)

        # Make y ramp exp(-2pi i kx x)
        xramp = torch.empty(K, x, 2, dtype=dtype, device=device)
        kx = (
            2
            * np.pi
            * fftfreq(x, dtype=dtype, device=device).view(1, x)
            * posn[:, 1].view(K, 1)
        )
        if units == "pixels":
            kx /= x

        xramp[..., 0] = torch.cos(kx)
        xramp[..., 1] = -torch.sin(kx)

        # Multiply both arrays together, view statements for
        # appropriate broadcasting to 2D
        return complex_mul(yramp.view(K, y, 1, 2), xramp.view(K, 1, x, 2))


def crop_window_to_flattened_indices_torch(indices: torch.Tensor, shape: list):
    # initialize array to hold flattened index in

    return (
        indices[-1].view(1, indices[-1].size(0))
        + indices[-2].view(indices[-2].size(0), 1) * shape[-1]
    ).flatten()


def crop_to_bandwidth_limit_torch(array: torch.Tensor, limit=2 / 3):
    """Crop an array to its bandwidth limit (ie remove superfluous array entries)"""

    from .numpy_utils import crop_window_to_flattened_indices

    # Check if array is complex or not
    complx = iscomplex(array)

    # Get array shape, taking into account final dimension of size 2 if the array
    # is complex
    gridshape = list(array.size())[-2 - int(complx) :][:2]

    # New shape of final dimensions
    newshape = tuple([int(round(gridshape[i] * limit)) for i in range(2)])

    # Indices of values to take
    ind = [
        (np.fft.fftfreq(newshape[i], 1 / newshape[i]).astype(np.int) + gridshape[i])
        % gridshape[i]
        for i in range(2)
    ]
    ind = torch.tensor(
        crop_window_to_flattened_indices(ind, gridshape).astype("int32"),
        dtype=torch.long,
    )

    # flatten final two dimensions of array
    flat_shape = tuple(array.size()[: -2 - int(complx)]) + (int(np.prod(gridshape)),)
    newshape = tuple(array.size()[: -2 - int(complx)]) + newshape

    if complx:
        flat_shape += (2,)
        newshape += (2,)
        return array.view(flat_shape)[..., ind, :].view(newshape)
    return array.view(flat_shape)[..., ind].view(newshape)


def size_of_bandwidth_limited_array(shape):
    return list(crop_to_bandwidth_limit_torch(torch.zeros(*shape)).size())


def detect(detector, diffraction_pattern):
    """Calculates the signal in a diffraction pattern detector even if the size
    of the diffraction pattern and the detector are mismatched, assumes that
    the zeroth coordinate in reciprocal space is in the top-left hand corner
    of the array."""
    minsize = min(detector.size()[-2:], diffraction_pattern.size()[-2:])

    wind = [fftfreq(minsize[i], torch.long, detector.device) for i in [0, 1]]
    Dwind = crop_window_to_flattened_indices_torch(wind, detector.size())
    DPwind = crop_window_to_flattened_indices_torch(wind, diffraction_pattern.size())
    return torch.sum(
        detector.flatten(-2, -1)[:, None, Dwind]
        * diffraction_pattern.flatten(-2, -1)[None, :, DPwind],
        dim=-1,
    )


def fourier_interpolate_2d_torch(
    ain, shapeout, correct_norm=True, qspace_in=False, qspace_out=False
):
    """Perfoms a fourier interpolation on array ain so that its shape matches
    that given by shapeout.

    Arguments:   
    ain      -- Input numpy array
    shapeout -- Shape of output array
    correct_norm -- If True normalization such that values are conserved, if false
                    normalization such that sum square is conserved
    qspace_in   -- If True expect a Fourier space input
    qspace_out  -- If True return a Fourier space output, otherwise return
                   in real space
    """
    dtype = ain.dtype
    inputComplex = iscomplex(ain)
    # Make input complex
    aout = torch.zeros(
        ain.shape[: -2 - int(inputComplex)] + (np.prod(shapeout), 2),
        dtype=dtype,
        device=ain.device,
    )

    # Get input dimensions
    npiyin, npixin = ain.size()[-2 - int(inputComplex) :][:2]
    npiyout, npixout = shapeout

    # Get Fourier interpolation masks
    # PyTorch does not yet do element-wise logic operations, so we have to do
    # this bit in numpy. Additionally, in Windows pytorch does not support
    # bool types so we have to convert this to a unsigned 8-bit integer.
    from .numpy_utils import Fourier_interpolation_masks

    maskin, maskout = [
        torch.from_numpy(x).flatten()
        for x in Fourier_interpolation_masks(npiyin, npixin, npiyout, npixout)
    ]

    # Now transfer over Fourier coefficients from input to output array
    if inputComplex:
        ain_ = ain
    else:
        ain_ = to_complex(ain).flatten(-3, -2)

    if not qspace_in:
        ain_ = torch.fft(ain_, signal_ndim=2)

    aout[..., maskout, :] = ain_.flatten(-3, -2)[..., maskin, :]

    # Fourier transform result with appropriate normalization
    aout = aout.reshape(ain.shape[: -2 - int(inputComplex)] + tuple(shapeout) + (2,))

    if not qspace_out:
        aout = torch.ifft(aout, signal_ndim=2)
    if correct_norm:
        aout *= np.prod(shapeout) / np.prod([npiyin, npixin])
    else:
        aout *= np.sqrt(np.prod(shapeout) / np.prod([npiyin, npixin]))

    # Return correct array data type
    if inputComplex:
        return aout
    return aout[re]
