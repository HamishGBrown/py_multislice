import numpy as np
from .utils.numpy_utils import q_space_array


class aberration:
    def __init__(self, Haider, Krivanek, Description, amplitude, angle, n, m):
        self.Haider = Haider
        self.Krivanek = Krivanek
        self.Description = Description
        self.amplitude = amplitude
        self.m = m
        self.n = n
        if m > 0:
            self.angle = angle
        else:
            self.angle = 0


def nyquist_sampling(rsize=None, resolution_limit=None, eV=None, alpha=None):
    """For resolution limit in units of inverse length calculate 
    sampling required to meet the Nyquist criterion. If array size in
    units of length is passed then return how many probe positions are 
    required otherwise just return the sampling. Alternatively pass 
    probe accelerating voltage (eV) in kV and probe forming aperture 
    (alpha) in mrad and the resolution limit in inverse length will be 
    calculated for you."""
    
    if eV is None and alpha is None:
        step_size = 1 / ( 4  * resolution_limit)
    elif resolution_limit is None:
        step_size = 1 / (4 * wavev(eV) * alpha * 1e-3)
    else:
        return None

    if rsize is None:
        return step_size
    else:
        return np.ceil(rsize/step_size).astype(np.int)


def aberration_starter_pack():
    """Creates the set of aberrations up to fifth order"""
    aberrations = []
    aberrations.append(aberration("C10", "C1", "Defocus          ", 0.0, 0.0, 1, 0))
    aberrations.append(aberration("C12", "A1", "2-Fold astig.    ", 0.0, 0.0, 1, 2))
    aberrations.append(aberration("C21", "B2", "Axial coma       ", 0.0, 0.0, 2, 1))
    aberrations.append(aberration("C23", "A2", "3-Fold astig.    ", 0.0, 0.0, 2, 3))
    aberrations.append(
        aberration("C30 = CS", "C3", "3rd order spher. ", 0.0, 0.0, 3, 0)
    )
    aberrations.append(aberration("C32", "S3", "Axial star aber. ", 0.0, 0.0, 3, 2))
    aberrations.append(aberration("C34", "A3", "4-Fold astig.    ", 0.0, 0.0, 3, 4))
    aberrations.append(aberration("C41", "B4", "4th order coma   ", 0.0, 0.0, 4, 1))
    aberrations.append(aberration("C43", "D4", "3-Lobe aberr.    ", 0.0, 0.0, 4, 3))
    aberrations.append(aberration("C45", "A4", "5-Fold astig     ", 0.0, 0.0, 4, 5))
    aberrations.append(
        aberration("C50 = CS5", "C5", "5th order spher. ", 0.0, 0.0, 5, 0)
    )
    aberrations.append(aberration("C52", "S5", "5th order star   ", 0.0, 0.0, 5, 2))
    aberrations.append(aberration("C54", "R5", "5th order rosette", 0.0, 0.0, 5, 4))
    aberrations.append(aberration("C56", "A5", "6-Fold astig.    ", 0.0, 0.0, 5, 6))
    return aberrations


def chi(q, qphi, lam, df=0.0, aberrations=[]):
    """calculates the aberration function chi as a function of
    reciprocal space extent q for an electron with wavelength lam.

    Parameters
    ----------
    q : number
        reciprocal space extent (Inverse angstroms).
    lam : number
        wavelength of electron (Inverse angstroms).
    aberrations : list
        A list object containing a set of the class aberration"""
    chi_ = (q * lam) ** 2 / 2 * df
    for ab in aberrations:
        chi_ += (
            (q * lam) ** (ab.n + 1)
            * float(ab.amplitude.get())
            / (ab.n + 1)
            * np.cos(ab.m * (qphi - float(ab.angle.get())))
        )
    return 2 * np.pi * chi_ / lam


def make_contrast_transfer_function(
    pix_dim,
    real_dim,
    eV,
    app,
    optic_axis=[0, 0],
    aperture_shift=[0, 0],
    df=0,
    aberrations=[],
    q=None,
    app_units="mrad",
):
    """Makes a contrast transfer function with pixel dimensions given in pix_dim
    and real_dimensions given by real_dim
    ---------
    pix_dim --- The pixel size of the grid
    real_dim --- The size of the grid in Angstrom
    eV --- The energy of the probe electrons in eV
    app --- The aperture in units specified by app_units, pass app = None for
            no aperture
    optic_axis --- allows the user to specify a different optic
                    axis
    aperture_shift --- Shift of the objective aperture relative
                        to the center of the array
    df --- Probe defocus in A, a negative value indicate overfocus
    aberrations --- List containing instances of class aberration
    q --- reciprocal space array, allows the user to reduce computation
        time somewhat
    app_units --- The units of the aperture size (A^-1 or mrad)
    """

    # Make reciprocal space array
    if q is None:
        q = q_space_array(pix_dim, real_dim[:2])

    # Get  electron wave number (inverse of wavelength)
    k = wavev(eV)

    if app is None:
        app_ = np.amax(np.abs(q))
    else:
        # Get aperture size in units of inverse Angstrom
        if app_units == "mrad":
            app_ = np.tan(app / 1000.0) * k
        else:
            app_ = app

    # Initialize the array to contain the CTF
    CTF = np.zeros(pix_dim, dtype=np.complex)

    # Calculate the magnitude of the reciprocal lattice grid
    # qarray1 accounts for a shift of the optic axis
    qarray1 = np.sqrt(np.square(q[0] - optic_axis[0]) + np.square(q[1] - optic_axis[1]))

    # qarray2 accounts for a shift of both the optic axis and the aperture
    qarray2 = np.sqrt(
        np.square(q[0] - optic_axis[0] - aperture_shift[0])
        + np.square(q[1] - optic_axis[1] - aperture_shift[1])
    )

    # Calculate azimuth of reciprocal space array in case it is required for
    # aberrations
    qphi = np.arctan2(q[0] - optic_axis[0], q[1] - optic_axis[1])

    # Only calculate CTF for region within the aperture
    mask = qarray2 < app_
    CTF[mask] = np.exp(-1j * chi(qarray1[mask], qphi[mask], 1.0 / k, df, aberrations))
    return CTF


def focused_probe(
    pix_dim,
    real_dim,
    eV,
    app,
    beam_tilt=[0, 0],
    aperture_shift=[0, 0],
    df=0,
    aberrations=[],
    q=None,
    app_units="mrad",
    qspace=False,
):
    """Makes a probe wave function with pixel dimensions given in pix_dim
    and real_dimensions given by real_dim
    ---------
    pix_dim --- The pixel size of the grid
    real_dim --- The size of the grid in Angstrom
    eV --- The energy of the probe electrons in electron volts
    app --- The apperture in units specified by app_units
    df --- Probe defocus in A, a negative value indicate overfocus
    aberrations --- list of probe aberrations of class pyms.Probe.aberration
    app_units --- The units of the aperture size ("A^-1" or "mrad")
    """
    probe = make_contrast_transfer_function(
        pix_dim,
        real_dim,
        eV,
        app,
        beam_tilt,
        aperture_shift,
        df,
        aberrations,
        q,
        app_units,
    )

    # Normalize the STEM probe so that its sum-squared intensity is unity
    probe /= np.sqrt(np.sum(np.square(np.abs(probe))))

    # Return real or diffraction space probe depending on user preference
    if qspace:
        return probe
    else:
        return np.fft.ifft2(probe, norm="ortho")


def plane_wave_illumination(
    gridshape, gridsize, tilt=[0, 0], eV=None, tilt_units="mrad", qspace=False
):
    """Create plane wave illumination with transverse momentum given by vector
       tilt in units of inverse Angstrom or mrad. To maintain periodicitiy of
       the wave function at the boundaries this tilt is rounded to the nearest
       pixel value. The wave function will be normalized such that sum of
       intensity is unity."""

    # Initiliaze array that contains wave function
    illum = np.zeros(gridshape, dtype=np.complex)

    # Case of an untilted plane wave (phase is zero everywhere)
    if tilt[0] == 0 and tilt[1] == 0:
        illum[:, :] = 1 / np.sqrt(np.product(gridshape))

        if qspace:
            return np.fft.fft2(illum)
        else:
            return illum

    # If units of the tilt are given in mrad, convert to inverse Angstrom
    if tilt_units == "mrad":
        k = wavev(eV)
        tilt_ = np.asarray(tilt) * 1e-3 * k
    else:
        tilt_ = tilt

    # Convert inverse Angstrom to pixel coordinates, this will be rounded
    # to the nearest pixel
    if tilt_units != "pixels":
        tilt_ = np.round(tilt_ * gridsize[:2]).astype(int)

    # Set the value of wavefunction amplitude such that after inverse Fourier
    # transform (and resulting division by the total number of pixels) the sum
    # of intensity will be 1
    illum[tilt_[0], tilt_[1]] = np.sqrt(np.product(gridshape))

    # Return wave function in real space
    if qspace:
        return illum
    else:
        return np.fft.ifft2(illum)


def wavev(E):
    """Calculates the relativistically corrected wavenumber k0 (reciprocal of
    the wavelength) for an electron of energy eV. See Eq. (2.5) in Kirkland's
    Advanced Computing in electron microscopy"""
    # Planck's constant times speed of light in eV Angstrom
    hc = 1.23984193e4
    # Electron rest mass in eV
    m0c2 = 5.109989461e5
    return np.sqrt(E * (E + 2 * m0c2)) / hc


def relativistic_mass_correction(E):
    """Gives the relativistic mass correction, m/m0 or gamma, for an electron
    with kinetic energy given by E in eV. Eq. (2.2) in Kirkland's Advanced
     Computing in electron microscopy"""
    # Electron rest mass in eV
    m0c2 = 5.109989461e5
    return (m0c2 + E) / m0c2
