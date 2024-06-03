"""Functions for emulating electron optics of a TEM."""

import numpy as np
import copy
from .utils.numpy_utils import q_space_array


class aberration:
    """A class describing electron lens aberrations."""

    def __init__(self, Krivanek, Haider, Description, amplitude, angle, n, m):
        """
        Initialize the lens aberration object.

        Parameters
        ----------
        Krivanek : str
            A string describing the aberration coefficient in Krivanek notation
            (C_mn)
        Haider : str
            A string describing the aberration coefficient in Haider notation
            (ie. A1, A2, B2)
        Description : str
            A string describing the colloquial name of the aberration ie. 2-fold
            astig.
        amplitude : float
            The amplitude of the aberration in Angstrom
        angle : float
            The angle of the aberration in radians
        n : int
            The principle aberration order
        m : int
            The rotational order of the aberration.
        """
        self.Krivanek = Krivanek
        self.Haider = Haider
        self.Description = Description
        self.amplitude = amplitude
        self.m = m
        self.n = n
        if m > 0:
            self.angle = angle
        else:
            self.angle = 0

    def __str__(self):
        """Return a string describing the aberration."""
        if self.m > 0:
            return (
                "{0:17s} ({1:2s}) -- {2:3s} = {3:9.2e} \u00E5 \u03B8 = "
                + "{4:4d}\u00B0 "
            ).format(
                self.Description,
                self.Haider,
                self.Krivanek,
                self.amplitude,
                int(np.rad2deg(self.angle)),
            )
        else:
            return " {0:17s} ({1:2s}) -- {2:3s} = {3:9.2e} \u00E5".format(
                self.Description, self.Haider, self.Krivanek, self.amplitude
            )


def Scherzer_defocus(eV, Cs):
    """Calculate the Scherzer defocus for a given voltage and Cs"""
    return -1.2 * np.sqrt(Cs / wavev(eV))


def depth_of_field(eV, alpha):
    """
    Calculate the probe depth of field (z-resolution) for a probe.

    Parameters
    ----------
    eV : float
        Probe accelerating voltage in electron volts
    alpha : float
        Probe forming semi-angle in mrad
    Returns
    -------
    dof : float
        The probe full-width-at-half-maximum (FWHM) depth of field
    """
    return 1.77 / wavev(eV) / alpha / alpha * 1e6


def aberration_starter_pack():
    """Create the set of aberrations up to fifth order."""
    aberrations = []
    aberrations.append(aberration("C10", "C1", "Defocus          ", 0.0, 0.0, 1, 0))
    aberrations.append(aberration("C12", "A1", "2-Fold astig.    ", 0.0, 0.0, 1, 2))
    aberrations.append(aberration("C23", "A2", "3-Fold astig.    ", 0.0, 0.0, 2, 3))
    aberrations.append(aberration("C21", "B2", "Axial coma       ", 0.0, 0.0, 2, 1))
    aberrations.append(aberration("C30", "C3", "3rd order spher. ", 0.0, 0.0, 3, 0))
    aberrations.append(aberration("C34", "A3", "4-Fold astig.    ", 0.0, 0.0, 3, 4))
    aberrations.append(aberration("C32", "S3", "Axial star aber. ", 0.0, 0.0, 3, 2))
    aberrations.append(aberration("C45", "A4", "5-Fold astig.    ", 0.0, 0.0, 4, 5))
    aberrations.append(aberration("C43", "D4", "3-Lobe aberr.    ", 0.0, 0.0, 4, 3))
    aberrations.append(aberration("C41", "B4", "4th order coma   ", 0.0, 0.0, 4, 1))
    aberrations.append(aberration("C50", "C5", "5th order spher. ", 0.0, 0.0, 5, 0))
    aberrations.append(aberration("C56", "A5", "6-Fold astig.    ", 0.0, 0.0, 5, 6))
    aberrations.append(aberration("C52", "S5", "5th order star   ", 0.0, 0.0, 5, 2))
    aberrations.append(aberration("C54", "R5", "5th order rosette", 0.0, 0.0, 5, 4))
    return aberrations


def chi(q, qphi, lam, df=0.0, aberrations=[]):
    r"""
    Calculate the aberration function, chi.

    Parameters
    ----------
    q : float or array_like
        Reciprocal space extent (Inverse angstroms).
    qphi : float or array_like
        Azimuth of grid in radians
    lam : float
        Wavelength of electron (Inverse angstroms).
    df : float, optional
        Defocus in Angstrom
    aberrations : list, optional
        A list containing a set of the class aberration, pass an empty list for
        an unaberrated contrast transfer function.
    Returns
    -------
    chi : float or array_like
        The aberration function, will be the same shape as `q`. This is used to
        calculate the probe wave function in reciprocal space.
    """
    qlam = q * lam
    chi_ = qlam**2 / 2 * df
    for ab in aberrations:
        chi_ += (
            qlam ** (ab.n + 1)
            * float(ab.amplitude)
            / (ab.n + 1)
            * np.cos(ab.m * (qphi - float(ab.angle)))
        )
    return 2 * np.pi * chi_ / lam


def make_contrast_transfer_function(
    pix_dim,
    real_dim,
    eV,
    app,
    optic_axis=[0, 0],
    aperture_shift=[0, 0],
    tilt_units="mrad",
    df=0,
    aberrations=[],
    q=None,
    app_units="mrad",
):
    """
    Make an electron lens contrast transfer function.

    Parameters
    ---------
    pix_dim : (2,) int array_like
        The pixel size of the grid
    real_dim : (2,) float array_like
        The size of the grid in Angstrom
    eV : float
        The energy of the probe electrons in eV
    app : float or None
        The aperture in units specified by app_units, pass `app` = None for
        no aperture
    optic_axis : (2,) array_like, optional
        allows the user to specify a different optic axis in units specified by
        `tilt_units`
    aperture_shift : (2,) array_like, optional
        Shift of the objective aperture relative to the center of the array
    tilt_units : string
        Units of the `optic_axis` or `aperture_shift` values, default is mrad
    df : float
        Probe defocus in A, a negative value indicate overfocus
    aberrations : array_like of aberration objects
        List containing instances of class aberration
    q :
        Precomputed reciprocal space array, allows the user to reduce
        computation time somewhat
    app_units : string
        The units of `app` (A^-1 or mrad)
    Returns
    -------
    ctf : array_like
        The lens contrast transfer function in reciprocal space
    """
    # Make reciprocal space array
    if q is None:
        q = q_space_array(pix_dim, real_dim[:2])

    # Get  electron wave number (inverse of wavelength)
    k = wavev(eV)

    # Convert tilts to units of inverse Angstrom
    optic_axis_ = convert_tilt_angles(
        optic_axis, tilt_units, real_dim, eV, invA_out=True
    )
    aperture_shift_ = convert_tilt_angles(
        aperture_shift, tilt_units, real_dim, eV, invA_out=True
    )

    if app is None:
        app_ = np.amax(np.abs(q))
    else:
        # Get aperture size in units of inverse Angstrom
        app_ = convert_tilt_angles(app, app_units, real_dim, eV, invA_out=True)

    # Initialize the array to contain the CTF
    CTF = np.zeros(pix_dim, dtype=complex)

    # Calculate the magnitude of the reciprocal lattice grid
    # qarray1 accounts for a shift of the optic axis
    qarray1 = np.sqrt(
        np.square(q[0] - optic_axis_[0]) + np.square(q[1] - optic_axis_[1])
    )

    # qarray2 accounts for a shift of both the optic axis and the aperture
    qarray2 = np.square(q[0] - optic_axis_[0] - aperture_shift_[0]) + np.square(
        q[1] - optic_axis_[1] - aperture_shift_[1]
    )

    # Calculate azimuth of reciprocal space array in case it is required for
    # aberrations
    qphi = np.arctan2(q[0] - optic_axis_[0], q[1] - optic_axis_[1])

    # Only calculate CTF for region within the aperture
    mask = qarray2 <= app_**2
    CTF[mask] = np.exp(-1j * chi(qarray1[mask], qphi[mask], 1.0 / k, df, aberrations))
    return CTF


def focused_probe(
    gridshape,
    rsize,
    eV,
    app,
    beam_tilt=[0, 0],
    aperture_shift=[0, 0],
    tilt_units="mrad",
    df=0,
    aberrations=[],
    q=None,
    app_units="mrad",
    qspace=False,
):
    """
    Make a focused electron probe wave function.

    Parameters
    ---------
    gridshape : (2,) array_like
        The pixel size of the grid
    rsize : (2,) array_like
        The size of the grid in Angstrom
    eV : float
        The energy of the probe electrons in electron volts
    app : float
        The probe-forming aperture in units specified by app_units, pass None
        if no probe forming aperture is to be used
    beam_tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) beam tilt. To maintain
        periodicity of the wave function at the boundaries this tilt is rounded
        to the nearest pixel value.
    aperture_shift : array_like, optional
        Allows the user to simulate a (small < 50 mrad) aperture shift. To
        maintain periodicity of the wave function at the boundaries this tilt
        is rounded to the nearest pixel value.
    tilt_units : string, optional
        Units of beam tilt and aperture shift, can be 'mrad','pixels' or 'invA'
    df : float, optional
        Probe defocus in A, a negative value indicate overfocus
    aberrations : list, optional
        A list of of probe aberrations of class pyms.Probe.aberration, pass an
        empty list for an un-aberrated probe
    app_units : string, optional
        The units of the aperture size ("invA", "pixels" or "mrad")
    qspace : bool, optional
        If True return the probe in reciprocal space
    Returns
    -------
    probe : complex (Y,X) np.ndarray
        The requested electron probe wave function
    """
    probe = make_contrast_transfer_function(
        gridshape,
        rsize,
        eV,
        app,
        beam_tilt,
        aperture_shift,
        tilt_units,
        df,
        aberrations,
        q,
        app_units,
    )

    # Normalize the STEM probe so that its sum-squared intensity is unity
    probe *= np.sqrt(np.prod(gridshape)) / np.sqrt(np.sum(np.square(np.abs(probe))))

    # Return real or diffraction space probe depending on user preference
    if not qspace:
        return np.fft.ifft2(probe)
    return probe


def plane_wave_illumination(
    gridshape, gridsize, eV, tilt=[0, 0], tilt_units="mrad", qspace=False
):
    """
    Generate plane wave illumination for input to multislice.

    The wave function will be normalized such that sum of intensity is unity in
    real space.

    Parameters
    ----------
    gridshape : (2,) array_like
        Pixel dimensions of the 2D grid
    gridsize : (2,) array_like
        Size of the grid in real space
    eV : float
        Probe energy in electron volts (irrelevant for untilted illumination)
    tilt : array_like, optional
        Allows the user to simulate a (small < 50 mrad) beam tilt, To maintain
        periodicity of the wave function at the boundaries this tilt is rounded
        to the nearest pixel value.
    tilt_units : string, optional
        Units of beam tilt, can be 'mrad','pixels' or 'invA'
    qspace : bool, optional
        Pass qspace = True to get the probe in momentum (q) space
    Returns
    ------
    illum : np.ndarray (Y,X)
    """
    # Initialize array that contains wave function
    illum = np.zeros(gridshape, dtype=complex)

    # Convert tilt to units of pixels
    tilt_ = convert_tilt_angles(tilt, tilt_units, gridsize, eV)

    # Case of an untilted plane wave (phase is zero everywhere)
    if tilt[0] == 0 and tilt[1] == 0:
        illum[:, :] = 1 / np.sqrt(np.prod(gridshape))

        if qspace:
            return np.fft.fft2(illum)
        else:
            return illum

    # Set the value of wavefunction amplitude such that after inverse Fourier
    # transform (and resulting division by the total number of pixels) the sum
    # of intensity will be 1
    illum[tilt_[0], tilt_[1]] = np.sqrt(np.prod(gridshape))

    # Return wave function in real space
    if qspace:
        return illum
    else:
        return np.fft.ifft2(illum)


def wavev(E):
    """
    Evaluate relativistically-corrected wavenumber (in inverse Angstrom) of electron with energy E.

    Energy E must be in electron-volts, see Eq. (2.5) in Kirkland's Advanced
    Computing in electron microscopy
    """
    # Planck's constant times speed of light in eV Angstrom
    hc = 1.23984193e4
    # Electron rest mass in eV
    m0c2 = 5.109989461e5
    return np.sqrt(E * (E + 2 * m0c2)) / hc


def relativistic_mass_correction(E):
    """
    Evaluate the relativistic mass correction for electron with energy E in eV.

    See Eq. (2.2) in Kirkland's Advanced Computing in electron microscopy.
    """
    # Electron rest mass in eV
    m0c2 = 5.109989461e5
    return (m0c2 + E) / m0c2


def simulation_result_with_Cc(
    func, Cc, deltaE, eV, args=[], kwargs={}, npoints=7, deltaEconv="1/e"
):
    """
    Perform a simulation using function, taking into account chromatic aberration.

    Pass in the function that simulates the multislice result, it is assumed
    that defocus is a variable named 'df' somewhere in the keyword argument
    list.

    Parameters
    ----------
    func : function
        Function that simulates the result of interest, ie. pyms.HRTEM. The
        defocus must be present in the keyword argument list as 'df'
    Cc : float
        Chromatic aberration coefficient in Angstroms
    deltaE : float
        Energy spread in electron volts using 1/e measure of spread (to convert
        from FWHM divide by 1.655, and divide by sqrt(2) to convert from
        standard deviation )
    eV : float
        (Mean) beam energy in electron volts
    args : list, optional
        Arguments for the method function used to propagate probes to the exit
        surface
    kwargs : Dict, optional
        Keyword arguments for the method function used to propagate probes to
        the exit surface
    npoints : int,optional
        Number of integration points in the Cc numerical integration
    Returns
    -------
    average : dict or array_like
        The simulation requested but averaged over the different defocus values
        to account for chromatic aberration.
    """
    # Check if a defocii has already been specified if not, assume that nominal
    # (mean) defocus is 0
    if "df" in kwargs.keys():
        nominal_df = kwargs["df"]
    else:
        nominal_df = 0

    # Get defocii to integrate over
    defocii = Cc_integration_points(Cc, deltaE, eV, npoints, deltaEconv) + nominal_df
    ndf = len(defocii)

    # Initialise the average result to None
    average = None

    # Now integrate the function over those defocus values
    for df in defocii:
        kwargs["df"] = df
        result = func(*args, **kwargs)

        # Assume that every function will either return a numpy array
        # (eg. pyms.HRTEM) or a list of numpy arrays (eg. pyms.STEM with both
        # conventional and 4D-STEM options) so average these results
        if isinstance(result, np.ndarray):
            if average is None:
                average = result / ndf
            else:
                average += result / ndf
        elif isinstance(result, dict):
            if average is None:
                average = copy.deepcopy(result)
                for key in average.keys():
                    if average[key] is not None:
                        average[key] /= ndf
            else:
                for key in average.keys():
                    if average[key] is not None:
                        average[key] += result[key] / ndf
        else:
            if average is None:
                average = [x / ndf for x in result]
            else:
                average = [x / ndf + y for x, y in zip(result, average)]
    return average


def Cc_integration_points(Cc, deltaE, eV, npoints=7, deltaEconv="1/e"):
    """
    Calculate the defocus integration points for simulating chromatic aberration.

    The integration points are selected by dividing the assumed gaussian defocus
    spread into npoints regions of equal probability, then finding the mean
    defocus in each of those regions.

    Parameters
    ----------
    Cc : float
        Chromatic aberration coefficient in Angstroms
    deltaE : float
        Energy spread in electron volts using 1/e measure of spread (to convert
        from FWHM divide by 1.655, and divide by sqrt(2) to convert from
        standard deviation )
    eV : float
        (Mean) beam energy in electron volts
    npoints : int,optional
        Number of integration points in the Cc numerical integration
    deltaEconv : float,optional
        The convention for deltaE, the energy spread, acceptable inputs are '1/e'
        the energy point that the probability density function drops to 1/e times
        its maximum value, 'std' for standard deviation and 'FWHM' for the full
        width at half maximum of the energy spread.
    Returns
    -------
    defocii : (`npoints`,) array_like
        The defocus integration points
    """
    # Import the error function (integral of Gaussian) and inverse error function
    # from scipy's special functions library
    from scipy.special import erfinv, erf

    # First divide the gaussian pdf into npoints different regions of equal
    # "area"
    partitions = erfinv(2 * (np.arange(npoints - 1) + 1) / npoints - 1)

    # Now calculate the mean (expectation value) within each partition
    x = np.zeros(npoints)
    prefactor = 1 / (2 * np.sqrt(np.pi))
    x[0] = -prefactor * np.exp(-partitions[0] ** 2) / (1 + erf(partitions[0])) * 2
    x[1:-1] = (
        prefactor
        * (np.exp(-partitions[:-1] ** 2) - np.exp(-partitions[1:] ** 2))
        / (erf(partitions[1:]) - erf(partitions[:-1]))
        * 2
    )
    x[-1] = prefactor * np.exp(-partitions[-1] ** 2) / (1 - erf(partitions[-1])) * 2

    # Multiply by 1/e spread of defocus values
    return x * Cc * convert_deltaE(deltaE, deltaEconv) / eV


def Cc_defocus_spread(df, Cc, deltaE, eV, deltaEconv):
    """
    Calculate the defocus spread for chromatic aberration.

    Evaluates the probability density function at defocus df for the defocus
    spread of a chromatic aberration (Cc) for (1/e) energy spread deltaE and
    beam energy eV in electron volts.


    Parameters
    ----------
    df : float or array_like
        defocus or defocii at which to evaluate the probability density function
    Cc : float
        Chromatic aberration coefficient in Angstroms
    deltaE : float
        Energy spread in electron volts using the measure given by deltaEconv
        (1/e measure of spread is default)
    eV : float
        (Mean) beam energy in electron volts
    deltaEconv : string,optional
        The convention for deltaE, the energy spread, acceptable inputs are '1/e'
        the energy point that the probability density function drops to 1/e times
        its maximum value, 'std' for standard deviation and 'FWHM' for the full
        width at half maximum of the energy spread
    Returns
    -------
    Cc_pdf : float or array_like
        the probability density function, will be the same size and shape as `df`
    """
    # Calculate defocus spread
    df_spread = Cc * convert_deltaE(deltaE, deltaEconv) / eV

    # Evaluate probability density function for given defocus df
    return np.exp(-df * df / df_spread / df_spread) / np.sqrt(np.pi) / df_spread


def convert_deltaE(deltaE, deltaEconv):
    """
    Convert the energy spread input to a 1/e spread.

    Parameters
    ----------
    deltaE : float
        Energy spread in electron volts using the measure given by deltaEconv
    deltaEconv : string
        The convention for deltaE, the energy spread, acceptable inputs are '1/e'
        the energy point that the probability density function drops to 1/e times
        its maximum value, 'std' for standard deviation and 'FWHM' for the full
        width at half maximum of the energy spread
    """
    if deltaEconv == "1/e":
        return deltaE
    elif deltaEconv == "FWHM":
        return deltaE / 2 / np.sqrt(np.log(2))
    elif deltaEconv == "std":
        return deltaE * np.sqrt(2)
    else:
        raise AttributeError(
            "detlaEconv "
            + deltaEconv
            + " input not recognized, needs to be one of '1/e', 'FWHM' or 'std'"
        )


def convert_tilt_angles(tilt, tilt_units, rsize, eV, invA_out=False):
    """
    Convert tilt to pixel or inverse Angstroms units regardless of input units.

    Input units can be mrad, pixels or inverse Angstrom

    Parameters
    ----------
    tilt : array_like
        Tilt in units of mrad, pixels or inverse Angstrom
    tilt_units : string
        Units of specimen and beam tilt, can be 'mrad','pixels' or 'invA'
    rsize : (2,) array_like
        The size of the grid in Angstrom
    eV : float
        Probe energy in electron volts
    invA_out : bool
        Pass True if inverse Angstrom units are desired.
    """
    # If units of the tilt are given in mrad, convert to inverse Angstrom
    if tilt_units == "mrad":
        k = wavev(eV)
        tilt_ = np.asarray(tilt) * 1e-3 * k
    else:
        tilt_ = tilt

    # If inverse Angstroms are requested our work here is done
    if invA_out:
        return tilt_

    # Convert inverse Angstrom to pixel coordinates, this will be rounded
    # to the nearest pixel
    if tilt_units != "pixels":
        tilt_ = np.round(tilt_ * rsize[:2]).astype(int)
    return tilt_
