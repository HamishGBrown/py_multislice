import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import copy
from re import split, match
from os.path import splitext
from .atomic_scattering_params import e_scattering_factors, atomic_symbol
from .Probe import wavev, relativistic_mass_correction
from .utils.numpy_utils import bandwidth_limit_array, q_space_array
from .utils.torch_utils import (
    sinc,
    roll_n,
    cx_from_numpy,
    cx_to_numpy,
    complex_mul,
    torch_c_exp,
    fourier_shift_array,
    amplitude,
    get_device,
)


def Xray_scattering_factor(Z, gsq, units="A"):
    # Bohr radius in Angstrom
    a0 = 0.529177
    # gsq = g**2
    return Z - 2 * np.pi ** 2 * a0 * gsq * electron_scattering_factor(
        Z, gsq, units=units
    )


def electron_scattering_factor(Z, gsq, units="VA"):
    ai = e_scattering_factors[Z - 1, 0:10:2]
    bi = e_scattering_factors[Z - 1, 1:10:2]

    # Planck's constant in kg Angstrom/s
    h = 6.62607004e-24
    # Electron rest mass in kg
    me = 9.10938356e-31
    # Electron charge in Coulomb
    qe = 1.60217662e-19

    fe = np.zeros_like(gsq)

    for i in range(5):
        fe += ai[i] * (2 + bi[i] * gsq) / (1 + bi[i] * gsq) ** 2

    # Result can be returned in units of Volt Angstrom ('VA') or Angstrom ('A')
    if units == "VA":
        return h ** 2 / (2 * np.pi * me * qe) * fe
    elif units == "A":
        return fe


def calculate_scattering_factors(gridshape, gridsize, elements):
    """Calculates the electron scattering factors on a reciprocal space
        grid of pixel size pixels assuming a unit cell tiling given by 
        tiling"""

    # Get reciprocal space array
    g = q_space_array(gridshape, gridsize)
    gsq = np.square(g[0]) + np.square(g[1])

    # Initialise scattering factor array
    fe = np.zeros((len(elements), *gridshape), dtype=np.float32)

    # Loop over unique elements
    for ielement, element in enumerate(elements):
        fe[ielement, :, :] = electron_scattering_factor(element, gsq)

    return fe


def find_equivalent_sites(positions, EPS=1e-3):
    """Finds equivalent atomic sites, ie two atoms sharing the same 
    postions with fractional occupancy, and return an index of these equivalent
    sites"""
    from scipy.spatial.distance import pdist

    natoms = positions.shape[0]
    # Calculate pairwise distance between each atomic site
    distance_matrix = pdist(positions)

    # Initialize index of equivalent sites (initially assume all sites are
    # independent)
    equivalent_sites = np.arange(natoms, dtype=np.int)

    # Find equivalent sites
    equiv = distance_matrix < EPS

    # If there are equivalent sites correct the index of equivalent sites
    if np.any(equiv):
        # Masking function to get indices from distance_matrix
        iu = np.mask_indices(natoms, np.triu, 1)

        # Get a list of equivalent sites
        sites = np.nonzero(equiv)[0]
        for site in sites:
            # Use the masking function to
            equivalent_sites[iu[1][site]] = iu[0][site]
    return equivalent_sites


def interaction_constant(E, units="rad/VA"):
    """Calculates the interaction constant, sigma, to convert electrostatic
    potential (in V Angstrom) to radians. Units of this constant are rad/(V
    Angstrom).  See Eq. (2.5) in Kirkland's Advanced Computing in electron
    microscopy """
    # Planck's constant in kg Angstrom /s
    h = 6.62607004e-24
    # Electron rest mass in kg
    me = 9.10938356e-31
    # Electron charge in Coulomb
    qe = 1.60217662e-19
    # Electron wave number (reciprocal of wavelength) in Angstrom
    k0 = wavev(E)
    # Relativistic electron mass correction
    gamma = relativistic_mass_correction(E)
    if units == "rad/VA":
        return 2 * np.pi * gamma * me * qe / k0 / h / h
    elif units == "rad/A":
        return gamma / k0


def rot_matrix(theta, u=np.asarray([0, 0, 1], dtype=np.float)):
    """Generates the 3D rotational matrix for a rotation of angle theta in 
    radians around axis given by vector u."""
    from numpy import sin, cos, pi

    c = cos(theta)
    s = sin(theta)
    ux, uy, uz = u
    R = np.zeros((3, 3))
    R[0, :] = [
        c + ux * ux * (1 - c),
        ux * uy * (1 - c) - uz * s,
        ux * uz * (1 - c) + uy * s,
    ]
    R[1, :] = [
        uy * uz * (1 - c) + uz * s,
        c + uy * uy * (1 - c),
        uy * uz * (1 - c) - ux * s,
    ]
    R[2, :] = [
        uz * ux * (1 - c) - uy * s,
        uz * uy * (1 - c) + ux * s,
        c + uz * uz * (1 - c),
    ]
    return R


class crystal:
    """ Elements in a crystal object:
     unitcell - An array containing the side lengths of the orthorhombic unit cell
     atomtypes - A string array containing the symbols of atomic elements in the cell
     natoms - Total number of atoms of each element within the cell
     atoms - An array of dimensions total number of atoms by 6 which for each atom
            contains the fractional cooordinates within the unit cell for each atom
            in the first three entries, the atomic number in the fourth entry,
            the atomic occupancy (not yet implemented in the multislice) in the
            fifth entry and mean squared atomic displacement in the sixth entry"""

    def __init__(
        self,
        fnam,
        temperature_factor_units="urms",
        atomic_coordinates="fractional",
        EPS=1e-2,
        psuedo_rational_tiling=1e-2,
    ):
        """Initializes a crystal object by reading in a *.p1 file, which is
        outputted by the vesta software:

        K. Momma and F. Izumi, "VESTA 3 for three-dimensional visualization of
        crystal, volumetric and morphology data," J. Appl. Crystallogr., 44,
        1272-1276 (2011).

        or a prismatic style *.xyz file
        """
        f = open(fnam, "r")

        ext = splitext(fnam)[1]

        # Read title
        self.Title = f.readline().strip()

        if ext == ".p1":

            # I have no idea what the second line in the p1 file format means
            # so ignore it
            f.readline()

            # Get unit cell vector - WARNING assume an orthorhombic unit cell
            self.unitcell = np.loadtxt(f, max_rows=3, dtype=np.float)

            # Check to see if unit cell is orthorhombic
            ortho = np.abs(np.sum(self.unitcell) - np.trace(self.unitcell)) < EPS

            # Get the atomic symbol of each element
            self.atomtypes = np.loadtxt(f, max_rows=1, dtype=str, ndmin=1)

            # Get atomic number from lookup table
            Zs = [
                atomic_symbol.index(self.atomtypes[i].strip())
                for i in range(self.atomtypes.shape[0])
            ]

            # Get the number of atoms of each type
            self.natoms = np.loadtxt(f, max_rows=1, dtype=int, ndmin=1)

            # Skip empty line
            f.readline()

            # Total number of atoms
            totnatoms = np.sum(self.natoms)
            # Intialize array containing atomic information
            self.atoms = np.zeros((totnatoms, 6))

            for i in range(totnatoms):
                atominfo = split(r"\s+", f.readline().strip())[:6]
                # First three entries are the atomic coordinates
                self.atoms[i, :3] = np.asarray(atominfo[:3], dtype=np.float)
                # Fourth entry is the atomic symbol
                self.atoms[i, 3] = atomic_symbol.index(
                    match("([A-Za-z]+)", atominfo[3]).group(0)
                )
                # Final entries are the fractional occupancy and the temperature (Debye-Waller)
                # factor
                self.atoms[i, 4:6] = np.asarray(atominfo[4:6], dtype=np.float)

            if ortho:
                # If unit cell is orthorhombic then extract unit cell
                # dimension
                self.unitcell = np.diag(self.unitcell)
            else:
                # If not orthorhombic attempt psuedo rational tiling
                # (only implemented for hexagonal structures)
                self.orthorhombic_supercell(EPS=EPS)

        elif ext == ".xyz":
            # Read in unit cell dimensions
            self.unitcell = np.asarray(
                [float(x) for x in split(r"\s+", f.readline().strip())[:3]]
            )

            atoms = []
            for line in f:

                # Look for end of file marker
                if line.strip() == "-1":
                    break
                # Otherwise parse line
                atoms.append(
                    np.array([float(x) for x in split(r"\s+", line.strip())[:6]])
                )

            # Now stack all atoms into numpy array
            atoms = np.stack(atoms, axis=0)

            # Rearrange columns of numpy array to match standard
            totnatoms = atoms.shape[0]
            self.atoms = np.zeros((totnatoms, 6))
            self.atoms[:, :3] = atoms[:, 1:4]
            self.atoms[:, 3] = atoms[:, 0]
            self.atoms[:, 4:6] = atoms[:, 4:6]
        else:
            print("File extension: {0} not recognized".format(ext))
            return None
        # If temperature factors are given as B then convert to urms
        if temperature_factor_units == "B":
            self.atoms[:, 5] /= 8 * np.pi ** 2
        elif temperature_factor_units == 'sqrturms':
            self.atoms[:, 5] = self.atoms[:,5] ** 2

        # If necessary, Convert atomic positions to fractional coordinates
        if atomic_coordinates == "cartesian":
            self.atoms[:, :3] /= self.unitcell[:3][np.newaxis, :]

    def orthorhombic_supercell(self, EPS):
        # If not orthorhombic attempt psuedo rational tiling
        # (only implemented for hexagonal structures)
        tiley = int(np.round(np.abs(self.unitcell[1, 0] / self.unitcell[0, 0]) / EPS))
        tilex = int(np.round(1 / EPS))

        # Remove common denominators
        from math import gcd

        g_ = gcd(tiley, tilex)
        while g_ > 1:
            tiley = tiley // g_
            tilex = tilex // g_
            g_ = gcd(tiley, tilex)

        # Make deepcopy of old unit cell
        import copy

        olduc = copy.deepcopy(self.unitcell)

        # Calculate length of unit cell sides
        self.unitcell = np.sqrt(np.sum(np.square(self.unitcell), axis=1))

        # Tile out atoms
        self.tile(tiley, tilex, 1)

        # Calculate size of old unit cell under tiling
        olduc = np.asarray([tiley, tilex, 1])[:, np.newaxis] * olduc

        self.unitcell = np.diag(olduc)

        # Now calculate fractional coordinates in new orthorhombic cell
        self.atoms[:, :3] = np.mod(
            self.atoms[:, :3] @ olduc @ np.diag(1 / self.unitcell), 1.0
        )

    def quickplot(self, atomscale=None, cmap=plt.get_cmap("Dark2")):
        """Makes a quick 3D scatter plot of the crystal"""
        from mpl_toolkits.mplot3d import Axes3D

        if atomscale is None:
            atomscale = 1e-3 * np.amax(self.unitcell)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        colors = cmap(self.atoms[:, 3] / np.amax(self.atoms[:, 3]))
        sizes = self.atoms[:, 3] ** (4) * atomscale
        ax.scatter(*[self.atoms[:, i] for i in range(3)], c=colors, s=sizes)
        for fun in [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]:
            fun(0, 1.0)
        plt.show(block=True)

    def output_vesta_xtl(self, fnam):
        """Outputs an .xtl file which is viewable by the vesta software:

        K. Momma and F. Izumi, "VESTA 3 for three-dimensional visualization of
        crystal, volumetric and morphology data," J. Appl. Crystallogr., 44,
        1272-1276 (2011).
        """
        f = open(splitext(fnam)[0] + ".xtl", "w")
        f.write("TITLE " + self.Title + "\n CELL \n")
        f.write("  {0:.2f} {1:.2f} {2:.2f} 90 90 90\n".format(*self.unitcell))
        f.write("SYMMETRY NUMBER 1\n SYMMETRY LABEL  P1\n ATOMS \n")
        f.write("NAME         X           Y           Z" + "\n")
        for i in range(self.atoms.shape[0]):
            f.write(
                "{0} {1:.4f} {2:.4f} {3:.4f}\n".format(
                    atomic_symbol[int(self.atoms[i, 3])], *self.atoms[i, :3]
                )
            )
        f.write("EOF")
        f.close()

    def output_xyz(self, fnam, atomic_coordinates="cartesian"):
        f = open(splitext(fnam)[0] + ".xyz", "w")
        f.write(self.Title + "\n {0:.4f} {1:.4f} {2:.4f}\n".format(*self.unitcell))
        for atom in self.atoms:
            f.write(
                "{0:d} {1:.4f} {2:.4f} {3:.4f} {4:.2f}  {5:.3f}\n".format(
                    int(atom[3]), *(atom[:3] * self.unitcell), *atom[4:6]
                )
            )
        f.write("-1")
        f.close()

    def make_transmission_functions(
        self,
        pixels,
        eV,
        subslices=[1.0],
        tiling=[1, 1],
        fe=None,
        displacements=True,
        fftout=True,
        dtype=None,
        device=None,
        fractional_occupancy=True,
    ):
        """Make the transmission functions for this crystal, which are the
           exponential of the specimen potential scaled by the interaction
           constant for electrons, sigma."""

        # Make the specimen electrostatic potential
        T = self.make_potential(
            pixels,
            subslices,
            tiling,
            fe=fe,
            displacements=displacements,
            device=device,
            dtype=dtype,
            fractional_occupancy=fractional_occupancy,
        )

        # Now take the complex exponential of the electrostatic potential
        # scaled by the electron interaction constant
        T = torch.fft(torch_c_exp(interaction_constant(eV) * T), signal_ndim=2)

        # Band-width limit the transmission function, see Earl Kirkland's book
        # for an discussion of why this is necessary
        for i in range(T.shape[0]):
            T[i] = bandwidth_limit_array(T[i])

        if fftout:
            return torch.ifft(T, signal_ndim=2)
        return T

    def make_potential(
        self,
        pixels,
        subslices=[1.0],
        tiling=[1, 1],
        bandwidth_limit=True,
        displacements=True,
        fractional_occupancy=True,
        fe=None,
        device=None,
        dtype=torch.float32,
    ):
        """Calculates the projected electrostatic potential for a 
            crystal on a pixel grid with dimensions specified by array 
            pixels. Subslicing the unit cell is achieved by passing an 
            array subslices that contains as its entries the depths at 
            which each subslice should be terminated in units of 
            fractional coordinates. Tiling of the unit cell (often
            necessary to make a sufficiently large simulation grid to 
            fit the probe) is achieved by passing the tiling factors in 
            the array tiling.
            """

        # Initialize device cuda if available, CPU if no cuda is available
        device = get_device(device)

        tiling_ = np.asarray(tiling[:2])
        gsize = np.asarray(self.unitcell[:2]) * tiling_
        psize = np.asarray(pixels)

        pixperA = np.asarray(pixels) / np.asarray(self.unitcell[:2]) / tiling_

        # Get a list of unique atomic elements
        elements = list(set(np.asarray(self.atoms[:, 3], dtype=np.int)))

        # Get number of unique atomic elements
        nelements = len(elements)
        nsubslices = len(subslices)
        # Build list of equivalent sites
        if fractional_occupancy: 
            equivalent_sites = find_equivalent_sites(self.atoms[:, :3], EPS=1e-3)

        # FDES method
        # Intialize potential array
        P = torch.zeros(
            np.prod([nelements, nsubslices, *pixels, 2]), device=device, dtype=dtype
        )

        # Construct a map of which atom corresponds to which slice
        islice = np.zeros((self.atoms.shape[0]), dtype=np.int)
        slice_stride = np.prod(pixels) * 2
        if nsubslices > 1:
            # Finds which slice atom can be found in
            # WARNING Assumes that the slices list ends with 1.0 and is in
            # ascending order
            for i in range(nsubslices):
                zmin = 0 if i == 0 else subslices[i - 1]
                atoms_in_slice = (self.atoms[:, 2] % 1.0 >= zmin) & (
                    self.atoms[:, 2] % 1.0 < subslices[i]
                )
                islice[atoms_in_slice] = i * slice_stride
            islice = torch.from_numpy(islice).type(torch.long).to(device)
        else:
            islice = 0
        # Make map a pytorch Tensor

        # Construct a map of which atom corresponds to which element
        element_stride = nsubslices * slice_stride
        ielement = (
            torch.tensor(
                [
                    elements.index(int(self.atoms[iatom, 3]))
                    for iatom in range(self.atoms.shape[0])
                ],
                dtype=torch.long,
                device=device,
            )
            * element_stride
        )

        if displacements:
            # Generate thermal displacements
            urms = torch.tensor(
                np.sqrt(self.atoms[:, 5])[:, np.newaxis] * pixperA[np.newaxis, :],
                dtype=P.dtype,
                device=device,
            ).view(self.atoms.shape[0], 2)

        # FDES algorithm implemented using the pytorch scatter_add function,
        # which takes a list of numbers and adds them to a corresponding list
        # of coordinates
        for tile in range(tiling[0] * tiling[1]):
            # For these atomic coordinates (in fractional coordinates) convert
            # to pixel coordinates
            posn = (
                (
                    self.atoms[:, :2]
                    + np.asarray([tile % tiling[0], tile // tiling[0]])[np.newaxis, :]
                )
                / tiling_
                * psize
            )
            posn = torch.from_numpy(posn).to(device).type(P.dtype)

            if displacements:
                # Add displacement sampled from normal distribution to account
                # for atomic thermal motion
                disp = (
                    torch.randn(self.atoms.shape[0], 2, dtype=P.dtype, device=device)
                    * urms
                )

                # If using fractional occupancy force atoms occupying equivalent
                # sites to have the same displacement
                if fractional_occupancy:
                    disp = disp[equivalent_sites, :]

                posn[:, :2] += disp

            yc = (
                torch.remainder(torch.ceil(posn[:, 0]).type(torch.long), pixels[0])
                * pixels[1]
                * 2
            )
            yf = (
                torch.remainder(torch.floor(posn[:, 0]).type(torch.long), pixels[0])
                * pixels[1]
                * 2
            )
            xc = torch.remainder(torch.ceil(posn[:, 1]).type(torch.long), pixels[1]) * 2
            xf = (
                torch.remainder(torch.floor(posn[:, 1]).type(torch.long), pixels[1]) * 2
            )

            yh = torch.remainder(posn[:, 0], 1.0)
            yl = 1.0 - yh
            xh = torch.remainder(posn[:, 1], 1.0)
            xl = 1.0 - xh

            # Account for fractional occupancy of atomic sites if requested
            if fractional_occupancy:
                xh *= torch.from_numpy(self.atoms[:, 4]).type(P.dtype).to(device)
                xl *= torch.from_numpy(self.atoms[:, 4]).type(P.dtype).to(device)

            # Each pixel is set to the overlap of a shifted rectangle in that pixel
            P.scatter_add_(0, ielement + islice + yc + xc, yh * xh)
            P.scatter_add_(0, ielement + islice + yc + xf, yh * xl)
            P.scatter_add_(0, ielement + islice + yf + xc, yl * xh)
            P.scatter_add_(0, ielement + islice + yf + xf, yl * xl)

        # Now view potential as a 4D array for next bit
        P = P.view(nelements, nsubslices, *pixels, 2)

        # FFT potential to reciprocal space
        for i in range(P.shape[0]):
            P[i] = torch.fft(P[i], signal_ndim=2)

        # Make sinc functions with appropriate singleton dimensions for pytorch
        # broadcasting /gridsize[0]*pixels[0] /gridsize[1]*pixels[1]
        sincy = (
            sinc(torch.from_numpy(np.fft.fftfreq(pixels[0])))
            .view([1, 1, pixels[0], 1, 1])
            .to(device)
            .type(P.dtype)
        )
        sincx = (
            sinc(torch.from_numpy(np.fft.fftfreq(pixels[1])))
            .view([1, 1, 1, pixels[1], 1])
            .to(device)
            .type(P.dtype)
        )
        # #Divide by sinc functions
        P /= sincy
        P /= sincx

        # Option to precalculate scattering factors and pass to program which
        # saves computation for
        if fe is None:
            fe_ = calculate_scattering_factors(psize, gsize, elements)
        else:
            fe_ = fe

        # Convolve with electron scattering factors using Fourier convolution theorem
        P *= torch.from_numpy(fe_).view(nelements, 1, *pixels, 1).to(device)

        norm = np.prod(pixels) / np.prod(self.unitcell[:2]) / np.prod(tiling)
        # Add atoms together
        P = norm * torch.sum(P, dim=0)

        # Only return real part
        return torch.ifft(P, signal_ndim=2)[..., 0]

    def rotate(self, theta, axis, origin=[0.5, 0.5, 0.5]):
        """Returns a copy of the crystal rotated an angle theta in radians about an axis and
        """
        new = copy.deepcopy(self)

        # Make rotation matrix, R, and  the point about which we rotate, O
        R = rot_matrix(theta, axis)
        O = np.asarray(origin)

        # Get atomic coordinates in cartesian (not fractional coordinates)
        new.atoms[:, :3] = self.atoms[:, :3] * self.unitcell[np.newaxis, :]

        # Apply rotation matrix to each atom coordinate
        new.atoms[:, :3] = (
            R @ (new.atoms[:, :3].T - O[:, np.newaxis]) + O[:, np.newaxis]
        ).T

        # Get new unit cell (assume vacuum padding)
        origin = np.amin(new.atoms, axis=0)
        new.unitcell = np.amax(new.atoms, axis=0) - origin
        new.atoms[:, :3] = (new.atoms[:, :3] - origin[np.newaxis, :3]) / new.unitcell[
            np.newaxis, :3
        ]

        # Return rotated crystal
        return new

    def rot90(self, k=1, axes=(0, 1)):
        """Rotates a crystal by 90 degrees in the plane specified by axes.

        Rotation direction is from the first towards the second axis.

        Parameters
        ----------
        k : integer
            Number of times the crystal is rotated by 90 degrees.
        axes: (2,) array_like
            The array is rotated in the plane defined by the axes.
            Axes must be different."""

        # Much of the following is adapted from the numpy.rot90 function
        axes = tuple(axes)
        if len(axes) != 2:
            raise ValueError("len(axes) must be 2.")

        k %= 4

        if k == 0:
            # Do nothing
            return
        if k == 2:
            # Reflect in both axes
            self.reflect(axes)
            return

        axes_list = np.arange(0, 3)
        (axes_list[axes[0]], axes_list[axes[1]]) = (
            axes_list[axes[1]],
            axes_list[axes[0]],
        )

        if k == 1:
            self.reflect([axes[1]])
            self.transpose(axes_list)
        else:
            # k == 3
            self.transpose(axes_list)
            self.reflect([axes[1]])

    def transpose(self, axes):
        self.atoms[:, :3] = self.atoms[:, axes]
        self.unitcell = self.unitcell[axes]

    def tile(self, x=1, y=1, z=1):
        """tiles the crystal out"""
        # Make copy of original crystal
        # new = copy.deepcopy(self)

        tiling = np.asarray([x, y, z])

        # Get atoms in unit cell
        natoms = self.atoms.shape[0]

        # Initialize new atom list
        newatoms = np.zeros((natoms * x * y * z, 6))

        # Calculate new unit cell size
        self.unitcell *= np.asarray([x, y, z])

        # tile out the integer amounts
        for j in range(int(x)):
            for k in range(int(y)):
                for l in range(int(z)):

                    # Calculate origin of this particular tile
                    origin = np.asarray([j, k, l])

                    # Calculate index of this particular tile
                    indx = j * int(y) * int(z) + k * int(z) + l

                    # Add new atoms to unit cell
                    newatoms[indx * natoms : (indx + 1) * natoms, :3] = (
                        self.atoms[:, :3] + origin[np.newaxis, :]
                    ) / tiling[np.newaxis, :]

                    # Copy other information about atoms
                    newatoms[indx * natoms : (indx + 1) * natoms, 3:] = self.atoms[
                        :, 3:
                    ]
        self.atoms = newatoms

    def concatenate_crystals(self, other, axis=2, side=1, eps=1e-3):
        """adds other crystal to the crystal object slice is added to the bottom (top being z =0)
        only works if slices are the same size or their x and y dimensions are integer multiples of
        each other """
        # Make deep copies of the crystal object and the slice
        # this is so that these objects remain untouched by the operation
        # of this function
        new = copy.deepcopy(self)
        other_ = copy.deepcopy(other)

        # Check if the two slices are the same size and
        # tile accordingly
        for ax in range(3):
            # If this axis is the concatenation axis, then it's not necessary
            # that the crystals are the same size
            if ax == axis:
                continue

            factor = self.unitcell[ax] / other.unitcell[ax]
            tile = [0, 0, 0]

            if factor > 1:
                tile[ax] = int(factor)
                other_ = other_.tile(*tile)
            elif factor < 1:
                tile[ax] = int(1 / factor)
                new = new.tile(*tile)

        axes = np.arange(3) != axis
        assert np.all(
            np.abs(new.unitcell[axes] - other_.unitcell[axes]) < eps
        ), "Crystal axes mismatch"

        # Update the thickness of the resulting
        # crystal object.
        new.unitcell[axis] = self.unitcell[axis] + other_.unitcell[axis]

        # Adjust fractional coordinates of atoms
        new.atoms[:, axis] /= new.unitcell[axis] / self.unitcell[axis]
        other_.atoms[:, axis] /= new.unitcell[axis] / other_.unitcell[axis]

        if side == 0:
            new.atoms[:, axis] += self.unitcell[axis] / new.unitcell[axis]
        else:
            other_.atoms[:, axis] += self.unitcell[axis] / new.unitcell[axis]
        new.atoms = np.concatenate([new.atoms, other_.atoms], axis=0)

        return new

    def reflect(self, axes):
        """Reflect crystal in each of the axes enumerated in list axes"""
        for ax in axes:
            self.atoms[:, ax] = 1 - self.atoms[:, ax]

    def slice(self, range, axis):
        """Make a slice of crystal object ranging from range[0] to range[1] through
        specified axis"""

        # Work out which atoms will stay in the sliced structure
        mask = np.logical_and(
            self.atoms[:, axis] >= range[0], self.atoms[:, axis] <= range[1]
        )

        # Make a copy of the crystal
        new = copy.deepcopy(self)

        # Put remaining atoms back in
        new.atoms = self.atoms[mask, :]

        # Adjust unit cell dimensions
        new.unitcell[axis] = (range[1] - range[0]) * self.unitcell[axis]

        # Adjust origin of atomic coordinates
        origin = np.zeros((3))
        origin[axis] = range[0]
        new.atoms[:, axis] = (
            (new.atoms[:, axis] - range[0]) * self.unitcell[axis] / new.unitcell[axis]
        )

        # Return modified crystal structure
        return new
