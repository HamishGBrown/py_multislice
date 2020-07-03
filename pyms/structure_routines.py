"""Structure module.

A collection of functions and classes for reading in and manipulating structures
and creating potential arrays for multislice simulation.
"""
import itertools
import ase
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy
from re import split, match
from os.path import splitext
from .atomic_scattering_params import e_scattering_factors, atomic_symbol
from .Probe import wavev, relativistic_mass_correction
from .utils.numpy_utils import bandwidth_limit_array, q_space_array, ensure_array
from .utils.torch_utils import sinc, torch_c_exp, get_device, cx_from_numpy


def remove_common_factors(nums):
    """Remove common divisible factors from a set of numbers."""
    nums = np.asarray(nums, dtype=np.int)
    g_ = np.gcd.reduce(nums)
    while g_ > 1:
        nums //= g_
        g_ = np.gcd.reduce(nums)
    return nums


def psuedo_rational_tiling(dim1, dim2, EPS):
    """
    Calculate the psuedo-rational tiling for matching objects of different dimensions.

    For two dimensions, dim1 and dim2, work out the multiplicative
    tiling so that those dimensions might be matched to within error EPS.
    """
    if np.any([dim1 == 0, dim2 == 0]):
        return 1, 1
    tile1 = int(np.round(np.abs(dim2 / dim1) / EPS))
    tile2 = int(np.round(1 / EPS))
    return remove_common_factors([tile1, tile2])


def Xray_scattering_factor(Z, gsq, units="A"):
    """
    Calculate the X-ray scattering factor for atom with atomic number Z.

    Parameters
    ----------
    Z : int
        Atomic number of atom of interest.
    gsq : float or array_like
        Reciprocal space value(s) in Angstrom squared at which to evaluate the
        X-ray scattering factor.
    units : string, optional
        Units in which to calculate X-ray scattering factor, can be 'A' for
        Angstrom, or 'VA' for volt-Angstrom.
    """
    # Bohr radius in Angstrom
    a0 = 0.529177
    # gsq = g**2
    return Z - 2 * np.pi ** 2 * a0 * gsq * electron_scattering_factor(
        Z, gsq, units=units
    )


def electron_scattering_factor(Z, gsq, units="VA"):
    """
    Calculate the electron scattering factor for atom with atomic number Z.

    Parameters
    ----------
    Z : int
        Atomic number of atom of interest.
    gsq : float or array_like
        Reciprocal space value(s) in Angstrom squared at which to evaluate the
        electron scattering factor.
    units : string, optional
        Units in which to calculate electron scattering factor, can be 'A' for
        Angstrom, or 'VA' for volt-Angstrom.
    """
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
    """Calculate the electron scattering factors on a reciprocal space grid.

    Parameters
    ----------
    gridshape : (2,) array_like
        pixel size of the grid
    gridsize : (2,) array_like
        Lateral real space sizing of the grid in Angstrom
    elements : (M,) array_like
        List of elements for which electron scattering factors are required

    Returns
    -------
    fe : (M, *gridshape)
        Array of electron scattering factors in reciprocal space for each
        element
    """
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
    """Find equivalent atomic sites in a list of atomic positions object.

    This function is used to detect two atoms sharing the same postions (are
    with EPS of each other) with fractional occupancy, and return an index of
    these equivalent sites.
    """
    # Import  the pair-wise distance function from scipy
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
    """
    Calculate the electron interaction constant, sigma.

    The electron interaction constant converts electrostatic potential (in V
    Angstrom) to radians. Units of this constant are rad/(V Angstrom).  See
    Eq. (2.5) in Kirkland's Advanced Computing in electron microscopy.
    """
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


def change_of_basis(coords, newuc, olduc):
    """Change of basis for structure unit cell."""
    return np.mod(coords[:, :3] @ olduc @ np.linalg.inv(newuc), 1.0)


def rot_matrix(theta, u=np.asarray([0, 0, 1], dtype=np.float)):
    """
    Generate a 3D rotational matrix.

    Parameters
    ----------
    theta : float
        Angle of rotation in radians
    u : (3,) array_like
        Axis of rotation
    """
    from numpy import sin, cos

    c = cos(theta)
    s = sin(theta)
    ux, uy, uz = u / np.linalg.norm(u)
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


class structure:
    """
    Class for simulation objects.

    Elements in a structure object:
    unitcell :
        An array containing the side lengths of the orthorhombic unit cell
    atoms :
        An array of dimensions total number of atoms by 6 which for each atom
        contains the fractional cooordinates within the unit cell for each atom
        in the first three entries, the atomic number in the fourth entry,
        the atomic occupancy (not yet implemented in the multislice) in the
        fifth entry and mean squared atomic displacement in the sixth entry
    Title :
        Short description of the object of output purposes
    """

    def __init__(self, unitcell, atoms, dwf, occ=None, Title="", EPS=1e-2):
        """Initialize a simulation object with necessary variables."""
        self.unitcell = np.asarray(unitcell)
        natoms = np.asarray(atoms).shape[0]

        if occ is None:
            occ = np.ones(natoms)

        self.atoms = np.concatenate(
            [atoms, occ.reshape(natoms, 1), np.asarray(dwf).reshape(natoms, 1)], axis=1
        )
        self.Title = Title

        # Up till now unitcell can be a 3 x 3 matrix with rows describing the
        # unit cell edges. If this is the case we need to make sure that the
        # unit cell is orthorhombic and find an orthorhombic tiling if this it
        # is not orthorhombic
        if self.unitcell.ndim > 1:
            # Check to see if unit cell is orthorhombic
            ortho = np.abs(np.sum(self.unitcell) - np.trace(self.unitcell)) < EPS

            if ortho:
                # If unit cell is orthorhombic then extract unit cell
                # dimension
                self.unitcell = np.diag(self.unitcell)
            else:
                # If not orthorhombic attempt psuedo rational tiling
                self.orthorhombic_supercell(EPS=EPS)

        # Check if there is any fractional occupancy of atom sites in
        # the sample
        self.fractional_occupancy = np.any(np.abs(self.atoms[:, 4] - 1.0) > 1e-3)

    @classmethod
    def fromfile(
        cls,
        fnam,
        temperature_factor_units="ums",
        atomic_coordinates="fractional",
        EPS=1e-2,
        T=None,
    ):
        """
        Read in a simulation object from a structure file.

        Appropriate structure files include *.p1 files, which is outputted by
        the vesta software:

        K. Momma and F. Izumi, "VESTA 3 for three-dimensional visualization of
        crystal, volumetric and morphology data," J. Appl. Crystallogr., 44,
        1272-1276 (2011).

        or a *.xyz file in the standard of the prismatic software

        Parameters
        ----------
        fnam : string
            Filepath of the structure file
        temperature_factor_units : string,optional
            Units of the Debye-Waller temperature factors in the structure file
            appropriate inputs are B (crystallographic temperature factor),
            urms (root mean squared displacement) and ums (mean squared
            displacement, the default)
        atomic_coordinates : string, optional
            Units of the atomic coordinates can be "fractional" or "cartesian"
        EPS : float,optional
            Tolerance for procedures such as psuedo-rational tiling for
            non-orthorhombic crystal unit cells
        T : (3,3) array_like or None
            An optional transformation matrix to be applied to the unit cell
            and the atomic coordinates
        """
        f = open(fnam, "r")

        ext = splitext(fnam)[1]

        # Read title
        Title = f.readline().strip()

        if ext == ".p1":

            # I have no idea what the second line in the p1 file format means
            # so ignore it
            f.readline()

            # Get unit cell vector - WARNING assume an orthorhombic unit cell
            unitcell = np.loadtxt(f, max_rows=3, dtype=np.float)

            # Get the atomic symbol of each element
            atomtypes = np.loadtxt(f, max_rows=1, dtype=str, ndmin=1)  # noqa

            # Get the number of atoms of each type
            natoms = np.loadtxt(f, max_rows=1, dtype=int, ndmin=1)

            # Skip empty line
            f.readline()

            # Total number of atoms
            totnatoms = np.sum(natoms)
            # Intialize array containing atomic information
            atoms = np.zeros((totnatoms, 6))
            dwf = np.zeros((totnatoms,))
            occ = np.zeros((totnatoms,))

            for i in range(totnatoms):
                atominfo = split(r"\s+", f.readline().strip())[:6]
                # First three entries are the atomic coordinates
                atoms[i, :3] = np.asarray(atominfo[:3], dtype=np.float)
                # Fourth entry is the atomic symbol
                atoms[i, 3] = atomic_symbol.index(
                    match("([A-Za-z]+)", atominfo[3]).group(0)
                )
                # Final entries are the fractional occupancy and the temperature
                # (Debye-Waller) factor
                occ[i] = atominfo[4]
                dwf[i] = atominfo[5]

        elif ext == ".xyz":
            # Read in unit cell dimensions
            unitcell = np.asarray(
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
            atoms_ = np.stack(atoms, axis=0)

            # Rearrange columns of numpy array to match standard
            totnatoms = atoms_.shape[0]
            atoms = np.zeros((totnatoms, 4))
            # Atomic coordinates
            atoms[:, :3] = atoms_[:, 1:4]
            # Atomic numbers (Z)
            atoms[:, 3] = atoms_[:, 0]
            # Fractional occupancy and Debye-Waller (temperature) factor
            dwf = atoms_[:, 5]
            occ = atoms_[:, 4]
        else:
            print("File extension: {0} not recognized".format(ext))
            return None

        # Close file
        f.close()

        # If temperature factors are given in any other format than mean square
        # (ums) convert to mean square. Acceptable formats are crystallographic
        # temperature factor B and root mean square (urms) displacements
        if temperature_factor_units == "B":
            dwf /= 8 * np.pi ** 2
        elif temperature_factor_units == "urms":
            dwf = dwf ** 2

        # If necessary, Convert atomic positions to fractional coordinates
        if atomic_coordinates == "cartesian":
            atoms[:, :3] /= unitcell[:3][np.newaxis, :]
            atoms[:, :3] = atoms[:, :3] % 1.0

        if T is not None:
            # Transform atoms to cartesian basis and then apply transformation
            # matrix
            atoms[:, :3] = (T @ unitcell @ atoms[:, :3].T).T

            # Apply transformation matrix to unit-cell
            unitcell = unitcell @ T.T

            # Apply inverse of unit cell
            atoms[:, :3] = (np.linalg.inv(unitcell) @ atoms[:, :3].T).T

        return cls(unitcell, atoms[:, :4], dwf, occ, Title, EPS=EPS)

    @classmethod
    def from_ase_cluster(cls, asecell, occupancy=None, Title="", dwf=None):
        """Initialize from Atomic Simulation Environment (ASE) cluster object."""
        unitcell = asecell.cell[:]
        natoms = asecell.numbers.shape[0]
        atoms = np.concatenate(
            [
                asecell.cell.scaled_positions(asecell.positions),
                asecell.numbers.reshape(natoms, 1),
            ],
            axis=1,
        )
        if occupancy is None:
            occ = np.ones(natoms)
        if dwf is None:
            dwf = np.ones(natoms) * 1 / np.pi ** 2 / 8
        return cls(unitcell, atoms, dwf, occ, Title)

    def to_ase_atoms(self):
        """Convert structure to Atomic Simulation Environment (ASE) atoms object."""
        scaled_positions = self.atoms[:, :3]
        numbers = self.atoms[:, 3].astype(np.int)
        cell = self.unitcell
        pbc = [True, True, True]
        return ase.Atoms(
            scaled_positions=scaled_positions, numbers=numbers, cell=cell, pbc=pbc
        )

    def orthorhombic_supercell(self, EPS=1e-2):
        """
        Create an orthorhombic supercell from a monoclinic crystal unit cell.

        If not orthorhombic attempt psuedo rational tiling of general
        monoclinic structure. Assumes that the self.unitcell matrix is lower
        triangular.
        """
        if not np.abs(np.dot(self.unitcell[0], self.unitcell[1])) < EPS:
            tiley, tilex = psuedo_rational_tiling(*self.unitcell[0:2, 0], EPS)

            # Make deepcopy of old unit cell

            olduc = copy.deepcopy(self.unitcell)

            # Tile out atoms
            self.tile(tiley, tilex, 1)

            # Calculate size of old unit cell under tiling
            olduc = np.asarray([tiley, tilex, 1])[:, np.newaxis] * olduc

            self.unitcell = copy.deepcopy(olduc)
            self.unitcell[1, 0] = 0.0

            # Now calculate fractional coordinates in new orthorhombic cell
            self.atoms[:, :3] = change_of_basis(self.atoms[:, :3], self.unitcell, olduc)
        else:
            self.unitcell[0, 1:] = 0.0
            self.unitcell[1, ::2] = 0.0

        # Now tile crystal in x and y
        tilez1, tiley = psuedo_rational_tiling(*self.unitcell[::-2, 0], EPS)
        tilez2, tilex = psuedo_rational_tiling(*self.unitcell[3:0:-1, 1], EPS)
        tilez = remove_common_factors([tilez1, tilez2, tilez1 * tilez2])[-1]
        tiley *= tilez // tilez1
        tilex *= tilez // tilez2

        olduc = copy.deepcopy(self.unitcell)

        # Tile out atoms
        self.tile(tiley, tilex, tilez)

        # Calculate size of old unit cell under tiling

        olduc = np.asarray([tiley, tilex, tilez])[:, np.newaxis] * olduc

        self.unitcell = copy.deepcopy(olduc)
        self.unitcell[2, 0:2] = 0.0

        # Now calculate fractional coordinates in new orthorhombic cell
        self.atoms[:, :3] = np.mod(
            self.atoms[:, :3] @ olduc @ np.linalg.inv(self.unitcell), 1.0
        )
        self.unitcell = np.diag(self.unitcell)

        # Check for negative values of self.unitcell and rectify
        for i in range(3):
            if self.unitcell[i] < 0:
                self.atoms[:, i] = (1.0 - self.atoms[:, i]) % 1.0
        self.unitcell = np.abs(self.unitcell)

    def quickplot(
        self, atomscale=None, cmap=plt.get_cmap("Dark2"), block=True, colors=None
    ):
        """
        Make a quick 3D scatter plot of the atomic sites within the structure.

        For more detailed visualization output the structure file to a file format
        readable by the Vesta software using output_vesta_xtl
        """
        from mpl_toolkits.mplot3d import Axes3D  # NOQA

        if atomscale is None:
            atomscale = 1e-3 * np.amax(self.unitcell)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        if colors is None:
            colors = cmap(self.atoms[:, 3] / np.amax(self.atoms[:, 3]))
        sizes = self.atoms[:, 3] * atomscale

        ax.scatter(
            *[self.atoms[:, i] * self.unitcell[i] for i in [1, 0, 2]], c=colors, s=sizes
        )

        ax.set_xlim3d(0.0, self.unitcell[1])
        ax.set_ylim3d(top=0.0, bottom=self.unitcell[0])
        ax.set_zlim3d(top=0.0, bottom=self.unitcell[2])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        plt.show(block=block)
        return fig

    def output_vesta_xtl(self, fnam):
        """Output an .xtl file which is viewable by the vesta software.

        See K. Momma and F. Izumi, "VESTA 3 for three-dimensional visualization
        of crystal, volumetric and morphology data," J. Appl. Crystallogr., 44,
        1272-1276 (2011).
        """
        f = open(splitext(fnam)[0] + ".xtl", "w")
        f.write("TITLE " + self.Title + "\n CELL \n")
        f.write("  {0:.5f} {1:.5f} {2:.5f} 90 90 90\n".format(*self.unitcell))
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

    def output_xyz(
        self, fnam, atomic_coordinates="cartesian", temperature_factor_units="sqrturms"
    ):
        """Output an .xyz structure file."""
        f = open(splitext(fnam)[0] + ".xyz", "w")
        f.write(self.Title + "\n {0:.4f} {1:.4f} {2:.4f}\n".format(*self.unitcell))

        if atomic_coordinates == "cartesian":
            coords = self.atoms[:, :3] * self.unitcell
        else:
            coords = self.atoms[:, :3]

        # If temperature factors are given as B then convert to urms
        if temperature_factor_units == "B":
            DWFs = self.atoms[:, 5] * 8 * np.pi ** 2
        elif temperature_factor_units == "sqrturms":
            DWFs = np.sqrt(self.atoms[:, 5])

        for coord, atom, DWF in zip(coords, self.atoms, DWFs):
            f.write(
                "{0:d} {1:.4f} {2:.4f} {3:.4f} {4:.2f}  {5:.3f}\n".format(
                    int(atom[3]), *coord, atom[4], DWF
                )
            )
        f.write("-1")
        f.close()

    def make_potential(
        self,
        pixels,
        subslices=[1.0],
        tiling=[1, 1],
        displacements=True,
        fractional_occupancy=True,
        fe=None,
        device=None,
        dtype=torch.float32,
        seed=None,
    ):
        """
        Generate the projected potential of the structure.

        Calculate the projected electrostatic potential for a structure on a
        pixel grid with dimensions specified by pixels. Subslicing the unit
        cell is achieved by passing an array subslices that contains as its
        entries the depths at which each subslice should be terminated in units
        of fractional coordinates. Tiling of the unit cell (often necessary to
        make a sufficiently large simulation grid to fit the probe) is achieved
        by passing the tiling factors in the array tiling.

        Parameters
        ----------
        pixels: int, (2,) array_like
            The pixel size of the grid on which to calculate the projected
            potentials
        subslices: float, array_like, optional
            An array containing the depths at which each slice ends as a fraction
            of the simulation unit-cell
        tiling: int, (2,) array_like, optional
            Tiling of the simulation object (often necessary to  make a
            sufficiently large simulation grid to fit the probe)
        displacements: bool, optional
            Pass displacements = False to turn off random displacements of the
            atoms due to thermal motion
        fractional_occupancy: bool, optional
            Pass fractional_occupancy = False to turn off fractional occupancy
            of atomic sites
        fe: float, array_like
            An array containing the electron scattering factors for the elements
            in the structure as calculated by the function
            calculate_scattering_factors, can be passed to save recalculating
            each time new potentials are generated
        device: torch.device
            Allows the user to control which device the calculations will occur
            on
        dtype: torch.dtype
            Controls the data-type of the output
        seed: int
            Seed for random number generator for atomic displacements.
        """
        # Initialize device cuda if available, CPU if no cuda is available
        device = get_device(device)

        # Ensure pixels is integer
        pixels_ = [int(x) for x in pixels]

        # Seed random number generator for displacements
        if seed is not None:
            torch.manual_seed(seed)

        tiling_ = np.asarray(tiling[:2])
        gsize = np.asarray(self.unitcell[:2]) * tiling_
        psize = np.asarray(pixels_)

        pixperA = np.asarray(pixels_) / np.asarray(self.unitcell[:2]) / tiling_

        # Get a list of unique atomic elements
        elements = list(set(np.asarray(self.atoms[:, 3], dtype=np.int)))

        # Get number of unique atomic elements
        nelements = len(elements)
        nsubslices = len(subslices)
        # Build list of equivalent sites if Fractional occupancy is to be
        # taken into account
        if fractional_occupancy and self.fractional_occupancy:
            equivalent_sites = find_equivalent_sites(self.atoms[:, :3], EPS=1e-3)

        # FDES method
        # Intialize potential array
        P = torch.zeros(
            np.prod([nelements, nsubslices, *pixels_, 2]), device=device, dtype=dtype
        )

        # Construct a map of which atom corresponds to which slice
        islice = np.zeros((self.atoms.shape[0]), dtype=np.int)
        slice_stride = np.prod(pixels_) * 2
        # if nsubslices > 1:
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
        # else:
        #     islice = 0
        # Make map a pytorch Tensor

        # Construct a map of which atom corresponds to which element
        element_stride = nsubslices * slice_stride
        ielement = torch.tensor(
            [
                element_stride * elements.index(int(self.atoms[iatom, 3]))
                for iatom in range(self.atoms.shape[0])
            ],
            dtype=torch.long,
            device=device,
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
                if fractional_occupancy and self.fractional_occupancy:
                    disp = disp[equivalent_sites, :]

                posn[:, :2] += disp

            yc = (
                torch.remainder(torch.ceil(posn[:, 0]).type(torch.long), pixels_[0])
                * pixels_[1]
                * 2
            )
            yf = (
                torch.remainder(torch.floor(posn[:, 0]).type(torch.long), pixels_[0])
                * pixels_[1]
                * 2
            )
            xc = (
                torch.remainder(torch.ceil(posn[:, 1]).type(torch.long), pixels_[1]) * 2
            )
            xf = (
                torch.remainder(torch.floor(posn[:, 1]).type(torch.long), pixels_[1])
                * 2
            )

            yh = torch.remainder(posn[:, 0], 1.0)
            yl = 1.0 - yh
            xh = torch.remainder(posn[:, 1], 1.0)
            xl = 1.0 - xh

            # Account for fractional occupancy of atomic sites if requested
            if fractional_occupancy and self.fractional_occupancy:
                xh *= torch.from_numpy(self.atoms[:, 4]).type(P.dtype).to(device)
                xl *= torch.from_numpy(self.atoms[:, 4]).type(P.dtype).to(device)

            # Each pixel is set to the overlap of a shifted rectangle in that pixel
            P.scatter_add_(0, ielement + islice + yc + xc, yh * xh)
            P.scatter_add_(0, ielement + islice + yc + xf, yh * xl)
            P.scatter_add_(0, ielement + islice + yf + xc, yl * xh)
            P.scatter_add_(0, ielement + islice + yf + xf, yl * xl)

        # Now view potential as a 4D array for next bit
        P = P.view(nelements, nsubslices, *pixels_, 2)

        # FFT potential to reciprocal space
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                P[i, j] = torch.fft(P[i, j], signal_ndim=2)

        # Make sinc functions with appropriate singleton dimensions for pytorch
        # broadcasting /gridsize[0]*pixels_[0] /gridsize[1]*pixels_[1]
        sincy = (
            sinc(torch.from_numpy(np.fft.fftfreq(pixels_[0])))
            .view([1, 1, pixels_[0], 1, 1])
            .to(device)
            .type(P.dtype)
        )
        sincx = (
            sinc(torch.from_numpy(np.fft.fftfreq(pixels_[1])))
            .view([1, 1, 1, pixels_[1], 1])
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
        P *= torch.from_numpy(fe_).view(nelements, 1, *pixels_, 1).to(device)

        norm = np.prod(pixels_) / np.prod(self.unitcell[:2]) / np.prod(tiling)
        # Add atoms together
        P = norm * torch.sum(P, dim=0)

        # Only return real part
        return torch.ifft(P, signal_ndim=2)[..., 0]

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
        seed=None,
        bandwidth_limit=2 / 3,
    ):
        """
        Make the transmission functions for the simulation object.

        Transmission functions are the exponential of the specimen electrostatic
        potential scaled by the interaction constant for electrons, sigma. These
        are used to model scattering by a thin slice of the object in the
        multislice algorithm

        Parameters:
        -----------
        pixels : array_like
            Output pixel grid
        eV : float
            Probe accelerating voltage in electron-volts
        subslices : array_like, optional
            An array containing the depths at which each slice ends as a fraction
            of the simulation unit-cell, used for simulation objects thicker
            than typical multislice slicing (about 2 Angstrom)
        tiling : array_like,optional
            Repeat tiling of the simulation object
        fe: array_like,optional
            An array containing the electron scattering factors for the elements
            in the simulation object as calculated by the function
            calculate_scattering_factors
        """
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
            seed=seed,
        )

        # Now take the complex exponential of the electrostatic potential
        # scaled by the electron interaction constant
        T = torch.fft(torch_c_exp(interaction_constant(eV) * T), signal_ndim=2)

        # Band-width limit the transmission function, see Earl Kirkland's book
        # for an discussion of why this is necessary
        for i in range(T.shape[0]):
            T[i] = bandwidth_limit_array(T[i], bandwidth_limit)

        if fftout:
            return torch.ifft(T, signal_ndim=2)
        return T

    def generate_slicing_figure(self, slices, show=True):
        """
        Generate slicing figure.

        Generate a slicing figure that to aid in setting up the slicing
        of the sample for multislice algorithm. This will show where each of the
        slices end for a chosen slicing relative to the atoms. To minimize
        errors, the atoms should sit as close to the top of the slice as possible.

        Parameters
        ----------
        slices: array_like, float
            An array containing the depths at which each slice ends as a fraction
            of the simulation unit-cell
        """
        fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

        coords = self.atoms[:, :3] * self.unitcell[None, :]
        # Projection down the x-axis
        for i in range(2):
            ax[i].plot(coords[:, i], coords[:, 2], "bo", label="Atoms")
            for j, slice_ in enumerate(slices):
                if j == 0:
                    label = "Slices"
                else:
                    label = "_"
                ax[i].plot(
                    [0, self.unitcell[i]],
                    [slice_ * self.unitcell[2], slice_ * self.unitcell[2]],
                    "r--",
                    label=label,
                )
            ax[i].set_xlim([0, self.unitcell[i]])
            ax[i].set_xlabel(["y", "x"][i])
            ax[i].set_ylim([self.unitcell[2], 0])
            ax[i].set_ylabel("z")
            ax[i].set_title("View down {0} axis".format(["x", "y"][i]))
        ax[0].legend()
        if show:
            plt.show(block=True)
        return fig

    def rotate(self, theta, axis, origin=[0.5, 0.5, 0.5]):
        """
        Rotate simulation object an amount an angle theta (in radians) about axis.

        Parameters
        ----------
        theta: float
            Angle to rotate simulation object by in radians
        axis: array_like
            Axis about which to rotate simulation object eg [0,0,1]

        Keyword arguments
        ------------------
        origin : array_like, optional
            Origin (in fractional coordinates) about which to rotate simulation
            object eg [0.5, 0.5, 0.5]
        """
        new = copy.deepcopy(self)

        # Make rotation matrix, R, and  the point about which we rotate, O
        R = rot_matrix(theta, axis)
        origin_ = np.asarray(origin) * self.unitcell

        # Get atomic coordinates in cartesian (not fractional coordinates)
        new.atoms[:, :3] = self.atoms[:, :3] * self.unitcell[np.newaxis, :]

        # Apply rotation matrix to each atom coordinate
        new.atoms[:, :3] = (new.atoms[:, :3] - origin_) @ R + origin_

        # Apply rotation matrix to cell vertices
        vertices = (
            np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
            * self.unitcell
            - origin_
        ) @ R + origin_

        # Get new unit cell from maximum range of unit cell vertices
        origin_ = np.amin(vertices, axis=0)
        new.unitcell = np.ptp(vertices, axis=0)

        # Convert atoms back into fractional coordinates in new unit cell
        new.atoms[:, :3] = ((new.atoms[:, :3] - origin_) / new.unitcell) % 1.0

        # Return rotated structure
        return new

    def rot90(self, k=1, axes=(0, 1)):
        """
        Rotates a structure by 90 degrees in the plane specified by axes.

        Rotation direction is from the first towards the second axis.

        Parameters
        ----------
        k : integer, optional
            Number of times the structure is rotated by 90 degrees.
        axes: (2,) array_like
            The array is rotated in the plane defined by the axes.
            Axes must be different.
        """
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

        return self

    def transpose(self, axes):
        """Transpose the axes of a simulation object."""
        self.atoms[:, :3] = self.atoms[:, axes]
        self.unitcell = self.unitcell[axes]
        return self

    def tile(self, x=1, y=1, z=1):
        """Make a repeat unit tiling of the simulation object."""
        # Make copy of original structure
        # new = copy.deepcopy(self)

        tiling = np.asarray([x, y, z], dtype=np.int)

        # Get atoms in unit cell
        natoms = self.atoms.shape[0]

        # Initialize new atom list
        newatoms = np.zeros((natoms * x * y * z, 6))

        # Calculate new unit cell size
        self.unitcell = self.unitcell * np.asarray([x, y, z])

        # tile out the integer amounts
        from itertools import product

        for j, k, l in product(*[np.arange(int(i)) for i in [x, y, z]]):

            # Calculate origin of this particular tile
            origin = np.asarray([j, k, l])

            # Calculate index of this particular tile
            indx = j * int(y) * int(z) + k * int(z) + l

            # Add new atoms to unit cell
            newatoms[indx * natoms : (indx + 1) * natoms, :3] = (
                self.atoms[:, :3] + origin[np.newaxis, :]
            ) / tiling[np.newaxis, :]

            # Copy other information about atoms
            newatoms[indx * natoms : (indx + 1) * natoms, 3:] = self.atoms[:, 3:]
        self.atoms = newatoms
        return self

    def concatenate(self, other, axis=2, side=1, eps=1e-2):
        """
        Concatenate two simulation objects.

        Adds other simulation object to the current object. other is added to
        the bottom (top being z =0) routine will attempt to tile objects to
        match dimensions.

        Parameters:
        other : structure class
            Object that will be concatenated onto the other object.
        axis : int, optional
            Axis along which the two structures will be joined.
        side : int, optional
            Determines which side the other structure will be added onto the
            first structure. If side == 0 the structures will be added onto each
            other at the origin, if side == 1 the structures will be added onto
            each other at the end.
        eps : float, optional
            Fractional tolerance of the psuedo rational tiling to make the
            structure dimensions perpendicular to the beam direction match.
        """
        # Make deep copies of the structure object and the slice this is so
        # that these objects remain untouched by the operation of this function
        new = copy.deepcopy(self)
        other_ = copy.deepcopy(other)

        tile1, tile2 = [np.ones(3, dtype=np.int) for i in range(2)]

        # Check if the two slices are the same size and
        # tile accordingly
        for ax in range(3):
            # If this axis is the concatenation axis, then it's not necessary
            # that the structures are the same size
            if ax == axis:
                continue
            # Calculate the psuedo-rational tiling
            if self.unitcell[ax] < other.unitcell[ax]:
                tile1[ax], tile2[ax] = psuedo_rational_tiling(
                    self.unitcell[ax], other.unitcell[ax], eps
                )
            else:
                tile2[ax], tile1[ax] = psuedo_rational_tiling(
                    other.unitcell[ax], self.unitcell[ax], eps
                )

            tile1[ax], tile2[ax] = psuedo_rational_tiling(
                self.unitcell[ax], other.unitcell[ax], eps
            )

        new = new.tile(*tile1)
        tiled_zdim = new.unitcell[axis]
        other_ = other_.tile(*tile2)

        # Update the thickness of the resulting structure object.
        new.unitcell[axis] = tiled_zdim + other_.unitcell[axis]

        # Adjust fractional coordinates of atoms, multiply by old unitcell
        # size to transform into cartesian coordinates and then divide by
        # the old unitcell size to transform into fractional coordinates
        # in the new basis
        new.atoms[:, axis] *= tiled_zdim / new.unitcell[axis]
        other_.atoms[:, axis] *= other_.unitcell[axis] / new.unitcell[axis]

        # Adjust the coordinates of the new or old atoms depending on which
        # side the new structure is to be added.
        if side == 0:
            new.atoms[:, axis] += other_.unitcell[axis] / new.unitcell[axis]
        else:
            other_.atoms[:, axis] += self.unitcell[axis] / new.unitcell[axis]

        # Concatenate adjusted atomic coordinates
        new.atoms = np.concatenate([new.atoms, other_.atoms], axis=0)

        # Concatenate titles
        new.Title = self.Title + " and " + other.Title

        return new

    def reflect(self, axes):
        """Reflect structure in each of the axes enumerated in list axes."""
        for ax in axes:
            self.atoms[:, ax] = (1 - self.atoms[:, ax]) % 1.0
        return self

    def resize(self, fraction, axis):
        """
        Resize (either crop or pad with vacuum) the simulation object.

        Resize the simulation object ranging such that the new axis runs from
        fraction[iax,0] to fraction[iax,1] on specified axis iax, slice_frac is
        in units of fractional coordinates. If fraction[iax,0] is < 0 then
        additional vacuum will be added, if > 0 then parts of the sample will
        be removed for axis[iax]. Likewise if fraction[iax,1] is > 1 then
        additional vacuum will be added, if < 1 then parts of the sample will
        be removed for axis[iax].

        Parameters
        ----------
        fraction : (nax,2) or (2,) array_like
            Describes the size of the new simulation object as a fraction of
            old simulation object dimensions.
        axis : int or (nax,) array_like
            The axes of the simulation object that wil lbe resized

        Returns
        -------
        New structure : pyms.structure object
            The resized structure object
        """
        ax = ensure_array(axis)
        frac = ensure_array(fraction)

        # Work out which atoms will stay in the sliced structure
        mask = np.ones((self.atoms.shape[0],), dtype=np.bool)
        for a, f in zip(ax, frac):
            atomsin = np.logical_and(self.atoms[:, a] >= f[0], self.atoms[:, a] <= f[1])
            mask = np.logical_and(atomsin, mask)

        # Make a copy of the structure
        new = copy.deepcopy(self)

        # Put remaining atoms back in
        new.atoms = self.atoms[mask, :]

        # Origin for atomic coordinates
        origin = np.zeros((3))

        for a, f in zip(ax, frac):
            # Adjust unit cell dimensions
            new.unitcell[a] = (f[1] - f[0]) * self.unitcell[a]

            # Adjust origin of atomic coordinates
            origin[a] = f[0]

        new.atoms[:, :3] = (new.atoms[:, :3] - origin) * self.unitcell / new.unitcell

        # Return modified structure
        return new

    def cshift(self, shift, axis):
        """
        Circular shift routine.

        Shift the atoms within the simulation cell an amount shift in fractional
        coordinates along specified axis (or axes if both shift and axis are
        array_like).

        Parameters
        ----------
        shift : array_like or int
            Amount in fractional coordinates to shift (each) axis.
        axis : array_like or int
            Axis or list of axes to apply shift(s) to.
        """

        def _cshift(atoms, x, ax):
            atoms[:, ax % 3] = np.mod(atoms[:, ax % 3] + x, 1.0)
            return atoms

        if hasattr(axis, "__len__"):
            for x, ax in zip(shift, axis):
                self.atoms = _cshift(self.atoms, x, ax)
        else:
            self.atoms = _cshift(self.atoms, shift, axis)

        return self


class layered_structure_transmission_function:
    """
    A class that mimics multislice transmission functions for a layered object.

    Useful for performing multislice calculations of heterostructures (epitaxially
    layered cystalline structures).
    """

    def __init__(
        self,
        gridshape,
        eV,
        structures,
        nslices,
        subslices,
        tilings=None,
        kwargs={},
        nT=5,
        dtype=torch.float32,
        device=None,
    ):
        """
        Generate a layered structure transmission function object.

        This function assumes that the lateral (x and y) cell sizes of the
        structures are identical,

        Input
        -----
        structures : (N,) array_like of pyms.Structure objects
            The input structures for which the transmission functions for a
            layered structure will be calculated.
        nslices : int (N,) array_like
            The number of units of each structure in the multilayer
        subslices : array_like (N,) of array_like
            Multislice subslicing for each object in the multilayer structure

        Returns
        -----
        self : layered_structure_transmission_function object
            This will behave like a normal transmission function array, if
            T = layered_structure_transmission_function(...,[structure1,structure2 etc])
            then T[0,islice,...] will return a transmission function from whichever
            structure islice happens to be in. T.Propagator[islice] returns the
            relevant multislice propagator
        """
        self.dtype = dtype
        self.device = get_device(device)
        self.nslicestot = np.sum(nslices)
        self.structures = structures

        if tilings is None:
            tilings = len(structures) * [[1, 1]]

        self.Ts = []
        self.nT = nT
        self.gridshape = gridshape
        self.tilings = tilings
        self.subslices = subslices
        self.eV = eV
        args = [gridshape, eV]

        for structure, subslices_, tiling in zip(structures, subslices, tilings):
            self.Ts.append(
                torch.stack(
                    [
                        structure.make_transmission_functions(
                            *args,
                            subslices=subslices_,
                            tiling=tiling,
                            **kwargs,
                            device=self.device,
                            dtype=self.dtype,
                        )
                        for i in range(nT)
                    ]
                )
            )

        self.slicemap = list(
            itertools.chain(
                *[len(subslices[i]) * n * [i] for i, n in enumerate(nslices)]
            )
        )
        nsubslices = [len(subslice) for subslice in subslices]
        self.subslicemap = list(
            itertools.chain(
                *[
                    (np.arange(nsubslices[i] * n) % nsubslices[i]).tolist()
                    for i, n in enumerate(nslices)
                ]
            )
        )
        self.N = len(self.slicemap)
        # Mimics the shape property of a numpy array
        self.shape = (self.nT, self.N, *self.gridshape, 2)
        self.Propagator = layered_structure_propagators(self)

    def dim(self):
        """Return the array dimension of the synthetic array."""
        return 4

    def __getitem__(self, ind):
        """
        __getitem__ method for the transmission function synthetic array.

        This enables the transmission function object to mimic a standard
        transmission function numpy or torch.Tensor array
        """
        it, islice = ind[:2]

        if isinstance(islice, int) or np.issubdtype(
            np.asarray(islice).dtype, np.integer
        ):
            T = self.Ts[self.slicemap[islice]][:, self.subslicemap[islice]]
        elif isinstance(islice, slice):
            islice_ = np.arange(*islice.indices(self.N))
            T = torch.stack(
                [self.Ts[self.slicemap[j]][:, self.subslicemap[j]] for j in islice_],
                axis=1,
            )
        else:
            raise TypeError("Invalid argument type.")

        if isinstance(it, int) or np.issubdtype(np.asarray(it).dtype, np.integer):
            return T[it]
        elif isinstance(it, slice):
            it_ = np.arange(*it.indices(self.nT))
            return T[it_]
        else:
            raise TypeError("Invalid argument type.")


class layered_structure_propagators:
    """
    A class that mimics multislice propagators for a layered object.

    Complements layered_transmission_function
    """

    def __init__(self, layered_T, propkwargs={}):
        """
        Generate a layered structure multislice propagator function object.

        This function assumes that the lateral (x and y) cell sizes of the
        structures are identical,

        Input
        -----
        T : layered_structure_transmission_function object
            This should contain all the neccessary information about the layered
            object to generate the propagators

        Returns
        -----
        self : layered_structure_propagators object
            This will behave like a normal propagator array, if
            P = layered_structure_propagators(T)
            then P[islice,...] will return a transmission function from whichever
            structure islice happens to be in.
        """
        from .py_multislice import make_propagators

        rsize = layered_T.structures[0].unitcell
        rsize[:2] *= np.asarray(layered_T.tilings[0])

        self.Ps = [
            make_propagators(
                layered_T.gridshape, rsize, layered_T.eV, subslices, **propkwargs
            )
            for subslices in layered_T.subslices
        ]

        self.Ps = list(itertools.chain(*self.Ps))
        self.Ps = [cx_from_numpy(P, layered_T.dtype, layered_T.device) for P in self.Ps]
        self.slicemap = layered_T.slicemap
        self.subslicemap = layered_T.subslicemap
        # Mimics the shape property of a numpy array
        self.shape = (layered_T.nslicestot, *layered_T.gridshape, 2)
        self.ndim = 4

    def dim(self):
        """Return the array dimension of the synthetic array."""
        return 3

    def __getitem__(self, islice):
        """
        __getitem__ method for the propagator synthetic array.

        This enables the propagator object to mimic a standard propagator numpy
        or torch.Tensor array
        """
        if isinstance(islice, int) or np.issubdtype(
            np.asarray(islice).dtype, np.integer
        ):
            return self.Ps[self.slicemap[islice]]
        elif isinstance(islice, slice):
            islice_ = np.arange(*islice.indices(self.N))
            return torch.stack([self.Ps[self.slicemap[j]] for j in islice_])
        else:
            raise TypeError("Invalid argument type.")
