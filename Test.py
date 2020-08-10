"""Unit tests for the py_multislice package."""

import copy
import unittest
import pyms
import numpy as np
import os
import torch
import matplotlib.pyplot as plt  # noqa

required_accuracy = 1e-4
scratch_cell_name = "test_cell.xyz"


def sumsqr(array):
    """Calculate the sum of squares for a single array."""
    if torch.is_tensor(array):
        if array.ndim < 1:
            return torch.abs(array) ** 2
        return torch.sum(pyms.utils.amplitude(array))
    return np.sum(np.square(np.abs(array)))


def sumsqr_diff(array1, array2):
    """Calculate the sum of squares normalized difference of two arrays."""
    return sumsqr(array1 - array2) / sumsqr(array1)


def clean_temp_structure():
    """Remove the temporary files made by the unit tests."""
    if os.path.exists(scratch_cell_name):
        os.remove(scratch_cell_name)


def SrTiO3():
    """Generate a SrTiO3 object for testing purposes."""
    atoms = np.asarray(
        [
            [0.000000, 0.000000, 0.000000, 38, 1.0, 0.7870e-2],
            [0.500000, 0.500000, 0.500000, 22, 1.0, 0.5570e-2],
            [0.500000, 0.500000, 0.000000, 8, 1.0, 0.9275e-2],
            [0.500000, 0.000000, 0.500000, 8, 1.0, 0.9275e-2],
            [0.000000, 0.500000, 0.500000, 8, 1.0, 0.9275e-2],
        ]
    )
    return pyms.structure([3.905, 3.905, 3.905], atoms[:, :4], dwf=atoms[:, 5])


def BaTiO3():
    """Generate a BaTiO3 object for testing purposes."""
    atoms = np.asarray(
        [
            [0.000000, 0.000000, 0.000000, 56, 1.0, 0.7870e-2],
            [0.500000, 0.500000, 0.500000, 22, 1.0, 0.5570e-2],
            [0.500000, 0.500000, 0.000000, 8, 1.0, 0.9275e-2],
            [0.500000, 0.000000, 0.500000, 8, 1.0, 0.9275e-2],
            [0.000000, 0.500000, 0.500000, 8, 1.0, 0.9275e-2],
        ]
    )
    return pyms.structure([4.01, 4.01, 4.01], atoms[:, :4], dwf=atoms[:, 5])


def make_graphene_file(fnam):
    """Make a graphene structure file for various testing purposes."""
    f = open(fnam, "w")
    f.write(
        "c4\n1.0\n 2.4672913551 0.0000000000 0.0000000000\n -1.2336456776 "
        + "2.1367369921 0.0000000000\n 0.0000000000 0.0000000000 "
        + "7.8030724525\n C\n 4\nDirect\n -0.000000000 -0.000000000 "
        + "0.250000000 C0 1.00000 1.00000 0.00000 0.00000 0.00000 0.00000 "
        + "0.00000 0.00000\n -0.000000000 -0.000000000 0.750000015 C1 "
        + "1.00000 1.00000 0.00000 0.00000 0.00000 0.00000 0.00000"
        + " 0.00000\n 0.333333014 0.666667038 0.250000000 C2 1.00000 1.00000"
        + " 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000\n 0.666666992"
        + " 0.333333017 0.750000015 C3 1.00000 1.00000 0.00000 0.00000 "
        + "0.00000 0.00000 0.00000 0.00000\n\n"
    )
    f.close()


def make_temp_structure(atoms=None, title="scratch", ucell=None, seed=0):
    """
    Make a random temporary structure to test routines on.

    Make a test structure and file for supplied or randomly generated
    atomic positions.
    """
    np.random.seed(seed)

    if ucell is None:
        ucell = np.random.random(3) * (10 - 1) + 1

    if atoms is None:

        # Number of atoms to be included in cell
        natoms = np.random.randint(1, 50, size=1)[0]
        shape = (natoms, 1)

        # Random atomic numbers
        Z = np.random.randint(2, 80, size=(natoms,)).reshape(shape)

        # Random positions
        pos = np.random.random((natoms, 3))

        # DWFs
        DWFs = np.random.random((natoms,)).reshape(shape) * (0.01 - 0.001) + 0.001

        # occupancies
        occ = 1.0 - np.random.random((natoms,)).reshape(shape)

        atoms = np.concatenate((Z, pos, occ, DWFs), axis=1)

    f = open(scratch_cell_name, "w")

    f.write("STRUCTURE FOR PYMS UNIT TESTS\n")
    f.write(" ".join(["{0:5.6f}".format(x) for x in ucell]) + "\n")
    for atom in atoms:
        f.write(
            "{0} ".format(int(atom[0]))
            + " ".join(["{0:7.5f}".format(x) for x in atom[1:]])
            + "\n"
        )
    f.write("-1")
    f.close()
    return atoms[:, [1, 2, 3, 0, 4, 5]], ucell


class Test_Structure_Methods(unittest.TestCase):
    """Tests for functions in structure.py ."""

    def test_remove_common_factors(self):
        """Test the function to remove common (integer) factors."""
        self.assertTrue(
            [x == y for x, y in zip(pyms.remove_common_factors([3, 9, 15]), [1, 3, 5])]
        )

    def test_psuedo_rational_tiling(self):
        """Test psuedo rational tiling function."""
        self.assertTrue(
            [
                x == y
                for x, y in zip(
                    pyms.psuedo_rational_tiling(3.905, 15.6, 1e-1), np.array([4, 1])
                )
            ]
        )

    def test_electron_scattering_factor(self):
        """
        Test electron scattering factor function.

        Test that the electron scattering factor function can replicate the
        results for Ag in Doyle, P. A. & Turner, P. S. (1968). Acta Cryst. A24,
        390–397
        """
        # Reciprocal space points
        g = 2 * np.concatenate(
            [
                np.arange(0.00, 0.51, 0.05),
                np.arange(0.60, 1.01, 0.1),
                np.arange(1.20, 2.01, 0.2),
                [2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
            ]
        )
        # Values from Doyle and Turner
        fe = np.asarray(
            [
                8.671,
                8.244,
                7.267,
                6.215,
                5.293,
                4.522,
                3.878,
                3.339,
                2.886,
                2.505,
                2.185,
                1.688,
                1.335,
                1.082,
                0.897,
                0.758,
                0.568,
                0.444,
                0.357,
                0.291,
                0.241,
                0.159,
                0.113,
                0.084,
                0.066,
                0.043,
                0.030,
            ]
        )
        sse = np.sum(
            np.square(fe - pyms.electron_scattering_factor(47, g ** 2, units="A"))
        )
        self.assertTrue(sse < 0.01)

    def test_calculate_scattering_factors(self):
        """Test calculate scattering factors against precomputed standard."""
        calc = pyms.calculate_scattering_factors([3, 3], [1, 1], [49])
        known = np.asarray(
            [
                [
                    [499.59366, 107.720055, 107.720055],
                    [107.720055, 65.67732, 65.67732],
                    [107.720055, 65.67732, 65.67732],
                ]
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.sum(np.square(known - calc)) < 0.01)

    def test_find_equivalent_sites(self):
        """
        Test the find_equivalent_sites function.

        Test against a list of sites, some of which are equivalent.
        """
        positions = np.asarray(
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.20008, 0.30008, 0.40008]]
        )
        self.assertTrue(
            np.all(
                np.equal(
                    pyms.find_equivalent_sites(positions),
                    np.asarray([0, 1, 0], dtype=np.int),
                )
            )
        )

    def test_interaction_constant(self):
        """
        Test interaction constant function.

        Test by replicating key values from Kirkland's textbook.
        """
        refEs = np.asarray([10, 60, 100, 200, 300, 800]) * 1e3
        refsigmas = np.asarray([2.599, 1.1356, 0.9244, 0.7288, 0.6526, 0.5503]) * 1e-3
        self.assertTrue(
            np.all(np.square(pyms.interaction_constant(refEs) - refsigmas) < 1e-12)
        )

    def test_rot_matrix(self):
        """Test rotation matrix by replicating 90 degree axis rotation matrices."""
        Rx = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
        Ry = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        Rz = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        Rx, Ry, Rz = [np.asarray(R) for R in [Rx, Ry, Rz]]

        Rxpass = np.all(np.abs(Rx - pyms.rot_matrix(np.pi / 2, [1, 0, 0])) < 1e-8)
        Rypass = np.all(np.abs(Ry - pyms.rot_matrix(np.pi / 2, [0, 1, 0])) < 1e-8)
        Rzpass = np.all(np.abs(Rz - pyms.rot_matrix(np.pi / 2, [0, 0, 1])) < 1e-8)

        self.assertTrue(np.all([Rxpass, Rypass, Rzpass]))

    def test_structure_file_open(self):
        """Test by writing a scratch random unit cell and loading it again."""
        atoms, ucell = make_temp_structure()

        cell = pyms.structure.fromfile(scratch_cell_name)

        atomspass = np.all(np.square(atoms - cell.atoms) < 1e-10)
        celldimpass = np.all(np.square(ucell - cell.unitcell) < 1e-10)

        self.assertTrue(atomspass and celldimpass)

    def test_orthorhombic_supercell(self):
        """Test by opening graphene (which has a hexagonal unit cell)."""
        fnam = scratch_cell_name.replace(".xyz", ".p1")
        make_graphene_file(fnam)
        graphene = pyms.structure.fromfile(fnam)
        os.remove(fnam)
        graphene.output_vesta_xtl("graphene.xtl")

        graphene_cell = np.asarray([2.46729136, 4.27347398, 7.80307245])
        graphene_pos = np.asarray(
            [
                0.0000,
                0.0000,
                0.2500,
                0.0000,
                0.0000,
                0.7500,
                1.0000,
                0.3333,
                0.2500,
                0.5000,
                0.1667,
                0.7500,
                0.5000,
                0.5000,
                0.2500,
                0.5000,
                0.5000,
                0.7500,
                0.5000,
                0.8333,
                0.2500,
                0.0000,
                0.6667,
                0.7500,
            ]
        ).reshape((8, 3))

        atomspass = np.all(np.square(graphene_pos - graphene.atoms[:, :3]) < 1e-7)
        celldimpass = np.all(np.square(graphene_cell - graphene.unitcell) < 1e-7)

        self.assertTrue(atomspass and celldimpass)

    def test_make_potential(self):
        """Test the make potential function against a precomputed standard."""
        make_temp_structure(seed=0)
        cell = pyms.structure.fromfile(scratch_cell_name)
        tiling = [2, 2]
        pot = (
            cell.make_potential([4, 4], subslices=[0.5, 1.0], tiling=tiling, seed=0)
            .cpu()
            .numpy()
        )
        precalc = np.asarray(
            [
                7.0271377563e00,
                1.0801711082e01,
                7.5602140426e00,
                9.8008451461e00,
                2.3832376480e01,
                3.2159790992e00,
                2.2516759872e01,
                2.6229567527e00,
                7.7249503135e00,
                9.8236846923e00,
                7.3711028099e00,
                1.0093936920e01,
                2.2842248916e01,
                2.7207412719e00,
                2.2528783798e01,
                2.6203308105e00,
                4.4570851325e00,
                1.3123485565e01,
                3.9715085029e00,
                1.1828842163e01,
                2.2986445426e00,
                5.0166683197e00,
                2.2372388839e00,
                5.0104198455e00,
                4.8299803733e00,
                1.1613823890e01,
                4.7839732170e00,
                1.1727484703e01,
                1.6152291297e00,
                4.5633144378e00,
                2.1869301795e00,
                5.7718343734e00,
            ]
        )
        within_spec = np.sum(np.square(precalc - pot.ravel())) < 1e-10
        self.assertTrue(within_spec)

    def test_resize(self):
        """Test the structure resize routine for cropping and padding with vaccuum."""
        unitcell = [5.0, 5.0, 5.0]
        atom = np.asarray([1 / 4, 1 / 4, 1 / 4]).reshape((1, 3))
        dwf = [0]
        cell = pyms.structure(unitcell, atom, dwf)
        resize = [[0.0, 0.5], [-0.25, 1.25]]
        axis = [0, 2]
        newcell = cell.resize(resize, axis)

        test_posn = (
            sumsqr_diff(newcell.atoms[0, :3], np.asarray([0.5, 0.25, 1 / 3])) < 1e-10
        )
        test_cell = sumsqr_diff(newcell.unitcell, np.asarray([2.5, 5.0, 7.5])) < 1e-10

        self.assertTrue(test_posn and test_cell)

    def test_rotate(self):
        """Test rotation routine on synthetic structure."""
        orig_posn = np.asarray(
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]
        )
        teststruc = pyms.structure(
            [2, 2, 2], copy.deepcopy(orig_posn), [0.0, 0.0, 0.0, 0.0]
        )
        struc = copy.deepcopy(teststruc)
        struc.rot90(1, (0, 2))
        teststruc = teststruc.rotate(np.pi / 2, [0, 1, 0])
        orig_posn[1:3] = orig_posn[2:0:-1]
        self.assertTrue(
            sumsqr_diff(orig_posn, teststruc.atoms[:, :3]) < 1e-10
            and sumsqr_diff(orig_posn, struc.atoms[:, :3]) < 1e-10
        )

    def testrot90(self):
        """Test of the rot90 function (under construction)."""
        # Make random test object
        atoms, ucell = make_temp_structure()

        cell = pyms.structure.fromfile(scratch_cell_name)

        newcell = copy.deepcopy(cell)
        newcell = newcell.rot90(k=1, axes=(0, 1))

    def test_multilayer_objects(self):
        """Ensure multislice of a multilayer object against equivalent calculation."""
        gridshape = [64, 64]
        eV = 3e5
        nslices = [3, 2]
        tiling = [2, 1]
        kwargs = {"displacements": False}
        subslices = [[0.5, 1.0], [0.25, 1.0]]
        nT = 2

        STO = SrTiO3()
        BTO = BaTiO3()
        BTO.unitcell[:] = STO.unitcell[:]

        # For the sake of the test make the BTO cell thicker
        BTO.unitcell[2] *= 1.4
        rsize = STO.unitcell[:2] * np.asarray(tiling)

        # Make multilayer object
        T = pyms.layered_structure_transmission_function(
            gridshape,
            eV,
            [STO, BTO],
            nslices,
            subslices,
            nT=nT,
            kwargs=kwargs,
            tilings=2 * [tiling],
        )

        # Make explicit equivalent multislice transmission functions and
        # propagators
        P1, T1 = pyms.multislice_precursor(
            STO,
            gridshape,
            eV,
            subslices[0],
            **kwargs,
            showProgress=False,
            tiling=tiling,
            nT=nT
        )
        P2, T2 = pyms.multislice_precursor(
            BTO,
            gridshape,
            eV,
            subslices[1],
            **kwargs,
            showProgress=False,
            tiling=tiling,
            nT=nT
        )
        # fig,ax = plt.subplots(ncols=3)
        # ax[0].imshow(P1.imag)
        # ax[1].imshow(P2[0].imag)
        # ax[2].imshow(P2[1].imag)
        # plt.show(block=True)

        # Check transmission function and propagators are the same
        sameT = sumsqr_diff(T[:, 0 : len(subslices[0])], T1) < 1e-10
        N = len(subslices[0]) * nslices[0]
        sameT = sameT and sumsqr_diff(T[:, N : N + len(subslices[1])], T2) < 1e-10
        sameP = sumsqr_diff(pyms.cx_to_numpy(T.Propagator[:N]), P1) < 1e-10
        PP = pyms.cx_to_numpy(T.Propagator[N : N + len(subslices[1])])
        sameP = sameP and sumsqr_diff(PP, P2) < 1e-10

        # Multislice with equivalent multislice objects
        illum = pyms.plane_wave_illumination(gridshape, rsize, eV)
        exit_wave = pyms.multislice(
            illum,
            nslices[0],
            P1,
            T1,
            tiling=tiling,
            output_to_bandwidth_limit=nslices[1] < 1,
        )
        midwave = copy.deepcopy(exit_wave)
        exit_wave = pyms.multislice(exit_wave, nslices[1], P2, T2, tiling=tiling)

        # Multislice with multilayer object
        illum = pyms.plane_wave_illumination(gridshape, rsize, eV)

        # Ensure that the synthetic multilayer accurately reproduces electron
        # waves at the middle and exit surface of the layer
        nslices = pyms.thickness_to_slices(
            [
                nslices[0] * STO.unitcell[2],
                nslices[0] * STO.unitcell[2] + nslices[1] * BTO.unitcell[2],
            ],
            T.unitcell[2],
            True,
            T.subslices,
        )
        midwave2 = pyms.multislice(
            illum,
            nslices[0],
            T.Propagator,
            T,
            tiling=tiling,
            output_to_bandwidth_limit=False,
        )
        exit_wave2 = pyms.multislice(
            midwave2, nslices[1], T.Propagator, T, tiling=tiling
        )
        # fig,ax = plt.subplots(ncols=3)
        # ax[0].imshow(np.angle(exit_wave))
        # ax[1].imshow(np.angle(exit_wave2))
        # ax[2].imshow(np.angle(exit_wave) - np.angle(exit_wave2))
        # plt.show(block=True)
        passmultislice = (
            sumsqr_diff(exit_wave, exit_wave2) < 1e-10
            and sumsqr_diff(midwave, midwave2) < 1e-10
        )
        self.assertTrue(sameT and passmultislice)


class Test_probe_methods(unittest.TestCase):
    """Test functions inside the Probe.py file."""

    def test_Cc_methods(self):
        """Test the chromatic aberration functions."""
        eV = 3e5
        deltaE = 0.8  # 1.2
        Cc = 1.2e7
        deltaEconv = "FWHM"
        npoints = 55
        defocii = pyms.Cc_integration_points(
            Cc, deltaE, eV, npoints=npoints, deltaEconv=deltaEconv
        )
        passintegrationpoints = (
            np.abs(
                np.std(defocii)
                - Cc * pyms.convert_deltaE(deltaE, deltaEconv) / eV / np.sqrt(2)
            )
            < 0.1
        )

        npoints = 3
        gridshape = [128, 128]
        app = 20
        thicknesses = [12]
        args = [SrTiO3(), gridshape, eV, app, thicknesses]
        kwargs = {"tiling": [4, 4], "showProgress": False}
        pyms.simulation_result_with_Cc(
            pyms.HRTEM,
            Cc,
            deltaE,
            eV,
            args=args,
            kwargs=kwargs,
            npoints=npoints,
            deltaEconv=deltaEconv,
        )

        args = [SrTiO3(), gridshape, eV, app, thicknesses]
        kwargs = {"tiling": [4, 4], "showProgress": False, "FourD_STEM": True}
        pyms.simulation_result_with_Cc(
            pyms.STEM_multislice,
            Cc,
            deltaE,
            eV,
            args=args,
            kwargs=kwargs,
            npoints=npoints,
            deltaEconv=deltaEconv,
        )

        self.assertTrue(passintegrationpoints)


class Test_ionization_methods(unittest.TestCase):
    """Tests for ionization methods."""

    def test_EFTEM(self):
        """
        Test the energy filtered TEM (EFTEM) routine.

        This is done by simulating EFTEM for single transition for a single
        oxygen atom, which should be an (almost) perfect image of that
        transition potential.
        """
        # A 5 x 5 x 5 Angstrom cell with an oxygen
        cell = [5, 5, 0.0001]
        atoms = [[0.0, 0.0, 0.0, 8]]
        crystal = pyms.structure(cell, atoms, dwf=[0.0])
        gridshape = [64, 64]
        eV = 3e5
        thicknesses = cell[2]
        subslices = [1.0]
        app = None
        # Target the oxygen K edge
        Ztarget = 8

        # principal and orbital angular momentum quantum numbers for bound state
        n = 1
        ell = 0
        lprime = 0
        from pyms import orbital, get_q_numbers_for_transition, transition_potential

        qnumbers = get_q_numbers_for_transition(ell, order=1)[1]
        bound_configuration = "1s2 2s2 2p4"
        excited_configuration = "1s1 2s2 2p4"
        epsilon = 1

        bound_orbital = orbital(Ztarget, bound_configuration, n, ell, lprime)
        excited_orbital = orbital(Ztarget, excited_configuration, 0, qnumbers[0], 10)

        # Calculate transition potential for this escited state
        Hn0 = transition_potential(
            bound_orbital,
            excited_orbital,
            gridshape,
            cell,
            qnumbers[2],
            qnumbers[1],
            eV,
        ).reshape((1, *gridshape))

        P, T = pyms.multislice_precursor(
            crystal, gridshape, eV, subslices, nT=1, showProgress=False
        )

        EFTEM_images = pyms.EFTEM(
            crystal,
            gridshape,
            eV,
            app,
            thicknesses,
            Ztarget,
            n,
            ell,
            epsilon,
            df=[0],
            nT=1,
            Hn0=Hn0,
            subslices=subslices,
            nfph=1,
            P=P,
            T=T,
            showProgress=False,
        )

        # The image of the transition is "lensed" by the atom that it is passed
        # through so we must account for this by multiplying the Hn0 by its transition
        # potential
        Hn0 *= pyms.utils.cx_to_numpy(T[0, 0]) / np.sqrt(np.prod(gridshape))

        result = np.fft.fftshift(np.squeeze(EFTEM_images))
        reference = np.fft.fftshift(
            np.abs(
                pyms.utils.fourier_interpolate_2d(
                    pyms.utils.bandwidth_limit_array(
                        Hn0[0], qspace_in=False, qspace_out=False
                    ),
                    result.shape,
                    "conserve_norm",
                )
            )
            ** 2
        )

        # import matplotlib.pyplot as plt
        # fig,ax = plt.subplots(nrows=2)
        # ax[0].imshow(reference)
        # ax[1].imshow(result)
        # plt.show(block=True)

        passEFTEM = np.sum(np.abs(result - reference) ** 2) < 1e-7
        self.assertTrue(passEFTEM)


class Test_py_multislice_Methods(unittest.TestCase):
    """Tests for functions inside of the py_multislice.py file."""

    def test_thickness_to_slices(self):
        """Test Angstorm thickness to multislice slices routine."""
        thicknesses = [10, 50, 100]
        unitcell = 10
        nslices = pyms.thickness_to_slices(thicknesses, unitcell)
        passslicing = sumsqr_diff(np.asarray([1, 5, 10]), nslices) < 1e-10

        subslices = (np.arange(10) + 1) / 10
        unitcell = 100
        nslices = pyms.thickness_to_slices(thicknesses, unitcell, True, subslices)
        reference = [np.arange(1), np.arange(1, 5), np.arange(5, 10)]
        passsubslicing = np.all([np.all(x == y) for x, y in zip(nslices, reference)])
        self.assertTrue(passslicing and passsubslicing)

    def test_propagator(self):
        """Test propagator against analytic result."""
        # Test of propagator function
        gridshape = [256, 256]
        rsize = [75, 75, 250]
        eV = 3e5

        P = pyms.make_propagators(gridshape, rsize, eV)
        k = np.fft.fftfreq(gridshape[1], d=rsize[1] / gridshape[1])
        ref = np.sin(-np.pi * k ** 2 * rsize[2] / pyms.wavev(eV))
        ref[np.abs(P[0, 0]) < 1e-6] = 0
        self.assertTrue(sumsqr_diff(ref, np.imag(P[0, 0])) < 1e-10)

    def test_nyquist_and_probe_raster(self):
        """Test probe positions and Nyquist routine."""
        # cell size chosen so that there will be exactly 10 x 30 probe positions
        rsize = [2.460935, 2.460935 * 3]
        eV = 3e5
        alpha = 20

        # electron wave vector is 50.79 inv A so maximum resolution with 20 mrad
        # aperture is 20e-3*50.79 = 1.0158736246 inv A sampling is therefor
        # 4 times that, ie. 4.063 scan positions per Angstrom or a probe step of
        # 0.24609
        nyquistpass = (
            np.abs(pyms.nyquist_sampling(eV=eV, alpha=alpha) - 0.246093608496175) < 1e-6
        )

        probe_posn = pyms.generate_STEM_raster(rsize, eV, alpha)
        probe_scanpass = sumsqr_diff(probe_posn[:, 0, 0], np.arange(10) / 10) < 1e-10
        probe_scanpass = (
            probe_scanpass
            and sumsqr_diff(probe_posn[0, :, 1], np.arange(30) / 30) < 1e-10
        )

        self.assertTrue(nyquistpass and probe_scanpass)

    def test_diffraction_pattern_resize(self):
        """Test the method for resizing diffraction patterns."""
        from skimage.data import astronaut

        im = np.fft.fftshift(np.sum(astronaut(), axis=2).astype(np.float32))
        im /= np.sum(im)
        # Assume that the original pattern measures 1 x 1 inverse Angstroms and is
        # 512 x 512 (size of astronaut image)
        rsize = im.shape
        gridshape = im.shape
        # Check cropping only
        FourD_STEM = [[256, 256]]
        gridout, resize, Ksize = pyms.workout_4DSTEM_datacube_DP_size(
            FourD_STEM, rsize, gridshape
        )

        imout1 = resize(im)

        # imout1 should just be im cropped by factor of two
        passtest = (
            sumsqr_diff(pyms.utils.crop(np.fft.ifftshift(im), FourD_STEM[0]), imout1)
            < 1e-10
        )
        passtest = sumsqr_diff(Ksize, [0.5, 0.5]) < 1e-10 and passtest

        # Check cropping  and interpolation
        FourD_STEM = [[512, 512], [1.5, 1.5]]

        gridout, resize, Ksize = pyms.workout_4DSTEM_datacube_DP_size(
            FourD_STEM, rsize, gridshape
        )

        imout2 = resize(im)

        # Make sure that imout2 has the same normalisation as bfore
        passtest = sumsqr_diff(np.sum(imout2), np.sum(im)) < 1e-10 and passtest
        passtest = (
            sumsqr_diff(*[np.asarray(x) for x in [Ksize, FourD_STEM[1]]]) < 1e-10
            and passtest
        )
        self.assertTrue(passtest)

    def test_STEM_routine(self):
        """
        Test  STEM multislice  by comparing result to weak phase object approximation.

        For a thin and weakly scattering object the STEM image of a sample
        is approximately equal to the convolution of the specimen potential
        in radians with the "phase contrast transfer function" we use this to
        test that the STEM routine is giving sensible results.
        """
        # Make random test object
        atoms, ucell = make_temp_structure()

        cell = pyms.structure.fromfile(scratch_cell_name)
        # Make it very thin
        cell.unitcell[2] = 1
        # Make all atoms lithium (a weakly scattering atom)
        cell.atoms[:, 3] = 3

        # Parameters for test
        tiling = [4, 4]
        sampling = 0.05
        gridshape = np.ceil(cell.unitcell[:2] * np.asarray(tiling) / sampling).astype(
            np.int
        )
        eV = 3e5
        app = 20
        df = 100

        rsize = np.asarray(tiling) * cell.unitcell[:2]
        probe = pyms.focused_probe(gridshape, rsize, eV, app, df=df, qspace=True)
        detector = pyms.make_detector(gridshape, rsize, eV, app, app / 2)

        # Get phase contrast transfer funciton in real space
        PCTF = np.real(
            np.fft.ifft2(pyms.STEM_phase_contrast_transfer_function(probe, detector))
        )
        P, T = pyms.multislice_precursor(
            cell, gridshape, eV, tiling=tiling, displacements=False, showProgress=False
        )

        # Calculate phase of transmission function
        phase = np.angle(pyms.utils.cx_to_numpy(T[0, 0]))

        # Weak phase object approximation = meansignal + PCTF * phase
        meansignal = np.sum(np.abs(probe) ** 2 * detector) / np.sum(np.abs(probe) ** 2)
        WPOA = meansignal + np.real(pyms.utils.convolve(PCTF, phase))

        # Perform Multislice calculation and then compare to weak phase object
        # result
        images = pyms.STEM_multislice(
            cell,
            gridshape,
            eV,
            app,
            thicknesses=1.01,
            df=df,
            detector_ranges=[[app / 2, app]],
            nfph=1,
            P=P,
            T=T,
            tiling=tiling,
            showProgress=False,
        )["STEM images"]

        # Testing the PRISM code takes ~3 mins on a machine with a consumer
        # GPU so generally this part of the test is skipped
        testPRISM = False
        if testPRISM:
            PRISMimages = pyms.STEM_PRISM(
                cell,
                gridshape,
                eV,
                app,
                1.01,
                df=df,
                detector_ranges=[[app / 2, app]],
                PRISM_factor=[1, 1],
                nfph=1,
                P=P,
                T=T,
                tiling=tiling,
                showProgress=False,
            )["STEM images"]
            PRISMimages = pyms.utils.fourier_interpolate_2d(
                np.tile(PRISMimages, tiling), gridshape
            )
        images = pyms.utils.fourier_interpolate_2d(np.tile(images, tiling), gridshape)

        # fig,ax = plt.subplots(nrows = 3+testPRISM)
        # ax[0].imshow(WPOA)
        # ax[1].imshow(images)
        # if testPRISM:
        #     ax[2].imshow(PRISMimages)
        # plt.show(block=True)

        # Test is passed if sum squared difference with weak phase object
        # approximation is within reason
        passTest = sumsqr_diff(WPOA, images) < 1e-8
        if testPRISM:
            passTest = passTest and sumsqr_diff(PRISMimages, images) < 1e-8
        self.assertTrue(passTest)

    def test_DPC(self):
        """Test DPC routine by comparing reconstruction to input potential."""
        gridshape = [256, 256]
        eV = 6e4
        app = 30
        tiling = [17 // 2, 10 // 2]

        fnam = scratch_cell_name.replace(".xyz", ".p1")
        make_graphene_file(fnam)

        Graphene = pyms.structure.fromfile(fnam)
        thicknesses = [Graphene.unitcell[2] - 0.1]

        P, T = pyms.multislice_precursor(
            Graphene,
            gridshape,
            eV,
            tiling=tiling,
            nT=1,
            displacements=False,
            showProgress=False,
        )
        result = pyms.STEM_multislice(
            Graphene,
            gridshape,
            eV,
            app,
            thicknesses,
            nfph=1,
            displacements=False,
            DPC=True,
            tiling=tiling,
            P=P,
            T=T,
            detector_ranges=[[0, app / 2]],
            showProgress=False,
        )

        # fig, ax = plt.subplots(ncols=4)
        rsize = Graphene.unitcell[:2] * np.asarray(tiling)
        probe = pyms.focused_probe(gridshape, rsize, eV, app)
        convolved_phase = pyms.utils.convolve(
            np.abs(probe) ** 2, np.angle(pyms.utils.cx_to_numpy(T[0, 0]))
        )
        convolved_phase -= np.amin(convolved_phase)

        reconstructed_phase = pyms.utils.fourier_interpolate_2d(
            np.tile(result["DPC"], tiling), gridshape
        )
        reconstructed_phase -= np.amin(reconstructed_phase)
        self.assertTrue(sumsqr_diff(convolved_phase, reconstructed_phase) < 1e-4)

    def test_Smatrix(self):
        """
        Testing scattering matrix and PRISM algorithms.

        Test the scattering matrix code by matching a basic multislice CBED
        calculation with the scattering matrix result. Every nth beam, where n
        is the PRISM "interpolation" factor, is removed before doing the
        multislice and the result is cropped in real space in the multislice
        result for consistency with the PRISM S-matrix approach.
        """
        # Make random test object
        atoms, ucell = make_temp_structure()

        cell = pyms.structure.fromfile(scratch_cell_name)

        # Parameters for test
        tiling = [4, 4]
        tiling = [1, 1]
        # cell.unitcell[:2] = [15,15]

        gridshape = [128, 64]
        eV = 3e5
        app = 20
        thicknesses = [20]
        subslices = [0.5, 1.0]
        df = -300
        probe_posn = [1.0, 1.0]
        # probe_posn = [0.5, 0.5]

        PRISM_factor = [2, 2]
        # PRISM_factor = [1, 1]
        nT = 2
        # seed = np.random.randint(0, 10, size=2)

        # Make propagator and transmission functions for multislice and
        # S-matrix calculations
        P, T = pyms.multislice_precursor(
            cell,
            gridshape,
            eV,
            subslices=subslices,
            tiling=tiling,
            nT=nT,
            displacements=False,
            showProgress=False,
            band_width_limiting=[None, None],
        )

        # S matrix calculation
        # Calculate scattering matrix
        rsize = np.asarray(tiling) * cell.unitcell[:2]
        nslices = np.ceil(thicknesses[0] / cell.unitcell[2]).astype(np.int)
        S = pyms.scattering_matrix(
            rsize,
            P,
            T,
            nslices,
            eV,
            app,
            PRISM_factor=PRISM_factor,
            tiling=tiling,
            # seed=seed,
            showProgress=False,
            GPU_streaming=False,
        )

        # Calculate probe and shift it to required position
        probe = pyms.focused_probe(gridshape, rsize, eV, app, df=df, qspace=True)
        probe = pyms.utils.cx_from_numpy(
            pyms.utils.fourier_shift(
                probe,
                np.asarray(probe_posn) / np.asarray(tiling),
                qspacein=True,
                qspaceout=True,
                pixel_units=False,
            )
        )

        probe_ = copy.deepcopy(probe)
        # Calculate Smatrix result
        S_CBED = S(
            probe_, thicknesses[0], posn=[np.asarray(probe_posn) / np.asarray(tiling)]
        )

        S_CBED_amp = np.fft.fftshift(pyms.utils.amplitude(S_CBED).cpu().numpy())
        S_CBED_amp /= np.prod(S.stored_gridshape)

        # Ensure consistency of PRISM STEM routine
        datacube = pyms.STEM_PRISM(
            cell,
            gridshape,
            eV,
            app,
            thicknesses,
            tiling=tiling,
            df=df,
            PRISM_factor=PRISM_factor,
            FourD_STEM=True,
            scan_posn=np.asarray(probe_posn).reshape((1, 1, 2)) / np.asarray(tiling),
            showProgress=False,
            S=S,
            nT=5,
            GPU_streaming=False,
            device_type=torch.device("cpu"),
        )["datacube"]

        # Remove every nth beam where n is the PRISM cropping factor
        yy = np.fft.fftfreq(gridshape[0], 1.0 / gridshape[0]).astype(np.int)
        xx = np.fft.fftfreq(gridshape[1], 1.0 / gridshape[1]).astype(np.int)
        mask = np.logical_and(
            np.logical_and(
                (yy % PRISM_factor[0] == 0)[:, None],
                (xx % PRISM_factor[1] == 0)[None, :],
            ),
            pyms.amplitude(probe).cpu().numpy() > 0,
        )
        probe[np.logical_not(mask)] = 0

        # Transform probe back to real space
        probe = torch.ifft(probe, signal_ndim=2)

        # Adjust normalization to account for prism cropping
        probe *= np.prod(PRISM_factor)

        # Do multislice
        probe = pyms.multislice(
            probe.view([1, *gridshape, 2]), nslices, P, T, return_numpy=False
        )
        # ax[1].imshow(np.abs(pyms.utils.cx_to_numpy(probe[0])))
        # plt.show(block=True)

        # Get output gridsize
        gridout = torch.squeeze(probe).shape[:2]

        # Now crop multislice result in real space
        grid = probe.shape[-3:-1]
        stride = [x // y for x, y in zip(grid, PRISM_factor)]
        halfstride = [x // 2 for x in stride]
        start = [
            int(np.round(probe_posn[i] * grid[i])) - halfstride[i] for i in range(2)
        ]
        windows = pyms.utils.crop_window_to_periodic_indices(
            [start[0], stride[0], start[1], stride[1]], gridout
        )

        new = torch.zeros(*gridout, 2, dtype=S.dtype, device=probe.device)
        for wind in windows:
            new[
                wind[0][0] : wind[0][0] + wind[0][1],
                wind[1][0] : wind[1][0] + wind[1][1],
                :,
            ] = probe[
                0,
                wind[0][0] : wind[0][0] + wind[0][1],
                wind[1][0] : wind[1][0] + wind[1][1],
                :,
            ]
        probe = new
        probe = torch.fft(probe, signal_ndim=2) / np.sqrt(np.prod(gridout))
        ms_CBED = np.fft.fftshift(
            np.squeeze(np.abs(pyms.utils.cx_to_numpy(probe)) ** 2)
        )

        fig, ax = plt.subplots(ncols=3)
        ax[0].imshow(ms_CBED)
        ax[1].imshow(datacube[0, 0])
        ax[2].imshow(S_CBED_amp[0])
        plt.show(block=True)

        # The test is passed if the result from multislice and S-matrix is
        # within numerical error and the S matrix premixed routine is also
        # behaving as expected
        Smatrixtestpass = (
            sumsqr_diff(ms_CBED, S_CBED_amp) < 1e-10
            and sumsqr_diff(S_CBED_amp, datacube[0, 0]) < 1e-10
        )
        self.assertTrue(Smatrixtestpass)


class Test_util_Methods(unittest.TestCase):
    """Test the utility functions for numpy and pytorch and some output functions."""

    def test_h5_output(self):
        """Test the hdf5 output by writing and then reading a file to it."""
        import h5py
        from pyms.utils import (
            initialize_h5_datacube_object,
            datacube_to_py4DSTEM_viewable,
        )

        # Make test dataset to write and then read from hdf5 files
        shape = (5, 5, 5, 5)
        test_dcube = np.random.random_sample(shape)

        # Get hdf5 file objects for on-the-fly reading of arrays
        dcube, f = initialize_h5_datacube_object(
            shape, "test.h5", dtype=test_dcube.dtype
        )

        # Write datacube using object
        dcube[:] = test_dcube
        f.close()

        # Read datacube back in
        f = h5py.File("test.h5", "r")
        dcubein = f["/4DSTEM_experiment/data/datacubes/datacube_0/datacube"]

        # Check consistency
        objwritepass = sumsqr_diff(dcubein, test_dcube) < 1e-10
        f.close()

        # Now test direct writing function
        datacube_to_py4DSTEM_viewable(test_dcube, "test.h5")
        f = h5py.File("test.h5", "r")
        dcubein = f["/4DSTEM_experiment/data/datacubes/datacube_0/datacube"]

        funcwritepass = sumsqr_diff(dcubein, test_dcube) < 1e-10
        f.close()
        os.remove("test.h5")
        # If both writing methods work then test is passed
        self.assertTrue(objwritepass and funcwritepass)

    def test_r_space_array(self):
        """Test the real space array function."""
        r = pyms.utils.r_space_array([5, 5], [2, 2])
        passx = sumsqr_diff(r[0][:, 0], np.asarray([0.0, 0.4, 0.8, -0.8, -0.4])) < 1e-10
        passy = sumsqr_diff(r[1][0, :], np.asarray([0.0, 0.4, 0.8, -0.8, -0.4])) < 1e-10
        self.assertTrue(passx and passy)

    def test_q_space_array(self):
        """Test the reciprocal space array function."""
        q = pyms.utils.q_space_array([3, 5], [1, 1])
        passx = sumsqr_diff(q[0][:, 0], np.asarray([0, 1, -1])) < 1e-10
        passy = sumsqr_diff(q[1][0, :], np.asarray([0, 1, 2, -2, -1])) < 1e-10
        self.assertTrue(passx and passy)

    def test_crop_periodic_rectangle(self):
        """Test the periodic cropping rectangle indices function."""
        from pyms.utils import crop_window_to_periodic_indices

        i1 = crop_window_to_periodic_indices([2, 2, 1, 3], [5, 5])
        i2 = (([2, 2], [1, 3]),)
        passTest = i1 == i2
        i1 = crop_window_to_periodic_indices([-1, 3, 1, 3], [5, 5])
        i2 = (([4, 1], [1, 3]), ([0, 2], [1, 3]))
        passTest = passTest and i1 == i2
        i1 = crop_window_to_periodic_indices([3, 4, 1, 3], [5, 5])
        i2 = (([3, 2], [1, 3]), ([0, 2], [1, 3]))
        passTest = passTest and i1 == i2
        i1 = crop_window_to_periodic_indices([4, 4, 3, 3], [5, 5])
        i2 = (([4, 1], [3, 2]), ([4, 1], [0, 1]), ([0, 3], [3, 2]), ([0, 3], [0, 1]))
        passTest = passTest and i1 == i2
        self.assertTrue(passTest)

    def test_crop_window(self):
        """Test cropping window function on known result."""
        indices = torch.as_tensor([[2, 3, 4], [1, 2, 3]])
        gridshape = [4, 4]

        # Test torch function
        grid = torch.zeros(*gridshape, dtype=torch.long).flatten()
        ind = pyms.utils.crop_window_to_flattened_indices_torch(indices, gridshape)
        grid[ind] = 1
        grid = grid.view(*gridshape)

        reference = torch.as_tensor(
            [[0, 1, 1, 1], [0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1]]
        )
        torchpass = sumsqr_diff(grid, reference).item() < 1e-7

        # Test numpy function
        grid = np.zeros(gridshape, dtype=np.int).ravel()
        ind = pyms.utils.crop_window_to_flattened_indices(
            indices.cpu().numpy(), gridshape
        )
        grid[ind] = 1
        grid = grid.reshape(*gridshape)
        numpypass = sumsqr_diff(grid, reference.cpu().numpy()).item() < 1e-7

        self.assertTrue(numpypass and torchpass)

    def test_crop_tobandwidthlimit(self):
        """Test the function that crops arrays to the maximum bandwidth limit."""
        passtest = True
        # refoutputshape = [85, 85]
        gridshape = [128, 128]
        in_ = np.ones(gridshape)
        bwlimited = pyms.utils.bandwidth_limit_array(in_)
        cropped = pyms.utils.crop_to_bandwidth_limit(bwlimited)
        passtest = passtest and sumsqr_diff(np.sum(bwlimited), np.sum(cropped)) < 1e-10
        gridshape = [127, 127]
        in_ = np.ones(gridshape)
        bwlimited = pyms.utils.bandwidth_limit_array(in_)
        cropped = pyms.utils.crop_to_bandwidth_limit(bwlimited)
        passtest = passtest and sumsqr_diff(np.sum(bwlimited), np.sum(cropped)) < 1e-10

        # Now test torch version
        gridshape = [128, 128]
        in_ = np.ones(gridshape)
        bwlimited = pyms.utils.bandwidth_limit_array(torch.from_numpy(in_))
        cropped = pyms.utils.crop_to_bandwidth_limit_torch(bwlimited)
        passtest = (
            passtest and sumsqr_diff(torch.sum(bwlimited), torch.sum(cropped)) < 1e-10
        )
        gridshape = [127, 127]
        in_ = np.ones(gridshape)
        bwlimited = pyms.utils.bandwidth_limit_array(torch.from_numpy(in_))
        cropped = pyms.utils.crop_to_bandwidth_limit(bwlimited)
        passtest = (
            passtest and sumsqr_diff(torch.sum(bwlimited), torch.sum(cropped)) < 1e-10
        )

        # pytorch test with complex numbers
        gridshape = [128, 128]
        in_ = np.ones(gridshape, dtype=np.complex64)
        bwlimited = pyms.utils.bandwidth_limit_array(pyms.utils.cx_from_numpy(in_))
        cropped = pyms.utils.crop_to_bandwidth_limit_torch(bwlimited)
        passtest = (
            passtest and sumsqr_diff(torch.sum(bwlimited), torch.sum(cropped)) < 1e-10
        )
        gridshape = [127, 127]
        in_ = np.ones(gridshape)
        bwlimited = pyms.utils.bandwidth_limit_array(pyms.utils.cx_from_numpy(in_))
        cropped = pyms.utils.crop_to_bandwidth_limit(bwlimited)
        passtest = (
            passtest and sumsqr_diff(torch.sum(bwlimited), torch.sum(cropped)) < 1e-10
        )
        self.assertTrue(passtest)

    def test_bandwidth_limit_array(self):
        """Test bandwidth limiting function by comparing to known result."""
        grid = [7, 7]
        testarray = np.ones(grid)
        referencearray = np.asarray(
            [
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                1,
                1,
            ]
        ).reshape(grid)
        bandlimited = pyms.utils.bandwidth_limit_array(testarray, 2 / 3)
        passeven = np.all(bandlimited - referencearray == 0)
        grid = [8, 8]
        testarray = np.ones(grid)
        referencearray = np.asarray(
            [
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
            ]
        ).reshape(grid)
        bandlimited = pyms.utils.bandwidth_limit_array(testarray, 2 / 3)
        passodd = np.all(bandlimited - referencearray == 0)
        self.assertTrue(np.all([passeven, passodd]))

    def test_convolution(self):
        """Test Fourier convolution by convolving with shifted delta function."""
        shape = [128, 128]
        delta = np.zeros(shape)
        delta[5, 5] = 1
        # Make a circle
        circle = pyms.make_contrast_transfer_function(shape, [20, 20], 3e5, 20)

        convolved = pyms.utils.convolve(circle, delta)
        shifted = np.roll(circle, [5, 5], axis=[-2, -1])
        passcomplex = sumsqr_diff(convolved, shifted) < 1e-10
        convolved = pyms.utils.convolve(np.abs(circle), delta)
        passreal = sumsqr_diff(convolved, shifted) < 1e-10
        self.assertTrue(passcomplex and passreal)

    def test_fourier_interpolation(self):
        """Test fourier interpolation of a cosine function."""
        a = np.zeros((4, 4), dtype=np.float32)

        # Put in one full period of a cosine function (a checkerboard 1,0,-1,0)
        a[0, :] = 1
        a[2, :] = -1

        # In the fourier interpolated version we should recover the value of
        # cos(pi/2) = 1/sqrt(2) at the interpolated intermediate points
        passY = (
            pyms.utils.fourier_interpolate_2d(a, (8, 8))[1, 0] - 1 / np.sqrt(2) < 1e-10
        )

        # Test in x direction for completeness
        passX = (
            pyms.utils.fourier_interpolate_2d(a.T, (8, 8)).T[1, 0] - 1 / np.sqrt(2)
            < 1e-10
        )

        # Test that the option 'conserve_norm' works too.
        passNorm = (
            np.sum(
                pyms.utils.fourier_interpolate_2d(a, (8, 8), norm="conserve_L2") ** 2
            )
            - np.sum(a ** 2)
            < 1e-10
        )

        numpyVersionPass = (passY and passX) and passNorm

        # test pytorch versions
        passY = (
            pyms.utils.fourier_interpolate_2d_torch(
                pyms.utils.cx_from_numpy(a), (8, 8)
            )[1, 0, 0]
            - 1 / np.sqrt(2)
            < 1e-10
        )
        # print(pyms.utils.fourier_interpolate_2d_torch(pyms.utils.cx_from_numpy(a.T),(8,8)).shape)
        passX = (
            pyms.utils.fourier_interpolate_2d_torch(
                pyms.utils.cx_from_numpy(a.T), (8, 8)
            )[0, 1, 0]
            - 1 / np.sqrt(2)
            < 1e-10
        )

        # Test that the option 'conserve_norm' works too.
        passNorm = (
            torch.sum(
                pyms.utils.fourier_interpolate_2d_torch(
                    pyms.utils.cx_from_numpy(a.T), (8, 8), norm="conserve_norm"
                )
                ** 2
            )
            - torch.sum(pyms.utils.cx_from_numpy(a) ** 2)
            < 1e-10
        )
        pytorchVersionPass = passY.item() and passX.item() and passNorm.item()

        self.assertTrue(pytorchVersionPass and numpyVersionPass)

    def test_numpy_fft_shift(self):
        """Test to see if fourier shift correctly shifts a pixel 2 to the right."""
        test_array = np.zeros((5, 5))
        test_array[0, 0] = 1
        shifted1 = pyms.utils.fourier_shift(test_array, [2, 3])
        test_array[0, 0] = 0
        test_array[2, 3] = 1
        self.assertTrue(
            sumsqr_diff(shifted1, test_array) < 1e-10
            and sumsqr_diff(shifted1, test_array) < 1e-10
        )

    def test_crop(self):
        """Test cropping method on scipy test image."""
        # Get astonaut image
        from skimage.data import astronaut

        im = np.sum(astronaut(), axis=2).astype(np.float32)

        passTest = True

        for c_func, img in zip(
            [pyms.utils.crop, pyms.utils.crop_torch],
            [im, torch.as_tensor(copy.deepcopy(im))],
        ):
            # Check that normal cropping works
            cropPass = (
                sumsqr_diff(
                    c_func(img, [256, 256]), img[128 : 512 - 128, 128 : 512 - 128]
                )
                < 1e-10
            )
            passTest = passTest and cropPass

            # If an output size larger than that of the input is requested then the
            # input array should be padded instead, check that this is working too.
            pad = c_func(img, [700, 700])
            pad[350 - 256 : 350 + 256, 350 - 256 : 350 + 256] -= img
            padPass = sumsqr(pad) < 1e-10
            passTest = passTest and padPass

        self.assertTrue(passTest)

    def test_Gaussian(self):
        """Test the 2D Gaussian function (check standard deviation of result)."""
        sigma = [4, 8]
        gridshape = [128, 128]
        rsize = [128, 128]

        Gaussian = pyms.utils.Gaussian(sigma, gridshape, rsize)
        r = pyms.utils.r_space_array(gridshape, rsize)
        sigmayout = np.sqrt(np.sum((r[0] ** 2) * Gaussian))
        sigmaxout = np.sqrt(np.sum((r[1] ** 2) * Gaussian))
        passTest = (
            sumsqr_diff(np.asarray(sigma), np.asarray([sigmayout, sigmaxout])) < 1e-10
        )
        self.assertTrue(passTest)

    def test_torch_complex_matmul(self):
        """Test complex matmul against numpy complex matrix multiplication."""
        k, m, n = 2, 3, 4
        a = np.random.randint(-5, 5, size=k * m).reshape(
            (k, m)
        ) + 1j * np.random.randint(-5, 5, size=k * m).reshape((k, m))
        b = np.random.randint(-5, 5, size=m * n).reshape(
            (m, n)
        ) + 1j * np.random.randint(-5, 5, size=m * n).reshape((m, n))

        c = sumsqr_diff(
            a @ b,
            pyms.utils.cx_to_numpy(
                pyms.utils.complex_matmul(
                    *[pyms.utils.cx_from_numpy(x) for x in [a, b]]
                )
            ),
        )
        self.assertTrue(c < 1e-10)

    def test_torch_complex_mul(self):
        """Multiply 1 + i and 3 + 4i to give -1 + 7 i."""
        a = torch.as_tensor([1, 1])
        b = torch.as_tensor([3, 4])
        c = pyms.utils.complex_mul(a, b)
        self.assertTrue(float(sumsqr_diff(c, torch.as_tensor([-1, 7]))) < 1e-10)

    def test_torch_c_exp(self):
        """Test exponential function by calculating e^{i pi} and e^{1 + i pi}."""
        # Test e^{i pi} = -1
        # Test complex input
        arg = np.asarray(np.pi + 1j)
        a = pyms.utils.torch_c_exp(pyms.utils.cx_from_numpy(arg))
        passcomplex = (a.data[0] + 1 / np.exp(1.0)) ** 2 < 1e-10 and a.data[
            1
        ] ** 2 < 1e-10

        # Test real input
        arg = torch.as_tensor([np.pi])
        a = pyms.utils.torch_c_exp(arg)
        passreal = (a[0, 0] + 1) ** 2 < 1e-10
        passtest = passreal and passcomplex
        self.assertTrue(passtest.item())

    def test_torch_sinc(self):
        """Test sinc function by calculating [sinc(0),sinc(pi/2),sinc(pi)]."""
        x = torch.as_tensor([0, 1 / 2, 1, 3 / 2])
        y = torch.as_tensor([1.0, 2 / np.pi, 0, -2 / 3 / np.pi])
        self.assertTrue(sumsqr_diff(y, pyms.utils.sinc(x)) < 1e-7)

    def test_amplitude(self):
        """Test modulus square function by calculating a few known examples."""
        # 1 + i, 1-i and i
        x = torch.as_tensor([[1, 1], [1, -1], [0, 1]])
        y = torch.as_tensor([2, 2, 1])
        passcomplex = sumsqr_diff(y, pyms.utils.amplitude(x)).item() < 1e-7
        # 1,2,4 (test function for real numbers)
        x = torch.as_tensor([1, 2, 4])
        y = torch.as_tensor([1, 4, 16])
        passreal = sumsqr_diff(y, pyms.utils.amplitude(x)).item() < 1e-7
        self.assertTrue(passcomplex and passreal)

    def test_roll_n(self):
        """Test the roll (circular shift of array) function."""
        shifted = torch.zeros((5, 5))
        shifted[0, 0] = 1
        for a, n in zip([-1, -2], [2, 3]):
            shifted = pyms.utils.roll_n(shifted, a, n)
        test_array = torch.zeros((5, 5))
        test_array[2, 3] = 1
        self.assertTrue(sumsqr_diff(shifted, test_array).item() < 1e-10)

    def test_cx_from_numpy(self):
        """Test function for converting complex numpy arrays to complex tensors."""
        in_ = np.asarray([1.0 + 1.0j, 2.0 - 2.0j])
        out_ = torch.as_tensor([[1.0, 1.0], [2.0, -2.0]], device=torch.device("cpu"))
        self.assertTrue(
            sumsqr_diff(
                out_, pyms.utils.cx_from_numpy(in_, device=torch.device("cpu"))
            ).item()
            < 1e-7
        )

    def test_cx_to_numpy(self):
        """Test function for converting complex numpy arrays to complex tensors."""
        out_ = np.asarray([1.0 + 1.0j, 2.0 - 2.0j])
        in_ = torch.as_tensor([[1.0, 1.0], [2.0, -2.0]], device=torch.device("cpu"))
        self.assertTrue(sumsqr_diff(out_, pyms.utils.cx_to_numpy(in_)).item() < 1e-7)

    def test_fftfreq(self):
        """Test function for calculating Fourier frequency grid."""
        even = torch.as_tensor([0.0, 1.0, -2.0, -1.0])
        odd = torch.as_tensor([0.0, 1.0, 2.0, -2.0, -1.0])
        passeven = sumsqr_diff(pyms.utils.fftfreq(4), even) < 1e-8
        passodd = sumsqr_diff(pyms.utils.fftfreq(5), odd) < 1e-8
        self.assertTrue(passeven and passodd)


if __name__ == "__main__":

    # Code to run a single test function
    # test = Test_util_Methods()
    # test.test_crop_periodic_rectangle()

    # run all test functions
    unittest.main()
    clean_temp_structure()
    import sys

    sys.exit()

    # A 5 x 5 x 5 Angstrom cell with an oxygen
    cell = [5, 5, 0.0001]
    atoms = [[0.0, 0.0, 0.0, 8]]
    crystal = pyms.structure(cell, atoms, dwf=[0.0])
    gridshape = [64, 64]
    eV = 3e5
    thicknesses = cell[2]
    subslices = [1.0]
    app = None
    # Target the oxygen K edge
    Ztarget = 8

    # principal and orbital angular momentum quantum numbers for bound state
    n = 1
    ell = 0
    lprime = 0
    from pyms import orbital, get_q_numbers_for_transition, transition_potential

    qnumbers = get_q_numbers_for_transition(ell, order=1)[1]
    bound_configuration = "1s2 2s2 2p4"
    excited_configuration = "1s1 2s2 2p4"
    epsilon = 1

    bound_orbital = orbital(Ztarget, bound_configuration, n, ell, lprime)
    excited_orbital = orbital(Ztarget, excited_configuration, 0, qnumbers[0], 10)

    # Calculate transition potential for this escited state
    Hn0 = transition_potential(
        bound_orbital, excited_orbital, gridshape, cell, qnumbers[2], qnumbers[1], eV
    ).reshape((1, *gridshape))

    P, T = pyms.multislice_precursor(
        crystal, gridshape, eV, subslices, nT=1, showProgress=False
    )

    EFTEM_images = pyms.EFTEM(
        crystal,
        gridshape,
        eV,
        app,
        thicknesses,
        Ztarget,
        n,
        ell,
        epsilon,
        df=[0],
        nT=1,
        Hn0=Hn0,
        subslices=subslices,
        nfph=1,
        P=P,
        T=T,
        showProgress=False,
    )

    # By reciprocity, a point detector should give identical results to
    # EFTEM so this is how we test the STEM routine as well.
    det = np.zeros((1, *gridshape))
    det[0, 0] = 1
    STEM_images = pyms.STEM_EELS_multislice(
        crystal,
        gridshape,
        eV,
        app,
        det,
        thicknesses,
        Ztarget,
        n,
        ell,
        epsilon,
        df=0,
        nT=1,
        P=P,
        T=T,
        showProgress=False,
        Hn0=Hn0,
    )

    # The image of the transition is "lensed" by the atom that it is passed
    # through so we must account for this by multiplying the Hn0 by the transmission
    # function
    Hn0 *= pyms.utils.cx_to_numpy(T[0, 0]) / np.sqrt(np.prod(gridshape))

    result = np.fft.fftshift(np.squeeze(EFTEM_images))
    reference = np.fft.fftshift(
        np.abs(
            pyms.utils.fourier_interpolate_2d(
                pyms.utils.bandwidth_limit_array(
                    Hn0[0], qspace_in=False, qspace_out=False
                ),
                result.shape,
                "conserve_norm",
            )
        )
        ** 2
    )

    # import matplotlib.pyplot as plt
    # fig,ax = plt.subplots(nrows=2)
    # ax[0].imshow(reference)
    # ax[1].imshow(result)
    # plt.show(block=True)

    passEFTEM = np.sum(np.abs(result - reference) ** 2) < 1e-7
    # self.assertTrue(passEFTEM)
