"""Unit tests for the py_multislice package."""

import copy
import unittest
import pyms
import numpy as np
import os
import torch

required_accuracy = 1e-4
scratch_cell_name = "test_cell.xyz"


def sumsqr(array):
    """Calculate the sum of squares for a single array."""
    return np.sum(np.square(array))


def sumsqr_diff(array1, array2):
    """Calculate the sum of squares normalized difference of two arrays."""
    return sumsqr(array1 - array2) / sumsqr(array1)


def clean_temp_structure():
    """Remove the temporary files made by the unit tests."""
    if os.path.exists(scratch_cell_name):
        os.remove(scratch_cell_name)


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


class Test_util_Methods(unittest.TestCase):
    """Test the utility functions for numpy and pytorch."""

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

        img = np.sum(astronaut(), axis=2).astype(np.float32)

        # Check that normal cropping works
        cropPass = (
            sumsqr_diff(
                pyms.utils.crop(img, [256, 256]), img[128 : 512 - 128, 128 : 512 - 128]
            )
            < 1e-10
        )

        # If an output size larger than that of the input is requested then the
        # input array should be padded instead, check that this is working too.
        pad = pyms.utils.crop(img, [700, 700])
        pad[350 - 256 : 350 + 256, 350 - 256 : 350 + 256] -= img
        padPass = sumsqr(pad) < 1e-10
        self.assertTrue(padPass and cropPass)

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

        numpyVersionPass = passY and passX

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
        pytorchVersionPass = passY.item() and passX.item()

        self.assertTrue(pytorchVersionPass and numpyVersionPass)


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

        cell = pyms.structure(scratch_cell_name)

        atomspass = np.all(np.square(atoms - cell.atoms) < 1e-10)
        celldimpass = np.all(np.square(ucell - cell.unitcell) < 1e-10)

        self.assertTrue(atomspass and celldimpass)

    def test_orthorhombic_supercell(self):
        """Test by opening graphene (which has a hexagonal unit cell)."""
        fnam = scratch_cell_name.replace(".xyz", ".p1")
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

        graphene = pyms.structure(fnam)
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
        cell = pyms.structure(scratch_cell_name)
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

    def testrot90(self):
        """Test of the rot90 function (under construction)."""
        # Make random test object
        atoms, ucell = make_temp_structure()

        cell = pyms.structure(scratch_cell_name)

        newcell = copy.deepcopy(cell)
        newcell = newcell.rot90(k=1, axes=(0, 1))


class Test_py_multislice_Methods(unittest.TestCase):
    """Tests for functions inside of the py_multislice.py file."""

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

        cell = pyms.structure(scratch_cell_name)

        # Parameters for test
        tiling = [4, 4]
        gridshape = [128, 128]
        eV = 3e5
        app = 20
        thicknesses = [20]
        subslices = [0.5, 1.0]
        df = -300
        probe_posn = [1.0, 1.0]
        PRISM_factor = [2, 2]
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
            GPU_streaming=True,
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
            scan_posn=np.asarray(probe_posn).reshape((1, 2)) / np.asarray(tiling),
            showProgress=False,
            S=S,
            nT=5,
            GPU_streaming=True,
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
        probe = torch.ifft(probe, signal_ndim=2)

        # Adjust normalization to account for prism cropping
        probe *= np.prod(PRISM_factor)

        # Do multislice
        probe = pyms.multislice(
            probe.view([1, *gridshape, 2]), nslices, P, T, return_numpy=False
        )

        # Get output gridsize
        gridout = torch.squeeze(probe).shape[:2]

        # Now crop multislice result in real space
        crop_ = [
            torch.arange(
                -gridout[i] // (2 * PRISM_factor[i]),
                gridout[i] // (2 * PRISM_factor[i]),
            )
            for i in range(2)
        ]
        window = pyms.utils.crop_window_to_flattened_indices_torch(
            [
                (crop_[i] + probe_posn[i] / tiling[i] * gridout[i]) % gridout[i]
                for i in range(2)
            ],
            gridout,
        )

        probe = torch.squeeze(probe).flatten(0, 1)
        new = torch.zeros(np.prod(gridout), 2, dtype=S.dtype, device=S.device)
        new[window] = probe[window]
        probe = new.reshape(*gridout, 2)
        probe = torch.fft(probe, signal_ndim=2) / np.sqrt(np.prod(gridout))
        ms_CBED = np.fft.fftshift(
            np.squeeze(np.abs(pyms.utils.cx_to_numpy(probe)) ** 2)
        )
        # import matplotlib.pyplot as plt

        # fig,ax = plt.subplots(nrows=3)
        # ax[0].imshow(np.squeeze(ms_CBED))
        # ax[1].imshow(np.squeeze(S_CBED_amp))
        # ax[2].imshow(datacube[0])
        # plt.show(block=True)
        # The test is passed if the result from multislice and S-matrix is
        # within numerical error and the S matrix wrapper routine is also
        # behaving as expected
        Smatrixtestpass = (
            sumsqr_diff(ms_CBED, S_CBED_amp) < 1e-10
            and sumsqr_diff(S_CBED_amp, datacube) < 1e-10
        )
        self.assertTrue(Smatrixtestpass)


if __name__ == "__main__":
    unittest.main()
    clean_temp_structure()
    # sys.exit()

    # atoms, ucell = make_temp_structure()

    # cell = pyms.structure(scratch_cell_name)

    # oldatoms = np.asarray(cell.atoms[:,:3]).T
    # oldcell = np.asarray(cell.unitcell)
    # print(cell.atoms[:,:3])

    # cell = cell.rot90(k=1, axes=(0, 1))

    # atoms = (cell.atoms[:,:3]).T

    # print(cell.atoms[:,:3])
    # atomspass = np.all(atoms.T[:2]==oldatoms.T[:2][::-1])
    # print(atomspass)
    # unitCellPass = np.all(oldcell[:2] == cell.unitcell[:2][::-1])
    # sys.exit()
    # newcell = copy.deepcopy(cell)
    # newcell = newcell.rot90(k=1, axes=(0, 2))
    # unitCellPass = unitCellPass and np.all(cell.unitcell[::2] ==
    # newcell.unitcell[::2][::-1])
    # print(newcell.unitcell,cell.unitcell,unitCellPass)

    # unittest.main()
    # clean_temp_structure()
