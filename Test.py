"""Unit tests for the py_multislice package."""

import copy
import unittest

from sklearn.datasets import make_swiss_roll
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
        # if array.ndim < 1:
        #     return torch.abs(array) ** 2
        return torch.sum(pyms.utils.amplitude(array))
    return np.sum(np.square(np.abs(array)))


def sumsqr_diff(array1, array2,norm=False):
    """Calculate the sum of squares normalized difference of two arrays."""
    if norm:
        return sumsqr(array1 - array2) / sumsqr(array1)
    else:
        return sumsqr(array1 - array2)


def clean_temp_structure():
    """Remove the temporary files made by the unit tests."""
    if os.path.exists(scratch_cell_name):
        os.remove(scratch_cell_name)


def Single_atom(Z=79, r=[0.5, 0.5, 0.5], R=[10.0, 10.0, 10.0],dwf = 0.0):
    """Single atom structure for testing."""
    atoms = np.asarray(
        [
            [*r, Z, 1.0, dwf],
        ]
    )
    return pyms.structure(R, atoms[:, :4], dwf=atoms[:, 5])


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


def make_temp_structure(atoms=None, title="scratch", ucell=None, seed=0, natoms=None):
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
        if natoms is None:
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
        390-397
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
        # fig,ax = plt.subplots()
        # ax.plot(g,pyms.electron_scattering_factor(47, g ** 2, units="A"),'b-')
        # ax.plot(g,fe,'r--')

        # plt.show(block=True)

        sse = np.sum(
            np.square(fe - pyms.electron_scattering_factor(47, g ** 2, units="A"))
        )
        self.assertTrue(sse < 0.01)

    # def test_calculate_scattering_factors(self):
    #     """Test calculate scattering factors against precomputed standard."""
    #     calc = pyms.calculate_scattering_factors([3, 3], [1, 1], [49])
    #     known = np.asarray(
    #         [
    #             [
    #                 [499.59366, 107.720055, 107.720055],
    #                 [107.720055, 65.67732, 65.67732],
    #                 [107.720055, 65.67732, 65.67732],
    #             ]
    #         ],
    #         dtype=pyms._float,
    #     )
    #     self.assertTrue(np.sum(np.square(known - calc)) < 0.01)

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
                    np.asarray([0, 1, 0], dtype=int),
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
        """Test the make potential function against the values in Kirkland's book."""

        Kirkland = np.asarray(
            [
                0.022206387004698444,
                4938.406696881273,
                0.023349930244864578,
                4706.494070900851,
                0.025310290085149373,
                4510.422487117403,
                0.027270649925434168,
                4336.488017632086,
                0.029231009765718963,
                4172.040882845968,
                0.031191369606003758,
                4023.405972558515,
                0.03315172944628855,
                3884.2583969702614,
                0.03560217924664455,
                3721.919558783966,
                0.038542719007071735,
                3546.9309409987377,
                0.04148325876749892,
                3386.7003994122633,
                0.04442379852792612,
                3243.336230624366,
                0.047854428248424494,
                3087.3222822375365,
                0.051775147928994084,
                2926.0375923511515,
                0.05569586760956367,
                2782.1463494132986,
                0.06010667725020445,
                2634.4601725957655,
                0.06500757685091645,
                2483.927795368473,
                0.07039856641169963,
                2332.552099501252,
                0.07709646253267267,
                2165.996668115312,
                0.08363099533362199,
                2029.258574355426,
                0.09098234473468995,
                1884.8026091140496,
                0.09931387405590035,
                1742.2290515510854,
                0.1086255832972531,
                1600.9731793630094,
                0.11891747245874827,
                1465.2755436653733,
                0.12969945158031465,
                1342.8026775484395,
                0.14048143070188102,
                1234.704560975747,
                0.15126340982344738,
                1140.4062039655255,
                0.16204538894501375,
                1056.4576666271578,
                0.17282736806658006,
                984.5839189059525,
                0.18360934718814642,
                917.3100910389039,
                0.1943913263097128,
                858.0861229166303,
                0.20517330543127915,
                804.6120546120546,
                0.21595528455284552,
                758.0378660887127,
                0.22673726367441188,
                712.6136575289111,
                0.23751924279597825,
                672.3643588050363,
                0.2483012219175446,
                634.9900099900096,
                0.259083201039111,
                601.6405910473704,
                0.26986518016067734,
                569.4411520682697,
                0.2806471592822437,
                540.116662998018,
                0.2914291384038101,
                512.5171438730749,
                0.30221111752537644,
                487.2175846752107,
                0.3129930966469428,
                463.6429954226551,
                0.32377507576850917,
                442.3683660971792,
                0.33455705489007553,
                419.36876682639286,
                0.3453390340116419,
                402.11906737330446,
                0.35612101313320826,
                382.56940799313634,
                0.36592281233463225,
                366.9169029338518,
                0.38846695049790736,
                334.27024952448664,
                0.3992489296194737,
                321.0454799437839,
                0.4100309087410401,
                307.82071036308207,
                0.4208128878626065,
                295.1709307641504,
                0.4315948669841728,
                283.096141146988,
                0.44237684610573924,
                272.17133149336496,
                0.45315882522730555,
                260.09654187620254,
                0.46688134410929916,
                247.44676227726995,
                0.47766332323086547,
                239.396902532495,
                0.4884453023524319,
                230.7720528059508,
                0.4953065617934287,
                222.1472030794057,
            ]
        ).reshape([392 - 335, 2])

        # Make single Silver atom
        R = 10
        cell = Single_atom(r=[0, 0, 0], Z=92, R=[R, R, 2])
        tiling = [2, 2]
        # Note that this sampling is insufficient sampling for agreement <0.1 A
        # but is chosen for faster testing
        # For better agreement increaes this to [1024,1024]
        gridshape = [128, 1024]

        pot = cell.make_potential(gridshape).cpu().numpy()

        # fig,a = plt.subplots(ncols=1)
        # xlim = [0.0,0.5]
        # ylim = [0,5e3]
        # a.plot(np.fft.fftfreq(gridshape[1],d=1)[:gridshape[1]//2]*R,pot[0,:gridshape[1]//2],'b-',label='pyms')
        # a.plot(Kirkland[:,0],Kirkland[:,1],'r--',label='Kirkland')
        # a.set_title('Potential of U atom')
        # a.legend()
        # a.set_ylabel('V(r)')
        # a.set_xlabel('r ($\\AA$)')
        # a.set_xlim(xlim)
        # a.set_ylim(ylim)

        # plt.show(block=True)

        from scipy.interpolate import interp1d

        kirk = interp1d(Kirkland[:, 0], Kirkland[:, 1])
        x = np.fft.fftfreq(gridshape[1], d=1)[: gridshape[1] // 2] * R
        pyms_pot = interp1d(x, pot[0, 0, : gridshape[1] // 2])
        xtest = np.linspace(0.1, 0.45)

        K = kirk(xtest)
        within_spec = (
            np.sum(np.square(kirk(xtest) - pyms_pot(xtest)) / np.sum(K ** 2)) < 1e-4
        )
        self.assertTrue(within_spec)

    def test_atom_placement(self):
        """Test that the code places atoms in the correct location with correct
           mean square vibration"""
        nT = 128
        gridshape = [128,128]
        # Make single Silver atom
        R = 10
        dwf = 0.02
        r = [0.25,0.25,0.25]
        cell = Single_atom(r=r, Z=2, R=[R, R, 2],dwf=dwf)

        pot = np.zeros([nT]+gridshape)
        for i in range(nT):
            pot[i] = cell.make_potential(gridshape).cpu().numpy()

        # Calculate COM in fraction of unit cell
        M = np.sum(pot,axis=(-2,-1))
        COMx = np.sum(pot*np.arange(gridshape[-1])/gridshape[-1],axis=(-2,-1))/M
        COMy = np.sum(pot*np.arange(gridshape[-2]).reshape([gridshape[-2],1])/gridshape[-2],axis=(-2,-1))/M
        COM = np.stack([COMy,COMx],axis=0)
        # fig,ax=plt.subplots(ncols=2)
        # ax[0].imshow(np.sum(pot,axis=0),origin='lower')
        # ax[1].plot(COMx,COMy,'bo')
        # plt.show(block=True)

        moving_average = np.cumsum(COM,axis=1)/(np.arange(nT)+1)
        moving_ms = np.cumsum(COMx**2+COMy**2)/(np.arange(nT)+1)/2 - np.average(moving_average**2,axis=0)
        moving_ms*= R**2

        # fig,ax = plt.subplots(ncols=2)
        # ax[0].plot(COMx,COMy,'bo')
        # ax[1].plot(moving_ms)
        # from matplotlib.patches import Circle
        # ax[0].add_patch(Circle(r[:2],radius=np.sqrt(dwf)/R))
        # plt.show(block=True)

        meanCOM = np.sum(COM,axis=1)/nT
        # 5 Sigma test, ie. 1/3.5 million probability of failing assuming correct functioning
        fiveSigma = dwf/nT*5**2
        sigmaofvariance = dwf*np.sqrt(2/(nT-1))
        muwithinspec = np.sum((meanCOM-r[0])**2) < fiveSigma
        sigmawithinspec = np.abs(moving_ms[-1]-dwf) < 5*sigmaofvariance

        withinspec = sigmawithinspec and muwithinspec
        self.assertTrue(withinspec)


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
            showProgress=False,
            tiling=tiling,
            nT=nT,
            **kwargs
        )
        P2, T2 = pyms.multislice_precursor(
            BTO,
            gridshape,
            eV,
            subslices[1],
            showProgress=False,
            tiling=tiling,
            nT=nT,
            **kwargs
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
        sameP = sumsqr_diff(T.Propagator[:N].cpu().numpy(), P1) < 1e-10
        PP = T.Propagator[N : N + len(subslices[1])].cpu().numpy()
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

    def test_EFTEM_and_multislice_STEM_EELS(self):
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
        gridshape = [128, 128]
        eV = 3e5
        thicknesses = cell[2]
        subslices = [1.0]
        app = 20  # pyms.max_grid_resolution(gridshape,crystal.unitcell[:2],eV=eV)
        # Target the oxygen K edge
        Ztarget = 8

        # principal and orbital angular momentum quantum numbers for bound state
        n = 1
        ell = 0
        lprime = 0
        from pyms import orbital, get_q_numbers_for_transition, transition_potential

        # qnumbers will contain the quantum numbers [lprime,mlprime,ml]
        qnumbers = get_q_numbers_for_transition(ell, order=1)[1]
        bound_configuration = "1s2 2s2 2p4"
        excited_configuration = "1s1 2s2 2p4"
        epsilon = 10

        bound_orbital = orbital(Ztarget, bound_configuration, n, ell, lprime)
        excited_orbital = orbital(
            Ztarget, excited_configuration, 0, qnumbers[0], epsilon
        )

        # Calculate transition potential for this excited state
        Hn0 = transition_potential(
            bound_orbital,
            excited_orbital,
            gridshape,
            cell,
            qnumbers[2],
            qnumbers[1],
            eV,
        ).reshape((1, *gridshape))
        # plt.imshow(np.abs(Hn0[0]))
        # plt.show(block=True)

        P, T = pyms.multislice_precursor(
            crystal, gridshape, eV, subslices, nT=1, showProgress=False
        )

        ctf = pyms.make_contrast_transfer_function(
            gridshape, crystal.unitcell[:2], eV, app
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
        result = np.fft.fftshift(np.squeeze(EFTEM_images))

        # By reciprocity, a point detector should give identical results to
        # EFTEM so this is how we test the STEM routine as well. We work out the
        # size of a single pixel in mrad to make such a point detector.
        detector_ranges = [np.amin(1 / crystal.unitcell[:2] * 1e3 / pyms.wavev(eV))]
        STEM_images = pyms.STEM_EELS_multislice(
            crystal,
            gridshape,
            eV,
            app,
            detector_ranges,
            thicknesses,
            Ztarget,
            n,
            ell,
            epsilon,
            ionization_cutoff=None,
            df=0,
            nT=1,
            nfph=1,
            P=P,
            T=T,
            showProgress=False,
            Hn0=Hn0,
        )
        # plt.imshow(STEM_images[0])
        # plt.show(block=True)
        STEM_images = np.fft.fftshift(
            pyms.utils.fourier_interpolate(
                STEM_images, result.shape, norm="conserve_val"
            )
        )

        # The image of the transition is "lensed" by the atom that it is passed
        # through so we must account for this by multiplying the Hn0 by the transmission
        # function
        Hn0 *= T[0, 0].cpu().numpy() / np.sqrt(np.prod(gridshape))

        reference = np.fft.fftshift(
            pyms.utils.fourier_interpolate(
                np.abs(np.fft.ifft2(ctf * np.fft.fft2(Hn0[0]))) ** 2,
                result.shape,
                "conserve_L1",
                qspace_out=False,
            )
        )

        # To get STEM and EFTEM images on same scale need to multiply by number
        # of pixels in STEM probe forming aperture function then divide by number
        # of pixels in STEM image
        STEM_adj = pyms.make_contrast_transfer_function(
            gridshape, crystal.unitcell[:2], eV, app
        )
        STEM_adj = np.sum(np.abs(STEM_adj) ** 2)
        STEM_images *= STEM_adj / np.prod(result.shape)

        # import matplotlib.pyplot as plt
        # fig,ax = plt.subplots(nrows=3)
        # ax[0].imshow(reference)
        # ax[1].imshow(result)
        # ax[2].imshow(np.squeeze(STEM_images))
        # plt.show(block=True)

        passEFTEM = np.sum(np.abs(result - reference) ** 2) < 1e-7
        passSTEMEELS = np.sum(np.abs(STEM_images - reference) ** 2) < 1e-7
        self.assertTrue(passEFTEM and passSTEMEELS)

    def test_PRISM_STEM_EELS(self):
        """
        Test that the PRISM STEM_EELS.

        Calculate a EELS image and then compare with standard PRISM STEM EELS,
        which is itself tested against the EFTEM routine.
        """
        # A STO crystal for testing
        structure = SrTiO3()
        thickness = 20
        # An even more basic test object
        # A 5 x 5 x 5 Angstrom cell with an oxygen
        # cell = [10, 10, 0.0001]
        # atoms = np.asarray([[0.5, 0.5, 0.0, 8]])#,[0.25,0.25,0.0,8]])
        # structure = pyms.structure(cell, atoms, dwf=np.zeros(len(atoms)))
        # thickness = 0.00009
        df = -10
        aberrations = pyms.aberration_starter_pack()
        aberrations[1].amplitude = 50
        structure.atoms[:, 5] = 0
        gridshape = [128, 128]
        tiling = [1, 1]  # [4,4]#[2, 2]
        rsize = np.asarray(tiling) * structure.unitcell[:2]
        PRISM_factor = [1, 1]  # tiling#[1,1]#[2, 2]
        Hn0_crop = None
        # Target the oxygen K edge
        Ztarget = 8
        eV = 3e5
        app = 20
        det = 30
        subslices = [1.0]

        # principal and orbital angular momentum quantum numbers for bound state
        n = 1
        ell = 0
        lprime = 0
        from pyms import orbital, get_q_numbers_for_transition, transition_potential

        # qnumbers will contain the quantum numbers [lprime,mlprime,ml]
        qnumbers = get_q_numbers_for_transition(ell, order=1)[1]
        bound_configuration = "1s2 2s2 2p4"
        excited_configuration = "1s1 2s2 2p4"
        epsilon = 10

        bound_orbital = orbital(Ztarget, bound_configuration, n, ell, lprime)
        excited_orbital = orbital(
            Ztarget, excited_configuration, 0, qnumbers[0], epsilon
        )

        # Calculate transition potential for this excited state
        # cropped_grid = pyms.utils.size_of_bandwidth_limited_array(gridshape)
        Hn0s = transition_potential(
            bound_orbital,
            excited_orbital,
            gridshape,
            rsize,
            qnumbers[2],
            qnumbers[1],
            eV,
        ).reshape((1, *gridshape))

        PRISM_result = pyms.STEM_EELS_PRISM(
            structure,
            gridshape,
            eV,
            app,
            det,
            thickness,
            Ztarget,
            n,
            ell,
            epsilon,
            nfph=1,
            df=df,
            aberrations=aberrations,
            Hn0s=pyms.utils.crop_to_bandwidth_limit(
                Hn0s, qspace_in=False, qspace_out=False, norm="conserve_val"
            ),
            PRISM_factor=PRISM_factor,
            Hn0_crop=Hn0_crop,
            subslices=subslices,
            nT=1,
            contr=None,
            tiling=tiling,
            showProgress=False,
            do_reverse_multislice=False,
        )

        ms_result = pyms.STEM_EELS_multislice(
            structure,
            gridshape,
            eV,
            app,
            det,
            thickness,
            Ztarget,
            n,
            ell,
            epsilon,
            df=df,
            aberrations=aberrations,
            nfph=1,
            Hn0=Hn0s,
            nT=1,
            contr=None,
            subslices=subslices,
            ionization_cutoff=None,
            showProgress=False,
        )
        # fig, ax = plt.subplots(ncols=3)
        # ax[0].imshow(PRISM_result)
        # ax[1].imshow(ms_result[0])
        # ax[2].imshow(ms_result[0]-PRISM_result)
        # plt.show(block=True)
        self.assertTrue(
            sumsqr_diff(ms_result[0], PRISM_result) / np.mean(ms_result[0]) < 1e-4
        )


class Test_py_multislice_Methods(unittest.TestCase):
    """Tests for functions inside of the py_multislice.py file."""

    def test_reverse_multislice(self):
        """Test the reverse multislice option."""
        structure = SrTiO3()
        gridshape = [512, 512]
        tiling = [4, 4]
        eV = 3e5
        subslicing = [0.5, 1.0]
        slices = pyms.generate_slice_indices(5, len(subslicing), subslicing=False)
        seed = np.random.randint(0, 2 ** 31 - 1, size=len(slices))
        P, T = pyms.multislice_precursor(
            structure,
            gridshape,
            eV,
            tiling=tiling,
            band_width_limiting=[1, 1],
            displacements=True,
            subslices=subslicing,
        )
        # print(T.shape)
        # fig,ax = plt.subplots(ncols=3)
        # ax[0].imshow(np.angle(P))
        # ax[1].imshow(np.imag(np.fft.ifft2(T[0,0].cpu().numpy())))
        # pot = structure.make_potential(
        #     gridshape,
        #     subslicing,
        #     [5,5],
        # )[0]

        # ax[2].imshow(pot.cpu().numpy())
        # ax[2].imshow(np.angle())
        # plt.show(block=True)

        rsize = structure.unitcell[:2] * np.asarray(tiling)

        # Make a plane wave probe
        probe = pyms.plane_wave_illumination(gridshape, rsize, eV)

        # Propagate forward
        exitprobe = pyms.multislice(
            probe,
            slices,
            P,
            T,
            output_to_bandwidth_limit=False,
            return_numpy=False,
            tiling=tiling,
            subslicing=True,
            seed=seed,
        )

        eprobe = exitprobe.cpu().numpy()

        # Now propagate back
        entranceprobe = pyms.multislice(
            exitprobe,
            slices[::-1],
            P,
            T,
            tiling=tiling,
            reverse=True,
            subslicing=True,
            seed=seed,
            output_to_bandwidth_limit=False,
        )
        # Make sure that errors are low
        self.assertTrue(
            np.std(np.angle(entranceprobe)) / np.std(np.angle(eprobe)) < 1e-2
        )

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

        im = np.fft.fftshift(np.sum(astronaut(), axis=2).astype(pyms._float))
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
        tiling = [2, 2]
        sampling = 0.01
        gridshape = np.ceil(cell.unitcell[:2] * np.asarray(tiling) / sampling).astype(
            int
        )
        eV = 3e5
        app = 20
        df = -100
        # df=0

        rsize = np.asarray(tiling) * cell.unitcell[:2]
        probe = pyms.focused_probe(gridshape, rsize, eV, app, df=df, qspace=True)
        detector = pyms.make_detector(gridshape, rsize, eV, app, app / 2)

        # Get phase contrast transfer funciton in real space
        PCTF = np.real(
            np.fft.ifft2(pyms.STEM_phase_contrast_transfer_function(probe, detector))
        )

        P, T = pyms.multislice_precursor(
            cell,
            gridshape,
            eV,
            tiling=tiling,
            displacements=False,
            showProgress=False,
            nT=1,
        )

        # Calculate phase of transmission function
        phase = np.angle(T[0, 0].cpu().numpy())

        # Weak phase object approximation = meansignal + PCTF * phase
        meansignal = np.sum(np.abs(probe) ** 2 * detector) / np.sum(np.abs(probe) ** 2)
        WPOA = np.real(pyms.utils.convolve(PCTF, phase))

        # Perform Multislice calculation and then compare to weak phase object
        # result
        images = (
            pyms.STEM_multislice(
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
            - meansignal
        )

        # Testing the PRISM code takes ~3 mins on a machine with a consumer
        # GPU so generally this part of the test is skipped
        testPRISM = True
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
            PRISMimages = np.tile(PRISMimages, tiling)

        images = np.tile(images, tiling)
        WPOA = pyms.utils.fourier_interpolate(WPOA, images.shape[-2:])

        # fig,ax = plt.subplots(nrows = 3+testPRISM)
        # ax[0].imshow(WPOA)
        # ax[1].imshow(images)
        # ax[2].imshow(images-WPOA)
        # if testPRISM:
        #     ax[-1].imshow(PRISMimages-meansignal)
        # plt.show(block=True)

        # Test is passed if sum squared difference with weak phase object
        # approximation is within reason
        passTest = (
            sumsqr_diff(WPOA, images) / np.sum(np.square(images)) / np.prod(gridshape)
            < 1e-5
        )
        if testPRISM:
            passTest = (
                passTest
                and sumsqr_diff(PRISMimages - meansignal, images)
                / np.sum(np.square(images))
                / np.prod(gridshape)
                < 1e-5
            )
        self.assertTrue(passTest)

    def test_DPC(self):
        """Test DPC, and by extension STEM, routine by comparing reconstruction to input potential."""
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
            np.abs(probe) ** 2, np.angle(T[0, 0].cpu().numpy())
        )
        convolved_phase -= np.amin(convolved_phase)

        reconstructed_phase = pyms.utils.fourier_interpolate(
            np.tile(result["DPC"], tiling), gridshape
        )
        reconstructed_phase -= np.amin(reconstructed_phase)

        fig,ax = plt.subplots(ncols=4)
        # from PIL import Image
        # Image.fromarray(convolved_phase).save('convolvedphase.tif')
        # ax[0].imshow(convolved_phase)
        # ax[1].imshow(reconstructed_phase)
        # Image.fromarray(reconstructed_phase).save('reconstructed_phase.tif')
        # ax[2].imshow(convolved_phase-reconstructed_phase)
        # ax[3].imshow((convolved_phase-reconstructed_phase),vmin=convolved_phase.min(),vmax=convolved_phase.max())
        # plt.show(block=True)
        # print(sumsqr_diff(convolved_phase, reconstructed_phase,norm=True) )
        self.assertTrue(
            sumsqr_diff(convolved_phase, reconstructed_phase,norm=True)
            < 1e-1
        )

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
        probe_posn = np.asarray([1.0, 1.0])
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
        device = pyms.utils.get_device()

        # S matrix calculation
        # Calculate scattering matrix
        rsize = np.asarray(tiling) * cell.unitcell[:2]
        nslices = np.ceil(thicknesses[0] / cell.unitcell[2]).astype(int)
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
        probe = torch.from_numpy(
            pyms.utils.fourier_shift(
                probe,
                np.asarray(probe_posn) / np.asarray(tiling),
                qspacein=True,
                qspaceout=True,
                pixel_units=False,
            )
        ).to(device)

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
        yy = np.fft.fftfreq(gridshape[0], 1.0 / gridshape[0]).astype(pyms._int)
        xx = np.fft.fftfreq(gridshape[1], 1.0 / gridshape[1]).astype(pyms._int)
        mask = np.logical_and(
            np.logical_and(
                (yy % PRISM_factor[0] == 0)[:, None],
                (xx % PRISM_factor[1] == 0)[None, :],
            ),
            pyms.amplitude(probe).cpu().numpy() > 0,
        )
        probe[np.logical_not(mask)] = 0

        # Transform probe back to real space
        probe = torch.fft.ifftn(probe, dim=(-2, -1))

        # Adjust normalization to account for prism cropping
        probe *= np.prod(PRISM_factor)

        # Do multislice
        probe = pyms.multislice(
            probe.view([1, *gridshape]), nslices, P, T, return_numpy=False
        )

        # Get output gridsize
        gridout = torch.squeeze(probe).shape[:2]

        # Now crop multislice result in real space
        grid = probe.shape[-2:]
        stride = [x // y for x, y in zip(grid, PRISM_factor)]
        halfstride = [x // 2 for x in stride]
        start = [
            int(np.round(probe_posn[i] * grid[i])) - halfstride[i] for i in range(2)
        ]
        windows = pyms.utils.crop_window_to_periodic_indices(
            [start[0], stride[0], start[1], stride[1]], gridout
        )

        new = torch.zeros(*gridout, dtype=S.dtype, device=probe.device)
        for wind in windows:
            new[
                wind[0][0] : wind[0][0] + wind[0][1],
                wind[1][0] : wind[1][0] + wind[1][1],
            ] = probe[
                0,
                wind[0][0] : wind[0][0] + wind[0][1],
                wind[1][0] : wind[1][0] + wind[1][1],
            ]
        probe = new
        probe = torch.fft.fftn(probe, dim=[-2, -1]) / np.sqrt(np.prod(gridout))
        ms_CBED = np.fft.fftshift(np.squeeze(np.abs(probe.cpu().numpy()) ** 2))

        # fig, ax = plt.subplots(ncols=3)
        # ax[0].imshow(ms_CBED)
        # ax[1].imshow(datacube[0, 0])
        # ax[2].imshow(S_CBED_amp[0])
        # plt.show(block=True)

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
        grid = np.zeros(gridshape, dtype=pyms._int).ravel()
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
        oddgridshape = [5, 128, 128]
        evengridshape = [5, 127, 127]
        in_ = np.ones(evengridshape)
        bwlimited = pyms.utils.bandwidth_limit_array(in_)
        cropped = np.fft.ifft2(pyms.utils.crop_to_bandwidth_limit(bwlimited))
        # Arrays are compared in real space since a 1/#pixels scaling is applied
        # to arrays with a standard FFT and the crop_to_bandwidth_limit
        # scales to account for this
        passtest = passtest and sumsqr_diff(sumsqr(np.fft.ifft2(bwlimited)), sumsqr(cropped)) < 1e-10
        in_ = np.ones(oddgridshape)
        bwlimited = pyms.utils.bandwidth_limit_array(in_)
        cropped = np.fft.ifft2(pyms.utils.crop_to_bandwidth_limit(bwlimited))
        passtest = passtest and sumsqr_diff(sumsqr(np.fft.ifft2(bwlimited)), sumsqr(cropped)) < 1e-10

        # Now test torch version
        in_ = np.ones(oddgridshape)
        bwlimited = pyms.utils.bandwidth_limit_array(torch.from_numpy(in_))
        cropped = torch.fft.ifft2(pyms.utils.crop_to_bandwidth_limit_torch(bwlimited))
        passtest = (
            passtest and sumsqr_diff(sumsqr(torch.fft.ifft2(bwlimited)), sumsqr(cropped)) < 1e-10
        )
        # fig,ax = plt.subplots(ncols=2)
        # ax[0].imshow(np.abs(bwlimited[1].cpu().numpy()))
        # ax[1].imshow(np.abs(torch.fft.fft2(cropped[1].cpu().numpy())))
        # plt.show(block=True)
        in_ = np.ones(evengridshape)
        bwlimited = torch.fft.ifft2(pyms.utils.bandwidth_limit_array(torch.from_numpy(in_)))
        cropped = torch.fft.ifft2(pyms.utils.crop_to_bandwidth_limit_torch(bwlimited))

        passtest = (
            passtest and sumsqr_diff(sumsqr(torch.fft.ifft2(bwlimited)), sumsqr(cropped)) < 1e-10
        )

        # pytorch test with complex numbers
        in_ = np.ones(oddgridshape, dtype=complex)
        bwlimited = pyms.utils.bandwidth_limit_array(torch.from_numpy(in_))
        cropped = torch.fft.ifft2(pyms.utils.crop_to_bandwidth_limit_torch(bwlimited))
        passtest = (
            passtest and sumsqr_diff(sumsqr(torch.fft.ifft2(bwlimited)), sumsqr(cropped)) < 1e-10
        )
        in_ = np.ones(evengridshape, dtype=complex)
        bwlimited = pyms.utils.bandwidth_limit_array(torch.from_numpy(in_))
        cropped = torch.fft.ifft2(pyms.utils.crop_to_bandwidth_limit_torch(bwlimited))
        passtest = (
            passtest and sumsqr_diff(sumsqr(torch.fft.ifft2(bwlimited)), sumsqr(cropped)) < 1e-10
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
        testarray = np.broadcast_to(testarray, [7] + grid)
        bandlimited = pyms.utils.bandwidth_limit_array(testarray, 2 / 3)
        passeven = np.all(bandlimited[3] - referencearray == 0)
        testtensor = torch.from_numpy(copy.deepcopy(testarray))
        bandlimited = pyms.utils.bandwidth_limit_array_torch(testtensor, 2 / 3)
        torchpasseven = np.all(bandlimited[3].cpu().numpy() - referencearray == 0)
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
        testarray = np.broadcast_to(testarray, [7] + grid)
        bandlimited = pyms.utils.bandwidth_limit_array(testarray, 2 / 3)
        passodd = np.all(bandlimited[3] - referencearray == 0)
        testtensor = torch.from_numpy(copy.deepcopy(testarray))
        bandlimited = pyms.utils.bandwidth_limit_array_torch(testtensor, 2 / 3)
        torchpassodd = np.all(bandlimited[3].cpu().numpy() - referencearray == 0)
        self.assertTrue(np.all([passeven, torchpasseven, passodd, torchpassodd]))

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
        oddgrid = [5, 5]
        evengrid = [4, 4]
        numpyVersionPass = True
        pytorchVersionPass = True
        for g in [evengrid, oddgrid]:
            a = np.zeros(g, dtype=pyms._float)
            device = pyms.utils.get_device()
            # Put in one full period of a cosine function
            a[:, :] = np.cos(2 * np.pi * np.fft.fftfreq(g[0]))
            a = a.T

            # In the fourier interpolated version we should recover the value of
            # cos(pi/2) = 1/sqrt(2) at the interpolated intermediate points
            passY = (
                np.abs(pyms.utils.fourier_interpolate(a, (8, 8))[1, 0] - 1 / np.sqrt(2))
                < 1e-8
            )

            # Test in x direction for completeness
            passX = (
                np.abs(
                    pyms.utils.fourier_interpolate(a.T, (8, 8)).T[1, 0] - 1 / np.sqrt(2)
                )
                < 1e-8
            )

            # Test that the option 'conserve_norm' works too.
            passNorm = (
                np.sum(
                    pyms.utils.fourier_interpolate(a, (8, 8), norm="conserve_L2") ** 2
                )
                - np.sum(a ** 2)
                < 1e-8
            )

            numpyVersionPass = numpyVersionPass and ((passY and passX) and passNorm)

            # test pytorch versions
            passY = (
                pyms.utils.fourier_interpolate_torch(
                    torch.from_numpy(a).to(device), (8, 8)
                )[1, 0]
                - 1 / np.sqrt(2)
                < 1e-8
            )

            passX = (
                pyms.utils.fourier_interpolate_torch(
                    torch.from_numpy(a.T).to(device), (8, 8)
                )[0, 1]
                - 1 / np.sqrt(2)
                < 1e-8
            )

            # Test that the option 'conserve_norm' works too.
            passNorm = (
                torch.sum(
                    pyms.utils.fourier_interpolate_torch(
                        torch.from_numpy(a.T).to(device), (8, 8), norm="conserve_norm"
                    ).abs()
                    ** 2
                )
                - torch.sum(torch.from_numpy(a).to(device) ** 2)
                < 1e-8
            )

            ptorchpass = passY.item() and passX.item() and passNorm.item()
            pytorchVersionPass = pytorchVersionPass and ptorchpass

        self.assertTrue(pytorchVersionPass and numpyVersionPass)

    def test_fft_shift(self):
        """Test to see if fourier shift correctly shifts a pixel 2 to the right."""
        test_array = np.zeros((5, 5))
        test_array[0, 0] = 1
        shifted1 = pyms.utils.fourier_shift(test_array, [2, 3])
        shifted2 = pyms.utils.fourier_shift_torch(torch.from_numpy(test_array), [2, 3])
        test_array[0, 0] = 0
        test_array[2, 3] = 1
        self.assertTrue(
            sumsqr_diff(shifted1, test_array) < 1e-10
            and sumsqr_diff(shifted2.cpu().numpy(), test_array) < 1e-10
        )

    def test_crop(self):
        """Test cropping method on scipy test image."""
        # Get astonaut image
        from skimage.data import astronaut

        im = np.sum(astronaut(), axis=2).astype(pyms._float)

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

    def test_torch_sinc(self):
        """Test sinc function by calculating [sinc(0),sinc(pi/2),sinc(pi)]."""
        x = torch.as_tensor([0, 1 / 2, 1, 3 / 2])
        y = torch.as_tensor([1.0, 2 / np.pi, 0, -2 / 3 / np.pi])
        self.assertTrue(sumsqr_diff(y, pyms.utils.sinc(x)) < 1e-7)

    def test_amplitude(self):
        """Test modulus square function by calculating a few known examples."""
        # 1 + i, 1-i and i
        x = torch.as_tensor([1 + 1j, 1 - 1j, 1j])
        y = torch.as_tensor([2, 2, 1])
        passcomplex = sumsqr_diff(y, pyms.utils.amplitude(x)).item() < 1e-7
        # 1,2,4 (test function for real numbers)
        x = torch.as_tensor([1, 2, 4])
        y = torch.as_tensor([1, 4, 16])
        passreal = sumsqr_diff(y, pyms.utils.amplitude(x)).item() < 1e-7
        self.assertTrue(passcomplex and passreal)

    # def test_fftfreq(self):
    #     """Test function for calculating Fourier frequency grid."""
    #     even = torch.as_tensor([0.0, 1.0, -2.0, -1.0])
    #     odd = torch.as_tensor([0.0, 1.0, 2.0, -2.0, -1.0])
    #     passeven = sumsqr_diff(pyms.utils.fftfreq(4), even) < 1e-8
    #     passodd = sumsqr_diff(pyms.utils.fftfreq(5), odd) < 1e-8
    #     self.assertTrue(passeven and passodd)


# def suite():
#     suite = unittest.TestSuite()
#     suite.addTest(Test_util_Methods())
#     return suite

# if __name__ == '__main__':
#     runner = unittest.TextTestRunner()
#     runner.run(suite())

if __name__ == "__main__":
    # test = Test_util_Methods()
    # test.test_crop_tobandwidthlimit()

    # test = Test_py_multislice_Methods()
    # test.test_DPC()
    # import sys; sys.exit()
    # test.test_nyquist_and_probe_raster()
    # test.test_DPC()

    # test.test_STEM_routine()
    # test.test_Smatrix()
    # import sys; sys.exit()
    # # test.test_propagator()
    # sys.exit()
    # test = Test_ionization_methods()
    # test.test_PRISM_STEM_EELS()
    # import sys;sys.exit()
    # suite = unittest.TestSuite(tests = (Test_util_Methods(),))
    # suite.run()
    # test.run()
    # test = Test_Structure_Methods()
    # test.test_atom_placement()

    # test.test_electron_scattering_factor()
    # test.test_make_potential()
    # import sys; sys.exit()
    # test.test_fourier_interpolation()
    # test = Test_py_multislice_Methods()
    # test.test_reverse_multislice()

    unittest.main()
    clean_temp_structure()
    import sys

    sys.exit()
