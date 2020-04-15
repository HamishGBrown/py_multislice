import unittest
import pyms
import numpy as np

required_accuracy = 1e-4

class TestStructureMethods(unittest.TestCase):

    def test_remove_common_factors(self):
        self.assertTrue([x == y for x,y in zip(pyms.remove_common_factors([3,9,15]),[1,3,5])])

    def test_psuedo_rational_tiling(self):
        self.assertTrue([x == y for x,y in zip(pyms.psuedo_rational_tiling(3.905,15.6,1e-1),np.array([4,1]))])

    def test_electron_scattering_factor(self):
        """
        Test that the electron scattering factor function can replicate the
        results for Ag in Doyle, P. A. & Turner, P. S. (1968). Acta Cryst. A24, 
        390–397
        """
        #Reciprocal space points
        g = 2*np.concatenate([np.arange(0.00,0.51,0.05),np.arange(0.60,1.01,0.1),np.arange(1.20,2.01,0.2),[2.5,3.0,3.5,4.0,5.0,6.0]])
        # Values from Doyle and Turner
        fe = np.asarray([8.671, 8.244, 7.267, 6.215, 5.293, 4.522, 3.878, 3.339, 2.886, 2.505, 2.185, 1.688, 1.335, 1.082, 0.897, 0.758, 0.568, 0.444, 0.357, 0.291, 0.241, 0.159, 0.113, 0.084, 0.066, 0.043, 0.030])
        sse = np.sum(np.square(fe-pyms.electron_scattering_factor(47,g**2,units='A')))
        self.assertTrue(sse<0.01)
        
    def test_calculate_scattering_factors(self):
        calc = pyms.calculate_scattering_factors([3,3], [1,1], [49])
        known = np.asarray([[[499.59366 , 107.720055, 107.720055],
                             [107.720055,  65.67732 ,  65.67732 ],
                             [107.720055,  65.67732 ,  65.67732 ]]], dtype=np.float32)
        self.assertTrue(np.sum(np.square(known-calc))<0.01)

if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    # g = 2*np.concatenate([np.arange(0.00,0.51,0.05),np.arange(0.60,1.01,0.1),np.arange(1.20,2.01,0.2),[2.5,3.0,3.5,4.0,5.0,6.0]])
    # # print(s)
    # fe = np.asarray([8.671, 8.244, 7.267, 6.215, 5.293, 4.522, 3.878, 3.339, 2.886, 2.505, 2.185, 1.688, 1.335, 1.082, 0.897, 0.758, 0.568, 0.444, 0.357, 0.291, 0.241, 0.159, 0.113, 0.084, 0.066, 0.043, 0.030])
    # fig,ax = plt.subplots()
    # ax.set_xlim([0,12])
    # ax.set_ylim([0,9])
    # ax.plot(g,pyms.electron_scattering_factor(47,g**2,units='A'),'k-')
    # ax.plot(g,fe,'rx')
    # print(np.sum(np.square(fe-pyms.electron_scattering_factor(47,g**2,units='A'))))
    # fig.savefig('Lobato_replication.pdf')
    # plt.show(block=True)
    unittest.main()