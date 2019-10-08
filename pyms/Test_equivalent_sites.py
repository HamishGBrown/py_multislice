import pyms
import numpy as np

sites = np.random.randn(10,2)

sites[6,:] = sites[2,:]

sites[9,:] = sites[4,:]

print('output should be 0,1,2,3,4,5,2,7,8,4,10')
print(pyms.equivalent_sites(sites))