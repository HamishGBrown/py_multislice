import cupy  as cp
import numpy as np

#TODO: The cupy multislice function allows smooth switching between GPU
#and cpu libraries by using if gpu: import cupy as cp else import numpy as cp
#an equivalent is needed for this whole library

def oned_shift(len,shift):
    '''Constructs a one dimensional shift array of array size 
    len that shifts an array number of pixels given by shift.'''
    #Check if the pixel length is even or not
    even = len%2 == 0
    
    #Create the Fourier space pixel coordinates of the shift 
    #array
    if (even):
        shiftarray = cp.empty((len))
        shiftarray[:len/2] = cp.arange(len/2)
        shiftarray[len/2:] = cp.arange(-len/2,0)
    else:
        shiftarray = cp.arange(-len/2+1,len/2+1)
    
    #The shift array is given mathematically as e^(-2pi i k 
    #Delta x) and this is what is returned.
    return cp.exp(-2*cp.pi*1j*shiftarray*shift)
    
def fourier_shift_array(arrayin, shift,qspace = False):
    '''Shifts a 2d array by an amount given in the tuple shift 
    using the Fourier shift theorem.'''
    
    #Get shift amounts and array dimensions from input
    shifty,shiftx = shift
    y,x = np.shape(arrayin)
    
    #Construct shift array
    shifty = oned_shift(y,shifty)
    shiftx = oned_shift(x,shiftx)
    shiftarray = shiftx[cp.newaxis,:]*shifty[:,cp.newaxis]

    #Now Fourier transform array and apply shift
    real = array_real(arrayin)
    if real: array = np.asarray(arrayin,dtype= np.complex)
    else: array = arrayin
    
    if qspace: array = ifft2(shiftarray*array)
    else: array = ifft2(shiftarray*fft2(array))
    
    if real: return np.real(array)
    else: return array