import numpy as np
import torch
from .py_multislice import make_propagators,scattering_matrix,generate_STEM_raster
from .utils.torch_utils import cx_from_numpy,cx_to_numpy,amplitude,complex_matmul,complex_mul
from .utils.numpy_utils import fourier_shift

def window_indices(center,windowsize,gridshape):
    """Makes indices for a cropped window centered at center with size given
    by windowsize on a grid"""
    window = []
    for i,wind in enumerate(windowsize): 
        indices = np.arange(-wind//2,wind//2,dtype=np.int) + wind%2
        indices += int(round(center[i]*gridshape[i]))
        indices = np.mod(indices,gridshape[i])
        window.append(indices)
    
    return (window[0][:,None]*gridshape[0]+window[1][None,:]).ravel()


def STEM_EELS_PRISM(crystal,gridshape,eV,app,det,thickness,Ztarget,n,l,ml,lprime,mlprime,epsilon,subslices=[1.0],device_type=None,tiling=[1,1]):

    #Choose GPU if available and CPU if not
    # Initialize device cuda if available, CPU if no cuda is available
    device = get_device(device_type)

    rsize = np.zeros(3)
    rsize[:3] = crystal.unitcell[:3]
    rsize[:2] *= np.asarray(tiling)


    nslices = int(np.ceil(thickness/crystal.unitcell[2]))
    P = pyms.make_propagators(gridshape,rsize,eV,subslices)
    nT = 5
    T = torch.zeros(nT,len(subslices),*gridshape,2,device=device)

    for i in range(nT):
        T[i] = crystal.make_transmission_functions(gridshape,eV,subslices,tiling)

    #Make transition potentials
    Ztarget = 8
    

    natoms = np.sum(crystal.atoms[:,3]==Ztarget)
    coords = np.zeros((natoms*np.product(tiling),3))
    coords[:natoms] = crystal.atoms[crystal.atoms[:,3]==Ztarget][:,:3]/np.asarray(tiling+[1])

    #Tile coordinates
    for i in range(np.product(tiling)):
        coords[i*natoms:(i+1)*natoms] = coords[:natoms]+np.asarray([(i%tiling[0])/tiling[0],(i//tiling[0])/tiling[1],0])[None,:]

    # Scattering matrix 1 propagates probe from surface of specimen to slice of
    # interest
    S1 = scattering_matrix(rsize,P,T,0,eV,app,batch_size=5,subslicing=True)
    
    # Scattering matrix 2 propagates probe from slice of interest to exit surface
    S2 = scattering_matrix(rsize,P,T,nslices*len(subslices),eV,app,batch_size=5,subslicing=True,transposed=True)
    #Link the slices and seeds of both scattering matrices
    S1.seed = S2.seed

    from .Ionization import orbital, transition_potential
    Hn0_crop = [min(Hn0_crop[i],S1.stored_gridshape[i]) for i in range(2)]
    boundOrbital = orbital(Ztarget,'1s2 2s2 2p6',n,l)
    freeOrbital = orbital(Ztarget,'1s1 2s2 2p6',0,lprime,epsilon=epsilon)
    Hn0 = transition_potential(boundOrbital, freeOrbital, Hn0_crop, [Hn0_crop[i]/gridshape[i]*rsize[i] for i in range(2)], ml, mlprime, eV, bandwidth_limiting=None,qspace=True)

    #Make probe wavefunction vectors for scan
    #Get kspace grid in units of inverse pixels
    ky,kx = [np.fft.fftfreq(gridshape[-2+i], d=1 / gridshape[-2+i]) for i in range(2)]

    #Generate scan positions in pixels
    scan = generate_STEM_raster(S1.S.shape[-2:],rsize[:2], eV, app)
    nprobe_posn = len(scan[0])*len(scan[1])

    scan_array = np.zeros((nprobe_posn,S1.S.shape[0]),dtype=np.complex)
    #TODO implement aberrations
    for i in range(S1.nbeams):
        scan_array[:,i] = (np.exp(-2*np.pi*1j*ky[S1.beams[i, 0]]*scan[0])[:,None]
                          *np.exp(-2*np.pi*1j*kx[S1.beams[i, 1]]*scan[1])[None,:]).ravel()

    scan_array = cx_from_numpy(scan_array,dtype = S1.dtype,device=device)

    #Initialize Image
    EELS_image = torch.zeros(len(scan[0])*len(scan[1]),dtype=S1.dtype,device=device)
    
    for islice in tqdm.tqdm(range(nslices*len(subslices)),desc='Slice'):
        
        #Propagate scattering matrices to this slice
        if islice>0:
            S1.Propagate(islice,P,T,subslicing=True,batch_size=5,showProgress=False)
            S2.Propagate(nslices*len(subslices)-islice,P,T,subslicing=True,batch_size=5,showProgress=False)

        S2.S = S2.S.reshape(S2.S.shape[0],np.product(S2.stored_gridshape),2)
        subslice= islice % S1.nsubslices
        atomsinslice = coords[np.logical_and(coords[:,2]>=subslice/S1.nsubslices,coords[:,2]<(subslice+1)/S1.nsubslices),:2]
        
        for atom in tqdm.tqdm(atomsinslice,'Transitions in slice'):
            
            windex = torch.from_numpy(window_indices(atom,Hn0_crop,S1.stored_gridshape))
            
            #Initialize matrix describing this transition event
            SHn0 = torch.zeros(S1.S.shape[0],S2.S.shape[0],2,dtype = S1.S.dtype,device=device)

            #Sub-pixel shift of Hn0
            Hn0_ = np.fft.fftshift(fourier_shift(Hn0,np.remainder(atom*np.asarray(gridshape),1.0),qspacein=True)).ravel()

            #Convert Hn0 to pytorchtensor
            Hn0_ = cx_from_numpy(Hn0_,dtype = S1.S.dtype,device=device)


            for i,S1component in enumerate(S1.S):
                Hn0S1 = pyms.utils.complex_mul(Hn0_,S1component.flatten(end_dim=-2)[windex,:])
                SHn0[i] = complex_matmul(S2.S[:,windex],Hn0S1)


            EELS_image += torch.sum(amplitude(complex_matmul(scan_array,SHn0)),axis=1)

        
        #Reshape scattering matrix S2 for propagation
        S2.S = S2.S.reshape((S2.S.shape[0],*S2.stored_gridshape,2))
    return EELS_image.cpu().numpy().reshape(len(scan[0]),len(scan[1]))