import numpy as np

class aberration:


    def __init__(self,Haider,Krivanek,Description,
                 amplitude,angle,n,m):
        self.Haider = Haider
        self.Krivanek = Krivanek
        self.Description = Description
        self.amplitude = amplitude
        self.m = m
        self.n = n
        if(m>0):
            self.angle = angle
        else:
            self.angle = 0

def nyquist_sampling(size,resolution_limit= None,eV = None,alpha = None):
    """For resolution limit in units of inverse length and array size in
    units of length calculate how many probe positions are required for
    nyquist sampling. Alternatively pass probe accelerating voltage and
    probe forming aperture and the resolution limit in inverse length
    will be calculated for you."""
    if eV is None and alpha is None: return np.ceil(4*np.asarray(size)*resolution_limit).astype(np.int)
    elif resolution_limit is None: return np.ceil(4*np.asarray(size)*wavev(eV)*alpha).astype(np.int)
    else: return None

def aberration_starter_pack():
    """Creates the set of aberrations up to fifth order"""
    aberrations = []
    aberrations.append(aberration('C10'      ,'C1','Defocus          ',0.0,0.0,1,0))
    aberrations.append(aberration('C12'      ,'A1','2-Fold astig.    ',0.0,0.0,1,2))
    aberrations.append(aberration('C21'      ,'B2','Axial coma       ',0.0,0.0,2,1))
    aberrations.append(aberration('C23'      ,'A2','3-Fold astig.    ',0.0,0.0,2,3))
    aberrations.append(aberration('C30 = CS' ,'C3','3rd order spher. ',0.0,0.0,3,0))
    aberrations.append(aberration('C32'      ,'S3','Axial star aber. ',0.0,0.0,3,2))
    aberrations.append(aberration('C34'      ,'A3','4-Fold astig.    ',0.0,0.0,3,4))
    aberrations.append(aberration('C41'      ,'B4','4th order coma   ',0.0,0.0,4,1))
    aberrations.append(aberration('C43'      ,'D4','3-Lobe aberr.    ',0.0,0.0,4,3))
    aberrations.append(aberration('C45'      ,'A4','5-Fold astig     ',0.0,0.0,4,5))
    aberrations.append(aberration('C50 = CS5','C5','5th order spher. ',0.0,0.0,5,0))
    aberrations.append(aberration('C52'      ,'S5','5th order star   ',0.0,0.0,5,2))
    aberrations.append(aberration('C54'      ,'R5','5th order rosette',0.0,0.0,5,4))
    aberrations.append(aberration('C56'      ,'A5','6-Fold astig.    ',0.0,0.0,5,6))
    return aberrations


def q_space_array(pixels,gridsize):
    '''Returns the appropriately scaled 2D reciprocal space array for pixel size
    given by pixels (#y pixels, #x pixels) and real space size given by gridsize
    (y size, x size)'''
    return np.meshgrid(*[np.fft.fftfreq(pixels[i])*pixels[i]/gridsize[i] for i
                        in range(1,-1,-1)])[::-1]

def chi(q,qphi,lam,df=0.0,aberrations = []):
    '''calculates the aberration function chi as a function of 
    reciprocal space extent q for an electron with wavelength lam.

    Parameters
    ----------
    q : number
        reciprocal space extent (Inverse angstroms).
    lam : number
        wavelength of electron (Inverse angstroms).
    aberrations : list
        A list object containing a set of the class aberration'''
    chi_ = (q*lam)**2/2*df/2
    for ab in aberrations: chi_ += (q*lam)**(ab.n+1)*float(ab.amplitude.get())/(ab.n+1)*np.cos(ab.m*(qphi-float(ab.angle.get())))
    return 2*np.pi*chi_/lam

def construct_illum(pix_dim,real_dim,eV,app,beam_tilt=[0,0],aperture_shift=[0,0],
    df=0,aberrations=[],q=None,app_units = 'mrad',qspace = False):
    '''Makes a probe wave function with pixel dimensions given in pix_dim
    and real_dimensions given by real_dim
    ---------
    pix_dim --- The pixel size of the grid
    real_dim --- The size of the grid in Angstrom
    keV --- The energy of the probe electrons in keVnews
    app --- The apperture in units specified by app_units
    df --- Probe defocus in A, a negative value indicate overfocus
    cs --- The 3rd order spherical aberration coefficient
    c5 --- The 5rd order spherical aberration coefficient
    app_units --- The units of the aperture size (A^-1 or mrad)
    '''
    if(q==None): q = q_space_array(pix_dim,real_dim[:2])

    k = wavev(eV)

    if app_units == 'mrad': app_ = np.tan(app/1000.0)*k
    else: app_= app

    probe = np.zeros(pix_dim,dtype=np.complex)

    qarray1 = np.sqrt(np.square(q[0]-beam_tilt[0])+np.square(q[1]-beam_tilt[1]))
    qarray2 = np.sqrt(np.square(q[0]-beam_tilt[0]-aperture_shift[0])
                     +np.square(q[1]-beam_tilt[1]-aperture_shift[1]))
    qarray = np.sqrt(np.square(q[0])+np.square(q[1]))
    qphi = np.arctan2(q[0],q[1])
    mask = qarray2<app_
    probe[mask] = np.exp(-1j*chi(qarray1[mask],qphi[mask],
                                   1.0/k,df,aberrations))
    probe /= np.sqrt(np.sum(np.square(np.abs(probe))))

    #Return real or diffraction space probe depending on user preference
    if(qspace): return probe
    else:   return np.fft.ifft2(probe,norm='ortho')

def wavev(E):
    """Calculates the relativistically corrected wavenumber k0 (reciprocal of
    the wavelength) for an electron of energy eV. See Eq. (2.5) in Kirkland's
    Advanced Computing in electron microscopy"""
    #Planck's constant times speed of light in eV Angstrom
    hc = 1.23984193e4
    #Electron rest mass in eV
    m0c2 = 5.109989461e5
    return np.sqrt( E*(E+ 2*m0c2 )) / hc
