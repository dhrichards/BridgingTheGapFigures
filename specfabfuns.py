from specfabpy import specfab as sf
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import signal as sig



def solve(gradu,dt,x,L=8):

    nt = dt.shape[0]
    D = 0.5*(gradu+np.transpose(gradu,(0,2,1)))
    W = 0.5*(gradu-np.transpose(gradu,(0,2,1)))

    effSR = np.sqrt(0.5*np.einsum('pij,pji->p',D,D))
    iota = x[:,0]
    lamb = x[:,2]*effSR
    beta = x[:,3]*effSR

    lm, nlm_len = sf.init(L) 
    f = np.zeros((nt,nlm_len), dtype=np.complex64) # State vector (array of expansion coefficients)
    f[0,0] = 1.0 # Normalized ODF at t=0

    a2 = np.zeros((nt,3,3))
    a4 = np.zeros((nt,3,3,3,3))
    a2[0,...] = sf.a2(f[0,:])
    a4[0,...] = sf.a4(f[0,:])

    for tt in np.arange(1,nt):
        nlm_prev = f[tt-1,:] # Previous solution
        
        M_LROT = sf.M_LROT(nlm_prev, D[tt-1], W[tt-1], iota[tt-1], 0) # Lattice rotation operator (nlm_len x nlm_len matrix) 
        M_REG  = sf.M_REG(nlm_prev, D[tt-1])                 # Regularization operator   (nlm_len x nlm_len matrix)
        M      = M_LROT + M_REG + lamb[tt-1]*sf.M_CDRX(nlm_prev)
        M += beta[tt-1]*sf.M_DDRX(nlm_prev,D[tt-1])



        f[tt,:] = nlm_prev + dt[tt-1]*np.matmul(M, nlm_prev) # Euler step
        f[tt,:] = sf.apply_bounds(f[tt,:]) # Apply spectral bounds if needed
        a2[tt,...] = sf.a2(f[tt,:]) # Compute a2 tensor from ODF

    return a2,a4,f




def solve_constant(gradu,dt,tmax,x,L=8):

    D = 0.5*(gradu+gradu.T)
    W = 0.5*(gradu-gradu.T)

    effSR = np.sqrt(0.5*np.einsum('ij,ji',D,D))
    iota = x[0]
    lamb = x[1]
    beta = x[2]


    t = np.arange(0,tmax,dt)
    nt = len(t)
    lm, nlm_len = sf.init(L) 
    f = np.zeros((nt,nlm_len), dtype=np.complex64) # State vector (array of expansion coefficients)
    f[0,0] = 1/np.sqrt(4*np.pi) # Normalized ODF at t=0

    a2 = np.zeros((nt,3,3))
    a2[0,...] = sf.a2(f[0,:])

    for tt in np.arange(1,nt):
        nlm_prev = f[tt-1,:] # Previous solution
        
        M_LROT = sf.M_LROT(nlm_prev, D, W, iota, 0) # Lattice rotation operator (nlm_len x nlm_len matrix) 
        M_REG  = sf.M_REG(nlm_prev, D)                 # Regularization operator   (nlm_len x nlm_len matrix)
        M      = M_LROT + M_REG + x[1]*sf.M_CDRX(nlm_prev)
        M += x[2]*sf.M_DDRX(nlm_prev,D)



        f[tt,:] = nlm_prev + dt*np.matmul(M, nlm_prev) # Euler step
        f[tt,:] = sf.apply_bounds(f[tt,:]) # Apply spectral bounds if needed
        a2[tt,...] = sf.a2(f[tt,:]) # Compute a2 tensor from ODF

    return t,a2,f,lm




class Plotting:
    def __init__(self,L,f):
        self.f = f
        self.L = L
        self.lm, self.nlm_len = sf.init(self.L)

    def J(self):
        m = self.lm[1,:]

        # get indices where m>=0
        idx = np.where(m>=0)
        return np.sum(np.abs(self.f[idx])**2)

    def plot(self,fig,ax,hemisphere=False,colorbar=True,vmax=None,**kwargs):
        
        nlat = 100
        nlon = 100
        theta_vals = np.linspace(0,np.pi,nlat)
        phi_vals = np.linspace(0,2*np.pi,nlon)

        #make phi contiguous so it includes 2pi
        phi_vals = np.append(phi_vals,2*np.pi)
        phi,theta = np.meshgrid(phi_vals,theta_vals)




        ra,dec = decra_from_polar(phi_vals,theta_vals)


        X, Y = np.meshgrid(ra, dec)
        

        fgrid = self.synth(theta,phi)
                            

        if hemisphere:
            

            fgrid = fgrid[0:nlat//2,:]
            X = X[0:nlat//2,:]
            Y = Y[0:nlat//2,:]

        F = fgrid
        pcol = ax.pcolormesh(X,Y,F,transform=ccrs.PlateCarree(),vmin=0,vmax=vmax)
        pcol.set_edgecolor('face')
        ax.set_aspect('equal')
        ax.axis('off')
        kwargs_gridlines = {'ylocs':np.arange(-90,90+30,30), \
                            'xlocs':np.arange(-360,+360,45),\
                                'linewidth':0.5, 'color':'black', 'alpha':0.25, \
                                    'linestyle':'-'}
        
        gl = ax.gridlines(crs=ccrs.PlateCarree(),**kwargs_gridlines)#,xlocs=[s_dir,s_dir+90,s_dir+180,s_dir+270])


        if hemisphere:
            gl.ylim = (0,90)

        geo = ccrs.RotatedPole()

        # colorbar for this axes -show max and min
        if colorbar:
            cb = fig.colorbar(pcol,ax=ax,orientation='horizontal',**kwargs)
            cb.set_label('ODF')
            vm = np.max(pcol.get_clim()[1])
            cb.set_ticks([0,vm/2,vm])
            # set sig fig in colorbar
            from matplotlib.ticker import FormatStrFormatter
            cb.ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        return pcol


    def synth(self,theta,phi):
        # synthesize spherical harmonics from spectral array
        F = np.real(np.sum([ self.f[ii]*sp.sph_harm(self.lm[1][ii], self.lm[0][ii], phi,theta) for ii in np.arange(self.nlm_len) ], axis=0))
        F[F<0] = 0 # hide numerical/truncation errors
        
        return F



def y0(nlm,lm):
    theta = np.linspace(0,   np.pi,   5000) # co-lat
    #y=0 so phi = pi/2
    phi = 0
    _,nlm_len = lm.shape
    F = np.zeros((nlm.shape[0],theta.shape[0]))
    for tt in range(nlm.shape[0]):
        F[tt,:] = np.real(np.sum([ nlm[tt,ii]*sp.sph_harm(lm[1][ii],\
                                                           lm[0][ii], phi,theta)\
                                                              for ii in np.arange(nlm_len) ],\
                                                                  axis=0))
        
    F[F<0] = 0 # hide numerical/truncation errors

    return theta,F



def init_fig():
    fig = plt.figure(figsize=(3,4))
    inclination, rot = 45, +135 # view angle
    prj, geo = ccrs.Orthographic(rot, 90-inclination), ccrs.Geodetic()
    ax = plt.subplot(projection=prj)
    ax.set_global() # show entire S^2
    return fig, ax, geo


def angle_between_peaks(theta,F):

    angle_between = np.zeros(F.shape[0])
    strength_ratio = np.zeros(F.shape[0])
    for tt in range(F.shape[0]):
        peaks, _ = sig.find_peaks(F[tt,:])
        if len(peaks) < 2:
            angle_between[tt] = np.nan
            strength_ratio[tt] = np.nan
        else:
            angle_between[tt] = theta[peaks[1]]-theta[peaks[0]]
            strength_ratio[tt] = F[tt,peaks[1]]/F[tt,peaks[0]]

    return angle_between,strength_ratio




def decra_from_polar(phi, theta):
    """ Convert from ra and dec to spherical polar coordinates.
    Parameters
    ----------
    phi, theta : float or numpy.array
        azimuthal and polar angle in radians
    Returns
    -------
    ra, dec : float or numpy.array
        Right ascension and declination in degrees.
    """
    ra = phi * (phi < np.pi) + (phi-2*np.pi)*(phi > np.pi)
    dec = np.pi/2-theta
    return ra/np.pi*180, dec/np.pi*180

