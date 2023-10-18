
import numpy as np

from scipy.signal import savgol_filter
import agedepth
from scipy.optimize import minimize_scalar
import densityfromdepth
import divide_data
import specfabfuns as sff
import parameters as prm
class path2d:
    def __init__(self,time,dt,xc,yc,data):
        
        self.t = np.arange(0,time,dt)
        self.nt = self.t.size
        self.gradu = np.zeros((self.nt,3,3))
        self.xp=np.zeros(self.nt)
        self.yp = np.zeros(self.nt)
        self.t=np.zeros(self.nt)
        self.s=np.zeros(self.nt)
        
        self.xp[-1]=xc
        self.yp[-1]=yc
        self.dt=dt

        
        
            
        
        for i in range(self.nt-1,0,-1):
            data.interps(self.xp[i],self.yp[i])

            
            self.gradu[i,0,0] = data.dudx
            self.gradu[i,0,1] = data.dudy
            self.gradu[i,1,0] = data.dvdx
            self.gradu[i,1,1] = data.dvdy            
            #umag = np.sqrt(up**2 + vp**2)
            #dt = k/umag
            
            if i==self.nt-1:
                self.xp[i-1] = self.xp[i] - data.vx*dt
                self.yp[i-1] = self.yp[i] - data.vy*dt
            else:
                self.xp[i-1] = self.xp[i] - 1.5*data.vx*dt + 0.5*vxp*dt
                self.yp[i-1] = self.yp[i] - 1.5*data.vy*dt + 0.5*vyp*dt
            
            self.t[i-1] = self.t[i] - dt
            ds = np.sqrt((self.xp[i-1]-self.xp[i])**2 + (self.yp[i-1]-self.yp[i])**2)
            self.s[i-1] = self.s[i] - ds

            #Store velocity for next timestep
            vxp = np.copy(data.vx)
            vyp = np.copy(data.vy)
            
        data.interps(self.xp[0],self.yp[0])   
        self.gradu[0,0,0] = data.dudx
        self.gradu[0,0,1] = data.dudy
        self.gradu[0,1,0] = data.dvdx
        self.gradu[0,1,1] = data.dvdy


        # Get physical data along path 
        data.interps(self.xp,self.yp)
        self.acc_gl = data.accumulation
        self.bed = data.bed
        self.surf = data.surface
        self.vx = data.vx
        self.vy = data.vy
        

        self.dbeddx = data.dbeddx
        self.dbeddy = data.dbeddy
        self.dsurfdx = data.dsurfdx
        self.dsurfdy = data.dsurfdy
        
        import gerber
        gerb = gerber.gerb()
        gerb.interps(self.s)
        gerb.interps_xy(self.xp,self.yp)
        self.f = np.nan_to_num(gerb.f,0)
        self.acc_gb = gerb.acc_s
        self.basalmelt = gerb.basalmelt
    

class path3d:
    def __init__(self,path2d,nt):
        self.nt = nt
        self.t = path2d.t[-nt:,...]
        self.dt = path2d.dt

        self.gradu = path2d.gradu[-nt:,...]
        #dw/dz from div u = 0 - inital guess not accounting for density change in firn
        self.gradu[:,2,2] = - self.gradu[:,0,0] - self.gradu[:,1,1]

        self.xp = path2d.xp[-nt:,...]
        self.yp = path2d.yp[-nt:,...]
        self.s = path2d.s[-nt:,...]


        surf_smooth = savgol_filter(path2d.surf,211,3)
        surfslope = np.gradient(surf_smooth,path2d.s)
        self.surfslope = surfslope[-nt:,...]
        self.meanslope = np.mean(surfslope)

        self.vx_s = path2d.vx[-nt:,...] # note these a surface velocities
        self.vy_s = path2d.vy[-nt:,...]

        self.bed = path2d.bed[-nt:,...]
        #self.acc = path2d.acc_gl[-nt:,...]
        self.surf = path2d.surf[-nt:,...]
        self.dbeddx = path2d.dbeddx[-nt:,...]
        self.dbeddy = path2d.dbeddy[-nt:,...]
        self.dsurfdx = path2d.dsurfdx[-nt:,...]
        self.dsurfdy = path2d.dsurfdy[-nt:,...]
        #self.basalmelt = path2d.basalmelt[-nt:,...]


    def optimizeacc(self): #optimize accumulation rate to match depth age relationship from gerber
        res = minimize_scalar(self.deptherror,method='bounded',bounds=(0,1000))
        self.acc = res.x
        self.depth(self.acc)



    def deptherror(self,acc): #error function for optimization
        self.depth(acc)
        depth = self.d[-1]

        error = np.sqrt((depth - agedepth.time2depth(-self.t[0]))**2)

        return error
    

    def depth(self,acc=0.1): #calculate depth of path using accumulation rate

                
        self.z=np.zeros(self.nt)
        self.vz=np.zeros(self.nt)
        self.d=np.zeros(self.nt)
        self.density=np.zeros(self.nt)
        

        

        vs = np.sqrt(self.vx_s**2 + self.vy_s**2)

        # Averaging to remove noise which can cause crossing streamlines due to
        # fluctuations in vertical velocity


        # assign initial values to vertical velocity and depth
        self.vz[0]= -acc + vs[0]*self.surfslope[-1]
        self.z[0]=self.surf[0]
        self.d[0] = 0

        self.density[0] = densityfromdepth.densityfromdepth(self.d[0])

        for i in range(self.nt-1):


            self.z[i+1] = self.z[i] + self.vz[i]*self.dt
            self.d[i+1] = self.surf[i+1]-self.z[i+1]
            self.density[i+1] = densityfromdepth.densityfromdepth(self.d[i+1])
            drhodt = (self.density[i+1]-self.density[i])/self.dt
            
            if i==0: # forward difference drhodt for first timestep
                self.gradu[i,2,2] = -self.gradu[i,0,0] - self.gradu[i,1,1]\
                        - drhodt/self.density[i]
                

            self.gradu[i+1,2,2] = -self.gradu[i+1,0,0] - self.gradu[i+1,1,1]\
                      - drhodt/self.density[i+1]

            
            self.vz[i+1] = self.vz[i]/(1-self.gradu[i,2,2]*self.dt) # check thi

            # dx = self.xp[i+1]-self.xp[i]
            # dy = self.yp[i+1]-self.yp[i]
            # self.vz[i+1] = ( -(self.vx_s[i]*self.vz[i]/dx) \
            #                 - (self.vy_s[i]*self.vz[i]/dy) \
            #                 + self.gradu[i,2,2]*self.vz[i] )/\
            #                     (1/self.dt - self.vx_s[i]/dx - self.vy_s[i]/dy)
        
        
            
            

            
        
        self.ztilde = (self.z-self.bed)/(self.surf-self.bed)       
        self.dtilde = 1-self.ztilde


    def Temperature(self):
        import egrip_temperature as et
        self.T = np.interp(self.d,et.depth,et.T)



    # def fabric(self,L=6,eps=1/3,x=None): #calculate fabric

    #     self.sh = spherical.spherical(L)


    #     # Calculate fabric tensor
    #     self.a2=np.zeros((self.nt,3,3))
    #     self.a4=np.zeros((self.nt,3,3,3,3))


        
    #     self.f0 = self.sh.fabricfromdiagonala2(eps)

    #     self.f=np.zeros((self.nt,self.f0.size),dtype='complex128')
    #     self.f[0,:]=self.f0
    #     self.a2[0,...] = self.sh.a2(self.f[0,:])
    #     self.a4[0,...] = self.sh.a4(self.f[0,:])

    #     for i in range(self.nt-1):
        
        
    #         #Update fabric with dt T[i] gradu[i]
    #         self.rk = solver.rk3iterate(self.T[i], self.gradu[i,...], self.sh,x=x)
    #         self.f[i+1,:] = self.rk.iterate(self.f[i,:], self.dt)

    #         # Update orientation tensors
    #         self.a2[i+1,...] = self.sh.a2(self.f[i+1,:].real)
    #         self.a4[i+1,...] = self.sh.a4(self.f[i+1,:].real)


    # def fabric_shtns(self,L=20,x=None):

    #     self.sc = solver_shtns(lmax=L,mmax=6)
        
    #     self.f0 = self.sc.f0

    #     self.f=np.zeros((self.nt,self.f0.size),dtype='complex128')
    #     self.f[0,:]=self.f0

    #     for i in range(self.nt-1):
    #         self.f[i+1,:] = self.sc.iterate(self.f[i,:], self.T[i], self.gradu[i,...], self.dt,x=x)
        
    #     self.a2 = self.sc.a2(self.f).real
    #     self.a4 = self.sc.a4(self.f).real

    # def fabric_mc(self,npoints,x=None):

    #     if x is None:
    #         x_tile = mc.parameters.Richards2021(self.T)
    #     elif x=='Reduced':
    #         x_tile = mc.parameters.Richards2021Reduced(self.T)
    #     else:
    #         x_tile = np.tile(x,(self.gradu.shape[0],1))

    #     dt_tile = np.tile(self.dt,(self.gradu.shape[0],))

    #     self.n,self.m,self.a2,self.a4 = mc.solve(npoints,self.gradu,dt_tile,x_tile)

    def fabric_sf(self,L=12,x=None):

        if x is None:
            x_tile = prm.Richards2021(self.T)
        elif x=='Reduced':
            x_tile = prm.Richards2021Reduced(self.T)
        else:
            x_tile = np.tile(x,(self.gradu.shape[0],1))

        dt_tile = np.tile(self.dt,(self.gradu.shape[0],))

        self.a2,self.a4,self.f = sff.solve(self.gradu,dt_tile,x_tile,L)



    

class divide:
    def __init__(self,dh=0.8,dt=1,location='GRIP',model='DJ',include_shear=False):

        self.dt = dt
        nt_max = 100000
        self.nt = nt_max
        self.dh = dh

        self.location = location
        self.include_shear = include_shear

        self.model = model

        self.t = np.arange(0,nt_max*dt,dt)

        self.gradu = np.zeros((nt_max,3,3))
        self.vz = np.zeros(nt_max)
        
        self.z = np.zeros(nt_max)
        self.d = np.zeros(nt_max)
        self.density = np.zeros(nt_max)

        self.data = divide_data.data(location)
        self.H = self.data.H
        self.acc = self.data.acc

        if include_shear:
            self.surface = query_point(self.data.xc,self.data.yc)

        


        

        
    def D_zz(self,depth):

        if self.model=='DJ':
            d_zz = self.Dansgard_Johnson(depth)
        elif self.model == 'Nye':
            d_zz = self.Nye(depth)

        return d_zz


    def Dansgard_Johnson(self,depth):
        #d_switch = 1750 #m, depth above which D_zz is constant, copying Castelnau (1996)
        d_switch = 0.66666*self.H
        # Change above
        # Integration of dansgard johnson profile to transition from vz_0 to 0 at base
        # vz_0 = -acc = int dw/dz dz = dzz_0*d_switch + 1/2*dzz_0*(H-d_switch)**2
        D_zz_0 = -self.acc/(d_switch + 0.5*(self.H-d_switch))

        depths = np.array([0,d_switch,self.H])
        d_zzs = np.array([D_zz_0,D_zz_0,0])

        return np.interp(depth,depths,d_zzs)

    def Nye(self,depth):
        d_zz = -self.acc/self.H
        return d_zz*np.ones_like(depth)

    def Temperature(self,d):
        return self.data.Temperature(d)


        
    def depth(self):

        self.z[0] = self.H
        self.vz[0] = -self.acc
        self.strain = np.zeros(self.nt)

        self.gradu[0,2,2] = self.D_zz(self.d[0])
        if self.include_shear:
            k = -self.gradu[0,2,2]/(self.surface.gradu[0,0] + self.surface.gradu[1,1])
            self.gradu[0,0:2,0:2] = k*self.surface.gradu[0:2,0:2]
        else:
            k = 0.5
            self.gradu[0,0,0] = self.gradu[0,1,1] = -self.gradu[0,2,2]/2
        
        i=0
        while self.d[i] < self.dh*self.H:
            self.z[i+1] = self.z[i] + self.vz[i]*self.dt
            self.d[i+1] = self.H-self.z[i+1]
            self.strain[i+1] = self.strain[i] + self.gradu[i,2,2]*self.dt
            
            self.gradu[i+1,2,2] = self.D_zz(self.d[i+1])
            self.gradu[i+1,0:2,0:2] = self.gradu[0,0:2,0:2]*self.gradu[i+1,2,2]/self.gradu[0,2,2]
            
            
            #self.vz[i+1] = self.vz[i]/(1-self.gradu[i,2,2]*self.dt) # check this
            self.vz[i+1] = self.vz[i] + self.gradu[i,2,2]*self.dt*self.vz[i] # check this
            i+=1

        self.nt = i
        self.z = self.z[:self.nt]
        self.d = self.d[:self.nt]
        self.vz = self.vz[:self.nt]
        self.strain = self.strain[:self.nt]
        self.gradu = self.gradu[:self.nt,...]
        self.t = self.t[:self.nt]
        self.dtilde = self.d/self.H

        self.T = self.data.Temperature(self.d)



# def fabric(path,npoints=5000,x='Richards',sr_type='SR',model='R1'):

#     if isinstance(x,str):
#         if x == 'Richards':
#             x_tile = mc.parameters.Richards2021(path.T)
#         elif x == 'Elmer':
#             x_tile = mc.parameters.Elmer(path.T)
#         elif x=='Reduced':
#             x_tile = mc.parameters.Richards2021Reduced(path.T)

            
#     else:    
#         x_tile = np.tile(x,(path.gradu.shape[0],1))

    

#     dt_tile = np.tile(path.dt,(path.gradu.shape[0],))



#     n,m,a2,a4 = mc.solve(npoints,path.gradu,dt_tile,x_tile,model,sr_type)


#     return a2,a4,n,m


def fabric_sf(path,L=12,x='Richards',sr_type='SR',model='R1'):

    if isinstance(x,str):
        if x == 'Richards':
            x_tile = prm.Richards2021(path.T)
        elif x == 'Elmer':
            x_tile = prm.Elmer(path.T)
        elif x=='Reduced':
            x_tile = prm.Richards2021Reduced(path.T)

            
    else:    
        x_tile = np.tile(x,(path.gradu.shape[0],1))

    

    dt_tile = np.tile(path.dt,(path.gradu.shape[0],))



    a2,a4,f = sff.solve(path.gradu,dt_tile,x_tile,L)


    return a2,a4,f

