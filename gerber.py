#%%
import numpy as np
from scipy.interpolate import interp1d,griddata
import pandas as pd
import pyproj

filename = 'UpstreamCorrection_data_10Jun2021.csv'

data = np.genfromtxt(filename,delimiter=',')

s = -np.flipud(data[3:,5])*1e3
surf = np.flipud(data[3:,10])
acc_s = np.flipud(data[3:,20])


line1 = pd.read_excel("./SupplementaryMaterial/UpstreamCorrection_data_10Jun2021.xls",
                     sheet_name=1)
line2 = pd.read_excel("./SupplementaryMaterial/UpstreamCorrection_data_10Jun2021.xls",
                     sheet_name=2)
line3 = pd.read_excel("./SupplementaryMaterial/UpstreamCorrection_data_10Jun2021.xls",
                     sheet_name=3)



P = pyproj.Proj('epsg:3413')

def LonLat_To_XY(Lon,Lat):
    return P(Lon,Lat)    


x1,y1 = LonLat_To_XY(line1['longitude'].to_numpy(),line1['latitude '].to_numpy())
acc1 = line1['acc rate [m/yr]'].to_numpy()
basalmelt1 = line1['basal melt [m/yr]'].to_numpy()
f1 = line1['basal sliding'].to_numpy()

x2,y2 = LonLat_To_XY(line2['longitude'].to_numpy(),line2['latitude '].to_numpy())
acc2 = line2['acc rate [m/yr]'].to_numpy()
basalmelt2 = line2['basal melt [m/yr]'].to_numpy()
f2 = line2['basal sliding'].to_numpy()

x3,y3 = LonLat_To_XY(line3['longitude'].to_numpy(),line3['latitude '].to_numpy())
x3 = x3[~np.isnan(x3)]
y3 = y3[~np.isnan(y3)]
acc3 = line3['acc rate [m/yr]'].to_numpy()
basalmelt3 = line3['basal melt [m/yr]'].to_numpy()
f3 = line3['basal sliding'].to_numpy()


x = np.concatenate((x1,x2,x3),axis=0)
y = np.concatenate((y1,y2,y3),axis=0)
acc = np.concatenate((acc1,acc2,acc3),axis=0)
basalmelt = np.concatenate((basalmelt1,basalmelt2,basalmelt3),axis=0)
f = np.concatenate((f1,f2,f3),axis=0)

nans = np.isnan(x)
y=y[~nans]
acc = acc[~nans]
basalmelt = basalmelt[~nans]
f = f[~nans]
x = x[~nans]


class gerb():
    def __init__(self):
        self.surf_interp = interp1d(s,surf,fill_value='extrapolate')
        self.acc_interp = interp1d(s,acc_s,fill_value='extrapolate')

    def interps(self,si):
        self.si = si

        self.surf_s = self.surf_interp(si)
        self.acc_s = self.acc_interp(si)

    def interps_xy(self,xi,yi):

        if xi.ndim==2:
            pts = np.reshape([xi,yi],(2,-1)).T
        elif xi.ndim==1:
            pts = np.stack((xi,yi),1)
        else:
            pts=[xi,yi]


        self.acc = griddata(np.stack((x,y),1),acc,pts)
        self.f = griddata(np.stack((x,y),1),f,pts)
        self.basalmelt = griddata(np.stack((x,y),1),basalmelt,pts)

        # self.acc2 = griddata(np.stack((x2,y2),1),acc2,pts)
        # self.f2 = griddata(np.stack((x2,y2),1),f2,pts)

        # self.acc3 = griddata(np.stack((x3,y3),1),acc3,pts)
        # self.f3 = griddata(np.stack((x3,y3),1),f3,pts)

        


        




# %%
