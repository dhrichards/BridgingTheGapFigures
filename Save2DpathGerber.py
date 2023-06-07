#%%

import numpy as np
import track
import pickle
import GreenlandSG as Greenland

dt=50

xc = 244691
yc = -1544921

data = Greenland.gl()

p=track.path2d(100000,dt,xc,yc,data)

filename = 'path2dSGdt'+str(dt)+'.pkl'

with open(filename, 'wb') as f:
   pickle.dump(p,f)

# %%
