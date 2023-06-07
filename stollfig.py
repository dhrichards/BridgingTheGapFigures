#%%
import numpy as np# numpy for arrays
from tqdm import tqdm
import track
import pickle
import copy
from tqdm import tqdm 
import agedepth
from matplotlib import pyplot as plt
with open('path2dSGdt10.pkl', 'rb') as f:
    path2d = pickle.load(f)




import stoll
stoll_d,e_s,e_z,e_n = stoll.eigenvalues(dmin=0,dmax=2000)
# From data we know smallest eigenvalue is approximately in streamline direction,
# and the largest eigenvalue is approximately perpendicular to the streamline direction.

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 11,
    "figure.autolayout" : True,
    
})

# Create vertical eigenvalue plot
fig,ax2 = plt.subplots(figsize=(3,4))


ax2.scatter(e_z,stoll_d,s=0.3,color='#1f77b4')
ax2.scatter(e_n,stoll_d,s=0.3,color='#ff7f0e')
ax2.scatter(e_s,stoll_d,s=0.3,color='#2ca02c')

ax2.set_xlabel('Eigenvalues of $\mathbf{A}^{(2)}$')
ax2.set_ylabel('Depth (m)')
ax2.set_ylim(0,2000)
#flip y axis
ax2.invert_yaxis()
ax2.grid()

fig.savefig('stolldata.png',format='png',dpi=400)