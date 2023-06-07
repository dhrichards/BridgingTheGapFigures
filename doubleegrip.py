#%%
import numpy as np# numpy for arrays
from tqdm import tqdm
import track
import pickle
import copy
from tqdm import tqdm 
import agedepth
from matplotlib import pyplot as plt
import mcfab as mc
with open('path2dSGdt10.pkl', 'rb') as f:
    path2d = pickle.load(f)





depthsupper = np.array([5,25,50,75,100,150,200,250])
depthslower = np.arange(375,1875,250)
depths = np.concatenate((depthsupper,depthslower))
#colors = ['#03045e', '#0077b6', '#00b4d8','#90e0ef','#caf0f8','#f72585','#7209b7','#3a0ca3','#4361ee','#4cc9f0']
#default colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


#depths = np.linspace(100,1800,16)
times=-agedepth.depth2time(depths)


paths=[]
for t in times:
    nt = path2d.nt - np.abs(path2d.t - t).argmin()
    paths.append(track.path3d(copy.deepcopy(path2d),nt))


for p in paths:
    p.optimizeacc()

import stoll
stoll_d,e_s,e_z,e_n = stoll.eigenvalues(dmin=depths[0],dmax=depths[-1])
# From data we know smallest eigenvalue is approximately in streamline direction,
# and the largest eigenvalue is approximately perpendicular to the streamline direction.


L=12
npoints = 10000


depths = np.zeros(len(paths))
ev_s = np.zeros(len(paths))
ev_n = np.zeros(len(paths))
ev_z = np.zeros(len(paths))

paths2 = copy.deepcopy(paths)
ev_s2 = np.zeros(len(paths))
ev_n2 = np.zeros(len(paths))
ev_z2 = np.zeros(len(paths))

for i in tqdm(range(len(paths))):
    paths[i].Temperature()
    paths2[i].Temperature()
    paths[i].fabric_mc(npoints)
    paths2[i].fabric_mc(npoints,x='Reduced')
    depths[i] = paths[i].d[-1]


    w,v = np.linalg.eig(paths[i].a2[-1,:2,:2])
    ev_n[i] = np.max(w)
    ev_s[i] = np.min(w)
    ev_z[i] = paths[i].a2[-1,2,2]

    w2,v2 = np.linalg.eig(paths2[i].a2[-1,:2,:2])
    ev_n2[i] = np.max(w2)
    ev_s2[i] = np.min(w2)
    ev_z2[i] = paths2[i].a2[-1,2,2]

fig,ax = plt.subplots()
ax.scatter(stoll_d,e_z,s=0.3,marker='.',color=colors[0])
ax.scatter(stoll_d,e_n,s=0.3,marker='.',color=colors[1])
ax.scatter(stoll_d,e_s,s=0.3,marker='.',color=colors[2])

ax.plot(depths,ev_z,color=colors[0])
ax.plot(depths,ev_n,color=colors[1])
ax.plot(depths,ev_s,color=colors[2])

ax.plot(depths,ev_z2,color=colors[0],linestyle='--')
ax.plot(depths,ev_n2,color=colors[1],linestyle='--')
ax.plot(depths,ev_s2,color=colors[2],linestyle='--')

ax.set_xlabel('Depth (m)')
ax.set_ylabel('Eigenvalue')



#%%
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes as geoaxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from meso_fab_mc import BuildHarmonics
import matplotlib.patheffects as path_effects

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 11,
    "figure.autolayout" : True,
    
})

L=8
mmax = 4
vmax = 1.5


def loadEGRIP(loc):

    if loc==0:
        filename = 'stereo_EGRIP266_2_20.txt'
        depth = 145.93
    elif loc==1:
        filename = 'stereo_EGRIP1906_6_20.txt'
        depth = 1048.3
    else:
        filename = 'stereo_EGRIP2635_4_20.txt'
        depth = 1499.07

    # load data as tab delimited with header
    data = np.loadtxt(filename, delimiter='\t', skiprows=1)

    # extract columns
    lon = data[:,0]
    lat = data[:,1]

    # convert to radians
    lon = lon*np.pi/180
    lat = lat*np.pi/180

    # convert to xyz
    x = np.cos(lat)*np.cos(lon)
    y = np.cos(lat)*np.sin(lon)
    z = np.sin(lat)

    # create array of xyz
    xyz = np.array([x,y,z]).T
    m = np.ones(len(xyz))
    a2 = np.einsum('pi,pj->ij',xyz,xyz)/len(xyz)
    w,v = np.linalg.eig(a2[:2,:2])
    epf_n = np.max(w)
    epf_s = np.min(w)
    epf_z =a2[2,2]

    w = np.array([epf_n,epf_z,epf_s])

    return xyz,w,depth

ev_exp = np.zeros((3,3))
d_exp = np.zeros(3)
for loc in range(3):
    _,ev_exp[loc,:],d_exp[loc] = loadEGRIP(loc)

import seaborn as sns
import cmocean
colors = sns.color_palette("deep", 3)
colors_bright = sns.color_palette("bright", 3)




# Create vertical eigenvalue plot
fig,ax = plt.subplots(figsize=(4,4))


ax.plot(ev_z,depths,linewidth=2,color =colors[0])
ax.plot(ev_n,depths,linewidth=2,color =colors[1])
ax.plot(ev_s,depths,linewidth=2,color =colors[2])

ax.plot(ev_z2,depths,linewidth=2,color =colors[0],linestyle='--')
ax.plot(ev_n2,depths,linewidth=2,color =colors[1],linestyle='--')
ax.plot(ev_s2,depths,linewidth=2,color =colors[2],linestyle='--')


ax.scatter(e_z,stoll_d,s=0.3,color=colors[0],alpha=0.5)
ax.scatter(e_n,stoll_d,s=0.3,color=colors[1],alpha=0.5)
ax.scatter(e_s,stoll_d,s=0.3,color=colors[2],alpha=0.5)

#Highlight these points
ax.scatter(ev_exp[:,1],d_exp,s=100,color=colors_bright[0],marker='x')
ax.scatter(ev_exp[:,0],d_exp,s=100,color=colors_bright[1],marker='x')
ax.scatter(ev_exp[:,2],d_exp,s=100,color=colors_bright[2],marker='x')


ax.set_title('Eigenvalues at EGRIP')


ax.set_xlabel('Eigenvalues of $\mathbf{A}^{(2)}$')
ax.set_ylabel('Depth (m)')
ax.set_ylim(0,2100)
#flip y axis
ax.invert_yaxis()
ax.grid()


nearestdepths = np.concatenate(([100],depthslower[1::2]))


#Custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='k', lw=2, label='SpecCAF'),
                     Line2D([0], [0], color='k', lw=2, linestyle='--', label=r"$\tilde{\lambda}' = \tilde{\lambda}/4$"),
                   Line2D([0], [0], marker='o', color='w', label='EGRIP',
                          markerfacecolor='k', markersize=4),
                          Line2D([0], [0], marker='x', color='w', label='Pole figures',
                            markerfacecolor='k', markersize=8,markeredgecolor='k')]
# put legend outside top right
ax.legend(handles=legend_elements,fontsize=9,ncol=2,loc='lower center')





# # Get aspect ratio in fig coords for axes
# def get_aspect(ax):
#     pos = ax.get_position()
#     return (pos.ymax-pos.ymin)/(pos.xmax-pos.xmin)


# ar = get_aspect(ax2)

# for n in nearestdepths:

#     j = np.abs(depths - n).argmin()
#     pcol=inset(fig,ax2,0.93,depths[j],j,r=0.1,vmax=vmax)


# cbax2 = ax2.inset_axes([0.9,-0.14,0.3,0.04])
# cbar=fig.colorbar(pcol, cax=cbax2, ticks=[0,vmax],orientation="horizontal", format='%.1f')
# cbar.set_label('$f^*$',labelpad=-10,fontsize=9)

fig.savefig('eigenvaluesdouble.pdf', bbox_inches='tight')

#%%
# Create three subplots sharing y axis, y is depth

fig,axs = plt.subplots(1, 3, sharey=True,figsize=(6,4))

colors = sns.color_palette("cmo.ice", 3)



# Plot smallest eigenvalue in left subplot
axs[0].plot(ev_s,depths,linewidth=2,color =colors[0],label='SpecCAF')
axs[0].plot(ev_s2,depths,linewidth=2,color =colors[1],label=r"$\tilde{\lambda}' = 0.25\tilde{\lambda}$")
axs[0].scatter(e_s,stoll_d,s=0.3,color=colors[2],alpha=0.5,label='EGRIP')

# Plot middle eigenvalue in middle subplot
axs[1].plot(ev_z,depths,linewidth=2,color =colors[0])
axs[1].plot(ev_z2,depths,linewidth=2,color =colors[1])
axs[1].scatter(e_z,stoll_d,s=0.3,color=colors[2],alpha=0.5)

# Plot largest eigenvalue in right subplot
axs[2].plot(ev_n,depths,linewidth=2,color =colors[0])
axs[2].plot(ev_n2,depths,linewidth=2,color =colors[1])
axs[2].scatter(e_n,stoll_d,s=0.3,color=colors[2],alpha=0.5)

#Highlight these points
axs[0].scatter(ev_exp[:,2],d_exp,s=100,color=colors[2],marker='x',label='Pole figures')
axs[1].scatter(ev_exp[:,1],d_exp,s=100,color=colors[2],marker='x')
axs[2].scatter(ev_exp[:,0],d_exp,s=100,color=colors[2],marker='x')

# Set titles
axs[0].set_xlabel('$e_s$')
axs[1].set_xlabel('$e_z$')
axs[2].set_xlabel('$e_n$')

# fig title
fig.suptitle('Eigenvalues at EGRIP')

# grids
for ax in axs:
    ax.grid()

# legend - move verticall down a bit
fig.legend(loc='lower center',ncol=4,bbox_to_anchor=(0.5, -0.07))
# Set y label
axs[0].set_ylabel('Depth (m)')
#flip y axis
axs[0].invert_yaxis()

fig.savefig('eigenvaluessplit.pdf', bbox_inches='tight')


#%%
L=8
mmax = 8
vmax=2

def J(odf):
    J=0
    Sff = 0
    for l in range(0,odf.L+1,2):
        Sff = 0*Sff
        for m in range(0,l+1,1):
            Sff=Sff+np.abs(odf.f[odf.sh.idx(l,abs(m))])**2
        J=J+Sff
    return J

loc = 2
xyz,epf,depth = loadEGRIP(loc=loc)


def angle_correction(xyz):
    angle_corrector = 124.94 #Westhoff average of two peaks
    
    a2 = np.einsum('pi,pj->ij',xyz,xyz)/len(xyz)
    w,v = np.linalg.eig(a2[:2,:2])

    # Find eigenvector corresponding to largest eigenvalue
    idx = np.argmax(w)
    v = v[:,idx]

    # Find angle between eigenvector and y axis
    angle_v =  90 - np.arctan2(v[1],v[0])*180/np.pi

    # Correct angle
    angle = angle_corrector - angle_v
    # Convert xyz to phi,theta
    phi = np.arctan2(xyz[:,1],xyz[:,0])
    theta = np.arccos(xyz[:,2])

    # Update phi
    phi = phi - angle*np.pi/180

    # Convert back to xyz
    xyz[:,0] = np.cos(phi)*np.sin(theta)
    xyz[:,1] = np.sin(phi)*np.sin(theta)
    xyz[:,2] = np.cos(theta)

    return xyz

xyz = angle_correction(xyz)
m = np.ones(len(xyz))

#get path with nearest depth
j = np.abs(depths-depth).argmin()

epf = np.sort(epf)
espec = np.sort(np.array([ev_n[j],ev_z[j],ev_s[j]]))
espec2 = np.sort(np.array([ev_n2[j],ev_z2[j],ev_s2[j]]))

n1 = paths[j].n[-1,...]
m1 = paths[j].m[-1,...]

n2 = paths2[j].n[-1,...]
m2 = paths2[j].m[-1,...]

fig,ax = plt.subplots(1,3,figsize=(6,3.3),subplot_kw=\
                      {'projection':ccrs.AzimuthalEquidistant(90,90)})
odf1 = BuildHarmonics(n1,m1,L,mmax)
odf2 = BuildHarmonics(n2,m2,L,mmax)
odf_exp = BuildHarmonics(xyz,m,L,mmax)

odf1.plot(fig,ax[0],hemisphere=True)
odf2.plot(fig,ax[1],hemisphere=True)
odf_exp.plot(fig,ax[2],hemisphere=True)


J1 = J(odf1)
J2 = J(odf2)
J_exp = J(odf_exp)

# ax[0].set_title('(a) SpecCAF'\
#                 +'\n'+r'$e_{1,2,3} = '+'{:.2f}, {:.2f}, {:.2f}'.format(espec[0],espec[1],espec[2])+'$')
# ax[1].set_title(r"(b) $\tilde{\lambda}' = \tilde{\lambda}/4$"\
#                 +'\n'+r'$e_{1,2,3} = '+'{:.2f}, {:.2f}, {:.2f}'.format(espec2[0],espec2[1],espec2[2])+'$')
# ax[2].set_title('(c) EGRIP ice core data'\
#                 +'\n'+r'$e_{1,2,3} = '+'{:.2f}, {:.2f}, {:.2f}'.format(epf[0],epf[1],epf[2])+'$')


ax[0].set_title('(a) SpecCAF \n $J={:.2f}$'.format(J1))
ax[1].set_title(r"(b) $\tilde{\lambda}' = 0.25\tilde{\lambda}$" +"\n$J={:.2f}$".format(J2))
ax[2].set_title('(c) EGRIP ice core data\n$J={:.2f}$'.format(J_exp))
#figure title
fig.suptitle('Pole figures at {:.0f} m'.format(depth),y=1.05)
fig.savefig('polefigs' + str(loc) +'.pdf', bbox_inches='tight')


#%%
# fig = plt.figure(figsize=(6.5,5.5))

# subfigs = fig.subfigures(nrows=2, ncols=1)

fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(6,6),subplot_kw=\
                        {'projection':ccrs.AzimuthalEquidistant(90,90)})




rowletter = ['(a)','(b)','(c)']

for row in range(2):
    

    loc = row+1
    xyz,epf,depth = loadEGRIP(loc=loc)
    xyz = angle_correction(xyz)
    m = np.ones(len(xyz))

    #get path with nearest depth
    j = np.abs(depths-depth).argmin()

    epf = np.sort(epf)
    espec = np.sort(np.array([ev_n[j],ev_z[j],ev_s[j]]))
    espec2 = np.sort(np.array([ev_n2[j],ev_z2[j],ev_s2[j]]))

    n1 = paths[j].n[-1,...]
    m1 = paths[j].m[-1,...]

    n2 = paths2[j].n[-1,...]
    m2 = paths2[j].m[-1,...]

    # ax = subfig.subplots(nrows=1, ncols=3,subplot_kw=\
    #                         {'projection':ccrs.AzimuthalEquidistant(90,90)})    
    ax = axs[row,:]
    odf1 = BuildHarmonics(n1,m1,L,mmax)
    odf2 = BuildHarmonics(n2,m2,L,mmax)
    odf_exp = BuildHarmonics(xyz,m,L,mmax)

    pcol1 = odf1.plot(fig,ax[0],hemisphere=True,colorbar=True,pad=0.05)
    pcol2 = odf2.plot(fig,ax[1],hemisphere=True,colorbar=True,pad=0.05)
    pcol_exp = odf_exp.plot(fig,ax[2],hemisphere=True, colorbar=True,pad=0.05)

    

    J1 = J(odf1)
    J2 = J(odf2)
    J_exp = J(odf_exp)

    fmax1 = pcol1.get_clim()[1]
    fmax2 = pcol2.get_clim()[1]
    fmax_exp = pcol_exp.get_clim()[1]

    def titlestr(J,fmax):
        return '\n $J={:.2f}$'.format(J)# + r'\; \rho^*_{max} =' + '{:.2f}$'.format(fmax)

    # add J and fmax to plot
    colnumeral  = ['i','ii','iii']
    
    ax[0].set_title('(' +colnumeral[0] + ') SpecCAF' + titlestr(J1,fmax1))
    ax[1].set_title('(' +colnumeral[1] + r") $\tilde{\lambda}' = \tilde{\lambda}/4$" + titlestr(J2,fmax2))
    ax[2].set_title('(' +colnumeral[2] + ') EGRIP ice core data' + titlestr(J_exp,fmax_exp))

    #subfig.suptitle(rowletter[row] + f' Pole figures at {depth:.0f} m',y=0.96)
    #fig.suptitle(rowletter[row] + f' Pole figures at {depth:.0f} m',y=0.96)


fig.text(0.5, 1.0, '(a) Pole figures at 1048 m', ha='center', va='center',fontsize=13)
fig.text(0.5, 0.5, '(b) Pole figures at 1499 m', ha='center', va='center',fontsize=13)

# Add custom colorbar from 0 to 1, horiztonal
# with customticklabels so max is \rho^*_{max}
# cbax = fig.add_axes([0.1, 0.03, 0.8, 0.015])
# cbar=fig.colorbar(pcol_exp, cax=cbax, ticks=[0,fmax_exp],orientation="horizontal")
# cbar.ax.set_xticklabels(['0',r'$\rho^*_{max}$'])
# cbar.set_label(r'$\rho^*$',labelpad=-10)

fig.savefig('polefigs.pdf',bbox_inches='tight')

    #Add vertical text relative to ax[0] centre in fig coords

    # #Get ax[0] centre in fig coords
    # ax0centre = ax[0].transAxes.transform([0,0.5])
    # #Get fig coords of ax[0] centre
    # ax0centre = fig.transFigure.inverted().transform(ax0centre)

    # fig.text(ax0centre[0]-0.1,ax0centre[1],f'Depth $= {depth:.0f}$ m',rotation=90,va='center',ha='center')


#%%

# Calculate effective strain along paths
# and plot against depth
depths = np.zeros(len(paths))
strains = np.zeros(len(paths))
times = np.zeros(len(paths))
for i in range(len(paths)):
    p = paths[i]
    D = 0.5*(p.gradu+np.transpose(p.gradu,(0,2,1)))
    effSR = np.sqrt(0.5*np.einsum('pij,pji->p',D,D))

    strain = np.cumsum(effSR)*p.dt

    p.strain = strain

    depths[i] = p.d[-1]
    strains[i] = strain[-1]
    times[i] = -p.t[0]

fig,ax = plt.subplots()
ax.plot(depths,strains)

plt.figure()
plt.plot(depths,times)

# interpolate to find value at depth=500
from scipy.interpolate import interp1d
f = interp1d(depths,strains)
f2 = interp1d(depths,times)
print(f(500))
print(f2(500))


