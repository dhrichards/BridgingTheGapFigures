#%%
import numpy as np# numpy for arrays
from tqdm import tqdm
import track
import pickle
import copy
import agedepth
from matplotlib import pyplot as plt
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
for i in tqdm(range(len(paths))):
    paths[i].Temperature()
    paths[i].fabric_sf(L)
    depths[i] = paths[i].d[-1]

    ## Due to lack of vertical shear we know one eigenvector is (0,0,1),
    # so eigenvalue is just the vertical component of the fabric tensor.
    # TODO: update to find closest to vertical eigenvalue
    w,v = np.linalg.eig(paths[i].a2[-1,:2,:2])
    ev_n[i] = np.max(w)
    ev_s[i] = np.min(w)
    ev_z[i] = paths[i].a2[-1,2,2]

fig,ax = plt.subplots()
ax.scatter(stoll_d,e_z,s=0.3,marker='.',color=colors[0])
ax.scatter(stoll_d,e_n,s=0.3,marker='.',color=colors[1])
ax.scatter(stoll_d,e_s,s=0.3,marker='.',color=colors[2])

ax.plot(depths,ev_z,color=colors[0])
ax.plot(depths,ev_n,color=colors[1])
ax.plot(depths,ev_s,color=colors[2])

ax.set_xlabel('Depth (m)')
ax.set_ylabel('Eigenvalue')



#%%
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes as geoaxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from buildharmonics import BuildHarmonics
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

def plot(ax,odf,s_dir,hemisphere=True):
    X,Y,F = odf.plot(hemisphere=hemisphere)
    pcol = ax.pcolormesh(X,Y,F,transform=ccrs.PlateCarree(),vmax=vmax,vmin=0)
    pcol.set_edgecolor('face')
    ax.set_aspect('equal')
    ax.axis('off')
    kwargs_gridlines = {'ylocs':np.arange(-90,90+30,30), \
                        'xlocs':np.arange(s_dir-360,s_dir+360,45),\
                              'linewidth':0.5, 'color':'black', 'alpha':0.25, \
                                'linestyle':'-'}
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(),**kwargs_gridlines)#,xlocs=[s_dir,s_dir+90,s_dir+180,s_dir+270])


    if hemisphere:
        gl.ylim = (0,90)

    geo = ccrs.RotatedPole()

    text = []
    text.append(ax.text(0,90,'$z$',transform=geo,ha='center',va='center',color='white'))
    #ax.text(90,0,'$y$',transform=geo,ha='center',va='center')
    #ax.text(0,0,'$x$',transform=geo,ha='center',va='center')
    text.append(ax.text(s_dir,0,'$s$',transform=geo,ha='center',va='center',color='white'))
    text.append(ax.text(s_dir+90,0,'$n$',transform=geo,ha='center',va='center',color='white'))
    #text.append(ax.text(s_dir+270,0,'$-n$',transform=geo,ha='center',va='center',color='white'))

    for tex in text:
        tex.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                       path_effects.Normal()])
    # ax.text(s_dir+180,0,'$-s$',transform=geo,ha='center',va='center',color = 'white')
    # ax.text(s_dir+270,0,'$-n$',transform=geo,ha='center',va='center')


    return pcol

def inset(fig,ax,insx,insy,j,tind=-1,r=0.1,lon=90,lat=90,hemisphere=True,vmax=0.7):
    centre=ax.transAxes.inverted().transform(ax.transData.transform((insx, insy)))
    long_s = np.arctan2(paths[j].vy_s[-1],paths[j].vx_s[-1])*180/np.pi

    ins = inset_axes(ax,width="100%", height="100%",
                bbox_to_anchor=(centre[0]-r,centre[1]-r,2*r,2*r),
                bbox_transform=ax.transAxes, loc='center',
                axes_class=geoaxes.GeoAxes,
                axes_kwargs=dict(map_projection=ccrs.AzimuthalEquidistant(long_s+90,lat)
                                 ))

    #odf = Reconstruct(paths[j].mmc.n,paths[j].mmc.m)
    odf = BuildHarmonics(paths[j].n[tind,...],paths[j].m[tind,...],L,mmax)
    pcol = odf.plot(fig,ins,hemisphere=hemisphere,vmax=vmax,colorbar=False)

    geo = ccrs.RotatedPole()

    s_dir = long_s
    text = []
    text.append(ins.text(0,90,'$z$',transform=geo,ha='center',va='center',color='white'))
    #ax.text(90,0,'$y$',transform=geo,ha='center',va='center')
    #ax.text(0,0,'$x$',transform=geo,ha='center',va='center')
    text.append(ins.text(s_dir,0,'$s$',transform=geo,ha='center',va='center',color='white'))
    text.append(ins.text(s_dir+90,0,'$n$',transform=geo,ha='center',va='center',color='white'))
    #text.append(ax.text(s_dir+270,0,'$-n$',transform=geo,ha='center',va='center',color='white'))

    for tex in text:
        tex.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                       path_effects.Normal()])

    return pcol
    
    




# Create vertical eigenvalue plot
fig,ax2 = plt.subplots(figsize=(4,4))


ax2.plot(ev_z,depths,linewidth=2,color =colors[0])
ax2.plot(ev_n,depths,linewidth=2,color =colors[1])
ax2.plot(ev_s,depths,linewidth=2,color =colors[2])

ax2.scatter(e_z,stoll_d,s=0.3,color=colors[0])
ax2.scatter(e_n,stoll_d,s=0.3,color=colors[1])
ax2.scatter(e_s,stoll_d,s=0.3,color=colors[2])

if vmax<1:
    ax2.set_title('Eigenvalues at EGRIP')
else:
    ax2.set_title(r"Eigenvalues at EGRIP, $\tilde{\lambda}' = \tilde{\lambda}/4$")



ax2.set_xlabel('Eigenvalues of $\mathbf{A}^{(2)}$')
ax2.set_ylabel('Depth (m)')
ax2.set_ylim(0,2000)
#flip y axis
ax2.invert_yaxis()
ax2.grid()


nearestdepths = np.concatenate(([100],depthslower[1::2]))


#Custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='k', lw=2, label='Model'),
                   Line2D([0], [0], marker='o', color='w', label='EGRIP',
                          markerfacecolor='k', markersize=4)]
# put legend outside top right
ax2.legend(handles=legend_elements,fontsize=9,ncol=2,loc='lower center')





# # Get aspect ratio in fig coords for axes
# def get_aspect(ax):
#     pos = ax.get_position()
#     return (pos.ymax-pos.ymin)/(pos.xmax-pos.xmin)


# ar = get_aspect(ax2)

for n in nearestdepths:

    j = np.abs(depths - n).argmin()
    pcol=inset(fig,ax2,0.93,depths[j],j,r=0.1,vmax=vmax)


cbax2 = ax2.inset_axes([0.9,-0.14,0.3,0.04])
cbar=fig.colorbar(pcol, cax=cbax2, ticks=[0,vmax],orientation="horizontal", format='%.1f')
cbar.set_label('$f^*$',labelpad=-10,fontsize=9)



if npoints>=10000:
    if vmax<1:
        fig.savefig('eigenvalues.pdf', bbox_inches='tight')
    else:
        fig.savefig('eigenvalueslam.pdf', bbox_inches='tight')


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import specfabfuns as sff
vmax = 0.7
def inset(fig,ax,insx,insy,j,tind=-1,r=0.1,lon=90,lat=90,hemisphere=False,vmax=0.7):
    centre=ax.transAxes.inverted().transform(ax.transData.transform((insx, insy)))
    long_s = np.arctan2(paths[j].vy_s[-1],paths[j].vx_s[-1])*180/np.pi

    ins = inset_axes(ax,width="100%", height="100%",
                bbox_to_anchor=(centre[0]-r,centre[1]-r,2*r,2*r),
                bbox_transform=ax.transAxes, loc='center',
                axes_class=geoaxes.GeoAxes,
                axes_kwargs=dict(map_projection=ccrs.Orthographic(long_s+90,lat)
                                 ))

    #odf = Reconstruct(paths[j].mmc.n,paths[j].mmc.m)
    # odf = BuildHarmonics(paths[j].n[tind,...],paths[j].m[tind,...],L,mmax)
    odf = sff.Plotting(L,paths[j].f[tind,...])
    pcol = odf.plot(fig,ins,hemisphere,vmax=vmax,colorbar=False)

    geo = ccrs.RotatedPole()
    
    s_dir = long_s
    text = []
    text.append(ins.text(0,90,'$z$',transform=geo,ha='center',va='center',color='white'))
    #ax.text(90,0,'$y$',transform=geo,ha='center',va='center')
    #ax.text(0,0,'$x$',transform=geo,ha='center',va='center')
    text.append(ins.text(s_dir,0,'$s$',transform=geo,ha='center',va='center',color='white'))
    text.append(ins.text(s_dir+90,0,'$n$',transform=geo,ha='center',va='center',color='white'))
    text.append(ins.text(s_dir+270,0,'$-n$',transform=geo,ha='center',va='center',color='white'))

    for tex in text:
        tex.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                       path_effects.Normal()])


    return pcol


''' Plot paths and surface and bed'''

fig,ax = plt.subplots(figsize=(7,5))

max = np.max(paths[-1].surf) +200
min = np.min(paths[-1].bed) - 100
ax.plot(-paths[-1].s/1e3,paths[-1].surf,color='k',linewidth=2)
ax.plot(-paths[-1].s/1e3,paths[-1].bed,color='k',linewidth=2)
ax.fill_between(-paths[-1].s/1e3,paths[-1].surf,max,color='#87ceeb')
ax.fill_between(-paths[-1].s/1e3,paths[-1].bed,min,color='#876445')


ax.set_xlim(0,-paths[-1].s[0]/1e3)
ax.set_ylim(min,max)
ax.invert_xaxis()
for p in paths:
    #ice blue color
    ax.plot(-p.s/1e3,p.z,'k--',linewidth=0.7)



ax.set_xlabel('Distance upstream (km)')
ax.set_ylabel('Height above sea-level (m)')


for n in nearestdepths:

    j = np.abs(depths - n).argmin()
    pcol=inset(fig,ax,-paths[j].s[-1]/1e3, paths[j].z[-1],j,r=0.08,lon=180,lat=20,vmax=vmax)
    

    ## Plot small figures along line
    splot = paths[j].s[0]
    ds = 50e3
    while splot<-40e3:
        ind = np.abs(paths[j].s-splot).argmin()
        inset(fig,ax,-paths[j].s[ind]/1e3, paths[j].z[ind],j,\
              r=0.07,tind=ind,lon=180,lat=20,hemisphere=False,vmax=vmax)

        splot = splot+ds

cbax = ax.inset_axes([0.1,0.25,0.3,0.04])
cbar=fig.colorbar(pcol, cax=cbax, ticks=[0,vmax],orientation="horizontal", format='%.1f')
cbar.set_label('$f^*$',labelpad=-10,fontsize=9)


if npoints>=10000:
    fig.savefig('paths.pdf', bbox_inches='tight')

#%%%%
# plot strain-rate T graph with strain-rate log scale
fig,ax = plt.subplots(figsize=(7,5))

secperyr = 365*24*60*60

for p in paths[-2:]:
    D = 0.5*(p.gradu + np.transpose(p.gradu,axes=(0,2,1)))
    effectiveSR = np.sqrt(0.5*np.einsum('pij,pji->p',D,D))/secperyr

    ax.scatter(effectiveSR,p.T)

ax.set_xlabel('Effective strain-rate (s$^{-1}$)')
ax.set_ylabel('Temperature (C)')

# set x log scale
ax.set_xscale('log')



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import netCDF4 as nc


fn = 'BedMachineGreenland-v5.nc'
ds = nc.Dataset(fn)



xmin = paths[-1].xp.min()-50e3
xmax = paths[-1].xp.max()+50e3
ymin = paths[-1].yp.min()-50e3
ymax = paths[-1].yp.max()+50e3


surface=np.squeeze(ds['surface'][:])
thickness=np.squeeze(ds['thickness'][:])
bed=np.squeeze(ds['bed'][:])
y = np.squeeze(ds['y'][:])
x = np.squeeze(ds['x'][:])

y=np.flip(y,0)
surface = np.flip(surface,0)
thickness = np.flip(thickness,0)
bed = np.flip(bed,0)


xminind = np.abs(x-xmin).argmin()
xmaxind = np.abs(x-xmax).argmin()
yminind = np.abs(y-ymin).argmin()
ymaxind = np.abs(y-ymax).argmin()


x = x[xminind:xmaxind]
y = y[yminind:ymaxind]
surface = surface[yminind:ymaxind,xminind:xmaxind]
thickness = thickness[yminind:ymaxind,xminind:xmaxind]
bed = bed[yminind:ymaxind,xminind:xmaxind]


Y,X = np.meshgrid(y,x)


''' Plot 3d paths'''

fig,ax = plt.subplots(subplot_kw={'projection': '3d'})

for p in paths[1:]:
    #ice blue color
    w,v = np.linalg.eig(p.a2)
    ind = np.argmax(np.abs(w),1)

    w = np.sort(w)

    ax.scatter(p.xp/1e3,p.yp/1e3,p.z,s=w[:,2])

    #ax.plot(p.xp/1e3,p.yp/1e3,p.z,)

ax.set_xlabel('x (km)')
ax.set_ylabel('y (km)')
ax.set_zlabel('z (m)')



ax.plot_surface(X/1e3,Y/1e3,surface.T,color='#87ceeb',alpha=0.3)
ax.plot_surface(X/1e3,Y/1e3,bed.T,color='#876445',alpha=1)


