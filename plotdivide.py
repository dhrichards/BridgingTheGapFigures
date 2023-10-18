#%%
import numpy as np
import matplotlib.pyplot as plt
import track
import divide_data
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.mpl.geoaxes as geoaxes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mcfab import BuildHarmonics
import matplotlib.patheffects as path_effects


plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 11,
    "figure.autolayout" : True,
    
})
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

npoints=10000
L=20
location = 'GRIP'#,'DomeF','DomeC','Talos']
#locations = ['DomeF']


param = 'Richards'
legend = 'Richards et al. (2021)'
sr_type = 'SR'
model = 'R1'


p = track.divide(dt=100,location=location,dh=0.9)
p.depth()




data =divide_data.data(location)

ev1 = data.largest_ev
ev2 = ev3 = (1 - ev1)/2



a2,a4,f = track.fabric_sf(p,L)

a2r,a4r,fr = track.fabric_sf(p,L,x='Reduced')

eigvals = np.linalg.eigvals(a2)

eigvals = np.sort(eigvals,axis=1)

eigvalsr = np.linalg.eigvals(a2r)

eigvalsr = np.sort(eigvalsr,axis=1)
#%%

fig,ax = plt.subplots(1,1,figsize=(4,4))

ax.scatter(ev2,data.largest_ev_d,color=colors[0],s=10)
ax.scatter(ev1,data.largest_ev_d,color=colors[1],s=10)
ax.scatter(ev3,data.largest_ev_d,color=colors[2],s=10)



ax.plot(eigvals[:,1],p.d,color=colors[0],linewidth=2)
ax.plot(eigvals[:,2],p.d,color=colors[1],linewidth=2)
ax.plot(eigvals[:,0],p.d,color=colors[2],linewidth=2)




vmax = 1.5
if vmax<1:
    ax.set_title('Eigenvalues at GRIP')
else:
    ax.set_title(r"Eigenvalues at GRIP, $\tilde{\lambda}' = \tilde{\lambda}/4$")


ax.set_xlabel('Eigenvalues of $\mathbf{A}^{(2)}$')
ax.set_ylabel('Depth (m)')
ax.grid()
ax.set_ylim(0,3500)
ax.invert_yaxis()



#Custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='k', lw=2, label='SpecCAF'),
                   Line2D([0], [0], marker='o', color='w', label='GRIP',
                          markerfacecolor='k', markersize=4)]
# put legend outside top right
ax.legend(handles=legend_elements,fontsize=9,ncol=2,loc='lower center')



# ar = get_aspect(ax2)

nearestdepths = np.linspace(0,p.d[-1],4)
L=8
mmax = 8



def inset(fig,ax,insx,insy,j,r=0.1,hemisphere=True,vmax=0.7):
    centre=ax.transAxes.inverted().transform(ax.transData.transform((insx, insy)))
    
    ins = inset_axes(ax,width="100%", height="100%",
                bbox_to_anchor=(centre[0]-r,centre[1]-r,2*r,2*r),
                bbox_transform=ax.transAxes, loc='center',
                axes_class=geoaxes.GeoAxes,
                axes_kwargs=dict(map_projection=ccrs.AzimuthalEquidistant(0,90)
                                 ))

    #odf = Reconstruct(paths[j].mmc.n,paths[j].mmc.m)
    odf = BuildHarmonics(n[j,...],m[j],L,mmax)
    pcol = odf.plot(fig,ins,hemisphere=hemisphere,vmax=vmax,colorbar=False)

    geo = ccrs.RotatedPole()

    
    text = []
    text.append(ins.text(0,90,'$z$',transform=geo,ha='center',va='center',color='white'))
    # #ax.text(90,0,'$y$',transform=geo,ha='center',va='center')
    # #ax.text(0,0,'$x$',transform=geo,ha='center',va='center')
    # text.append(ins.text(s_dir,0,'$s$',transform=geo,ha='center',va='center',color='white'))
    # text.append(ins.text(s_dir+90,0,'$n$',transform=geo,ha='center',va='center',color='white'))
    # #text.append(ax.text(s_dir+270,0,'$-n$',transform=geo,ha='center',va='center',color='white'))

    for tex in text:
        tex.set_path_effects([path_effects.Stroke(linewidth=1, foreground='black'),
                       path_effects.Normal()])

    return pcol



for d in nearestdepths:

    j = np.abs(p.d - d).argmin()
    pcol=inset(fig,ax,1.15,p.d[j],j,r=0.1,vmax=vmax)


cbax2 = ax.inset_axes([0.9,-0.12,0.3,0.04])
cbar=fig.colorbar(pcol, cax=cbax2, ticks=[0,vmax],orientation="horizontal", format='%.1f')
cbar.set_label('$f^*$',labelpad=-10,fontsize=9)

if vmax<1:
    fig.savefig('gripcomparison.pdf',bbox_inches='tight')
else:
    fig.savefig('gripcomparisonlam.pdf',bbox_inches='tight')



#%%
import seaborn as sns
colors = sns.color_palette("deep", 3)

fig,ax = plt.subplots(1,1,figsize=(4,4))

ax.scatter(ev2,data.largest_ev_d,color=colors[0],s=6)
ax.scatter(ev1,data.largest_ev_d,color=colors[1],s=6)
ax.scatter(ev3,data.largest_ev_d,color=colors[2],s=6)



ax.plot(eigvals[:,1],p.d,color=colors[0],linewidth=2)
ax.plot(eigvals[:,2],p.d,color=colors[1],linewidth=2)
ax.plot(eigvals[:,0],p.d,color=colors[2],linewidth=2)

ax.plot(eigvalsr[:,1],p.d,color=colors[0],linewidth=2,linestyle='--')
ax.plot(eigvalsr[:,2],p.d,color=colors[1],linewidth=2,linestyle='--')
ax.plot(eigvalsr[:,0],p.d,color=colors[2],linewidth=2,linestyle='--')



ax.set_title('Eigenvalues at GRIP')


ax.set_xlabel('Eigenvalues of $\mathbf{A}^{(2)}$')
ax.set_ylabel('Depth (m)')
ax.grid()
ax.set_ylim(0,3700)
ax.invert_yaxis()


from matplotlib.lines import Line2D
#Custom legend
legend_elements = [Line2D([0], [0], color='k', lw=2, label='SpecCAF'),
                     Line2D([0], [0], color='k', lw=2, linestyle='--', label=r"$\tilde{\lambda}' = \tilde{\lambda}/4$"),
                   Line2D([0], [0], marker='o', color='w', label='GRIP',
                          markerfacecolor='k', markersize=4)]
# put legend outside top right
ax.legend(handles=legend_elements,fontsize=9,ncol=2,loc='lower center')

fig.savefig('gripcomparisondouble.pdf',bbox_inches='tight')


#%%
# plot 3 vertical subplots for each eigenvalue, sharing y axis
fig,ax = plt.subplots(1,3,figsize=(6,4),sharey=True)

import cmocean
colors = sns.color_palette("deep", 3)


#plot smallest eigenvalue in left plot, etc.
ax[0].plot(eigvals[:,0],p.d,color=colors[0],linewidth=2,label='SpecCAF')
ax[1].plot(eigvals[:,1],p.d,color=colors[0],linewidth=2)
ax[2].plot(eigvals[:,2],p.d,color=colors[0],linewidth=2)

ax[0].plot(eigvalsr[:,0],p.d,color=colors[1],linewidth=2,label=r"$\tilde{\lambda}' = 0.25\tilde{\lambda}$")
ax[1].plot(eigvalsr[:,1],p.d,color=colors[1],linewidth=2)
ax[2].plot(eigvalsr[:,2],p.d,color=colors[1],linewidth=2)

ax[0].scatter(ev3,data.largest_ev_d,color=colors[2],s=6,label='GRIP')
ax[1].scatter(ev2,data.largest_ev_d,color=colors[2],s=6)
ax[2].scatter(ev1,data.largest_ev_d,color=colors[2],s=6)

ax[0].set_xlabel('$e_1$')
ax[1].set_xlabel('$e_2$')
ax[2].set_xlabel('$e_3$')

ax[0].set_ylabel('Depth (m)')

fig.suptitle('Eigenvalues at GRIP')

for a in ax:
    a.grid()
    a.invert_yaxis()

fig.legend(loc='lower center',ncol=3,bbox_to_anchor=(0.5,-0.08))

fig.savefig('gripcomparisonsplit.pdf',bbox_inches='tight')