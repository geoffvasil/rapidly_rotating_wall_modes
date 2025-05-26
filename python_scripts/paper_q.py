
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import h5py
import publication_settings
import pickle
from dedalus.extras import plot_tools
import brewer2mpl
import dedalus.public as de

matplotlib.rcParams.update(publication_settings.params)

color_map = ('spectral', 'diverging',11)
b2m = brewer2mpl.get_map(*color_map, reverse=True)
cmap1 = b2m.mpl_colormap
color_map = ('PRGn', 'diverging',11)
b2m = brewer2mpl.get_map(*color_map)
cmap2 = b2m.mpl_colormap

dpi = 600

t_mar, b_mar, l_mar, r_mar = (0.5, 0.4, 0.45, 0.1)
h_slice, w_slice = (1., 2.5)
h_pad = 0.4

h_cbar, w_cbar = (0.05, w_slice)
w_pad = 0.2

h_total = t_mar + 2*h_cbar + 2*h_slice + h_pad + b_mar
w_total = l_mar + 1*w_pad + 2*w_slice + r_mar

width = 7.1
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# slices
slice_axes = []
for i in range(2):
    for j in range(2):
        left = (l_mar + i*w_pad + i*w_slice) / w_total
        bottom = 1 - (t_mar + j*h_pad + (j+1)*(h_cbar + h_slice) ) / h_total
        width = w_slice / w_total
        height = h_slice / h_total
        slice_axes.append(fig.add_axes([left, bottom, width, height]))

# cbars
cbar_axes = []
for i in range(2):
    for j in range(2):
        left = (l_mar + i*w_pad + i*w_slice) / w_total
        bottom = 1 - (t_mar + (j+1)*h_cbar + j*(h_pad + h_slice) ) / h_total
        width = w_cbar / w_total
        height = h_cbar / h_total
        cbar_axes.append(fig.add_axes([left, bottom, width, height]))

fns = ['', '_sf']

c_im1s = []
c_im2s = []
cbar1s = []
cbar2s = []

for i, fn in enumerate(fns):

    # load slice data
    if i == 0:
        data = pickle.load(open('R4%s/data.pkl' %fn, 'rb'))
    else:
        data = pickle.load(open('R4%s/data.pkl' %fn, 'rb'))
    T = data['T_wall'][:,:,0]
    shift = np.argmin(T[1,:])
    
    divq_old = data['divq'][:,:,0]
    divq_old = np.hstack((divq_old, divq_old[:,0][:,None]))
    divq = np.copy(divq_old)
    divq[:, -shift:] = divq_old[:, :shift] 
    divq[:, :-shift] = divq_old[:, shift:] 
    curlq_old = data['curlq'][:,:,0]
    curlq_old = np.hstack((curlq_old, curlq_old[:,0][:,None]))
    curlq = np.copy(curlq_old)
    curlq[:, -shift:] = curlq_old[:, :shift]
    curlq[:, :-shift] = curlq_old[:, shift:]
    z = data['z'][:,0,0]
    Nz = len(z)
    z = z[:Nz//2+1]
    divq = divq[:Nz//2+1]
    curlq = curlq[:Nz//2+1]
    phi = data['phi'][0,:,0]
    phi = np.concatenate([phi, [2*np.pi]])
    
    # normalization
    divq *= 2
    curlq *= 2
    
    xm, ym = plot_tools.quad_mesh(phi, z)
    c_im1 = slice_axes[2*i  ].pcolormesh(xm,ym,divq,cmap=cmap1)
    c_im2 = slice_axes[2*i+1].pcolormesh(xm,ym,curlq,cmap=cmap2)

    cbar1s.append(fig.colorbar(c_im1, cax=cbar_axes[2*i  ], orientation='horizontal', ticks=MaxNLocator(nbins=5)))
    cbar2s.append(fig.colorbar(c_im2, cax=cbar_axes[2*i+1], orientation='horizontal', ticks=MaxNLocator(nbins=5)))

for i in range(2):
    for j in range(2):
        slice_axes[2*i+j].xaxis.set_major_locator(ticker.FixedLocator([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]))
        if j == 1:
            slice_axes[2*i+j].set_xlabel(r'$\phi$')
            slice_axes[2*i+j].xaxis.set_major_formatter(ticker.FixedFormatter([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$']))
        else:
            slice_axes[2*i+j].xaxis.set_major_formatter(ticker.FixedFormatter([r'',r'',r'',r'',r'']))

for i in range(2):
    for j in range(2):
        slice_axes[2*i+j].yaxis.set_major_locator(ticker.FixedLocator([0, 0.25, 0.5]))
        if i == 0:
            slice_axes[2*i+j].set_ylabel(r'$z$')
            slice_axes[2*i+j].yaxis.set_major_formatter(ticker.FixedFormatter([r'$0$',r'$H/2$',r'$H$']))
        else:
            slice_axes[2*i+j].yaxis.set_major_formatter(ticker.FixedFormatter([r'',r'',r'',r'',r'']))

for i in range(4):
    cbar_axes[i].xaxis.set_ticks_position('top')
    cbar_axes[i].xaxis.set_label_position('top')

for cbar1 in cbar1s:
    cbar1.ax.tick_params(labelsize=8)
for cbar2 in cbar2s:
    cbar2.ax.tick_params(labelsize=8)
for i in range(2):
    cbar_axes[2*i  ].text(0.5,6,r'$\nabla\cdot q$',va='center',ha='center',fontsize=10,transform=cbar_axes[2*i].transAxes)
    cbar_axes[2*i+1].text(0.5,6,r'$\nabla\times q$',va='center',ha='center',fontsize=10,transform=cbar_axes[2*i+1].transAxes)

cbar_axes[0].text(0.5,9,r'${\rm no} \ {\rm slip}$',va='center',ha='center',fontsize=10,transform=cbar_axes[0].transAxes)
cbar_axes[2].text(0.5,9,r'${\rm stress-free}$',va='center',ha='center',fontsize=10,transform=cbar_axes[2].transAxes)


plt.savefig('figures/q_wall.png',dpi=dpi)


