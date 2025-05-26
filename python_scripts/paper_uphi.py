
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import ticker
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

dpi = 300

t_mar, b_mar, l_mar, r_mar = (0.27, 0.2, 0.2, 0.05)
h_slice, w_slice = (1, 0.4)

h_cbar, w_cbar = (0.025, w_slice)
w_pad = 0.35

h_total = t_mar + h_cbar + h_slice + b_mar
w_total = l_mar + 1*w_pad + 2*w_slice + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# slices
slice_axes = []
for i in range(2):
    left = (l_mar + i*w_pad + i*w_slice) / w_total
    bottom = 1 - (t_mar + h_cbar + h_slice ) / h_total
    width = w_slice / w_total
    height = h_slice / h_total
    slice_axes.append(fig.add_axes([left, bottom, width, height]))

# cbars
cbar_axes = []
for i in range(2):
    left = (l_mar + i*w_pad + i*w_slice) / w_total
    bottom = 1 - (t_mar + h_cbar ) / h_total
    width = w_cbar / w_total
    height = h_cbar / h_total
    cbar_axes.append(fig.add_axes([left, bottom, width, height]))

# load slice data
c_im = []
cbar = []
for i, R in enumerate([4, 4]):
    if i==0:
        ending=''
    elif i==1:
        ending='_sf'
    data = pickle.load(open('R%i%s/data.pkl' %(R, ending), 'rb'))
    uphi = data['uphi_axi'][:,:]
    z = data['z'][:,0,0]
    Nz = len(z)
    z = z[:Nz//2+1]
    uphi = uphi[:Nz//2+1]
    r = data['r'][0,0,:]
    r[0] = 0
    r[-1] = 0.2

    # normalization
    uphi *= 2
    xm, ym = plot_tools.quad_mesh(r, z)
    c_im.append((slice_axes[i].pcolormesh(xm,ym,uphi,cmap=cmap1)))
    cbar.append(fig.colorbar(c_im[i], cax=cbar_axes[i], orientation='horizontal', ticks=MaxNLocator(nbins=4)))

for i in range(2):
    slice_axes[i].xaxis.set_major_locator(ticker.FixedLocator([0, 0.1, 0.2]))
    slice_axes[i].set_xlabel(r'$r$')
    slice_axes[i].xaxis.set_major_formatter(ticker.FixedFormatter([r'$0$',r'$\Lambda/4$',r'$\Lambda/2$']))

for i in range(2):
    slice_axes[i].yaxis.set_major_locator(ticker.FixedLocator([0, 0.25, 0.5]))
    slice_axes[i].set_ylabel(r'$z$')
    slice_axes[i].yaxis.set_major_formatter(ticker.FixedFormatter([r'$0$',r'$H/2$',r'$H$']))

for i in range(2):
    cbar_axes[i].xaxis.set_ticks_position('top')
    cbar_axes[i].xaxis.set_label_position('top')

for cb in cbar:
    cb.ax.tick_params(labelsize=8)
cbar_axes[0].text(0.5,6.5,r'$\overline{u_{\phi}}$',va='center',ha='center',fontsize=10,transform=cbar_axes[0].transAxes)
cbar_axes[0].text(0.5,10,r'${\rm no} \ {\rm slip}$',va='center',ha='center',fontsize=10,transform=cbar_axes[0].transAxes)
cbar_axes[1].text(0.5,6.5,r'$\overline{u_{\phi}}$',va='center',ha='center',fontsize=10,transform=cbar_axes[1].transAxes)
cbar_axes[1].text(0.5,10,r'${\rm stress-free}$',va='center',ha='center',fontsize=10,transform=cbar_axes[1].transAxes)

plt.savefig('figures/uphi.png', dpi=dpi)


