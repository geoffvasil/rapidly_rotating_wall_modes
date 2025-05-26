
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

color_map = ('RdBu', 'diverging',11)
b2m = brewer2mpl.get_map(*color_map, reverse=True)
cmap1 = b2m.mpl_colormap

dpi = 300

t_mar, b_mar, l_mar, r_mar = (0.2, 0.2, 0.28, 0.07)
h_slice, w_slice = (1., 3)

h_cbar, w_cbar = (0.05, w_slice)
h_pad = h_cbar

h_total = t_mar + h_cbar + h_pad + h_slice + b_mar
w_total = l_mar + w_slice + r_mar

width = 7.1
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# slices
left = (l_mar) / w_total
bottom = 1 - (t_mar + h_cbar + h_pad + h_slice) / h_total
width = w_slice / w_total
height = h_slice / h_total
slice_axes = fig.add_axes([left, bottom, width, height])

# cbars
left = (l_mar) / w_total
bottom = 1 - (t_mar + h_cbar) / h_total
width = w_cbar / w_total
height = h_cbar / h_total
cbar_axes = fig.add_axes([left, bottom, width, height])

data = pickle.load(open('R2/data_hov.pkl', 'rb'))
t = data['t']
phi = data['phi'].ravel()
phi[-1] = 2*np.pi
T = data['T_hov']

# normalization
T *= 2

xm, ym = plot_tools.quad_mesh(t, phi)
c_im = slice_axes.pcolormesh(xm, ym, T.T, cmap=cmap1)
cbar = fig.colorbar(c_im, cax = cbar_axes, orientation='horizontal', ticks=MaxNLocator(nbins=5))

slice_axes.set_xlabel(r'$t$')
slice_axes.set_ylabel(r'$\phi$')
slice_axes.yaxis.set_major_locator(ticker.FixedLocator([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]))
slice_axes.yaxis.set_major_formatter(ticker.FixedFormatter([r'$0$',r'$\pi/2$',r'$\pi$',r'$3\pi/2$',r'$2\pi$']))
slice_axes.xaxis.set_major_locator(ticker.FixedLocator([0, 0.05, 0.1, 0.15, 0.2, 0.25]))
slice_axes.xaxis.set_major_formatter(ticker.FixedFormatter([r'$0$',r'$0.05$',r'$0.1$',r'$0.15$',r'$0.2$',r'$0.25$']))
slice_axes.axis([0, 0.25, 0, 2*np.pi])

cbar_axes.xaxis.set_ticks_position('top')
cbar_axes.xaxis.set_label_position('top')
cbar.ax.tick_params(labelsize=8)
cbar_axes.text(0.5,4,r'$T(r=0.99\Gamma/2,z=H/2)$',va='center',ha='center',fontsize=10,transform=cbar_axes.transAxes)

plt.savefig('figures/T_hov.png',dpi=dpi)


