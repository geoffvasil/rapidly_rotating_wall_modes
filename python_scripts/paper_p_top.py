
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import h5py
import publication_settings
import pickle
from dedalus.extras import plot_tools
import brewer2mpl
import dedalus.public as de

matplotlib.rcParams.update(publication_settings.params)

color_map = ('PuOr', 'diverging',11)
b2m = brewer2mpl.get_map(*color_map, reverse=True)
cmap1 = b2m.mpl_colormap

dpi = 300

t_mar, b_mar, l_mar, r_mar = (0.4, 0.1, 0.1, 0.35)
h_slice, w_slice = (1., 1)
h_pad = 0.2


h_cbar, w_cbar = (0.05, w_slice)
h_cbpad = h_cbar
w_pad = 0.5

h_total = t_mar + 2*h_pad + 3*h_cbar + 3*h_cbpad + 3*h_slice + b_mar
w_total = l_mar + 1*w_pad + 2*w_slice + r_mar

width = 5.5
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# slices
slice_axes = []
for i in range(2):
  for j in range(3):
    left = (l_mar + i*w_pad + i*w_slice) / w_total
    bottom = 1 - (t_mar + (j+1)*h_cbar + (j+1)*h_cbpad + j*h_pad + (j+1)*h_slice ) / h_total
    width = w_slice / w_total
    height = h_slice / h_total
    slice_axes.append(fig.add_axes([left, bottom, width, height]))

# cbars
cbar_axes = []
for i in range(2):
  for j in range(3):
    left = (l_mar + i*w_pad + i*w_slice) / w_total
    bottom = 1 - (t_mar + (j+1)*h_cbar + j*h_cbpad + j*h_pad + j*h_slice ) / h_total
    width = w_cbar / w_total
    height = h_cbar / h_total
    cbar_axes.append(fig.add_axes([left, bottom, width, height]))

# load slice data
R_list = [2, 3, 4]
SF_list = [False, True]

c_im = []
cbars = []
for i in range(2):
    for j in range(3):
        SF = SF_list[i]
        R = R_list[j]

        filename = 'R'+str(R)
        if SF:
            filename += '_sf'

        data = pickle.load(open(filename + '/data.pkl', 'rb'))
        T = data['T_wall'][:,:,0]
        shift = np.argmin(T[1,:])
        p_old = data['p_top'][0,:,:]
        #p_old = np.vstack((p_old, p_old[0,:][None,:]))
        p = np.copy(p_old)
        p[-shift:, :] = p_old[:shift, :]
        p[:-shift, :] = p_old[shift:, :]
        r = data['r'][0,0,:]
        r[0] = 0
        r[-1] = 0.2
        phi = data['phi'][0,:,0]

        phim, rm = plot_tools.quad_mesh(phi, r)
        xm, ym = rm*np.cos(phim), rm*np.sin(phim)
        c_im.append(slice_axes[3*i+j].pcolormesh(xm,ym,p.T,cmap=cmap1))
        slice_axes[3*i+j].axis('off')
        phi = np.concatenate([phi, [2*np.pi]])
        slice_axes[3*i+j].plot(0.2*np.cos(phi), 0.2*np.sin(phi), color='k', linewidth=2, clip_on=False)

        cbars.append(fig.colorbar(c_im[3*i+j], cax=cbar_axes[3*i+j], orientation='horizontal', ticks=MaxNLocator(nbins=4)))

for i in range(6):
    cbar_axes[i].xaxis.set_ticks_position('top')
    cbar_axes[i].xaxis.set_label_position('top')
    cbars[i].ax.tick_params(labelsize=8)
    for tick in cbars[i].ax.get_xticklabels():
        tick.set_fontname("stix")
    if i % 3 == 0:
        cbar_axes[i].text(0.5,5,r'$p(z=H)$',va='center',ha='center',fontsize=10,transform=cbar_axes[i].transAxes)

cbar_axes[0].text(0.5,8,r'${\rm no} \ {\rm slip}$',va='center',ha='center',fontsize=10,transform=cbar_axes[0].transAxes)
cbar_axes[3].text(0.5,8,r'${\rm stress-free}$',va='center',ha='center',fontsize=10,transform=cbar_axes[3].transAxes)

for i in range(3, 6):
    slice_axes[i].text(1.2,0.5,r'$R=%iR_c$' %(i-1),va='center',ha='center',fontsize=10,transform=slice_axes[i].transAxes)

plt.savefig('figures/p_top.png', dpi=dpi)


