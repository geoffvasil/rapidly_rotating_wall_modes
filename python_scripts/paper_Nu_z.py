
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


t_mar, b_mar, l_mar, r_mar = (0.22, 0.25, 0.28, 0.05)
h_slice, w_slice = (1., 1./publication_settings.golden_mean)

h_total = t_mar + h_slice + b_mar
w_total = l_mar + w_slice + r_mar

width = 3.4
scale = width/w_total

fig = plt.figure(1, figsize=(scale * w_total,
                             scale * h_total))

# slices
left = (l_mar) / w_total
bottom = 1 - (t_mar + h_slice) / h_total
width = w_slice / w_total
height = h_slice / h_total
slice_axes = fig.add_axes([left, bottom, width, height])

# load data

data = pickle.load(open('R4/data.pkl', 'rb'))
A = np.pi*0.2**2
# normalize by A and R
F_c = data['F_c'][:,0]/A/112.95141083744463/4
F_d = data['F_d'][:,0,0]/A/112.95141083744463/4
z = data['z_hres'][:,0,0]

data = pickle.load(open('R4_sf/data.pkl', 'rb'))
A = np.pi*0.2**2
# normalize by A and R
F2_c = data['F_c'][:,0]/A/112.95141083744463/4
F2_d = data['F_d'][:,0,0]/A/112.95141083744463/4
z2 = data['z_hres'][:,0,0]

Nz = len(z)
z = z[:Nz//2+1]
F_c = F_c[:Nz//2+1]
F_d = F_d[:Nz//2+1]

Nz = len(z2)
z2 = z2[:Nz//2+1]
F2_c = F2_c[:Nz//2+1]
F2_d = F2_d[:Nz//2+1]

slice_axes.plot(F_c, z, linewidth=2, color='MidnightBlue', label=r'$F_c$')
slice_axes.plot(F2_c, z2, linewidth=2, color='MidnightBlue', linestyle='--', label=r'$F_c$')
slice_axes.plot(-F_d, z, linewidth=2, color='Firebrick', label=r'$F_d$')
slice_axes.plot(-F2_d, z2, linewidth=2, color='Firebrick', linestyle='--', label=r'$F_d$')
slice_axes.plot(F_c-F_d, z, linewidth=2, color='DarkGoldenrod', label=r'$F_t$')
slice_axes.plot(F2_c-F2_d, z2, linewidth=2, color='DarkGoldenrod', linestyle='--', label=r'$F_t$')

lg = slice_axes.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3)
lg.draw_frame(False)

slice_axes.set_ylabel(r'$z$')
slice_axes.set_xlabel(r'$Nu-1$')
slice_axes.xaxis.set_major_locator(ticker.FixedLocator([0, 1, 2, 3]))
slice_axes.xaxis.set_major_formatter(ticker.FixedFormatter([r'$0$',r'$1$',r'$2$',r'$3$']))
slice_axes.yaxis.set_major_locator(ticker.FixedLocator([0, 0.25, 0.5]))
slice_axes.yaxis.set_major_formatter(ticker.FixedFormatter([r'$0$',r'$H/2$',r'$H$']))

plt.savefig('figures/Nu_z.pdf')


