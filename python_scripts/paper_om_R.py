
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


t_mar, b_mar, l_mar, r_mar = (0.05, 0.27, 0.45, 0.05)
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
R_NS = np.array([1.01, 1.02, 1.05, 1.1, 1.2, 1.5, 2, 3, 4, 5, 6])
R_SF = np.array([2, 3, 4])

om_NS = np.array([-139.5, -141.0, -145.2, -151.9, -163.6, -188.3, -191.8, -121.4, -56.6, -5.6, 40.8])/(-2)/69.027
om_SF = np.array([-218.2, -306.4, -430.0])/(-2)/69.027

slice_axes.scatter(R_NS, om_NS, marker='x', color='MidnightBlue', label=r'${\rm no} \ {\rm slip}$')
slice_axes.scatter(R_SF, om_SF, marker='*', color='Firebrick', label=r'${\rm stress-free}$')
slope = (om_NS[0]-1)/0.01
print(slope)
slice_axes.plot([1, 1.45], [1, 1+slope*0.45], color='k', linestyle='--')
slice_axes.axhline(0, linestyle='--', linewidth=0.5, color='k', zorder=-1)

lg = slice_axes.legend(loc='upper left')
lg.draw_frame(False)

slice_axes.set_ylabel(r'$\omega/\omega_c$')
slice_axes.set_xlabel(r'$R/R_c$')
slice_axes.yaxis.set_major_locator(ticker.FixedLocator([0, 1, 2, 3]))
slice_axes.yaxis.set_major_formatter(ticker.FixedFormatter([r'$0$',r'$1$',r'$2$',r'$3$']))
slice_axes.xaxis.set_major_locator(ticker.FixedLocator([1, 2, 3, 4, 5, 6]))
slice_axes.xaxis.set_major_formatter(ticker.FixedFormatter([r'$1$', r'$2$',r'$3$',r'$4$',r'$5$',r'$6$']))

plt.savefig('figures/om_R.pdf')

