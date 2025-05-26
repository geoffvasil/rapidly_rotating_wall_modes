
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


t_mar, b_mar, l_mar, r_mar = (0.05, 0.27, 0.25, 0.05)
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
A = np.pi*0.2**2

Nu_NS = []
R_NS = []

for i, R in zip(['1p01','1p02','1p05','1p1','1p2','1p5'],[1.01,1.02,1.05,1.1,1.2,1.5]):
    data = pickle.load(open('R%s/data.pkl' %i, 'rb'))
    # normalize by A and R
    Nu_NS.append(-data['F_d'][0,0,0]/A/112.95141083744463/R)
    R_NS.append(R)

for i in range(2,7):
    data = pickle.load(open('R%i/data.pkl' %i, 'rb'))
    # normalize by A and R
    Nu_NS.append(-data['F_d'][0,0,0]/A/112.95141083744463/i)
    R_NS.append(i)

Nu_SF = []
R_SF = []
for i in range(2, 5):
    data = pickle.load(open('R%i_sf/data.pkl' %i, 'rb'))
    # normalize by A and R
    Nu_SF.append(-data['F_d'][0,0,0]/A/112.95141083744463/i)
    R_SF.append(i)

R_NS = np.array(R_NS)
Nu_NS = np.array(Nu_NS)
Nu_SF = np.array(Nu_SF)

slice_axes.scatter(R_NS, Nu_NS+1, marker='x', color='MidnightBlue', label=r'${\rm no} \ {\rm slip}$')
slice_axes.scatter(R_SF, Nu_SF+1, marker='*', color='Firebrick', label=r'${\rm stress-free}$')
slope = (Nu_NS[0])/0.01
print(slope)
slice_axes.plot([1, 1.75], [1, 1+slope*0.75], color='k', linestyle='--')
slice_axes.set_ylim([1, None])

lg = slice_axes.legend()
lg.draw_frame(False)

slice_axes.set_ylabel(r'$Nu$')
slice_axes.set_xlabel(r'$R/R_c$')
slice_axes.yaxis.set_major_locator(ticker.FixedLocator([1, 2, 3, 4]))
slice_axes.yaxis.set_major_formatter(ticker.FixedFormatter([r'$1$',r'$2$',r'$3$',r'$4$']))
slice_axes.xaxis.set_major_locator(ticker.FixedLocator([1, 2, 3, 4, 5, 6]))
slice_axes.xaxis.set_major_formatter(ticker.FixedFormatter([r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$']))

plt.savefig('figures/Nu_R.pdf')

