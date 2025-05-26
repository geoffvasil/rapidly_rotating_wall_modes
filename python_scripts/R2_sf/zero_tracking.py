
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import h5py
import pickle
from dedalus.extras import plot_tools
from scipy.interpolate import CubicSpline
from scipy.optimize import newton

# load slice data
T = []
t = []
for i in range(1, 3):
    f = h5py.File('slices_rst/slices_rst_s%i.h5' %i)
    dset = f['tasks/T(r=0.99)']
    T.append(np.array(f['tasks/T(r=0.99)'][:,15,:,0]))
    phi = np.array(dset.dims[2]['phi'])
    t.append(np.array(dset.dims[0]['sim_time']))

t = np.hstack(t)
T = np.vstack(T)
Nt = len(t)

phi_p = np.concatenate((phi, [2*np.pi]))

roots = []
for i in range(Nt):
    Ti = T[-1-i]
    T_interp = CubicSpline(phi_p, np.concatenate((Ti, [Ti[0]])), bc_type='periodic')
    if i == 0:
        roots.append(newton(T_interp, np.pi))
    else:
        roots.append(newton(T_interp, roots[-1]))

roots = np.array(roots[::-1])

plt.plot(t, roots)

i_fit = (Nt//4)*3
fit = np.polyfit(t[i_fit:], roots[i_fit:], 1)
plt.plot(t, np.poly1d(fit)(t))
print(fit)

plt.savefig('test.png')

