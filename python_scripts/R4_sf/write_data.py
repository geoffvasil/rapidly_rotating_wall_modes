import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from dedalus.core import operators
import pickle
import h5py

# Parameters
Lz = 1
r0 = 0.2
# empirical R0 for this problem
R0 = 112.95141083744463
R = 2*R0

Nz = 96
Nphi = 512
Nr = 192

dealias = 1
stop_sim_time = 1.5
timestepper = d3.SBDF2
safety = 0.2
max_timestep = safety * Lz / Nz * 0.1
dtype = np.float64
mesh = None

# Bases
z_coord = d3.Coordinate('z')
polar_coords = d3.PolarCoordinates('phi', 'r')
cyl_coords = [z_coord, polar_coords]
dist = d3.Distributor(cyl_coords, dtype=dtype, mesh=mesh)
z_basis = d3.Fourier(z_coord, size=Nz, bounds=(0, Lz), dealias=dealias, dtype=dtype)
disk_basis = d3.DiskBasis(polar_coords, shape=(Nphi, Nr), radius=r0, dealias=dealias, dtype=dtype)
z, phi, r = dist.local_grids(z_basis, disk_basis, scales=1)
z_hres = dist.local_grid(z_basis, scale=4)

f = h5py.File('snapshots_rst4/snapshots_rst4_s23.h5')
p = dist.Field(bases=(z_basis, disk_basis))
p.load_from_hdf5(f, 0, task='<Field 22492311246144>')
qz = dist.Field(bases=(z_basis, disk_basis.edge))
qz.load_from_hdf5(f, 0, task='<Field 22492311249600>')
qphi = dist.Field(bases=(z_basis, disk_basis.edge))
qphi.load_from_hdf5(f, 0, task='<Field 22492311246240>')

dz = lambda A: d3.Differentiate(A, z_coord)
dphi = lambda A: d3.Differentiate(A, polar_coords[0])/r0
T = dz(p)

divq = dphi(qphi) + dz(qz)
divq = divq.evaluate()
divq.change_scales(1)
divq = divq['g']

curlq = dphi(qz) - dz(qphi)
curlq = curlq.evaluate()
curlq.change_scales(1)
curlq = curlq['g']

T_wall = T(r=r0).evaluate()
#T_wall.change_scales((4, 4, 4))
T_wall = T_wall['g']
p_wall = p(r=r0).evaluate()['g']
T_mid = T(z=0.25).evaluate()['g']
p_top = p(z=0.5).evaluate()
p_top = p_top['g']
T_int = np.sum(T['g'][:Nz//2],axis=0)

u = d3.grad(p, polar_coords).evaluate()
#u.change_scales(4)
u = u['g']
uphi_axi = np.mean(u[1], axis=1)

conv_flux = (qz*T(r=r0)).evaluate()
conv_flux.change_scales((4,1,1))

conv_flux = np.sum(conv_flux['g']*r0*2*np.pi/Nphi, axis=1)

cond_flux = d3.integ(dz(T), polar_coords).evaluate()
cond_flux.change_scales((4,1,1))
cond_flux = cond_flux['g']

data = {'T_wall':T_wall, 'p_wall':p_wall, 'T_mid':T_mid, 'p_top':p_top, 'T_int': T_int, 'F_c':conv_flux, 'F_d':cond_flux, 'qz': qz['g'], 'qphi':qphi['g'], 'divq':divq, 'curlq':curlq, 'uphi_axi':uphi_axi, 'z':z, 'z_hres':z_hres, 'r': r, 'phi':phi}

f = open('data.pkl', 'wb')
pickle.dump(data, f)
f.close()

