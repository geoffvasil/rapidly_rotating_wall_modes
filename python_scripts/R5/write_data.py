import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
import h5py
from dedalus.core import operators
import pickle

# Parameters: double cover, so H=0.5
Lz = 1
r0 = 0.2
# empirical R0 for this problem
R0 = 112.95141083744463
R = 5*R0

Nz = 128
Nphi = 512
Nr = 256

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
z, phi, r = dist.local_grids(z_basis, disk_basis)
z_hres = dist.local_grid(z_basis, scale=4)

f = h5py.File('snapshots_rst2/snapshots_rst2_s31.h5')
p = dist.Field(bases=(z_basis, disk_basis))
p.load_from_hdf5(f, 0, task='<Field 22332332398272>')
qz = dist.Field(bases=(z_basis, disk_basis.edge))
qz.load_from_hdf5(f, 0, task='<Field 22332332401344>')

dz = lambda A: d3.Differentiate(A, z_coord)
T = dz(p)

T_wall = T(r=r0).evaluate()['g']
p_wall = p(r=r0).evaluate()['g']
T_mid = T(z=0.25).evaluate()['g']
p_top = p(z=0.5).evaluate()['g']
T_int = np.sum(T['g'][:Nz//2]*0.5/Nz,axis=0)

conv_flux = (qz*T(r=r0)).evaluate()
conv_flux.change_scales((4,1,1))

conv_flux = np.sum(conv_flux['g']*r0*2*np.pi/Nphi, axis=1)

cond_flux = d3.integ(dz(T), polar_coords).evaluate()
cond_flux.change_scales((4,1,1))
cond_flux = cond_flux['g']

data = {'T_wall':T_wall, 'p_wall':p_wall, 'T_mid':T_mid, 'p_top':p_top, 'T_int': T_int, 'F_c':conv_flux, 'F_d':cond_flux, 'z':z, 'z_hres':z_hres, 'r': r, 'phi':phi}

f = open('data.pkl', 'wb')
pickle.dump(data, f)
f.close()
