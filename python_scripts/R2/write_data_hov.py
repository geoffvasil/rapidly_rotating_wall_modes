import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from dedalus.core import operators
import pickle
import h5py

# Parameters: double cover, so H=0.5
Lz = 1
r0 = 0.2
# empirical R0 for this problem
R0 = 112.95141083744463
R = 2*R0

Nz = 64
Nphi = 128
Nr = 64

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
z, phi, r = dist.local_grids(z_basis, disk_basis, scales=4)

T_wall = dist.Field(bases=(z_basis, disk_basis.edge))

T = []
t = []
for i in range(1, 8):
    f = h5py.File('slices/slices_s%i.h5' %i)
    dset = f['tasks/T(r=0.99)']
    for j in range(len(dset.dims[0]['sim_time'])):
        T_wall.preset_scales(1)
        T_wall['g'] = dset[j]
        T_wall.change_scales(4)
        T.append(T_wall['g'][63,:,0])
        t.append(np.array(dset.dims[0]['sim_time'][j]))

T = np.array(T)
t = np.array(t)

data = {'T_hov':T, 't':t, 'phi':phi} 

f = open('data_hov.pkl', 'wb')
pickle.dump(data, f)
f.close()
