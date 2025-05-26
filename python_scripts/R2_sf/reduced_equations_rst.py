import numpy as np
import dedalus.public as d3
import logging
import h5py
logger = logging.getLogger(__name__)
from dedalus.core import operators

# Parameters: double cover, so H=0.5
Lz = 1
r0 = 0.2
# empirical R0 for this problem
R0 = 112.95141083744463
R = 2*R0
Pr = 1

Nz = 64
Nphi = 128
Nr = 64

dealias = 1
stop_sim_time = 1.5
timestepper = d3.SBDF2
safety = 0.2
max_timestep = safety * Lz / Nz * 0.1
dtype = np.float64
mesh = [8, 8]

# Bases
z_coord = d3.Coordinate('z')
polar_coords = d3.PolarCoordinates('phi', 'r')
cyl_coords = [z_coord, polar_coords]
dist = d3.Distributor(cyl_coords, dtype=dtype, mesh=mesh)
z_basis = d3.Fourier(z_coord, size=Nz, bounds=(0, Lz), dealias=dealias, dtype=dtype)
disk_basis = d3.DiskBasis(polar_coords, shape=(Nphi, Nr), radius=r0, dealias=dealias, dtype=dtype)
z, phi, r = dist.local_grids(z_basis, disk_basis)

# Fields
T = dist.Field(bases=(z_basis, disk_basis))
p = dist.Field(bases=(z_basis, disk_basis))

qphi = dist.Field(bases=(z_basis, disk_basis.edge))
qz = dist.Field(bases=(z_basis, disk_basis.edge))

tau_T = dist.Field(bases=(z_basis, disk_basis.edge))
tau_p1 = dist.Field(bases=disk_basis.edge)
tau_p2 = dist.Field(bases=disk_basis.edge)

K = dist.Field(bases=(z_basis))
K['g'] = sum(np.sin(2*k*np.pi*z/Lz)/(k*np.pi/Lz) for k in range(1,Nz//2))
K = d3.Grid(K)

# Substitutions
lift_basis = disk_basis.derivative_basis(2)
lift = lambda A: d3.Lift(A, lift_basis, -1)

lift_basis2 = disk_basis.derivative_basis(4)
lift2 = lambda A, n: d3.Lift(A, lift_basis2, n)

rvec = dist.VectorField(polar_coords, disk_basis.radial_basis)
rvec['g'][1] = r

dz = lambda A: d3.Differentiate(A, z_coord)
dphi = lambda A: d3.Differentiate(A, polar_coords[0])/r0
dr = lambda A: d3.Differentiate(A, polar_coords[1])
Grad = lambda A: d3.grad(A, coordsys=polar_coords)

# curl(A z-hat) = - z-hat x grad(A)
curl = lambda A: -d3.skew(Grad(A))
H = lambda A: operators.HilbertTransform(A, z_coord)

U = -curl(p)
T = dz(p)
zeta = d3.div(Grad(p))
qphi = -H(qz)

vars = [p, qz]
taus = [tau_T, tau_p1, tau_p2]

# p is cos only
p.valid_modes[1::2] = False
# T is sin only
tau_T.valid_modes[::2] = False
# dr(T) is sin only
# qz is sin only
qz.valid_modes[::2] = False

problem = d3.IVP(vars+taus, namespace=locals())

problem.add_equation("dt(T) - div(Grad(T)) - dz(dz(T)) + lift(tau_T) = -U@Grad(T)", condition='nz!=0')
problem.equations[-1]["valid_modes"][::2] = False # sin only
problem.add_equation("-(rvec@Grad(T))(r=r0)/r0 + R*qz = qphi*dphi(T(r=r0)) + qz*dz(T(r=r0))", condition='nz!=0')
problem.equations[-1]["valid_modes"][::2] = False # sin only
problem.add_equation("dphi(qphi) + dz(qz) - (rvec@U)(r=r0)/r0 = 0", condition='nz!=0')
problem.equations[-1]["valid_modes"][1::2] = False # cos only

problem.add_equation("dt(zeta) - Pr*div(Grad(zeta)) + lift2(tau_p1, -1) + lift2(tau_p2, -2) = -div(U*zeta)", condition='nz==0')
problem.add_equation("p(r=r0) = 0", condition='nz==0')
problem.add_equation("2*Pr*(rvec@Grad(p))(r=r0)/r0 = - ( 3*qphi*dphi(qphi) - qphi*dphi(p(r=r0)) )", condition='nz==0')

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
f = h5py.File('snapshots_s22.h5')
p.load_from_hdf5(f, 0, task='<Field 22285498504944>')
dt = f['scales/timestep'][0]/10

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots_rst', sim_dt=0.01, max_writes=1, parallel='gather')
snapshots.add_tasks(solver.state)

slices = solver.evaluator.add_file_handler('slices_rst', sim_dt=0.0005, max_writes=100, parallel='gather')
slices.add_task(d3.integ(p, 'z')/Lz, name='<p>')
slices.add_task(T(z=0.25), name='T(z=0.25)')
slices.add_task(T(z=0), name='T(z=0)')
slices.add_task(T(phi=0), name='T(phi=0)')
slices.add_task(T(phi=np.pi), name='T(phi=pi)')
slices.add_task(T(r=0.95*r0), name='T(r=0.95)')
slices.add_task(T(r=0.99*r0), name='T(r=0.99)')

# CFL
CFL = d3.CFL(solver, initial_dt=dt, cadence=1, safety=safety, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(U)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((T**2)/2, name='PE')
flow.add_property((U@U)/2, name='KE')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e, max(T)=%e, max(U)=%e' %(solver.iteration, solver.sim_time, timestep, np.sqrt(flow.max('PE')), np.sqrt(flow.max('KE'))))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

