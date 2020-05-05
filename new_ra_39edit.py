"""
Simulation script for 2D moist Rayleigh-Benard convection. Eqns are non-dimensional

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `process.py` script in this
folder can be used to merge distributed save files from parallel runs and plot
the snapshots from the command line.

To run, join, and plot using 4 processes, for instance, you could use:
$ mpiexec -n 4 python3 rayleigh_benard.py
$ mpiexec -n 4 python3 process.py join snapshots
$ mpiexec -n 4 python3 process.py plot snapshots/*.h5

On a single process, this should take ~15 minutes to run.

"""

import os
import numpy as np
from mpi4py import MPI
import time

from dedalus import public as de
from dedalus.extras import flow_tools
import logging
logger = logging.getLogger(__name__)

# Parameters
Lx, Lz = (20., 1.)
q0val= 0.000
T1ovDTval = 5.5
betaval =1.201

# Create bases and domain
x_basis = de.Fourier('x',768, interval=(0, Lx), dealias=2/3) #'x'
z_basis = de.Chebyshev('z',129, interval=(0, Lz), dealias=2/3) #'z'
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64) #mesh = [1]?

# 2D Boussinesq hydrodynamics
problem = de.IVP(domain,
                           variables=['p','b','u','w','bz','uz','wz','temp','q','qz'])
                           #param_names=['Eu','Prandtl','Ra','M','tau','S','K2','aDT','T1ovDT','beta','T1','deltaT'])
# introduce parameters
problem.parameters['Eu'] = 1.0
problem.parameters['Prandtl'] = 1.0
problem.parameters['Ra'] = 1000000.0
problem.parameters['M'] = 50.0
problem.parameters['S'] = 1.0
problem.parameters['beta']=betaval
problem.parameters['K2'] = 4e-10
problem.parameters['tau'] = 0.00005
problem.parameters['aDT'] = 3.00
#problem.parameters['aDT'] = 2.86
problem.parameters['T1ovDT'] = T1ovDTval
problem.parameters['T1'] = T1ovDTval
problem.parameters['deltaT'] = 1.00
#problem.substitutions['q_diff'] = "q-b"

problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(b) - (dx(dx(b)) + dz(bz))             = - u*dx(b) - w*bz+M*0.5*(1.0+tanh(100000.*(q-K2*exp(aDT*temp))))*(q-K2*exp(aDT*temp))/tau")
problem.add_equation("dt(q) - S*(dx(dx(q)) + dz(qz))             = - u*dx(q) - w*qz-0.5*(1.0+tanh(100000.*(q-K2*exp(aDT*temp))))*(q-K2*exp(aDT*temp))/tau")
problem.add_equation("dt(u) - Prandtl*(dx(dx(u)) + dz(uz)) + Eu*dx(p)     = - u*dx(u) - w*uz")
problem.add_equation("dt(w) - Prandtl*(dx(dx(w)) + dz(wz)) + Eu*dz(p) - Prandtl*Ra*b = - u*dx(w) - w*wz")
problem.add_equation("bz - dz(b) = 0")
problem.add_equation("qz - dz(q) = 0")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_equation("dz(temp)-bz = -beta")
problem.add_bc("left(b) = T1ovDT")
problem.add_bc("left(q) = K2*exp(aDT*T1ovDT)")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("left(temp)=T1ovDT")
problem.add_bc("right(b) = T1ovDT-1.0+beta")
problem.add_bc("right(q) = K2*exp(aDT*(T1ovDT-1.0))")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")







#problem.expand(domain) #possibly need to delete, leave in for now 

# Build solver
ts = de.timesteppers.SBDF3
#solver = de.solvers.IVP(problem, domain, ts)
#logger.info('Solver built')

solver = problem.build_solver(ts)
logger.info('Solver built')

# Initial conditions #unchanged
x = domain.grid(0)
z = domain.grid(1)
b = solver.state['b']
bz = solver.state['bz']
q = solver.state['q']
qz = solver.state['qz']


# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
print('this is:' +str( domain.local_grid_shape))
#pert =  1e-3 * np.random.standard_normal(domain.local_grid_shape) * (zt - z) * (z - zb) #commented out because it was never used again
#b['g'] = -0.0*(z - pert)
b['g'] = T1ovDTval-(1.00-betaval)*z
b.differentiate('z', out=bz)
#q['g'] = q0val*np.exp(-betaval*z/T0val)+1e-2*np.exp(-((x-1.0)/0.01)^2)*np.exp(-((z-0.5)/0.01)^2)
#q['g'] = q0val*np.exp(-betaval*z/T0val)+(1e-2)*np.exp(-((z-0.5)*(z-0.5)/0.02))*np.exp(-((x-1.0)*(x-1.0)/0.02))
q['g'] = (2e-2)*np.exp(-((z-0.1)*(z-0.1)/0.005))*np.exp(-((x-1.0)*(x-1.0)/0.005))
q.differentiate('z', out=qz)

# Integration parameters
dt = 1e-6
solver.stop_sim_time = 6.5
solver.stop_wall_time = 6000 * 60.
solver.stop_iteration = np.inf

CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=100, safety=0.3,
                     max_change=1.5, min_change=0.5, max_dt=0.125, threshold=0.05)
CFL.add_velocities(('u', 'w'))
# CFL routines
evaluator = solver.evaluator
evaluator.vars['grid_delta_x'] = domain.grid_spacing(0)
evaluator.vars['grid_delta_z'] = domain.grid_spacing(1)

cfl_cadence = 100
#cfl_variables = evaluator.add_dictionary_handler(iter=cfl_cadence)
#cfl_variables.add_task('u/grid_delta_x', name='f_u')
#cfl_variables.add_task('w/grid_delta_z', name='f_w')
#cfl_variables.add_task('b', name='f_b')

#def cfl_dt():
 #   if z.size > 0:
 #       max_f_u = np.max(np.abs(cfl_variables.fields['f_u']['g']))
  #      max_f_w = np.max(np.abs(cfl_variables.fields['f_w']['g']))
  #  else:
   #     max_f_u = max_f_w = 0
   # max_f = max(max_f_u, max_f_w)
   # if max_f > 0:
#        min_t = 1 / max_f
   # else:
    #    min_t = np.inf
   # return min_t

#safety = 0.3
#dt_array = np.zeros(1, dtype=np.float64)
#def update_dt(dt):
 #   new_dt = min(max(0.5*dt, min(safety*cfl_dt(), 1.01*dt)),0.00005)
  #  if domain.distributor.size > 1:
   #     dt_array[0] = new_dt
    #    domain.distributor.comm_cart.Allreduce(MPI.IN_PLACE, dt_array, op=MPI.MIN)
     #   new_dt = dt_array[0]
   # return new_dt

# Analysis
snapshots = evaluator.add_file_handler('snapshots', sim_dt=0.005, max_writes=50)
snapshots.add_task("q")
snapshots.add_task("b")
snapshots.add_task("temp")
snapshots.add_task("u")
snapshots.add_task("w")

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        solver.step(dt)

        if (solver.iteration - 1) % cfl_cadence == 0:
            dt =CFL.compute_dt()# update_dt(dt)
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()

    # Print statistics
    logger.info('Run time: %f' %(end_time-start_time))
    logger.info('Iterations: %i' %solver.iteration)
