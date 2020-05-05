"""
Dedalus script for 2D Rayleigh-Benard convection.

This script uses a Fourier basis in the x direction with periodic boundary
conditions.  The equations are scaled in units of the buoyancy time (Fr = 1).

This script can be ran serially or in parallel, and uses the built-in analysis
framework to save data snapshots in HDF5 files.  The `merge_procs` command can
be used to merge distributed analysis sets from parallel runs, and the
`plot_slices.py` script can be used to plot the snapshots.

To run, merge, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ mpiexec -n 4 python3 plot_slices.py snapshots/*.h5

This script can restart the simulation from the last save of the original
output to extend the integration.  This requires that the output files from
the original simulation are merged, and the last is symlinked or copied to
`restart.h5`.

To run the original example and the restart, you could use:
    $ mpiexec -n 4 python3 rayleigh_benard.py
    $ mpiexec -n 4 python3 -m dedalus merge_procs snapshots
    $ ln -s snapshots/snapshots_s2.h5 restart.h5
    $ mpiexec -n 4 python3 rayleigh_benard.py

The simulations should take a few process-minutes to run.

"""

import numpy as np
from mpi4py import MPI
import time
import pathlib

from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)


# Parameters done
Lx, Lz = (20., 1.)
q0val= 0.000
T1ovDTval = 5.5
betaval =1.201

# Create bases and domain done 
x_basis = de.Fourier('x', 768, interval=(0, Lx), dealias=2/3)
z_basis = de.Chebyshev('z', 129, interval=(0, Lz), dealias=2/3)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)


# 2D Boussinesq hydrodynamics
problem = de.IVP(domain, variables=['p','b','u','w','bz','uz','wz','temp','q','qz']) #done
################################################################
problem.meta['p','b','u','w']['z']['dirichlet'] = True #see if keeping this in changes anything 
################################################################
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
#done

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
#done


problem.add_bc("left(b) = T1ovDT")
problem.add_bc("left(q) = K2*exp(aDT*T1ovDT)")
problem.add_bc("left(u) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("left(temp)=T1ovDT")
problem.add_bc("right(b) = T1ovDT-1.0+beta")
problem.add_bc("right(q) = K2*exp(aDT*(T1ovDT-1.0))")
problem.add_bc("right(u) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")  #maybe dx??

#done

#problem.expand(domain)

# Build solver
solver = problem.build_solver(de.timesteppers.SBDF3)
logger.info('Solver built')
#done

# Initial conditions

x = domain.grid(0)
z = domain.grid(1)
b = solver.state['b']
bz = solver.state['bz']
q = solver.state['q']
qz = solver.state['qz']

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
#pert =  1e-3 * np.random.standard_normal(domain.local_grid_shape) * (zt - z) * (z - zb)
#b['g'] = -0.0*(z - pert)
b['g'] = T1ovDTval-(1.00-betaval)*z
b.differentiate('z', out=bz)
#q['g'] = q0val*np.exp(-betaval*z/T0val)+1e-2*np.exp(-((x-1.0)/0.01)^2)*np.exp(-((z-0.5)/0.01)^2)
#q['g'] = q0val*np.exp(-betaval*z/T0val)+(1e-2)*np.exp(-((z-0.5)*(z-0.5)/0.02))*np.exp(-((x-1.0)*(x-1.0)/0.02))
q['g'] = (2e-2)*np.exp(-((z-0.1)*(z-0.1)/0.005))*np.exp(-((x-1.0)*(x-1.0)/0.005))
q.differentiate('z', out=qz)
#done

# Integration parameters
dt = 1e-6
solver.stop_sim_time = 6.5
solver.stop_wall_time = 6000 * 60.
solver.stop_iteration = np.inf


# Analysis
snapshots = evaluator.add_file_handler('snapshots', sim_dt=0.005, max_writes=50)
snapshots.add_task("q")
snapshots.add_task("b")
snapshots.add_task("temp")
snapshots.add_task("u")
snapshots.add_task("w")

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=100, safety=0.3,
                     max_change=1.5, min_change=0.5, max_dt=0.00005, threshold=0.05)
CFL.add_velocities(('u', 'w'))

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=100)
flow.add_property("sqrt(u*u + w*w) / R", name='Re')

# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.proceed:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10000 == 0:
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))
            logger.info('Max Re = %f' %flow.max('Re'))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
