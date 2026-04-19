# Fuel-Spill-Fate-and-Transport
Convection–Diffusion–Decay Modelling
File:
- fuel_spill_and_transport.py       : Code containing functions for dt, one-step update, and concentration simulation.

How to reproduce (Python 3 + numpy + matplotlib):
1) Import functions from afuel_spill_and_transport.py and call run_simulation with the parameters from the brief.
2) Run python fuel_spill_and_transport.py
3) For tasks 5 & 6, number of nodes are defined in lines 99-102 and can be uncommented or commented to determine the variation in the simulation due to the timestep. 
4) Contour plots and midpoint graphs for initial conditions and non-conservative conditions are 

Baseline settings used in this solution:
  L=100.0 m, W=60.0 m, Dx=687.2 m^2/yr, Dy=34.89 m^2/yr, vx=34.69 m/yr, k=4.6 1/yr,
  cmax=13.68 mg/L, wc=4.25 m, tau_c=40.0 yr, dx=dy=1.0 m, tmax=2.5 yr, tplot=0.5 yr.

Notes:
- Time step was chosen via the conservative CFL expression.
- Upwind in x for convection (vx>0), central for diffusion, forward Euler in time.
- Boundary conditions: Dirichlet at x=0; zero-gradient at x=L; zero second-derivative (no-flux) at y=0, y=W.
