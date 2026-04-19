"""
Discretization and modelling of a 2D convection-diffusion-decay PDE

Governing eq: 
del_c/del_t + v_x*del_c/del_x = D_x*del^2_c/del_x^2 + D_y*del^2_c/del_y^2 - kc

Determines the concentration of a contaminant wrt x, y, and t in an aquifer 
before reaching a creek 
"""


import numpy as np
import matplotlib.pyplot as plt

# Problem parameters

L = 100     #m
W = 60      #m
Dx = 687.2  #m**2/yr
Dy = 34.89  #m**2/yr
vx = 34.69  #m**2/yr
k = 4.6     #yr**-1
cmax = 13.68 #mg/L
wc = 4.25   #m
tauc = 40.0 #yr

###############################################################################

# Task 3

# Defining the governing PDE for concentration at inflow boundary
# Dirichlet conditions for c(0,y,t) at x = 0

def c_0(y, t):
    return 0.5*cmax*np.exp(-t/tauc) * (
    np.tanh(10.0*((y - W/2.0) + wc/2.0)) - np.tanh(10.0*((y - W/2.0) - wc/2.0))
    )

# Stable timestep using CFL=0.45
def dt_stable(dx, dy, Dx, Dy, vx):
    dy = dx
    CFL = 0.45
    dt_par = (Dx/dx**2)+(Dy/dy**2)+(vx/dx)
    dt_stable =CFL/dt_par
    
    return dt_stable

###############################################################################

# Task 4
# Advance the stable timestep by one iteration
def advance_dt(c, t, dt, dx, dy, Dx, Dy, vx, k, W):
    ny, nx = c.shape

    rx  = Dx * dt / dx**2
    ry  = Dy * dt / dy**2
    s = vx * dt / dx
    l = k  * dt

    cnew = c.copy()

    # Update boundary conditions
    c_ij   = c[1:-1, 1:-1]
    c_im1j = c[1:-1, 0:-2]
    c_ip1j = c[1:-1, 2:  ]
    c_ijm1 = c[0:-2, 1:-1]
    c_ijp1 = c[2:  , 1:-1]

    # Update interior node 
    cnew[1:-1, 1:-1] = (
        (1 - 2*rx - 2*ry - s - l)*c_ij
        + (rx + s)*c_im1j
        +  rx       *c_ip1j
        +  ry       *c_ijm1
        +  ry       *c_ijp1
    )

    # boundaries at t+dt
    y = np.linspace(0, W, ny)

    cnew[:, 0] = c_0(y, t+dt)
    cnew[:, -1] = cnew[:, -2]
    cnew[0, :]  = 2*cnew[1, :]  - cnew[2, :]
    cnew[-1, :] = 2*cnew[-2, :] - cnew[-3, :]

    return cnew

###############################################################################

#Tasks 5 & 6

tmax   = 2.5   #yr
tplot  = 0.5   #yr

# Uncomment different lines for diff grid spacing
#Number of nodes in the x and y directions 
nx, ny = 201, 121   # 0.5m 
#nx, ny = 101, 61   # 1m 
#nx, ny = 51, 31    # 2m
#nx, ny = 401, 241  # 0.25m

def run_scenario(nx=201, ny=121, tmax=2.5, tplot=0.5,
                 Dx=Dx, Dy=Dy, vx=vx, k=k, W=W):

    x = np.linspace(0, L, nx)
    dx = x[1] - x[0]

    y = np.linspace(0, W, ny)
    dy = y[1] - y[0]

    dt = dt_stable(dx, dy, Dx, Dy, vx)
    print("Stable dt =", dt)

    c = np.zeros((ny, nx))
    t = 0.0
    next_plot = 0.0
    times, fields = [], []

    j_mid = np.argmin(np.abs(y - W/2))
    i_right = nx - 1
    mid_series = []

    while t < tmax - 1e-14:

        if t + 1e-12 >= next_plot:
            times.append(next_plot)
            fields.append(c.copy())
            next_plot += tplot

        dt_eff = min(dt, tmax - t)
        c = advance_dt(c, t, dt_eff, dx, dy, Dx, Dy, vx, k, W)
        t += dt_eff

        mid_series.append((t, c[j_mid, i_right]))

    if abs(times[-1] - tmax) > 1e-10:
        times.append(tmax)
        fields.append(c.copy())

    return x, y, np.array(times), fields, np.array(mid_series)


# Contour plots for the progression of c in domain
# Plot every 0.5 years for a total of 2.5 years 
def plot_contours(x, y, times, fields, vmin=0, vmax=cmax, nlevels=20):
    X, Y = np.meshgrid(x, y)
    levels = np.linspace(vmin, vmax, nlevels)

    for t, C in zip(times, fields):
        plt.figure()
        cs = plt.contourf(X, Y, C, levels=levels, vmin=vmin, vmax=vmax)
        plt.colorbar(cs)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title(f"Concentration, t={t:.2f} yr")
        fname = f"contour_t_{t:.2f}yr.png"
        plt.savefig(fname, dpi=150)
        print("Saved:", fname)
        plt.show()

#Plot c at midpoint x_mid
def plot_midpoint_c(series, label, fname):
    plt.figure()
    plt.plot(series[:,0], series[:,1], '.-', label=label)
    plt.axhline(0.95, linestyle='--')
    plt.xlabel("t (yr)")
    plt.ylabel("c at (x=L, y=W/2)")
    plt.grid()
    plt.legend()
    plt.savefig(fname, dpi=150)
    print("Saved:", fname)
    plt.show()

# Determining if concentration reaches max
def report_peak(tag, series):
    cM = series[:,1].max()
    tM = series[series[:,1].argmax(), 0]
    flag = "exceeds" if cM > 0.95 else "does NOT exceed"
    print(f"[{tag}] Max = {cM:.3f} at t={tM:.3f} → {flag} threshold")

###############################################################################

if __name__ == "__main__":

    x, y, times, fields, mid_base = run_scenario()

    plot_contours(x, y, times, fields)
    plot_midpoint_c(mid_base, "Baseline", "series_baseline.png")
    report_peak("Baseline", mid_base)

    x, y, _, _, mid_d2 = run_scenario(Dx=2*Dx, Dy=2*Dy)
    plot_midpoint_c(mid_d2, "2x Diffusion", "series_2x_diffusion.png")
    report_peak("Diffusion", mid_d2)

    x, y, _, _, mid_v2 = run_scenario(vx=2*vx)
    plot_midpoint_c(mid_v2, "2x Convection", "series_2x_convection.png")
    report_peak("Convection", mid_v2)

    x, y, _, _, mid_k05 = run_scenario(k=0.5*k)
    plot_midpoint_c(mid_k05, "0.5x Decay", "series_05x_decay.png")
    report_peak("Decay", mid_k05)
