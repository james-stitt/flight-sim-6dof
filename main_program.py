import matplotlib.pyplot as plt
import numpy as np

from numerical_integrators import numerical_integration_methods
from governing_equations import flat_earth_eom

# =============================================================================
# Part 1: Initialization of simulation
# =============================================================================

# Set initial conditions (these conditions may be loaded from an aircraft
# trim routine in future versions of the code)
u0_bf_mps  = []
v0_bf_mps  = []
w0_bf_mps  = []
p0_bf_rps  = []
q0_bf_rps  = []
r0_bf_rps  = []
phi0_rad   = []
theta0_rad = []
psi0_rad   = []
p10_n_m    = []
p20_n_m    = []
p30_n_m    = []

# Assign initial conditions to an array
x0 = np.array([
    u0_bf_mps,   # x-axis body-fixed velocity (m/s)
    v0_bf_mps,   # y-axis body-fixed velocity (m/s)
    w0_bf_mps,   # z-axis body-fixed velocity (m/s)
    p0_bf_rps,   # roll rate (rad/s)
    q0_bf_rps,   # pitch rate (rad/s)
    r0_bf_rps,   # yaw rate (rad/s)
    phi0_rad,    # roll angle (rad)
    theta0_rad,  # pitch angle (rad)
    psi0_rad,    # yaw angle (rad)
    p10_n_m,     # x-axis position (N*m)
    p20_n_m,     # y-axis position (N*m)
    p30_n_m,     # z-axis position (N*m)
])

# Make the initial condition array a column vector
x0 = x0.transpose(); 
nx0 = x0.size

# Set time conditions
t0_s = 0.0
tf_s = 10.0
h_s = 0.01

# =============================================================================
# Part 2: Numerically approximate solutions to the governing equations
# =============================================================================

# Preallocate the solution array
t_s = np.arange(t0_s, tf_s + h_s, h_s); 
nt_s = t_s.size
x = np.empty((nx0, nt_s), dtype=float)

# Assign the initial condition, x0, to solution array, x
x[:, 0] = x0;

# Numerically solve
#t_s, x = numerical_integration_methods.forward_euler(flat_earth_eom.flat_earth_eom, t_s, x, h_s)
t_s, x = numerical_integration_methods.rk4(flat_earth_eom.flat_earth_eom, t_s, x, h_s)


# Data post-processing actions

# =============================================================================
# Part 3: Plot data
# =============================================================================

# Create subplots and set layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))  # 1 row, 2 columns

# Plot line 1 on first subplot
ax1.plot(t_s, x[0, :], label='Line 1')
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Line 1 Value')
ax1.set_title('Line 1')
ax1.grid(True)

# Plot line 2 on second subplot
ax2.plot(t_s, x[1, :], label='Line 2')
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Line 2 Value')
ax2.set_title('Line 2')
ax2.grid(True)

# Remaining plots to appear below

plt.show()