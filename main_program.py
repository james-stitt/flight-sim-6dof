import math
import matplotlib.pyplot as plt
import numpy as np
import ussa1976

from numerical_integrators import numerical_integration_methods
from governing_equations import flat_earth_eom
#from tools.Interpolators import fastInterp1
from vehicle_models.sphere import spheres

# =============================================================================
# Part 1: Initialization of simulation
# =============================================================================

# Atmospheric Data
atmosphere = ussa1976.compute()

amod = {
    "alt_m" : atmosphere["z"].values, \
    "rho_kgpm3" : atmosphere["rho"].values, \
    "c_mps" : atmosphere["cs"].values, \
    "g_mps2" : ussa1976.core.compute_gravity(atmosphere["z"].values)
}

# Define Vehicle
vmod = spheres.BowlingBall()

# Set initial conditions (these conditions may be loaded from an aircraft
# trim routine in future versions of the code)
u0_bf_mps  = 0.0001
v0_bf_mps  = 0
w0_bf_mps  = 0
p0_bf_rps  = 0
q0_bf_rps  = 0
r0_bf_rps  = 0
phi0_rad   = 0*math.pi/180
theta0_rad = -90*math.pi/180
psi0_rad   = 0
p10_n_m    = 0
p20_n_m    = 0
p30_n_m    = -30000

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
tf_s = 100.0
h_s = 0.005

# =============================================================================
# Part 2: Numerically approximate solutions to the governing equations
# =============================================================================

# Preallocate the solution array
t_s = np.arange(t0_s, tf_s + h_s, h_s) 
nt_s = t_s.size
x = np.empty((nx0, nt_s), dtype=float)

# Assign the initial condition, x0, to solution array, x
x[:, 0] = x0

# Numerically solve
#t_s, x = numerical_integration_methods.forward_euler(flat_earth_eom.flat_earth_eom, t_s, x, h_s)
t_s, x = numerical_integration_methods.rk4(flat_earth_eom.flat_earth_eom, t_s, x, h_s, vmod, amod)


# Data post-processing actions

# Altitude (NED z is down, so altitude is -z)
Altitude_m = -x[11, :]

# Atmosphere interpolation using standard numpy (or fastInterp1)
Cs_mps    = np.interp(Altitude_m, amod["alt_m"], amod["c_mps"])
Rho_kgpm3 = np.interp(Altitude_m, amod["alt_m"], amod["rho_kgpm3"])

# Airspeed: norm across axes [u, v, w]
True_Airspeed_mps = np.linalg.norm(x[0:3, :], axis=0)

# Aerodynamic angles (np.arctan2 avoids divide-by-zero errors)
Alpha_rad = np.arctan2(x[2, :], x[0, :])
Beta_rad  = np.arcsin(np.clip(np.divide(x[1, :], True_Airspeed_mps, out=np.zeros_like(x[1, :]), where=True_Airspeed_mps != 0), -1.0, 1.0))

# Mach number
Mach = True_Airspeed_mps / Cs_mps

# =============================================================================
# Part 3: Plot data
# =============================================================================

# Create subplots and set layout
fig, axes = plt.subplots(2, 4, figsize=(10, 6))
fig.set_facecolor('black')

# Axial velocity u^b_CM/n
axes[0, 0].plot(t_s, x[0,:], color='yellow')
axes[0, 0].set_xlabel('Time [s]', color='white')
axes[0, 0].set_ylabel('u [m/s]', color='white')
axes[0, 0].grid(True)
axes[0, 0].set_facecolor('black')
axes[0, 0].tick_params(colors = 'white')

# y-axis velocity v^b_CM/n
axes[0, 1].plot(t_s, x[1,:], color='yellow')
axes[0, 1].set_xlabel('Time [s]', color='white')
axes[0, 1].set_ylabel('v [m/s]', color='white')
axes[0, 1].grid(True)
axes[0, 1].set_facecolor('black')
axes[0, 1].tick_params(colors = 'white')

# z-axis velocity w^b_CM/n
axes[0, 2].plot(t_s, x[2,:], color='yellow')
axes[0, 2].set_xlabel('Time [s]', color='white')
axes[0, 2].set_ylabel('w [m/s]', color='white')
axes[0, 2].grid(True)
axes[0, 2].set_facecolor('black')
axes[0, 2].tick_params(colors = 'white')

# Roll angle, phi
axes[0, 3].plot(t_s, x[6,:], color='yellow')
axes[0, 3].set_xlabel('Time [s]', color='white')
axes[0, 3].set_ylabel('phi [rad]', color='white')
axes[0, 3].grid(True)
axes[0, 3].set_facecolor('black')
axes[0, 3].tick_params(colors = 'white')

# Roll rate p^b_b/n
axes[1, 0].plot(t_s, x[3,:], color='yellow')
axes[1, 0].set_xlabel('Time [s]', color='white')
axes[1, 0].set_ylabel('p [r/s]', color='white')
axes[1, 0].grid(True)
axes[1, 0].set_facecolor('black')
axes[1, 0].tick_params(colors = 'white')

# Pitch rate q^b_b/n
axes[1, 1].plot(t_s, x[4,:], color='yellow')
axes[1, 1].set_xlabel('Time [s]', color='white')
axes[1, 1].set_ylabel('q [r/s]', color='white')
axes[1, 1].grid(True)
axes[1, 1].set_facecolor('black')
axes[1, 1].tick_params(colors = 'white')

# Yaw rate r^b_b/n
axes[1, 2].plot(t_s, x[5,:], color='yellow')
axes[1, 2].set_xlabel('Time [s]', color='white')
axes[1, 2].set_ylabel('r [r/s]', color='white')
axes[1, 2].grid(True)
axes[1, 2].set_facecolor('black')
axes[1, 2].tick_params(colors = 'white')

# Pitch angle, theta
axes[1, 3].plot(t_s, x[7,:], color='yellow')
axes[1, 3].set_xlabel('Time [s]', color='white')
axes[1, 3].set_ylabel('theta [rad]', color='white')
axes[1, 3].grid(True)
axes[1, 3].set_facecolor('black')
axes[1, 3].tick_params(colors = 'white')

plt.tight_layout()
#plt.savefig('saved_figures/sphere_drop_test_1.png')
plt.show()

# Create subplots and configure dark theme layout
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
fig.patch.set_facecolor('black')

plot_color = 'magenta'

# 1. Angle of Attack
ax1.plot(t_s, np.rad2deg(Alpha_rad), color=plot_color, linewidth=1.8)
ax1.set_xlabel('Time [s]', color='white')
ax1.set_ylabel('Angle of Attack [deg]', color='white')
ax1.set_ylim([-30, 30])
ax1.set_yticks(np.arange(-30, 31, 10))

# 2. Angle of Side Slip
ax2.plot(t_s, np.rad2deg(Beta_rad), color=plot_color, linewidth=1.8)
ax2.set_xlabel('Time [s]', color='white')
ax2.set_ylabel('Angle of Side Slip [deg]', color='white')
ax2.set_ylim([-30, 30])
ax2.set_yticks(np.arange(-30, 31, 10))

# 3. Mach Number
ax3.plot(t_s, Mach, color=plot_color, linewidth=1.8)
ax3.set_xlabel('Time [s]', color='white')
ax3.set_ylabel('Mach Number', color='white')
ax3.set_ylim(bottom=0.0)

# Format axes and spines across all subplots
for ax in (ax1, ax2, ax3):
    ax.set_facecolor('black')
    ax.grid(True, color='gray', alpha=0.45)
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

plt.tight_layout()
plt.show()

# Position variables from NED coordinates
North_m    = x[9, :]
East_m     = x[10, :]
Altitude_m = -x[11, :]

# Create 2x3 subplots with dark theme
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
fig.patch.set_facecolor('black')

plot_color = '#00f5ff'

# --- Row 1: Position vs Time ---
# North vs Time
axes[0, 0].plot(t_s, North_m, color=plot_color, linewidth=1.8)
axes[0, 0].set_xlabel('Time [s]', color='white')
axes[0, 0].set_ylabel('North [m]', color='white')

# East vs Time
axes[0, 1].plot(t_s, East_m, color=plot_color, linewidth=1.8)
axes[0, 1].set_xlabel('Time [s]', color='white')
axes[0, 1].set_ylabel('East [m]', color='white')

# Altitude vs Time
axes[0, 2].plot(t_s, Altitude_m, color=plot_color, linewidth=1.8)
axes[0, 2].set_xlabel('Time [s]', color='white')
axes[0, 2].set_ylabel('Altitude [m]', color='white')

# --- Row 2: Trajectory Projections ---
# North vs East (Horizontal plane)
axes[1, 0].plot(East_m, North_m, color=plot_color, linewidth=1.8)
axes[1, 0].set_xlabel('East [m]', color='white')
axes[1, 0].set_ylabel('North [m]', color='white')

# Altitude vs East (Vertical plane)
axes[1, 1].plot(East_m, Altitude_m, color=plot_color, linewidth=1.8)
axes[1, 1].set_xlabel('East [m]', color='white')
axes[1, 1].set_ylabel('Altitude [m]', color='white')

# Altitude vs North (Vertical plane)
axes[1, 2].plot(North_m, Altitude_m, color=plot_color, linewidth=1.8)
axes[1, 2].set_xlabel('North [m]', color='white')
axes[1, 2].set_ylabel('Altitude [m]', color='white')

# Style all axes
for ax in axes.flat:
    ax.set_facecolor('black')
    ax.grid(True, color='gray', alpha=0.45)
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')

plt.tight_layout()
plt.show()