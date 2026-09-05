import math
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ussa1976

from numerical_integrators import numerical_integration_methods
from governing_equations import flat_earth_eom
from vehicle_models.dicts import spheres

# =============================================================================
# 1. CONSTANTS & UNIT CONVERSIONS
# =============================================================================

d2r = math.pi / 180.0
r2d = 180.0 / math.pi
f2m = 1/3.048

# =============================================================================
# 2. VEHICLE & ENVIRONMENT CONFIGURATION
# =============================================================================

# Define vehicle model
vmod = spheres.NASASpheroid()
veh_name = vmod["short_name"]
model_3d_path = vmod.get("model_path", "ufo")

# Atmospheric Data
atmosphere = ussa1976.compute()
amod = {
    "alt_m": atmosphere["z"].values,
    "rho_kgpm3": atmosphere["rho"].values,
    "c_mps": atmosphere["cs"].values,
    "g_mps2": ussa1976.core.compute_gravity(atmosphere["z"].values),
}

# =============================================================================
# 3. SIMULATION SETUP & INITIAL CONDITIONS
# =============================================================================

# Time parameters
t0_s = 0.0
tf_s = 30.0
h_s = 0.01

# Data export toggle
save_full_csv = False
save_flightgear_csv = True
save_data_dir = './case_studies/verification'

# Plot export toggles
save_dir = "./saved_figures"
save_plot_6dof = False
save_plot_airdata = False
save_plot_ned = False

# Initial state variables
u0_bf_mps = 100
v0_bf_mps = 0
w0_bf_mps = 0.0
p0_bf_rps = 0.0 * d2r
q0_bf_rps = 0.0 * d2r
r0_bf_rps = 0.0 * d2r
phi0_rad = 0.0 * d2r
theta0_rad = 0.0 * d2r
psi0_rad = 0.0 * d2r
p10_n_m = 0.0
p20_n_m = 0.0
p30_n_m = -10000.0 * f2m

# Pack initial conditions into state array
x0 = np.array([
    u0_bf_mps,   # 0: x body-fixed velocity (m/s)
    v0_bf_mps,   # 1: y body-fixed velocity (m/s)
    w0_bf_mps,   # 2: z body-fixed velocity (m/s)
    p0_bf_rps,   # 3: Roll rate (rad/s)
    q0_bf_rps,   # 4: Pitch rate (rad/s)
    r0_bf_rps,   # 5: Yaw rate (rad/s)
    phi0_rad,    # 6: Roll angle (rad)
    theta0_rad,  # 7: Pitch angle (rad)
    psi0_rad,    # 8: Yaw angle (rad)
    p10_n_m,     # 9: North position (m)
    p20_n_m,     # 10: East position (m)
    p30_n_m,     # 11: Down position (m)
]).transpose()

nx0 = x0.size

# =============================================================================
# 4. NUMERICAL INTEGRATION
# =============================================================================

t_s = np.arange(t0_s, tf_s + h_s, h_s)
nt_s = t_s.size
x = np.empty((nx0, nt_s), dtype=float)
x[:, 0] = x0

# Solve using Runge-Kutta 4th Order
t_s, x = numerical_integration_methods.rk4(
    flat_earth_eom.flat_earth_eom, t_s, x, h_s, vmod, amod
)

# =============================================================================
# 5. DATA POST-PROCESSING
# =============================================================================

# --- Positions (NED) ---
North_m = x[9, :]
East_m = x[10, :]
Down_m = x[11, :]
Altitude_m = -Down_m

# --- Atmospheric Interpolation & Air Data ---
Cs_mps = np.interp(Altitude_m, amod["alt_m"], amod["c_mps"])
Rho_kgpm3 = np.interp(Altitude_m, amod["alt_m"], amod["rho_kgpm3"])
True_Airspeed_mps = np.linalg.norm(x[0:3, :], axis=0)

Alpha_rad = np.arctan2(x[2, :], x[0, :])
Beta_rad = np.arcsin(
    np.clip(
        np.divide(
            x[1, :],
            True_Airspeed_mps,
            out=np.zeros_like(x[1, :]),
            where=True_Airspeed_mps != 0,
        ),
        -1.0,
        1.0,
    )
)
Mach = True_Airspeed_mps / Cs_mps

# --- NED Velocities via Direction Cosine Matrix (Body to NED) ---
c_phi, s_phi = np.cos(x[6, :]), np.sin(x[6, :])
c_th, s_th = np.cos(x[7, :]), np.sin(x[7, :])
c_psi, s_psi = np.cos(x[8, :]), np.sin(x[8, :])

v_North_mps = (
    (c_th * c_psi) * x[0, :]
    + (s_phi * s_th * c_psi - c_phi * s_psi) * x[1, :]
    + (c_phi * s_th * c_psi + s_phi * s_psi) * x[2, :]
)
v_East_mps = (
    (c_th * s_psi) * x[0, :]
    + (s_phi * s_th * s_psi + c_phi * c_psi) * x[1, :]
    + (c_phi * s_th * s_psi - s_phi * c_psi) * x[2, :]
)
v_Down_mps = (
    (-s_th) * x[0, :]
    + (s_phi * c_th) * x[1, :]
    + (c_phi * c_th) * x[2, :]
)
v_Altitude_mps = -v_Down_mps

# =============================================================================
# 5b. SAVE SIMULATION DATA (FULL & FLIGHTGEAR CSV)
# =============================================================================

import pandas as pd

# Export toggles & directory
save_full_csv = True
save_flightgear_csv = True
save_data_dir = "./case_studies/verification"

# Reference geodetic anchor for Flat-Earth to WGS-84 conversion
lat0_deg = 28.5721   # Reference latitude (e.g., Kennedy Space Center)
lon0_deg = -80.6480  # Reference longitude
R_earth = 6378137.0  # Equatorial radius (m)

if save_full_csv or save_flightgear_csv:
    os.makedirs(save_data_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Full Dataset Export (Comprehensive Engineering Log)
# -----------------------------------------------------------------------------
if save_full_csv:
    full_data = {
        # Time
        "time_s": t_s,
        # Body-frame translational velocities
        "u_mps": x[0, :],
        "v_mps": x[1, :],
        "w_mps": x[2, :],
        # Body-frame rotational rates
        "p_rps": x[3, :],
        "q_rps": x[4, :],
        "r_rps": x[5, :],
        # Euler angles
        "phi_deg": np.rad2deg(x[6, :]),
        "theta_deg": np.rad2deg(x[7, :]),
        "psi_deg": np.rad2deg(x[8, :]),
        # NED positions
        "north_m": North_m,
        "east_m": East_m,
        "altitude_m": Altitude_m,
        # NED velocities
        "v_north_mps": v_North_mps,
        "v_east_mps": v_East_mps,
        "v_altitude_mps": v_Altitude_mps,
        # Air data & atmosphere
        "tas_mps": True_Airspeed_mps,
        "mach": Mach,
        "alpha_deg": np.rad2deg(Alpha_rad),
        "beta_deg": np.rad2deg(Beta_rad),
        "rho_kgpm3": Rho_kgpm3,
        "speed_of_sound_mps": Cs_mps,
    }

    full_df = pd.DataFrame(full_data)
    full_csv_path = os.path.join(save_data_dir, f"{veh_name}_full_data.csv")
    full_df.to_csv(full_csv_path, index=False)

# -----------------------------------------------------------------------------
# 2. FlightGear Replay Export (Generic Protocol Compatible)
# -----------------------------------------------------------------------------
if save_flightgear_csv:
    # Coordinate transformation: Flat-Earth NED to Geodetic
    lat_deg = lat0_deg + (North_m / R_earth) * r2d
    lon_deg = lon0_deg + (East_m / (R_earth * np.cos(np.deg2rad(lat0_deg)))) * r2d
    alt_ft = Altitude_m * 3.28084

    fg_df = pd.DataFrame({
        "time_s": t_s,
        "lat_deg": lat_deg,
        "lon_deg": lon_deg,
        "alt_ft": alt_ft,
        "roll_deg": np.rad2deg(x[6, :]),
        "pitch_deg": np.rad2deg(x[7, :]),
        "heading_deg": np.rad2deg(x[8, :]),
    })

    fg_csv_path = os.path.join(save_data_dir, f"{veh_name}_flightgear_replay.csv")

    # Write model path as metadata comment header, then write trajectory data
    with open(fg_csv_path, "w") as f:
        f.write(f"# model_path: {model_3d_path}\n")
        fg_df.to_csv(f, index=False, header=False)

# =============================================================================
# 6. VISUALIZATION
# =============================================================================

if any([save_plot_6dof, save_plot_airdata, save_plot_ned]):
    os.makedirs(save_dir, exist_ok=True)

def apply_dark_theme(ax, xlabel="", ylabel="", title=""):
    """Helper to maintain consistent styling across subplots."""
    ax.set_facecolor("black")
    ax.grid(True, color="gray", alpha=0.35)
    ax.tick_params(colors="white")
    ax.set_xlabel(xlabel, color="white")
    ax.set_ylabel(ylabel, color="white")
    if title:
        ax.set_title(title, color="white", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")

# -----------------------------------------------------------------------------
# Plot 1: 6 Degrees of Freedom (Velocities, Rates, & Attitude)
# -----------------------------------------------------------------------------
fig_6dof, axes_6dof = plt.subplots(3, 3, figsize=(13, 8))
fig_6dof.patch.set_facecolor("black")
fig_6dof.suptitle(f"{veh_name} - 6 Degrees of Freedom", color="white", fontsize=14)

c_6dof = "yellow"

# Row 1: Body Velocities (Translational DoF)
axes_6dof[0, 0].plot(t_s, x[0, :], color=c_6dof)
apply_dark_theme(axes_6dof[0, 0], xlabel="Time [s]", ylabel="u [m/s]", title="Axial Velocity (u)")

axes_6dof[0, 1].plot(t_s, x[1, :], color=c_6dof)
apply_dark_theme(axes_6dof[0, 1], xlabel="Time [s]", ylabel="v [m/s]", title="Side Velocity (v)")

axes_6dof[0, 2].plot(t_s, x[2, :], color=c_6dof)
apply_dark_theme(axes_6dof[0, 2], xlabel="Time [s]", ylabel="w [m/s]", title="Normal Velocity (w)")

# Row 2: Body Angular Rates (Rotational DoF)
axes_6dof[1, 0].plot(t_s, x[3, :], color=c_6dof)
apply_dark_theme(axes_6dof[1, 0], xlabel="Time [s]", ylabel="p [rad/s]", title="Roll Rate (p)")

axes_6dof[1, 1].plot(t_s, x[4, :], color=c_6dof)
apply_dark_theme(axes_6dof[1, 1], xlabel="Time [s]", ylabel="q [rad/s]", title="Pitch Rate (q)")

axes_6dof[1, 2].plot(t_s, x[5, :], color=c_6dof)
apply_dark_theme(axes_6dof[1, 2], xlabel="Time [s]", ylabel="r [rad/s]", title="Yaw Rate (r)")

# Row 3: Euler Angles
axes_6dof[2, 0].plot(t_s, np.rad2deg(x[6, :]), color=c_6dof)
apply_dark_theme(axes_6dof[2, 0], xlabel="Time [s]", ylabel="phi [deg]", title="Roll Angle (phi)")

axes_6dof[2, 1].plot(t_s, np.rad2deg(x[7, :]), color=c_6dof)
apply_dark_theme(axes_6dof[2, 1], xlabel="Time [s]", ylabel="theta [deg]", title="Pitch Angle (theta)")

axes_6dof[2, 2].plot(t_s, np.rad2deg(x[8, :]), color=c_6dof)
apply_dark_theme(axes_6dof[2, 2], xlabel="Time [s]", ylabel="psi [deg]", title="Yaw Angle (psi)")

fig_6dof.tight_layout()

if save_plot_6dof:
    fig_6dof.savefig(
        os.path.join(save_dir, f"{veh_name}_6dof.png"),
        facecolor=fig_6dof.get_facecolor(),
    )

# -----------------------------------------------------------------------------
# Plot 2: Air Data (Aerodynamic Angles, Mach, True Airspeed)
# -----------------------------------------------------------------------------
fig_air, axes_air = plt.subplots(2, 2, figsize=(11, 7))
fig_air.patch.set_facecolor("black")
fig_air.suptitle(f"{veh_name} - Air Data", color="white", fontsize=14)

c_air = "magenta"

axes_air[0, 0].plot(t_s, np.rad2deg(Alpha_rad), color=c_air, linewidth=1.8)
apply_dark_theme(axes_air[0, 0], xlabel="Time [s]", ylabel="Alpha [deg]", title="Angle of Attack")

axes_air[0, 1].plot(t_s, np.rad2deg(Beta_rad), color=c_air, linewidth=1.8)
apply_dark_theme(axes_air[0, 1], xlabel="Time [s]", ylabel="Beta [deg]", title="Angle of Sideslip")

axes_air[1, 0].plot(t_s, Mach, color=c_air, linewidth=1.8)
axes_air[1, 0].set_ylim(bottom=0.0)
apply_dark_theme(axes_air[1, 0], xlabel="Time [s]", ylabel="Mach [-]", title="Mach Number")

axes_air[1, 1].plot(t_s, True_Airspeed_mps, color=c_air, linewidth=1.8)
axes_air[1, 1].set_ylim(bottom=0.0)
apply_dark_theme(axes_air[1, 1], xlabel="Time [s]", ylabel="Airspeed [m/s]", title="True Airspeed")

fig_air.tight_layout()

if save_plot_airdata:
    fig_air.savefig(
        os.path.join(save_dir, f"{veh_name}_air_data.png"),
        facecolor=fig_air.get_facecolor(),
    )

# -----------------------------------------------------------------------------
# Plot 3: NED Kinematics (Positions and Velocities)
# -----------------------------------------------------------------------------
fig_ned, axes_ned = plt.subplots(2, 3, figsize=(13, 7))
fig_ned.patch.set_facecolor("black")
fig_ned.suptitle(f"{veh_name} - NED Kinematics", color="white", fontsize=14)

c_ned = "#00f5ff"

# Row 1: NED Positions
axes_ned[0, 0].plot(t_s, North_m, color=c_ned, linewidth=1.8)
apply_dark_theme(axes_ned[0, 0], xlabel="Time [s]", ylabel="North [m]", title="North Position")

axes_ned[0, 1].plot(t_s, East_m, color=c_ned, linewidth=1.8)
apply_dark_theme(axes_ned[0, 1], xlabel="Time [s]", ylabel="East [m]", title="East Position")

axes_ned[0, 2].plot(t_s, Altitude_m, color=c_ned, linewidth=1.8)
apply_dark_theme(axes_ned[0, 2], xlabel="Time [s]", ylabel="Altitude [m]", title="Altitude (-Down)")

# Row 2: NED Velocities
axes_ned[1, 0].plot(t_s, v_North_mps, color=c_ned, linewidth=1.8)
apply_dark_theme(axes_ned[1, 0], xlabel="Time [s]", ylabel="v_North [m/s]", title="North Velocity")

axes_ned[1, 1].plot(t_s, v_East_mps, color=c_ned, linewidth=1.8)
apply_dark_theme(axes_ned[1, 1], xlabel="Time [s]", ylabel="v_East [m/s]", title="East Velocity")

axes_ned[1, 2].plot(t_s, v_Altitude_mps, color=c_ned, linewidth=1.8)
apply_dark_theme(axes_ned[1, 2], xlabel="Time [s]", ylabel="v_Altitude [m/s]", title="Climb Rate (-v_Down)")

fig_ned.tight_layout()

if save_plot_ned:
    fig_ned.savefig(
        os.path.join(save_dir, f"{veh_name}_ned_data.png"),
        facecolor=fig_ned.get_facecolor(),
    )

# =============================================================================
# 7. DISPLAY ALL WINDOWS SIMULTANEOUSLY
# =============================================================================
# Calling plt.show() once here displays all open figures at the same time.
plt.show()