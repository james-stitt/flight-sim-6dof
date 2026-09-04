import math
import numpy as np


def flat_earth_eom(t, x, vmod, amod):
    """
    Arguments:
        t - time [s], scalar
        x - state vector at time t, numpy array
            x[0] - u_b_mps, axial velocity of CM w.r.t. intertial frame resolved in aircraft fixed body axis
            x[1] - v_b_mps, lateral velocity of CM w.r.t. intertial frame resolved in aircraft fixed body axis
            x[2] - w_b_mps, vertical velocity of CM w.r.t. intertial frame resolved in aircraft fixed body axis
            x[3] - p_b_rps, roll angular velocity of body fixed frame w.r.t. inertial frame
            x[4] - q_b_rps, pitch angular velocity of body fixed frame w.r.t. inertial frame
            x[5] - r_b_rps, yaw angular velocity of body fixed frame w.r.t. inertial frame            
            x[6] - phi_rad, roll angle
            x[7] - theta_rad, pitch angle
            x[8] - psi_rad, yaw angle
            x[9] - p1_n_m, x position of aircraft resolved in NED frame
            x[10] - p2_n_m, y position of aircraft resolved in NED frame
            x[11] - p3_n_m, z position of aircraft resolved in NED frame
        vmod - vehicle model data, dictionary
        amod - airodynamic model data, dictionary

    Returns:
        dx - time derivative of state vector x
    """

    # Preallocation of dx
    dx = np.empty((12,), dtype=float)

    # Assign variable names
    u_b_mps = x[0]
    v_b_mps = x[1]
    w_b_mps = x[2]
    p_b_rps = x[3]
    q_b_rps = x[4]
    r_b_rps = x[5]
    phi_rad = x[6]
    theta_rad = x[7]
    psi_rad = x[8]
    p1_n_m = x[9]
    p2_n_m = x[10]
    p3_n_m = x[11]

    # Pre-Compute Trig Funcs
    s_phi = math.sin(phi_rad)
    s_theta = math.sin(theta_rad)
    s_psi = math.sin(psi_rad)
    c_phi = math.cos(phi_rad)
    c_theta = math.cos(theta_rad)
    c_psi = math.cos(psi_rad)
    t_the = math.tan(theta_rad)

    v_b     = x[0:3]  # [u_b_mps, v_b_mps, w_b_mps]
    omega_b = x[3:6]  # [p_b_rps, q_b_rps, r_b_rps]

    # Mass and moments of inertia
    m_kg = vmod['m_kg']
    I = vmod['I']
    I_inv = vmod['I_inv']

    # Current altitude
    h_m = -p3_n_m

    # Atmosphere Model
    rho_interp_kgpm3 = np.interp(h_m, amod["alt_m"], amod["rho_kgpm3"])
    c_interp_mps     = np.interp(h_m, amod["alt_m"], amod["c_mps"])


    # Air Data
    true_airspeed_mps = np.linalg.norm(v_b)
    qbar_kgpms2       = 0.5 * rho_interp_kgpm3 * true_airspeed_mps**2

    alpha_rad = math.atan2(w_b_mps, u_b_mps)
    if true_airspeed_mps > 1e-5:
        beta_rad = math.asin(max(-1.0, min(1.0, v_b_mps / true_airspeed_mps)))
    else:
        beta_rad = 0.0
    s_alpha   = math.sin(alpha_rad)
    c_alpha   = math.cos(alpha_rad)
    s_beta    = math.sin(beta_rad)
    c_beta    = math.cos(beta_rad)

    # Gravity in NED frame
    gz_interp_n_mps2 = np.interp(h_m, amod["alt_m"], amod['g_mps2'])

    # Resolve gravity to body frame
    gx_b_mps2 = -math.sin(theta_rad) * gz_interp_n_mps2
    gy_b_mps2 = math.sin(phi_rad) * math.cos(theta_rad) * gz_interp_n_mps2
    gz_b_mps2 = math.cos(phi_rad) * math.cos(theta_rad) * gz_interp_n_mps2

    g_b = np.array([gx_b_mps2, gy_b_mps2, gz_b_mps2])

    # Aerodynamic Forces
    drag_kgmps2 = vmod["CD_approx"]*qbar_kgpms2*vmod["Aref_m2"]
    side_kgmps2 = 0
    lift_kgmps2 = 0

    # External Forces
    R_w_to_b = np.array([
        [ c_alpha * c_beta, -c_alpha * s_beta, -s_alpha],
        [           s_beta,            c_beta,      0.0],
        [ s_alpha * c_beta, -s_alpha * s_beta,  c_alpha]
    ])

    # F_aero_w = [Drag, Side, Lift]
    F_b = -R_w_to_b @ np.array([drag_kgmps2, side_kgmps2, lift_kgmps2])

    # External Moments
    l_b_kgm2ps2 = 0
    m_b_kgm2ps2 = 0
    n_b_kgm2ps2 = 0

    M_b = np.array([l_b_kgm2ps2, m_b_kgm2ps2, n_b_kgm2ps2])

    # Translational dynamics: d/dt(v_b) = F/m + g - (omega x v)
    dx[0:3] = (F_b / m_kg) + g_b - np.cross(omega_b, v_b)

    # Rotational dynamics: d/dt(omega_b) = I^-1 * (M - (omega x (I @ omega)))
    dx[3:6] = I_inv @ (M_b - np.cross(omega_b, I @ omega_b))

    # Kinematic equations
    sec_the = 1.0 / c_theta  # Note: singular at theta = +/- 90 deg

    T_euler = np.array([
        [1.0,  s_phi * t_the,    c_phi * t_the],
        [0.0,  c_phi,           -s_phi        ],
        [0.0,  s_phi * sec_the,  c_phi * sec_the]
    ])

    # omega_b = np.array([p_b_rps, q_b_rps, r_b_rps]) or slice x[3:6]
    dx[6:9] = T_euler @ omega_b

    # Position (Navigation) Equations
    # Direction Cosine Matrix: Body frame to NED frame (R_b_to_n)
    R_b_to_n = np.array([
        [c_theta * c_psi,  s_phi * s_theta * c_psi - c_phi * s_psi,  c_phi * s_theta * c_psi + s_phi * s_psi],
        [c_theta * s_psi,  s_phi * s_theta * s_psi + c_phi * c_psi,  c_phi * s_theta * s_psi - s_phi * c_psi],
        [       -s_theta,                          s_phi * c_theta,                          c_phi * c_theta]
    ])

    # Position (Navigation) equations: d/dt(p_ned) = R_b_to_n @ v_b
    dx[9:12] = R_b_to_n @ v_b  # where v_b = x[0:3]

    return dx
    
