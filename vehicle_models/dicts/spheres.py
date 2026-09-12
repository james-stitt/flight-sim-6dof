import numpy as np


def BowlingBall():
    r_sphere_m = 0.1
    m_sphere_kg = 9
    J_sphere_kgm2 = 0.4 * m_sphere_kg * r_sphere_m**2

    vmod = {
        "short_name": "bowling_ball",
        # Relative or absolute path to the 3D model geometry
        "model_path": "./vehicle_models/3d_models/sphere.xml",
        "m_kg": 5,
        "Jxz_b_kgm2": 0,
        "Jxx_b_kgm2": J_sphere_kgm2,
        "Jyy_b_kgm2": J_sphere_kgm2,
        "Jzz_b_kgm2": J_sphere_kgm2,
        "CD": 0.47,
        "Aref_m2": np.pi * r_sphere_m**2,
    }

    vmod["I"] = np.array([
        [vmod["Jxx_b_kgm2"], 0.0, -vmod["Jxz_b_kgm2"]],
        [0.0, vmod["Jyy_b_kgm2"], 0.0],
        [-vmod["Jxz_b_kgm2"], 0.0, vmod["Jzz_b_kgm2"]],
    ])

    vmod["I_inv"] = np.linalg.inv(vmod["I"])
    return vmod


def NASASpheroid():
    slug_to_kg = 14.5939029
    ft_to_m = 0.3048

    m_kg = 1.0 * slug_to_kg
    J_diag_kgm2 = 3.6 * slug_to_kg * (ft_to_m**2)
    Aref_m2 = 0.1963495 * (ft_to_m**2)

    vmod = {
        "short_name": "NASA_spheroid",
        # Relative or absolute path to the 3D model geometry
        "model_path": "./vehicle_models/3d_models/Spitfire/spitfire9.ac",
        "m_kg": m_kg,
        "Jxz_b_kgm2": 0.0,
        "Jxx_b_kgm2": J_diag_kgm2,
        "Jyy_b_kgm2": J_diag_kgm2,
        "Jzz_b_kgm2": J_diag_kgm2,
        "CD": 0.1,
        "CL": 0.0,
        "CY": 0.0,
        "Cl": 0.0,
        "Cm": 0.0,
        "Cn": 0.0,
        "Aref_m2": Aref_m2,
    }

    vmod["I"] = np.array([
        [vmod["Jxx_b_kgm2"], 0.0, -vmod["Jxz_b_kgm2"]],
        [0.0, vmod["Jyy_b_kgm2"], 0.0],
        [-vmod["Jxz_b_kgm2"], 0.0, vmod["Jzz_b_kgm2"]],
    ])

    vmod["I_inv"] = np.linalg.inv(vmod["I"])
    return vmod