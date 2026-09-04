import numpy as np

def BowlingBall():
    r_sphere_m = 0.1
    m_sphere_kg = 9
    J_sphere_kgm2 = 0.4*m_sphere_kg*r_sphere_m**2


    vmod = {
        "m_kg": 5,
        "Jxz_b_kgm2": 0,
        "Jxx_b_kgm2": J_sphere_kgm2,
        "Jyy_b_kgm2": J_sphere_kgm2,
        "Jzz_b_kgm2": J_sphere_kgm2,
        "CD_approx": 0.47,
        "Aref_m2": np.pi * r_sphere_m**2
    }

    vmod["I"] = np.array([
        [ vmod["Jxx_b_kgm2"],                 0.0, -vmod["Jxz_b_kgm2"]],
        [                 0.0,  vmod["Jyy_b_kgm2"],                 0.0],
        [-vmod["Jxz_b_kgm2"],                 0.0,  vmod["Jzz_b_kgm2"]],
    ])

    vmod["I_inv"] = np.linalg.inv(vmod["I"])


    return vmod