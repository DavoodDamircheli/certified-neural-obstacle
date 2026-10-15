"""v14 A4: split the global-Lipschitz Level-2 remainder E_h by component.

The weighted bound needs separate upper bounds U_r, U_p, U_c; a single E_h
cannot be split afterwards, so every remainder term of specification section 7 is
assigned to the density it bounds:

    L_sm: (2 Lambda/alpha) m_R Lip(R_PDE) + 2 m_sig Lip(R_sig)    -> r
    L_sm: 2 kappa m_gap L_gamma                                    -> p
    L_sm: m_lam L_gamma + m_gap Lip(lambda_psi)                    -> c
    L_Psi, E_h_unr, E_h_dat                                        -> p
    L_Phi0: 2 m_0 (Lip(u0) + ||grad_x u_theta||_inf)               -> r (initial mismatch)
    L_Phi0: 2 m_gap L_gamma, all of L_PhiT                         -> p (traces of w)

U_r = rho_r_h^2 + E_h^r, U_p = rho_p_h^2 + E_h^p, U_c = rho_c_h^2 + E_h^c, and
U_r + U_p + U_c == rho_tilde_h^2 + E_h up to float rounding (regression test).
Status: unvalidated_enclosure (valid in exact arithmetic, float64 evaluation,
midpoint-quadrature raw estimator).
"""

from __future__ import annotations

from certified_obstacle_nn.certification.network_bounds import decompose_L_sm_and_L_Psi

STATUS = "unvalidated_enclosure"


def split_global_remainder(problem, bounds, remainder, h: float, rho_r: float, rho_p: float, rho_c: float) -> dict:
    a, b = problem.omega_bounds
    omega = b - a
    Q = problem.T * omega
    terms = decompose_L_sm_and_L_Psi(problem, bounds)
    sm = terms["L_sm_terms"]
    psi = terms["L_Psi_terms"]
    L_r = sm["T1_pde_residual"] + sm["T2_sig_residual"]
    L_p = sm["T3_kappa_gap"]
    L_c = sm["T4_lambda_Lgamma"] + sm["T5_gap_lip_lambda"]
    L_psi = sum(psi.values())

    phi0_r = 2.0 * bounds.m_0 * (problem.u0_lip + bounds.grad_x_u_inf)
    phi0_p = 2.0 * bounds.m_gap * bounds.L_gamma
    E_r = L_r * h * Q + phi0_r * h * omega
    E_p = (L_p + L_psi) * h * Q + remainder.E_h_unr + remainder.E_h_dat + (phi0_p + bounds.L_PhiT) * h * omega
    E_c = L_c * h * Q
    return {
        "E_h_r": E_r,
        "E_h_p": E_p,
        "E_h_c": E_c,
        "U_r": rho_r**2 + E_r,
        "U_p": rho_p**2 + E_p,
        "U_c": rho_c**2 + E_c,
        "status": STATUS,
    }
