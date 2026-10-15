"""Steps D-E (TODO Task C): docs/history/LEVEL2_SLACK_DIAGNOSTICS.csv.

For every global constant entering L_sm (and L_gamma, L_Psi, m_*), compare the rigorous global
bound with the sampled max / median of the corresponding pointwise quantity on 20,000 scrambled
Sobol points of Q (seed 777, independent of every training / evaluation set). Pointwise gradients
are exact forward-mode jets (certification/level2/pointwise.py, ops.Jet) -- DIAGNOSTICS ONLY:
nothing sampled here ever enters a certificate.

python3 -m certified_obstacle_nn.experiments.level2.slack_diagnostics
"""

import csv
import math

import torch

from certified_obstacle_nn.certification.level2.fields import fields_at
from certified_obstacle_nn.certification.level2.problems import make_problem
from certified_obstacle_nn.experiments.level2.common import REPO_ROOT, load_case

C2 = 4.0 / (3.0 * math.sqrt(3.0))


def sobol(n, m, seed=777):
    return torch.quasirandom.SobolEngine(m, scramble=True, seed=seed).draw(n).to(torch.float64)


def _norm(gr):
    return gr.pow(2).sum(-1).sqrt()


def sampled(prob, n=20000):
    P = sobol(n, prob.m)
    F = fields_at(prob, P, grad=True)
    La = prob.Lambda / prob.alpha
    gam, gt, gg = F["gamma"], F["gamma_t"], F["ggrad"]
    out = {
        "m_R": F["R"].val.abs(), "L_R": _norm(F["R"].grad),
        "m_gap": gam.val.abs(), "L_gamma": _norm(gam.grad),
        "m_gap_t": gt.val.abs(), "lip_dt_gamma": _norm(gt.grad),
        "m_gap_x": torch.stack([v.val for v in gg], -1).pow(2).sum(-1).sqrt(),
        "lip_grad_gamma": torch.stack([_norm(v.grad) for v in gg], -1).pow(2).sum(-1).sqrt(),
        "m_sig": torch.stack([v.val for v in F["Rsig"]], -1).pow(2).sum(-1).sqrt(),
        "lip_R_sig": torch.stack([_norm(v.grad) for v in F["Rsig"]], -1).pow(2).sum(-1).sqrt(),
        "m_lam": F["lam"].val.abs(), "lip_lambda": _norm(F["lam"].grad),
    }
    # Psi = A |grad gamma|^2 + La gamma_t^2 ; grad Psi = 2 A sum gx_i grad gx_i + 2 La gt grad gt
    gPsi = 2 * prob.A * sum(v.val[:, None] * v.grad for v in gg) + 2 * La * gt.val[:, None] * gt.grad
    out["L_Psi"] = _norm(gPsi)
    return out


def rigorous_1d(model, meta):
    from certified_obstacle_nn.certification.network_bounds import compute_triple_bounds, decompose_L_sm_and_L_Psi
    pb = meta["problem"]
    b = compute_triple_bounds(pb, model)
    dec = decompose_L_sm_and_L_Psi(pb, b)
    lip_grad_gamma = math.hypot(pb.g_tx_inf, pb.g_xx_inf) + b.lip_grad_x_u
    lip_dt_gamma = math.hypot(pb.g_tt_inf, pb.g_tx_inf) + b.lip_d_t_u
    rig = {"m_R": b.m_R, "L_R": b.lip_R_PDE, "m_gap": b.m_gap, "L_gamma": b.L_gamma, "m_gap_t": b.m_gap_t,
           "lip_dt_gamma": lip_dt_gamma, "m_gap_x": b.m_gap_x, "lip_grad_gamma": lip_grad_gamma,
           "m_sig": b.m_sig, "lip_R_sig": b.lip_R_sig, "m_lam": b.m_lam, "lip_lambda": b.lip_lambda,
           "L_Psi": b.L_Psi, "L_sm": b.L_sm}
    return rig, dec["L_sm_terms"], {"lip_div_s": b.lip_div_x_sigma, "lip_dt_u": b.lip_d_t_u,
                                    "lip_f": b.lip_f, "Pi_flux": b.flux_bounds.gradient_bound}


def rigorous_2d(model, meta):
    from certified_obstacle_nn.certification.network_bounds import compute_network_bounds
    from certified_obstacle_nn.experiments.expanding_disk_2d.level2_diagnostic import data_bounds, triple_bounds
    cfg = meta["cfg"]
    db = data_bounds(cfg.kappa)
    bd = triple_bounds(model, cfg, db)
    sb, fb, mb = (compute_network_bounds(n) for n in (model.state, model.flux, model.mult))
    lip_grad_u = db["grad2B_inf"] * sb.output_bound + 2 * db["gradB_inf"] * sb.gradient_bound \
        + db["B_inf"] * sb.gradient_lipschitz_bound
    lip_dt_u = db["gradB_inf"] * sb.gradient_bound + db["B_inf"] * sb.gradient_lipschitz_bound
    lip_grad_gamma = math.hypot(db["g_tx_inf"], db["g_xx_inf"]) + lip_grad_u
    lip_dt_gamma = math.hypot(db["g_tt_inf"], db["g_tx_inf"]) + lip_dt_u
    rig = {"m_R": bd["m_R_apriori"], "L_R": bd["L_R"], "m_gap": bd["m_gap"], "L_gamma": bd["L_gamma"],
           "m_gap_t": bd["m_gap_t"], "lip_dt_gamma": lip_dt_gamma, "m_gap_x": bd["m_gap_x"],
           "lip_grad_gamma": lip_grad_gamma, "m_sig": bd["m_sig"], "lip_R_sig": bd["lip_R_sig"],
           "m_lam": bd["m_lam"], "lip_lambda": 2 * mb.output_bound * mb.gradient_bound, "L_Psi": bd["L_Psi"],
           "L_sm": bd["L_sm_old"]}
    return rig, bd["L_sm_terms_old"], {"lip_div_s": 2 * fb.gradient_lipschitz_bound, "lip_dt_u": lip_dt_u,
                                       "lip_f": math.hypot(db["f_t_inf"], db["f_x_inf"]),
                                       "Pi_flux": fb.gradient_bound}


def main():
    rows = []
    for case in ("1d", "2d_base", "2d_long"):
        model, meta = load_case(case)
        prob = make_problem(model, meta)
        rig, terms, extra = (rigorous_1d if case == "1d" else rigorous_2d)(model, meta)
        smp = sampled(prob)
        La = prob.Lambda / prob.alpha
        Lsm = rig["L_sm"]
        ck = meta["checkpoint"].replace(str(REPO_ROOT) + "/", "")
        for q in ("L_gamma", "L_R", "m_R", "m_gap", "m_gap_x", "m_gap_t", "m_sig", "lip_R_sig", "m_lam",
                  "lip_lambda", "L_Psi", "lip_grad_gamma", "lip_dt_gamma"):
            v = smp[q]
            mx, md = float(v.max()), float(v.median())
            rows.append({"case": case, "checkpoint": ck, "quantity": q, "rigorous_bound": rig[q], "sampled_max": mx,
                         "sampled_median": md, "bound_to_sampled_max_ratio": rig[q] / mx,
                         "contribution_to_Lsm_fraction": "", "notes": "pointwise quantity; 20k Sobol pts"})
        # the five L_sm terms, sampled analogue = same product with sampled maxima
        smax = {k: float(smp[k].max()) for k in smp}
        samp_terms = {"T1_pde_residual": 2 * La * smax["m_R"] * smax["L_R"],
                      "T2_sig_residual": 2 * smax["m_sig"] * smax["lip_R_sig"],
                      "T3_kappa_gap": 2 * prob.kappa * smax["m_gap"] * smax["L_gamma"],
                      "T4_lambda_Lgamma": smax["m_lam"] * smax["L_gamma"],
                      "T5_gap_lip_lambda": smax["m_gap"] * smax["lip_lambda"]}
        for k, val in terms.items():
            rows.append({"case": case, "checkpoint": ck, "quantity": k, "rigorous_bound": val,
                         "sampled_max": samp_terms[k], "sampled_median": "",
                         "bound_to_sampled_max_ratio": val / samp_terms[k],
                         "contribution_to_Lsm_fraction": val / Lsm,
                         "notes": "L_sm term; sampled = same product of sampled maxima"})
        rows.append({"case": case, "checkpoint": ck, "quantity": "L_sm", "rigorous_bound": Lsm,
                     "sampled_max": sum(samp_terms.values()), "sampled_median": "",
                     "bound_to_sampled_max_ratio": Lsm / sum(samp_terms.values()),
                     "contribution_to_Lsm_fraction": 1.0, "notes": "sum of T1..T5"})
        # sub-terms of L_R (fractions of L_R)
        for k, val in extra.items():
            rows.append({"case": case, "checkpoint": ck, "quantity": k, "rigorous_bound": val, "sampled_max": "",
                         "sampled_median": "", "bound_to_sampled_max_ratio": "",
                         "contribution_to_Lsm_fraction": "",
                         "notes": f"global constant; fraction of L_R = {val / rig['L_R']:.4f}"
                                  if k != "Pi_flux" else "Pi of the flux net (Lip S)"})
        print(case, {r["quantity"]: f"{r['bound_to_sampled_max_ratio']:.3g}" for r in rows
                     if r["case"] == case and r["bound_to_sampled_max_ratio"] != ""})
    out = REPO_ROOT / "docs/history/LEVEL2_SLACK_DIAGNOSTICS.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
