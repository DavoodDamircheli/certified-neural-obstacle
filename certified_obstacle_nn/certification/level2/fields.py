"""Assemble the Level-2 fields (R_PDE, R_sig, gamma, d_t gamma, grad_x gamma, lambda) at points
(exact) or over boxes (sound enclosures), optionally together with their space-time gradients.

  fields_at(prob, P, grad=False)          P [N, m] tensor -> dict of tensors (or Jets of tensors)
  fields_on_boxes(prob, lo, hi, mode, grad=False)
        mode 'interval' (Tier 1) or 'affine' (Tier 3) -> dict of Aff (or Jets of Aff)
"""

from __future__ import annotations

import torch

from certified_obstacle_nn.certification.level2.aff import Aff
from certified_obstacle_nn.certification.level2.netenc import mlp_enclose
from certified_obstacle_nn.certification.level2.ops import Jet
from certified_obstacle_nn.certification.level2.pointwise import mlp_jet


def _sel(v, dim, i):
    return v.select(dim, i) if torch.is_tensor(v) else v.sel(dim, i)


def _net_dict(prob, val_jac_hess, grad):
    """val_jac_hess: dict name -> (val [B,out], jac [B,out,m], hess [B,out,m,m] or None), every
    derivative taken w.r.t. ALL m = 1+d coordinates (index 0 = t, 1+i = x_i)."""
    d = prob.d
    Nv, Nj, Nh = val_jac_hess["state"]
    Sv, Sj, Sh = val_jac_hess["flux"]
    Mv, Mj, _ = val_jac_hess["mult"]
    if not grad:
        N = _sel(Nv, 1, 0)
        Nt = _sel(_sel(Nj, 1, 0), 1, 0)
        Nx = [_sel(_sel(Nj, 1, 0), 1, 1 + i) for i in range(d)]
        S = [_sel(Sv, 1, i) for i in range(d)]
        dS = [_sel(_sel(Sj, 1, i), 1, 1 + i) for i in range(d)]
        M = _sel(Mv, 1, 0)
        return {"N": N, "N_t": Nt, "N_x": Nx, "S": S, "dS": dS, "M": M}
    Nj0, Nh0 = _sel(Nj, 1, 0), _sel(Nh, 1, 0)
    N = Jet(_sel(Nv, 1, 0), Nj0)
    Nt = Jet(_sel(Nj0, 1, 0), _sel(Nh0, 1, 0))
    Nx = [Jet(_sel(Nj0, 1, 1 + i), _sel(Nh0, 1, 1 + i)) for i in range(d)]
    S = [Jet(_sel(Sv, 1, i), _sel(Sj, 1, i)) for i in range(d)]
    dS = [Jet(_sel(_sel(Sj, 1, i), 1, 1 + i), _sel(_sel(Sh, 1, i), 1, 1 + i)) for i in range(d)]
    M = Jet(_sel(Mv, 1, 0), _sel(Mj, 1, 0))
    return {"N": N, "N_t": Nt, "N_x": Nx, "S": S, "dS": dS, "M": M}


def _coords(P_list, m, grad, like):
    if not grad:
        return P_list
    out = []
    for j, p in enumerate(P_list):
        if torch.is_tensor(p):
            gj = torch.zeros(p.shape + (m,), dtype=p.dtype)
            gj[..., j] = 1.0
            out.append(Jet(p, gj))
        else:
            gj = torch.zeros(p.c.shape + (m,), dtype=p.c.dtype)
            gj[..., j] = 1.0
            out.append(Jet(p, type(p).const(gj, p.m)))
    return out


def fields_at(prob, P: torch.Tensor, grad: bool = False):
    order = 2 if grad else 1
    vjh = {"state": mlp_jet(prob.state, P, order=order), "flux": mlp_jet(prob.flux, P, order=order),
           "mult": mlp_jet(prob.mult, P, order=1 if grad else 0)}
    net = _net_dict(prob, vjh, grad)
    coords = _coords([P[:, j] for j in range(prob.m)], prob.m, grad, P)
    return prob.fields_from_net(coords, net)


def _zono_nets(prob, Z, max_sym):
    """Tier 3z: each network with its own symbols, then laid out in one global symbol list."""
    from certified_obstacle_nn.certification.level2.netenc_zono import mlp_enclose_zono, relayout
    m = Z.m
    outs = {"state": mlp_enclose_zono(prob.state, Z, 1, max_sym), "flux": mlp_enclose_zono(prob.flux, Z, 1, max_sym),
            "mult": mlp_enclose_zono(prob.mult, Z, 0, max_sym)}
    total = m + sum(o[2] - m for o in outs.values())
    off = m
    vjh = {}
    for k, (v, j, M) in outs.items():
        v2 = relayout(v, m, off, total)
        j2 = relayout(j, m, off, total) if j is not None else None
        vjh[k] = (v2, j2, None)
        off += M - m
    return vjh


def fields_on_boxes(prob, lo: torch.Tensor, hi: torch.Tensor, mode: str = "affine", grad: bool = False,
                    max_sym: int = 128):
    if mode == "zono":
        assert not grad
        Z = Aff.from_box(lo, hi, affine=True)
        net = _net_dict(prob, _zono_nets(prob, Z, max_sym), False)
        coords = [Z.sel(1, j) for j in range(prob.m)]
        return prob.fields_from_net(coords, net)
    if mode == "tm2":
        from certified_obstacle_nn.certification.level2.tm import TM
        Z = TM.from_box(lo, hi)
    else:
        Z = Aff.from_box(lo, hi, affine=(mode == "affine"))
    order = 2 if grad else 1
    vjh = {"state": mlp_enclose(prob.state, Z, order=order), "flux": mlp_enclose(prob.flux, Z, order=order),
           "mult": mlp_enclose(prob.mult, Z, order=1 if grad else 0)}
    net = _net_dict(prob, vjh, grad)
    coords = _coords([Z.sel(1, j) for j in range(prob.m)], prob.m, grad, Z)
    return prob.fields_from_net(coords, net)


def u0_at(prob, X: torch.Tensor):
    return prob.u0([X[:, i] for i in range(prob.d)])


def state_u_at(prob, P: torch.Tensor):
    N, _, _ = mlp_jet(prob.state, P, order=0)
    return prob.mask([P[:, 1 + i] for i in range(prob.d)]) * N[:, 0]
