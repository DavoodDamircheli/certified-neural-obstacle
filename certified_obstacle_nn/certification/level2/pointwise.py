"""Exact pointwise forward-mode jets of a tanh MLP (value, Jacobian, optional Hessian).

For z^0 = p, a^k = W_k z^{k-1} + b_k, z^k = tanh(a^k) (k < L), N = W_L z^{L-1} + b_L:
    dz^k   = s^k * (W_k dz^{k-1}),                         s = 1 - tanh^2 = tanh'
    d2z^k  = s^k * (W_k d2z^{k-1}) + s'^k * (W_k dz^{k-1}) (x) (W_k dz^{k-1}),   s' = -2 tanh s
Plain float64 tensors, no autograd. Used for (i) cell-center values inside the enclosures,
(ii) diagnostics. Agreement with torch.autograd is tested in certification/tests.
"""

import torch


def mlp_layers(mlp):
    return [(layer.weight.detach(), layer.bias.detach()) for layer in mlp.layers]


def mlp_jet(mlp, P: torch.Tensor, dirs=None, order: int = 1):
    """P [N, m]. dirs: list of input indices to differentiate along (default all m).
    Returns (val [N,out], jac [N,out,k] or None, hess [N,out,k,k] or None)."""
    layers = mlp_layers(mlp)
    m = P.shape[1]
    dirs = list(range(m)) if dirs is None else list(dirs)
    W, b = layers[0]
    a = P @ W.T + b
    da = W[:, dirs].expand(P.shape[0], -1, -1) if order >= 1 else None  # [N,n,k]
    d2a = None
    for W, b in layers[1:]:
        z = torch.tanh(a)
        s = 1.0 - z * z
        if order >= 1:
            dz = s[..., None] * da
            if order >= 2:
                sp = -2.0 * z * s
                outer = da[..., :, None] * da[..., None, :]
                d2z = sp[..., None, None] * outer
                if d2a is not None:
                    d2z = d2z + s[..., None, None] * d2a
            da = torch.einsum("ij,njk->nik", W, dz)
            if order >= 2:
                d2a = torch.einsum("ij,njkl->nikl", W, d2z)
        a = z @ W.T + b
    if order >= 2 and d2a is None:
        d2a = torch.zeros(a.shape + (len(dirs), len(dirs)), dtype=a.dtype)
    return a, (da if order >= 1 else None), (d2a if order >= 2 else None)
