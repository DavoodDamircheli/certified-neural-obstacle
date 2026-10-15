"""Physical-units conversion for a triple trained on normalized targets
(specification section 10, American put: "Train on normalized ubar=u/K, gbar=g/K,
fbar=f/K ... Convert back to physical units before certifying").

Dividing the WHOLE PDE (specification section 0) by K --
    d_tau(u) - A*u_xx + b*u_x + c*u - lambda = f
    => d_tau(ubar) - A*ubar_xx + b*ubar_x + c*ubar - lambdabar = fbar
   (ubar=u/K, fbar=f/K, and lambdabar := lambda/K falls out of the SAME
    division, not a separate choice)
-- shows u, the flux sigma = -A*grad(u) (linear in u, so it inherits the
same 1/K), and the multiplier lambda all scale by the SAME factor K when
converting the normalized-trained representation back to physical units.
PhysicalUnitsTriple applies that one factor uniformly to all three
sub-networks' outputs; nothing here retrains or re-derives anything.
"""

import torch


class _ScaleOutput:
    def __init__(self, base_net, scale: float):
        self.base_net = base_net
        self.scale = scale

    def __call__(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.scale * self.base_net(t, x)


class PhysicalUnitsTriple:
    """Duck-typed triple (matches certification/stress_corruption.py's
    CorruptedTriple pattern): wraps a triple trained on NormalizedAmericanPutProblem
    targets, exposing physical-unit state_net/flux_net/multiplier_net.
    .eval()/.parameters() delegate to the base triple -- scaling by a
    constant K introduces no new learnable parameters."""

    def __init__(self, base_triple, K: float):
        self.base_triple = base_triple
        self.K = K
        self.state_net = _ScaleOutput(base_triple.state_net, K)
        self.flux_net = _ScaleOutput(base_triple.flux_net, K)
        self.multiplier_net = _ScaleOutput(base_triple.multiplier_net, K)

    def eval(self):
        self.base_triple.eval()

    def parameters(self):
        return self.base_triple.parameters()
