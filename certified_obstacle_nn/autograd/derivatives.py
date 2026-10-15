"""First-derivative extraction for the neural triple, specification section 0.

Only first derivatives are ever computed here: d_t u, grad_x u, d_x sigma. No
second derivatives are taken anywhere in this module.

Two entry points, deliberately kept separate rather than parameterized by a
flag, because they serve mutually exclusive callers with opposite memory
requirements:

  compute_derivatives      -- the TRAINING path. create_graph=True is
                               required so that a training loss built from
                               these derivatives can still be backpropagated
                               into theta/phi/psi -- that is not a second
                               derivative, it is keeping the first-derivative
                               computation itself differentiable.

  eval_triple_derivatives  -- the CERTIFICATION path. create_graph=False,
                               since certification never takes a second
                               derivative or a further optimizer step (specification
                               section 12 invariant 2), so the graph-of-the-graph
                               create_graph=True builds is pure waste there --
                               roughly half of certification's peak memory in
                               the raw estimator (see
                               docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md
                               section 1). Every returned tensor is
                               `.detach()`-ed at the boundary so no graph
                               survives into the density code, and model
                               parameters are asserted frozen on entry since
                               this function must never be reachable from a
                               training loop. A thin wrapper around
                               eval_derivatives_detached below (same
                               create_graph=False body), adding only the
                               NeuralTriple-specific frozen-parameter assertion.

  eval_derivatives_detached -- the general create_graph=False evaluator
                               eval_triple_derivatives wraps. Takes three bare
                               (t, x) -> tensor callables, exactly like
                               compute_derivatives does, rather than a
                               NeuralTriple -- for callers that need
                               create_graph=False on something that is not
                               (and cannot be, e.g. a problem's closed-form
                               exact solution) a NeuralTriple with parameters
                               to assert frozen. errors/true_error.py uses this
                               directly for both the network triple's
                               sub-networks and the independent reference
                               solution, since it evaluates both through the
                               identical code path.

`t` and `x` must already be part of an active autograd graph (leaf tensors
with requires_grad=True, or otherwise connected to one) when passed in to
compute_derivatives. This function does not clone, detach, or rescale them --
doing so would silently disconnect them from whatever graph the caller built
(including any upstream input reparametrization, e.g. the American-put
xi = 2(x-L)/(R-L) - 1 remap of specification section 10) and corrupt every
downstream residual. eval_triple_derivatives is the opposite: it takes
ownership of `t`/`x`, detaching and re-attaching requires_grad itself, since
its whole point is to guarantee no incoming graph is retained.
"""

import torch


def compute_derivatives(state_net, flux_net, multiplier_net, t: torch.Tensor, x: torch.Tensor):
    u = state_net(t, x)
    sigma = flux_net(t, x)
    lam = multiplier_net(t, x)

    (u_t,) = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
    (u_x,) = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)
    (sigma_x,) = torch.autograd.grad(sigma, x, grad_outputs=torch.ones_like(sigma), create_graph=True, retain_graph=True)

    return u, sigma, lam, u_t, u_x, sigma_x


def eval_triple_derivatives(model, t: torch.Tensor, x: torch.Tensor):
    """Certification-only sibling of compute_derivatives. Same five outputs
    (u, sigma, lam, u_t, u_x, sigma_x) and the same values to machine
    precision (tests/test_autograd.py checks this against compute_derivatives
    directly) -- only the memory behavior differs. `model` is the full
    NeuralTriple (not three separate callables), since this function's own
    invariant check needs `model.parameters()`.

    Three deliberate departures from compute_derivatives, each a real failure
    mode if gotten wrong:

    - create_graph=False. Halves peak memory for this step (measured
      directly, docs/history/MEMORY_OOM_DEBUGGING_AND_SAFE_RUN_POLICY.md section 1)
      because no backward-of-the-backward graph is retained.
    - NOT wrapped in torch.no_grad(). We still need autograd w.r.t. the
      INPUTS (t, x) -- torch.no_grad() here would silently produce zero
      gradients (or an error), not a speedup. torch.enable_grad() is used
      explicitly instead so this is correct even if a caller wraps its own
      call site in torch.no_grad().
    - u_t and u_x come from ONE torch.autograd.grad call over both inputs at
      once, not two calls with retain_graph=True -- one backward pass through
      u's graph instead of two, and the graph can be freed immediately after
      (retain_graph defaults to create_graph=False) instead of kept alive
      for a second call.
    """
    for p in model.parameters():
        assert not p.requires_grad, (
            "eval_triple_derivatives: certification invariant violated -- a model "
            "parameter has requires_grad=True (this function must never run inside "
            "a training loop, specification section 12 invariant 2)"
        )
    return eval_derivatives_detached(model.state_net, model.flux_net, model.multiplier_net, t, x)


def eval_derivatives_detached(state_net, flux_net, multiplier_net, t: torch.Tensor, x: torch.Tensor):
    """create_graph=False sibling of compute_derivatives, for bare (t, x) ->
    tensor callables rather than a NeuralTriple -- see module docstring for
    why this exists separately from eval_triple_derivatives (which adds only
    a frozen-parameter assertion on top of this). Same departures from
    compute_derivatives as eval_triple_derivatives: create_graph=False, NOT
    torch.no_grad() (autograd w.r.t. the INPUTS is still required), ONE
    torch.autograd.grad call for (u_t, u_x) instead of two.
    """
    t = t.detach().requires_grad_(True)
    x = x.detach().requires_grad_(True)

    with torch.enable_grad():
        u = state_net(t, x)
        sig = flux_net(t, x)
        lam = multiplier_net(t, x)

        u_t, u_x = torch.autograd.grad(u.sum(), (t, x), create_graph=False)
        (sig_x,) = torch.autograd.grad(sig.sum(), (x,), create_graph=False)

    return tuple(v.detach() for v in (u, sig, lam, u_t, u_x, sig_x))


def eval_gap_derivatives(problem, t: torch.Tensor, x: torch.Tensor, u_t: torch.Tensor, u_x: torch.Tensor):
    """Certification-only helper: (d_t gamma, grad_x gamma) for gamma = g - u,
    given ALREADY-COMPUTED u_t, u_x (e.g. from eval_triple_derivatives).

    Never re-derives u_t/u_x by backpropagating through u a second time --
    that would require the network's forward graph to still be alive, which
    eval_triple_derivatives deliberately does not keep (everything it returns
    is detached). Instead this takes the cheap path specification section 5
    licenses (grad_x w = 1_{gamma>0} grad_x gamma, d_t w = 1_{gamma>0} d_t
    gamma -- the same identity, one derivative order up): gamma_t = g_t - u_t,
    gamma_x = g_x - u_x, where g_t/g_x come from a SEPARATE, tiny backward
    pass through g ALONE -- g is a simple closed-form function of (t, x),
    independent of the trained networks and hence orders of magnitude cheaper
    than a second pass through the network would be.
    """
    t2 = t.detach().requires_grad_(True)
    x2 = x.detach().requires_grad_(True)
    with torch.enable_grad():
        g = problem.g(t2, x2)
        # allow_unused=True: some problems' g is genuinely t-independent
        # (American put, specification section 10 -- g depends on x alone), so
        # g.sum() has no path back to t2 in the graph at all. Without
        # allow_unused, autograd.grad raises outright instead of returning
        # the mathematically correct answer (g_t=0) -- Stefan's own g DOES
        # depend on t, so this never surfaced there.
        g_t, g_x = torch.autograd.grad(g.sum(), (t2, x2), create_graph=False, allow_unused=True)
        if g_t is None:
            g_t = torch.zeros_like(t2)
        if g_x is None:
            g_x = torch.zeros_like(x2)
    return (g_t.detach() - u_t), (g_x.detach() - u_x)
