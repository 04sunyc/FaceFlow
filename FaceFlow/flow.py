"""Conditional flow matching and tangent-sphere integration."""
import torch
from .geometry import (isotropic_direction, residual_direction, slerp,
                       exp_step, project_tangent, unit, compose)


def flow_loss(net, a, b, generator=None):
    """Return the flow-matching loss and geometric diagnostics.

    Inputs a and b are normalized embeddings of the same identity.
    """
    _, theta, u1 = residual_direction(a, b)
    u0 = isotropic_direction(a, generator)
    t = torch.rand(a.shape[0], 1, device=a.device, dtype=a.dtype, generator=generator)
    ut, udot, om = slerp(u0, u1, t)
    v = project_tangent(net(ut, t, a, theta), a, ut)
    loss = ((v - udot) ** 2).sum(-1).mean()
    with torch.no_grad():
        diag = {"omega_mean": om.mean().item(),
                "speed_err": (udot.norm(dim=-1, keepdim=True) - om).abs().max().item(),
                "theta_mean": theta.mean().item()}
    return loss, diag


@torch.no_grad()
def sample_u(net, A, theta, n_steps=16, generator=None):
    """Integrate the velocity field; a zero field preserves the initial direction."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    u = isotropic_direction(A, generator)
    h = 1.0 / n_steps
    for k in range(n_steps):
        t = torch.full_like(theta, k * h)
        v = project_tangent(net(u, t, A, theta), A, u)
        u = exp_step(u, v, h)
        u = unit(project_tangent(u, A))
    return u


@torch.no_grad()
def sample_A_prime(net, A, theta, n_steps=16, generator=None):
    """Sample a direction and combine it with the reference identity."""
    return compose(A, theta, sample_u(net, A, theta, n_steps, generator))


def uniform_cosine_theta(n, lb=0.5, device="cpu", generator=None):
    """Sample a cosine uniformly in [lb, 1] and return its angle."""
    if not 0 < lb <= 1 or n < 1:
        raise ValueError("require 0 < lb <= 1 and a positive sample count")
    c = lb + (1 - lb) * torch.rand(n, 1, device=device, generator=generator)
    return torch.arccos(c)
