"""Batched spherical geometry for vectors of shape (B, d).

Identity anchors A and tangent directions u are unit vectors with u orthogonal
to A. Callers supply normalized inputs unless a function normalizes its output.
"""
import torch

EPS = 1e-6


def unit(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(EPS)


def project_tangent(v, A, u=None):
    """Project v orthogonally to A and, optionally, u.

    A and u must be orthogonal unit vectors when both are supplied.
    """
    v = v - (v * A).sum(-1, keepdim=True) * A
    if u is not None:
        v = v - (v * u).sum(-1, keepdim=True) * u
    return v


def isotropic_direction(A, generator=None):
    """Sample a uniform unit direction orthogonal to A."""
    n = torch.randn(A.shape, device=A.device, dtype=A.dtype, generator=generator)
    return unit(project_tangent(n, A))


def residual_direction(a, b):
    """Recover similarity, angle, and direction from a normalized identity pair."""
    s = (a * b).sum(-1, keepdim=True)
    theta = torch.arccos(s.clamp(-1 + EPS, 1 - EPS))
    u = unit(b - s * a)
    return s, theta, u


def slerp(u0, u1, t):
    """Return the geodesic point, velocity, and arc angle.

    Endpoints u0 and u1 must be unit vectors in the same tangent space.
    """
    cos_om = (u0 * u1).sum(-1, keepdim=True).clamp(-1 + EPS, 1 - EPS)
    om = torch.arccos(cos_om)
    so = torch.sin(om).clamp_min(EPS)
    ut = (torch.sin((1.0 - t) * om) * u0 + torch.sin(t * om) * u1) / so
    udot = om * (-torch.cos((1.0 - t) * om) * u0 + torch.cos(t * om) * u1) / so
    return ut, udot, om


def exp_step(u, v, h):
    """Take an exponential-map step of arc length h * ||v||."""
    nv = v.norm(dim=-1, keepdim=True).clamp_min(EPS)
    ang = h * nv
    return torch.cos(ang) * u + torch.sin(ang) * (v / nv)


def compose(A, theta, u):
    """Combine an identity anchor, angle, and tangent direction."""
    return torch.cos(theta) * A + torch.sin(theta) * u
