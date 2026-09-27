"""CPU checks for spherical geometry, zero-field sampling, and flow loss."""
import math, sys
import torch
from FaceFlow.geometry import (unit, project_tangent, isotropic_direction,
                              residual_direction, slerp, exp_step, compose)
from FaceFlow.model import FaceFlow
from FaceFlow.flow import sample_u, flow_loss

torch.manual_seed(0)
D, B = 512, 256
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


A = unit(torch.randn(B, D, dtype=torch.float64))
u0 = isotropic_direction(A)
u1 = isotropic_direction(A)
t = torch.rand(B, 1, dtype=torch.float64)

print("1  Base distribution and projection")
check("u0 is a unit vector", (u0.norm(dim=-1) - 1).abs().max() < 1e-12,
      f"max err {(u0.norm(dim=-1)-1).abs().max():.2e}")
check("u0 ⊥ A", (u0 * A).sum(-1).abs().max() < 1e-12,
      f"max |<u0,A>| {(u0*A).sum(-1).abs().max():.2e}")
w = torch.randn(B, D, dtype=torch.float64)
p1 = project_tangent(w, A, u0)
p2 = project_tangent(p1, A, u0)
check("projection is idempotent", (p1 - p2).abs().max() < 1e-12, f"max diff {(p1-p2).abs().max():.2e}")

print("2  Geodesic interpolation")
ut, udot, om = slerp(u0, u1, t)
check("|u_t| == 1", (ut.norm(dim=-1) - 1).abs().max() < 1e-11)
check("u_t ⊥ A", (ut * A).sum(-1).abs().max() < 1e-11)
check("|udot| == Omega (constant speed)", (udot.norm(dim=-1, keepdim=True) - om).abs().max() < 1e-10,
      f"max err {(udot.norm(dim=-1,keepdim=True)-om).abs().max():.2e}")
check("udot ⊥ u_t", (udot * ut).sum(-1).abs().max() < 1e-10)
check("udot ⊥ A", (udot * A).sum(-1).abs().max() < 1e-11)
e0, _, _ = slerp(u0, u1, torch.zeros(B, 1, dtype=torch.float64))
e1, _, _ = slerp(u0, u1, torch.ones(B, 1, dtype=torch.float64))
check("t=0 recovers u0", (e0 - u0).abs().max() < 1e-11)
check("t=1 recovers u1", (e1 - u1).abs().max() < 1e-11)
print(f"     Omega mean {om.mean().item():.4f} rad (high-dimensional limit: pi/2 = {math.pi/2:.4f}), "
      f"std {om.std().item():.4f}")

print("3  Residual decomposition and reconstruction")
a = unit(torch.randn(B, D, dtype=torch.float64))
b = unit(torch.randn(B, D, dtype=torch.float64))
s, th, ur = residual_direction(a, b)
check("compose(a, theta, u) == b", (compose(a, th, ur) - b).abs().max() < 1e-11,
      f"max err {(compose(a,th,ur)-b).abs().max():.2e}")
check("<b, a> == cos(theta)", ((compose(a, th, ur) * a).sum(-1, keepdim=True) - s).abs().max() < 1e-12)

print("4  Exponential-map step")
v = project_tangent(torch.randn(B, D, dtype=torch.float64), A, u0)
un = exp_step(u0, v, 0.1)
check("|u| == 1 after integration", (un.norm(dim=-1) - 1).abs().max() < 1e-11)
check("u remains orthogonal to A", (un * A).sum(-1).abs().max() < 1e-11)

print("5  Zero-field sampling")
net = FaceFlow(d=D, w=128, depth=3).double().eval()
raw = net(u0, t, A, th)
check("initial network output is zero", raw.abs().max() == 0, f"max |out| {raw.abs().max():.2e}")
g = torch.Generator().manual_seed(123)
torch.manual_seed(7)
us = sample_u(net, A, th, n_steps=16)
torch.manual_seed(7)
ub = isotropic_direction(A)
check("zero-field samples match initial directions within tolerance", (us - ub).abs().max() < 1e-12,
      f"max diff {(us-ub).abs().max():.2e}")
check("sampled directions have unit norm", (us.norm(dim=-1) - 1).abs().max() < 1e-11)
check("sampled directions are orthogonal to A", (us * A).sum(-1).abs().max() < 1e-11)

print("6  Loss and diagnostics")
loss, diag = flow_loss(net, a, b)
check("zero-field loss is positive", loss.item() > 0, f"loss {loss.item():.4f}")
check("speed_err is near zero", diag["speed_err"] < 1e-9, f"{diag['speed_err']:.2e}")

print()
print("All checks passed" if ok else "Some checks failed")
sys.exit(0 if ok else 1)
