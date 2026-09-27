"""Conditional velocity MLP with adaLN-Zero residual blocks.

The modulation and output layers are zero-initialized, giving a zero initial
velocity field.
"""
import math
import torch
import torch.nn as nn


def fourier(x, n_freq=128, max_freq=1000.0):
    """Map scalar inputs of shape (B, 1) to (B, 2 * n_freq) Fourier features."""
    freqs = torch.exp(
        torch.linspace(0.0, math.log(max_freq), n_freq, device=x.device, dtype=x.dtype)
    )
    a = x * freqs * (2 * math.pi)
    return torch.cat([a.sin(), a.cos()], dim=-1)


class Block(nn.Module):
    def __init__(self, w, mlp_ratio=4):
        super().__init__()
        self.norm = nn.LayerNorm(w, elementwise_affine=False)
        h = int(w * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(w, h), nn.GELU(), nn.Linear(h, w))
        self.ada = nn.Linear(w, 3 * w)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, h, c):
        shift, scale, gate = self.ada(c).chunk(3, dim=-1)
        return h + gate * self.mlp(self.norm(h) * (1 + scale) + shift)


class FaceFlow(nn.Module):
    """Identity- and angle-conditioned velocity network."""

    def __init__(self, d=512, w=768, depth=8, mlp_ratio=4, n_freq=128,
                 theta_max=math.pi / 2):
        super().__init__()
        self.d, self.theta_max, self.n_freq = d, theta_max, n_freq
        self.inp = nn.Linear(d, w)
        f = 2 * n_freq
        self.emb_t = nn.Sequential(nn.Linear(f, w), nn.SiLU(), nn.Linear(w, w))
        self.emb_th = nn.Sequential(nn.Linear(f, w), nn.SiLU(), nn.Linear(w, w))
        self.emb_A = nn.Linear(d, w)
        self.blocks = nn.ModuleList([Block(w, mlp_ratio) for _ in range(depth)])
        self.out = nn.Linear(w, d)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, ut, t, A, theta):
        """Return unprojected (B, d) velocities; t and theta have shape (B, 1)."""
        c = (self.emb_t(fourier(t, self.n_freq))
             + self.emb_th(fourier(theta / self.theta_max, self.n_freq))
             + self.emb_A(A))
        h = self.inp(ut)
        for blk in self.blocks:
            h = blk(h, c)
        return self.out(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
