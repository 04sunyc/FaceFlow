"""Same-identity pairs and angle-balanced sampling of normalized embeddings."""
import numpy as np
import torch


def load_embeddings(emb_path, label_path, dtype=np.float32):
    emb = np.load(emb_path)
    lab = np.load(label_path)
    assert emb.ndim == 2 and lab.ndim == 1 and len(emb) == len(lab), \
        f"shape mismatch: embeddings {emb.shape}, labels {lab.shape}"
    norms = np.linalg.norm(emb.astype(np.float32), axis=1)
    emb = emb.astype(dtype)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    return emb, lab.astype(np.int64), norms


def build_pairs(emb, labels, min_cos=0.10, max_pairs_per_id=2000, seed=0, verbose=True,
                max_cos=0.9999):
    """Return pair indices and cosines in [min_cos, max_cos].

    The upper cutoff excludes near-duplicates with degenerate residual directions.
    Pairs are capped per identity before angle-balanced sampling.
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(labels, kind="stable")
    lab_sorted = labels[order]
    bounds = np.flatnonzero(np.diff(lab_sorted)) + 1
    groups = np.split(order, bounds)

    I, J, C = [], [], []
    dropped = dup = 0
    for g in groups:
        n = len(g)
        if n < 2:
            continue
        X = emb[g]
        G = X @ X.T
        iu, ju = np.triu_indices(n, k=1)
        c = G[iu, ju]
        keep = (c >= min_cos) & (c <= max_cos)
        dropped += int((~keep).sum())
        dup += int((c > max_cos).sum())
        iu, ju, c = iu[keep], ju[keep], c[keep]
        if len(c) > max_pairs_per_id:
            sel = rng.choice(len(c), max_pairs_per_id, replace=False)
            iu, ju, c = iu[sel], ju[sel], c[sel]
        I.append(g[iu]); J.append(g[ju]); C.append(c)

    I = np.concatenate(I).astype(np.int32)
    J = np.concatenate(J).astype(np.int32)
    C = np.concatenate(C).astype(np.float32)
    if verbose:
        print(f"[pairs] kept {len(C):,}  |  dropped {dropped:,}"
              f" (cos > {max_cos}: {dup:,} near-duplicate pairs)"
              f"  |  cos range [{C.min():.3f}, {C.max():.3f}]  median {np.median(C):.3f}")
    return I, J, C


class PairSampler:
    """Balance batches across equal-width angle bins."""

    def __init__(self, I, J, C, n_bins=20, seed=0, device="cpu"):
        if n_bins < 1 or len(C) == 0 or len(I) != len(C) or len(J) != len(C):
            raise ValueError("nonempty aligned pairs and positive n_bins are required")
        if not np.isfinite(C).all() or np.any(np.abs(C) > 1):
            raise ValueError("pair cosines must be finite and in [-1, 1]")
        self.I = torch.from_numpy(I).to(device)
        self.J = torch.from_numpy(J).to(device)
        self.C = torch.from_numpy(C).to(device)
        self.device = device
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        theta = np.arccos(C.astype(np.float64))
        self.edges = np.linspace(theta.min(), theta.max(), n_bins + 1)
        b = np.clip(np.digitize(theta, self.edges) - 1, 0, n_bins - 1)
        self.bins = [torch.from_numpy(np.flatnonzero(b == k).astype(np.int64)).to(device)
                     for k in range(n_bins)]
        self.bins = [x for x in self.bins if len(x) > 0]
        self.sizes = [len(x) for x in self.bins]

    def sample(self, bs):
        if bs < 1:
            raise ValueError("batch size must be positive")
        per, rem = divmod(bs, len(self.bins))
        extra = set(torch.randperm(len(self.bins), generator=self.g)[:rem].tolist())
        picks = []
        for k, idx in enumerate(self.bins):
            m = per + int(k in extra)
            if m == 0:
                continue
            sel = torch.randint(len(idx), (m,), generator=self.g)
            picks.append(idx[sel.to(idx.device)])
        p = torch.cat(picks)
        p = p[torch.randperm(len(p), generator=self.g).to(p.device)]
        return self.I[p].long(), self.J[p].long()


def split_identities(labels, val_frac=0.05, seed=0):
    """Return training and held-out masks with disjoint identity sets."""
    rng = np.random.default_rng(seed)
    ids = np.unique(labels)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val].tolist())
    is_val = np.array([l in val_ids for l in labels])
    return ~is_val, is_val
