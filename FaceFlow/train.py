"""Train FaceFlow from ElasticFace embeddings and aligned identity labels.

Embeddings have shape (N, 512) and are normalized on load. Labels are an
integer array of shape (N,).
"""
import argparse, json, math, os, time
import numpy as np
import torch
from scripts.path_args import path_string

from FaceFlow.data import load_embeddings, build_pairs, PairSampler, split_identities
from FaceFlow.model import FaceFlow, EMA
from FaceFlow.flow import flow_loss


def get_args():
    p = argparse.ArgumentParser(description="Train the Face Flow velocity field.")
    p.add_argument("--emb", type=path_string, default=None, required=True)
    p.add_argument("--labels", type=path_string, default=None, required=True)
    p.add_argument("--out", type=path_string, default=None, required=True)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--mlp-ratio", type=int, default=4)
    p.add_argument("--min-cos", type=float, default=0.10,
                   help="minimum cosine similarity for same-identity training pairs")
    p.add_argument("--max-pairs-per-id", type=int, default=2000)
    p.add_argument("--theta-bins", type=int, default=20)
    p.add_argument("--val-frac", type=float, default=0.05, help="fraction of identities held out")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no-bf16", dest="bf16", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * step / max(1, args.warmup)
    p = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def main():
    args = get_args()
    os.makedirs(args.out, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)

    print(f"[data] loading {args.emb}")
    emb, lab, norms = load_embeddings(args.emb, args.labels)
    print(f"[data] {len(emb):,} embeddings | {len(np.unique(lab)):,} identities | d = {emb.shape[1]}")
    print(f"[data] input norm median {np.median(norms):.2f} "
          f"5th/95th percentiles {np.percentile(norms,5):.2f}/{np.percentile(norms,95):.2f}")

    tr_mask, va_mask = split_identities(lab, args.val_frac, args.seed)
    print(f"[data] training {tr_mask.sum():,} / held out {va_mask.sum():,} (identity-disjoint)")

    I, J, C = build_pairs(emb[tr_mask], lab[tr_mask], args.min_cos,
                          args.max_pairs_per_id, args.seed)
    np.save(os.path.join(args.out, "train_pair_cos.npy"), C)

    E = torch.from_numpy(emb[tr_mask]).to(dev)

    sampler = PairSampler(I, J, C, args.theta_bins, args.seed, device="cpu")
    print(f"[data] {len(sampler.bins)} angle bins, pairs per bin "
          f"min {min(sampler.sizes):,} / max {max(sampler.sizes):,}")

    net = FaceFlow(d=emb.shape[1], w=args.width, depth=args.depth,
                      mlp_ratio=args.mlp_ratio).to(dev)
    print(f"[model] {net.n_params()/1e6:.1f} M parameters")
    with torch.no_grad():
        probe = net(torch.randn(4, emb.shape[1], device=dev),
                    torch.rand(4, 1, device=dev),
                    torch.randn(4, emb.shape[1], device=dev),
                    torch.rand(4, 1, device=dev))
    assert probe.abs().max().item() == 0.0, "expected a zero-initialized velocity field"
    print("[model] zero-initialized velocity field verified")

    ema = EMA(net, args.ema)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd,
                            betas=(0.9, 0.95))
    use_amp = args.bf16 and dev.type == "cuda"
    json.dump(vars(args), open(os.path.join(args.out, "config.json"), "w"),
              indent=2, ensure_ascii=False)

    t0, run = time.time(), 0.0
    for step in range(1, args.steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = lr_at(step, args)
        i, j = sampler.sample(args.batch)
        a, b = E[i.to(dev)], E[j.to(dev)]
        if torch.rand(1).item() < 0.5:
            a, b = b, a
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, diag = flow_loss(net, a.float(), b.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
        opt.step()
        ema.update(net)
        run += loss.item()

        if step % args.log_every == 0:
            el = time.time() - t0
            print(f"step {step:>7,}/{args.steps:,}  loss {run/args.log_every:.5f}  "
                  f"lr {lr_at(step,args):.2e}  |grad| {gn:.2f}  "
                  f"Ω {diag['omega_mean']:.3f}  θ {diag['theta_mean']:.3f}  "
                  f"{step/el:.1f} it/s", flush=True)
            assert diag["speed_err"] < 1e-3, "Slerp speed does not match the geodesic arc angle"
            run = 0.0

        if step % args.ckpt_every == 0 or step == args.steps:
            torch.save({"step": step, "model": net.state_dict(),
                        "ema": ema.shadow, "args": vars(args)},
                       os.path.join(args.out, "ckpt.pt"))
            print(f"[ckpt] saved step {step}")

    print(f"[done] {(time.time()-t0)/60:.1f} minutes -> {args.out}/ckpt.pt")


if __name__ == "__main__":
    main()
