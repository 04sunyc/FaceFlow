#!/usr/bin/env python3
"""Sample identity embeddings with a trained FaceFlow model."""
from __future__ import annotations

import argparse
from scripts.path_args import data_path
import json
from pathlib import Path

import numpy as np
import torch

from FaceFlow.flow import sample_u, uniform_cosine_theta
from FaceFlow.geometry import compose, unit
from FaceFlow.model import FaceFlow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', type=data_path, required=True, help='UIFace reference embeddings', default=None)
    parser.add_argument('--checkpoint', type=data_path, required=True, default=None)
    parser.add_argument('--output', type=data_path, required=True, default=None)
    parser.add_argument('--identities', type=int, default=10000)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--lb', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if min(args.identities, args.samples, args.batch) < 1 or not 0 < args.lb <= 1:
        raise ValueError('invalid sample count, batch size, or lb')
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    bank = np.load(args.bank, mmap_mode='r')
    if bank.ndim != 2 or bank.shape[1] != 512 or len(bank) < args.identities:
        raise ValueError('bank must contain enough 512-D reference embeddings')
    reference = np.array(bank[:args.identities], dtype=np.float32)
    norms = np.linalg.norm(reference, axis=1, keepdims=True)
    if not np.isfinite(reference).all() or np.any(norms < 1e-8):
        raise ValueError('invalid reference embeddings')
    reference /= norms

    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = payload['args']
    net = FaceFlow(d=512, w=cfg['width'], depth=cfg['depth'],
                   mlp_ratio=cfg['mlp_ratio']).to(device).eval()
    net.load_state_dict(payload['ema'], strict=True)
    net.requires_grad_(False)
    checkpoint_step = int(payload['step'])
    del payload

    count = args.identities * args.samples
    angle_rng = torch.Generator(device='cpu').manual_seed(args.seed)
    direction_rng = torch.Generator(device=device).manual_seed(args.seed)
    theta = uniform_cosine_theta(count, args.lb, generator=angle_rng)

    args.output.mkdir(parents=True)
    contexts = np.lib.format.open_memmap(args.output / 'contexts.npy', mode='w+',
                                         dtype=np.float32, shape=(args.identities, args.samples, 512))
    flat = contexts.reshape(count, 512)
    with torch.inference_mode():
        for start in range(0, count, args.batch):
            stop = min(start + args.batch, count)
            ids = np.arange(start, stop) // args.samples
            anchor = unit(torch.from_numpy(reference[ids]).to(device))
            angles_batch = theta[start:stop].to(device)
            direction = sample_u(net, anchor, angles_batch, 16, direction_rng)
            result = compose(anchor, angles_batch, direction)
            flat[start:stop] = result.cpu().numpy()
    contexts.flush()
    np.save(args.output / 'identity_ids.npy', np.arange(args.identities, dtype=np.int64))
    np.save(args.output / 'reference_embeddings.npy', reference)
    np.save(args.output / 'angles_degrees.npy', torch.rad2deg(theta).numpy().reshape(args.identities, args.samples))
    manifest = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    manifest.update(method='FaceFlow', angle_law='uniform_cosine',
                    ode_steps=16, weights='ema',
                    checkpoint_step=checkpoint_step)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
