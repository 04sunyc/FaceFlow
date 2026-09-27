from __future__ import annotations

import argparse
from scripts.path_args import data_path
import hashlib
import importlib.util
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_CACHE = None
DEFAULT_CHECKPOINT = None
DEFAULT_OUT = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_model(checkpoint: Path, device: torch.device, iresnet_source: Path) -> torch.nn.Module:
    if not iresnet_source.is_file():
        raise FileNotFoundError(f"missing UIFace/ElasticFace model code: {iresnet_source}")
    spec = importlib.util.spec_from_file_location("uiface_iresnet", iresnet_source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {iresnet_source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = module.iresnet100(num_features=512)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def read_cache(cache_dir: Path) -> tuple[dict[str, Any], np.memmap, np.memmap]:
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing cache metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = tuple(int(x) for x in metadata.get("shape", []))
    if len(shape) != 4 or shape[1:] != (112, 112, 3):
        raise ValueError(f"expected cache shape (N,112,112,3), got {shape}")
    count = int(metadata["num_samples"])
    if shape[0] != count:
        raise ValueError(f"metadata count/shape mismatch: {count} vs {shape}")
    image_name = metadata.get("files", {}).get("images", "images.uint8.mmap")
    label_name = metadata.get("files", {}).get("labels", "labels.int64.mmap")
    image_path = cache_dir / image_name
    label_path = cache_dir / label_name
    images = np.memmap(image_path, mode="r", dtype=np.uint8, shape=shape)
    labels = np.memmap(label_path, mode="r", dtype=np.int64, shape=(count,))
    return metadata, images, labels


def extract(
    cache_dir: Path,
    checkpoint: Path,
    output: Path,
    device: torch.device,
    batch_size: int,
    limit: int | None,
    iresnet_source: Path,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    metadata, images, labels = read_cache(cache_dir)
    count = len(labels) if limit is None else min(int(limit), len(labels))
    if count < 1:
        raise ValueError("selected cache is empty")
    if limit is not None and int(limit) < 1:
        raise ValueError("--limit must be positive")

    output.mkdir(parents=True, exist_ok=True)
    emb_path = output / "embeddings.npy"
    labels_path = output / "labels.npy"
    manifest_path = output / "manifest.json"
    if any(path.exists() for path in (emb_path, labels_path, manifest_path)):
        raise FileExistsError(
            f"output already contains extraction artifacts: {output}; "
            "choose a new directory"
        )

    model = load_model(checkpoint, device, iresnet_source)
    embeddings = np.empty((count, 512), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(start + batch_size, count)
            batch = torch.from_numpy(np.array(images[start:end], copy=True))
            batch = batch.permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
            batch = (batch - 127.5) / 127.5
            features = F.normalize(model(batch), dim=1)
            embeddings[start:end] = features.float().cpu().numpy()
            if start == 0 or end == count or end % (batch_size * 100) == 0:
                print(f"{end}/{count}", flush=True)

    labels_out = np.asarray(labels[:count], dtype=np.int64).copy()
    atomic_npy(emb_path, embeddings)
    atomic_npy(labels_path, labels_out)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "embedding_model": "ElasticFace-Arc R100",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "preprocess": {
            "source": "decoded C-WF HuggingFace image Parquet cache",
            "input_shape": [112, 112, 3],
            "channel_order": "RGB",
            "scale": "(pixel - 127.5) / 127.5",
            "output": "L2-normalized float32",
        },
        "cache": {
            "path": str(cache_dir.resolve()),
            "metadata_sha256": sha256_file(cache_dir / "metadata.json"),
            "fingerprint": metadata.get("fingerprint"),
            "source_format": metadata.get("source_format"),
            "source_root": metadata.get("source_root"),
        },
        "rows": count,
        "identities": int(len(np.unique(labels_out))),
        "embedding_shape": list(embeddings.shape),
        "label_shape": list(labels_out.shape),
        "embeddings": str(emb_path),
        "labels": str(labels_path),
        "embeddings_sha256": sha256_file(emb_path),
        "labels_sha256": sha256_file(labels_path),
        "device": str(device),
        "batch_size": batch_size,
        "limit": limit,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=data_path, default=None, required=True)
    parser.add_argument("--checkpoint", type=data_path, default=None, required=True)
    parser.add_argument("--output", type=data_path, default=None, required=True)
    parser.add_argument("--iresnet-source", type=data_path, required=True, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="positive smoke-test row limit")
    args = parser.parse_args()
    extract(
        args.cache_dir,
        args.checkpoint,
        args.output,
        torch.device(args.device),
        args.batch_size,
        args.limit,
        args.iresnet_source,
    )


if __name__ == "__main__":
    main()
