# FaceFlow

Flow matching for learning intra-class variation in face embeddings.

## Setup

Python 3.11+.

```bash
pip install -r requirements.txt
cp paths.example.env paths.env
```

Fill in the paths in `paths.env`, then load them:

```bash
set -a; source paths.env; set +a
```

## Training

Inputs: embeddings `(N, 512)` and integer identity labels `(N,)`, saved as NumPy arrays.

```bash
python train.py --emb "$EMBEDDINGS" --labels "$LABELS" \
  --out "$TRAIN_OUTPUT" --seed 1337
```

## Sampling

```bash
python sample.py --bank "$REFERENCE_BANK" --checkpoint "$FLOW_CHECKPOINT" \
  --output "$CONTEXT_DIR" --identities 10000 --samples 50
```

Outputs embedding conditions, not images. Use `--device cpu` to sample without CUDA.
Training and sampling require new output directories.

## Image generation

Requires an external UIFace adapter and weights, configured through `paths.env`.

```bash
bash scripts/generate.sh
```

## Tests

```bash
python tests.py
python -m unittest discover -s scripts/tests -v
```

See each script's `--help` for options.
