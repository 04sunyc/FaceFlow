# FaceFlow

## Abstract

Generating synthetic data for face recognition (FR) training offers a practical alternative to authentic face datasets and supports privacy protection. Existing identity-conditioned generative models can synthesize photorealistic faces with high identity consistency, yet they typically exhibit limited intra-class diversity. Perturbation of the identity condition has been introduced to alleviate this limitation, but existing methods constrain only the perturbation magnitude and sample the perturbation direction uniformly at random in the identity embedding space, without modeling the intra-class variation of real faces. We propose FaceFlow, a flow-based model trained on same-identity pairs to learn an identity-conditioned flow whose endpoint follows the high-dimensional intra-class distribution. FaceFlow guides the perturbation of identity embeddings, which are then decoded by a pre-trained generator to construct synthetic face datasets. FR models trained on the resulting synthetic datasets outperform those trained on datasets constructed using state-of-the-art synthetic-data methods on standard benchmarks and surpass 94% average accuracy at a training scale of 0.5M synthetic images. These results show that FaceFlow enables the reliable generation of diverse intra-class samples while maintaining high identity consistency.

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
  --out "$TRAIN_OUTPUT"
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
