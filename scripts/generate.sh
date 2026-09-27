#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
: "${UIFACE_GENERATOR_ROOT:?Path to the external UIFace precomputed-context adapter}"
: "${DIFFUSION_CHECKPOINT:?Frozen UIFace EMA checkpoint}"
: "${DIFFUSION_CONFIG:?UIFace model configuration}"
: "${VQ_ENCODER:?Frozen autoencoder encoder weights}"
: "${VQ_DECODER:?Frozen autoencoder decoder weights}"
: "${CONTEXT_DIR:?Output directory of sample.py}"
: "${IMAGE_OUTPUT:?New directory for generated images}"
test ! -e "$IMAGE_OUTPUT"

exec "$PYTHON" "$UIFACE_GENERATOR_ROOT/sample_precomputed.py" \
  --config-path "$ROOT/generator_configs" --config-name generator \
  "checkpoint.path=$DIFFUSION_CHECKPOINT" \
  "diffusion_cfg_path=$DIFFUSION_CONFIG" \
  "VQEncoder_path=$VQ_ENCODER" "VQDecoder_path=$VQ_DECODER" \
  "sampling.precomputed_contexts_file=$CONTEXT_DIR/contexts.npy" \
  "sampling.context_ids_file=$CONTEXT_DIR/identity_ids.npy" \
  "sampling.save_dir=$IMAGE_OUTPUT" "sampling.seed=${SEED:-1337}"
