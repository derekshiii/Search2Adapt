# VFM Setup

Search2Adapt uses two vision foundation models as frozen proposal generators inside the search space:

| Model | Role | Parameters | Conditioning |
|---|---|---|---|
| **BiomedParse v2** (3D) | semantic proposals | ~371M | text prompt |
| **SAM 3** | geometric / boundary proposals | ~862M | text prompt + bounding box |

Both run in **their own conda environments** and are launched as subprocesses by the `Executor` agent, so this package never imports `torch`, `hydra` or `sam3` at runtime. This keeps the core install light and isolates dependency conflicts.

---

## BiomedParse (3D)

### 1. Checkout

```bash
git clone https://github.com/microsoft/BiomedParse.git
cd BiomedParse
```

The checkout must contain the model code under `src/` and the hydra configuration under `configs/model/biomedparse_3D`.

### 2. Environment

```bash
conda create -n biomedparse python=3.11 -y
conda activate biomedparse
pip install -r requirements.txt
pip install -e .
pip install hydra-core          # if not already a dependency
```

### 3. Checkpoint

Download the 3D checkpoint and point `SEARCH2ADAPT_BIOMEDPARSE_CHECKPOINT` at it. The runner expects the hydra config `configs/model/biomedparse_3D` with
`_target_: src.model.biomedparse_3D.BiomedParseModel`.

### 4. Environment variables

```bash
export SEARCH2ADAPT_BIOMEDPARSE_ROOT=/path/to/BiomedParse
export SEARCH2ADAPT_BIOMEDPARSE_CHECKPOINT=/path/to/biomedparse_3D.ckpt
export SEARCH2ADAPT_BIOMEDPARSE_PYTHON=/path/to/envs/biomedparse/bin/python
# optional: override the config dir if it is not <root>/configs/model
export SEARCH2ADAPT_BIOMEDPARSE_CONFIG_DIR=/path/to/BiomedParse/configs/model
```

### 5. Smoke test

```bash
$SEARCH2ADAPT_BIOMEDPARSE_PYTHON -m search2adapt.vfm.biomedparse_3d_standalone \
  --input_dir  /tmp/bp_npz \
  --output_dir /tmp/bp_out \
  --checkpoint "$SEARCH2ADAPT_BIOMEDPARSE_CHECKPOINT" \
  --biomedparse_root "$SEARCH2ADAPT_BIOMEDPARSE_ROOT" \
  --gpus 0
```

> The vendored backend (`search2adapt/vfm/biomedparse_backend/`) is adapted from
> [microsoft/BiomedParse](https://github.com/microsoft/BiomedParse) (Apache-2.0). Unlike the upstream
> script it resolves the hydra config **by directory** rather than by working directory, so it works
> regardless of the current working directory.

---

## SAM 3

### 1. Checkout

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
```

### 2. Environment

SAM 3 requires a recent PyTorch with CUDA. Install it in its own environment:

```bash
conda create -n sam3 python=3.11 -y
conda activate sam3
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
```

The runner also needs `opencv-python` and `nibabel`.

### 3. Checkpoint

```bash
export SEARCH2ADAPT_SAM3_ROOT=/path/to/sam3
export SEARCH2ADAPT_SAM3_CHECKPOINT=/path/to/sam3/sam3/sam3.pt
export SEARCH2ADAPT_SAM3_PYTHON=/path/to/envs/sam3/bin/python
```

### 4. How 3D inference works

SAM 3 ships as a **video** predictor. The standalone runner:

1. loads the NIfTI volume and transposes it so the slice axis becomes the frame axis,
2. writes the slices as a frame sequence,
3. prompts the model once per organ (`text` and/or normalized `bounding_box`) on a **key slice**,
4. propagates the mask along the sequence in both directions,
5. reassembles the frames into a volume and writes `{case_id}.nii.gz`.

Each organ gets an **independent session**, so masks of one organ cannot leak into the next.

### 5. Config file format

The runner takes a JSON file mapping case name → prompts:

```json
{
  "AMOS_0001": {
    "image": "/path/to/imagesTr/AMOS_0001_0000.nii.gz",
    "slice_axis": 2,
    "prompts": [
      {"obj_id": 1, "text": "liver", "key_slice": 120},
      {"obj_id": 2, "text": "spleen", "key_slice": 130, "box": [40, 60, 220, 300]}
    ]
  }
}
```

| Field | Meaning |
|---|---|
| `image` | path to the input `.nii.gz` |
| `slice_axis` | axis to slice along (optional; defaults to the axis with the most slices) |
| `prompts[].obj_id` | integer identifier of the object within the case |
| `prompts[].text` | text prompt for the organ |
| `prompts[].box` | `[x0, y0, x1, y1]` bounding box in slice pixel coordinates (optional) |
| `prompts[].key_slice` | index of the slice the prompt is placed on |

### 6. Smoke test

```bash
$SEARCH2ADAPT_SAM3_PYTHON -m search2adapt.vfm.sam3_3d_standalone \
  --json_config /tmp/sam3_config.json \
  --output_dir  /tmp/sam3_out \
  --checkpoint  "$SEARCH2ADAPT_SAM3_CHECKPOINT" \
  --sam3_root   "$SEARCH2ADAPT_SAM3_ROOT" \
  --gpus 0
```

### GPU fallback

`SAM3Tool` tries the configured GPU list first (e.g. `0,1`). If the run fails with a NCCL or
out-of-memory error it automatically retries on a single GPU — preferring the last GPU, then the
first — and remembers the working configuration for the rest of the session.

---

## Dual-expert mode

`--initial-method dual_expert` runs BiomedParse first, derives per-organ bounding boxes from its
output, and then refines each organ with SAM 3 using text + box prompts. The two masks are fused
with a union strategy. This needs **both** environments to be configured.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: hydra` | BiomedParse env missing hydra | `pip install hydra-core` in that env |
| `ModuleNotFoundError: sam3` | `SEARCH2ADAPT_SAM3_ROOT` not on `sys.path` | set the variable, or pass `--sam3_root` |
| CUDA OOM in SAM 3 | two GPUs saturated | the automatic single-GPU fallback handles this |
| `Could not find config` | hydra config dir mismatch | set `SEARCH2ADAPT_BIOMEDPARSE_CONFIG_DIR` |
| Empty prompt map | no organs resolved for the case | check the label map produced by the Perceptor |
