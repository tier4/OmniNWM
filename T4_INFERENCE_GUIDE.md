# T4 Dataset Inference Setup for OmniNWM

## Overview
This setup enables running OmniNWM inference on T4 datasets with all 3 inference tasks:
1. **Standard Inference** - Trajectory-to-video generation using T4 ground truth trajectories
2. **OOD/Nuplan Style Inference** - Manual trajectory input for out-of-distribution testing
3. **VLA Closed-Loop Inference** - Long-term generation with VLA policy (10 rounds)

## Environment Setup

### 1. UV Environment
```bash
# Environment is already created at /mnt/nvme2/OmniNWM/.venv
source .venv/bin/activate
```

### 2. Dependencies Installed
- PyTorch 2.4.0 with CUDA 12.1
- ColossalAI 0.5.0
- All required packages from requirements.txt
- OmniNWM package with CUDA extensions

### 3. Transformers Patch Applied
The required patch to `transformers/modeling_utils.py` has been applied:
- Changed `is_torch_greater_or_equal("2.3")` to `is_torch_greater_or_equal("2.5")`

## Downloaded Checkpoints

All checkpoints are downloaded to `./pretrained/`:

### OmniNWM Weights
- DiT model: `pretrained/ckpt/dit/model/` (6 shards, ~23GB)
- Occupancy model: `pretrained/ckpt/occ/occ.pth` (1.4GB)
- VLA model: `pretrained/ckpt/vla/OmniNWM-VLA/`

### Open-Sora-v2 Weights
- `pretrained/hunyuan_vae.safetensors` (471MB)
- `pretrained/Open_Sora_v2.safetensors` (23GB)

## T4 Dataset Preparation

### Dataset Structure
The T4 dataset is located at `/mnt/nvme2/T4_datasets/` with:
- 401 scenes
- ~1.5M frames total
- 6 cameras per frame (5 real + 1 black image for back camera)

### Preprocessing Script
`tools/prepare_t4_dataset.py`:
- Converts T4 annotations to OmniNWM-compatible pickle format
- Maps T4 camera names to OmniNWM view order
- Creates black images for missing back camera
- Extracts ego trajectories from ground truth

### Generated Files
- `data/t4_infos.pkl` - Main dataset file with 1,586,120 samples
- Black images at `data/CAM_BACK/black.jpg`

## Inference Configurations

### 1. Standard Inference
**Config:** `configs/inference/infer_t4_standard.py`
- Uses T4 ground truth trajectories
- 33 frames, 448x800 resolution
- Processes first 100 samples by default
- Output: `./outputs/t4_standard_inference/`

### 2. OOD/Nuplan Style Inference  
**Config:** `configs/inference/infer_t4_ood.py`
- Manual trajectory input (straight line example)
- 17 frames, 448x800 resolution
- For testing out-of-distribution scenarios
- Output: `./outputs/t4_ood_inference/`

### 3. VLA Closed-Loop Inference
**Config:** `configs/inference/infer_t4_vla.py`
- 10 rounds of auto-regressive generation
- 33 frames per round (321 total)
- Uses VLA policy for action control
- Output: `./outputs/t4_vla_closedloop/`

## Camera Configuration

### View Order
```python
view_order = [
    "CAM_FRONT_LEFT",   # T4: CAM_FRONT_LEFT_WIDE
    "CAM_FRONT",        # T4: CAM_FRONT
    "CAM_FRONT_RIGHT",  # T4: CAM_FRONT_RIGHT_WIDE
    "CAM_BACK_RIGHT",   # T4: CAM_BACK_RIGHT_WIDE
    "CAM_BACK",         # Black image (no back camera in T4)
    "CAM_BACK_LEFT",    # T4: CAM_BACK_LEFT_WIDE
]
```

### Back Camera Handling
Since T4 doesn't have a back camera, we use a **black image** instead:
- Black image created at: `data/CAM_BACK/black.jpg`
- Resolution: 1920x1080 (8-bit black)
- Uses FRONT camera intrinsics/extrinsics as placeholder

## Running Inference

### Option 1: Run All Tasks
```bash
source .venv/bin/activate
cd /mnt/nvme2/OmniNWM

# Run all 3 inference tasks
python tools/run_t4_inference.py --task all --num-samples 5
```

### Option 2: Run Individual Tasks
```bash
# Standard inference
python tools/inference.py configs/inference/infer_t4_standard.py

# OOD inference
python tools/inference.py configs/inference/infer_t4_ood.py

# VLA closed-loop inference
python tools/inference.py configs/inference/infer_t4_vla.py
```

### Option 3: Multi-GPU Inference
```bash
# Standard inference on 8 GPUs
torchrun --nproc-per-node 8 tools/inference.py configs/inference/infer_t4_standard.py

# OOD inference
torchrun --nproc-per-node 8 tools/inference.py configs/inference/infer_t4_ood.py

# VLA closed-loop inference  
torchrun --nproc-per-node 8 tools/inference.py configs/inference/infer_t4_vla.py
```

## Output Structure

Each inference task creates:
```
outputs/
├── t4_standard_inference/
│   ├── video/           # Generated videos
│   ├── image/           # Sample frames
│   └── comparisons/     # Side-by-side with originals
├── t4_ood_inference/
│   └── ...
└── t4_vla_closedloop/
    └── ...
```

## Visualization and Comparison

### Compare Generated vs Original
```bash
python tools/compare_outputs.py \
    --output-dir outputs/t4_standard_inference \
    --t4-root /mnt/nvme2/T4_datasets \
    --num-samples 5
```

This creates:
- `comparisons/` directory with side-by-side videos
- `grid_visualization.jpg` - Grid of first frames

## Important Notes

### Memory Requirements
- Standard inference: ~40GB GPU memory
- VLA closed-loop: ~60GB GPU memory (due to longer sequences)
- Recommended: Use 8 GPUs with gradient checkpointing enabled

### T4 Dataset Limitations
1. **No depth annotations** - Set `use_depth=False` in configs
2. **No segmentation annotations** - Set `use_seg=False` in configs
3. **No back camera** - Using black image placeholder
4. **Different camera intrinsics** - T4 uses different resolution than nuScenes

### Trajectory Extraction
Trajectories are extracted from T4 ego poses:
- X, Y positions relative to first frame
- Heading (yaw) from quaternion rotation
- Used as input for trajectory-conditioned generation

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` to 1 (already set)
- Reduce `num_frames` (e.g., 17 instead of 33)
- Enable `use_low_men_vae_infer=True` (already set)
- Use more GPUs with `torchrun`

### Missing Camera Data
If a scene is missing camera data, it's skipped during preprocessing. The pickle file contains only valid samples.

### Black Camera Issues
The black image for back camera is created during preprocessing. If missing, regenerate with:
```bash
python tools/prepare_t4_dataset.py --t4-root /mnt/nvme2/T4_datasets --output data/t4_infos.pkl
```

## Next Steps

1. **Run inference** using the commands above
2. **Compare outputs** using the comparison script
3. **Visualize results** with the grid visualization
4. **Adjust trajectories** in OOD config for different scenarios
5. **Tune sampling parameters** (num_steps, shift, etc.) for quality/speed trade-off

## Files Created

- `configs/inference/infer_t4_standard.py` - Standard inference config
- `configs/inference/infer_t4_ood.py` - OOD inference config  
- `configs/inference/infer_t4_vla.py` - VLA closed-loop config
- `tools/prepare_t4_dataset.py` - T4 dataset preprocessing
- `tools/run_t4_inference.py` - Inference runner script
- `tools/compare_outputs.py` - Comparison visualization script
- `data/t4_infos.pkl` - Preprocessed T4 dataset (1.5M samples)
- `setup.py` - Modified for relative paths
- `requirements.txt` - Updated for compatibility
