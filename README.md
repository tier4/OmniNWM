<div align="center">

# OmniNWM

**Omniscient Driving Navigation World Models**

[![Paper](https://img.shields.io/badge/Paper-Arxiv-red?style=for-the-badge&logo=arxiv)](https://arxiv.org/abs/2510.18313)
[![Project Page](https://img.shields.io/badge/Project-Page-green?style=for-the-badge&logo=google-chrome)](https://arlo0o.github.io/OmniNWM/)
[![Huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-yellow?style=for-the-badge)](https://huggingface.co/Arlolo0/OmniNWM/tree/main)
<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-orange?style=for-the-badge)](LICENSE) -->

> **OmniNWM** is a unified panoramic navigation world model that advances autonomous driving simulation by jointly generating multi-modal states (RGB, semantics, depth, 3D occupancy), enabling precise action control via normalized Plücker ray-maps, and facilitating closed-loop evaluation through occupancy-based dense rewards.

---

![Teaser](assets/teaser.png)

</div>

<br>

## ✨ Key Features

| Feature | Description |
|-----------|-------------|
| **Multi-modal Generation** | Jointly generates RGB, semantic, depth, and 3D occupancy in panoramic views |
| **Precise Camera Control** | Normalized Plücker ray-maps for pixel-level trajectory interpretation |
| **Long-term Stability** | Flexible forcing strategy enables auto-regressive generation beyond GT length |
| **Closed-loop Evaluation** | Occupancy-based dense rewards enable realistic driving policy evaluation |
|**Zero-shot Generalization** | Transfers across datasets and camera configurations without fine-tuning |

---

## 🏗️ Architecture

![Architecture](assets/overall.png)

---

## 💥 News
- **[2026/02]**  Implementation details of [OmniNWM-VLA](omninwm/models/OmniNWM-VLA) released!
- **[2026/02]**  Training/Inference code and pre-trained weights released!
- **[2025/10]**  Paper available on [arXiv](https://arxiv.org/abs/2510.18313).
- **[2025/09]**  Project demo live on the [Project Page](https://arlo0o.github.io/OmniNWM/).

---

## 🛠️ Quickstart

### 1. Installation



> **Manual Patch Required**: After installation, you must manually patch `transformers` for compatibility. See step 4 below.

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Ma-Zhuang/OmniNWM.git
    cd OmniNWM
    ```

2.  **Create directories**
    ```bash
    mkdir -p pretrained data
    ```

3.  **Install dependencies** (Recommended: `torch >= 2.4.0`)
    ```bash
    pip install -v -e .
    pip install "huggingface_hub[cli]"
    ```

4.  **Apply Patch** 
    Locate `transformers/modeling_utils.py` (usually in your conda env `site-packages`) and modify the version check:
    ```python
    # Find this line:
    if self._tp_plan is not None and is_torch_greater_or_equal("2.3"):
    
    # Change to:
    if self._tp_plan is not None and is_torch_greater_or_equal("2.5"):
    ```

### 2. Model Download

Download the official checkpoints and auxiliary models.

**OmniNWM Weights:**
```bash
pip install "huggingface_hub[cli]"
huggingface-cli download Arlolo0/OmniNWM --local-dir ./pretrained
```

**Open-Sora-v2 Weights:**
```bash
huggingface-cli download hpcai-tech/Open-Sora-v2 --local-dir ./pretrained
```

### 3. Data Preparation

1.  **nuScenes Dataset**: Download **Trainval** splits (Full dataset v1.0) from the [official website](https://www.nuscenes.org/download) and place in `./data/nuscenes`.
2.  **Depth Annotations**: Download from [HuggingFace](https://huggingface.co/datasets/Arlolo0/12HZ-Depth/tree/main).
3.  **Segmentation Annotations**: Download from [HuggingFace](https://huggingface.co/datasets/Arlolo0/12HZ-Segmentation/tree/main).

**Expected Directory Structure:**
```none
OmniNWM
├── assets
├── build
├── configs
├── data
│   ├── nuscenes
│   │   ├── samples
│   │   │   ├── CAM_BACK
│   │   │   ├── CAM_BACK_LEFT
│   │   │   ├── ...
│   │   │   ├── CAM_FRONT_RIGHT
│   │   ├── sweeps
│   │   │   ├── CAM_BACK
│   │   │   ├── CAM_BACK_LEFT
│   │   │   ├── ...
│   │   │   ├── CAM_FRONT_RIGHT
│   ├── nuscenes_12hz_depth_unzip
│   │   ├── adf04...
│   │   ├── adf06...
│   │   ├── ...
│   │   ├── ecd00...
│   ├── nuscenes_seg
│   │   ├── samples_seg
│   │   │   ├── CAM_BACK
│   │   │   ├── CAM_BACK_LEFT
│   │   │   ├── ...
│   │   │   ├── CAM_FRONT_RIGHT
│   │   ├── sweeps_seg
│   │   │   ├── CAM_BACK
│   │   │   ├── CAM_BACK_LEFT
│   │   │   ├── ...
│   │   │   ├── CAM_FRONT_RIGHT
│   ├── nuscenes_interp_12Hz_infos_train_with_bid_caption.pkl
│   ├── nuscenes_interp_12Hz_infos_val_with_bid_caption.pkl
├── omninwm
├── pretrained
│   ├── hunyuan_vae.safetensors
│   ├── occ.pth
│   ├── Open_Sora_v2.safetensors
├── tools
```

### 4.  OmniNWM-VLA
Check [OmniNWM-VLA](omninwm/models/OmniNWM-VLA) for more implementation details, including :

-  Integrated SSR-MIDI for tri-modal fusion 
-  ShareGPT format dataset generation pipeline 
-  codebase setup with nuScenes support

## 🚀 Usage

### Inference (Trajectory-to-Video)

Generate videos from trajectories. Ensure you update the checkpoint path in `configs/inference/infer.py` before running.

| Task | Command | Description |
| :--- | :--- | :--- |
| **Standard Inference** | `torchrun --nproc-per-node 8 tools/inference.py configs/inference/infer.py` | Multi-GPU, nuScenes 448x800, 6 cams, 33 frames |
| **OOD Nuplan Inference** | `torchrun --nproc-per-node 8 tools/inference.py configs/inference/infer_nuplan.py` | nuPlan dataset, manual trajectory input |
| **VLA Closed-Loop Test** | `torchrun --nproc-per-node 8 tools/inference.py configs/inference/infer_with_occ_vla.py` | Closed-loop test with occupancy prediction (321 frames) |

### Training

Training is divided into stages for stability.

```bash
# Stage 1: Small resolution, short video, single model output
bash dist_train_mlp.sh configs/train/stage_1.py

# Stage 2: Small resolution, short video, multi-model output
bash dist_train_mlp.sh configs/train/stage_2.py

# Stage 3: High resolution, long/short video, multi-model output
bash dist_train_mlp.sh configs/train/stage_3.py
```

### T4 Finetuning with Local MLflow

This repository now includes a complete T4 finetuning pipeline with local MLflow tracking.

#### T4 Data Sources

- RGB root: `--t4-root` (raw T4 scenes with `annotation/*.json` and camera files)
- Segmentation GT (SAM3): `dataset.seg_root` (default: `/mnt/nvme3/T4_datasets_sam3`)
- Depth GT (PriorDA): `dataset.depth_root` (default: `/mnt/nvme1/data/T4_datasets_priorda_depth`)
- Ego-mask zeroing: same 3 masks as T4 inference (`CAM_FRONT`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`)

#### Build Train/Val Pickles

```bash
python tools/prepare_t4_dataset_train.py \
  --t4-root /mnt/nvme2/T4_datasets \
  --train-output data/t4_infos_train.pkl \
  --val-output data/t4_infos_val.pkl
```

Generated pickle fields:
- `metadata`: split statistics
- `scene_tokens`: token sequence per scene (for clip sampling)
- `infos`: per-frame camera records (`cams[*].data_path`, intrinsics/extrinsics, timestamp, token)

Clip extraction in training:
- dense clips: sliding window with stride 1
- sparse clips: first `scene[::6]`, then sliding window with stride 1

#### Start Local MLflow UI

```bash
bash tools/run_mlflow_local.sh
```

Default local backend/artifacts:
- DB: `mlruns_omninwm/mlflow.db`
- Artifacts: `mlruns_omninwm/artifacts`
- UI: `http://<host>:5001` (default host `0.0.0.0`)

#### Phase A Finetune

```bash
NPROC_PER_NODE=8 bash tools/run_t4_finetune_mlflow.sh configs/train/t4_finetune.py
```

Key properties of `configs/train/t4_finetune.py`:
- 5-view setup (`CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK_RIGHT`, `CAM_BACK_LEFT`)
- Stage-3 style variable buckets: `224x400@33` and `448x800@17`
- Flexible forcing enabled (`use_multi_level_noise=True`)
- Partial-freeze finetuning (adapters + final layer + selected cross-view layers)
- Occupancy head not explicitly optimized (focus on video-generation objective)

#### Phase B Finetune (continued training)

```bash
NPROC_PER_NODE=8 bash tools/run_t4_finetune_mlflow.sh configs/train/t4_finetune_phaseB.py \
  --load /path/to/phaseA_checkpoint
```

`configs/train/t4_finetune_phaseB.py` keeps adapters/head trainable and additionally unfreezes late single-stream blocks (`single_blocks.34~37`) with lower LR.

#### Notes

- MLflow run names are fixed by default:
  - Phase A: `omninwm_t4_phaseA`
  - Phase B: `omninwm_t4_phaseB`
- Override config values from CLI if needed:
  - `--dataset.seg_root ...`
  - `--dataset.depth_root ...`
  - `--mlflow_tracking_uri ...`
  - `--mlflow_artifact_location ...`

---

## 📚 Citation

If you find OmniNWM useful for your research, please consider citing:

```bibtex
@article{li2025omninwm,
  title={OmniNWM: Omniscient Driving Navigation World Models},
  author={Li, Bohan and Ma, Zhuang and Du, Dalong and Peng, Baorui and Liang, Zhujin and Liu, Zhenqiang and Ma, Chao and Jin, Yueming and Zhao, Hao and Zeng, Wenjun and others},
  journal={arXiv preprint arXiv:2510.18313},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## ❤️ Acknowledgments

Built upon excellent open-source projects including [OpenSora](https://github.com/hpcaitech/Open-Sora) and [Qwen-VL](https://github.com/QwenLM/Qwen-VL).

<div align="center">

**🌟 Star us on GitHub if you like this project! 🌟**

</div>
