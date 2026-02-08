_base_ = [
    "../base/mmdit_config.py",
    "../base/vae_config.py",
    "../base/data_config.py",
]

# -----------------------------
# Core finetune switches
# -----------------------------
use_multi_level_noise = True  # keep Flexible Forcing behavior
traj_ctrl = True
use_depth = True
use_seg = True
is_multi_view = True

# 5-view setup (drop CAM_BACK)
view_order = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
]

# cross-view neighborhood for 5-view ring
cross_view_list = [19, 28, 37]
# Use _delete_ to fully override 6-view base map during config merge.
mv_order_map = {
    "_delete_": True,
    0: [4, 1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 0],
}

# -----------------------------
# Dataset settings (Stage-3 style variable bucket)
# -----------------------------
num_frames = 33
depth_scale = 650

bucket_config = {
    "_delete_": True,
    "224x400r": {
        33: (1.0, 1),
    },
    "448x800r": {
        17: (1.0, 1),
    },
}

dataset = dict(
    pkl_path="data/t4_infos_train.pkl",
    transform_name="resize_crop",
    fps_max=24,
    memory_efficient=False,
    view_order=view_order,
    use_depth=use_depth,
    use_seg=use_seg,
    traj_ctrl=traj_ctrl,
    num_frames=num_frames,
    max_depth=100,
    dataset_name="t4",
    scene_token="scene_tokens",
    depth_scale=depth_scale,
    seg_png_format=True,
    depth_png_format=True,
    opencv_num_threads=1,
    seg_root="/mnt/nvme3/T4_datasets_sam3",
    depth_root="/mnt/nvme1/data/T4_datasets_priorda_depth",
    # robust mapping: supports both OmniNWM names and raw T4 names
    t4_camera_map={
        "CAM_FRONT_LEFT": "CAM_FRONT_LEFT_WIDE",
        "CAM_FRONT": "CAM_FRONT",
        "CAM_FRONT_RIGHT": "CAM_FRONT_RIGHT_WIDE",
        "CAM_BACK_RIGHT": "CAM_BACK_RIGHT_WIDE",
        "CAM_BACK_LEFT": "CAM_BACK_LEFT_WIDE",
        "CAM_FRONT_LEFT_WIDE": "CAM_FRONT_LEFT_WIDE",
        "CAM_FRONT_RIGHT_WIDE": "CAM_FRONT_RIGHT_WIDE",
        "CAM_BACK_RIGHT_WIDE": "CAM_BACK_RIGHT_WIDE",
        "CAM_BACK_LEFT_WIDE": "CAM_BACK_LEFT_WIDE",
    },
    # use the same 3 ego masks as inference config
    ray_mask_zeroing=True,
    ray_mask_paths={
        "CAM_FRONT": "outputs/ego_car_masks/final/front_mask_448x800.png",
        "CAM_BACK_LEFT": "outputs/ego_car_masks/final/back_left_mask_448x800.png",
        "CAM_BACK_RIGHT": "outputs/ego_car_masks/final/back_right_mask_448x800.png",
    },
    video_attr_list=[
        dict(height=224, width=400, frames=33),
        dict(height=448, width=800, frames=17),
    ],
)

condition_config = dict(
    i2v_head=5,
)

# -----------------------------
# Model settings
# -----------------------------
in_channels = 192
grad_ckpt_settings = (100, 100)
model = dict(
    # Set to your stage-3 / released OmniNWM checkpoint
    from_pretrained="pretrained/ckpt/dit/model",
    strict_load=False,
    fused_qkv=False,
    use_liger_rope=True,
    grad_ckpt_settings=grad_ckpt_settings,
    in_channels=in_channels,
    use_depth=use_depth,
    use_seg=use_seg,
    use_multi_level_noise=use_multi_level_noise,
    mv_order_map=mv_order_map,
    cross_view_list=cross_view_list,
)

# -----------------------------
# Optimization settings
# -----------------------------
lr = 1e-5
eps = 1e-15

optim = dict(
    cls="HybridAdam",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
    adamw_mode=True,
)

warmup_steps = 0
update_warmup_steps = True
use_cosine_scheduler = False

grad_clip = 1.0
accumulation_steps = 1
ema_decay = None

# Freeze strategy:
# 1) freeze most transformer blocks
# 2) train input adapters + final layer
# 3) optionally train cross-view modules with lower LR
freeze_strategy = dict(
    enable=True,
    freeze_all=True,
    freeze_transformer_blocks=True,
    unfreeze_patterns=[
        "img_in*",
        "cond_in*",
        "traj_cond_in*",
        "time_in*",
        "final_layer*",
    ],
    unfreeze_cross_view=True,
    param_groups=[
        dict(
            name="adapter_and_head",
            lr_mult=1.0,
            patterns=[
                "img_in*",
                "cond_in*",
                "traj_cond_in*",
                "time_in*",
                "final_layer*",
            ],
        ),
        dict(
            name="cross_view",
            lr_mult=0.5,
            patterns=[
                "single_blocks.*.q_mv_proj*",
                "single_blocks.*.k_mv_proj*",
                "single_blocks.*.v_mv_mlp*",
                "single_blocks.*.linear_mv*",
                "single_blocks.*.connector*",
            ],
        ),
    ],
)

# -----------------------------
# Runtime settings
# -----------------------------
prefetch_factor = 2
num_workers = 4
persistent_workers = True
num_bucket_build_workers = 2

# Disable first-step torch.compile overhead for faster startup.
compile_ae_encoder = False

dtype = "bf16"
plugin = "zero2"
plugin_config = dict(
    reduce_bucket_size_in_m=128,
    overlap_allgather=False,
)

grad_checkpoint = True
pin_memory_cache_pre_alloc_numels = None
async_io = False

seed = 42
outputs = "./outputs/t4_finetune"
epochs = 30
log_every = 10
# Save checkpoints more frequently (by update steps, not by epoch).
# With accumulation_steps=1, this is effectively every 100 iterations.
ckpt_every = 100
keep_n_latest = 20
save_master_weights = True
load_master_weights = True

# Optional resume.
# load = "/path/to/last_checkpoint"
# start_from_scratch = False

# Trackers
wandb = False
wandb_project = "omninwm"

mlflow = True
mlflow_experiment = "omninwm_t4_shared"
mlflow_run_name = "omninwm_t4_phaseA"
mlflow_tracking_uri = "sqlite:///mlruns_omninwm/mlflow.db"
mlflow_artifact_location = "mlruns_omninwm/artifacts"
mlflow_log_checkpoints = False
mlflow_tags = dict(
    project="omninwm",
    dataset="t4",
    task="finetune_stage3_5view",
    views="5",
    occ_head="frozen_or_unused",
)
