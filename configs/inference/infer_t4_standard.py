calculate_reward = False
use_multi_level_noise = False
use_low_men_vae_infer = True
traj_ctrl = True
use_depth = True  # Enable Prior-Depth-Anything
use_seg = True  # Enable SAM3 segmentation
depth_scale = 650  # Prior-Depth-Anything scale factor
view_order = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]
depth = 19
depth_single_blocks = 38
cross_view_list = list(range(depth, depth_single_blocks, 9))
is_multi_view = True
mv_order_map = {
    0: [5, 1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 5],
    5: [4, 0],
}
num_round = 1
num_frames = 33
height = 448
width = 800
start_index = 975  # Randomly selected sample
end_index = 976  # Process only 1 sample
batch_size = 1

# T4 camera mapping for depth/seg loading
T4_CAMERA_MAP = {
    "CAM_FRONT_LEFT": "CAM_FRONT_LEFT_WIDE",
    "CAM_FRONT": "CAM_FRONT",
    "CAM_FRONT_RIGHT": "CAM_FRONT_RIGHT_WIDE",
    "CAM_BACK_RIGHT": "CAM_BACK_RIGHT_WIDE",
    "CAM_BACK": "CAM_FRONT",  # Using front for back
    "CAM_BACK_LEFT": "CAM_BACK_LEFT_WIDE",
}

# Dataset settings for T4 with Prior-Depth-Anything and SAM3
dataset = dict(
    type="nuscenes_video",
    pkl_path="/mnt/nvme2/T4_processed/t4_infos_val.pkl",  # Use preprocessed T4 pickle
    transform_name="resize_crop",
    fps_max=24,
    vmaf=False,
    memory_efficient=False,
    view_order=view_order,
    use_depth=use_depth,
    use_seg=use_seg,
    depth_scale=depth_scale,  # Prior-Depth-Anything uses 650
    seg_png_format=True,  # SAM3 is stored as PNG
    depth_png_format=True,  # Prior-Depth-Anything is PNG
    seg_root="/mnt/nvme3/T4_datasets_sam3",  # SAM3 segmentation
    depth_root="/mnt/nvme1/data/T4_datasets_priorda_depth",  # Prior-Depth-Anything
    t4_camera_map=T4_CAMERA_MAP,  # Map OmniNWM camera names to T4 names
    traj_ctrl=traj_ctrl,
    max_depth=100,
    num_frames=num_frames,
    height=height,
    width=width,
    is_train=False,
    dataset_name="t4",
)

grad_ckpt_settings = (100, 100)

condition_config = dict(
    i2v_head=5,
)

# Use 192 channels (RGB + Depth + Seg)
in_channels = 192

# Define model components
model = dict(
    type="flux",
    from_pretrained="pretrained/ckpt/dit/model",
    strict_load=False,
    fused_qkv=False,
    use_liger_rope=True,
    grad_ckpt_settings=grad_ckpt_settings,
    in_channels=in_channels,
    hidden_size=3072,
    mlp_ratio=4.0,
    num_heads=24,
    depth=depth,
    depth_single_blocks=depth_single_blocks,
    axes_dim=[16, 56, 56],
    theta=10_000,
    qkv_bias=True,
    mv_order_map=mv_order_map,
    cross_view_list=cross_view_list,
    use_depth=use_depth,
    use_seg=use_seg,
    use_multi_level_noise=use_multi_level_noise,
)

ae = dict(
    type="hunyuan_vae",
    from_pretrained="pretrained/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)

is_causal_vae = True

# Optimization settings
lr = 1e-4
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

grad_clip = 1.0
accumulation_steps = 1
ema_decay = None

# Acceleration settings - single GPU
prefetch_factor = 4
num_workers = 2
num_bucket_build_workers = 16

dtype = "bf16"
plugin = None  # Disable distributed plugin for single GPU inference
plugin_config = None

grad_checkpoint = True

async_io = False

sampling_option = dict(
    num_frames=num_frames,
    num_steps=50,
    shift=True,
    temporal_reduction=4,
    is_causal_vae=True,
    method="i2v",
    seed=42,
    width=width,
    height=height,
    num_round=num_round,
)

# Other settings
seed = 42
outputs = "./outputs/t4_standard_inference"
save_dir = "./outputs/t4_standard_inference"
epochs = 10000
log_every = 1
ckpt_every = 100
keep_n_latest = 200000
wandb_project = "omninwm"

save_master_weights = True
load_master_weights = True
load = None
start_from_scratch = True

# Occupancy model (optional, for visualization)
occ = dict(
    type="occnet",
    from_pretrained="pretrained/ckpt/occ/occ.pth",
)

# VLA model (optional, for closed-loop)
vla = dict(
    type="omninwm_vla",
    from_pretrained="pretrained/ckpt/vla/OmniNWM-VLA",
)
