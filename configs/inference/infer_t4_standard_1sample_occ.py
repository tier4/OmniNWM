_base_ = [
    "../base/occ_config.py",
]

# Standard inference config for T4 - 1 sample with OCC visualization and fixed 3x2 layout
calculate_reward = False
use_multi_level_noise = False
use_low_men_vae_infer = True
traj_ctrl = True
use_depth = True
use_seg = True
depth_scale = 650

view_order = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
]

depth = 19
depth_single_blocks = 38
cross_view_list = list(range(depth, depth_single_blocks, 9))
is_multi_view = True
mv_order_map = {
    0: [4, 1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3, 0],
}
num_round = 1
num_frames = 33
height = 448
width = 800
start_index = 0
end_index = 1
batch_size = 1

dataset = dict(
    type="nuscenes_video",
    pkl_path="outputs/t4_candidate_pickles/pickles/candidate_14_start_33.pkl",
    transform_name="resize_crop",
    fps_max=24,
    vmaf=False,
    memory_efficient=False,
    view_order=view_order,
    use_depth=use_depth,
    use_seg=use_seg,
    depth_scale=depth_scale,
    seg_png_format=True,
    depth_png_format=True,
    seg_root="/mnt/nvme3/T4_datasets_sam3",
    depth_root="/mnt/nvme1/data/T4_datasets_priorda_depth",
    seg_class_map={
        0: 0, 1: 0, 2: 5, 3: 3, 4: 3, 5: 3, 6: 5, 7: 4, 8: 0,
        9: 3, 10: 3, 11: 1, 12: 0, 13: 7, 14: 8, 15: 0, 16: 8, 17: 2,
    },
    t4_camera_map={
        "CAM_FRONT_LEFT": "CAM_FRONT_LEFT_WIDE",
        "CAM_FRONT": "CAM_FRONT_WIDE",
        "CAM_FRONT_WIDE": "CAM_FRONT_WIDE",
        "CAM_FRONT_RIGHT": "CAM_FRONT_RIGHT_WIDE",
        "CAM_BACK_RIGHT": "CAM_BACK_RIGHT_WIDE",
        "CAM_BACK_LEFT": "CAM_BACK_LEFT_WIDE",
    },
    ray_mask_zeroing=True,
    ray_mask_paths={
        "CAM_FRONT": "outputs/ego_car_masks/final/front_mask_448x800.png",
        "CAM_BACK_LEFT": "outputs/ego_car_masks/final/back_left_mask_448x800.png",
        "CAM_BACK_RIGHT": "outputs/ego_car_masks/final/back_right_mask_448x800.png",
    },
    traj_ctrl=traj_ctrl,
    max_depth=100,
    num_frames=num_frames,
    height=height,
    width=width,
    is_train=False,
    dataset_name="t4",
)

grad_ckpt_settings = (100, 100)
condition_config = dict(i2v_head=5)
in_channels = 192

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
prefetch_factor = 4
num_workers = 2
num_bucket_build_workers = 16
dtype = "bf16"
plugin = None
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
show_step_bar = True
step_log_every = 5
seed = 42
outputs = "./outputs/t4_standard_candidate14_start33_occ"
save_dir = "./outputs/t4_standard_candidate14_start33_occ"
epochs = 10000
log_every = 1
ckpt_every = 100
keep_n_latest = 200000
wandb_project = "omninwm"
save_master_weights = True
load_master_weights = True
load = None
start_from_scratch = True

# OCC model config (copied from base/occ_config.py) for explicit availability.
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 40]
lss_downsample = [4, 4, 4]

voxel_size = 1.0
voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_channels = [80, 160, 320, 640]
empty_idx = 0
num_cls = 17
visible_mask = False

cascade_ratio = 4
sample_from_voxel = True
sample_from_img = True

data_config = {
    'cams': view_order,
    'Ncams': len(view_order),
    'input_size': (896, 1600),
    'src_size': (900, 1600),
    'resize': (0.0, 0.0),
    'rot': (0.0, 0.0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

numC_Trans = 80
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)

occ_model = dict(
    type='OccNet',
    loss_norm=True,
    pretrained="pretrained/ckpt/occ/occ.pth",
    img_backbone=dict(
        pretrained=None,
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        with_cp=True,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        loss_depth_weight=3.,
        loss_depth_type='kld',
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        vp_megvii=False),
    occ_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        n_input_channels=numC_Trans,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    occ_encoder_neck=dict(
        type='FPN3D',
        with_cp=True,
        in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
    ),
    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        cascade_ratio=cascade_ratio,
        sample_from_voxel=sample_from_voxel,
        sample_from_img=sample_from_img,
        final_occ_size=occ_size,
        fine_topk=15000,
        empty_idx=empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0,
        ),
    ),
    empty_idx=empty_idx,
)
