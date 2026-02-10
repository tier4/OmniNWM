_base_ = [
    "./t4_finetune.py",
]

# =============================================================================
# Deep Adaptation finetune for T4
# =============================================================================
# Phase A (t4_finetune.py) trained only adapters + cross-view modules for
# ~4800 steps but loss plateaued at ~0.039 since step ~1600.  The frozen
# backbone cannot represent the T4 domain well enough.
#
# This config unfreezes a significant portion of the network while keeping
# early double-stream layers frozen to preserve low-level features.
# It loads from a Phase A checkpoint to avoid wasting prior training.
#
# Unfrozen components (in addition to adapters):
#   - All 38 single_blocks   (single-stream transformer, bulk of generation)
#   - Top 7 double_blocks (12-18)  (high-level dual-stream interaction)
#   - All cross-view modules (blocks 19, 28, 37)
#
# Frozen components:
#   - double_blocks 0-11  (low-level features, well-transferable)
#   - VAE encoder         (always frozen)
#
# Optimization: match original paper style (constant 1e-4, no warmup/decay/cosine).
# If OOM, uncomment the reduced bucket_config at the bottom of this file.
# =============================================================================

lr = 1e-4

epochs = 30
ckpt_every = 200
keep_n_latest = 20

# Load from Phase A's latest checkpoint
load = "./outputs/t4_finetune_deep/omninwm_t4_phaseA/epoch0-global_step200"
start_from_scratch = True  # only load model weights, fresh optimizer

# -----------------------------------------------------------------------------
# Freeze strategy: unfreeze single blocks + top double blocks
# -----------------------------------------------------------------------------
freeze_strategy = dict(
    enable=True,
    freeze_all=True,
    freeze_transformer_blocks=False,
    freeze_patterns=[
        "double_blocks.0.*",
        "double_blocks.1.*",
        "double_blocks.2.*",
        "double_blocks.3.*",
        "double_blocks.4.*",
        "double_blocks.5.*",
        "double_blocks.6.*",
        "double_blocks.7.*",
        "double_blocks.8.*",
        "double_blocks.9.*",
        "double_blocks.10.*",
        "double_blocks.11.*",
    ],
    unfreeze_patterns=[
        "img_in*",
        "cond_in*",
        "traj_cond_in*",
        "time_in*",
        "final_layer*",
        "single_blocks.*",
        "double_blocks.12.*",
        "double_blocks.13.*",
        "double_blocks.14.*",
        "double_blocks.15.*",
        "double_blocks.16.*",
        "double_blocks.17.*",
        "double_blocks.18.*",
    ],
    unfreeze_cross_view=True,
    param_groups=[
        dict(
            name="adapter_and_head",
            lr_mult=1.0,        # 1e-4
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
            lr_mult=0.5,        # 5e-5
            patterns=[
                "single_blocks.*.q_mv_proj*",
                "single_blocks.*.k_mv_proj*",
                "single_blocks.*.v_mv_mlp*",
                "single_blocks.*.linear_mv*",
                "single_blocks.*.connector*",
            ],
        ),
        dict(
            name="late_double_blocks",
            lr_mult=0.2,        # 2e-5
            patterns=[
                "double_blocks.12.*",
                "double_blocks.13.*",
                "double_blocks.14.*",
                "double_blocks.15.*",
                "double_blocks.16.*",
                "double_blocks.17.*",
                "double_blocks.18.*",
            ],
        ),
        # Cross-view params already matched by earlier group, so this
        # only catches remaining non-cross-view params in single blocks.
        dict(
            name="single_blocks",
            lr_mult=0.4,        # 4e-5
            patterns=[
                "single_blocks.*",
            ],
        ),
    ],
)

# Match original paper: no weight decay, no cosine, no warmup, grad_clip=1.0
optim = dict(
    cls="HybridAdam",
    lr=lr,
    eps=1e-15,
    weight_decay=0.0,
    adamw_mode=True,
)

# Reduced low-res sequence length slightly to fit memory with ~70% params unfrozen
bucket_config = {
    "_delete_": True,
    "224x400r": {
        33: (1.0, 1),
    },
    "448x800r": {
        13: (1.0, 1),
    },
}
num_frames = 33
dataset = dict(
    num_frames=33,
    video_attr_list=[
        dict(height=224, width=400, frames=33),
        dict(height=448, width=800, frames=13),
    ],
)

outputs = "./outputs/t4_finetune_deep"

mlflow_run_name = "omninwm_t4_deep"
mlflow_tags = dict(
    project="omninwm",
    dataset="t4",
    task="finetune_t4_deep_adaptation",
    views="5",
    occ_head="frozen_or_unused",
)
