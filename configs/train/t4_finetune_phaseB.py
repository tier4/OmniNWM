_base_ = [
    "./t4_finetune.py",
]

# Phase B: continue from Phase A checkpoint with lower LR
lr = 5e-6
optim = dict(
    lr=lr,
)

# Optional: set this to your Phase A checkpoint path when launching directly from config
# load = "/path/to/phaseA/checkpoint"

# Slightly longer refinement
epochs = 20
ckpt_every = 400

# Stage-2 finetune strategy:
# - keep adapters/head trainable
# - additionally unfreeze top single-stream blocks for controlled adaptation
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
        "single_blocks.34*",
        "single_blocks.35*",
        "single_blocks.36*",
        "single_blocks.37*",
    ],
    # We already unfreeze selected full blocks above; keep this off to avoid
    # globally unfreezing cross-view modules in all layers.
    unfreeze_cross_view=False,
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
            name="late_single_blocks",
            lr_mult=0.5,
            patterns=[
                "single_blocks.34*",
                "single_blocks.35*",
                "single_blocks.36*",
                "single_blocks.37*",
            ],
        ),
    ],
)

mlflow_tags = dict(
    project="omninwm",
    dataset="t4",
    task="finetune_stage3_5view_phaseB",
    views="5",
    occ_head="frozen_or_unused",
)

mlflow_run_name = "omninwm_t4_phaseB"
