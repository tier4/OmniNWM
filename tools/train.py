from IPython import embed
import gc
import fnmatch
import math
import os
import random
import subprocess
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from pprint import pformat
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
gc.disable()
import torchvision.transforms as transforms
from einops import rearrange, repeat
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from colossalai.booster import Booster
from colossalai.utils import set_seed
from tqdm import tqdm
from omninwm.acceleration.checkpoint import (
    GLOBAL_ACTIVATION_MANAGER,
    set_grad_checkpoint,
)
from omninwm.acceleration.parallel_states import get_data_parallel_group
from omninwm.datasets.aspect import bucket_to_shapes
from omninwm.datasets.dataloader import prepare_dataloader
from omninwm.datasets.pin_memory_cache import PinMemoryCache
from omninwm.models.mmdit.distributed import MMDiTPolicy
from omninwm.registry import DATASETS, MODELS, build_module
from omninwm.utils.ckpt import (
    CheckpointIO,
    model_sharding,
    record_model_param_shape,
    rm_checkpoints,
)
from omninwm.utils.config import (
    config_to_name,
    create_experiment_workspace,
    parse_configs,
)
from omninwm.utils.logger import create_logger
from omninwm.utils.misc import (
    NsysProfiler,
    Timers,
    all_reduce_mean,
    create_tensorboard_writer,
    is_log_process,
    log_cuda_max_memory,
    log_cuda_memory,
    log_model_params,
    print_mem,
    to_torch_dtype,
)
from omninwm.utils.optimizer import create_lr_scheduler, create_optimizer
from omninwm.utils.sampling import (
    get_res_lin_function,
    pack,
    prepare,
    time_shift,
    unpack
)
from omninwm.utils.train import (
    create_colossalai_plugin,
    prepare_visual_condition_causal,
    set_eps,
    set_lr,
    setup_device,
    update_ema,
)
torch.backends.cudnn.benchmark = False  # True leads to slow down in conv3d


def _flatten_mlflow_params(data, prefix=""):
    flat = {}
    if isinstance(data, dict):
        for key, value in data.items():
            key = str(key)
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(_flatten_mlflow_params(value, full_key))
            else:
                if isinstance(value, (list, tuple)):
                    value = str(value)
                elif value is None:
                    value = "None"
                elif not isinstance(value, (str, int, float, bool)):
                    value = str(value)
                value = str(value)
                # MLflow param key/value length limits
                flat[full_key[:240]] = value[:500]
    return flat


def _normalize_mlflow_tracking_uri(tracking_uri: str, project_root: Path) -> str:
    if not tracking_uri:
        tracking_uri = "sqlite:///mlruns/mlflow.db"

    if tracking_uri.startswith("sqlite:///"):
        db_path = tracking_uri[len("sqlite:///") :]
        if not os.path.isabs(db_path):
            db_path = str(project_root / db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        tracking_uri = f"sqlite:///{db_path}"
    return tracking_uri


def _normalize_mlflow_artifact_location(artifact_location: str, project_root: Path):
    if artifact_location is None:
        return None
    if "://" in artifact_location:
        return artifact_location
    if not os.path.isabs(artifact_location):
        artifact_location = str(project_root / artifact_location)
    os.makedirs(artifact_location, exist_ok=True)
    return artifact_location


def _log_mlflow_params_safely(mlflow_client, params: dict):
    items = list(params.items())
    if not items:
        return
    chunk_size = 100
    for start in range(0, len(items), chunk_size):
        mlflow_client.log_params(dict(items[start : start + chunk_size]))


def _matches_pattern(name: str, pattern: str) -> bool:
    if any(ch in pattern for ch in "*?[]"):
        return fnmatch.fnmatch(name, pattern)
    return pattern in name


def _matches_any_pattern(name: str, patterns: list[str]) -> bool:
    return any(_matches_pattern(name, p) for p in patterns)


def _configure_finetune_params(model: torch.nn.Module, cfg, logger):
    freeze_cfg = cfg.get("freeze_strategy", None)
    if not freeze_cfg or not freeze_cfg.get("enable", False):
        return None

    named_params = list(model.named_parameters())

    freeze_patterns = list(freeze_cfg.get("freeze_patterns", []))
    if freeze_cfg.get("freeze_transformer_blocks", False):
        freeze_patterns.extend(["double_blocks.*", "single_blocks.*"])

    unfreeze_patterns = list(freeze_cfg.get("unfreeze_patterns", []))
    if freeze_cfg.get("unfreeze_cross_view", False):
        unfreeze_patterns.extend(
            [
                "single_blocks.*.q_mv_proj*",
                "single_blocks.*.k_mv_proj*",
                "single_blocks.*.v_mv_mlp*",
                "single_blocks.*.linear_mv*",
                "single_blocks.*.connector*",
            ]
        )

    if freeze_cfg.get("freeze_all", True):
        for _, p in named_params:
            p.requires_grad = False

    if freeze_patterns:
        for name, p in named_params:
            if _matches_any_pattern(name, freeze_patterns):
                p.requires_grad = False

    if unfreeze_patterns:
        for name, p in named_params:
            if _matches_any_pattern(name, unfreeze_patterns):
                p.requires_grad = True

    trainable = [name for name, p in named_params if p.requires_grad]
    frozen = [name for name, p in named_params if not p.requires_grad]
    logger.info(
        "Freeze strategy applied. trainable=%s frozen=%s",
        len(trainable),
        len(frozen),
    )
    if trainable:
        logger.info("Trainable param samples: %s", trainable[:30])

    param_groups_cfg = freeze_cfg.get("param_groups", None)
    if not param_groups_cfg:
        return None

    base_lr = float(cfg.optim.lr)
    default_wd = float(cfg.optim.get("weight_decay", 0.0))
    assigned_names = set()
    param_groups = []

    for idx, group_cfg in enumerate(param_groups_cfg):
        patterns = list(group_cfg.get("patterns", []))
        if not patterns:
            continue
        group_name = group_cfg.get("name", f"group_{idx}")
        group_lr = float(group_cfg.get("lr", base_lr * float(group_cfg.get("lr_mult", 1.0))))
        group_wd = float(group_cfg.get("weight_decay", default_wd))
        params = []
        names = []
        for name, p in named_params:
            if (not p.requires_grad) or (name in assigned_names):
                continue
            if _matches_any_pattern(name, patterns):
                params.append(p)
                names.append(name)
                assigned_names.add(name)
        if params:
            param_groups.append(
                {
                    "name": group_name,
                    "params": params,
                    "lr": group_lr,
                    "weight_decay": group_wd,
                }
            )
            logger.info(
                "Param group %s: %s tensors, lr=%s, weight_decay=%s",
                group_name,
                len(params),
                group_lr,
                group_wd,
            )
            logger.info("Param group %s samples: %s", group_name, names[:20])
        else:
            logger.warning("Param group %s matched 0 parameters.", group_name)

    remaining_params = []
    remaining_names = []
    for name, p in named_params:
        if p.requires_grad and name not in assigned_names:
            remaining_params.append(p)
            remaining_names.append(name)

    if remaining_params:
        param_groups.append(
            {
                "name": "default",
                "params": remaining_params,
                "lr": base_lr,
                "weight_decay": default_wd,
            }
        )
        logger.info(
            "Default param group: %s tensors, lr=%s, weight_decay=%s",
            len(remaining_params),
            base_lr,
            default_wd,
        )
        logger.info("Default param group samples: %s", remaining_names[:20])

    # torch optimizer does not support "name" key in groups
    for group in param_groups:
        group.pop("name", None)

    return param_groups


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs()

    # == get dtype & device ==
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32   = False
    device, coordinator = setup_device()
    grad_ckpt_buffer_size = cfg.get("grad_ckpt_buffer_size", 0)
    if grad_ckpt_buffer_size > 0:
        GLOBAL_ACTIVATION_MANAGER.setup_buffer(grad_ckpt_buffer_size, dtype)
    checkpoint_io = CheckpointIO()
    set_seed(cfg.get("seed", 1024))
    PinMemoryCache.force_dtype = dtype
    pin_memory_cache_pre_alloc_numels = cfg.get("pin_memory_cache_pre_alloc_numels", None)
    PinMemoryCache.pre_alloc_numels = pin_memory_cache_pre_alloc_numels

    # == init ColossalAI booster ==
    plugin_type = cfg.get("plugin", "zero2")
    plugin_config = cfg.get("plugin_config", {})
    plugin_kwargs = {}
    if plugin_type == "hybrid":
        plugin_kwargs["custom_policy"] = MMDiTPolicy
    plugin = create_colossalai_plugin(
        plugin=plugin_type,
        dtype=cfg.get("dtype", "bf16"),
        grad_clip=cfg.get("grad_clip", 0),
        **plugin_config,
        **plugin_kwargs,
    )

    booster = Booster(plugin=plugin)

    seq_align = plugin_config.get("sp_size", 1)

    # == init exp_dir ==
    exp_name, exp_dir = create_experiment_workspace(
        cfg.get("outputs", "./outputs"),
        model_name=config_to_name(cfg),
        config=cfg.to_dict(),
        exp_name=cfg.get("exp_name", None),  # useful for automatic restart to specify the exp_name
    )

    if is_log_process(plugin_type, plugin_config):
        print(f"changing {exp_dir} to share")
        os.system(f"chgrp -R share {exp_dir}")

    # == init logger, tensorboard & wandb ==
    logger = create_logger(exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    tb_writer = None
    mlflow_client = None
    mlflow_active = False
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(
                project=cfg.get("wandb_project", "Open-Sora"),
                name=exp_name,
                config=cfg.to_dict(),
                dir=exp_dir,
            )

    num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    tp_size = cfg["plugin_config"].get("tp_size", 1)
    sp_size = cfg["plugin_config"].get("sp_size", 1)
    pp_size = cfg["plugin_config"].get("pp_size", 1)
    num_groups = num_gpus // (tp_size * sp_size * pp_size)
    logger.info("Number of GPUs: %s", num_gpus)
    logger.info("Number of groups: %s", num_groups)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    cache_pin_memory = pin_memory_cache_pre_alloc_numels is not None
    num_workers = int(cfg.get("num_workers", 4))
    prefetch_factor = cfg.get("prefetch_factor", None)
    persistent_workers = bool(cfg.get("persistent_workers", num_workers > 0))
    if num_workers <= 0:
        prefetch_factor = None
        persistent_workers = False
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", None),
        num_workers=num_workers,
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        cache_pin_memory=cache_pin_memory,
        num_groups=num_groups,
    )
    print_mem("before prepare_dataloader")
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    print_mem("after prepare_dataloader")
    num_steps_per_epoch = len(dataloader)


    # ======================================================
    # 4. build model
    # ======================================================
    logger.info("Building models...")
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).train()
    if cfg.get("grad_checkpoint", True):
        set_grad_checkpoint(model)
    optimizer_param_groups = _configure_finetune_params(model, cfg, logger)
    log_cuda_memory("diffusion")
    log_model_params(model)

    # == build EMA model ==
    if cfg.get("ema_decay", None) is not None:
        ema = deepcopy(model).cpu().eval().requires_grad_(False)
        ema_shape_dict = record_model_param_shape(ema)
        logger.info("EMA model created.")
    else:
        ema = ema_shape_dict = None
        logger.info("No EMA model created.")
    log_cuda_memory("EMA")

    model_ae = build_module(cfg.ae, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
    del model_ae.decoder
    log_cuda_memory("autoencoder")
    log_model_params(model_ae)
    if cfg.get("compile_ae_encoder", False):
        logger.info(
            "Compiling VAE encoder with torch.compile (dynamic=%s)...",
            cfg.get("compile_ae_dynamic", True),
        )
        model_ae.encode = torch.compile(
            model_ae.encoder,
            dynamic=cfg.get("compile_ae_dynamic", True),
        )

    # == setup optimizer ==
    optimizer = create_optimizer(model, cfg.optim, param_groups=optimizer_param_groups)

    # == setup lr scheduler ==
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer,
        num_steps_per_epoch=num_steps_per_epoch,
        epochs=cfg.get("epochs", 1000),
        warmup_steps=cfg.get("warmup_steps", None),
        use_cosine_scheduler=cfg.get("use_cosine_scheduler", False),
    )

    log_cuda_memory("optimizer")

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")

    # == boosting ==
    torch.set_default_dtype(dtype)

    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosted model for distributed training")
    log_cuda_memory("boost")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    log_step = acc_step = 0
    running_loss = 0.0
    timers = Timers(record_time=cfg.get("record_time", False), record_barrier=cfg.get("record_barrier", False))

    nsys = NsysProfiler(
        warmup_steps=cfg.get("nsys_warmup_steps", 2),
        num_steps=cfg.get("nsys_num_steps", 2),
        enabled=cfg.get("nsys", False),
    )
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    load_master_weights = cfg.get("load_master_weights", False)
    save_master_weights = cfg.get("save_master_weights", False)
    start_epoch = cfg.get("start_epoch", None)
    start_step = cfg.get("start_step", None)

    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint from %s", cfg.load)
        lr_scheduler_to_load = lr_scheduler
        if cfg.get("update_warmup_steps", False):
            lr_scheduler_to_load = None

        ret = checkpoint_io.load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=None if cfg.get("start_from_scratch", False) else optimizer,
            lr_scheduler=None if cfg.get("start_from_scratch", False) else lr_scheduler_to_load,
            sampler=None if cfg.get("start_from_scratch", False) else sampler,
            include_master_weights=load_master_weights,
        )
        start_epoch = start_epoch if start_epoch is not None else ret[0]
        start_step = start_step if start_step is not None else ret[1]
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, ret[0], ret[1])
        # load optimizer and scheduler will overwrite some of the hyperparameters, so we need to reset them
        set_lr(optimizer, lr_scheduler, cfg.optim.lr, cfg.get("initial_lr", None))
        set_eps(optimizer, cfg.optim.eps)

        if cfg.get("update_warmup_steps", False):
            assert (
                cfg.get("warmup_steps", None) is not None
            ), "you need to set warmup_steps in order to pass --update-warmup-steps True"
            # set_warmup_steps(lr_scheduler, cfg.warmup_steps)
            lr_scheduler.step(start_epoch * num_steps_per_epoch + start_step)
            logger.info("The learning rate starts from %s", optimizer.param_groups[0]["lr"])

    if start_step is not None:
        # if start step exceeds data length, go to next epoch
        if start_step > num_steps_per_epoch:
            start_epoch = (
                start_epoch + start_step // num_steps_per_epoch
                if start_epoch is not None
                else start_step // num_steps_per_epoch
            )
            start_step = start_step % num_steps_per_epoch
    else:
        start_step = 0
    sampler.set_step(start_step)
    start_epoch = start_epoch if start_epoch is not None else 0
    logger.info("Starting from epoch %s step %s", start_epoch, start_step)

    # == sharding EMA model ==
    if ema is not None:
        model_sharding(ema)
        ema = ema.to(device)
        log_cuda_memory("sharding EMA")

    # == init mlflow ==
    if coordinator.is_master() and cfg.get("mlflow", False):
        import mlflow

        project_root = Path(__file__).resolve().parents[1]
        tracking_uri = _normalize_mlflow_tracking_uri(
            cfg.get("mlflow_tracking_uri", "sqlite:///mlruns/mlflow.db"),
            project_root,
        )
        artifact_location = _normalize_mlflow_artifact_location(
            cfg.get("mlflow_artifact_location", "mlruns/artifacts"),
            project_root,
        )

        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = cfg.get("mlflow_experiment", "omninwm")
        try:
            if artifact_location is not None:
                client = mlflow.tracking.MlflowClient()
                exp = client.get_experiment_by_name(experiment_name)
                if exp is None:
                    try:
                        client.create_experiment(
                            name=experiment_name,
                            artifact_location=artifact_location,
                        )
                    except TypeError:
                        # Older/newer MLflow API compatibility.
                        client.create_experiment(name=experiment_name)
            mlflow.set_experiment(experiment_name=experiment_name)
        except Exception as e:
            logger.warning(
                "Failed to set MLflow experiment with artifact_location (%s), fallback to default. err=%s",
                artifact_location,
                e,
            )
            mlflow.set_experiment(experiment_name)

        mlflow_run_name = cfg.get("mlflow_run_name", None) or exp_name
        mlflow.start_run(run_name=mlflow_run_name)
        mlflow_active = True
        mlflow_client = mlflow

        mlflow.set_tag("status", "RUNNING")
        mlflow.set_tag("exp_name", exp_name)
        mlflow.set_tag("exp_dir", exp_dir)
        mlflow.set_tag("config_path", cfg.get("config_path", ""))
        mlflow.set_tag("plugin", str(plugin_type))
        mlflow.set_tag("num_gpus", str(num_gpus))
        mlflow.set_tag("num_steps_per_epoch", str(num_steps_per_epoch))

        tags = cfg.get("mlflow_tags", None)
        if isinstance(tags, dict):
            mlflow.set_tags({str(k): str(v) for k, v in tags.items()})

        flat_cfg = _flatten_mlflow_params(cfg.to_dict())
        _log_mlflow_params_safely(mlflow, flat_cfg)
        mlflow.log_metric("dataset_size", len(dataset), step=0)

        config_path = os.path.join(exp_dir, "config.txt")
        if os.path.exists(config_path):
            mlflow.log_artifact(config_path, artifact_path="config")
        logger.info(
            "MLflow enabled. tracking_uri=%s, experiment=%s, run_name=%s",
            tracking_uri,
            experiment_name,
            mlflow_run_name,
        )

    # =======================================================
    # 5. training iter
    # =======================================================
    sigma_min = cfg.get("sigma_min", 1e-5)
    accumulation_steps = cfg.get("accumulation_steps", 1)
    ckpt_every = cfg.get("ckpt_every", 0)

    @torch.no_grad()
    def prepare_inputs(batch):
        inp = dict()
        x = batch["video"]
        ray_map_x = batch.get("ray_map",None)
        depth_x = batch.get("depth",None)
        seg_x = batch.get("seg",None)
        ori_bs = x.shape[0]
        if cfg.get("is_multi_view",False):
            x = rearrange(x,'b n c t h w -> (b n) c t h w')
            if seg_x is not None:
                seg_x = rearrange(seg_x,'b n c t h w -> (b n) c t h w')
            if depth_x is not None:
                depth_x = rearrange(depth_x,'b n c t h w -> (b n) c t h w')
            if ray_map_x is not None:
                ray_map_x = rearrange(ray_map_x,'b n c t h w -> (b n) c t h w')
        bs = x.shape[0]

        num_cam = bs//ori_bs
        
        # == encode video ==
        with nsys.range("encode_video"), timers["encode_video"]:
            # == prepare condition ==
            if cfg.get("condition_config", None) is not None:
                x_0, cond = prepare_visual_condition_causal(x, cfg.condition_config, model_ae)
                cond = pack(cond, patch_size=cfg.get("patch_size", 2))
                inp["cond"] = cond
            else:
                x_0 = model_ae.encode(x)

            if depth_x is not None:
                depth_x_0 = model_ae.encode(depth_x)
                x_0 = torch.cat([x_0,depth_x_0],dim=1)
            
            if seg_x is not None:
                seg_x_0 = model_ae.encode(seg_x)
                x_0 = torch.cat([x_0,seg_x_0],dim=1)

        # == prepare timestep ==
        # follow SD3 time shift, shift_alpha = 1 for 256px and shift_alpha = 3 for 1024px
        shift_alpha = get_res_lin_function()((x_0.shape[-1] * x_0.shape[-2]) // 4)
        # add temporal influence
        shift_alpha *= math.sqrt(x_0.shape[-3])  # for image, T=1 so no effect
        if cfg.get("use_multi_level_noise",False):
            t_bs,t_c,t_num_frames,t_height,t_width = x_0.shape
            t = torch.sigmoid(torch.randn((t_bs,t_num_frames), device=device))
        else:
            t = torch.sigmoid(torch.randn((ori_bs), device=device))

        t = time_shift(shift_alpha, t).to(dtype)
        if not cfg.get("use_multi_level_noise",False):
            t = repeat(t,'b ... -> (b t) ...',t=num_cam)

        # == encode text ==
        with nsys.range("encode_text"), timers["encode_text"]:
                inp_ = prepare(
                    x_0,
                    ray_map_x,
                    patch_size=cfg.get("patch_size", 2),
                )
                inp.update(inp_)
                inp['x_ori'] = x_0

        t_rev = 1 - t
        if cfg.get("use_multi_level_noise",False):
            x_1 = torch.randn_like(x_0, dtype=torch.float32).to(device, dtype)
            x_t = t_rev[:,None, :, None, None] * x_0 + (1 - (1 - sigma_min) * t_rev[:,None, :, None, None]) * x_1
            x_0 = pack(x_0, patch_size=cfg.get("patch_size", 2))
            x_1 = pack(x_1, patch_size=cfg.get("patch_size", 2))
            x_t = pack(x_t, patch_size=cfg.get("patch_size", 2))
        else:
            x_0 = pack(x_0, patch_size=cfg.get("patch_size", 2))
            x_1 = torch.randn_like(x_0, dtype=torch.float32).to(device, dtype)
            x_t = t_rev[:, None, None] * x_0 + (1 - (1 - sigma_min) * t_rev[:, None, None]) * x_1     

        inp["img"] = x_t
        inp["timesteps"] = t.to(dtype)
        inp["guidance"] = torch.full((x_t.shape[0],), cfg.get("guidance", 4), device=x_t.device, dtype=x_t.dtype)
        return inp, x_0, x_1
    
    def run_iter(inp, x_0, x_1):
        with nsys.range("forward"), timers["forward"]:
            # if cfg.get('use_multi_level_noise',False) and random.random() < 0.5:
            if cfg.get('use_multi_level_noise',False) and False:
                infer_x = inp['img'].clone()
                infer_cond = inp['cond'].clone()
                with torch.no_grad():
                    infer_model_pred = model.eval()(**inp)
                infer_num_frames, infer_height, infer_width = inp['x_ori'].shape[2:]
                infer_x_res = infer_x - infer_model_pred
                infer_x_res_unpack = unpack(infer_x_res, infer_height*8, infer_width*8, infer_num_frames, patch_size=cfg.get("patch_size", 2))
                infer_cond_unpack = unpack(infer_cond, infer_height*8, infer_width*8, infer_num_frames, patch_size=cfg.get("patch_size", 2))
                infer_cond_unpack[:,1:,:1] = infer_x_res_unpack[:,:16,:1]
                inp['cond'] = pack(infer_cond_unpack, patch_size=cfg.get("patch_size", 2))
                    
            # model_pred = model.train()(**inp)  # B, T, L
            model_pred = model(**inp)  # B, T, L
            v_t = (1 - sigma_min) * x_1 - x_0
            loss = F.mse_loss(model_pred.float(), v_t.float(), reduction="mean")
            # torch.cuda.empty_cache()
        
        loss_item = all_reduce_mean(loss.data.clone().detach()).item()
        # == backward & update ==
        with nsys.range("backward"), timers["backward"]:
            ctx = (
                booster.no_sync(model, optimizer)
                if cfg.get("plugin", "zero2") in ("zero1", "zero1-seq") and (step + 1) % accumulation_steps != 0
                else nullcontext()
            )
            with ctx:
                booster.backward(loss=(loss / accumulation_steps), optimizer=optimizer)

        with nsys.range("optim"), timers["optim"]:
            if (step + 1) % accumulation_steps == 0:
                booster.checkpoint_io.synchronize()
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        # == update EMA ==
        if ema is not None:
            with nsys.range("update_ema"), timers["update_ema"]:
                update_ema(
                    ema,
                    model.unwrap(),
                    optimizer=optimizer,
                    decay=cfg.get("ema_decay", 0.9999),
                )
        # torch.cuda.empty_cache()
        return loss_item
    
    # =======================================================
    # 6. training loop
    # =======================================================
    run_status = "RUNNING"
    try:
        for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
            sampler.set_epoch(epoch)
            dataloader_iter = iter(dataloader)
            logger.info("Beginning epoch %s...", epoch)

        # == training loop in an epoch ==
            with tqdm(
                enumerate(dataloader_iter, start=start_step),
                desc=f"Epoch {epoch}",
                disable=not is_log_process(plugin_type, plugin_config),
                initial=start_step,
                total=num_steps_per_epoch,
            ) as pbar:
                pbar_iter = iter(pbar)
                # prefetch one for non-blocking data loading
                def fetch_data():
                    step, batch = next(pbar_iter)
                    pinned_video = batch["video"]
                    batch["video"] = pinned_video.to(device, dtype, non_blocking=True)
                    if batch.get('depth',None) is not None:
                        pinned_depth = batch["depth"]
                        batch["depth"] = pinned_depth.to(device, dtype, non_blocking=True)
                    if batch.get('seg',None) is not None:
                        pinned_seg_video = batch["seg"]
                        batch["seg"] = pinned_seg_video.to(device, dtype, non_blocking=True)
                    if batch.get('ray_map',None) is not None:
                        pinned_ray_map = batch["ray_map"]
                        batch["ray_map"] = pinned_ray_map.to(device, dtype, non_blocking=True)

                    return batch, step, pinned_video
                
                batch_, step_, pinned_video_ = fetch_data()

                for _ in range(start_step, num_steps_per_epoch):
                    nsys.step()
                    # == load data ===
                    with nsys.range("load_data"), timers["load_data"]:
                        batch, step, pinned_video = batch_, step_, pinned_video_
                        if step + 1 < num_steps_per_epoch:
                            # only fetch new data if not last step
                            batch_, step_, pinned_video_ = fetch_data()
                    # == run iter ==
                    with nsys.range("iter"), timers["iter"]:
                        inp, x_0, x_1 = prepare_inputs(batch)
                        if cache_pin_memory:
                            dataloader_iter.remove_cache(pinned_video)
                        loss = run_iter(inp, x_0, x_1)

                    # == update log info ==
                    if loss is not None:
                        running_loss += loss

                    # == log config ==
                    global_step = epoch * num_steps_per_epoch + step
                    actual_update_step = (global_step + 1) // accumulation_steps
                    log_step += 1
                    acc_step += 1

                    # == logging ==
                    if (global_step + 1) % accumulation_steps == 0:
                        if actual_update_step % cfg.get("log_every", 1) == 0:
                            if is_log_process(plugin_type, plugin_config):
                                avg_loss = running_loss / log_step
                                global_grad_norm = optimizer.get_grad_norm()
                                # progress bar
                                pbar.set_postfix(
                                    {
                                        "loss": avg_loss,
                                        "global_grad_norm": global_grad_norm,
                                        "step": step,
                                        "global_step": global_step,
                                        # "actual_update_step": actual_update_step,
                                        "lr": optimizer.param_groups[0]["lr"],
                                    }
                                )
                                # tensorboard
                                if tb_writer is not None:
                                    tb_writer.add_scalar("loss", loss, actual_update_step)
                                # wandb
                                if cfg.get("wandb", False):
                                    wandb_dict = {
                                        "iter": global_step,
                                        "acc_step": acc_step,
                                        "epoch": epoch,
                                        "loss": loss,
                                        "avg_loss": avg_loss,
                                        "lr": optimizer.param_groups[0]["lr"],
                                        "eps": optimizer.param_groups[0]["eps"],
                                        "global_grad_norm": global_grad_norm,
                                    }
                                    if cfg.get("record_time", False):
                                        wandb_dict.update(timers.to_dict())
                                    wandb.log(wandb_dict, step=actual_update_step)

                                # mlflow
                                if mlflow_active:
                                    mlflow_metrics = {
                                        "iter": global_step,
                                        "acc_step": acc_step,
                                        "epoch": epoch,
                                        "loss": float(loss),
                                        "avg_loss": float(avg_loss),
                                        "lr": float(optimizer.param_groups[0]["lr"]),
                                        "eps": float(optimizer.param_groups[0]["eps"]),
                                        "global_grad_norm": float(global_grad_norm),
                                    }
                                    if cfg.get("record_time", False):
                                        for key, value in timers.to_dict().items():
                                            if isinstance(value, (int, float)):
                                                mlflow_metrics[f"time/{key}"] = float(value)
                                    mlflow_client.log_metrics(mlflow_metrics, step=actual_update_step)

                            running_loss = 0.0
                            log_step = 0

                    # == checkpoint saving ==
                    # uncomment below 3 lines to forcely clean cache
                    with nsys.range("clean_cache"), timers["clean_cache"]:
                        if ckpt_every > 0 and actual_update_step % ckpt_every == 0 and coordinator.is_master():
                            subprocess.run("sudo drop_cache", shell=True)

                    with nsys.range("checkpoint"), timers["checkpoint"]:
                        if ckpt_every > 0 and actual_update_step % ckpt_every == 0:
                            # mannual garbage collection
                            gc.collect()

                            save_dir = checkpoint_io.save(
                                booster,
                                exp_dir,
                                model=model,
                                ema=ema,
                                optimizer=None,
                                lr_scheduler=lr_scheduler,
                                sampler=sampler,
                                epoch=epoch,
                                step=step + 1,
                                global_step=global_step + 1,
                                batch_size=cfg.get("batch_size", None),
                                actual_update_step=actual_update_step,
                                ema_shape_dict=ema_shape_dict,
                                async_io=cfg.get("async_io", False),
                                include_master_weights=save_master_weights,
                            )

                            if is_log_process(plugin_type, plugin_config):
                                os.system(f"chgrp -R share {save_dir}")

                            logger.info(
                                "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                                epoch,
                                step + 1,
                                actual_update_step,
                                save_dir,
                            )

                            if mlflow_active and cfg.get("mlflow_log_checkpoints", False):
                                mlflow_client.log_artifacts(save_dir, artifact_path=f"checkpoints/step_{actual_update_step}")

                            # remove old checkpoints
                            rm_checkpoints(exp_dir, keep_n_latest=cfg.get("keep_n_latest", -1))
                            logger.info("Removed old checkpoints and kept %s latest ones.", cfg.get("keep_n_latest", -1))
                    # uncomment below 3 lines to benchmark checkpoint
                    # if ckpt_every > 0 and actual_update_step % ckpt_every == 0:
                    #     booster.checkpoint_io._sync_io()
                    #     checkpoint_io._sync_io()
                    # == terminal timer ==
                    if cfg.get("record_time", False):
                        print(timers.to_str(epoch, step))

            sampler.reset()
            start_step = 0

        log_cuda_max_memory("final")
        run_status = "COMPLETED"
    except Exception:
        run_status = "FAILED"
        raise
    finally:
        if mlflow_active:
            try:
                mlflow_client.set_tag("status", run_status)
                mlflow_client.end_run()
            except Exception as e:
                logger.warning("Failed to finalize MLflow run: %s", e)
        if coordinator.is_master() and cfg.get("wandb", False):
            wandb.finish()

if __name__ == "__main__":
    main()
