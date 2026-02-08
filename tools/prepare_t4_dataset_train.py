"""Build T4 train/val pickle files for OmniNWM finetuning.

The generated pickle format is training-compatible with `nuscenes_video`:
- includes `scene_tokens` for clip sampling
- each info has `token`, `cams[*].data_path`, ego/global pose, and lidar fields
"""

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from pyquaternion import Quaternion


OMNI_VIEW_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
]

# Map raw T4 camera names to OmniNWM camera names.
RAW_TO_OMNI = {
    "CAM_FRONT_LEFT_WIDE": "CAM_FRONT_LEFT",
    "CAM_FRONT_WIDE": "CAM_FRONT",
    "CAM_FRONT": "CAM_FRONT",
    "CAM_FRONT_RIGHT_WIDE": "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT_WIDE": "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT_WIDE": "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT": "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT": "CAM_BACK_LEFT",
}

REQUIRED_OMNI_CAMS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK_LEFT",
]


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_copy_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _build_camera_record(
    scene_dir: Path,
    sample_data: Dict,
    calibrated_sensor: Dict,
    ego_pose: Dict,
) -> Optional[Dict]:
    filename = sample_data.get("filename")
    if not filename:
        return None

    data_path = scene_dir / filename
    if not data_path.exists():
        return None

    cam_intrinsic = np.asarray(
        calibrated_sensor.get(
            "camera_intrinsic",
            [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
        ),
        dtype=np.float32,
    )

    sensor2ego_rotation = calibrated_sensor.get("rotation", [1, 0, 0, 0])
    sensor2ego_translation = np.asarray(
        calibrated_sensor.get("translation", [0, 0, 0]),
        dtype=np.float32,
    )
    sensor2lidar_rotation = Quaternion(sensor2ego_rotation).rotation_matrix.astype(
        np.float32
    )

    ego2global_rotation = ego_pose.get("rotation", [1, 0, 0, 0])
    ego2global_translation = np.asarray(
        ego_pose.get("translation", [0, 0, 0]),
        dtype=np.float32,
    )

    return {
        "data_path": str(data_path),
        "img_path": str(data_path),
        "cam_intrinsic": cam_intrinsic,
        "camera_intrinsics": cam_intrinsic.copy(),
        "sensor2ego_rotation": sensor2ego_rotation,
        "sensor2ego_translation": sensor2ego_translation,
        "sensor2lidar_rotation": sensor2lidar_rotation,
        "sensor2lidar_translation": sensor2ego_translation.copy(),
        "ego2global_rotation": ego2global_rotation,
        "ego2global_translation": ego2global_translation,
    }


def _duplicate_back_camera(front_cam: Dict) -> Dict:
    return {k: _safe_copy_value(v) for k, v in front_cam.items()}


def _find_scene_dirs(t4_root: Path) -> List[Path]:
    scenes = []
    for ann_dir in t4_root.rglob("annotation"):
        scene_dir = ann_dir.parent
        if (
            (ann_dir / "sample_data.json").exists()
            and (ann_dir / "ego_pose.json").exists()
            and (ann_dir / "calibrated_sensor.json").exists()
            and (ann_dir / "sensor.json").exists()
        ):
            scenes.append(scene_dir)
    scenes = sorted(set(scenes))
    return scenes


def _build_scene_infos(
    scene_dir: Path,
    scene_uid: str,
    max_frames_per_scene: int,
    min_frames_per_scene: int,
) -> Tuple[List[Dict], List[str]]:
    ann_dir = scene_dir / "annotation"

    sample_data = _load_json(ann_dir / "sample_data.json")
    ego_poses = {x["token"]: x for x in _load_json(ann_dir / "ego_pose.json")}
    calibrated_sensors = {
        x["token"]: x for x in _load_json(ann_dir / "calibrated_sensor.json")
    }
    sensors = {x["token"]: x for x in _load_json(ann_dir / "sensor.json")}

    by_omni_cam: Dict[str, List[Dict]] = defaultdict(list)

    for sd in sample_data:
        calib = calibrated_sensors.get(sd.get("calibrated_sensor_token"))
        if calib is None:
            continue
        sensor = sensors.get(calib.get("sensor_token"))
        if sensor is None:
            continue
        raw_cam = sensor.get("channel", "")
        omni_cam = RAW_TO_OMNI.get(raw_cam)
        if omni_cam is None:
            continue
        by_omni_cam[omni_cam].append(
            {
                "sample_data": sd,
                "calib": calib,
                "raw_cam": raw_cam,
            }
        )

    for cam in REQUIRED_OMNI_CAMS:
        if cam not in by_omni_cam or len(by_omni_cam[cam]) == 0:
            return [], []
        by_omni_cam[cam].sort(key=lambda x: x["sample_data"]["timestamp"])

    frame_count = min(len(by_omni_cam[cam]) for cam in REQUIRED_OMNI_CAMS)
    if max_frames_per_scene > 0:
        frame_count = min(frame_count, max_frames_per_scene)
    if frame_count < min_frames_per_scene:
        return [], []

    infos = []
    scene_tokens = []

    for frame_idx in range(frame_count):
        front_entry = by_omni_cam["CAM_FRONT"][frame_idx]
        ego_pose_token = front_entry["sample_data"].get("ego_pose_token")
        ego_pose = ego_poses.get(ego_pose_token)
        if ego_pose is None:
            continue

        cams = {}
        skip_frame = False
        for omni_cam in REQUIRED_OMNI_CAMS:
            entry = by_omni_cam[omni_cam][frame_idx]
            cam_record = _build_camera_record(
                scene_dir=scene_dir,
                sample_data=entry["sample_data"],
                calibrated_sensor=entry["calib"],
                ego_pose=ego_pose,
            )
            if cam_record is None:
                skip_frame = True
                break
            cams[omni_cam] = cam_record

        if skip_frame:
            continue

        # T4 usually has no rear camera; duplicate front camera to keep 6-view format.
        cams["CAM_BACK"] = _duplicate_back_camera(cams["CAM_FRONT"])

        token = f"{scene_uid}_{frame_idx:06d}"
        info = {
            "scene_token": scene_uid,
            "token": token,
            "timestamp": int(front_entry["sample_data"]["timestamp"]),
            "frame_idx": frame_idx,
            "cams": {view: cams[view] for view in OMNI_VIEW_ORDER},
        }
        infos.append(info)
        scene_tokens.append(token)

    return infos, scene_tokens


def _pack_dataset(scene_payloads: List[Dict], version: str, description: str) -> Dict:
    infos = []
    scene_tokens = []
    scene_ids = []

    for payload in scene_payloads:
        infos.extend(payload["infos"])
        scene_tokens.append(payload["scene_tokens"])
        scene_ids.append(payload["scene_uid"])

    return {
        "metadata": {
            "version": version,
            "description": description,
            "num_scenes": len(scene_payloads),
            "num_samples": len(infos),
        },
        "scene_tokens": scene_tokens,
        "scene_ids": scene_ids,
        "infos": infos,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare T4 train/val pickle files for OmniNWM finetuning"
    )
    parser.add_argument(
        "--t4-root",
        type=str,
        default="/mnt/nvme2/T4_datasets",
        help="T4 dataset root directory",
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default="data/t4_infos_train.pkl",
        help="Output pickle path for train split",
    )
    parser.add_argument(
        "--val-output",
        type=str,
        default="data/t4_infos_val.pkl",
        help="Output pickle path for val split",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Scene-level validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=-1,
        help="Use at most N scenes (-1 means all)",
    )
    parser.add_argument(
        "--max-frames-per-scene",
        type=int,
        default=-1,
        help="Use at most N frames from each scene (-1 means all)",
    )
    parser.add_argument(
        "--min-frames-per-scene",
        type=int,
        default=33,
        help="Skip scenes shorter than this threshold",
    )
    args = parser.parse_args()

    t4_root = Path(args.t4_root)
    if not t4_root.exists():
        raise FileNotFoundError(f"T4 root does not exist: {t4_root}")

    scene_dirs = _find_scene_dirs(t4_root)
    if args.max_scenes > 0:
        scene_dirs = scene_dirs[: args.max_scenes]

    print(f"Found {len(scene_dirs)} candidate scenes under: {t4_root}")

    scene_payloads = []
    for scene_idx, scene_dir in enumerate(scene_dirs):
        scene_uid = f"scene_{scene_idx:06d}"
        infos, scene_tokens = _build_scene_infos(
            scene_dir=scene_dir,
            scene_uid=scene_uid,
            max_frames_per_scene=args.max_frames_per_scene,
            min_frames_per_scene=args.min_frames_per_scene,
        )
        if len(infos) == 0:
            continue
        scene_payloads.append(
            {
                "scene_uid": scene_uid,
                "scene_dir": str(scene_dir),
                "infos": infos,
                "scene_tokens": scene_tokens,
            }
        )

    if len(scene_payloads) == 0:
        raise RuntimeError("No valid T4 scenes found for training pickle generation.")

    rng = random.Random(args.seed)
    rng.shuffle(scene_payloads)

    num_scenes = len(scene_payloads)
    num_val = int(num_scenes * args.val_ratio)
    if args.val_ratio > 0 and num_scenes > 1 and num_val == 0:
        num_val = 1
    if num_val >= num_scenes:
        num_val = num_scenes - 1

    val_payloads = scene_payloads[:num_val]
    train_payloads = scene_payloads[num_val:]
    if len(train_payloads) == 0:
        train_payloads = val_payloads
        val_payloads = []

    train_data = _pack_dataset(
        train_payloads,
        version="t4-v1.0-train",
        description="T4 training split for OmniNWM finetuning",
    )
    val_data = _pack_dataset(
        val_payloads,
        version="t4-v1.0-val",
        description="T4 validation split for OmniNWM finetuning",
    )

    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)

    with open(train_output, "wb") as f:
        pickle.dump(train_data, f)
    with open(val_output, "wb") as f:
        pickle.dump(val_data, f)

    print(
        f"Saved train split: {train_output} "
        f"(scenes={train_data['metadata']['num_scenes']}, samples={train_data['metadata']['num_samples']})"
    )
    print(
        f"Saved val split: {val_output} "
        f"(scenes={val_data['metadata']['num_scenes']}, samples={val_data['metadata']['num_samples']})"
    )


if __name__ == "__main__":
    main()
