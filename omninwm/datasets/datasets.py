import os
import random
import math
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, Any
from IPython import embed
import numpy as np
import torch
from pyquaternion import Quaternion
import cv2
from packaging import version as pver

# 导入本地模块
from omninwm.registry import DATASETS
from omninwm.datasets.utils import (
    load_pkl,
    build_clips,
    EfficientParquet,
)
from omninwm.datasets.config import DatasetConfig

try:
    from turbojpeg import TurboJPEG, TJPF_RGB
except Exception:
    TurboJPEG = None
    TJPF_RGB = None

_TURBOJPEG_DECODER = None
_TURBOJPEG_DECODER_PID = None
_RAY_GRID_CACHE = {}
_RAY_GRID_CACHE_MAX_ITEMS = 64


def _get_turbojpeg_decoder(enabled: bool):
    global _TURBOJPEG_DECODER, _TURBOJPEG_DECODER_PID
    if not enabled or TurboJPEG is None:
        return None
    pid = os.getpid()
    if _TURBOJPEG_DECODER is None or _TURBOJPEG_DECODER_PID != pid:
        try:
            _TURBOJPEG_DECODER = TurboJPEG()
            _TURBOJPEG_DECODER_PID = pid
        except Exception:
            _TURBOJPEG_DECODER = None
            _TURBOJPEG_DECODER_PID = None
    return _TURBOJPEG_DECODER


def _resize_crop_to_fill_cv2(
    image: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize+center-crop to target (H, W), equivalent to resize_crop_to_fill."""
    h, w = image.shape[:2]
    th, tw = target_size
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        resized = cv2.resize(image, (sw, sh), interpolation=interpolation)
        i, j = 0, int(round((sw - tw) / 2.0))
    else:
        sh, sw = round(h * rw), tw
        resized = cv2.resize(image, (sw, sh), interpolation=interpolation)
        i, j = int(round((sh - th) / 2.0)), 0
    return resized[i : i + th, j : j + tw]


def _normalize_rgb_uint8_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """Convert RGB uint8 HWC image to normalized CHW tensor in [-1, 1]."""
    image_rgb = np.ascontiguousarray(image_rgb)
    image_t = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
    return (image_t - 0.5) / 0.5


def _load_rgb_tensor_cv2(
    image_path: str, target_size: Tuple[int, int], use_turbojpeg: bool = True
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Load image with OpenCV and return normalized tensor + original (W, H)."""
    rgb = None
    ext = os.path.splitext(image_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        jpeg = _get_turbojpeg_decoder(use_turbojpeg)
        if jpeg is not None and TJPF_RGB is not None:
            try:
                with open(image_path, "rb") as f:
                    rgb = jpeg.decode(f.read(), pixel_format=TJPF_RGB)
            except Exception:
                rgb = None

    if rgb is None:
        bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    h, w = rgb.shape[:2]
    rgb = _resize_crop_to_fill_cv2(rgb, target_size, interpolation=cv2.INTER_LINEAR)
    return _normalize_rgb_uint8_to_tensor(rgb), (w, h)


def _get_ray_grid(
    H: int,
    W: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
):
    """Cache flattened pixel grid [1,1,HW] for ray generation."""
    dev = torch.device(device)
    key = (H, W, dev.type, dev.index, str(dtype))
    cached = _RAY_GRID_CACHE.get(key, None)
    if cached is not None:
        return cached

    j, i = torch.meshgrid(
        torch.arange(H, device=dev, dtype=dtype),
        torch.arange(W, device=dev, dtype=dtype),
        indexing="ij",
    )
    i = i.reshape(1, 1, H * W) + 0.5
    j = j.reshape(1, 1, H * W) + 0.5
    if len(_RAY_GRID_CACHE) >= _RAY_GRID_CACHE_MAX_ITEMS:
        _RAY_GRID_CACHE.pop(next(iter(_RAY_GRID_CACHE)))
    _RAY_GRID_CACHE[key] = (i, j)
    return i, j


def get_ray_map_from_vla(
    intrinsics_list,
    extrinsics_list,
    traj_list,
    H,
    W,
    view_order={
        "CAM_FRONT_LEFT": 0,
        "CAM_FRONT": 1,
        "CAM_FRONT_RIGHT": 2,
        "CAM_BACK_RIGHT": 3,
        "CAM_BACK": 4,
        "CAM_BACK_LEFT": 5,
    },
    P=np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    ),
):
    sample_index_list = list(range(len(traj_list)))
    sample_index = sample_index_list[::4]
    sampled_traj_list = [traj_list[i] for i in sample_index]

    multi_view_ray_map_list = []
    for frame_id in range(len(sampled_traj_list)):
        ray_map_list = []

        rel_ego2global = np.eye(4)
        x = sampled_traj_list[frame_id][0]
        y = sampled_traj_list[frame_id][1]
        heading = sampled_traj_list[frame_id][2]
        heading_matrix = NuscenesVideoDataset.eulerAngles2rotationMat(
            [0.0, 0.0, heading]
        )
        rel_ego2global[:3, :3] = P @ heading_matrix
        rel_ego2global[:3, 3] = [x, y, 0]
        CAM_FRONT_INDEX = view_order["CAM_FRONT"]
        for view_name in view_order.keys():
            view_id = view_order[view_name]
            current_raymap_camera_intrinsics = (
                intrinsics_list[view_id][frame_id].float().cpu()
            )
            cam2ego = extrinsics_list[view_id][frame_id].float().cpu().numpy()
            front_cam2ego = (
                extrinsics_list[CAM_FRONT_INDEX][frame_id].float().cpu().numpy()
            )

            current_camera_c2w = np.dot(
                np.dot(front_cam2ego, np.linalg.inv(cam2ego)),
                np.dot(rel_ego2global, cam2ego),
            )
            current_camera_c2w = torch.from_numpy(current_camera_c2w)
            ram_map_k = torch.zeros(1, 1, 4)
            ray_map_c2w = torch.zeros(1, 1, 4, 4)
            ray_map_c2w[0, 0] = current_camera_c2w
            ram_map_k[:, :, 0] = current_raymap_camera_intrinsics[0, 0]
            ram_map_k[:, :, 1] = current_raymap_camera_intrinsics[1, 1]
            ram_map_k[:, :, 2] = current_raymap_camera_intrinsics[0, 2]
            ram_map_k[:, :, 3] = current_raymap_camera_intrinsics[1, 2]

            ray_map_ori = NuscenesVideoDataset.ray_condition(
                ram_map_k, ray_map_c2w, H, W, device="cpu"
            )
            ray_map = ray_map_ori[0, 0].permute(2, 0, 1)
            ray_map_list.append(ray_map)

        ray_map = torch.stack(ray_map_list)
        multi_view_ray_map_list.append(ray_map)

    multi_view_ray_map = torch.stack(multi_view_ray_map_list).permute(1, 2, 0, 3, 4)

    return multi_view_ray_map.unsqueeze(0)


@DATASETS.register_module("nuscenes_video")
class NuscenesVideoDataset(torch.utils.data.Dataset):
    """NuScenes 多视图视频文本数据集加载器"""

    def __init__(self, **kwargs):
        """
        初始化数据集

        Args:
            **kwargs: 数据集配置参数，参见DatasetConfig类
        """
        # 将配置参数转换为数据类
        self.config = DatasetConfig(**kwargs)

        # 将配置参数复制到实例变量以便向后兼容
        for key, value in self.config.__dict__.items():
            setattr(self, key, value)

        # Avoid OpenCV oversubscribing CPU threads in multi-worker dataloading.
        opencv_num_threads = int(getattr(self, "opencv_num_threads", 1))
        if opencv_num_threads >= 0:
            cv2.setNumThreads(opencv_num_threads)
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass

        # Small process-local path caches for repeated path parsing.
        self._t4_path_cache = {}
        self._seg_path_cache = {}
        self._depth_path_cache = {}

        # 计算分割最大ID
        self.seg_max_id = (max(self.seg_class_map.values()) * 1000) + 500

        # 加载PKL数据
        self._load_pkl_data()
        if self.dataset_name == "nuplan":
            self.P = np.array(
                [
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1],
                ]
            )
        elif self.dataset_name == "nuscenes":
            self.P = np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            )
        # 设置推理参数
        if not self.is_train:
            self.infer_num_frames = self.num_frames
            self.infer_height = self.height
            self.infer_width = self.width

        # 设置最大FPS
        if self.fps_max is None:
            self.fps_max = 999999999

        # 转换为高效存储格式（如果启用）
        if self.memory_efficient:
            self._convert_to_efficient()

        # 初始化ray-map mask（用于几何条件层面的soft mask）
        self._init_ray_masks()

        # Precompute fast segmentation lookup tables.
        # raw seg id -> compact class id -> RGB color
        max_seg_raw_id = max(self.seg_class_map.keys()) if self.seg_class_map else 0
        self._seg_class_lut = np.zeros(max(256, max_seg_raw_id + 1), dtype=np.uint8)
        for raw_id, cls_id in self.seg_class_map.items():
            if raw_id >= 0:
                if raw_id >= self._seg_class_lut.shape[0]:
                    pad = raw_id + 1 - self._seg_class_lut.shape[0]
                    self._seg_class_lut = np.pad(self._seg_class_lut, (0, pad), mode="constant")
                self._seg_class_lut[raw_id] = cls_id
        self._seg_color_lut = np.asarray(self.seg_color_map, dtype=np.uint8)

    def _load_pkl_data(self) -> None:
        """加载PKL数据并构建数据片段"""
        self.pkl_data = load_pkl(self.pkl_path)

        if self.scene_token in self.pkl_data:
            self.metadata, self.version, self.data, self.data_infos = build_clips(
                self.pkl_data,
                self.num_frames,
                self.is_train,
                self.scene_token,
                infer_for_test=self.infer_for_test,
            )
        else:
            self.data_infos = list(
                sorted(self.pkl_data["infos"], key=lambda e: e["timestamp"])
            )
            self.metadata = self.pkl_data["metadata"]
            self.version = self.metadata["version"]

            # 构建数据片段 - store indices instead of actual data
            self.data = []
            for data_idx in range(len(self.data_infos)):
                # Create a list of num_frames indices (all pointing to the same sample for T4)
                temp_list = [data_idx] * self.num_frames
                self.data.append(temp_list)

    def _convert_to_efficient(self) -> None:
        """转换为高效存储格式"""
        if self.memory_efficient:
            addition_data_path = self.pkl_path.split(".")[0]
            self._data = self.data
            self.data = EfficientParquet(self._data, addition_data_path)

    def _init_ray_masks(self) -> None:
        """
        初始化用于ray-map zeroing的二值mask。
        Mask为每个相机独立加载，后续按ray-map分辨率做最近邻缩放。
        """
        self._ray_mask_cache = {}
        self._ray_mask_by_view = {}

        if not getattr(self, "ray_mask_zeroing", False):
            return

        ray_mask_paths = getattr(self, "ray_mask_paths", None)
        if not ray_mask_paths:
            print(
                "[NuscenesVideoDataset] ray_mask_zeroing=True but ray_mask_paths is empty."
            )
            return

        for view_name, mask_path in ray_mask_paths.items():
            if view_name not in self.view_order:
                continue
            if mask_path is None:
                continue
            if not os.path.exists(mask_path):
                print(
                    f"[NuscenesVideoDataset] ray mask not found for {view_name}: {mask_path}"
                )
                continue

            mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                print(
                    f"[NuscenesVideoDataset] failed to read ray mask for {view_name}: {mask_path}"
                )
                continue

            # >0区域视为需要屏蔽的区域，后续对应ray-map将置零
            self._ray_mask_by_view[view_name] = (mask_gray > 0).astype(np.float32)

        if self._ray_mask_by_view:
            loaded_views = ", ".join(sorted(self._ray_mask_by_view.keys()))
            print(
                f"[NuscenesVideoDataset] ray-map zeroing enabled for views: {loaded_views}"
            )

    def _get_resized_ray_mask(
        self, view_name: str, target_h: int, target_w: int
    ) -> Optional[torch.Tensor]:
        """
        获取指定视图在目标分辨率下的mask张量。
        返回shape: [1, 1, H, W]，值域{0,1}，1表示屏蔽区域。
        """
        if view_name not in self._ray_mask_by_view:
            return None

        cache_key = (view_name, target_h, target_w)
        if cache_key in self._ray_mask_cache:
            return self._ray_mask_cache[cache_key]

        mask_np = self._ray_mask_by_view[view_name]
        if mask_np.shape != (target_h, target_w):
            mask_np = cv2.resize(
                mask_np,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_np = (mask_np > 0.5).astype(np.float32)

        mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        self._ray_mask_cache[cache_key] = mask_t
        return mask_t

    def _apply_ray_map_zeroing(self, ray_map_sample: torch.Tensor) -> torch.Tensor:
        """
        在ray-map层面对mask区域做zeroing，避免模型将其解释为黑色物体。
        输入shape: [V, 6, T, H, W]
        """
        if not getattr(self, "ray_mask_zeroing", False):
            return ray_map_sample
        if not self._ray_mask_by_view:
            return ray_map_sample

        out = ray_map_sample.clone()
        _, _, _, h, w = out.shape

        for view_idx, view_name in enumerate(self.view_order):
            mask_t = self._get_resized_ray_mask(view_name, h, w)
            if mask_t is None:
                continue

            keep = 1.0 - mask_t.to(dtype=out.dtype)
            out[view_idx] = out[view_idx] * keep

        return out

    def _rand_another(self) -> int:
        """
        随机获取另一个数据索引

        Returns:
            int: 随机选择的索引
        """
        return np.random.randint(0, len(self))

    @staticmethod
    def eulerAngles2rotationMat(theta):
        """
        Calculates Rotation Matrix given euler angles.
        :param theta: 1-by-3 list [rx, ry, rz] angle in degree
        :return:
        RPY
        """
        R_x = np.array(
            [
                [1, 0, 0],
                [0, math.cos(theta[0]), -math.sin(theta[0])],
                [0, math.sin(theta[0]), math.cos(theta[0])],
            ]
        )

        R_y = np.array(
            [
                [math.cos(theta[1]), 0, math.sin(theta[1])],
                [0, 1, 0],
                [-math.sin(theta[1]), 0, math.cos(theta[1])],
            ]
        )

        R_z = np.array(
            [
                [math.cos(theta[2]), -math.sin(theta[2]), 0],
                [math.sin(theta[2]), math.cos(theta[2]), 0],
                [0, 0, 1],
            ]
        )
        R = np.dot(R_z, np.dot(R_y, R_x))
        return R

    @staticmethod
    def rotation_matrix_to_euler_angles(R: np.ndarray) -> np.ndarray:
        """
        将旋转矩阵转换为欧拉角

        Args:
            R: 3x3旋转矩阵

        Returns:
            np.ndarray: 欧拉角 (x, y, z)
        """
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])

    @staticmethod
    def get_resize_crop_param(
        ori_size: Tuple[int, int], target_size: Tuple[int, int]
    ) -> Tuple[float, float, int, int]:
        """
        计算调整大小和裁剪参数

        Args:
            ori_size: 原始尺寸 (width, height)
            target_size: 目标尺寸 (height, width)

        Returns:
            Tuple: (rh, rw, i, j) - 高度缩放比、宽度缩放比、垂直偏移、水平偏移
        """
        w, h = ori_size  # PIL格式 (W, H)
        th, tw = target_size
        rh, rw = th / h, tw / w

        if rh > rw:  # 高度缩放比更大，以高度为基准
            sh, sw = th, round(w * rh)
            i, j = 0, int(round((sw - tw) / 2.0))
        else:  # 宽度缩放比更大，以宽度为基准
            sh, sw = round(h * rw), tw
            i, j = int(round((sh - th) / 2.0)), 0

        return rh, rw, i, j

    @staticmethod
    def custom_meshgrid(*args) -> Tuple[torch.Tensor, ...]:
        """
        自定义网格生成，兼容不同版本的PyTorch

        Args:
            *args: 输入张量

        Returns:
            Tuple[torch.Tensor, ...]: 网格张量
        """
        if pver.parse(torch.__version__) < pver.parse("1.10"):
            return torch.meshgrid(*args)
        else:
            return torch.meshgrid(*args, indexing="ij")

    @staticmethod
    def ray_condition(
        K: torch.Tensor,
        c2w: torch.Tensor,
        H: int,
        W: int,
        device: str,
        flip_flag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算光线条件（Plücker坐标）

        Args:
            K: 相机内参矩阵 [B, V, 4]
            c2w: 相机到世界变换矩阵 [B, V, 4, 4]
            H: 图像高度
            W: 图像宽度
            device: 设备
            flip_flag: 翻转标志

        Returns:
            torch.Tensor: Plücker坐标 [B, V, H, W, 6]
        """
        B, V = K.shape[:2]

        # 创建像素网格
        i_base, j_base = _get_ray_grid(H, W, device, dtype=c2w.dtype)
        i = i_base.expand(B, V, H * W)
        j = j_base.expand(B, V, H * W)

        # 处理翻转情况
        if flip_flag is not None and torch.sum(flip_flag).item() > 0:
            j_flip, i_flip = NuscenesVideoDataset.custom_meshgrid(
                torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
                torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype),
            )
            i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
            j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
            i[:, flip_flag, ...] = i_flip
            j[:, flip_flag, ...] = j_flip

        # 分解内参矩阵
        fx, fy, cx, cy = K.chunk(4, dim=-1)  # B, V, 1

        # 计算光线方向
        zs = torch.ones_like(i)  # [B, V, HxW]
        xs = (i - cx) / fx * zs
        ys = (j - cy) / fy * zs
        zs = zs.expand_as(ys)

        # 构建光线方向并归一化
        directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
        directions = directions / directions.norm(dim=-1, keepdim=True)

        # 变换到世界坐标系
        rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
        rays_o = c2w[..., :3, 3]  # B, V, 3
        rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3

        # 计算Plücker坐标
        rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # B, V, HW, 3
        plucker = torch.cat([rays_dxo, rays_d], dim=-1)
        plucker = plucker.reshape(B, V, H, W, 6)  # B, V, H, W, 6

        return plucker

    def _get_camera_params(
        self,
        data: Dict,
        view_name: str,
        frame_idx: int,
        rel_ego2global_dict: Dict,
        ori_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        获取相机参数

        Args:
            data: 数据字典
            view_name: 视图名称
            frame_idx: 帧索引
            rel_ego2global_dict: 相对位姿字典
            ori_size: 原始图像尺寸
            target_size: 目标图像尺寸

        Returns:
            Dict: 包含相机参数的字典
        """
        cam_info = data["cams"][view_name]

        # 获取原始内参
        if "camera_intrinsics" in cam_info:
            current_camera_intrinsics = cam_info["camera_intrinsics"].copy()
        else:
            current_camera_intrinsics = cam_info["cam_intrinsic"].copy()
        current_camera_intrinsics = current_camera_intrinsics.astype(np.float32)

        ori_K = torch.from_numpy(current_camera_intrinsics.copy())

        # 计算调整后的内参
        rh, rw, i, j = self.get_resize_crop_param(ori_size, target_size)
        K = current_camera_intrinsics.copy()
        K[0, 0] *= rw
        K[1, 1] *= rh
        K[0, 2] *= rw
        K[1, 2] *= rh
        K[0, 2] -= j
        K[1, 2] -= i
        K = torch.from_numpy(K.astype(np.float32, copy=False))

        # 计算lidar2cam变换
        lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"]).astype(
            np.float32
        )
        lidar2cam_t = (
            cam_info["sensor2lidar_translation"].astype(np.float32) @ lidar2cam_r.T
        )
        lidar2cam_rt = np.eye(4).astype(np.float32)
        lidar2cam_rt[:3, :3] = lidar2cam_r.T
        lidar2cam_rt[3, :3] = -lidar2cam_t
        cam2lidar = torch.from_numpy(lidar2cam_rt.T)

        # 计算cam2ego变换
        cam2ego = np.eye(4, dtype=np.float32)
        cam2ego[:3, :3] = Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix
        cam2ego = torch.from_numpy(cam2ego)

        # 计算原始cam2ego变换（包含平移）
        cam2ego_ori = np.eye(4, dtype=np.float32)
        cam2ego_ori[:3, :3] = Quaternion(
            cam_info["sensor2ego_rotation"]
        ).rotation_matrix
        cam2ego_ori[:3, 3] = cam_info["sensor2ego_translation"]
        cam2ego_ori = torch.from_numpy(cam2ego_ori)

        # 计算c2w变换
        c2w = torch.from_numpy(
            np.dot(rel_ego2global_dict, cam2ego_ori.numpy()).astype(np.float32)
        )

        # 计算raymap相关参数（如果启用轨迹控制）
        raymap_params = {}
        if self.traj_ctrl:
            front_cam2ego = np.eye(4, dtype=np.float32)
            front_cam2ego[:3, :3] = Quaternion(
                data["cams"]["CAM_FRONT"]["sensor2ego_rotation"]
            ).rotation_matrix

            current_camera_c2w = np.dot(
                np.dot(front_cam2ego, np.linalg.inv(cam2ego.numpy())),
                np.dot(rel_ego2global_dict, cam2ego.numpy()),
            )
            raymap_c2w = torch.from_numpy(current_camera_c2w.astype(np.float32))

            # 计算raymap内参
            if "camera_intrinsics" in data["cams"]["CAM_FRONT"]:
                raymap_intrinsics = (
                    data["cams"]["CAM_FRONT"]["camera_intrinsics"]
                    .copy()
                    .astype(np.float32)
                )
            else:
                raymap_intrinsics = (
                    data["cams"]["CAM_FRONT"]["cam_intrinsic"]
                    .copy()
                    .astype(np.float32)
                )

            ray_map_rh, ray_map_rw, ray_map_i, ray_map_j = self.get_resize_crop_param(
                ori_size, (target_size[0] // 8, target_size[1] // 8)
            )

            raymap_intrinsics[0, 0] *= ray_map_rw
            raymap_intrinsics[1, 1] *= ray_map_rh
            raymap_intrinsics[0, 2] *= ray_map_rw
            raymap_intrinsics[1, 2] *= ray_map_rh
            raymap_intrinsics[0, 2] -= ray_map_j
            raymap_intrinsics[1, 2] -= ray_map_i
            raymap_K = torch.from_numpy(raymap_intrinsics.astype(np.float32))

            raymap_params = {
                "raymap_c2w": raymap_c2w,
                "raymap_K": raymap_K,
            }

        return {
            "K": K,
            "ori_K": ori_K,
            "cam2lidar": cam2lidar,
            "cam2ego": cam2ego,
            "cam2ego_ori": cam2ego_ori,
            "c2w": c2w,
            **raymap_params,
        }

    def _load_segmentation(
        self, image_path: str, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        加载分割掩码

        Args:
            image_path: 图像路径
            target_size: 目标尺寸

        Returns:
            torch.Tensor: 分割掩码
        """
        # Check if using T4 format with custom seg_root
        if (
            hasattr(self, "seg_root")
            and self.seg_root
            and hasattr(self, "t4_camera_map")
        ):
            seg_path = self._seg_path_cache.get(image_path)
            if seg_path is None:
                chunk_name, camera_name, filename = self._parse_t4_image_path(image_path)
                t4_camera = self.t4_camera_map.get(camera_name, camera_name)
                # SAM3 format: /mnt/nvme3/T4_datasets_sam3/{chunk_name}/{camera}/{filename}.png
                if hasattr(self, "seg_png_format") and self.seg_png_format:
                    seg_filename = filename.replace(".jpg", ".png")
                else:
                    seg_filename = filename
                seg_path = os.path.join(self.seg_root, chunk_name, t4_camera, seg_filename)
                self._seg_path_cache[image_path] = seg_path
        else:
            # Original nuScenes format
            seg_path = (
                image_path.replace("/nuscenes/", "/nuscenes_seg/")
                .replace(".jpg", ".png")
                .replace("/samples/", "/samples_seg/")
                .replace("/sweeps/", "/sweeps_seg/")
            )

        # Load segmentation as id map; T4 SAM3 masks are typically PNG id maps.
        seg_mask_ori = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if seg_mask_ori is None:
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")

        if seg_mask_ori.ndim == 3:
            # Keep previous behavior of using the first channel as class id source.
            seg_ids = seg_mask_ori[:, :, 0]
        else:
            seg_ids = seg_mask_ori

        # Vectorized remap: raw seg id -> compact class id -> RGB color.
        if seg_ids.max(initial=0) >= self._seg_class_lut.shape[0]:
            max_val = int(seg_ids.max())
            pad = max_val + 1 - self._seg_class_lut.shape[0]
            self._seg_class_lut = np.pad(self._seg_class_lut, (0, pad), mode="constant")
        class_ids = self._seg_class_lut[seg_ids]
        seg_mask = self._seg_color_lut[class_ids]

        seg_mask = _resize_crop_to_fill_cv2(
            seg_mask, target_size, interpolation=cv2.INTER_NEAREST
        )
        seg_mask_norm = _normalize_rgb_uint8_to_tensor(seg_mask)

        # Return raw id map for depth post-processing (sky mask, etc.).
        return seg_mask_norm, seg_ids

    def _load_depth(
        self,
        token: str,
        view_name: str,
        image_shape: Tuple[int, int, int],
        image_path: str,
        seg_mask_ori: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        加载深度图

        Args:
            token: 数据token
            view_name: 视图名称
            image_shape: 图像形状 (C, H, W)
            image_path: 图像路径 (for T4 format)
            seg_mask_ori: 原始分割掩码

        Returns:
            torch.Tensor: 深度图
        """
        # Check if using T4 format with custom depth_root
        if (
            hasattr(self, "depth_root")
            and self.depth_root
            and hasattr(self, "t4_camera_map")
        ):
            depth_path = self._depth_path_cache.get(image_path)
            if depth_path is None:
                chunk_name, camera_name, filename = self._parse_t4_image_path(image_path)
                t4_camera = self.t4_camera_map.get(camera_name, camera_name)
                # Format: /mnt/nvme1/data/T4_datasets_priorda_depth/{chunk_name}/{camera}/{filename}.png
                if hasattr(self, "depth_png_format") and self.depth_png_format:
                    depth_filename = filename.replace(".jpg", ".png")
                else:
                    depth_filename = filename
                depth_path = os.path.join(
                    self.depth_root, chunk_name, t4_camera, depth_filename
                )
                self._depth_path_cache[image_path] = depth_path

            # Load PNG depth
            depth_map_ori = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            if depth_map_ori is None:
                raise FileNotFoundError(f"Depth file not found: {depth_path}")

            # Convert uint16 to float32 using depth_scale (650 for Prior-Depth-Anything)
            if depth_map_ori.dtype == np.uint16:
                depth_scale = getattr(self, "depth_scale", 650)
                depth_map_ori = depth_map_ori.astype(np.float32) / depth_scale
        else:
            # Original nuScenes format
            depth_path = (
                Path("data/nuscenes_12hz_depth_unzip")
                / token
                / view_name
                / "refined_depth.npz"
            )

            # 加载深度图
            depth_map_ori = np.load(depth_path)["depth_pred"]

        # 调整大小并限制最大深度
        depth_map_resize = cv2.resize(depth_map_ori, (image_shape[2], image_shape[1]))
        depth_map_resize[depth_map_resize > self.max_depth] = self.max_depth

        # 处理天空区域（分割ID为2）
        if seg_mask_ori is None:
            # 如果没有提供分割掩码，则跳过
            pass
        else:
            seg_mask_resize = cv2.resize(
                seg_mask_ori,
                (image_shape[2], image_shape[1]),
                interpolation=cv2.INTER_NEAREST,
            )
            if seg_mask_resize.ndim == 3:
                seg_mask_resize = seg_mask_resize[:, :, 0]
            depth_map_resize[seg_mask_resize == 2] = self.max_depth

        # 归一化深度图
        depth_map_resize = depth_map_resize[:, :, None]
        depth_map_resize = (
            np.concatenate(
                (depth_map_resize, depth_map_resize, depth_map_resize), axis=2
            )
            / self.max_depth
        )

        depth_map_tensor = torch.from_numpy(depth_map_resize).permute(2, 0, 1)
        depth_map_norm = (depth_map_tensor - 0.5) / 0.5

        return depth_map_norm

    def get_data_info(
        self,
        index: int,
        num_frames: int,
        height: int,
        width: int,
        epsilon: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        获取数据信息

        Args:
            index: 数据索引
            num_frames: 帧数
            height: 图像高度
            width: 图像宽度
            epsilon: 数值稳定性参数

        Returns:
            Dict: 包含所有数据信息的字典
        """
        # 验证索引范围
        if self.infer_for_test and (
            index < self.infer_start_index or index >= self.infer_end_index
        ):
            raise ValueError(
                f"Index {index} out of range [{self.infer_start_index}, {self.infer_end_index})"
            )

        # 获取数据信息列表
        if self.is_occ_infer:
            data_infos = self.data[index]
        else:
            if self.infer_index is not None and not self.is_train:
                data_index_list = self.data[self.infer_index]
            else:
                data_index_list = self.data[index]
            # NOTE:
            # The clip sampler may return non-contiguous frame indices (e.g., custom scene token lists).
            # Always gather frames by explicit indices instead of assuming contiguous slices.
            data_infos = [self.data_infos[data_idx] for data_idx in data_index_list[:num_frames]]

        # 初始化返回字典和存储列表
        ret = {}
        multi_view_video_list = []
        multi_view_ray_map_list = []
        multi_view_depth_list = []
        multi_view_seg_list = []
        c2w_all_list = []
        raymap_c2w_all_list = []
        cam2ego_all_list = []
        ori_cam2ego_all_list = []
        K_all_list = []
        ori_K_all_list = []
        raymap_k_all_list = []
        all_cam2lidar_list = []
        all_img_path_list = []
        all_traj_list = []
        ray_sample_index = list(range(num_frames))[::4] if self.traj_ctrl else []
        ray_sample_index_set = set(ray_sample_index)

        # Reuse decoded tensors within one sample when duplicated frame paths appear.
        enable_local_cache = self.transform_name == "resize_crop"
        image_cache = {}
        seg_cache = {}
        depth_cache = {}

        # 处理每一帧
        for frame_idx, data in enumerate(data_infos):
            # 初始化当前帧的存储列表
            video_list = []
            depth_list = []
            seg_list = []
            c2w_list = []
            raymap_c2w_list = []
            K_list = []
            ori_K_list = []
            raymap_k_list = []
            cam2lidar_list = []
            img_path_list = []
            cam2ego_list = []
            ori_cam2ego_list = []

            # 计算相对位姿
            rel_ego2global_dict = self._calculate_relative_poses(
                data, frame_idx, num_frames, data_infos
            )
            token = data["token"]

            # 处理每个视图
            for view_name in self.view_order:
                # 获取相机信息
                cam_info = data["cams"][view_name]
                cam_path = cam_info["data_path"]

                temp_path = self._resolve_image_path(cam_path)

                # 加载并处理图像
                image_key = (temp_path, height, width)
                if enable_local_cache and image_key in image_cache:
                    image, ori_size = image_cache[image_key]
                else:
                    image, ori_size = _load_rgb_tensor_cv2(
                        temp_path,
                        (height, width),
                        use_turbojpeg=getattr(self, "use_turbojpeg", True),
                    )
                    if enable_local_cache:
                        image_cache[image_key] = (image, ori_size)
                video_list.append(image)
                img_path_list.append(temp_path)

                # 获取相机参数
                cam_params = self._get_camera_params(
                    data,
                    view_name,
                    frame_idx,
                    rel_ego2global_dict,
                    ori_size,
                    (height, width),
                )

                K_list.append(cam_params["K"])
                ori_K_list.append(cam_params["ori_K"])
                cam2lidar_list.append(cam_params["cam2lidar"])
                seg_mask_for_depth = None
                # 加载分割掩码（如果启用）
                if self.use_seg and self.is_train:
                    try:
                        seg_key = (temp_path, height, width)
                        if enable_local_cache and seg_key in seg_cache:
                            seg_mask, seg_mask_for_depth = seg_cache[seg_key]
                        else:
                            seg_mask, seg_mask_for_depth = self._load_segmentation(
                                temp_path, (height, width)
                            )
                            if enable_local_cache:
                                seg_cache[seg_key] = (seg_mask, seg_mask_for_depth)
                        seg_list.append(seg_mask)
                    except Exception as e:
                        print(f"Failed to load segmentation for {temp_path}: {e}")
                        return None
                else:
                    seg_list.append(image)

                # 加载深度图（如果启用）
                if self.use_depth and self.is_train:
                    try:
                        depth_key = (token, view_name, temp_path, height, width)
                        if enable_local_cache and depth_key in depth_cache:
                            depth_map = depth_cache[depth_key]
                        else:
                            depth_map = self._load_depth(
                                token,
                                view_name,
                                image.shape,
                                temp_path,
                                seg_mask_for_depth,
                            )
                            if enable_local_cache:
                                depth_cache[depth_key] = depth_map
                        depth_list.append(depth_map)
                    except Exception as e:
                        print(f"Failed to load depth for {token}/{view_name}: {e}")
                        return None
                else:
                    depth_list.append(image)

                # 存储其他相机参数
                cam2ego_list.append(cam_params["cam2ego"])
                ori_cam2ego_list.append(cam_params["cam2ego_ori"])
                c2w_list.append(cam_params["c2w"])

                # 处理raymap相关参数（如果启用轨迹控制）
                if self.traj_ctrl and "raymap_c2w" in cam_params:
                    raymap_c2w_list.append(cam_params["raymap_c2w"])
                    raymap_k_list.append(cam_params["raymap_K"])

            # 将当前帧的数据堆叠并添加到总列表中
            multi_view_video_list.append(torch.stack(video_list))
            multi_view_depth_list.append(torch.stack(depth_list))
            multi_view_seg_list.append(torch.stack(seg_list))

            K_all_list.append(torch.stack(K_list))
            ori_K_all_list.append(torch.stack(ori_K_list))
            c2w_all_list.append(torch.stack(c2w_list))
            all_cam2lidar_list.append(torch.stack(cam2lidar_list))
            all_img_path_list.append(img_path_list)

            # 处理轨迹控制相关数据
            if self.traj_ctrl:
                # Ray-map is only consumed at sampled timesteps (::4). Avoid computing dropped frames.
                if frame_idx in ray_sample_index_set:
                    # Compute all camera ray-maps together to reduce Python/kernel launch overhead.
                    raymap_k_t = torch.stack(raymap_k_list)  # [V, 3/4, 3/4]
                    raymap_c2w_t = torch.stack(raymap_c2w_list)  # [V, 4, 4]
                    ram_map_k = torch.zeros(1, raymap_k_t.shape[0], 4, dtype=raymap_k_t.dtype)
                    ray_map_c2w = raymap_c2w_t.unsqueeze(0)
                    ram_map_k[0, :, 0] = raymap_k_t[:, 0, 0]
                    ram_map_k[0, :, 1] = raymap_k_t[:, 1, 1]
                    ram_map_k[0, :, 2] = raymap_k_t[:, 0, 2]
                    ram_map_k[0, :, 3] = raymap_k_t[:, 1, 2]

                    ray_map_ori = self.ray_condition(
                        ram_map_k,
                        ray_map_c2w,
                        image.shape[1] // 8,
                        image.shape[2] // 8,
                        device="cpu",
                    )
                    # [1, V, H, W, 6] -> [V, 6, H, W]
                    ray_map_views = ray_map_ori[0].permute(0, 3, 1, 2).contiguous()
                    multi_view_ray_map_list.append(ray_map_views)
                raymap_k_all_list.append(torch.stack(raymap_k_list))
                cam2ego_all_list.append(torch.stack(cam2ego_list))
                ori_cam2ego_all_list.append(torch.stack(ori_cam2ego_list))
                raymap_c2w_all_list.append(torch.stack(raymap_c2w_list))

                # 计算轨迹真值
                rel_angles = self.rotation_matrix_to_euler_angles(
                    rel_ego2global_dict[:3, :3]
                )
                all_traj_list.append(
                    torch.cat(
                        [
                            torch.from_numpy(rel_ego2global_dict[:3, 3]).to(
                                torch.float32
                            )[:, None],
                            torch.from_numpy(rel_angles).to(torch.float32)[:, None],
                        ],
                        dim=1,
                    )
                )

        # 将所有帧的数据堆叠并重新排列维度
        multi_view_video = torch.stack(multi_view_video_list).permute(1, 2, 0, 3, 4)
        ret["video"] = multi_view_video  # 需要 B C T H W

        if self.use_depth:
            multi_view_depth = torch.stack(multi_view_depth_list).permute(1, 2, 0, 3, 4)
            ret["depth"] = multi_view_depth

        if self.use_seg:
            multi_view_seg = torch.stack(multi_view_seg_list).permute(1, 2, 0, 3, 4)
            ret["seg"] = multi_view_seg

        # 堆叠相机参数
        c2w = torch.stack(c2w_all_list).permute(1, 0, 2, 3)
        K = torch.stack(K_all_list).permute(1, 0, 2, 3)
        cam2lidar = torch.stack(all_cam2lidar_list).permute(1, 0, 2, 3)
        ori_K = torch.stack(ori_K_all_list).permute(1, 0, 2, 3)

        ret.update(
            {
                "c2w": c2w,
                "cam2lidar": cam2lidar,
                "cam_k": K,
                "ori_cam_k": ori_K,
                "index": index,
                "id": index,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "fps": 12,
                "path": all_img_path_list,
            }
        )

        # 处理轨迹控制相关输出
        if self.traj_ctrl:
            multi_view_ray_map = torch.stack(multi_view_ray_map_list).permute(
                1, 2, 0, 3, 4
            )
            all_traj_seq = torch.stack(all_traj_list)
            ret["traj_gt"] = all_traj_seq

            # ray-map is already computed only on sampled timesteps.
            sample_index = ray_sample_index
            multi_view_ray_map_sample = multi_view_ray_map
            multi_view_ray_map_sample = self._apply_ray_map_zeroing(
                multi_view_ray_map_sample
            )

            ret.update(
                {
                    "ray_map": multi_view_ray_map_sample,
                    "raymap_k": torch.stack(raymap_k_all_list).permute(1, 0, 2, 3)[
                        :, sample_index
                    ],
                    "cam2ego": torch.stack(cam2ego_all_list).permute(1, 0, 2, 3)[
                        :, sample_index
                    ],
                    "ori_cam2ego": torch.stack(ori_cam2ego_all_list).permute(
                        1, 0, 2, 3
                    )[:, sample_index],
                    "ori_cam2ego_all": torch.stack(ori_cam2ego_all_list).permute(
                        1, 0, 2, 3
                    ),
                }
            )

        return ret

    def _calculate_relative_poses(
        self, data: Dict, frame_idx: int, num_frames: int, data_infos: list
    ) -> Dict[str, np.ndarray]:
        """
        计算相对位姿

        Args:
            data: 当前帧数据
            frame_idx: 帧索引
            num_frames: 总帧数

        Returns:
            Dict: 各视图的相对位姿字典
        """
        # for view_name in self.view_order:
        if self.is_train or self.test_index is None:
            # 使用第一帧作为参考
            first = data_infos[0]["cams"]["CAM_FRONT"]
            e2g_r = first["ego2global_rotation"]
            e2g_t = first["ego2global_translation"]

            # 当前帧的位姿
            e2g_r_s = data["cams"]["CAM_FRONT"]["ego2global_rotation"]
            e2g_t_s = data["cams"]["CAM_FRONT"]["ego2global_translation"]
            e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
        else:
            # 使用测试索引
            test_data = self.data_infos[
                self.data[self.test_index][0] : self.data[self.test_index][0]
                + num_frames
            ][frame_idx]["cams"]["CAM_FRONT"]

            first = self.data_infos[
                self.data[self.test_index][0] : self.data[self.test_index][0]
                + num_frames
            ][0]["cams"]["CAM_FRONT"]

            e2g_r = first["ego2global_rotation"]
            e2g_t = first["ego2global_translation"]
            e2g_r_s = test_data["ego2global_rotation"]
            e2g_t_s = test_data["ego2global_translation"]
            e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

        # 构建变换矩阵
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        first_e2g = np.eye(4, dtype=np.float32)
        first_e2g[:3, :3] = e2g_r_mat.astype(np.float32)
        first_e2g[:3, 3] = np.asarray(e2g_t, dtype=np.float32)

        current_e2g = np.eye(4, dtype=np.float32)
        current_e2g[:3, :3] = e2g_r_s_mat.astype(np.float32)
        current_e2g[:3, 3] = np.asarray(e2g_t_s, dtype=np.float32)

        # 计算相对变换
        temp_rel_ego2global = np.dot(np.linalg.inv(first_e2g), current_e2g)
        rel_ego2global = np.eye(4, dtype=np.float32)
        rel_ego2global[:3, :3] = temp_rel_ego2global[:3, :3].astype(np.float32)
        rel_ego2global[:3, 3] = temp_rel_ego2global[:3, 3].astype(np.float32)

        return rel_ego2global

    def _resolve_image_path(self, cam_path: str) -> str:
        """
        解析图像路径

        Args:
            cam_path: 原始相机路径

        Returns:
            str: 解析后的路径
        """
        # 根据实际情况调整路径解析逻辑
        if cam_path.startswith("../data/nuscenes/"):
            cam_path = cam_path.replace("../data/nuscenes/", "data/nuscenes/")

        return cam_path

    def _parse_t4_image_path(self, image_path: str) -> Tuple[str, str, str]:
        """
        Parse T4 image path into (chunk_name, camera_name, filename).
        """
        norm_path = image_path.replace("\\", "/")
        cached = self._t4_path_cache.get(norm_path)
        if cached is not None:
            return cached

        parts = norm_path.split("/")
        chunk_name = ""
        camera_name = ""
        filename = ""

        for i, part in enumerate(parts):
            if part == "t4_datasets" and i + 1 < len(parts):
                chunk_name = parts[i + 1]
            if part == "data" and i + 1 < len(parts):
                camera_name = parts[i + 1]
                if i + 2 < len(parts):
                    filename = parts[i + 2]

        parsed = (chunk_name, camera_name, filename)
        self._t4_path_cache[norm_path] = parsed
        return parsed

    def __getitem__(self, index: Union[int, str]) -> Dict[str, Any]:
        """
        获取数据项

        Args:
            index: 索引（整数或字符串）

        Returns:
            Dict: 数据项字典
        """
        if self.is_train:
            # print(index)
            index, num_frames, height, width = map(int, index.split("-"))
        else:
            num_frames = self.infer_num_frames
            height = self.infer_height
            width = self.infer_width

        max_retries = int(getattr(self, "max_sample_retries", 100))
        retries = 0
        while retries < max_retries:
            ret = self.get_data_info(index, num_frames, height, width)
            if ret is not None:
                return ret
            else:
                print(f"video {index}")
                index = self._rand_another()
                print(f"new video {index}")
                retries += 1

        raise RuntimeError(
            f"NuscenesVideoDataset failed to fetch a valid sample after {max_retries} retries. "
            "Please check depth/seg paths and dataset integrity."
        )

    def __len__(self) -> int:
        """
        返回数据集长度

        Returns:
            int: 数据集大小
        """
        return len(self.data)


@DATASETS.register_module("infer_video")
class InferVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        multi_view_path_list=[
            {
                "CAM_FRONT_LEFT": "data/20240918-124054_91vp/frames/video4/20240918-124800/frame-003398_1726635020554.jpg",
                "CAM_FRONT": "data/20240918-124054_91vp/frames/video1/20240918-124800/frame-003398_1726635020555.jpg",
                "CAM_FRONT_RIGHT": "data/20240918-124054_91vp/frames/video8/20240918-124800/frame-003398_1726635020554.jpg",
                "CAM_BACK_RIGHT": "data/20240918-124054_91vp/frames/video7/20240918-124800/frame-003398_1726635020554.jpg",
                "CAM_BACK": "data/20240918-124054_91vp/frames/video6/20240918-124800/frame-003398_1726635020554.jpg",
                "CAM_BACK_LEFT": "data/20240918-124054_91vp/frames/video5/20240918-124800/frame-003398_1726635020554.jpg",
            },
        ],
        multi_view_intrinsics_list=[
            {
                "CAM_FRONT_LEFT": [  # CAM_L0
                    [1.21895984e03, 0.00000000e00, 9.57363475e02, 0.00000000e00],
                    [0.00000000e00, 1.22456364e03, 5.41585682e02, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 1.00000000e00, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ],
                "CAM_FRONT": [  # CAM_F0
                    [1.21895984e03, 0.00000000e00, 9.57363475e02, 0.00000000e00],
                    [0.00000000e00, 1.22456364e03, 5.41585682e02, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 1.00000000e00, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ],
                "CAM_FRONT_RIGHT": [  # CAM_R0
                    [1.21895984e03, 0.00000000e00, 9.57363475e02, 0.00000000e00],
                    [0.00000000e00, 1.22456364e03, 5.41585682e02, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 1.00000000e00, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ],
                "CAM_BACK_RIGHT": [  # CAM_R1
                    [1.21895984e03, 0.00000000e00, 9.57363475e02, 0.00000000e00],
                    [0.00000000e00, 1.22456364e03, 5.41585682e02, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 1.00000000e00, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ],
                "CAM_BACK": [  # CAM_B0
                    [1.21895984e03, 0.00000000e00, 9.57363475e02, 0.00000000e00],
                    [0.00000000e00, 1.22456364e03, 5.41585682e02, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 1.00000000e00, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ],
                "CAM_BACK_LEFT": [  # CAM_L2
                    [1.21895984e03, 0.00000000e00, 9.57363475e02, 0.00000000e00],
                    [0.00000000e00, 1.22456364e03, 5.41585682e02, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 1.00000000e00, 0.00000000e00],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
                ],
            },
        ],
        multi_view_extrinsics_list=[
            {
                "CAM_FRONT_LEFT": [  # CAM_L0
                    [0.86201404, 0.01009335, 0.50678391, 0.0],
                    [-0.50663421, -0.01425351, 0.86204328, 0.0],
                    [0.01592435, -0.99984747, -0.00717309, 0.0],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0],
                ],
                "CAM_FRONT": [  # CAM_F0
                    [0.00126075, 0.01200641, 0.99992713, 0.0],
                    [-0.9999793, 0.00632358, 0.00118489, 0.0],
                    [-0.00630889, -0.99990792, 0.01201414, 0.0],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0],
                ],
                "CAM_FRONT_RIGHT": [  # CAM_R0
                    [-8.63190382e-01, 2.07420307e-02, 5.04452310e-01, 0.0],
                    [-5.04293909e-01, 1.26489588e-02, -8.63439434e-01, 0.0],
                    [-2.42902837e-02, -9.99704842e-01, -4.58379799e-04, 0.0],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0],
                ],
                "CAM_BACK_RIGHT": [  # CAM_R1
                    [-0.81506139, 0.01696596, -0.57912613, 0.0],
                    [0.57907397, -0.00834085, -0.81523234, 0.0],
                    [-0.0186616, -0.99982128, -0.00302624, 0.0],
                    [0.0000e00, 0.0000e00, 0.0000e00, 1.0],
                ],
                "CAM_BACK": [  # CAM_B0
                    [0.025928, 0.02265023, -0.99940718, 0.0],
                    [0.99930294, -0.02744795, 0.02530322, 0.0],
                    [-0.02685855, -0.99936659, -0.02334611, 0.0],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.0],
                ],
                "CAM_BACK_LEFT": [  # CAM_L2
                    [0.80922905, 0.01857087, -0.58719968, 0.0],
                    [0.58742686, -0.01054935, 0.8092085, 0.0],
                    [0.00883314, -0.99977189, -0.01944587, 0.0],
                    [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.0],
                ],
            },
        ],
        traj_list=[
            {
                "x": [
                    0.0000,
                    0.3917,
                    0.5821,
                    0.9530,
                    1.3174,
                    1.5121,
                    1.8937,
                    2.2628,
                    2.4431,
                    2.7968,
                    3.1456,
                    3.3225,
                    3.6700,
                    4.0174,
                    4.1858,
                    4.5238,
                    4.8631,
                ],
                "y": [
                    0.0000e00,
                    7.3542e-04,
                    2.7183e-03,
                    9.3033e-03,
                    1.9415e-02,
                    2.6329e-02,
                    4.2032e-02,
                    6.2932e-02,
                    7.5704e-02,
                    1.0607e-01,
                    1.4249e-01,
                    1.6319e-01,
                    2.1026e-01,
                    2.6655e-01,
                    2.9804e-01,
                    3.6682e-01,
                    4.4582e-01,
                ],
                "heading": [
                    0.0000e00,
                    8.3332e-03,
                    1.2621e-02,
                    2.2932e-02,
                    3.4672e-02,
                    4.1350e-02,
                    5.5275e-02,
                    7.0716e-02,
                    7.8813e-02,
                    9.7313e-02,
                    1.1750e-01,
                    1.2824e-01,
                    1.5042e-01,
                    1.7532e-01,
                    1.8826e-01,
                    2.1578e-01,
                    2.4484e-01,
                ],
            },
        ],
        num_frames=17,
        height=224,
        width=400,
        max_depth=100.0,
        transform_name: str = "resize_crop",
        dataset_name: str = "nuscenes",
        memory_efficient=False,
        use_turbojpeg: bool = True,
    ):
        self.memory_efficient = memory_efficient
        self.use_turbojpeg = use_turbojpeg
        self.data = multi_view_intrinsics_list
        self.multi_view_path_list = multi_view_path_list
        self.multi_view_intrinsics_list = multi_view_intrinsics_list
        self.multi_view_extrinsics_list = multi_view_extrinsics_list
        self.traj_list = traj_list
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.transform_name = transform_name
        self.dataset_name = dataset_name
        if self.dataset_name == "nuplan":
            self.P = np.array(
                [
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1],
                ]
            )
        elif self.dataset_name == "nuscenes":
            self.P = np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            )
        else:
            raise ValueError(f"Unsupported dataset name: {self.dataset_name}")

        self.seg_color_map = [
            (0, 0, 0),  # object
            (0, 207, 191),  # road
            (135, 206, 235),  # sky
            (255, 158, 0),  # car
            (0, 0, 230),  # person
            (255, 61, 99),  # bicycle
            (192, 192, 192),  # lane
            (75, 0, 75),  # sidewalk
            (0, 175, 0),  # vegentation
        ]

        self.max_depth = max_depth

        if self.memory_efficient:
            self._convert_to_efficient()

    def _convert_to_efficient(self) -> None:
        """转换为高效存储格式"""
        if self.memory_efficient:
            addition_data_path = self.pkl_path.split(".")[0]
            self._data = self.data
            self.data = EfficientParquet(self._data, addition_data_path)

    def get_data_info(
        self,
        index: int,
        num_frames: int,
        height: int,
        width: int,
        epsilon: float = 1e-6,
    ):
        camera_dict = self.multi_view_path_list[index]
        intrinsics_dict = self.multi_view_intrinsics_list[index]
        extrinsics_dict = self.multi_view_extrinsics_list[index]
        traj_dict = self.traj_list[index]

        ret = {}
        multi_view_video_list = []
        multi_view_ray_map_list = []
        # multi_view_depth_list = []
        # multi_view_seg_list = []
        c2w_all_list = []
        raymap_c2w_all_list = []
        cam2ego_all_list = []
        ori_cam2ego_all_list = []
        K_all_list = []
        ori_K_all_list = []
        raymap_k_all_list = []
        all_img_path_list = []
        all_traj_list = []
        view_order = list(camera_dict.keys())

        for frame_id in range(num_frames):
            video_list = []
            ray_map_list = []
            # depth_list = []
            # seg_list = []
            c2w_list = []
            raymap_c2w_list = []
            K_list = []
            ori_K_list = []
            raymap_k_list = []
            img_path_list = []
            cam2ego_list = []
            ori_cam2ego_list = []

            x = traj_dict["x"][frame_id]
            y = traj_dict["y"][frame_id]
            heading = traj_dict["heading"][frame_id]
            rel_ego2global = np.eye(4)
            heading_matrix = NuscenesVideoDataset.eulerAngles2rotationMat(
                [0.0, 0.0, heading]
            )
            rel_ego2global[:3, :3] = self.P @ heading_matrix
            rel_ego2global[:3, 3] = [x, y, 0]

            all_traj_list.append(
                torch.cat(
                    [
                        torch.from_numpy(rel_ego2global[:3, 3])[:, None],
                        torch.from_numpy(
                            NuscenesVideoDataset.rotation_matrix_to_euler_angles(
                                heading_matrix
                            )
                        )[:, None],
                    ],
                    dim=1,
                )
            )

            for view_name in view_order:
                img_path = camera_dict[view_name]
                image, ori_size = _load_rgb_tensor_cv2(
                    img_path,
                    (height, width),
                    use_turbojpeg=getattr(self, "use_turbojpeg", True),
                )
                img_path_list.append(img_path)
                video_list.append(image)
                # seg_list.append(image)
                # depth_list.append(image)
                front_cam2ego = np.eye(4)
                front_cam2ego[:3, :3] = np.array(extrinsics_dict["CAM_FRONT"])[:3, :3]
                cam2ego = np.eye(4)
                cam2ego[:3, :3] = np.array(extrinsics_dict[view_name])[:3, :3]
                cam2ego_list.append(torch.from_numpy(cam2ego))
                cam2ego_ori = np.eye(4)
                cam2ego_ori[:3, :3] = np.array(extrinsics_dict[view_name])[:3, :3]
                cam2ego_ori[:3, 3] = np.array(extrinsics_dict[view_name])[:3, 3]
                ori_cam2ego_list.append(torch.from_numpy(cam2ego_ori))

                c2w = torch.from_numpy(np.dot(rel_ego2global, cam2ego_ori))
                c2w_list.append(c2w)

                current_camera_c2w = np.dot(
                    np.dot(front_cam2ego, np.linalg.inv(cam2ego)),
                    np.dot(rel_ego2global, cam2ego),
                )
                current_camera_c2w = torch.from_numpy(current_camera_c2w)
                raymap_c2w_list.append(current_camera_c2w)

                current_raymap_camera_intrinsics = np.array(
                    intrinsics_dict["CAM_FRONT"]
                ).copy()
                ori_K_list.append(
                    torch.from_numpy(current_raymap_camera_intrinsics.copy())
                )
                ray_map_rh, ray_map_rw, ray_map_i, ray_map_j = (
                    NuscenesVideoDataset.get_resize_crop_param(
                        ori_size, (height // 8, width // 8)
                    )
                )
                current_raymap_camera_intrinsics[0, 0] *= ray_map_rw
                current_raymap_camera_intrinsics[1, 1] *= ray_map_rh
                current_raymap_camera_intrinsics[0, 2] *= ray_map_rw
                current_raymap_camera_intrinsics[1, 2] *= ray_map_rh
                current_raymap_camera_intrinsics[0, 2] -= ray_map_j
                current_raymap_camera_intrinsics[1, 2] -= ray_map_i
                current_raymap_camera_intrinsics = torch.from_numpy(
                    current_raymap_camera_intrinsics
                )
                K_list.append(current_raymap_camera_intrinsics)
                raymap_k_list.append(current_raymap_camera_intrinsics)

                ram_map_k = torch.zeros(1, 1, 4)
                ray_map_c2w = torch.zeros(1, 1, 4, 4)
                ray_map_c2w[0, 0] = current_camera_c2w
                ram_map_k[:, :, 0] = current_raymap_camera_intrinsics[0, 0]
                ram_map_k[:, :, 1] = current_raymap_camera_intrinsics[1, 1]
                ram_map_k[:, :, 2] = current_raymap_camera_intrinsics[0, 2]
                ram_map_k[:, :, 3] = current_raymap_camera_intrinsics[1, 2]

                ray_map_ori = NuscenesVideoDataset.ray_condition(
                    ram_map_k,
                    ray_map_c2w,
                    image.shape[1] // 8,
                    image.shape[2] // 8,
                    device="cpu",
                )
                ray_map = ray_map_ori[0, 0].permute(2, 0, 1)
                ray_map_list.append(ray_map)

            multi_view_video_list.append(torch.stack(video_list))
            # multi_view_depth_list.append(torch.stack(depth_list))
            # multi_view_seg_list.append(torch.stack(seg_list))

            K_all_list.append(torch.stack(K_list))
            ori_K_all_list.append(torch.stack(ori_K_list))
            c2w_all_list.append(torch.stack(c2w_list))
            all_img_path_list.append(img_path_list)

            multi_view_ray_map_list.append(torch.stack(ray_map_list))
            raymap_k_all_list.append(torch.stack(raymap_k_list))
            cam2ego_all_list.append(torch.stack(cam2ego_list))
            ori_cam2ego_all_list.append(torch.stack(ori_cam2ego_list))
            raymap_c2w_all_list.append(torch.stack(raymap_c2w_list))

        multi_view_video = torch.stack(multi_view_video_list).permute(1, 2, 0, 3, 4)
        ret["video"] = multi_view_video  # 需要 B C T H W
        c2w = torch.stack(c2w_all_list).permute(1, 0, 2, 3)
        K = torch.stack(K_all_list).permute(1, 0, 2, 3)
        ori_K = torch.stack(ori_K_all_list).permute(1, 0, 2, 3)

        ret.update(
            {
                "c2w": c2w,
                "cam_k": K,
                "ori_cam_k": ori_K,
                "index": index,
                "id": index,
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "fps": 12,
                "path": all_img_path_list,
            }
        )

        multi_view_ray_map = torch.stack(multi_view_ray_map_list).permute(1, 2, 0, 3, 4)
        all_traj_seq = torch.stack(all_traj_list)
        ret["traj_gt"] = all_traj_seq
        sample_index = list(range(multi_view_ray_map.shape[2]))[::4]
        multi_view_ray_map_sample = multi_view_ray_map[:, :, sample_index]
        ret.update(
            {
                "ray_map": multi_view_ray_map_sample,
                "raymap_k": torch.stack(raymap_k_all_list).permute(1, 0, 2, 3)[
                    :, sample_index
                ],
                "cam2ego": torch.stack(cam2ego_all_list).permute(1, 0, 2, 3)[
                    :, sample_index
                ],
                "ori_cam2ego": torch.stack(ori_cam2ego_all_list).permute(1, 0, 2, 3)[
                    :, sample_index
                ],
                "ori_cam2ego_all": torch.stack(ori_cam2ego_all_list).permute(
                    1, 0, 2, 3
                ),
            }
        )

        return ret

    def getitem(self, index: str) -> dict:
        num_frames = self.num_frames
        height = self.height
        width = self.width
        ret = self.get_data_info(index, num_frames, height, width)
        return ret

    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.multi_view_path_list)
