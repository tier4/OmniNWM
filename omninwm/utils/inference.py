import os
import cv2
import torch
import numpy as np
import json
import torch.nn.functional as F
from PIL import Image
from enum import Enum
import torch.nn as nn
from io import BytesIO
from IPython import embed
from einops import rearrange
from omninwm.datasets.utils import save_sample, colorize
from omninwm.utils.vis_occ import occ_draw

import matplotlib.pyplot as plt

class SamplingMethod(Enum):
    I2V = "i2v"  # for open sora video generation

def concat_6_views_pt(imgs, oneline=False):
    if oneline:
        imgs = rearrange(imgs, "NC C T H W -> C T H (NC W)")
    else:
        if imgs.shape[0] == 6:
            imgs_up = rearrange(imgs[:3], "NC C T H W -> C T H (NC W)")
            imgs_down = rearrange(imgs[3:], "NC C T H W -> C T H (NC W)")
            imgs = torch.cat([imgs_up, imgs_down], dim=2)
        elif imgs.shape[0] == 5:
            # Keep the original 2x3 layout and pad the missing CAM_BACK slot with black.
            imgs_up = rearrange(imgs[:3], "NC C T H W -> C T H (NC W)")
            black_back = torch.zeros_like(imgs[:1])
            imgs_down = rearrange(
                torch.cat([imgs[3:4], black_back, imgs[4:5]], dim=0),
                "NC C T H W -> C T H (NC W)",
            )
            imgs = torch.cat([imgs_up, imgs_down], dim=2)
        elif imgs.shape[0] == 3:
            imgs = rearrange(imgs[:3], "NC C T H W -> C T H (NC W)")
        elif imgs.shape[0] == 4:
            imgs_up = rearrange(imgs[:2], "NC C T H W -> C T H (NC W)")
            imgs_down = rearrange(imgs[2:], "NC C T H W -> C T H (NC W)")
            imgs = torch.cat([imgs_up, imgs_down], dim=2)
        elif imgs.shape[0] == 1:
            return rearrange(imgs[:1], "NC C T H W -> C T H (NC W)")
    return imgs

def get_save_path_name(
    save_dir,
    sub_dir,
    save_prefix="",
    name=None,
    fallback_name=None,
    index=None,
    num_sample_pos=None,  # idx for prompt as path
):
    """
    Get the save path for the generated samples.
    """
    if name is not None:
        fname = save_prefix + name
    elif fallback_name is not None:
        fname = f"{save_prefix + fallback_name}_{index:04d}"
    else:
        fname = f"{num_sample_pos}_{index:04d}"

    return os.path.join(save_dir, sub_dir, fname)

def get_names_from_path(path):
    """
    Get the filename and extension from a path.

    Args:
        path (str): The path to the file.

    Returns:
        tuple[str, str]: The filename and the extension.
    """
    filename = os.path.basename(path)
    name, _ = os.path.splitext(filename)
    return name


def normalize(x, value_range=(-1, 1),only_clamp=False):
    low, high = value_range
    x.clamp_(min=low, max=high)
    if not only_clamp:
        x.sub_(low).div_(max(high - low, 1e-5))
    return x

def process_depth(
    depth_latent,
    max_depth=100
):
    depth_mean = depth_latent.mean(dim=1,keepdim=True)
    depth = (normalize(depth_mean)*max_depth).to("cpu", torch.float32)
    return depth

def process_seg(
    seg_latent,
    seg_color_map,
    devices
):
    torch.cuda.empty_cache()
    seg_norm = normalize(seg_latent.clone())
    seg_denorm = seg_norm.clone().mul_(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
    color_map_array = torch.tensor(seg_color_map).unsqueeze(0).repeat(seg_denorm.shape[2],1,1).to(devices,torch.uint8)
    semantic_list = []
    color_map_array = torch.tensor(seg_color_map).to(devices,torch.uint8)
    for i in range(seg_denorm.shape[2]):
        current_semantic = seg_denorm[:,:,i:i+1].unsqueeze(0).repeat(color_map_array.shape[0], 1, 1, 1, 1, 1).permute(4, 5, 1, 3, 0, 2).to(devices, torch.uint8)
        current_distance = torch.linalg.norm(current_semantic.float() - color_map_array, dim=5)
        current_semantic = torch.argmin(current_distance, dim=4).to('cpu')  # H, W
        torch.cuda.empty_cache()
        semantic_list.append(current_semantic.permute(2, 3, 0, 1).to(torch.uint8))

    seg_mask = torch.cat(semantic_list,dim=1)
    
    return seg_norm, seg_mask.int()

def vis_input_traj(
    batch,
    rewards_list=None,
    arrow_color="#2C32D5",
    traj_color="#5B85C7",
    traj_label="GT/Input Ego Trajectory",
    height = 448,
    width = 800
):
    traj_gt = batch['traj_gt']
    all_traj_img = []
    traj_b, traj_t, _,_ = traj_gt.shape

    vis_rewards = traj_t==len(rewards_list)

    # new_traj_gt = traj_gt
    sample_index_list = list(range(traj_t))
    sample_index = sample_index_list[::1]
    traj_position = traj_gt[:,sample_index]
    history_x = []
    history_y = []
    history_headings = []
    x_max = traj_position[0,:,1,0].max()
    x_min = traj_position[0,:,1,0].min()
    y_max = traj_position[0,:,0,0].max()
    y_min = traj_position[0,:,0,0].min()


    ylim_max = 40
    ylim_min = -5

    if ylim_max<y_max:
        ylim_max = y_max+15
        

    if ylim_min > y_min:
        ylim_min = y_min-15

    xlim_max = int(ylim_max/2)
    xlim_min = int(-ylim_max/2)

    if xlim_max<x_max:
        xlim_max = x_max+15
        xlim_min = -(xlim_max)

    # if xlim_min > x_min:
    #     xlim_min = x_min-15

    for traj_idx in range(traj_t):
        buffer_ = BytesIO()
        traj_pose = traj_position[0,traj_idx,:2,0]
        heading = traj_position[0,traj_idx,2,1]
        # 累积历史数据
        history_x.append(traj_pose[1])
        history_y.append(traj_pose[0])
        history_headings.append(heading)
        fig, ax = plt.subplots()
        ax.plot(history_x, history_y, traj_color, linewidth=4, alpha=0.7, label=traj_label,marker='o',linestyle='-',markersize=2)
        current_angle = history_headings[-1]
        arrow_len = 0.1*2
        dx = arrow_len * np.sin(current_angle)
        dy = arrow_len * np.cos(current_angle)
        ax.arrow(history_x[-1], history_y[-1], dx, dy, head_width=1.4, fc=arrow_color, ec=arrow_color)

        ax.set_aspect('equal')
        ax.set_facecolor('#f5f5f5')
        ax.grid(color='#e0e0e0', linestyle='--')
        ax.set_xlabel('Y (m)')
        ax.set_ylabel('X (m)')
        ax.set_xlim([xlim_max, xlim_min])
        ax.set_ylim([-5, ylim_max])
        ax.set_title('BEV Vehicle Trajectory')
        ax.legend()
        plt.savefig(buffer_,format = 'png')
        buffer_.seek(0)
        traj_img = Image.open(buffer_)
        traj_img = np.asarray(traj_img)[:,:,:3]
        buffer_.close()
        plt.close()
        traj_img_resize_width = int((width*3)/2)

        traj_img_resize_height = int(height*2)

        traj_img_resize = cv2.resize(traj_img,(traj_img_resize_width,traj_img_resize_height))
        traj_img_resize_norm = (traj_img_resize/255)
        traj_img_tensor = torch.from_numpy(traj_img_resize_norm).permute(2,0,1)
        all_traj_img.append(traj_img_tensor)
    
    all_traj_img = torch.stack(all_traj_img,dim=1)
    
    if vis_rewards:
        all_reward_img = []
        history_reward_x = []
        history_reward_y = []
        max_index = 100
        if traj_t > 100:
            max_index = traj_t+10
        for traj_idx in range(traj_t):
            buffer_ = BytesIO()
            curr_reward = rewards_list[traj_idx].item()*100
            history_reward_x.append(traj_idx)
            history_reward_y.append(curr_reward)
            fig, ax = plt.subplots()
            ax.plot(
                history_reward_x,
                history_reward_y,
                "#2E8B57",  # 换个颜色（海绿色），区别于碰撞 reward 的橙色
                linewidth=2.0,
                alpha=0.7,
                label="Total Reward",
                marker='o',
                linestyle='-',
                markersize=2
            )

            ax.set_aspect('equal')
            ax.set_facecolor('#f5f5f5')
            ax.grid(color='#e0e0e0', linestyle='--')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Total Reward Value x 100')
            ax.set_xlim([0, max_index])  # 与 collision reward 保持一致
            ax.set_ylim([0.0, 100])
            ax.set_title('Total Reward')
            ax.legend()

            plt.savefig(buffer_, format='png')
            buffer_.seek(0)
            reward_img = Image.open(buffer_)
            reward_img = np.asarray(reward_img)[:, :, :3]
            buffer_.close()
            plt.close()
            reward_img_resize_width = int((width*3)/2)
            reward_img_resize_height = int(height*2)

            reward_img_resize = cv2.resize(reward_img,(reward_img_resize_width,reward_img_resize_height))
            reward_img_resize_norm = (reward_img_resize/255)
            reward_img_tensor = torch.from_numpy(reward_img_resize_norm).permute(2,0,1)
            all_reward_img.append(reward_img_tensor)
        all_reward_img = torch.stack(all_reward_img,dim=1)

        all_traj_img = torch.cat([all_traj_img,all_reward_img],dim=3)

    return all_traj_img

def _align_panel(panel, ref):
    """Align panel to ref shape [C, T, H, W]."""
    if panel is None:
        return torch.zeros_like(ref)

    # Channels
    if panel.shape[0] == 1:
        panel = panel.repeat(3, 1, 1, 1)
    elif panel.shape[0] > 3:
        panel = panel[:3]
    elif panel.shape[0] < 3:
        panel = torch.cat([panel, panel[:1].repeat(3 - panel.shape[0], 1, 1, 1)], dim=0)

    # Time
    if panel.shape[1] > ref.shape[1]:
        panel = panel[:, : ref.shape[1]]
    elif panel.shape[1] < ref.shape[1]:
        pad_t = ref.shape[1] - panel.shape[1]
        panel = torch.cat([panel, panel[:, -1:].repeat(1, pad_t, 1, 1)], dim=1)

    # Spatial
    if panel.shape[2] != ref.shape[2] or panel.shape[3] != ref.shape[3]:
        panel = F.interpolate(
            panel.permute(1, 0, 2, 3),
            size=(ref.shape[2], ref.shape[3]),
            mode="bilinear",
            align_corners=False,
        ).permute(1, 0, 2, 3)
    return panel


def conbind_video(
    pred,
    gt=None,
    seg=None,
    depth=None,
    occ_video=None,
    traj=None,
):
    """Fixed 3x2 layout:
    Row1: pred | gt
    Row2: seg  | depth
    Row3: occ  | traj
    """
    gt = _align_panel(gt, pred)
    seg = _align_panel(seg, pred)
    occ_video = _align_panel(occ_video, pred)
    depth = _align_panel(depth, pred)
    traj = _align_panel(traj, pred)
    row1 = torch.cat([pred, gt], dim=3)
    row2 = torch.cat([seg, depth], dim=3)
    row3 = torch.cat([occ_video, traj], dim=3)
    return torch.cat([row1, row2, row3], dim=2)


def process_occ_res(occ_output_list,voxel_size=0.2,width=1400,height=1400):
    if occ_output_list is None:
        return None
    occ_img_list = []
    # torch.Size([3, 5, 448, 1200])
    for occ_res in occ_output_list:
        occ_img = occ_draw(
            occ_res,
            width=width,
            height=height
        )

        occ_img_list.append(occ_img)

    occ_video = torch.stack(occ_img_list,dim=1)
    return occ_video

def process_multi_modal(
    cfg,x,batch,dataset,devices
):
    rgb_video = x['rgb_video']
    depth_video = x['depth_video']
    seg_video = x['seg_video']
    rgb_denorm = concat_6_views_pt(normalize(rgb_video.clone()))
    single_modal_video_shape = rgb_denorm.shape

    occ_output_list = x.get('occ_output_list',None)
    occ_video = process_occ_res(
        occ_output_list=occ_output_list,
        voxel_size=cfg.get('voxel_size',0.2),
        width=single_modal_video_shape[-1],
        height=single_modal_video_shape[-2],
    )


    depth_map = None
    seg_map = None
    depth_vis_cat = None

    if depth_video is not None:
        depth_map = process_depth(
            depth_video,
            dataset.max_depth
        )
        # Convert depth to [0,1] for visualization and keep camera layout consistent.
        depth_vis = (depth_map / max(dataset.max_depth, 1e-6)).clamp(0.0, 1.0)
        depth_vis_cat = concat_6_views_pt(depth_vis)
    else:
        depth_map = None
        depth_vis_cat = None

    gt_video = batch.get("video", None)
    if isinstance(gt_video, torch.Tensor):
        if gt_video.ndim == 6:
            gt_video = gt_video[0]
        gt_denorm = concat_6_views_pt(normalize(gt_video.clone()))
    else:
        gt_denorm = None


    if seg_video is not None:
        seg_vis, seg_map = process_seg(
            seg_video,
            dataset.seg_color_map,
            devices=devices
        )
        seg_vis_cat = concat_6_views_pt(seg_vis)
    else:
        seg_vis_cat = None

    # Calculate True Safety-Aware Rewards (Collision + Road Bound + Proprioceptive)
    rewards_list = calculate_rewards(
        cfg = cfg,
        depth_map = depth_map,
        seg_map = seg_map,
        occ_map = occ_output_list, 
        batch=batch,
    )

    traj_vis = vis_input_traj(
        batch=batch,
        rewards_list=rewards_list,
        height=max(single_modal_video_shape[-2] // 2, 1),
        width=max(int(single_modal_video_shape[-1] * 2 / 3), 1),
    )

    gen_vis = conbind_video(
        pred=rgb_denorm,
        gt=gt_denorm,
        seg=seg_vis_cat,
        depth=depth_vis_cat,
        occ_video=occ_video,
        traj=traj_vis,
    )
    return gen_vis, depth_map, seg_map, occ_output_list, rewards_list

def get_obs_and_roadbound(
    cfg,
    batch,
    depth_pred,
    semantic_pred
):
    all_obstacle_pcs = []
    all_dynamic_pcs = []
    all_road_edge_maps = []

    c2w = batch['ori_cam2ego_all'][0]
    cam_k = batch['cam_k'][0]
    num_view = len(cfg.get("view_order", []))

    # Obstacle: everything except road(1) and sky(2)
    obstacle_mask = (semantic_pred != 1) & (semantic_pred != 2)
    dynamic_mask = torch.isin(semantic_pred, torch.tensor([3, 4, 5], device=semantic_pred.device))  # car, person, bike
    road_mask = (semantic_pred == 1)

    _, num_frames, _, _ = depth_pred.shape

    for t in range(num_frames):
        view_c2w_t = c2w[:, t]
        view_cam_k_t = cam_k[:, t]
        depth_t = depth_pred[:, t].cpu().numpy()
        seg_t = semantic_pred[:, t].cpu().numpy()

        obstacle_depth_t = depth_t.copy()
        obstacle_depth_t[~obstacle_mask[:, t].cpu().numpy()] = 0
        dynamic_depth_t = depth_t.copy()
        dynamic_depth_t[~dynamic_mask[:, t].cpu().numpy()] = 0

        obs_pts = []
        dyn_pts = []

        for v in range(min(3, num_view)):  # front 3 views
            K = view_cam_k_t[v].numpy()
            c2w_v = view_c2w_t[v].numpy()
            d_obs = obstacle_depth_t[v]
            d_dyn = dynamic_depth_t[v]

            # Obstacle points
            y_obs, x_obs = np.nonzero(d_obs > 0)
            if len(x_obs) > 0:
                pts_cam = np.linalg.inv(K[:3,:3]) @ np.stack([x_obs, y_obs, np.ones_like(x_obs)], axis=0)
                pts_cam *= d_obs[y_obs, x_obs]
                pts_world = (c2w_v @ np.concatenate([pts_cam, np.ones((1, len(x_obs)))], axis=0))[:3].T
                obs_pts.append(pts_world)

            # Dynamic points
            y_dyn, x_dyn = np.nonzero(d_dyn > 0)
            if len(x_dyn) > 0:
                pts_cam = np.linalg.inv(K[:3,:3]) @ np.stack([x_dyn, y_dyn, np.ones_like(x_dyn)], axis=0)
                pts_cam *= d_dyn[y_dyn, x_dyn]
                pts_world = (c2w_v @ np.concatenate([pts_cam, np.ones((1, len(x_dyn)))], axis=0))[:3].T
                dyn_pts.append(pts_world)

        all_obstacle_pcs.append(np.concatenate(obs_pts, axis=0) if obs_pts else np.empty((0,3)))
        all_dynamic_pcs.append(np.concatenate(dyn_pts, axis=0) if dyn_pts else np.empty((0,3)))

        # Road edge (front view only)
        # Use view index 1 = CAM_FRONT
        road_front = road_mask[1, t].cpu().numpy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        eroded = cv2.erode(road_front.astype(np.uint8), kernel)
        edge_map = road_front & (~eroded.astype(bool))
        all_road_edge_maps.append(edge_map)
    
    return all_obstacle_pcs, all_dynamic_pcs, all_road_edge_maps


def calculate_rewards(
    cfg,depth_map,seg_map,occ_map,batch
):
    if not cfg.get('calculate_reward', False):
        return []
    
    total_reward_list = []
    collision_penalty_reward_list = []
    roadbound_reward_list = []
    proprioceptive_reward_list = []

    depth_pred = depth_map[:,0]
    depth_pred[depth_pred >= 25] = -1.0
    depth_pred[depth_pred == 0] = -1.0

    # ===== 2. Back-project point clouds =====
    all_obstacle_pcs, all_dynamic_pcs, all_road_edge_maps = get_obs_and_roadbound(cfg, batch, depth_pred, seg_map)

    # ===== 3. Compute rewards =====
    ego_l_half = 2.0  # ±2m in x
    ego_w_half = 1.0  # ±1m in y
    road_buffer = 1.0  # meters

    # FIX: Use FIXED cam2ego for front camera (t=0)
    # Shape of batch['cam2ego']: [B, N_views, T_actual, 4, 4]
    # But camera extrinsics are constant → use t=0
    # T_actual = batch['cam2ego'].shape[2]
    cam2ego_front_fixed = batch['cam2ego'][0, 1, 0].cpu().numpy()  # always t=0
    cam_k_front_fixed = batch['cam_k'][0, 1, 0].cpu().numpy()      # also fixed
    _, num_frames, _, _ = depth_pred.shape

    for t in range(num_frames):
        # -- Ego pose --
        # traj_pt = vis_vla_traj[-1][t]
        traj_gt = batch['traj_gt'][0][t]
        x, y, theta = traj_gt[0,0].item(), traj_gt[1,0].item(), traj_gt[2,1].item()
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        corners_local = np.array([[ ego_l_half,  ego_w_half],
                                [ ego_l_half, -ego_w_half],
                                [-ego_l_half, -ego_w_half],
                                [-ego_l_half,  ego_w_half]])
        corners_world = (R @ corners_local.T).T + np.array([x, y])
        ego_x_min, ego_x_max = corners_world[:,0].min(), corners_world[:,0].max()
        ego_y_min, ego_y_max = corners_world[:,1].min(), corners_world[:,1].max()
        ego_center = np.array([x, y])

        # -- (A) Collision Penalty --
        obs_pc = all_obstacle_pcs[t]
        dyn_pc = all_dynamic_pcs[t]
        coll_cost = 0.0

        if obs_pc.size > 0:
            inside = (obs_pc[:,0] >= ego_x_min) & (obs_pc[:,0] <= ego_x_max) & \
                    (obs_pc[:,1] >= ego_y_min) & (obs_pc[:,1] <= ego_y_max)
            if inside.any():
                coll_cost += 1.0
            else:
                d_min = np.linalg.norm(obs_pc[:,:2] - ego_center, axis=1).min()
                if d_min < 3.0:
                    coll_cost += 0.3 * (3.0 - d_min) / 3.0

        if dyn_pc.size > 0:
            inside_dyn = (dyn_pc[:,0] >= ego_x_min) & (dyn_pc[:,0] <= ego_x_max) & \
                        (dyn_pc[:,1] >= ego_y_min) & (dyn_pc[:,1] <= ego_y_max)
            if inside_dyn.any():
                coll_cost += 2.0
            else:
                d_min_dyn = np.linalg.norm(dyn_pc[:,:2] - ego_center, axis=1).min()
                if d_min_dyn < 5.0:
                    coll_cost += 0.6 * (5.0 - d_min_dyn) / 5.0

        coll_reward = np.exp(-2.0 * min(coll_cost / 3.0, 1.0))
        collision_penalty_reward_list.append(coll_reward)

        # -- (B) Road Bound Penalty --
        # Use FIXED extrinsics (camera rigidly mounted)

        ego2cam = np.linalg.inv(cam2ego_front_fixed)
        point_ego = np.array([x, y, 0.0, 1.0])
        point_cam = ego2cam @ point_ego
        # TODO
        if point_cam[2] > 0:
            uv = cam_k_front_fixed @ point_cam[:3]
            u, v = int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))
            H, W = depth_pred.shape[2], depth_pred.shape[3]  # ✅ (N, T, H, W)
            if 0 <= v < H and 0 <= u < W:
                edge_pixels = np.argwhere(all_road_edge_maps[t])
                if edge_pixels.size > 0:
                    dist_px = np.min(np.sqrt((edge_pixels[:,0] - v)**2 + (edge_pixels[:,1] - u)**2))
                    depth_val = depth_pred[1, t, v, u].item()
                    focal = cam_k_front_fixed[0, 0]
                    px_to_m = depth_val / focal if focal > 0 and depth_val > 0 else 0.1
                    dist_m = dist_px * px_to_m
                else:
                    dist_m = float('inf')
            else:
                dist_m = 0.0

        else:
            dist_m = 0.0

        if dist_m < road_buffer:
            road_cost = (road_buffer - dist_m) / road_buffer
            road_r = np.exp(-3.0 * road_cost)
        else:
            road_r = 1.0
        roadbound_reward_list.append(road_r)

        # -- (C) Proprioceptive Reward TODO
        proprio_r = 1.0

        # -- (D) Total Reward --
        w_coll = 0.5
        w_road = 0.5
        w_prop = 0.2
        total_r = w_coll * coll_reward + w_road * road_r #+ w_prop * proprio_r
        total_reward_list.append(total_r)

    return total_reward_list

def process_and_save(
    x: dict,
    batch: dict,
    cfg: dict,
    sub_dir: str,
    generate_sampling_option: dict,
    epoch: int,
    start_index: int,
    dataset: torch.utils.data.Dataset,
    devices = None
):
    """
    Process the generated samples and save them to disk.
    """
    fps_save = cfg.get("fps_save", 10)
    save_dir = cfg.save_dir
    save_path = get_save_path_name(
        save_dir,
        sub_dir,
        save_prefix=cfg.get("save_prefix", ""),
        index=start_index,
        num_sample_pos=epoch,
    )
    video,depth_map,seg_map,occ_map,rewards_list = process_multi_modal(
        cfg=cfg,
        x=x,
        batch=batch,
        dataset=dataset,
        devices = devices
    )
    
    # save_sample mutates the input tensor in-place; keep frame export on an untouched copy.
    video_for_mp4 = video.clone()
    save_sample(video_for_mp4, save_path=save_path, fps=fps_save,normalize=False)
    if cfg.get("save_frames", True):
        # Save the same stitched content as frame-wise PNGs.
        frame_dir = f"{save_path}_frames"
        os.makedirs(frame_dir, exist_ok=True)
        video_u8 = (
            video.clone()
            .mul(255)
            .add(0.5)
            .clamp(0, 255)
            .permute(1, 2, 3, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        for frame_idx, frame in enumerate(video_u8):
            Image.fromarray(frame).save(
                os.path.join(frame_dir, f"frame_{frame_idx:04d}.png")
            )

    if occ_map is not None:
        occ_points = [int(len(frame_occ)) for frame_occ in occ_map]
        occ_stats = {
            "num_frames": int(len(occ_points)),
            "non_empty_frames": int(sum(p > 0 for p in occ_points)),
            "points_per_frame": occ_points,
            "points_min": int(min(occ_points) if occ_points else 0),
            "points_max": int(max(occ_points) if occ_points else 0),
            "points_mean": float(sum(occ_points) / len(occ_points)) if occ_points else 0.0,
        }
        with open(f"{save_path}_occ_stats.json", "w", encoding="utf-8") as f:
            json.dump(occ_stats, f, ensure_ascii=False, indent=2)


def prepare_inference_condition(
    z: torch.Tensor,
    batch: dict = None,
    causal: bool = True,
    model_ae: nn.Module = None
) -> torch.Tensor:
    """
    Prepare the visual condition for the model, using causal vae.

    Args:
        z (torch.Tensor): The latent noise tensor, of shape [B, C, T, H, W]
        mask_cond (dict): The condition configuration.
        ref_list: list of lists of media (image/video) for i2v and v2v condition, of shape [C, T', H, W]; len(ref_list)==B; ref_list[i] is the list of media for the generation in batch idx i, we use a list of media for each batch item so that it can have multiple references. For example, ref_list[i] could be [ref_image_1, ref_image_2] for i2v_loop condition.

    Returns:
        torch.Tensor: The visual condition tensor.
    """
    # x has shape [b, c, t, h, w], where b is the batch size
    device = next(model_ae.parameters()).device
    dtype = next(model_ae.parameters()).dtype
    B, _, T, H, W = z.shape
    C = model_ae.z_channels
    masks = torch.zeros(B, 1, T, H, W).to(device,dtype)
    masked_z = torch.zeros(B, C, T, H, W).to(device,dtype)
    for i in range(B):
        masks[i, :, 0, :, :] = 1
        masked_z[i, :, :1, :, :] = model_ae.encode(batch['video'][i, :, :1, :, :].unsqueeze(0)).squeeze(0)
    return masks, masked_z
