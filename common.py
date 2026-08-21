# common.py
# =============================================================================
# 共享模块：模型定义、数据工具、评估函数、数据增强
# =============================================================================

import os
import sys
import json
import math
import copy
import random
from datetime import datetime
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from mmfi_lib.mmfi import (
    MMFi_Database,
    MMFi_Dataset,
    make_dataset,
    make_dataloader,
)
from mmfi_lib.evaluate import calulate_error

# ----------------------------- 随机种子 -----------------------------
seed = int(os.environ.get("TRAIN_SEED", 0))
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# ----------------------------- Pose Normalization -----------------------------
def normalize_pose2d(pose2d):
    root = (pose2d[..., 11:12, :] + pose2d[..., 12:13, :]) / 2
    return pose2d - root


def normalize_pose3d(pose3d):
    root = (pose3d[:, 11:12, :] + pose3d[:, 12:13, :]) / 2
    return pose3d - root


def pose_root3d(pose3d):
    return (pose3d[:, 11:12, :] + pose3d[:, 12:13, :]) / 2


def root_relative_pose_loss(pred3d, gt3d):
    return F.l1_loss(pred3d - pose_root3d(pred3d), gt3d - pose_root3d(gt3d))


def root_position_loss(pred3d, gt3d):
    return F.l1_loss(pose_root3d(pred3d), pose_root3d(gt3d))


COCO17_BONES = (
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
)


def bone_length_loss(pred3d, gt3d, bones=COCO17_BONES):
    losses = []
    for i, j in bones:
        pred_len = torch.norm(pred3d[:, i, :] - pred3d[:, j, :], dim=-1)
        gt_len = torch.norm(gt3d[:, i, :] - gt3d[:, j, :], dim=-1)
        losses.append(F.l1_loss(pred_len, gt_len))
    return torch.stack(losses).mean()


def subject_robust_l1_loss(pred3d, gt3d, subjects, robust_weight=0.0):
    per_sample = torch.abs(pred3d - gt3d).mean(dim=(1, 2))
    mean_loss = per_sample.mean()
    if robust_weight <= 0 or not subjects:
        return mean_loss

    group_losses = []
    for subject in sorted(set(subjects)):
        idx = [i for i, s in enumerate(subjects) if s == subject]
        if idx:
            group_losses.append(per_sample[idx].mean())

    if not group_losses:
        return mean_loss
    worst_group = torch.stack(group_losses).max()
    return (1.0 - robust_weight) * mean_loss + robust_weight * worst_group


def denormalize_pose3d(pose3d_normalized, root):
    return pose3d_normalized + root


# ----------------------------- WiFi Data Augmentation -----------------------------
def augment_wifi(wifi, noise_std=0.05, freq_mask_prob=0.1, time_mask_prob=0.1):
    if noise_std > 0:
        wifi = wifi + torch.randn_like(wifi) * noise_std
    if freq_mask_prob > 0 and wifi.dim() >= 4:
        freq_dim = wifi.shape[-2]
        freq_mask = (torch.rand(freq_dim, device=wifi.device) > freq_mask_prob).float()
        if wifi.dim() == 5:
            freq_mask = freq_mask.view(1, 1, 1, -1, 1)
        else:
            freq_mask = freq_mask.view(1, 1, -1, 1)
        wifi = wifi * freq_mask
    if time_mask_prob > 0 and wifi.dim() == 5:
        T = wifi.shape[1]
        time_mask = (torch.rand(T, device=wifi.device) > time_mask_prob).float()
        time_mask[-1] = 1.0
        time_mask = time_mask.view(1, -1, 1, 1, 1)
        wifi = wifi * time_mask
    return wifi.clamp(0, 1)


def mixup_data(wifi, rgb, gt, alpha=0.4):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    B   = wifi.size(0)
    idx = torch.randperm(B, device=wifi.device)
    mixed_wifi = lam * wifi + (1 - lam) * wifi[idx]
    mixed_rgb  = lam * rgb  + (1 - lam) * rgb[idx]
    mixed_gt   = lam * gt   + (1 - lam) * gt[idx]
    return mixed_wifi, mixed_rgb, mixed_gt, lam, idx


# ----------------------------- Logger -----------------------------
class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        parent = os.path.dirname(log_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.log_file = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ----------------------------- IO utils -----------------------------
def make_run_dir(mode: str) -> str:
    root = "strict_offline_runs"
    os.makedirs(root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, f"{mode}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(obj, path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fmt_metrics(mpjpe, pa, pck):
    return (f"MPJPE={mpjpe*1000:.1f}mm | PA-MPJPE={pa*1000:.1f}mm | "
            f"PCK@20={pck.get('pck@20', 0):.1f}% | PCK@50={pck.get('pck@50', 0):.1f}%")


# ----------------------------- Temporal Window Dataset Wrapper -----------------------------
class TemporalWindowWrapper(torch.utils.data.Dataset):
    def __init__(self, base_ds, window: int = 16, stride: int = 1,
                 require_modalities: Optional[List[str]] = None,
                 include_rgb_window: bool = False):
        self.base_ds = base_ds
        self.window  = int(window)
        self.stride  = int(stride)
        self.require_modalities = require_modalities
        self.include_rgb_window = bool(include_rgb_window)
        self.indices = []

        for i in range(len(base_ds)):
            item = base_ds.data_list[i]
            if "idx" not in item:
                continue
            if item["idx"] >= (self.window - 1) * self.stride:
                ok = True
                if require_modalities is not None:
                    for m in require_modalities:
                        key = m + "_path"
                        if key in item and not os.path.isfile(item[key]):
                            ok = False
                            break
                if ok:
                    self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def _build_frame_path(self, current_path: str, frame_idx0: int) -> str:
        dir_path = os.path.dirname(current_path)
        ext      = os.path.splitext(current_path)[1]
        return os.path.join(dir_path, f"frame{frame_idx0 + 1:03d}{ext}")

    def __getitem__(self, j):
        base_i = self.indices[j]
        item   = self.base_ds.data_list[base_i]
        sample = self.base_ds[base_i]

        if "input_wifi-csi" not in sample:
            raise ValueError("TemporalWindowWrapper expects 'wifi-csi' in modality.")

        wifi_path_cur = item["wifi-csi_path"]
        idx_cur       = item["idx"]

        wifi_frames = []
        rgb_frames = []
        for t in range(self.window):
            idx_t  = idx_cur - (self.window - 1 - t) * self.stride
            path_t = self._build_frame_path(wifi_path_cur, idx_t)
            if not os.path.isfile(path_t):
                raise FileNotFoundError(path_t)
            wifi_t = self.base_ds.read_frame(path_t)
            wifi_frames.append(np.array(wifi_t))
            if self.include_rgb_window:
                rgb_path_cur = item.get("rgb_path")
                if rgb_path_cur is None:
                    raise ValueError("include_rgb_window=True requires rgb modality.")
                rgb_path_t = self._build_frame_path(rgb_path_cur, idx_t)
                if not os.path.isfile(rgb_path_t):
                    raise FileNotFoundError(rgb_path_t)
                rgb_t = self.base_ds.read_frame(rgb_path_t)
                rgb_frames.append(np.array(rgb_t))

        wifi_arr = np.stack(wifi_frames, axis=0)
        sample["input_wifi-csi_window"] = torch.FloatTensor(wifi_arr)
        if self.include_rgb_window:
            sample["input_rgb_window"] = torch.FloatTensor(np.stack(rgb_frames, axis=0))
        sample["idx_window_end"]        = idx_cur
        return sample


def split_heldout_windows(heldout_ds, cfg):
    """Split the held-out window dataset into validation/test subsets.

    This follows the HPE-Li reference protocol: MMFi's original validation
    partition is treated as a held-out pool, then split 1:1 by default.
    Set heldout_split.unit="sequence" to keep all windows from the same
    subject-action sequence on the same side of the split.
    """
    split_cfg = cfg.get("heldout_split", {})
    unit = str(split_cfg.get("unit", "window")).lower()
    test_size = float(split_cfg.get("test_size", 0.5))
    random_seed = int(split_cfg.get("random_seed", 41))

    if not 0.0 < test_size < 1.0:
        raise ValueError("heldout_split.test_size must be between 0 and 1.")
    if len(heldout_ds) < 2:
        raise ValueError("Need at least two held-out samples to split validation/test.")

    rng = np.random.default_rng(random_seed)

    if unit in ("window", "sample"):
        indices = np.arange(len(heldout_ds))
        rng.shuffle(indices)
    elif unit in ("sequence", "subject_action"):
        groups = {}
        for window_j, base_i in enumerate(heldout_ds.indices):
            item = heldout_ds.base_ds.data_list[base_i]
            key = (item.get("scene"), item.get("subject"), item.get("action"))
            groups.setdefault(key, []).append(window_j)
        if len(groups) < 2:
            raise ValueError("Need at least two held-out sequences to split validation/test.")
        group_keys = np.array(list(groups.keys()), dtype=object)
        rng.shuffle(group_keys)

        n_test_groups = int(math.ceil(len(group_keys) * test_size))
        n_test_groups = min(max(n_test_groups, 1), len(group_keys) - 1)
        test_keys = set(map(tuple, group_keys[:n_test_groups].tolist()))
        test_indices = []
        val_indices = []
        for key, group_indices in groups.items():
            if key in test_keys:
                test_indices.extend(group_indices)
            else:
                val_indices.extend(group_indices)
        return (
            torch.utils.data.Subset(heldout_ds, sorted(val_indices)),
            torch.utils.data.Subset(heldout_ds, sorted(test_indices)),
        )
    elif unit == "subject":
        groups = {}
        for window_j, base_i in enumerate(heldout_ds.indices):
            item = heldout_ds.base_ds.data_list[base_i]
            key = item.get("subject")
            groups.setdefault(key, []).append(window_j)
        if len(groups) < 2:
            raise ValueError("Need at least two held-out subjects to split validation/test.")
        group_keys = np.array(list(groups.keys()), dtype=object)
        rng.shuffle(group_keys)

        n_test_groups = int(math.ceil(len(group_keys) * test_size))
        n_test_groups = min(max(n_test_groups, 1), len(group_keys) - 1)
        test_keys = set(group_keys[:n_test_groups].tolist())
        test_indices = []
        val_indices = []
        for key, group_indices in groups.items():
            if key in test_keys:
                test_indices.extend(group_indices)
            else:
                val_indices.extend(group_indices)
        return (
            torch.utils.data.Subset(heldout_ds, sorted(val_indices)),
            torch.utils.data.Subset(heldout_ds, sorted(test_indices)),
        )
    else:
        raise ValueError(
            "heldout_split.unit must be one of: window, sequence, subject.")

    n_test = int(math.ceil(len(indices) * test_size))
    n_test = min(max(n_test, 1), len(indices) - 1)
    test_indices = indices[:n_test].tolist()
    val_indices = indices[n_test:].tolist()

    val_ds = torch.utils.data.Subset(heldout_ds, val_indices)
    test_ds = torch.utils.data.Subset(heldout_ds, test_indices)
    return val_ds, test_ds


def heldout_split_enabled(cfg):
    return bool(cfg.get("heldout_split", {}).get("enabled", False))


def get_cross_subject_val_test_subjects(cfg):
    split_cfg = cfg.get("cross_subject_split", {})
    heldout_subjects = list(split_cfg.get("val_dataset", {}).get("subjects") or [])
    heldout_cfg = cfg.get("heldout_split", {})
    val_subjects = list(heldout_cfg.get("val_subjects") or [])
    test_subjects = list(heldout_cfg.get("test_subjects") or [])

    if not val_subjects or not test_subjects:
        raise ValueError(
            "cross_subject_split with heldout_split.unit='subject' requires "
            "heldout_split.val_subjects and heldout_split.test_subjects.")

    overlap = sorted(set(val_subjects) & set(test_subjects))
    if overlap:
        raise ValueError(f"Val/test subjects overlap: {overlap}")

    missing = sorted((set(val_subjects) | set(test_subjects)) - set(heldout_subjects))
    if missing:
        raise ValueError(
            f"Val/test subjects must be drawn from cross_subject_split.val_dataset.subjects. "
            f"Unexpected subjects: {missing}")

    unused = sorted(set(heldout_subjects) - (set(val_subjects) | set(test_subjects)))
    if unused:
        raise ValueError(f"Held-out subjects not assigned to val/test: {unused}")

    return val_subjects, test_subjects


def make_cross_subject_heldout_dataset(dataset_root, cfg, subjects):
    cfg_subject = copy.deepcopy(cfg)
    cfg_subject["split_to_use"] = "cross_subject_split"
    cfg_subject["cross_subject_split"]["val_dataset"]["subjects"] = list(subjects)
    _, heldout_ds_base = make_dataset(dataset_root, cfg_subject)
    return heldout_ds_base


def collate_temporal(batch):
    out = {}
    out["modality"]       = batch[0]["modality"]
    out["scene"]          = [b["scene"]  for b in batch]
    out["subject"]        = [b["subject"] for b in batch]
    out["action"]         = [b["action"]  for b in batch]
    out["idx"]            = [b.get("idx", None) for b in batch]
    out["idx_window_end"] = [b.get("idx_window_end", None) for b in batch]
    out["output"]         = torch.stack([torch.as_tensor(b["output"]).float() for b in batch], dim=0)
    out["input_wifi-csi_window"] = torch.stack([b["input_wifi-csi_window"] for b in batch], dim=0)
    if "input_rgb" in batch[0]:
        out["input_rgb"] = torch.stack([torch.as_tensor(b["input_rgb"]).float() for b in batch], dim=0)
    if "input_rgb_window" in batch[0]:
        out["input_rgb_window"] = torch.stack([b["input_rgb_window"] for b in batch], dim=0)
    return out


# ----------------------------- CBAM Module -----------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )

    def forward(self, x):
        B, C, H, W = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(B, C))
        max_out = self.mlp(self.max_pool(x).view(B, C))
        return x * torch.sigmoid(avg_out + max_out).view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


# ----------------------------- WiFi Frame Encoder with CBAM -----------------------------
class WiFiFrameEncoderCBAM(nn.Module):
    def __init__(self, in_ch=3, d_model=512, use_cbam=True, dropout=0.3):
        super().__init__()
        self.use_cbam = use_cbam
        self.dropout  = nn.Dropout2d(dropout * 0.5)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        if use_cbam:
            self.cbam_stem = CBAM(32, reduction=8, spatial_kernel=5)

        self.layer1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        if use_cbam:
            self.cbam1 = CBAM(64, reduction=8, spatial_kernel=5)

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        if use_cbam:
            self.cbam2 = CBAM(128, reduction=16, spatial_kernel=5)

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(256, d_model), nn.LayerNorm(d_model), nn.Dropout(dropout))

    def forward(self, x):
        x = self.stem(x)
        if self.use_cbam: x = self.cbam_stem(x)
        x = self.dropout(x)
        x = self.layer1(x)
        if self.use_cbam: x = self.cbam1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        if self.use_cbam: x = self.cbam2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return self.fc(self.pool(x).flatten(1))


# ----------------------------- Temporal WiFi Encoder -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TemporalWiFiEncoderCBAM(nn.Module):
    def __init__(self, in_ch=3, d_model=512, window=16, use_cbam=True,
                 dropout=0.1, num_layers=2, dim_feedforward=1024):
        super().__init__()
        self.frame_encoder   = WiFiFrameEncoderCBAM(in_ch, d_model, use_cbam, dropout)
        self.temporal_pe     = PositionalEncoding(d_model, max_len=window + 16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu')
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.temporal_pool = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    def forward(self, x):
        B, T = x.shape[:2]
        if x.shape[-1] == 3:
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        x_flat      = x.view(B * T, x.shape[2], x.shape[3], x.shape[4])
        frame_feats = self.frame_encoder(x_flat).view(B, T, -1)
        frame_feats = self.temporal_pe(frame_feats)
        return self.temporal_pool(
            self.temporal_transformer(frame_feats)[:, -1, :])


# ----------------------------- RGB Encoder -----------------------------
class EnhancedRGBEncoder(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.joint_embed = nn.Linear(2, 64)
        self.pos_embed   = nn.Parameter(torch.zeros(1, 17, 64))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=4, dim_feedforward=256,
            dropout=dropout, batch_first=True, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Sequential(
            nn.Linear(64 * 17, 512), nn.ReLU(inplace=True),
            nn.Linear(512, d_model), nn.LayerNorm(d_model))

    def forward(self, rgb):
        x = self.joint_embed(rgb) + self.pos_embed
        return self.fc(self.transformer(x).flatten(1))


class TemporalRGBEncoder(nn.Module):
    def __init__(self, d_model=512, window=16, dropout=0.1,
                 num_layers=2, dim_feedforward=1024):
        super().__init__()
        self.frame_encoder = EnhancedRGBEncoder(d_model, dropout)
        self.temporal_pe = PositionalEncoding(d_model, max_len=window + 16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation='gelu')
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.temporal_pool = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

    def forward(self, rgb_window):
        if rgb_window.dim() == 3:
            return self.frame_encoder(rgb_window)
        B, T = rgb_window.shape[:2]
        frame_feats = self.frame_encoder(
            rgb_window.reshape(B * T, *rgb_window.shape[2:])
        ).view(B, T, -1)
        frame_feats = self.temporal_pe(frame_feats)
        return self.temporal_pool(
            self.temporal_transformer(frame_feats)[:, -1, :])


# ----------------------------- Pose Head -----------------------------
class PoseHead(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, hidden=512):
        super().__init__()
        hidden = int(hidden)
        mid = max(128, hidden // 2)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, mid),     nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(mid, 17 * 3))

    def forward(self, feat):
        return self.net(feat).view(-1, 17, 3)


class RootRelativePoseHead(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, hidden=512):
        super().__init__()
        hidden = int(hidden)
        mid = max(128, hidden // 2)
        self.rel_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, mid), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(mid, 17 * 3))
        self.root_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 3))

    def forward(self, feat):
        rel = self.rel_head(feat).view(-1, 17, 3)
        rel = rel - pose_root3d(rel)
        root = self.root_head(feat).view(-1, 1, 3)
        return rel + root


class GraphPoseHead(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, hidden=512, num_layers=2):
        super().__init__()
        hidden = int(hidden)
        self.joint_queries = nn.Parameter(torch.randn(17, hidden) * 0.02)
        self.feat_proj = nn.Linear(d_model, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.rel_out = nn.Linear(hidden, 3)
        self.root_head = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, 3))

        adj = torch.eye(17)
        for i, j in COCO17_BONES:
            adj[i, j] = 1.0
            adj[j, i] = 1.0
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("adj", adj)

    def forward(self, feat):
        B = feat.size(0)
        base = self.feat_proj(feat).unsqueeze(1)
        x = base + self.joint_queries.unsqueeze(0).expand(B, -1, -1)
        x = self.input_norm(x)
        for layer, norm in zip(self.layers, self.norms):
            msg = torch.einsum("ij,bjh->bih", self.adj, x)
            x = norm(x + layer(msg))
        rel = self.rel_out(x)
        rel = rel - pose_root3d(rel)
        root = self.root_head(feat).view(-1, 1, 3)
        return rel + root


def build_pose_head(head_type="mlp", d_model=512, dropout=0.1, hidden=512):
    head_type = (head_type or "mlp").lower()
    if head_type in ("mlp", "plain"):
        return PoseHead(d_model, dropout, hidden)
    if head_type in ("root", "root_relative"):
        return RootRelativePoseHead(d_model, dropout, hidden)
    if head_type in ("graph", "graph_root", "topology"):
        return GraphPoseHead(d_model, dropout, hidden)
    raise ValueError(f"Unknown pose_head_type: {head_type}")


# ----------------------------- Complete Models -----------------------------
class TemporalWiFiStudentCBAM(nn.Module):
    def __init__(self, in_ch=3, d_model=512, window=16, use_cbam=True,
                 dropout=0.1, pose_head_hidden=512,
                 num_layers=2, dim_feedforward=1024,
                 pose_head_type="mlp"):
        super().__init__()
        self.wifi_encoder = TemporalWiFiEncoderCBAM(
            in_ch, d_model, window, use_cbam, dropout,
            num_layers, dim_feedforward)
        self.pose_head = build_pose_head(
            pose_head_type, d_model, dropout, pose_head_hidden)

    def forward(self, wifi_window):
        feat = self.wifi_encoder(wifi_window)
        return self.pose_head(feat), feat


class EnhancedFusionTeacher(nn.Module):
    def __init__(self, in_ch=3, d_model=512, dropout=0.1):
        super().__init__()
        self.wifi_encoder = WiFiFrameEncoderCBAM(
            in_ch, d_model, use_cbam=False, dropout=dropout)
        self.rgb_encoder  = EnhancedRGBEncoder(d_model, dropout)
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 2))
        self.pose_head = PoseHead(d_model, dropout)

    def forward(self, wifi, rgb, return_feat=False):
        fw = self.wifi_encoder(wifi)
        fv = self.rgb_encoder(rgb)
        w  = F.softmax(self.gate(torch.cat([fw, fv], dim=-1)), dim=-1)
        fused = w[:, 0:1] * fw + w[:, 1:2] * fv
        pred  = self.pose_head(fused)
        return (pred, fused) if return_feat else pred
#new_thing
class RGBOnlyTeacher(nn.Module):
    """纯 RGB Teacher：接口尽量与 EnhancedFusionTeacher 保持一致。"""
    def __init__(self, d_model=512, dropout=0.1, temporal_rgb=False,
                 window=16, temporal_layers=2, dim_feedforward=1024):
        super().__init__()
        self.temporal_rgb = bool(temporal_rgb)
        if self.temporal_rgb:
            self.rgb_encoder = TemporalRGBEncoder(
                d_model=d_model, window=window, dropout=dropout,
                num_layers=temporal_layers,
                dim_feedforward=dim_feedforward)
        else:
            self.rgb_encoder = EnhancedRGBEncoder(d_model, dropout)
        self.pose_head = PoseHead(d_model, dropout)

    def forward(self, wifi, rgb, return_feat=False):
        del wifi
        feat = self.rgb_encoder(rgb)
        pred = self.pose_head(feat)
        return (pred, feat) if return_feat else pred

class TemporalLUPICMC_CBAM(nn.Module):
    def __init__(self, in_ch=3, d_model=512, d_proj=128, window=16,
                 use_cbam=True, dropout=0.1,
                 num_layers=2, dim_feedforward=1024,
                 pose_head_hidden=512,
                 pose_head_type="mlp"):
        super().__init__()
        self.wifi_encoder = TemporalWiFiEncoderCBAM(
            in_ch, d_model, window, use_cbam, dropout,
            num_layers, dim_feedforward)
        self.pose_head = build_pose_head(
            pose_head_type, d_model, dropout, pose_head_hidden)
        self.proj_w = nn.Sequential(
            nn.Linear(d_model, d_proj), nn.ReLU(inplace=True),
            nn.Linear(d_proj, d_proj))
        self.proj_v = nn.Sequential(
            nn.Linear(d_model, d_proj), nn.ReLU(inplace=True),
            nn.Linear(d_proj, d_proj))

    def forward_wifi(self, wifi_window):
        feat = self.wifi_encoder(wifi_window)
        return self.pose_head(feat), feat

    def forward(self, wifi_window):
        return self.forward_wifi(wifi_window)


# ----------------------------- InfoNCE Loss -----------------------------
def info_nce_loss(z_w, z_v, tau=0.1):
    z_w = F.normalize(z_w, dim=-1)
    z_v = F.normalize(z_v, dim=-1)
    B   = z_w.size(0)
    logits = torch.mm(z_w, z_v.t()) / tau
    labels = torch.arange(B, device=z_w.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)) / 2


# ----------------------------- Evaluation Functions -----------------------------
def eval_temporal_model(model, val_loader, device, model_type="student",
                        use_root_relative=False):
    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            wifi_window = batch["input_wifi-csi_window"].to(device)
            gt          = batch["output"].to(device)
            out         = model(wifi_window)
            pred3d      = out[0] if isinstance(out, tuple) else out

            gt_np   = gt.cpu().numpy()
            pred_np = pred3d.cpu().numpy()

            if use_root_relative:
                root = ((gt[:, 11:12, :] + gt[:, 12:13, :]) / 2).cpu().numpy()
                preds_all.append(pred_np + root)
            else:
                preds_all.append(pred_np)

            gts_all.append(gt_np)

    preds_all = np.concatenate(preds_all, axis=0)
    gts_all   = np.concatenate(gts_all,   axis=0)
    mpjpe, pa, pck = calulate_error(preds_all, gts_all)
    return mpjpe, pa, pck


def eval_teacher_model(model, val_loader, device, use_root_relative=False):
    model.eval()
    preds_all, gts_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            wifi_window = batch["input_wifi-csi_window"].to(device)
            wifi_cur    = wifi_window[:, -1, :, :, :]
            if getattr(model, "temporal_rgb", False) and "input_rgb_window" in batch:
                rgb = batch["input_rgb_window"].to(device)
            else:
                rgb = batch["input_rgb"].to(device)
            gt          = batch["output"].to(device)
            #if use_root_relative:
            #    rgb = normalize_pose2d(rgb)
            #pred3d = model(wifi_cur, rgb)

            #gt_np   = gt.cpu().numpy()
            #pred_np = pred3d.cpu().numpy()
            if use_root_relative:
                rgb = normalize_pose2d(rgb)

            out = model(wifi_cur, rgb)
            pred3d = out[0] if isinstance(out, tuple) else out

            gt_np = gt.cpu().numpy()
            pred_np = pred3d.cpu().numpy()

            if use_root_relative:
                root = ((gt[:, 11:12, :] + gt[:, 12:13, :]) / 2).cpu().numpy()
                preds_all.append(pred_np + root)
            else:
                preds_all.append(pred_np)

            gts_all.append(gt_np)

    preds_all = np.concatenate(preds_all, axis=0)
    gts_all   = np.concatenate(gts_all,   axis=0)
    mpjpe, pa, pck = calulate_error(preds_all, gts_all)
    return mpjpe, pa, pck


# ----------------------------- 数据加载工具 -----------------------------
def _protocol_actions(config):
    all_actions = [
        'A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09',
        'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18',
        'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27']
    if config['protocol'] == 'protocol1':
        return [
            'A02', 'A03', 'A04', 'A05', 'A13', 'A14', 'A17', 'A18',
            'A19', 'A20', 'A21', 'A22', 'A23', 'A27']
    if config['protocol'] == 'protocol2':
        return [
            'A01', 'A06', 'A07', 'A08', 'A09', 'A10', 'A11', 'A12',
            'A15', 'A16', 'A24', 'A25', 'A26']
    return all_actions


def _resolve_actions(action_cfg, protocol_actions):
    return protocol_actions if action_cfg == 'all' else action_cfg


def _split_dataset_config(config, split_name, dataset_names):
    protocol_actions = _protocol_actions(config)
    split_cfg = config[split_name]
    result = {}
    for dataset_name in dataset_names:
        entry = split_cfg[dataset_name]
        default_split = 'training' if dataset_name == 'train_dataset' else 'validation'
        if dataset_name == 'test_dataset':
            default_split = 'test'
        data_form = entry.get('data_form')
        if data_form is None:
            actions = _resolve_actions(entry['actions'], protocol_actions)
            data_form = {subject: actions for subject in entry['subjects']}
        else:
            data_form = {
                subject: _resolve_actions(actions, protocol_actions)
                for subject, actions in data_form.items()
            }
        result[dataset_name] = {
            'modality': config['modality'],
            'split': entry.get('split', default_split),
            'data_form': data_form,
        }
    return result


def make_four_way_dataset(dataset_root, config):
    database = MMFi_Database(dataset_root)
    config_dataset = _split_dataset_config(
        config,
        'four_way_split',
        ('train_dataset', 'teacher_val_dataset', 'student_val_dataset', 'test_dataset'),
    )
    train_dataset = MMFi_Dataset(
        database, config['data_unit'], **config_dataset['train_dataset'])
    teacher_val_dataset = MMFi_Dataset(
        database, config['data_unit'], **config_dataset['teacher_val_dataset'])
    student_val_dataset = MMFi_Dataset(
        database, config['data_unit'], **config_dataset['student_val_dataset'])
    test_dataset = MMFi_Dataset(
        database, config['data_unit'], **config_dataset['test_dataset'])
    return train_dataset, teacher_val_dataset, student_val_dataset, test_dataset


def make_teacher_student_dataset(dataset_root, config):
    database = MMFi_Database(dataset_root)
    config_dataset = _split_dataset_config(
        config,
        'teacher_student_split',
        ('train_dataset', 'teacher_val_dataset', 'student_val_dataset'),
    )
    train_dataset = MMFi_Dataset(
        database, config['data_unit'], **config_dataset['train_dataset'])
    teacher_val_dataset = MMFi_Dataset(
        database, config['data_unit'], **config_dataset['teacher_val_dataset'])
    student_val_dataset = MMFi_Dataset(
        database, config['data_unit'], **config_dataset['student_val_dataset'])
    return train_dataset, teacher_val_dataset, student_val_dataset


def load_config(config_file, split):
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    env_protocol = os.environ.get("MMFI_PROTOCOL")
    if env_protocol:
        if env_protocol not in ("protocol1", "protocol2", "protocol3"):
            raise ValueError(
                "MMFI_PROTOCOL must be one of: protocol1, protocol2, protocol3")
        cfg["protocol"] = env_protocol
    cfg["data_unit"]    = "frame"
    cfg["split_to_use"] = split
    cfg["modality"]     = "wifi-csi|rgb"
    env_teacher_train = os.environ.get("TEACHER_CV_TRAIN_SUBJECTS")
    env_teacher_val = os.environ.get("TEACHER_CV_VAL_SUBJECTS")
    if split == "teacher_student_split" and (env_teacher_train or env_teacher_val):
        split_cfg = cfg.setdefault("teacher_student_split", {})
        if env_teacher_train:
            split_cfg.setdefault("train_dataset", {})["subjects"] = [
                s.strip() for s in env_teacher_train.split(",") if s.strip()]
        if env_teacher_val:
            split_cfg.setdefault("teacher_val_dataset", {})["subjects"] = [
                s.strip() for s in env_teacher_val.split(",") if s.strip()]
    env_teacher_train_form = os.environ.get("TEACHER_CV_TRAIN_FORM_JSON")
    env_teacher_val_form = os.environ.get("TEACHER_CV_VAL_FORM_JSON")
    if split == "teacher_student_split" and (
            env_teacher_train_form or env_teacher_val_form):
        split_cfg = cfg.setdefault("teacher_student_split", {})
        if env_teacher_train_form:
            split_cfg.setdefault("train_dataset", {})["data_form"] = (
                json.loads(env_teacher_train_form))
        if env_teacher_val_form:
            val_form = json.loads(env_teacher_val_form)
            split_cfg.setdefault("teacher_val_dataset", {})["data_form"] = val_form
            split_cfg.setdefault("student_val_dataset", {})["data_form"] = val_form
    heldout_cfg = cfg.setdefault("heldout_split", {})
    env_enabled = os.environ.get("HELDOUT_SPLIT_ENABLED")
    if env_enabled is not None:
        heldout_cfg["enabled"] = env_enabled.strip().lower() in (
            "1", "true", "yes", "y", "on")
    env_unit = os.environ.get("HELDOUT_SPLIT_UNIT")
    if env_unit:
        heldout_cfg["unit"] = env_unit
        if env_unit.strip().lower() == "subject":
            heldout_cfg["enabled"] = True
    env_test_size = os.environ.get("HELDOUT_SPLIT_TEST_SIZE")
    if env_test_size:
        heldout_cfg["test_size"] = float(env_test_size)
    env_seed = os.environ.get("HELDOUT_SPLIT_SEED")
    if env_seed:
        heldout_cfg["random_seed"] = int(env_seed)
    env_val_subjects = os.environ.get("HELDOUT_VAL_SUBJECTS")
    if env_val_subjects:
        heldout_cfg["val_subjects"] = [
            s.strip() for s in env_val_subjects.split(",") if s.strip()]
    env_test_subjects = os.environ.get("HELDOUT_TEST_SUBJECTS")
    if env_test_subjects:
        heldout_cfg["test_subjects"] = [
            s.strip() for s in env_test_subjects.split(",") if s.strip()]
    return cfg


def get_loaders(dataset_root, cfg, window, stride, batch_size, val_batch_size,
                role="student", num_workers=8, include_rgb_window=False,
                eval_num_workers=None):
    split_enabled = heldout_split_enabled(cfg)
    if cfg.get("split_to_use") == "four_way_split":
        train_ds_base, teacher_val_ds_base, student_val_ds_base, _ = make_four_way_dataset(dataset_root, cfg)
        val_ds_base = teacher_val_ds_base if role == "teacher" else student_val_ds_base
        split_heldout = False
    elif cfg.get("split_to_use") == "teacher_student_split":
        train_ds_base, teacher_val_ds_base, student_val_ds_base = make_teacher_student_dataset(dataset_root, cfg)
        val_ds_base = teacher_val_ds_base if role == "teacher" else student_val_ds_base
        split_heldout = False
    elif (cfg.get("split_to_use") == "cross_subject_split"
          and cfg.get("heldout_split", {}).get("unit") == "subject"):
        train_ds_base, _ = make_dataset(dataset_root, cfg)
        val_subjects, _ = get_cross_subject_val_test_subjects(cfg)
        val_ds_base = make_cross_subject_heldout_dataset(
            dataset_root, cfg, val_subjects)
        split_heldout = False
    else:
        train_ds_base, val_ds_base = make_dataset(dataset_root, cfg)
        split_heldout = split_enabled
    train_ds = TemporalWindowWrapper(
        train_ds_base, window=window, stride=stride,
        require_modalities=["wifi-csi"],
        include_rgb_window=include_rgb_window)
    heldout_ds = TemporalWindowWrapper(
        val_ds_base, window=window, stride=stride,
        require_modalities=["wifi-csi"],
        include_rgb_window=include_rgb_window)
    if split_heldout:
        val_ds, _ = split_heldout_windows(heldout_ds, cfg)
    else:
        val_ds = heldout_ds
    rng = torch.manual_seed(cfg.get("init_rand_seed", 0))
    val_workers = num_workers if eval_num_workers is None else eval_num_workers
    train_loader = make_dataloader(
        train_ds, True, rng, batch_size=batch_size,
        collate_fn_padd=collate_temporal, num_workers=num_workers)
    val_loader = make_dataloader(
        val_ds, False, rng, batch_size=val_batch_size,
        collate_fn_padd=collate_temporal, num_workers=val_workers)
    return train_ds, val_ds, train_loader, val_loader


def get_test_loader(dataset_root, cfg, window, stride, batch_size, num_workers=8,
                    include_rgb_window=False):
    split_enabled = heldout_split_enabled(cfg)
    if cfg.get("split_to_use") == "four_way_split":
        _, _, _, test_ds_base = make_four_way_dataset(dataset_root, cfg)
        test_ds = TemporalWindowWrapper(
            test_ds_base, window=window, stride=stride,
            require_modalities=["wifi-csi"],
            include_rgb_window=include_rgb_window)
    elif cfg.get("split_to_use") == "teacher_student_split":
        raise ValueError("teacher_student_split has no independent test set.")
    elif (cfg.get("split_to_use") == "cross_subject_split"
          and cfg.get("heldout_split", {}).get("unit") == "subject"):
        _, test_subjects = get_cross_subject_val_test_subjects(cfg)
        test_ds_base = make_cross_subject_heldout_dataset(
            dataset_root, cfg, test_subjects)
        test_ds = TemporalWindowWrapper(
            test_ds_base, window=window, stride=stride,
            require_modalities=["wifi-csi"],
            include_rgb_window=include_rgb_window)
    else:
        _, heldout_ds_base = make_dataset(dataset_root, cfg)
        heldout_ds = TemporalWindowWrapper(
            heldout_ds_base, window=window, stride=stride,
            require_modalities=["wifi-csi"],
            include_rgb_window=include_rgb_window)
        if split_enabled:
            _, test_ds = split_heldout_windows(heldout_ds, cfg)
        else:
            test_ds = heldout_ds
    rng = torch.manual_seed(cfg.get("init_rand_seed", 0))
    test_loader = make_dataloader(
        test_ds, False, rng, batch_size=batch_size,
        collate_fn_padd=collate_temporal, num_workers=num_workers)
    return test_ds, test_loader


# ----------------------------- 共享参数解析 -----------------------------
def add_common_args(parser):
    """添加所有脚本共享的参数"""
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")

    # Training
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--val_batch_size", type=int,   default=16)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-3)
    parser.add_argument("--warmup_epochs",  type=int,   default=3)
    parser.add_argument("--log_every",      type=int,   default=1000)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--lr_patience",    type=int,   default=3)
    parser.add_argument("--num_workers",    type=int,   default=8)
    parser.add_argument("--eval_num_workers", type=int, default=None,
                        help="Validation/test DataLoader workers; defaults to --num_workers")

    # Model
    parser.add_argument("--d_model",  type=int,   default=512)
    parser.add_argument("--window",   type=int,   default=16)
    parser.add_argument("--stride",   type=int,   default=1)
    parser.add_argument("--use_cbam", action="store_true", default=True)
    parser.add_argument("--no_cbam",  action="store_false", dest="use_cbam")
    parser.add_argument("--dropout",  type=float, default=0.15)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward",    type=int, default=1024)
    parser.add_argument("--pose_head_hidden",   type=int, default=512)
    parser.add_argument("--pose_head_type", type=str, default="mlp",
                        choices=["mlp", "root", "root_relative", "graph", "graph_root", "topology"])

    # 标准化
    parser.add_argument("--use_root_relative", action="store_true", default=False)
    parser.add_argument("--no_root_relative",  action="store_false", dest="use_root_relative")

    # 数据增强
    parser.add_argument("--aug_noise",     type=float, default=0.08)
    parser.add_argument("--aug_freq_mask", type=float, default=0.15)
    parser.add_argument("--aug_time_mask", type=float, default=0.15)
    parser.add_argument("--mixup_alpha",   type=float, default=0.4)

    # 早停
    parser.add_argument("--patience", type=int, default=8)

    # Split
    parser.add_argument("--split", type=str, default="cross_subject_split",
                        choices=["random_split", "cross_subject_split", "cross_scene_split",
                                 "four_way_split", "teacher_student_split"])

    return parser
