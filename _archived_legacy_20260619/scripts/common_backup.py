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

from mmfi_lib.mmfi import make_dataset, make_dataloader
from mmfi_lib.evaluate import calulate_error

# ----------------------------- 随机种子 -----------------------------
seed = int(os.environ.get("TRAIN_SEED", 0))
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# ----------------------------- Pose Normalization -----------------------------
def normalize_pose2d(pose2d):
    root = (pose2d[:, 11:12, :] + pose2d[:, 12:13, :]) / 2
    return pose2d - root


def normalize_pose3d(pose3d):
    root = (pose3d[:, 11:12, :] + pose3d[:, 12:13, :]) / 2
    return pose3d - root


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
                 require_modalities: Optional[List[str]] = None):
        self.base_ds = base_ds
        self.window  = int(window)
        self.stride  = int(stride)
        self.require_modalities = require_modalities
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
        for t in range(self.window):
            idx_t  = idx_cur - (self.window - 1 - t) * self.stride
            path_t = self._build_frame_path(wifi_path_cur, idx_t)
            if not os.path.isfile(path_t):
                raise FileNotFoundError(path_t)
            wifi_t = self.base_ds.read_frame(path_t)
            wifi_frames.append(np.array(wifi_t))

        wifi_arr = np.stack(wifi_frames, axis=0)
        sample["input_wifi-csi_window"] = torch.FloatTensor(wifi_arr)
        sample["idx_window_end"]        = idx_cur
        return sample


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


# ----------------------------- Pose Head -----------------------------
class PoseHead(nn.Module):
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 512), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256),     nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, 17 * 3))

    def forward(self, feat):
        return self.net(feat).view(-1, 17, 3)


# ----------------------------- Complete Models -----------------------------
class TemporalWiFiStudentCBAM(nn.Module):
    def __init__(self, in_ch=3, d_model=512, window=16, use_cbam=True, dropout=0.1):
        super().__init__()
        self.wifi_encoder = TemporalWiFiEncoderCBAM(
            in_ch, d_model, window, use_cbam, dropout)
        self.pose_head = PoseHead(d_model, dropout)

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
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
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
                 num_layers=2, dim_feedforward=1024):
        super().__init__()
        self.wifi_encoder = TemporalWiFiEncoderCBAM(
            in_ch, d_model, window, use_cbam, dropout,
            num_layers, dim_feedforward)
        self.pose_head   = PoseHead(d_model, dropout)
        self.proj_w = nn.Sequential(
            nn.Linear(d_model, d_proj), nn.ReLU(inplace=True),
            nn.Linear(d_proj, d_proj))
        self.proj_v = nn.Sequential(
            nn.Linear(d_model, d_proj), nn.ReLU(inplace=True),
            nn.Linear(d_proj, d_proj))

    def forward_wifi(self, wifi_window):
        feat = self.wifi_encoder(wifi_window)
        return self.pose_head(feat), feat


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
            rgb         = batch["input_rgb"].to(device)
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
def load_config(config_file, split):
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["data_unit"]    = "frame"
    cfg["split_to_use"] = split
    cfg["modality"]     = "wifi-csi|rgb"
    return cfg


def get_loaders(dataset_root, cfg, window, stride, batch_size, val_batch_size):
    train_ds_base, val_ds_base = make_dataset(dataset_root, cfg)
    train_ds = TemporalWindowWrapper(
        train_ds_base, window=window, stride=stride,
        require_modalities=["wifi-csi"])
    val_ds = TemporalWindowWrapper(
        val_ds_base, window=window, stride=stride,
        require_modalities=["wifi-csi"])
    rng = torch.manual_seed(cfg.get("init_rand_seed", 0))
    train_loader = make_dataloader(
        train_ds, True, rng, batch_size=batch_size,
        collate_fn_padd=collate_temporal, num_workers=5)
    val_loader = make_dataloader(
        val_ds, False, rng, batch_size=val_batch_size,
        collate_fn_padd=collate_temporal, num_workers=5)
    return train_ds, val_ds, train_loader, val_loader


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
    parser.add_argument("--log_every",      type=int,   default=200)
    parser.add_argument("--device",         default="cuda")
    parser.add_argument("--lr_patience",    type=int,   default=3)

    # Model
    parser.add_argument("--d_model",  type=int,   default=512)
    parser.add_argument("--window",   type=int,   default=16)
    parser.add_argument("--stride",   type=int,   default=1)
    parser.add_argument("--use_cbam", action="store_true", default=True)
    parser.add_argument("--no_cbam",  action="store_false", dest="use_cbam")
    parser.add_argument("--dropout",  type=float, default=0.15)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward",    type=int, default=1024)

    # 标准化
    parser.add_argument("--use_root_relative", action="store_true", default=False)
    parser.add_argument("--no_root_relative",  action="store_false", dest="use_root_relative")

    # 数据增强
    parser.add_argument("--aug_noise",     type=float, default=0.08)
    parser.add_argument("--aug_freq_mask", type=float, default=0.15)
    parser.add_argument("--aug_time_mask", type=float, default=0.15)
    parser.add_argument("--mixup_alpha",   type=float, default=0.3)

    # 早停
    parser.add_argument("--patience", type=int, default=8)

    # Split
    parser.add_argument("--split", type=str, default="cross_subject_split",
                        choices=["random_split", "cross_subject_split", "cross_scene_split"])

    return parser