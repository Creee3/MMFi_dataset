# train_lupi_rgb_teacher_old.py
# =============================================================================
# LUPI + CMC + CBAM + Knowledge Distillation 训练脚本
# 使用纯 RGB Teacher 做蒸馏，其他逻辑尽量与 train_lupi.py 保持一致。
# =============================================================================

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import (
    Logger, make_run_dir, save_json, count_params, fmt_metrics,
    augment_wifi, normalize_pose2d, normalize_pose3d, mixup_data,
    info_nce_loss,
    TemporalLUPICMC_CBAM, RGBOnlyTeacher,
    eval_temporal_model,
    load_config, get_loaders, add_common_args
)


def train_lupi_rgb_teacher(args, train_loader, val_loader, device, teacher_ckpt=None):
    run_dir  = make_run_dir("temporal_lupi_rgb_teacher_cmc_cbam_distill")
    log_path = os.path.join(run_dir, "log.txt")
    logger   = Logger(log_path)
    sys.stdout = logger

    print("=" * 60)
    print("🚀 Training: Temporal LUPI + CMC + CBAM + RGB Teacher Distillation")
    print(f"Run dir: {run_dir}")
    print(f"Split: {args.split}")
    print(f"Window: {args.window}, CBAM: {args.use_cbam}")
    print(f"Dropout: {args.dropout}, Weight decay: {args.weight_decay}")
    print(f"★ transformer_layers: {args.transformer_layers}, "
          f"dim_feedforward: {args.dim_feedforward}")
    print(f"★ SWA: start_epoch={args.swa_start}, swa_lr={args.swa_lr}")
    print(f"lambda_cmc: {args.lambda_cmc}, tau: {args.cmc_tau}")
    print(f"lambda_distill: {args.lambda_distill}")
    print(f"Augmentation: noise={args.aug_noise}, freq_mask={args.aug_freq_mask}, "
          f"time_mask={args.aug_time_mask}")
    print(f"★ Mixup alpha: {args.mixup_alpha}  (0=关闭)")
    print(f"★ Scheduler: ReduceLROnPlateau (factor=0.5, patience={args.lr_patience})")
    print(f"Early stopping patience: {args.patience}")
    print(f"Root-relative normalization: {args.use_root_relative}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model = TemporalLUPICMC_CBAM(
        in_ch=3, d_model=args.d_model, d_proj=args.cmc_proj_dim,
        window=args.window, use_cbam=args.use_cbam, dropout=args.dropout,
        num_layers=args.transformer_layers,
        dim_feedforward=args.dim_feedforward
    ).to(device)

    teacher = None
    if teacher_ckpt is not None and os.path.isfile(teacher_ckpt):
        teacher = RGBOnlyTeacher(
            d_model=args.d_model, dropout=args.dropout
        ).to(device)
        ckpt  = torch.load(teacher_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)
        teacher.load_state_dict(state, strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"✅ Loaded RGB-only Teacher from: {teacher_ckpt}")
        print(f"   Teacher will provide pose distillation supervision")

    else:
        print("⚠️  No teacher checkpoint provided, pose distillation disabled")

    print(f"Parameters: {count_params(model):,}")

    train_params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(train_params, lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=args.lr_patience, min_lr=1e-6)

    swa_model   = torch.optim.swa_utils.AveragedModel(model)
    swa_started = False

    best_path = os.path.join(run_dir, "best.pth")
    hist_path = os.path.join(run_dir, "history.json")

    history, best_metric, no_improve = [], 1e9, 0

    class WiFiOnlyWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, w):
            return self.m.forward_wifi(w)

    for ep in range(1, args.epochs + 1):
        model.train()
        running = running_pose = running_cmc = running_distill = 0.0

        if args.warmup_epochs > 0 and ep <= args.warmup_epochs:
            warm_lr = args.lr * ep / args.warmup_epochs
            for g in opt.param_groups:
                g["lr"] = warm_lr

        current_lr = opt.param_groups[0]["lr"]

        for i, batch in enumerate(train_loader, 1):
            wifi_window = batch["input_wifi-csi_window"].to(device)
            rgb         = batch["input_rgb"].to(device)
            gt          = batch["output"].to(device)

            if (args.aug_noise > 0 or args.aug_freq_mask > 0
                    or args.aug_time_mask > 0):
                wifi_window = augment_wifi(
                    wifi_window, args.aug_noise,
                    args.aug_freq_mask, args.aug_time_mask)

            if args.use_root_relative:
                rgb = normalize_pose2d(rgb)
                gt  = normalize_pose3d(gt)

            lam = 1.0
            if args.mixup_alpha > 0:
                wifi_window, rgb, gt, lam, _ = mixup_data(
                    wifi_window, rgb, gt, alpha=args.mixup_alpha)

            pred3d, feat_w = model.forward_wifi(wifi_window)
            loss_pose = F.l1_loss(pred3d, gt)

            teacher_pred = None
            feat_v_teacher = None
            if teacher is not None and (args.lambda_cmc > 0 or args.lambda_distill > 0):
                with torch.no_grad():
                    teacher_pred, feat_v_teacher = teacher(None, rgb, return_feat=True)

            loss_cmc = torch.tensor(0.0, device=device)
            if args.lambda_cmc > 0 and feat_v_teacher is not None:
                z_w = model.proj_w(feat_w)
                z_v = model.proj_v(feat_v_teacher)
                loss_cmc = info_nce_loss(z_w, z_v, tau=args.cmc_tau)

            loss_distill = torch.tensor(0.0, device=device)
            if args.lambda_distill > 0 and teacher_pred is not None:
                loss_distill = F.l1_loss(pred3d, teacher_pred.detach())

            loss = (loss_pose
                    + args.lambda_cmc * loss_cmc
                    + args.lambda_distill * loss_distill)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            opt.step()

            running         += loss.item()
            running_pose    += loss_pose.item()
            running_cmc     += loss_cmc.item()
            running_distill += loss_distill.item()

            if i % args.log_every == 0:
                print(f"  batch {i}/{len(train_loader)} "
                      f"loss={running/i:.4f} (pose={running_pose/i:.4f}, "
                      f"cmc={running_cmc/i:.4f}, distill={running_distill/i:.4f}) "
                      f"lam={lam:.3f} lr={current_lr:.6f}")

        wifi_only = WiFiOnlyWrapper(model).to(device)
        mpjpe, pa, pck = eval_temporal_model(
            wifi_only, val_loader, device,
            use_root_relative=args.use_root_relative)
        if ep > args.warmup_epochs:
            scheduler.step(pa)
        print(f"Epoch {ep}: val(WiFi-only) {fmt_metrics(mpjpe, pa, pck)}")

        if args.swa_start > 0 and ep >= args.swa_start:
            swa_model.update_parameters(model)
            swa_started = True

        history.append({
            "epoch": ep, "mpjpe": mpjpe, "pa_mpjpe": pa,
            "pck@20": pck.get("pck@20", 0),
            "pck@50": pck.get("pck@50", 0),
            "lr": current_lr,
            "train_loss":    running         / len(train_loader),
            "train_pose":    running_pose    / len(train_loader),
            "train_cmc":     running_cmc     / len(train_loader),
            "train_distill": running_distill / len(train_loader),
        })

        if pa < best_metric:
            best_metric = pa
            no_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       best_path)
            print(f"  ✅ New best! PA-MPJPE={pa*1000:.1f}mm | "
                  f"PCK@20={pck.get('pck@20',0):.1f}% | "
                  f"PCK@50={pck.get('pck@50',0):.1f}%")
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"  ⏹ Early stopping! No improvement for "
                      f"{args.patience} epochs")
                break

        save_json(history, hist_path)

    if swa_started:
        print("\n🔄 SWA: 更新 BatchNorm 统计量...")
        swa_model.train()
        with torch.no_grad():
            for batch in train_loader:
                wifi = batch["input_wifi-csi_window"].to(device)
                swa_model.module.forward_wifi(wifi)

        class SWAWiFiWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, w):
                return self.m.module.forward_wifi(w)

        swa_wifi_only = SWAWiFiWrapper(swa_model).to(device)
        swa_mpjpe, swa_pa, swa_pck = eval_temporal_model(
            swa_wifi_only, val_loader, device,
            use_root_relative=args.use_root_relative)
        print(f"★ SWA model: {fmt_metrics(swa_mpjpe, swa_pa, swa_pck)}")

        swa_save_path = os.path.join(run_dir, "best_swa.pth")
        torch.save(
            {"model": swa_model.module.state_dict(), "args": vars(args)},
            swa_save_path)

        if swa_pa < best_metric:
            best_metric = swa_pa
            print(f"  ✅ SWA model is new best! Saved to best_swa.pth")
        else:
            print(f"  ℹ️  SWA model saved "
                  f"(PA-MPJPE={swa_pa*1000:.1f}mm, "
                  f"best regular={best_metric*1000:.1f}mm)")

    print("=" * 60)
    print(f"✅ RGB-Teacher LUPI+CMC+CBAM+Distill finished! "
          f"Best PA-MPJPE: {best_metric*1000:.1f}mm")
    print(f"   transformer_layers={args.transformer_layers}, "
          f"dim_feedforward={args.dim_feedforward}")
    print("=" * 60)
    sys.stdout = logger.terminal
    logger.close()
    return best_metric, run_dir


def main():
    parser = argparse.ArgumentParser(
        description="LUPI + CMC + Distillation Training (RGB-only Teacher)")
    add_common_args(parser)

    parser.add_argument("--teacher_ckpt", type=str, default=None,
                        help="Path to RGB-only teacher checkpoint (.pth)")
    parser.add_argument("--swa_start", type=int,   default=10)
    parser.add_argument("--swa_lr",    type=float, default=5e-5)
    parser.add_argument("--lambda_cmc",       type=float, default=0.2)
    parser.add_argument("--cmc_tau",          type=float, default=0.1)
    parser.add_argument("--cmc_proj_dim",     type=int,   default=128)
    parser.add_argument("--cmc_stopgrad_rgb", action="store_true")
    parser.add_argument("--cmc_freeze_rgb",   action="store_true")
    parser.add_argument("--lambda_distill", type=float, default=1.0)

    args = parser.parse_args()

    seed = int(os.environ.get("TRAIN_SEED", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(args.config_file, args.split)

    print(f"Split: {args.split} | Window: {args.window} | Stride: {args.stride}")
    print(f"Dropout: {args.dropout} | Mixup alpha: {args.mixup_alpha}")
    print(f"lambda_cmc: {args.lambda_cmc} | lambda_distill: {args.lambda_distill}")

    train_ds, val_ds, train_loader, val_loader = get_loaders(
        args.dataset_root, cfg, args.window, args.stride,
        args.batch_size, args.val_batch_size)
    print(f"Train samples (windowed): {len(train_ds)}")
    print(f"Val   samples (windowed): {len(val_ds)}")

    if args.teacher_ckpt is None:
        print("⚠️  No --teacher_ckpt provided! Distillation will be disabled.")

    best_metric, run_dir = train_lupi_rgb_teacher(
        args, train_loader, val_loader, device,
        teacher_ckpt=args.teacher_ckpt)
    print(f"\n📊 Result: PA-MPJPE = {best_metric*1000:.1f}mm")
    print(f"📁 Run dir: {run_dir}")


if __name__ == "__main__":
    main()
