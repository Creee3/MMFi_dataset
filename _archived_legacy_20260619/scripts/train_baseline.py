# train_baseline.py
# =============================================================================
# WiFi Baseline 训练脚本 (Temporal WiFi + CBAM, 无RGB)
# =============================================================================

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from common import (
    Logger, make_run_dir, save_json, count_params, fmt_metrics,
    augment_wifi, normalize_pose3d,
    TemporalWiFiStudentCBAM,
    eval_temporal_model,
    load_config, get_loaders, add_common_args
)

from torch.cuda.amp import autocast, GradScaler



def train_baseline(args, train_loader, val_loader, device):
    run_dir  = make_run_dir("temporal_baseline_cbam")
    log_path = os.path.join(run_dir, "log.txt")
    logger   = Logger(log_path)
    sys.stdout = logger

    print("=" * 60)
    print("🚀 Training: Temporal WiFi Baseline + CBAM")
    print(f"Run dir: {run_dir}")
    print(f"Split: {args.split}")
    print(f"Window: {args.window}, CBAM: {args.use_cbam}")
    print(f"Dropout: {args.dropout}, Weight decay: {args.weight_decay}")
    print(f"Augmentation: noise={args.aug_noise}, freq_mask={args.aug_freq_mask}, "
          f"time_mask={args.aug_time_mask}")
    print(f"Mixup alpha: {args.mixup_alpha}")
    print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience={args.lr_patience})")
    print(f"Early stopping patience: {args.patience}")
    print(f"Root-relative normalization: {args.use_root_relative}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model = TemporalWiFiStudentCBAM(
        in_ch=3, d_model=args.d_model, window=args.window,
        use_cbam=args.use_cbam, dropout=args.dropout
    ).to(device)
    print(f"Parameters: {count_params(model):,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=args.lr_patience, min_lr=1e-6)

    best_path = os.path.join(run_dir, "best.pth")
    last_path = os.path.join(run_dir, "last.pth")
    hist_path = os.path.join(run_dir, "history.json")

    history, best_metric, no_improve = [], 1e9, 0

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        if args.warmup_epochs > 0 and ep <= args.warmup_epochs:
            warm_lr = args.lr * ep / args.warmup_epochs
            for g in opt.param_groups:
                g["lr"] = warm_lr

        current_lr = opt.param_groups[0]["lr"]

        for i, batch in enumerate(train_loader, 1):
            wifi_window = batch["input_wifi-csi_window"].to(device)
            gt          = batch["output"].to(device)

            if args.aug_noise > 0 or args.aug_freq_mask > 0 or args.aug_time_mask > 0:
                wifi_window = augment_wifi(
                    wifi_window, args.aug_noise,
                    args.aug_freq_mask, args.aug_time_mask)

            if args.use_root_relative:
                gt = normalize_pose3d(gt)

            if args.mixup_alpha > 0:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                idx = torch.randperm(wifi_window.size(0), device=device)
                wifi_window = lam * wifi_window + (1 - lam) * wifi_window[idx]
                gt          = lam * gt          + (1 - lam) * gt[idx]

            pred3d, _ = model(wifi_window)
            loss      = F.l1_loss(pred3d, gt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item()

            if i % args.log_every == 0:
                print(f"  batch {i}/{len(train_loader)} "
                      f"loss={running/i:.4f} lr={current_lr:.6f}")

        mpjpe, pa, pck = eval_temporal_model(
            model, val_loader, device,
            use_root_relative=args.use_root_relative)
        if ep > args.warmup_epochs:
            scheduler.step(pa)
        print(f"Epoch {ep}: val {fmt_metrics(mpjpe, pa, pck)}")

        history.append({
            "epoch": ep, "mpjpe": mpjpe, "pa_mpjpe": pa,
            "pck@20": pck.get("pck@20", 0),
            "pck@50": pck.get("pck@50", 0),
            "lr": current_lr
        })
        torch.save({"model": model.state_dict(), "args": vars(args)}, last_path)

        if pa < best_metric:
            best_metric = pa
            no_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
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

    print("=" * 60)
    print(f"✅ Baseline+CBAM finished! Best PA-MPJPE: {best_metric*1000:.1f}mm")
    print("=" * 60)
    sys.stdout = logger.terminal
    logger.close()
    return best_metric, run_dir


def main():
    parser = argparse.ArgumentParser(
        description="WiFi Baseline Training (Temporal + CBAM)")
    add_common_args(parser)
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

    train_ds, val_ds, train_loader, val_loader = get_loaders(
        args.dataset_root, cfg, args.window, args.stride,
        args.batch_size, args.val_batch_size)
    print(f"Train samples (windowed): {len(train_ds)}")
    print(f"Val   samples (windowed): {len(val_ds)}")

    best_metric, run_dir = train_baseline(args, train_loader, val_loader, device)
    print(f"\n📊 Result: PA-MPJPE = {best_metric*1000:.1f}mm")
    print(f"📁 Run dir: {run_dir}")


if __name__ == "__main__":
    main()