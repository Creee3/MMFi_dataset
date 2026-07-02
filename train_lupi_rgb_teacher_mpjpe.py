# train_lupi_rgb_teacher_mpjpe.py
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
    info_nce_loss, root_relative_pose_loss, root_position_loss,
    bone_length_loss, subject_robust_l1_loss,
    TemporalLUPICMC_CBAM, RGBOnlyTeacher,
    eval_temporal_model,
    load_config, get_loaders, get_test_loader, add_common_args
)
from train_baseline_mpjpe import (
    checkpoint_payload,
    maybe_save_topk_checkpoint,
    maybe_save_tail_checkpoint,
    build_topk_soup,
    build_tail_soup,
)


def train_lupi_rgb_teacher(args, train_loader, val_loader, device, teacher_ckpt=None):
    run_dir  = args.run_dir or make_run_dir("temporal_lupi_rgb_teacher_cmc_cbam_distill")
    os.makedirs(run_dir, exist_ok=True)
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
    print(f"cmc_start_epoch: {args.cmc_start_epoch}, "
          f"cmc_ramp_epochs: {args.cmc_ramp_epochs}, "
          f"cmc_encoder_grad_scale: {args.cmc_encoder_grad_scale}")
    print(f"lambda_distill: {args.lambda_distill}")
    print(f"lambda_rel_pose: {args.lambda_rel_pose}, lambda_root: {args.lambda_root}")
    print(f"lambda_bone: {args.lambda_bone}, pose_head_type: {args.pose_head_type}")
    print(f"subject_robust_weight: {args.subject_robust_weight}")
    print(f"Top-k checkpoints: {args.topk_ckpts}, soup: {args.topk_soup}")
    print(f"Tail checkpoints: {args.tail_ckpts}, soup: {args.tail_soup}")
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
        dim_feedforward=args.dim_feedforward,
        pose_head_hidden=args.pose_head_hidden,
        pose_head_type=args.pose_head_type
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

    swa_model   = None
    swa_started = False

    best_path = os.path.join(run_dir, "best.pth")
    last_path = os.path.join(run_dir, "last.pth")
    hist_path = os.path.join(run_dir, "history.json")

    history, topk_entries, tail_entries = [], [], []
    best_metric, best_source, no_improve = 1e9, "none", 0

    class WiFiOnlyWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, w):
            return self.m.forward_wifi(w)

    for ep in range(1, args.epochs + 1):
        model.train()
        running = running_pose = running_cmc = running_distill = 0.0
        running_rel = running_root = running_bone = 0.0

        if args.warmup_epochs > 0 and ep <= args.warmup_epochs:
            warm_lr = args.lr * ep / args.warmup_epochs
            for g in opt.param_groups:
                g["lr"] = warm_lr

        current_lr = opt.param_groups[0]["lr"]
        if ep < args.cmc_start_epoch:
            cmc_weight = 0.0
        elif args.cmc_ramp_epochs > 0:
            cmc_weight = args.lambda_cmc * min(
                1.0, (ep - args.cmc_start_epoch + 1) / args.cmc_ramp_epochs)
        else:
            cmc_weight = args.lambda_cmc

        for i, batch in enumerate(train_loader, 1):
            wifi_window = batch["input_wifi-csi_window"].to(device)
            rgb         = batch["input_rgb"].to(device)
            gt          = batch["output"].to(device)
            subjects    = batch.get("subject", [])

            if (args.aug_noise > 0 or args.aug_freq_mask > 0
                    or args.aug_time_mask > 0):
                wifi_window = augment_wifi(
                    wifi_window, args.aug_noise,
                    args.aug_freq_mask, args.aug_time_mask)

            if args.use_root_relative:
                rgb = normalize_pose2d(rgb)
                gt  = normalize_pose3d(gt)

            lam = 1.0
            subjects_for_loss = subjects
            if args.mixup_alpha > 0:
                wifi_window, rgb, gt, lam, _ = mixup_data(
                    wifi_window, rgb, gt, alpha=args.mixup_alpha)
                subjects_for_loss = []

            pred3d, feat_w = model.forward_wifi(wifi_window)
            loss_pose = subject_robust_l1_loss(
                pred3d, gt, subjects_for_loss,
                robust_weight=args.subject_robust_weight)
            loss_rel = torch.tensor(0.0, device=device)
            if args.lambda_rel_pose > 0:
                loss_rel = root_relative_pose_loss(pred3d, gt)
            loss_root = torch.tensor(0.0, device=device)
            if args.lambda_root > 0:
                loss_root = root_position_loss(pred3d, gt)
            loss_bone = torch.tensor(0.0, device=device)
            if args.lambda_bone > 0:
                loss_bone = bone_length_loss(pred3d, gt)

            teacher_pred = None
            feat_v_teacher = None
            if teacher is not None and (args.lambda_cmc > 0 or args.lambda_distill > 0):
                with torch.no_grad():
                    teacher_pred, feat_v_teacher = teacher(None, rgb, return_feat=True)

            loss_cmc = torch.tensor(0.0, device=device)
            if cmc_weight > 0 and feat_v_teacher is not None:
                if args.cmc_encoder_grad_scale < 1.0:
                    feat_w_for_cmc = (feat_w.detach()
                                      + args.cmc_encoder_grad_scale
                                      * (feat_w - feat_w.detach()))
                else:
                    feat_w_for_cmc = feat_w
                z_w = model.proj_w(feat_w_for_cmc)
                z_v = model.proj_v(feat_v_teacher)
                loss_cmc = info_nce_loss(z_w, z_v, tau=args.cmc_tau)

            loss_distill = torch.tensor(0.0, device=device)
            if args.lambda_distill > 0 and teacher_pred is not None:
                loss_distill = F.l1_loss(pred3d, teacher_pred.detach())

            loss = (loss_pose
                    + args.lambda_rel_pose * loss_rel
                    + args.lambda_root * loss_root
                    + args.lambda_bone * loss_bone
                    + cmc_weight * loss_cmc
                    + args.lambda_distill * loss_distill)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            opt.step()

            running         += loss.item()
            running_pose    += loss_pose.item()
            running_rel     += loss_rel.item()
            running_root    += loss_root.item()
            running_bone    += loss_bone.item()
            running_cmc     += loss_cmc.item()
            running_distill += loss_distill.item()

            if i % args.log_every == 0:
                print(f"  batch {i}/{len(train_loader)} "
                      f"loss={running/i:.4f} (pose={running_pose/i:.4f}, "
                      f"rel={running_rel/i:.4f}, root={running_root/i:.4f}, "
                      f"bone={running_bone/i:.4f}, "
                      f"cmc={running_cmc/i:.4f}, distill={running_distill/i:.4f}) "
                      f"cmc_w={cmc_weight:.4f} lam={lam:.3f} lr={current_lr:.6f}")

        wifi_only = WiFiOnlyWrapper(model).to(device)
        mpjpe, pa, pck = eval_temporal_model(
            wifi_only, val_loader, device,
            use_root_relative=args.use_root_relative)
        if ep > args.warmup_epochs:
            scheduler.step(mpjpe)
        print(f"Epoch {ep}: val(WiFi-only) {fmt_metrics(mpjpe, pa, pck)}")

        if args.swa_start > 0 and ep >= args.swa_start:
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
            swa_model.update_parameters(model)
            swa_started = True

        history.append({
            "epoch": ep, "mpjpe": mpjpe, "pa_mpjpe": pa,
            "pck@20": pck.get("pck@20", 0),
            "pck@50": pck.get("pck@50", 0),
            "lr": current_lr,
            "cmc_weight": cmc_weight,
            "train_loss":    running         / len(train_loader),
            "train_pose":    running_pose    / len(train_loader),
            "train_rel":     running_rel     / len(train_loader),
            "train_root":    running_root    / len(train_loader),
            "train_bone":    running_bone    / len(train_loader),
            "train_cmc":     running_cmc     / len(train_loader),
            "train_distill": running_distill / len(train_loader),
        })
        torch.save(checkpoint_payload(model, args, "regular"), last_path)

        if mpjpe < best_metric:
            best_metric = mpjpe
            best_source = "regular"
            no_improve = 0
            torch.save(checkpoint_payload(model, args, "regular"), best_path)
            print(f"  ✅ New best! MPJPE={mpjpe*1000:.1f}mm | "
                  f"PCK@20={pck.get('pck@20',0):.1f}% | "
                  f"PCK@50={pck.get('pck@50',0):.1f}%")
        else:
            no_improve += 1

        topk_entries = maybe_save_topk_checkpoint(
            args, run_dir, model, ep, mpjpe, pa, pck, topk_entries)
        tail_entries = maybe_save_tail_checkpoint(
            args, run_dir, model, ep, mpjpe, pa, pck, tail_entries)

        save_json(history, hist_path)

        if args.patience > 0 and no_improve >= args.patience:
            print(f"  ⏹ Early stopping! No improvement for "
                  f"{args.patience} epochs")
            break

    best_metric, soup_source = build_topk_soup(
        args, run_dir, model, val_loader, device,
        topk_entries, best_metric, best_path)
    if soup_source is not None:
        best_source = soup_source

    best_metric, tail_source = build_tail_soup(
        args, run_dir, model, val_loader, device,
        tail_entries, best_metric, best_path)
    if tail_source is not None:
        best_source = tail_source

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
        swa_payload = {
            "model": swa_model.module.state_dict(),
            "args": vars(args),
            "checkpoint_source": "swa",
        }
        torch.save(swa_payload, swa_save_path)

        if swa_mpjpe < best_metric:
            best_metric = swa_mpjpe
            best_source = "swa"
            torch.save(swa_payload, best_path)
            print(f"  ✅ SWA model is new best! Saved to best.pth and best_swa.pth")
        else:
            print(f"  ℹ️  SWA model saved "
                  f"(MPJPE={swa_mpjpe*1000:.1f}mm, "
                  f"best regular={best_metric*1000:.1f}mm)")

    print("=" * 60)
    print(f"✅ RGB-Teacher LUPI+CMC+CBAM+Distill finished! "
          f"Best MPJPE: {best_metric*1000:.1f}mm")
    print(f"   transformer_layers={args.transformer_layers}, "
          f"dim_feedforward={args.dim_feedforward}")
    print("=" * 60)
    save_json({
        "status": "complete",
        "best_val_mpjpe": best_metric,
        "checkpoint_source": best_source,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": "best.pth",
        "topk_ckpts": int(getattr(args, "topk_ckpts", 0) or 0),
        "topk_soup": bool(getattr(args, "topk_soup", False)),
        "topk_soup_replace_best": bool(getattr(args, "topk_soup_replace_best", False)),
        "tail_ckpts": int(getattr(args, "tail_ckpts", 0) or 0),
        "tail_soup": bool(getattr(args, "tail_soup", False)),
        "tail_soup_replace_best": bool(getattr(args, "tail_soup_replace_best", False)),
    }, os.path.join(run_dir, "train_complete.json"))
    sys.stdout = logger.terminal
    logger.close()
    return best_metric, run_dir


def evaluate_best_on_test(args, run_dir, device, test_loader):
    best_path = os.path.join(run_dir, "best.pth")
    if not os.path.isfile(best_path):
        print("⚠️  No best checkpoint found for final test evaluation.")
        return None

    model = TemporalLUPICMC_CBAM(
        in_ch=3, d_model=args.d_model, d_proj=args.cmc_proj_dim,
        window=args.window, use_cbam=args.use_cbam, dropout=args.dropout,
        num_layers=args.transformer_layers,
        dim_feedforward=args.dim_feedforward,
        pose_head_hidden=getattr(args, "pose_head_hidden", 512),
        pose_head_type=getattr(args, "pose_head_type", "mlp")
    ).to(device)
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    class WiFiOnlyWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, w):
            return self.m.forward_wifi(w)

    wifi_only = WiFiOnlyWrapper(model).to(device)
    mpjpe, pa, pck = eval_temporal_model(
        wifi_only, test_loader, device,
        use_root_relative=args.use_root_relative)
    test_result = {
        "mpjpe": mpjpe,
        "pa_mpjpe": pa,
        "pck@20": pck.get("pck@20", 0.0),
        "pck@50": pck.get("pck@50", 0.0),
        "checkpoint": os.path.basename(best_path),
        "checkpoint_source": ckpt.get("checkpoint_source", "unknown"),
    }
    save_json(test_result, os.path.join(run_dir, "final_test.json"))
    print(f"\n🧪 Final test: {fmt_metrics(mpjpe, pa, pck)}")
    return test_result


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
    parser.add_argument("--cmc_start_epoch", type=int, default=1,
                        help="Epoch from which CMC starts contributing")
    parser.add_argument("--cmc_ramp_epochs", type=int, default=0,
                        help="Linearly ramp CMC weight over this many epochs")
    parser.add_argument("--cmc_encoder_grad_scale", type=float, default=1.0,
                        help="Scale CMC gradients flowing into the WiFi encoder")
    parser.add_argument("--lambda_distill", type=float, default=1.0)
    parser.add_argument("--lambda_rel_pose", type=float, default=0.0,
                        help="Auxiliary root-relative 3D pose loss weight")
    parser.add_argument("--lambda_root", type=float, default=0.0,
                        help="Auxiliary pelvis/root position loss weight")
    parser.add_argument("--lambda_bone", type=float, default=0.0,
                        help="Auxiliary bone-length consistency loss weight")
    parser.add_argument("--subject_robust_weight", type=float, default=0.0,
                        help="Blend average pose loss with worst-subject pose loss")
    parser.add_argument("--topk_ckpts", type=int, default=0,
                        help="Keep the best K validation checkpoints for diagnostics/soup; 0 disables")
    parser.add_argument("--topk_soup", action="store_true",
                        help="Average the saved top-k checkpoints and compare the soup on validation")
    parser.add_argument("--topk_soup_replace_best", action="store_true",
                        help="Allow top-k soup to overwrite best.pth when it improves validation")
    parser.add_argument("--tail_ckpts", type=int, default=0,
                        help="Keep the last K epoch checkpoints for diagnostics/soup; 0 disables")
    parser.add_argument("--tail_soup", action="store_true",
                        help="Average the saved tail checkpoints and compare the soup on validation")
    parser.add_argument("--tail_soup_replace_best", action="store_true",
                        help="Allow tail soup to overwrite best.pth when it improves validation")
    parser.add_argument("--skip_final_test", action="store_true",
                        help="Only train/save checkpoints; final-test evaluation can run separately")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Optional explicit run directory for isolated execution")

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
        args.batch_size, args.val_batch_size, role="student",
        num_workers=args.num_workers)
    print(f"Train samples (windowed): {len(train_ds)}")
    print(f"Val   samples (windowed): {len(val_ds)}")

    if args.teacher_ckpt is None:
        print("⚠️  No --teacher_ckpt provided! Distillation will be disabled.")

    best_metric, run_dir = train_lupi_rgb_teacher(
        args, train_loader, val_loader, device,
        teacher_ckpt=args.teacher_ckpt)
    final_test = None
    if args.split == "four_way_split" and not args.skip_final_test:
        test_ds, test_loader = get_test_loader(
            args.dataset_root, cfg, args.window, args.stride,
            args.val_batch_size, num_workers=args.num_workers)
        print(f"Test  samples (windowed): {len(test_ds)}")
        final_test = evaluate_best_on_test(args, run_dir, device, test_loader)
    elif args.skip_final_test:
        print("🧪 Final test skipped; evaluate best.pth in a separate process.")
    print(f"\n📊 Best val MPJPE = {best_metric*1000:.1f}mm")
    if final_test is not None:
        print(f"🧪 Final test MPJPE = {final_test['mpjpe']*1000:.1f}mm")
    print(f"📁 Run dir: {run_dir}")


if __name__ == "__main__":
    main()
