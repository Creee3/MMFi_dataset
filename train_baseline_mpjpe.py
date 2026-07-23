# train_baseline_mpjpe.py
# =============================================================================
# WiFi Baseline 训练脚本 — MPJPE 优化版 (scheduler + best ckpt 按 MPJPE 选取)
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
    augment_wifi, normalize_pose3d, root_relative_pose_loss,
    root_position_loss, bone_length_loss, subject_robust_l1_loss,
    TemporalWiFiStudentCBAM,
    eval_temporal_model,
    load_config, get_loaders, get_test_loader, add_common_args
)

from torch.cuda.amp import autocast, GradScaler



def checkpoint_payload(model, args, source, extra=None):
    payload = {
        "model": model.state_dict(),
        "args": vars(args),
        "checkpoint_source": source,
    }
    if extra:
        payload.update(extra)
    return payload


def maybe_save_topk_checkpoint(args, run_dir, model, ep, mpjpe, pa, pck, topk_entries):
    k = int(getattr(args, "topk_ckpts", 0) or 0)
    if k <= 0 or not np.isfinite(mpjpe):
        return topk_entries

    if len(topk_entries) >= k:
        worst = max(item["mpjpe"] for item in topk_entries)
        if mpjpe >= worst:
            return topk_entries

    ckpt_name = f"topk_epoch_{ep:03d}.pth"
    ckpt_path = os.path.join(run_dir, ckpt_name)
    entry = {
        "checkpoint": ckpt_name,
        "epoch": int(ep),
        "mpjpe": float(mpjpe),
        "mpjpe_mm": float(mpjpe * 1000.0),
        "pa_mpjpe": float(pa),
        "pa_mpjpe_mm": float(pa * 1000.0),
        "pck@20": float(pck.get("pck@20", 0.0)),
        "pck@50": float(pck.get("pck@50", 0.0)),
    }
    torch.save(
        checkpoint_payload(
            model, args, "topk_regular",
            {"epoch": int(ep), "val_metric": entry}
        ),
        ckpt_path,
    )

    topk_entries.append(entry)
    topk_entries.sort(key=lambda item: item["mpjpe"])
    removed = []
    while len(topk_entries) > k:
        removed.append(topk_entries.pop())

    for item in removed:
        old_path = os.path.join(run_dir, item["checkpoint"])
        if old_path != ckpt_path and os.path.isfile(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass

    save_json({"topk": topk_entries}, os.path.join(run_dir, "topk_checkpoints.json"))
    print(f"  ☆ Top-{k} checkpoint kept: {ckpt_name} "
          f"(val MPJPE={mpjpe*1000:.1f}mm)")
    return topk_entries


def maybe_save_tail_checkpoint(args, run_dir, model, ep, mpjpe, pa, pck, tail_entries):
    k = int(getattr(args, "tail_ckpts", 0) or 0)
    if k <= 0 or not np.isfinite(mpjpe):
        return tail_entries

    ckpt_name = f"tail_epoch_{ep:03d}.pth"
    ckpt_path = os.path.join(run_dir, ckpt_name)
    entry = {
        "checkpoint": ckpt_name,
        "epoch": int(ep),
        "mpjpe": float(mpjpe),
        "mpjpe_mm": float(mpjpe * 1000.0),
        "pa_mpjpe": float(pa),
        "pa_mpjpe_mm": float(pa * 1000.0),
        "pck@20": float(pck.get("pck@20", 0.0)),
        "pck@50": float(pck.get("pck@50", 0.0)),
    }
    torch.save(
        checkpoint_payload(
            model, args, "tail_regular",
            {"epoch": int(ep), "val_metric": entry}
        ),
        ckpt_path,
    )

    tail_entries.append(entry)
    tail_entries.sort(key=lambda item: item["epoch"])
    removed = []
    while len(tail_entries) > k:
        removed.append(tail_entries.pop(0))

    for item in removed:
        old_path = os.path.join(run_dir, item["checkpoint"])
        if old_path != ckpt_path and os.path.isfile(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass

    save_json({"tail": tail_entries}, os.path.join(run_dir, "tail_checkpoints.json"))
    print(f"  ☆ Tail-{k} checkpoint kept: {ckpt_name} "
          f"(val MPJPE={mpjpe*1000:.1f}mm)")
    return tail_entries


def average_checkpoint_states(ckpt_paths):
    ckpts = [torch.load(path, map_location="cpu") for path in ckpt_paths]
    states = [ckpt["model"] for ckpt in ckpts]
    keys = list(states[0].keys())
    key_set = set(keys)
    for state in states[1:]:
        if set(state.keys()) != key_set:
            raise ValueError("Cannot soup checkpoints with different state_dict keys")

    soup_state = {}
    for key in keys:
        base = states[0][key]
        if torch.is_tensor(base) and base.is_floating_point():
            acc = base.detach().clone().float()
            for state in states[1:]:
                acc.add_(state[key].detach().float())
            acc.div_(len(states))
            soup_state[key] = acc.to(dtype=base.dtype)
        elif torch.is_tensor(base):
            soup_state[key] = base.detach().clone()
        else:
            soup_state[key] = base
    return soup_state


def build_topk_soup(args, run_dir, model, val_loader, device, topk_entries,
                    best_metric, best_path):
    if not getattr(args, "topk_soup", False):
        return best_metric, None

    ckpt_paths = [
        os.path.join(run_dir, item["checkpoint"])
        for item in topk_entries
        if os.path.isfile(os.path.join(run_dir, item["checkpoint"]))
    ]
    if len(ckpt_paths) < 2:
        print("  ℹ️  Top-k soup skipped; fewer than 2 top-k checkpoints exist.")
        return best_metric, None

    print(f"\n🍲 Top-k soup: averaging {len(ckpt_paths)} checkpoints...")
    soup_state = average_checkpoint_states(ckpt_paths)
    model.load_state_dict(soup_state, strict=True)
    soup_mpjpe, soup_pa, soup_pck = eval_temporal_model(
        model, val_loader, device,
        use_root_relative=args.use_root_relative)

    soup_metric = {
        "mpjpe": float(soup_mpjpe),
        "mpjpe_mm": float(soup_mpjpe * 1000.0),
        "pa_mpjpe": float(soup_pa),
        "pa_mpjpe_mm": float(soup_pa * 1000.0),
        "pck@20": float(soup_pck.get("pck@20", 0.0)),
        "pck@50": float(soup_pck.get("pck@50", 0.0)),
    }
    soup_payload = {
        "model": soup_state,
        "args": vars(args),
        "checkpoint_source": "topk_soup",
        "source_checkpoints": topk_entries,
        "val_metric": soup_metric,
    }
    soup_path = os.path.join(run_dir, "topk_soup.pth")
    torch.save(soup_payload, soup_path)
    save_json({
        "checkpoint": "topk_soup.pth",
        "source_checkpoints": topk_entries,
        "val_metric": soup_metric,
    }, os.path.join(run_dir, "topk_soup.json"))
    print(f"★ Top-k soup model: {fmt_metrics(soup_mpjpe, soup_pa, soup_pck)}")

    if getattr(args, "topk_soup_replace_best", False) and soup_mpjpe < best_metric:
        best_metric = soup_mpjpe
        torch.save(soup_payload, best_path)
        print("  ✅ Top-k soup is new best! Saved to best.pth and topk_soup.pth")
        return best_metric, "topk_soup"

    if soup_mpjpe < best_metric:
        print(f"  ℹ️  Top-k soup has better validation MPJPE, but best.pth keeps "
              f"the regular validation checkpoint. Use topk_soup.pth for "
              f"separate final-test diagnosis.")
        return best_metric, None

    print(f"  ℹ️  Top-k soup saved separately "
          f"(MPJPE={soup_mpjpe*1000:.1f}mm, "
          f"best={best_metric*1000:.1f}mm)")
    return best_metric, None


def build_tail_soup(args, run_dir, model, val_loader, device, tail_entries,
                    best_metric, best_path):
    if not getattr(args, "tail_soup", False):
        return best_metric, None

    ckpt_paths = [
        os.path.join(run_dir, item["checkpoint"])
        for item in tail_entries
        if os.path.isfile(os.path.join(run_dir, item["checkpoint"]))
    ]
    if len(ckpt_paths) < 2:
        print("  ℹ️  Tail soup skipped; fewer than 2 tail checkpoints exist.")
        return best_metric, None

    print(f"\n🍲 Tail soup: averaging {len(ckpt_paths)} final checkpoints...")
    soup_state = average_checkpoint_states(ckpt_paths)
    model.load_state_dict(soup_state, strict=True)
    soup_mpjpe, soup_pa, soup_pck = eval_temporal_model(
        model, val_loader, device,
        use_root_relative=args.use_root_relative)

    soup_metric = {
        "mpjpe": float(soup_mpjpe),
        "mpjpe_mm": float(soup_mpjpe * 1000.0),
        "pa_mpjpe": float(soup_pa),
        "pa_mpjpe_mm": float(soup_pa * 1000.0),
        "pck@20": float(soup_pck.get("pck@20", 0.0)),
        "pck@50": float(soup_pck.get("pck@50", 0.0)),
    }
    soup_payload = {
        "model": soup_state,
        "args": vars(args),
        "checkpoint_source": "tail_soup",
        "source_checkpoints": tail_entries,
        "val_metric": soup_metric,
    }
    soup_path = os.path.join(run_dir, "tail_soup.pth")
    torch.save(soup_payload, soup_path)
    save_json({
        "checkpoint": "tail_soup.pth",
        "source_checkpoints": tail_entries,
        "val_metric": soup_metric,
    }, os.path.join(run_dir, "tail_soup.json"))
    print(f"★ Tail soup model: {fmt_metrics(soup_mpjpe, soup_pa, soup_pck)}")

    if getattr(args, "tail_soup_replace_best", False) and soup_mpjpe < best_metric:
        best_metric = soup_mpjpe
        torch.save(soup_payload, best_path)
        print("  ✅ Tail soup is new best! Saved to best.pth and tail_soup.pth")
        return best_metric, "tail_soup"

    print("  ℹ️  Tail soup saved separately for final-test diagnosis.")
    return best_metric, None


def train_baseline(args, train_loader, val_loader, device):
    run_dir  = args.run_dir or make_run_dir("temporal_baseline_cbam")
    os.makedirs(run_dir, exist_ok=True)
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
    print(f"lambda_rel_pose: {args.lambda_rel_pose}, lambda_root: {args.lambda_root}")
    print(f"lambda_bone: {args.lambda_bone}, pose_head_type: {args.pose_head_type}")
    print(f"subject_robust_weight: {args.subject_robust_weight}")
    print(f"Top-k checkpoints: {args.topk_ckpts}, soup: {args.topk_soup}")
    print(f"Tail checkpoints: {args.tail_ckpts}, soup: {args.tail_soup}")
    print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience={args.lr_patience})")
    print(f"SWA start epoch: {args.swa_start}")
    print(f"Early stopping patience: {args.patience}")
    print(f"Root-relative normalization: {args.use_root_relative}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model = TemporalWiFiStudentCBAM(
        in_ch=3, d_model=args.d_model, window=args.window,
        use_cbam=args.use_cbam, dropout=args.dropout,
        pose_head_hidden=args.pose_head_hidden,
        num_layers=args.transformer_layers,
        dim_feedforward=args.dim_feedforward,
        pose_head_type=args.pose_head_type
    ).to(device)
    print(f"Parameters: {count_params(model):,}")

    opt = optim.AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=args.lr_patience, min_lr=1e-6)
    swa_model = None
    swa_started = False

    best_path = os.path.join(run_dir, "best.pth")
    swa_path = os.path.join(run_dir, "best_swa.pth")
    last_path = os.path.join(run_dir, "last.pth")
    hist_path = os.path.join(run_dir, "history.json")

    history, topk_entries, tail_entries = [], [], []
    best_metric, best_source, no_improve = 1e9, "none", 0

    for ep in range(1, args.epochs + 1):
        model.train()
        running = running_pose = running_rel = running_root = running_bone = 0.0

        if args.warmup_epochs > 0 and ep <= args.warmup_epochs:
            warm_lr = args.lr * ep / args.warmup_epochs
            for g in opt.param_groups:
                g["lr"] = warm_lr

        current_lr = opt.param_groups[0]["lr"]

        for i, batch in enumerate(train_loader, 1):
            wifi_window = batch["input_wifi-csi_window"].to(device)
            gt          = batch["output"].to(device)
            subjects    = batch.get("subject", [])

            if args.aug_noise > 0 or args.aug_freq_mask > 0 or args.aug_time_mask > 0:
                wifi_window = augment_wifi(
                    wifi_window, args.aug_noise,
                    args.aug_freq_mask, args.aug_time_mask)

            if args.use_root_relative:
                gt = normalize_pose3d(gt)

            subjects_for_loss = subjects
            if args.mixup_alpha > 0:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                idx = torch.randperm(wifi_window.size(0), device=device)
                wifi_window = lam * wifi_window + (1 - lam) * wifi_window[idx]
                gt          = lam * gt          + (1 - lam) * gt[idx]
                subjects_for_loss = []

            pred3d, _ = model(wifi_window)
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
            loss = (loss_pose
                    + args.lambda_rel_pose * loss_rel
                    + args.lambda_root * loss_root
                    + args.lambda_bone * loss_bone)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running      += loss.item()
            running_pose += loss_pose.item()
            running_rel  += loss_rel.item()
            running_root += loss_root.item()
            running_bone += loss_bone.item()

            if i % args.log_every == 0:
                print(f"  batch {i}/{len(train_loader)} "
                      f"loss={running/i:.4f} (pose={running_pose/i:.4f}, "
                      f"rel={running_rel/i:.4f}, root={running_root/i:.4f}, "
                      f"bone={running_bone/i:.4f}) "
                      f"lr={current_lr:.6f}")

        mpjpe, pa, pck = eval_temporal_model(
            model, val_loader, device,
            use_root_relative=args.use_root_relative)
        if ep > args.warmup_epochs:
            scheduler.step(mpjpe)
        print(f"Epoch {ep}: val {fmt_metrics(mpjpe, pa, pck)}")

        history.append({
            "epoch": ep, "mpjpe": mpjpe, "pa_mpjpe": pa,
            "pck@20": pck.get("pck@20", 0),
            "pck@50": pck.get("pck@50", 0),
            "lr": current_lr,
            "train_loss": running / len(train_loader),
            "train_pose": running_pose / len(train_loader),
            "train_rel": running_rel / len(train_loader),
            "train_root": running_root / len(train_loader),
            "train_bone": running_bone / len(train_loader),
        })
        torch.save(checkpoint_payload(model, args, "regular"), last_path)

        if args.swa_start > 0 and ep >= args.swa_start:
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(model)
            swa_model.update_parameters(model)
            swa_started = True

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
        print("\n🔄 SWA: updating BatchNorm statistics...")
        swa_model.train()
        with torch.no_grad():
            for batch in train_loader:
                wifi = batch["input_wifi-csi_window"].to(device)
                swa_model.module.forward(wifi)

        class SWAWiFiWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, w):
                return self.m.module.forward(w)

        swa_wrapper = SWAWiFiWrapper(swa_model).to(device)
        swa_mpjpe, swa_pa, swa_pck = eval_temporal_model(
            swa_wrapper, val_loader, device,
            use_root_relative=args.use_root_relative)
        print(f"★ SWA model: {fmt_metrics(swa_mpjpe, swa_pa, swa_pck)}")
        swa_payload = {
            "model": swa_model.module.state_dict(),
            "args": vars(args),
            "checkpoint_source": "swa",
        }
        torch.save(swa_payload, swa_path)
        if swa_mpjpe < best_metric:
            best_metric = swa_mpjpe
            best_source = "swa"
            torch.save(swa_payload, best_path)
            print("  ✅ SWA model is new best! Saved to best.pth and best_swa.pth")
        else:
            print(f"  ℹ️  SWA model saved separately "
                  f"(MPJPE={swa_mpjpe*1000:.1f}mm, "
                  f"best={best_metric*1000:.1f}mm)")

    print("=" * 60)
    print(f"✅ Baseline+CBAM finished! Best MPJPE: {best_metric*1000:.1f}mm")
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

    model = TemporalWiFiStudentCBAM(
        in_ch=3, d_model=args.d_model, window=args.window,
        use_cbam=args.use_cbam, dropout=args.dropout,
        pose_head_hidden=getattr(args, "pose_head_hidden", 512),
        num_layers=getattr(args, "transformer_layers", 2),
        dim_feedforward=getattr(args, "dim_feedforward", 1024),
        pose_head_type=getattr(args, "pose_head_type", "mlp")
    ).to(device)
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    mpjpe, pa, pck = eval_temporal_model(
        model, test_loader, device,
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
        description="WiFi Baseline Training (Temporal + CBAM)")
    add_common_args(parser)
    parser.add_argument("--lambda_rel_pose", type=float, default=0.0,
                        help="Auxiliary root-relative 3D pose loss weight")
    parser.add_argument("--lambda_root", type=float, default=0.0,
                        help="Auxiliary pelvis/root position loss weight")
    parser.add_argument("--lambda_bone", type=float, default=0.0,
                        help="Auxiliary bone-length consistency loss weight")
    parser.add_argument("--subject_robust_weight", type=float, default=0.0,
                        help="Blend average pose loss with worst-subject pose loss")
    parser.add_argument("--swa_start", type=int, default=0,
                        help="Epoch from which to start SWA averaging; 0 disables SWA")
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

    print(f"Protocol: {cfg.get('protocol', 'unknown_protocol')}")
    print(f"Split: {args.split} | Window: {args.window} | Stride: {args.stride}")
    heldout_cfg = cfg.get("heldout_split", {})
    print("Held-out split: "
          f"enabled={heldout_cfg.get('enabled', False)} | "
          f"unit={heldout_cfg.get('unit', 'window')} | "
          f"test_size={heldout_cfg.get('test_size', 0.5)} | "
          f"seed={heldout_cfg.get('random_seed', 41)}")
    print(f"Dropout: {args.dropout} | Mixup alpha: {args.mixup_alpha}")

    train_ds, val_ds, train_loader, val_loader = get_loaders(
        args.dataset_root, cfg, args.window, args.stride,
        args.batch_size, args.val_batch_size, num_workers=args.num_workers)
    print(f"Train samples (windowed): {len(train_ds)}")
    print(f"Val   samples (windowed): {len(val_ds)}")

    best_metric, run_dir = train_baseline(args, train_loader, val_loader, device)
    if args.split != "teacher_student_split" and not args.skip_final_test:
        test_ds, test_loader = get_test_loader(
            args.dataset_root, cfg, args.window, args.stride,
            args.val_batch_size, num_workers=args.num_workers)
        print(f"Test  samples (windowed): {len(test_ds)}")
        evaluate_best_on_test(args, run_dir, device, test_loader)
    elif args.skip_final_test:
        print("🧪 Final test skipped; evaluate best.pth in a separate process.")
    print(f"\n📊 Result: MPJPE = {best_metric*1000:.1f}mm")
    print(f"📁 Run dir: {run_dir}")


if __name__ == "__main__":
    main()
