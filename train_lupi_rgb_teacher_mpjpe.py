# train_lupi_rgb_teacher_mpjpe.py
# =============================================================================
# Clean S2P2 WiFi + RGB-privileged CMC training script.
#
# Main line only:
#   WiFi temporal pose loss + RGB feature CMC + Mixup.
# Kept as the only small CMC-side improvement:
#   adaptive CMC.
# Removed from this main script:
#   pose distillation, graph/root/bone heads/losses, subject-robust loss,
#   SWA, checkpoint soup, temporal RGB teacher.
# =============================================================================

import argparse
import os
import random
import sys
from datetime import datetime

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import (
    Logger,
    RGBOnlyTeacher,
    TemporalLUPICMC_CBAM,
    augment_wifi,
    count_params,
    eval_temporal_model,
    fmt_metrics,
    get_loaders,
    get_test_loader,
    info_nce_loss,
    load_config,
    make_run_dir,
    mixup_data,
    save_json,
)


def load_trusted_checkpoint(path):
    """Load a checkpoint written by this local training pipeline."""
    try:
        # PyTorch 2.6 defaults to weights_only=True. Resumable checkpoints also
        # contain optimizer and NumPy RNG state, so they need full loading.
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Compatibility with older PyTorch versions without weights_only.
        return torch.load(path, map_location="cpu")


def checkpoint_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    if not isinstance(state, dict):
        return False
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except (TypeError, ValueError, RuntimeError):
        return False
    return True


def checkpoint_payload(model, args, source, extra=None):
    payload = {
        "model": model.state_dict(),
        "args": vars(args),
        "checkpoint_source": source,
    }
    if extra:
        payload.update(extra)
    return payload


class WiFiOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, wifi_window):
        return self.model.forward_wifi(wifi_window)


def load_rgb_teacher(teacher_ckpt, args, device):
    if teacher_ckpt is None or not os.path.isfile(teacher_ckpt):
        return None

    teacher = RGBOnlyTeacher(
        d_model=args.d_model,
        dropout=args.dropout,
        temporal_rgb=False,
    ).to(device)
    ckpt = load_trusted_checkpoint(teacher_ckpt)
    state = ckpt.get("model", ckpt)
    teacher.load_state_dict(state, strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def cmc_weight_for_epoch(args, epoch):
    if epoch < args.cmc_start_epoch:
        return 0.0
    if args.cmc_ramp_epochs > 0:
        progress = (epoch - args.cmc_start_epoch + 1) / args.cmc_ramp_epochs
        return args.lambda_cmc * min(1.0, max(0.0, progress))
    return args.lambda_cmc


def train_lupi_rgb_teacher(args, train_loader, val_loader, device, teacher_ckpt=None):
    run_dir = args.run_dir or make_run_dir("s2p2_wifi_cmc_mpjpe")
    os.makedirs(run_dir, exist_ok=True)
    logger = Logger(os.path.join(run_dir, "log.txt"))
    sys.stdout = logger

    print("=" * 60)
    print("Training: WiFi + RGB-privileged CMC")
    print(f"Run dir: {run_dir}")
    print(f"Protocol: {args.protocol}")
    print(f"Split: {args.split}")
    print(f"Window: {args.window}, Stride: {args.stride}, CBAM: {args.use_cbam}")
    print(f"Dropout: {args.dropout}, Weight decay: {args.weight_decay}")
    print(f"Transformer layers: {args.transformer_layers}, dim_feedforward: {args.dim_feedforward}")
    print(f"lambda_cmc: {args.lambda_cmc}, tau: {args.cmc_tau}, proj_dim: {args.cmc_proj_dim}")
    print(f"cmc_start_epoch: {args.cmc_start_epoch}, cmc_ramp_epochs: {args.cmc_ramp_epochs}")
    print(f"cmc_encoder_grad_scale: {args.cmc_encoder_grad_scale}")
    print(f"Adaptive CMC: {args.adaptive_cmc} "
          f"(factor={args.adaptive_cmc_factor}, min_scale={args.adaptive_cmc_min_scale})")
    print(f"Mixup alpha: {args.mixup_alpha}")
    print(f"Augmentation: noise={args.aug_noise}, freq_mask={args.aug_freq_mask}, time_mask={args.aug_time_mask}")
    print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience={args.lr_patience}, min_lr={args.min_lr})")
    print(f"Early stopping patience: {args.patience}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model = TemporalLUPICMC_CBAM(
        in_ch=3,
        d_model=args.d_model,
        d_proj=args.cmc_proj_dim,
        window=args.window,
        use_cbam=args.use_cbam,
        dropout=args.dropout,
        num_layers=args.transformer_layers,
        dim_feedforward=args.dim_feedforward,
        pose_head_hidden=args.pose_head_hidden,
    ).to(device)

    teacher = load_rgb_teacher(teacher_ckpt, args, device)
    if teacher is None:
        print("No RGB teacher checkpoint loaded. CMC loss will be disabled.")
    else:
        print(f"Loaded RGB teacher for CMC features from: {teacher_ckpt}")

    print(f"Parameters: {count_params(model):,}")

    train_params = [p for p in model.parameters() if p.requires_grad]
    print("Initializing optimizer...")
    optimizer = optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    best_path = os.path.join(run_dir, "best.pth")
    last_path = os.path.join(run_dir, "last.pth")
    hist_path = os.path.join(run_dir, "history.json")

    history = []
    best_metric = 1e9
    no_improve = 0
    cmc_adapt_scale = 1.0
    start_epoch = 1
    wifi_only = WiFiOnlyWrapper(model).to(device)

    if args.resume_last and os.path.isfile(last_path):
        ckpt = load_trusted_checkpoint(last_path)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        history = ckpt.get("history", [])
        if not history and os.path.isfile(hist_path):
            try:
                import json
                with open(hist_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except (OSError, ValueError):
                history = []
        best_metric = float(ckpt.get(
            "best_metric",
            min((row.get("mpjpe", 1e9) for row in history), default=1e9),
        ))
        no_improve = int(ckpt.get("no_improve", 0))
        cmc_adapt_scale = float(ckpt.get("cmc_adapt_scale", 1.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        restored_rng = restore_rng_state(ckpt.get("rng_state"))
        print(f"Resuming from last.pth at epoch {start_epoch}/{args.epochs}")
        if not restored_rng:
            print("Resume checkpoint has no RNG state; continuing from its saved model state.")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        running_pose = 0.0
        running_cmc = 0.0

        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            warm_lr = args.lr * epoch / args.warmup_epochs
            for group in optimizer.param_groups:
                group["lr"] = warm_lr

        current_lr = optimizer.param_groups[0]["lr"]
        cmc_weight = cmc_weight_for_epoch(args, epoch) * cmc_adapt_scale
        print(
            f"Epoch {epoch} train start: "
            f"batches={len(train_loader)} cmc_w={cmc_weight:.4f} lr={current_lr:.6f}"
        )

        for step, batch in enumerate(train_loader, 1):
            wifi_window = batch["input_wifi-csi_window"].to(device)
            rgb = batch["input_rgb"].to(device)
            gt = batch["output"].to(device)

            if args.aug_noise > 0 or args.aug_freq_mask > 0 or args.aug_time_mask > 0:
                wifi_window = augment_wifi(
                    wifi_window,
                    args.aug_noise,
                    args.aug_freq_mask,
                    args.aug_time_mask,
                )

            lam = 1.0
            if args.mixup_alpha > 0:
                wifi_window, rgb, gt, lam, _ = mixup_data(
                    wifi_window,
                    rgb,
                    gt,
                    alpha=args.mixup_alpha,
                )

            pred3d, feat_w = model.forward_wifi(wifi_window)
            loss_pose = F.l1_loss(pred3d, gt)

            loss_cmc = torch.tensor(0.0, device=device)
            if cmc_weight > 0 and teacher is not None:
                with torch.no_grad():
                    _, feat_rgb = teacher(None, rgb, return_feat=True)
                if args.cmc_encoder_grad_scale < 1.0:
                    feat_w_for_cmc = (
                        feat_w.detach()
                        + args.cmc_encoder_grad_scale * (feat_w - feat_w.detach())
                    )
                else:
                    feat_w_for_cmc = feat_w
                z_w = model.proj_w(feat_w_for_cmc)
                z_v = model.proj_v(feat_rgb)
                loss_cmc = info_nce_loss(z_w, z_v, tau=args.cmc_tau)

            loss = loss_pose + cmc_weight * loss_cmc

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, max_norm=1.0)
            optimizer.step()

            running += loss.item()
            running_pose += loss_pose.item()
            running_cmc += loss_cmc.item()

            if step % args.log_every == 0 or step == len(train_loader):
                print(
                    f"  batch {step}/{len(train_loader)} "
                    f"loss={running/step:.4f} "
                    f"pose={running_pose/step:.4f} "
                    f"cmc={running_cmc/step:.4f} "
                    f"cmc_w={cmc_weight:.4f} "
                    f"lam={lam:.3f} "
                    f"lr={current_lr:.6f}"
                )

        mpjpe, pa, pck = eval_temporal_model(wifi_only, val_loader, device)
        if epoch > args.warmup_epochs:
            scheduler.step(mpjpe)
        print(f"Epoch {epoch}: val(WiFi-only) {fmt_metrics(mpjpe, pa, pck)}")

        row = {
            "epoch": epoch,
            "mpjpe": mpjpe,
            "pa_mpjpe": pa,
            "pck@20": pck.get("pck@20", 0.0),
            "pck@50": pck.get("pck@50", 0.0),
            "lr": current_lr,
            "cmc_weight": cmc_weight,
            "cmc_adapt_scale": cmc_adapt_scale,
            "train_loss": running / len(train_loader),
            "train_pose": running_pose / len(train_loader),
            "train_cmc": running_cmc / len(train_loader),
        }
        history.append(row)

        if mpjpe < best_metric:
            best_metric = mpjpe
            no_improve = 0
            torch.save(
                checkpoint_payload(model, args, "best", {"epoch": epoch, "val_metric": row}),
                best_path,
            )
            print(
                f"  New best! MPJPE={mpjpe*1000:.1f}mm | "
                f"PCK@20={pck.get('pck@20', 0.0):.1f}% | "
                f"PCK@50={pck.get('pck@50', 0.0):.1f}%"
            )
        else:
            no_improve += 1
            if args.adaptive_cmc and args.lambda_cmc > 0:
                old_scale = cmc_adapt_scale
                cmc_adapt_scale = max(
                    args.adaptive_cmc_min_scale,
                    cmc_adapt_scale * args.adaptive_cmc_factor,
                )
                if cmc_adapt_scale < old_scale:
                    print(f"  Adaptive CMC scale: {old_scale:.3f} -> {cmc_adapt_scale:.3f}")

        save_json(history, hist_path)
        torch.save(
            checkpoint_payload(
                model,
                args,
                "last",
                {
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                    "best_metric": best_metric,
                    "no_improve": no_improve,
                    "cmc_adapt_scale": cmc_adapt_scale,
                    "rng_state": checkpoint_rng_state(),
                },
            ),
            last_path,
        )

        if args.patience > 0 and no_improve >= args.patience:
            print(f"  Early stopping: no improvement for {args.patience} epochs")
            break

    print("=" * 60)
    print(f"WiFi + CMC finished. Best MPJPE: {best_metric*1000:.1f}mm")
    print("=" * 60)
    save_json(
        {
            "status": "complete",
            "best_val_mpjpe": best_metric,
            "checkpoint": "best.pth",
            "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        os.path.join(run_dir, "train_complete.json"),
    )

    sys.stdout = logger.terminal
    logger.close()
    return best_metric, run_dir


def evaluate_best_on_eval(args, run_dir, device, eval_loader):
    best_path = os.path.join(run_dir, "best.pth")
    if not os.path.isfile(best_path):
        print("No best checkpoint found for evaluation.")
        return None

    model = TemporalLUPICMC_CBAM(
        in_ch=3,
        d_model=args.d_model,
        d_proj=args.cmc_proj_dim,
        window=args.window,
        use_cbam=args.use_cbam,
        dropout=args.dropout,
        num_layers=args.transformer_layers,
        dim_feedforward=args.dim_feedforward,
        pose_head_hidden=args.pose_head_hidden,
    ).to(device)
    ckpt = load_trusted_checkpoint(best_path)
    model.load_state_dict(ckpt["model"], strict=True)
    wifi_only = WiFiOnlyWrapper(model).to(device)

    mpjpe, pa, pck = eval_temporal_model(wifi_only, eval_loader, device)
    result = {
        "mpjpe": mpjpe,
        "pa_mpjpe": pa,
        "pck@20": pck.get("pck@20", 0.0),
        "pck@50": pck.get("pck@50", 0.0),
        "checkpoint": os.path.basename(best_path),
        "checkpoint_source": ckpt.get("checkpoint_source", "unknown"),
    }
    save_json(result, os.path.join(run_dir, "final_test.json"))
    print(f"\nEvaluation: {fmt_metrics(mpjpe, pa, pck)}")
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Clean WiFi + RGB-privileged CMC training")
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_num_workers", type=int, default=None)

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--use_cbam", action="store_true", default=True)
    parser.add_argument("--no_cbam", action="store_false", dest="use_cbam")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--pose_head_hidden", type=int, default=512)

    parser.add_argument("--aug_noise", type=float, default=0.08)
    parser.add_argument("--aug_freq_mask", type=float, default=0.15)
    parser.add_argument("--aug_time_mask", type=float, default=0.15)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument(
        "--split",
        type=str,
        default="cross_subject_split",
        choices=["random_split", "cross_subject_split", "cross_scene_split"],
    )
    parser.add_argument("--teacher_ckpt", type=str, default=None)
    parser.add_argument("--lambda_cmc", type=float, default=0.2)
    parser.add_argument("--cmc_tau", type=float, default=0.1)
    parser.add_argument("--cmc_proj_dim", type=int, default=128)
    parser.add_argument("--cmc_start_epoch", type=int, default=1)
    parser.add_argument("--cmc_ramp_epochs", type=int, default=0)
    parser.add_argument("--cmc_encoder_grad_scale", type=float, default=1.0)
    parser.add_argument("--adaptive_cmc", action="store_true")
    parser.add_argument("--adaptive_cmc_factor", type=float, default=0.85)
    parser.add_argument("--adaptive_cmc_min_scale", type=float, default=0.35)
    parser.add_argument("--skip_final_test", action="store_true")
    parser.add_argument("--resume_last", action="store_true")
    parser.add_argument("--run_dir", type=str, default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    seed = int(os.environ.get("TRAIN_SEED", 0))
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(args.config_file, args.split)
    args.protocol = cfg.get("protocol", "unknown_protocol")
    print(f"Protocol: {args.protocol}")
    print(f"Split: {args.split} | Window: {args.window} | Stride: {args.stride}")
    heldout_cfg = cfg.get("heldout_split", {})
    print(
        "Held-out split: "
        f"enabled={heldout_cfg.get('enabled', False)} | "
        f"unit={heldout_cfg.get('unit', 'window')} | "
        f"test_size={heldout_cfg.get('test_size', 0.5)} | "
        f"seed={heldout_cfg.get('random_seed', 41)}"
    )
    print(f"Dropout: {args.dropout} | Mixup alpha: {args.mixup_alpha}")
    print(f"lambda_cmc: {args.lambda_cmc}")

    train_ds, val_ds, train_loader, val_loader = get_loaders(
        args.dataset_root,
        cfg,
        args.window,
        args.stride,
        args.batch_size,
        args.val_batch_size,
        role="student",
        num_workers=args.num_workers,
        eval_num_workers=args.eval_num_workers,
        include_rgb_window=False,
    )
    print(f"Train samples (windowed): {len(train_ds)}")
    print(f"Val   samples (windowed): {len(val_ds)}")

    best_metric, run_dir = train_lupi_rgb_teacher(
        args,
        train_loader,
        val_loader,
        device,
        teacher_ckpt=args.teacher_ckpt,
    )

    final_eval = None
    if not args.skip_final_test:
        eval_ds, eval_loader = get_test_loader(
            args.dataset_root,
            cfg,
            args.window,
            args.stride,
            args.val_batch_size,
            num_workers=(args.num_workers if args.eval_num_workers is None else args.eval_num_workers),
            include_rgb_window=False,
        )
        print(f"Eval samples (windowed): {len(eval_ds)}")
        final_eval = evaluate_best_on_eval(args, run_dir, device, eval_loader)
    else:
        print("Final evaluation skipped; evaluate best.pth separately if needed.")

    print(f"\nBest val MPJPE = {best_metric*1000:.1f}mm")
    if final_eval is not None:
        print(f"Final eval MPJPE = {final_eval['mpjpe']*1000:.1f}mm")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
