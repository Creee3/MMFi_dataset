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
    augment_wifi, RGBOnlyTeacher, eval_teacher_model,
    root_relative_pose_loss, root_position_loss, bone_length_loss,
    load_config, get_loaders, get_test_loader
)


def dropout_rgb_history(rgb_window, drop_prob):
    """Randomly replace historical RGB frames with the current frame."""
    if drop_prob <= 0 or rgb_window.dim() != 4:
        return rgb_window
    B, T = rgb_window.shape[:2]
    if T <= 1:
        return rgb_window
    keep_current = rgb_window[:, -1:, :, :]
    mask = torch.rand(B, T, 1, 1, device=rgb_window.device) < drop_prob
    mask[:, -1] = False
    return torch.where(mask, keep_current.expand_as(rgb_window), rgb_window)


def default_teacher_run_dir(args):
    rgb_mode = "temporal" if args.temporal_rgb else "single"
    protocol = getattr(args, "protocol", "unknown_protocol")
    mode = f"T1_rgb_teacher_{rgb_mode}_{args.split}_{protocol}_mpjpe"
    return make_run_dir(mode)


def train_t1_rgb_teacher_mpjpe(args, train_loader, val_loader, device):
    run_dir = args.run_dir or default_teacher_run_dir(args)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "log.txt")
    logger = Logger(log_path)
    sys.stdout = logger

    print("=" * 60)
    print("Training: T1 RGB-only Teacher (MPJPE optimized)")
    print(f"Run dir: {run_dir}")
    print(f"Split: {args.split}")
    print(f"Window: {args.window}, Stride: {args.stride}")
    print(f"Dropout: {args.dropout}, Weight decay: {args.weight_decay}")
    print(f"Augmentation: noise={args.aug_noise}, freq_mask={args.aug_freq_mask}, "
          f"time_mask={args.aug_time_mask}")
    print(f"Mixup alpha: {args.mixup_alpha}")
    print(f"RGB history dropout: {args.rgb_history_dropout}")
    print(f"Aux losses: rel={args.lambda_rel_pose}, root={args.lambda_root}, "
          f"bone={args.lambda_bone}")
    print(f"Scheduler: ReduceLROnPlateau (factor=0.5, patience={args.lr_patience})")
    print(f"Early stopping patience: {args.patience}")
    print(f"Root-relative normalization: {args.use_root_relative}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model = RGBOnlyTeacher(
        d_model=args.d_model,
        dropout=args.dropout,
        temporal_rgb=args.temporal_rgb,
        window=args.window,
    ).to(device)
    print(f"Parameters: {count_params(model):,}")
    print(f"Temporal RGB teacher: {args.temporal_rgb}")

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=args.lr_patience, min_lr=1e-6
    )

    best_path = os.path.join(run_dir, "best.pth")
    last_path = os.path.join(run_dir, "last.pth")
    hist_path = os.path.join(run_dir, "history.json")

    history = []
    best_metric = 1e9
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        running_pose = running_rel = running_root = running_bone = 0.0

        if args.warmup_epochs > 0 and ep <= args.warmup_epochs:
            warm_lr = args.lr * ep / args.warmup_epochs
            for g in opt.param_groups:
                g["lr"] = warm_lr

        current_lr = opt.param_groups[0]["lr"]

        for i, batch in enumerate(train_loader, 1):
            wifi_window = batch["input_wifi-csi_window"].to(device)
            rgb_key = "input_rgb_window" if args.temporal_rgb else "input_rgb"
            rgb = batch[rgb_key].to(device)
            gt = batch["output"].to(device)
            if args.temporal_rgb and args.rgb_history_dropout > 0:
                rgb = dropout_rgb_history(rgb, args.rgb_history_dropout)

            if args.aug_noise > 0 or args.aug_freq_mask > 0 or args.aug_time_mask > 0:
                wifi_window = augment_wifi(
                    wifi_window, args.aug_noise,
                    args.aug_freq_mask, args.aug_time_mask
                )

            lam = None
            idx = None
            if args.mixup_alpha > 0:
                lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
                idx = torch.randperm(wifi_window.size(0), device=device)
                wifi_window = lam * wifi_window + (1 - lam) * wifi_window[idx]
                rgb = lam * rgb + (1 - lam) * rgb[idx]
                gt = lam * gt + (1 - lam) * gt[idx]

            wifi_cur = wifi_window[:, -1]
            pred3d, _ = model(wifi_cur, rgb, return_feat=True)
            loss_pose = F.l1_loss(pred3d, gt)
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
            running += loss.item()
            running_pose += loss_pose.item()
            running_rel += loss_rel.item()
            running_root += loss_root.item()
            running_bone += loss_bone.item()

            if i % args.log_every == 0:
                lam_str = f"{lam:.3f}" if lam is not None else "1.000"
                print(f"  batch {i}/{len(train_loader)} "
                      f"loss={running/i:.4f} pose={running_pose/i:.4f} "
                      f"rel={running_rel/i:.4f} root={running_root/i:.4f} "
                      f"bone={running_bone/i:.4f} "
                      f"lam={lam_str} lr={current_lr:.6f}")

        mpjpe, pa, pck = eval_teacher_model(
            model, val_loader, device, use_root_relative=args.use_root_relative
        )
        if ep > args.warmup_epochs:
            scheduler.step(mpjpe)
        print(f"Epoch {ep}: val {fmt_metrics(mpjpe, pa, pck)}")

        history.append({
            "epoch": ep,
            "train_loss": running / len(train_loader),
            "train_pose": running_pose / len(train_loader),
            "train_rel": running_rel / len(train_loader),
            "train_root": running_root / len(train_loader),
            "train_bone": running_bone / len(train_loader),
            "mpjpe": mpjpe,
            "pa_mpjpe": pa,
            "pck@20": pck.get("pck@20", 0),
            "pck@50": pck.get("pck@50", 0),
            "lr": current_lr,
        })
        torch.save({"model": model.state_dict(), "args": vars(args)}, last_path)

        if mpjpe < best_metric:
            best_metric = mpjpe
            no_improve = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            print(f"  New best! MPJPE={mpjpe*1000:.1f}mm | "
                  f"PCK@20={pck.get('pck@20', 0):.1f}% | "
                  f"PCK@50={pck.get('pck@50', 0):.1f}%")
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"  Early stopping! No improvement for {args.patience} epochs")
                break

        save_json(history, hist_path)

    print("=" * 60)
    print(f"T1 RGB-only Teacher finished! Best MPJPE: {best_metric*1000:.1f}mm")
    print("=" * 60)
    sys.stdout = logger.terminal
    logger.close()
    return best_metric, run_dir


def evaluate_best_on_test(args, run_dir, device, test_loader):
    best_path = os.path.join(run_dir, "best.pth")
    if not os.path.isfile(best_path):
        print("No best checkpoint found for final test evaluation.")
        return None

    model = RGBOnlyTeacher(
        d_model=args.d_model,
        dropout=args.dropout,
        temporal_rgb=args.temporal_rgb,
        window=args.window,
    ).to(device)
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    mpjpe, pa, pck = eval_teacher_model(
        model, test_loader, device, use_root_relative=args.use_root_relative
    )
    test_result = {
        "mpjpe": mpjpe,
        "pa_mpjpe": pa,
        "pck@20": pck.get("pck@20", 0.0),
        "pck@50": pck.get("pck@50", 0.0),
        "checkpoint": os.path.basename(best_path),
    }
    save_json(test_result, os.path.join(run_dir, "final_test.json"))
    print(f"\nFinal test: {fmt_metrics(mpjpe, pa, pck)}")
    return test_result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train T1 RGB-only Teacher with MPJPE selection"
    )
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("config_file", type=str)
    parser.add_argument("--split", type=str, default="cross_subject_split")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--lr_patience", type=int, default=3)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--aug_noise", type=float, default=0.10)
    parser.add_argument("--aug_freq_mask", type=float, default=0.20)
    parser.add_argument("--aug_time_mask", type=float, default=0.20)
    parser.add_argument("--mixup_alpha", type=float, default=0.6)
    parser.add_argument("--rgb_history_dropout", type=float, default=0.0,
                        help="For --temporal_rgb, randomly replace historical RGB frames with the current frame.")
    parser.add_argument("--lambda_rel_pose", type=float, default=0.0,
                        help="Auxiliary root-relative pose loss weight.")
    parser.add_argument("--lambda_root", type=float, default=0.0,
                        help="Auxiliary pelvis/root position loss weight.")
    parser.add_argument("--lambda_bone", type=float, default=0.0,
                        help="Auxiliary bone-length consistency loss weight.")
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--skip_final_test", action="store_true")
    parser.add_argument("--temporal_rgb", action="store_true",
                        help="Use a temporal RGB 2D-pose sequence teacher instead of single-frame RGB.")
    parser.add_argument("--use_root_relative", action="store_true")
    parser.add_argument("--no_root_relative", dest="use_root_relative", action="store_false")
    parser.set_defaults(use_root_relative=False)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    seed = int(os.environ.get("TRAIN_SEED", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = load_config(args.config_file, args.split)
    args.protocol = cfg.get("protocol", "unknown_protocol")
    heldout_cfg = cfg.get("heldout_split", {})
    print(f"Protocol: {args.protocol}")
    print(f"Split: {args.split} | Window: {args.window} | Stride: {args.stride}")
    print("Held-out split: "
          f"enabled={heldout_cfg.get('enabled', False)} | "
          f"unit={heldout_cfg.get('unit', 'window')} | "
          f"test_size={heldout_cfg.get('test_size', 0.5)} | "
          f"seed={heldout_cfg.get('random_seed', 41)}")
    train_ds, val_ds, train_loader, val_loader = get_loaders(
        args.dataset_root, cfg, args.window, args.stride,
        args.batch_size, args.val_batch_size, role="teacher",
        num_workers=args.num_workers,
        include_rgb_window=args.temporal_rgb,
    )
    print(f"Train samples (windowed): {len(train_ds)}")
    print(f"Val   samples (windowed): {len(val_ds)}")

    best_metric, run_dir = train_t1_rgb_teacher_mpjpe(
        args, train_loader, val_loader, device
    )
    final_test = None
    if args.split != "teacher_student_split" and not args.skip_final_test:
        test_ds, test_loader = get_test_loader(
            args.dataset_root, cfg, args.window, args.stride,
            args.val_batch_size, num_workers=args.num_workers,
            include_rgb_window=args.temporal_rgb,
        )
        print(f"Test  samples (windowed): {len(test_ds)}")
        final_test = evaluate_best_on_test(args, run_dir, device, test_loader)
    elif args.skip_final_test:
        print("Final test skipped; evaluate best.pth in a separate process.")
    print(f"\nResult: MPJPE = {best_metric*1000:.1f}mm")
    if final_test is not None:
        print(f"Final test MPJPE = {final_test['mpjpe']*1000:.1f}mm")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
