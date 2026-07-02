"""
Compare final-test metrics for multiple checkpoints from the same run.

This is a lightweight diagnostic for strict four-way experiments. It evaluates
existing checkpoints such as best.pth, topk_soup.pth, tail_soup.pth,
best_swa.pth, and last.pth on the same final test loader, then writes a sidecar
JSON without overwriting final_test.json.
"""

import argparse
import os

import torch

from common import eval_temporal_model, fmt_metrics, get_test_loader, load_config, save_json
from evaluate_final_test_mpjpe import as_namespace, build_model


def evaluate_checkpoint(ckpt_path, model_type, train_args, test_loader, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    eval_model, load_target = build_model(model_type, train_args)
    load_target.load_state_dict(ckpt["model"], strict=True)
    eval_model = eval_model.to(device)

    mpjpe, pa, pck = eval_temporal_model(
        eval_model,
        test_loader,
        device,
        use_root_relative=getattr(train_args, "use_root_relative", False),
    )
    return {
        "checkpoint": os.path.basename(ckpt_path),
        "checkpoint_source": ckpt.get("checkpoint_source", "unknown"),
        "mpjpe": float(mpjpe),
        "pa_mpjpe": float(pa),
        "pck@20": float(pck.get("pck@20", 0.0)),
        "pck@50": float(pck.get("pck@50", 0.0)),
        "mpjpe_mm": float(mpjpe * 1000.0),
        "pa_mpjpe_mm": float(pa * 1000.0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare strict four-way final-test metrics across checkpoints."
    )
    parser.add_argument("dataset_root")
    parser.add_argument("config_file")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--model_type", choices=["lupi", "baseline"], default="baseline")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["best.pth", "topk_soup.pth", "tail_soup.pth", "best_swa.pth", "last.pth"],
    )
    parser.add_argument("--split", default="four_way_split")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_batch_size", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    existing = []
    for name in args.checkpoints:
        path = os.path.join(args.run_dir, name)
        if os.path.isfile(path):
            existing.append(path)
        else:
            print(f"Skip missing checkpoint: {name}")
    if not existing:
        raise FileNotFoundError(f"No requested checkpoints found in {args.run_dir}")

    first_ckpt = torch.load(existing[0], map_location="cpu")
    train_args = as_namespace(first_ckpt.get("args", {}), args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cfg = load_config(args.config_file, args.split)
    test_ds, test_loader = get_test_loader(
        args.dataset_root,
        cfg,
        train_args.window,
        train_args.stride,
        train_args.val_batch_size,
        num_workers=args.num_workers,
    )

    print(f"Device: {device}")
    print(f"Run dir: {args.run_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Test samples (windowed): {len(test_ds)}")
    print(f"num_workers: {args.num_workers}")

    results = []
    for ckpt_path in existing:
        item = evaluate_checkpoint(
            ckpt_path,
            args.model_type,
            train_args,
            test_loader,
            device,
        )
        results.append(item)
        print(
            f"{item['checkpoint']}: {fmt_metrics(item['mpjpe'], item['pa_mpjpe'], item)} "
            f"source={item['checkpoint_source']}"
        )

    results = sorted(results, key=lambda x: x["mpjpe"])
    output = {
        "run_dir": args.run_dir,
        "model_type": args.model_type,
        "n_test": int(len(test_ds)),
        "results": results,
    }
    out_path = args.output or os.path.join(args.run_dir, "final_test_checkpoints.json")
    save_json(output, out_path)

    best = results[0]
    print(
        f"\nBest checkpoint: {best['checkpoint']} "
        f"MPJPE={best['mpjpe_mm']:.2f}mm PA={best['pa_mpjpe_mm']:.2f}mm"
    )
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
