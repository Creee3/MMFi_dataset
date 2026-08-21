"""
train_pose_wrapper.py

统一适配器：将 MetaFi / HPELI / DT-Pose 三个模型接入贝叶斯超参数调优调度器。

功能：
  - 接受命令行参数 --lr / --batch_size / --model_name 覆盖 YAML 配置
  - 训练结束后输出 history.json（格式与 run_bayesian_hparam_tuning.py 兼容）
  - 打印 "Run dir: <path>" 让调度器自动定位结果目录

用法（调度器会自动调用，也可以手动测试）：
  python train_pose_wrapper.py <config_file> \
      --model_name dtpose \
      --lr 3e-4 \
      --batch_size 128

支持的 model_name:
  metafi   → MetaFi（ResNet34 + ChannelTransformer，从头训练）
  hpeli    → HPELI（SKUnit CNN，从头训练）
  dtpose   → DT-Pose（MAE 预训练 + ViT_Pose_Decoder 微调）
"""

import os
import sys
import json
import argparse
import copy
from datetime import datetime

import yaml
import numpy as np
import torch

# ── 项目内部导入（路径根据你的项目结构调整）──────────────────
from feeder.mmfi import make_dataset, make_dataloader
from model.model import MAE_ViT, ViT_Pose_Decoder
from model.metafi.mynetwork import metafinet, metafi_weights_init
from model.hpeli.hpeli import hpelinet, hpeli_weights_init
from utils import setup_seed, calulate_error, compute_pck_pckh


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────
def make_run_dir(model_name: str) -> str:
    root = "strict_offline_runs"
    os.makedirs(root, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root, f"dtpose_{model_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(obj, path: str):
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.floating, float)):
                return float(o)
            if isinstance(o, (np.integer, int)):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, cls=NpEncoder)


def build_model(model_name: str, config: dict, device: str, pretrained_path=None):
    """根据 model_name 构建模型和优化器。"""
    dataset = config["dataset_name"]
    num_person = config.get("num_person", 1)

    if model_name == "metafi":
        if dataset == "mmfi-csi":
            model = metafinet(num_keypoints=17, num_coor=3,
                              num_person=num_person, dataset=dataset).to(device)
        else:
            raise ValueError(f"MetaFi does not support dataset: {dataset}")
        model.apply(metafi_weights_init)
        return model

    elif model_name == "hpeli":
        if dataset == "mmfi-csi":
            model = hpelinet(num_keypoints=17, num_coor=3, subcarrier_num=114,
                             num_person=num_person, dataset=dataset).to(device)
        else:
            raise ValueError(f"HPELI does not support dataset: {dataset}")
        model.apply(hpeli_weights_init)
        return model

    elif model_name == "dtpose":
        if pretrained_path is not None and os.path.isfile(pretrained_path):
            print(f"  ✅ Loading pretrained weights: {pretrained_path}")
            pretrained = torch.load(pretrained_path, map_location="cpu",
                                    weights_only=False)
            encoder = pretrained.encoder
        else:
            print("  ⚠️  No pretrained weights, using random encoder")
            encoder = MAE_ViT(
                image_size=(114, 10), patch_size=(2, 2),
                encoder_layer=4, encoder_head=4,
                decoder_layer=2, decoder_head=4, emb_dim=256
            ).encoder

        model = ViT_Pose_Decoder(
            encoder, keypoints=17, coor_num=3,
            token_num=285, dataset=dataset
        ).to(device)
        return model

    else:
        raise ValueError(f"Unknown model_name: {model_name}. "
                         f"Choose from: metafi, hpeli, dtpose")


def build_optimizer(model_name: str, model, lr: float):
    """根据模型类型选择合适的优化器（与原版 train_pose.py 保持一致）。
    
    原版对应关系（mmfi-csi）：
      metafi  → SGD, momentum=0.9（从头训练）
      hpeli   → SGD, momentum=0.9（从头训练）
      dtpose  → SGD, weight_decay=0.01（预训练微调，原版无 momentum）
    """
    if model_name == "metafi":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif model_name == "hpeli":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif model_name == "dtpose":
        trainable = filter(lambda p: p.requires_grad, model.parameters())
        return torch.optim.SGD(trainable, lr=lr, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")


# ─────────────────────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────────────────────
def train(config: dict, model_name: str, lr: float, batch_size: int):
    run_dir = make_run_dir(model_name)
    print(f"Run dir: {run_dir}")   # ← 调度器靠这行定位结果目录

    hist_path = os.path.join(run_dir, "history.json")
    best_path = os.path.join(run_dir, "best.pt")

    seed = int(os.environ.get("TRAIN_SEED", config.get("seed", 42)))
    setup_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 数据加载 ─────────────────────────────────────────────
    dataset_name = config["dataset_name"]
    load_batch_size = min(config.get("max_device_batch_size", batch_size), batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    rng_generator = torch.manual_seed(seed)

    if dataset_name == "mmfi-csi":
        train_dataset, val_dataset = make_dataset(
            config.get("training_semi", False),
            config["dataset_root"], config
        )
        train_loader = make_dataloader(
            train_dataset, is_training=True,
            generator=rng_generator, batch_size=load_batch_size
        )
        val_loader = make_dataloader(
            val_dataset, is_training=False,
            generator=rng_generator, batch_size=load_batch_size
        )
    else:
        raise NotImplementedError(
            f"Dataset {dataset_name} not yet supported in wrapper. "
            f"Add it following the mmfi-csi pattern."
        )

    # ── 构建模型 & 优化器 ─────────────────────────────────────
    pretrained_path = config.get("pretrained_model_path", None)
    model = build_model(model_name, config, device, pretrained_path)
    optim = build_optimizer(model_name, model, lr)

    print(f"\n{'=' * 60}")
    print(f"🚀 Training: {model_name.upper()}")
    print(f"   Run dir:    {run_dir}")
    print(f"   lr:         {lr}")
    print(f"   batch_size: {batch_size}")
    print(f"   epochs:     {config['total_epoch']}")
    print(f"   device:     {device}")
    print(f"{'=' * 60}\n")

    # ── 训练循环 ─────────────────────────────────────────────
    history = []
    best_pampjpe = float("inf")
    step_count = 0
    optim.zero_grad()

    for epoch in range(config["total_epoch"]):
        model.train()
        train_losses = []
        step_count = 0

        for batch_data in train_loader:
            step_count += 1
            csi_data  = batch_data["input_wifi-csi"].to(device)
            pose_gt   = batch_data["output"].to(device)

            predicted_pose, _ = model(csi_data)
            loss = torch.mean(torch.norm(predicted_pose - pose_gt, dim=-1))
            loss.backward()

            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # ── 验证 ─────────────────────────────────────────────
        model.eval()
        mpjpe_list, pampjpe_list = [], []
        pck_iter = [[] for _ in range(5)]  # PCK@50/40/30/20/10

        with torch.no_grad():
            for batch_data in val_loader:
                val_csi     = batch_data["input_wifi-csi"].to(device)
                val_pose_gt = batch_data["output"].to(device)

                predicted_val_pose, _ = model(val_csi)

                for idx, pct in enumerate([0.5, 0.4, 0.3, 0.2, 0.1]):
                    pck_iter[idx].append(
                        compute_pck_pckh(
                            predicted_val_pose.permute(0, 2, 1).cpu().numpy(),
                            val_pose_gt.permute(0, 2, 1).cpu().numpy(),
                            pct, align=False, dataset=dataset_name
                        )
                    )
                mpjpe, pampjpe, _, _ = calulate_error(
                    predicted_val_pose.cpu().numpy(),
                    val_pose_gt.cpu().numpy(), align=False
                )
                mpjpe_list   += mpjpe.tolist()
                pampjpe_list += pampjpe.tolist()

        avg_mpjpe    = float(np.mean(mpjpe_list))
        avg_pampjpe  = float(np.mean(pampjpe_list))
        # pck_overall: [PCK@50, PCK@40, PCK@30, PCK@20, PCK@10]
        pck_overall  = [float(np.mean(pck_iter[i], axis=0)[17]) for i in range(5)]

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"MPJPE={avg_mpjpe*1000:.1f}mm | "
            f"PA-MPJPE={avg_pampjpe*1000:.1f}mm | "
            f"PCK@20={pck_overall[3]:.1f}% | "
            f"PCK@50={pck_overall[0]:.1f}%"
        )

        # ── 记录 history（格式与 BO 调度器兼容）──────────────
        history.append({
            "epoch":    epoch,
            "mpjpe":    avg_mpjpe,       # 单位：米（与你的 WiFi 脚本一致）
            "pa_mpjpe": avg_pampjpe,     # BO 调度器用这个字段做排序
            "pck@20":   pck_overall[3],
            "pck@50":   pck_overall[0],
            "train_loss": avg_train_loss,
        })
        save_json(history, hist_path)

        # ── 保存最优模型 ──────────────────────────────────────
        if avg_pampjpe < best_pampjpe:
            best_pampjpe = avg_pampjpe
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "pa_mpjpe": avg_pampjpe},
                       best_path)
            print(f"  ✅ New best! PA-MPJPE={avg_pampjpe*1000:.1f}mm")

    print(f"\n{'=' * 60}")
    print(f"✅ {model_name.upper()} finished! "
          f"Best PA-MPJPE: {best_pampjpe*1000:.1f}mm")
    print(f"   Run dir: {run_dir}")
    print(f"{'=' * 60}")

    return best_pampjpe, run_dir


# ─────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Unified wrapper: MetaFi / HPELI / DT-Pose → BO scheduler"
    )
    parser.add_argument("config_file", type=str,
                        help="YAML 配置文件路径（pose_config.yaml）")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["metafi", "hpeli", "dtpose"],
                        help="要训练的模型")
    parser.add_argument("--lr", type=float, default=None,
                        help="学习率（覆盖 YAML 中的 base_learning_rate）")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="batch size（覆盖 YAML 中的 batch_size）")
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖 YAML
    if args.lr is not None:
        config["base_learning_rate"] = args.lr
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    lr         = config.get("base_learning_rate", 3e-4)
    batch_size = config.get("batch_size", 128)

    best_metric, run_dir = train(config, args.model_name, lr, batch_size)

    print(f"\n📊 Result: PA-MPJPE = {best_metric*1000:.1f}mm")
    print(f"📁 Run dir: {run_dir}")


if __name__ == "__main__":
    main()
