"""
run_debug_ablation.py
快速消融实验

Part 1: WiFi Baseline 组件消融 (V0~V4)
  V0: 单帧CNN（最简）
  V1: + 时序窗口(32帧Transformer)
  V2: + CBAM
  V3: + 数据增强
  V4: + Mixup (=完整Baseline)

Part 2: Teacher WiFi分支消融 (T0~T2)
  T0: WiFi+RGB融合（现有Teacher）
  T1: 纯RGB（删WiFi）
  T2: 纯WiFi单帧（删RGB）

用法:
  python run_debug_ablation.py data_base config.yaml
"""

import os, sys, json, time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from common import (
    count_params,
    augment_wifi,
    TemporalWiFiStudentCBAM,
    EnhancedFusionTeacher,
    WiFiFrameEncoderCBAM,
    EnhancedRGBEncoder,
    PoseHead,
    eval_temporal_model,
    eval_teacher_model,
    load_config, get_loaders,
)


# ================================================================
#  新模型类（不修改 common.py）
# ================================================================

class SingleFrameBaseline(nn.Module):
    """V0用：单帧WiFi → CNN → PoseHead，没有Transformer，没有CBAM"""
    def __init__(self, in_ch=3, d_model=512, use_cbam=False, dropout=0.1):
        super().__init__()
        self.encoder = WiFiFrameEncoderCBAM(in_ch, d_model, use_cbam=use_cbam, dropout=dropout)
        self.pose_head = PoseHead(d_model, dropout)

    def forward(self, wifi_window):
        # wifi_window: [B, W, 3, 114, 10]
        wifi_cur = wifi_window[:, -1]      # 只取最后1帧 [B, 3, 114, 10]
        feat = self.encoder(wifi_cur)       # [B, 512]
        pred = self.pose_head(feat)         # [B, 17, 3]
        return pred, feat


class RGBOnlyTeacher(nn.Module):
    """T1用：纯RGB → 编码 → PoseHead，完全删掉WiFi"""
    def __init__(self, d_model=512, dropout=0.1):
        super().__init__()
        self.rgb_encoder = EnhancedRGBEncoder(d_model, dropout)
        self.pose_head = PoseHead(d_model, dropout)

    def forward(self, wifi_cur, rgb):
        fv = self.rgb_encoder(rgb)          # [B, 512]
        pred = self.pose_head(fv)           # [B, 17, 3]
        return pred, fv


class WiFiOnlyTeacher(nn.Module):
    """T2用：纯WiFi单帧 → 编码 → PoseHead，完全删掉RGB"""
    def __init__(self, in_ch=3, d_model=512, dropout=0.1):
        super().__init__()
        self.wifi_encoder = WiFiFrameEncoderCBAM(in_ch, d_model, use_cbam=False, dropout=dropout)
        self.pose_head = PoseHead(d_model, dropout)

    def forward(self, wifi_cur, rgb):
        fw = self.wifi_encoder(wifi_cur)    # [B, 512]
        pred = self.pose_head(fw)           # [B, 17, 3]
        return pred, fw


# ================================================================
#  统一训练函数
# ================================================================

def train_one_variant(
    variant_name, model, train_loader, val_loader, device,
    epochs=30, lr=3e-4, weight_decay=1e-3, warmup_epochs=3, lr_patience=3,
    is_teacher=False,       # Teacher模式：单帧WiFi+RGB
    use_augment=True,
    use_mixup=True,
    aug_noise=0.10, aug_freq_mask=0.20, aug_time_mask=0.20,
    mixup_alpha=0.4,
    results_prefix="debug_ablation",
):
    results_dir = os.path.join("experiment_results", results_prefix, variant_name)
    os.makedirs(results_dir, exist_ok=True)

    hist_path = os.path.join(results_dir, "history.json")
    if os.path.isfile(hist_path):
        print(f"\n⏭  {variant_name} 已完成，跳过")
        with open(hist_path) as f:
            h = json.load(f)
        return min(h, key=lambda e: e["pa_mpjpe"])

    print(f"\n{'='*70}")
    print(f"▶ {variant_name}")
    print(f"  is_teacher={is_teacher}, augment={use_augment}, mixup={use_mixup}")
    print(f"  Parameters: {count_params(model):,}")
    print(f"{'='*70}")

    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=lr_patience, min_lr=1e-6)

    best_pa = 1e9
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        running, n_batch = 0.0, 0
        t0 = time.time()

        # 学习率预热
        if ep <= warmup_epochs:
            for g in opt.param_groups:
                g["lr"] = lr * ep / warmup_epochs

        for batch in train_loader:
            wifi_window = batch["input_wifi-csi_window"].to(device)
            gt = batch["output"].to(device)

            # 数据增强
            if use_augment:
                wifi_window = augment_wifi(
                    wifi_window, aug_noise, aug_freq_mask, aug_time_mask)

            # Mixup
            lam, idx = None, None
            if use_mixup and mixup_alpha > 0:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(wifi_window.size(0), device=device)
                wifi_window = lam * wifi_window + (1 - lam) * wifi_window[idx]
                gt = lam * gt + (1 - lam) * gt[idx]

            # 前向传播
            if is_teacher:
                rgb = batch["input_rgb"].to(device)
                if use_mixup and lam is not None:
                    rgb = lam * rgb + (1 - lam) * rgb[idx]
                wifi_cur = wifi_window[:, -1]
                pred3d, _ = model(wifi_cur, rgb)
            else:
                pred3d, _ = model(wifi_window)

            loss = F.l1_loss(pred3d, gt)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            running += loss.item()
            n_batch += 1

        avg_loss = running / max(n_batch, 1)

        # 验证
        if is_teacher:
            mpjpe, pa, pck = eval_teacher_model(
                model, val_loader, device, use_root_relative=False)
        else:
            mpjpe, pa, pck = eval_temporal_model(
                model, val_loader, device, use_root_relative=False)

        if ep > warmup_epochs:
            scheduler.step(pa)

        cur_lr = opt.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # pck 处理
        # pck 处理
        if isinstance(pck, dict):
            pck20 = round(pck.get("pck@20", pck.get("PCK@20", 0)), 2)
            pck50 = round(pck.get("pck@50", pck.get("PCK@50", 0)), 2)
        elif isinstance(pck, (list, tuple)):
            pck20, pck50 = round(pck[0], 2), round(pck[1], 2)
        else:
            pck20, pck50 = 0, round(float(pck), 2)

        record = {
            "epoch": ep,
            "train_loss": round(avg_loss, 5),
            "mpjpe": round(mpjpe, 5),
            "pa_mpjpe": round(pa, 5),
            "pck@20": pck20,
            "pck@50": pck50,
            "lr": cur_lr,
        }
        history.append(record)

        is_best = pa < best_pa
        if is_best:
            best_pa = pa
            torch.save({"model": model.state_dict()},
                       os.path.join(results_dir, "best.pth"))

        tag = " ✅" if is_best else ""
        print(f"  Ep {ep:2d}/{epochs} | loss={avg_loss:.4f} | "
              f"MPJPE={mpjpe*1000:.1f} PA={pa*1000:.1f} "
              f"PCK@50={pck50:.1f} | lr={cur_lr:.6f} | {elapsed:.0f}s{tag}")

    # 保存
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    best_ep = min(history, key=lambda e: e["pa_mpjpe"])
    print(f"\n  📊 {variant_name} 最佳: "
          f"PA={best_ep['pa_mpjpe']*1000:.1f} "
          f"MPJPE={best_ep['mpjpe']*1000:.1f} "
          f"PCK@50={best_ep['pck@50']:.1f} "
          f"(epoch {best_ep['epoch']})")

    return best_ep


# ================================================================
#  Part 1: WiFi Baseline 组件消融
# ================================================================

def run_baseline_ablation(train_loader, val_loader, device, results_prefix="debug_ablation"):
    print(f"\n{'🔷'*35}")
    print("📌 Part 1: WiFi Baseline 组件消融")
    print(f"{'🔷'*35}")

    variants = [
        # (名称, 模型, 训练配置)

        # V0: 单帧CNN，无CBAM，无增强，无Mixup
        ("V0_single_frame", 
         SingleFrameBaseline(in_ch=3, d_model=512, use_cbam=False, dropout=0.15),
         dict(use_augment=False, use_mixup=False)),

        # V1: 32帧时序窗口，无CBAM，无增强，无Mixup
        ("V1_add_temporal",
         TemporalWiFiStudentCBAM(in_ch=3, d_model=512, window=32, use_cbam=False, dropout=0.15),
         dict(use_augment=False, use_mixup=False)),

        # V2: 32帧 + CBAM，无增强，无Mixup
        ("V2_add_cbam",
         TemporalWiFiStudentCBAM(in_ch=3, d_model=512, window=32, use_cbam=True, dropout=0.15),
         dict(use_augment=False, use_mixup=False)),

        # V3: 32帧 + CBAM + 数据增强，无Mixup
        ("V3_add_augment",
         TemporalWiFiStudentCBAM(in_ch=3, d_model=512, window=32, use_cbam=True, dropout=0.15),
         dict(use_augment=True, use_mixup=False)),

        # V4: 32帧 + CBAM + 数据增强 + Mixup = 完整Baseline
        ("V4_full_baseline",
         TemporalWiFiStudentCBAM(in_ch=3, d_model=512, window=32, use_cbam=True, dropout=0.15),
         dict(use_augment=True, use_mixup=True)),
    ]

    results = {}
    for vname, model, cfg in variants:
        best = train_one_variant(
            variant_name=vname, model=model,
            train_loader=train_loader, val_loader=val_loader, device=device,
            epochs=30, lr=3e-4, weight_decay=1e-3,
            warmup_epochs=3, lr_patience=3,
            is_teacher=False,
            results_prefix=results_prefix,
            **cfg,
        )
        if best:
            results[vname] = best

    return results


# ================================================================
#  Part 2: Teacher WiFi分支消融
# ================================================================

def run_teacher_ablation(train_loader, val_loader, device, results_prefix="debug_ablation"):
    print(f"\n{'🔶'*35}")
    print("📌 Part 2: Teacher WiFi分支消融")
    print(f"{'🔶'*35}")

    variants = [
        # T0: WiFi + RGB 融合（现有Teacher）
        ("T0_fusion_teacher",
         EnhancedFusionTeacher(in_ch=3, d_model=512, dropout=0.15)),

        # T1: 纯RGB（完全删掉WiFi分支）
        ("T1_rgb_only_teacher",
         RGBOnlyTeacher(d_model=512, dropout=0.15)),

        # T2: 纯WiFi单帧（完全删掉RGB分支）
        ("T2_wifi_only_teacher",
         WiFiOnlyTeacher(in_ch=3, d_model=512, dropout=0.15)),
    ]

    results = {}
    for vname, model in variants:
        best = train_one_variant(
            variant_name=vname, model=model,
            train_loader=train_loader, val_loader=val_loader, device=device,
            epochs=30, lr=3e-4, weight_decay=1e-3,
            warmup_epochs=3, lr_patience=3,
            is_teacher=True,
            use_augment=True, use_mixup=True,
            results_prefix=results_prefix,
        )
        if best:
            results[vname] = best

    return results


# ================================================================
#  结果汇总
# ================================================================

def print_summary(results_prefix="debug_ablation"):
    root = f"experiment_results/{results_prefix}"

    print(f"\n{'='*90}")
    print("📊 消融实验结果汇总")
    print(f"{'='*90}")

    all_results = {}
    for vname in sorted(os.listdir(root)):
        hp = os.path.join(root, vname, "history.json")
        if not os.path.isfile(hp):
            continue
        with open(hp) as f:
            h = json.load(f)
        all_results[vname] = min(h, key=lambda e: e["pa_mpjpe"])

    # ── Part 1 ──
    print(f"\n📋 Part 1: WiFi Baseline 组件消融")
    print("-" * 88)
    print(f"  {'变体':<22s} {'累加组件':<25s} "
          f"{'PA-MPJPE(mm)':<14s} {'MPJPE(mm)':<12s} {'PCK@50(%)':<10s} {'增量'}")
    print("-" * 88)

    b_order = ["V0_single_frame", "V1_add_temporal", "V2_add_cbam",
               "V3_add_augment", "V4_full_baseline"]
    b_desc = {
        "V0_single_frame":  "单帧CNN（最简）",
        "V1_add_temporal":  "+ 时序窗口(32帧)",
        "V2_add_cbam":      "+ CBAM注意力",
        "V3_add_augment":   "+ 数据增强",
        "V4_full_baseline": "+ Mixup (=完整Baseline)",
    }

    prev_pa = None
    for v in b_order:
        if v not in all_results:
            print(f"  {v:<22s}  ⚠️ 缺失")
            continue
        r = all_results[v]
        pa = r["pa_mpjpe"] * 1000
        mp = r["mpjpe"] * 1000
        p50 = r.get("pck@50", 0)
        delta = ""
        if prev_pa is not None:
            d = prev_pa - pa
            delta = f"↓{d:.1f}" if d > 0 else f"↑{-d:.1f}"
        prev_pa = pa
        print(f"  {v:<22s} {b_desc[v]:<25s} {pa:<14.1f} {mp:<12.1f} {p50:<10.1f} {delta}")

    # ── Part 2 ──
    print(f"\n📋 Part 2: Teacher WiFi分支消融")
    print("-" * 88)
    print(f"  {'变体':<25s} {'输入':<25s} "
          f"{'PA-MPJPE(mm)':<14s} {'MPJPE(mm)':<12s} {'PCK@50(%)':<10s}")
    print("-" * 88)

    t_order = ["T0_fusion_teacher", "T1_rgb_only_teacher", "T2_wifi_only_teacher"]
    t_desc = {
        "T0_fusion_teacher":     "WiFi + RGB (门控融合)",
        "T1_rgb_only_teacher":   "纯RGB (无WiFi)",
        "T2_wifi_only_teacher":  "纯WiFi单帧 (无RGB)",
    }

    for v in t_order:
        if v not in all_results:
            print(f"  {v:<25s}  ⚠️ 缺失")
            continue
        r = all_results[v]
        pa = r["pa_mpjpe"] * 1000
        mp = r["mpjpe"] * 1000
        p50 = r.get("pck@50", 0)
        print(f"  {v:<25s} {t_desc[v]:<25s} {pa:<14.1f} {mp:<12.1f} {p50:<10.1f}")

    # ── 分析 ──
    if "T0_fusion_teacher" in all_results and "T1_rgb_only_teacher" in all_results:
        t0 = all_results["T0_fusion_teacher"]["pa_mpjpe"] * 1000
        t1 = all_results["T1_rgb_only_teacher"]["pa_mpjpe"] * 1000
        diff = t0 - t1
        if diff > 0:
            print(f"\n  💡 纯RGB比融合Teacher低 {diff:.1f}mm → WiFi分支是拖累")
        else:
            print(f"\n  💡 融合Teacher比纯RGB低 {-diff:.1f}mm → WiFi分支有帮助")

    if "V0_single_frame" in all_results and "T2_wifi_only_teacher" in all_results:
        v0 = all_results["V0_single_frame"]["pa_mpjpe"] * 1000
        t2 = all_results["T2_wifi_only_teacher"]["pa_mpjpe"] * 1000
        print(f"\n  💡 V0(单帧Baseline)={v0:.1f}mm vs T2(单帧Teacher)={t2:.1f}mm")
        print(f"     两者结构相同，差距来自训练配置（增强/Mixup等）")

    print(f"\n{'='*90}")

    sp = os.path.join(root, "summary.json")
    with open(sp, "w") as f:
        json.dump({k: {kk: round(vv, 5) if isinstance(vv, float) else vv
                       for kk, vv in v.items()} for k, v in all_results.items()},
                  f, indent=2)
    print(f"📄 → {sp}")


# ================================================================
#  主入口
# ================================================================

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python run_debug_ablation.py <dataset_root> <config_file> [variant]")
        print("例如: python run_debug_ablation.py data_base config.yaml")
        print("      python run_debug_ablation.py data_base config.yaml T1_rgb_only_teacher")
        sys.exit(1)

    dataset_root, config_file = sys.argv[1], sys.argv[2]

    # 解析剩余参数
    only_variant = None
    results_prefix = "debug_ablation"  # 默认保存到 experiment_results/debug_ablation/
    extra = sys.argv[3:]
    for a in extra:
        if a.startswith("--prefix="):
            results_prefix = a.split("=", 1)[1]
        elif not a.startswith("--"):
            only_variant = a

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 加载数据（window=32，V0内部只取最后1帧）
    cfg = load_config(config_file, "cross_subject_split")
    train_ds, val_ds, train_loader, val_loader = get_loaders(
        dataset_root, cfg, window=32, stride=2,
        batch_size=128, val_batch_size=128)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    print(f"\n{'='*70}")
    print(f"🔬 快速消融实验")
    if only_variant:
        print(f"   只跑: {only_variant}")
    else:
        print(f"   Part 1: WiFi Baseline 组件消融 (V0~V4) × 1次")
        print(f"   Part 2: Teacher WiFi分支消融 (T0~T2) × 1次")
        print(f"   总计: 7 个变体")
    print(f"   每个变体: 30 epochs, 不使用 root-relative")
    print(f"   时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    t_start = time.time()

    all_variants = {
        # Part 1: Baseline
        "V0_single_frame": dict(
            model=SingleFrameBaseline(in_ch=3, d_model=512, use_cbam=False, dropout=0.15),
            is_teacher=False, use_augment=False, use_mixup=False),
        "V1_add_temporal": dict(
            model=TemporalWiFiStudentCBAM(in_ch=3, d_model=512, window=32, use_cbam=False, dropout=0.15),
            is_teacher=False, use_augment=False, use_mixup=False),
        "V2_add_cbam": dict(
            model=TemporalWiFiStudentCBAM(in_ch=3, d_model=512, window=32, use_cbam=True, dropout=0.15),
            is_teacher=False, use_augment=False, use_mixup=False),
        "V3_add_augment": dict(
            model=TemporalWiFiStudentCBAM(in_ch=3, d_model=512, window=32, use_cbam=True, dropout=0.15),
            is_teacher=False, use_augment=True, use_mixup=False),
        "V4_full_baseline": dict(
            model=TemporalWiFiStudentCBAM(in_ch=3, d_model=512, window=32, use_cbam=True, dropout=0.15),
            is_teacher=False, use_augment=True, use_mixup=True),
        # Part 2: Teacher
        "T0_fusion_teacher": dict(
            model=EnhancedFusionTeacher(in_ch=3, d_model=512, dropout=0.15),
            is_teacher=True, use_augment=True, use_mixup=True),
        "T1_rgb_only_teacher": dict(
            model=RGBOnlyTeacher(d_model=512, dropout=0.15),
            is_teacher=True, use_augment=True, use_mixup=True),
        "T2_wifi_only_teacher": dict(
            model=WiFiOnlyTeacher(in_ch=3, d_model=512, dropout=0.15),
            is_teacher=True, use_augment=True, use_mixup=True),
    }

    if only_variant:
        if only_variant not in all_variants:
            print(f"❌ 未找到变体: {only_variant}")
            print(f"   可用: {', '.join(all_variants.keys())}")
            sys.exit(1)
        cfg = all_variants[only_variant]
        train_one_variant(
            variant_name=only_variant, model=cfg["model"],
            train_loader=train_loader, val_loader=val_loader, device=device,
            epochs=30, lr=3e-4, weight_decay=1e-3,
            warmup_epochs=3, lr_patience=3,
            is_teacher=cfg["is_teacher"],
            use_augment=cfg["use_augment"],
            use_mixup=cfg["use_mixup"],
            results_prefix=results_prefix)
    else:
        run_baseline_ablation(train_loader, val_loader, device, results_prefix)
        run_teacher_ablation(train_loader, val_loader, device, results_prefix)

    print(f"\n⏱  总耗时: {(time.time()-t_start)/3600:.1f} 小时")
    print_summary(results_prefix)