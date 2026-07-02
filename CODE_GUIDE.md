# MMFi 项目代码指南

## 项目概述

基于 WiFi CSI 的 3D 人体姿态估计，使用 MMFi 数据集。核心技术路线：

- **WiFi CSI** 作为输入（推理时仅需 WiFi）
- **RGB 2D 关键点** 作为特权信息（LUPI，仅训练时使用）
- **Teacher-Student 架构**：RGB Teacher 提供 CMC 对比学习监督
- **时序 Transformer**：32 帧滑动窗口 + stride=2
- **跨个体评估**：cross_subject_split（Protocol 2）

---

## 核心文件

### `common.py` — 共享模块（核心，约 23000 行）

所有训练脚本的公共依赖。包含：

| 模块 | 说明 |
|------|------|
| `WiFiFrameEncoderCBAM` | 单帧 WiFi CSI 编码器（CNN + 可选 CBAM 注意力） |
| `TemporalWiFiEncoderCBAM` | 时序 WiFi 编码器（CNN + CBAM + 2 层 Transformer） |
| `EnhancedRGBEncoder` | RGB 2D 关键点编码器（Linear + PosEmbed + 2 层 Transformer + FC） |
| `PoseHead` | 姿态回归头（512→512→256→51，输出 17×3） |
| `TemporalWiFiStudentCBAM` | **WiFi Baseline 模型**：TemporalWiFiEncoderCBAM + PoseHead |
| `RGBOnlyTeacher` | **纯 RGB Teacher**：RGB 编码器 → PoseHead（2D→3D lifting，不用 WiFi） |
| `WiFiFrameEncoderCBAM` | 单帧 WiFi 编码器（用于 Teacher 的 WiFi 分支） |
| `EnhancedFusionTeacher` | WiFi+RGB 融合 Teacher（门控融合，已弃用） |
| `TemporalLUPICMC_CBAM` | **CMC Student 模型**：WiFi 编码器 + PoseHead + 投影头（proj_w/proj_v） |
| `augment_wifi()` | WiFi CSI 数据增强（噪声、频率 mask、时间 mask） |
| `mixup_data()` | Mixup 增强（WiFi + RGB + GT 三元组混合） |
| `info_nce_loss()` | InfoNCE 对比损失（用于 CMC） |
| `eval_temporal_model()` | 时序模型验证（WiFi-only 推理） |
| `eval_teacher_model()` | Teacher 模型验证（WiFi + RGB 推理） |
| `load_config()` | 加载数据集配置，设置 `data_unit='frame'`, `modality='wifi-csi|rgb'` |
| `get_loaders()` | 数据加载：make_dataset → TemporalWindowWrapper → make_dataloader |
| `add_common_args()` | 所有脚本共享的命令行参数（lr, bs, epochs, window, stride 等） |

关键默认值：`use_root_relative=True`（已被各实验脚本覆盖为 False）

### `common_backup.py`

`common.py` 的备份副本。

---

## 训练脚本

### `train_baseline.py` — WiFi Baseline 训练

```
入口: python train_baseline.py <dataset_root> <config_file> [args...]
模型: TemporalWiFiStudentCBAM（无 RGB）
输入: WiFi CSI 时序窗口 [B, 32, 3, 114, 10]
训练: L1 loss + 可选 Mixup + 可选 WiFi 增广
```

被 `run_ablation_best_hparams.py` 的变体 A 和 B 调用。

**已修改**（2026-05-20）：添加了 `TRAIN_SEED` 环境变量支持。

### `train_lupi_rgb_teacher.py` — LUPI + CMC 训练

```
入口: python train_lupi_rgb_teacher.py <dataset_root> <config_file> [args...]
模型: TemporalLUPICMC_CBAM（WiFi Student）
Teacher: RGBOnlyTeacher（加载 T1 checkpoint，冻结）
训练: L1 loss + λ_cmc * InfoNCE + λ_distill * L1_distill
```

CMC 流程：Teacher 提取 RGB 特征 → proj_v 投影 → 与 Student 的 WiFi 特征 proj_w 投影做 InfoNCE 对齐。

**已修改**（2026-05-20）：添加了 `TRAIN_SEED` 环境变量支持。

### `train_teacher.py` — 融合 Teacher 训练（从 .pyc 恢复，已弃用）

```
入口: python train_teacher.py <dataset_root> <config_file> [args...]
模型: EnhancedFusionTeacher（WiFi + RGB 门控融合）
```

与当前 `common.py` 兼容，但项目已不再使用此脚本。T1 teacher 不是用这个训的。

### `train_lupi.py` — LUPI + CMC + Distill 训练（从 .pyc 恢复，已弃用）

```
入口: python train_lupi.py <dataset_root> <config_file> [args...]
模型: 旧版 TemporalLUPICMC_CBAM（含 forward_rgb 方法）
```

与当前 `common.py` 不兼容（调用了 `model.forward_rgb()` 和 `model.rgb_encoder`，当前模型无这些接口）。**不可直接运行**。

---

## 实验编排脚本

### `run_ablation_best_hparams.py` — 消融实验（主要脚本）

```
入口: python run_ablation_best_hparams.py <dataset_root> <config_file> [--variant X]

变体（Bayesian HPO 最优参数 lr=1.03e-4, bs=64, no CBAM）:
  A: 纯时序 WiFi（train_baseline.py, mixup_alpha=0）
  B: 时序 + Mixup（train_baseline.py, mixup_alpha=0.4）
  C: 时序 + CMC（train_lupi_rgb_teacher.py, λ_cmc=0.2, distill=0）
  D: 时序 + Mixup + CMC（train_lupi_rgb_teacher.py, mixup+cmc）

每个变体: N seeds × 30 epochs（默认 5 seeds，可修改 SEEDS 变量）
结果保存: experiment_results/ablation_best_hparams/<variant>/seed_XX/
断点续跑: 检查 history.json 是否存在自动跳过
```

通过 subprocess 调用 `train_baseline.py` 和 `train_lupi_rgb_teacher.py`，通过 `TRAIN_SEED` 环境变量控制种子。

### `run_debug_ablation.py` — 快速消融实验（自包含）

```
入口: python run_debug_ablation.py <dataset_root> <config_file> [variant] [--prefix=X]

Part 1: WiFi Baseline 组件消融 (V0~V4，各 1 seed × 30 epochs)
  V0: 单帧 CNN（最简）
  V1: + 时序窗口 (32帧 Transformer)
  V2: + CBAM 注意力
  V3: + 数据增强
  V4: + Mixup (= 完整 Baseline)

Part 2: Teacher WiFi 分支消融 (T0~T2，各 1 seed × 30 epochs)
  T0: WiFi + RGB 门控融合
  T1: 纯 RGB（删 WiFi 分支）
  T2: 纯 WiFi 单帧（删 RGB 分支）

结果保存: experiment_results/<prefix>/<variant>/
默认 prefix=debug_ablation
```

完全自包含，不依赖 subprocess。训练循环和模型定义均在文件内。参数：lr=3e-4, bs=128, warmup=3。

### `run_bayesian_hparam_tuning.py` — Bayesian 超参数搜索

```
入口: python run_bayesian_hparam_tuning.py <dataset_root> <config_file>

4 种模式: temporal_only / mixup / cbam / cbam_mixup
每种模式: 15 次 Optuna TPE 试验 × 3 seeds
搜索空间: lr [1e-4, 5e-3], bs [16,32,64,128,256], mixup_α [0.1, 1.0]
结果保存: experiment_results/bayesian_hparam_tuning/
```

最优结果：mixup 模式，lr=1.03e-04, bs=64, no CBAM, PA-MPJPE 91.98mm。

### `run_cmc_mixup_all_splits.py` — CMC+Mixup 全划分实验

```
入口: python run_cmc_mixup_all_splits.py <dataset_root> <config_file> [--protocol X] [--split Y]

3 种 Protocol:
  protocol1 — 仅日常活动 (13 类)
  protocol2 — 仅康复活动 (14 类)
  protocol3 — 全部活动 (27 类)

3 种 Split:
  random_split      — 80/20 随机划分
  cross_scene_split — 跨场景 (E01/E02/E03 → E04)
  cross_subject_split — 跨个体 (32 → 8 subjects)

固定配置: CMC + Mixup, lr=1.03e-4, bs=64, no CBAM
每种组合: 5 seeds × 30 epochs
结果保存: experiment_results/cmc_mixup_all_splits/<protocol>_<split>/seed_XX/
```

通过临时 yaml 文件替换 protocol 字段实现 protocol 切换，不修改原 config.yaml。通过 subprocess 调用 `train_lupi_rgb_teacher.py`。

### `main.py` — 统一入口（框架代码）

```
入口: python main.py <dataset_root> <config_file> --mode <MODE>

支持模式: baseline / teacher / teacher_rgb_only / teacher_wifi_only / 
          lupi / auto_cmc / auto_teacher_cmc / ablation / main_experiments

依赖: train_teacher.py, train_lupi.py（后者与当前 common.py 不兼容）
```

是一个框架性入口。ablation 模式通过 subprocess 递归调用自身。部分模式因依赖不兼容而不可用。

### 旧版 `run_temporal_*.py` 系列（历史遗留）

| 文件 | 说明 |
|------|------|
| `run_temporal_only_cmc_comparison_new.py` | 纯时序 vs 时序+CMC |
| `run_temporal_mixup_cmc_comparison.py` | 时序+Mixup vs 时序+Mixup+CMC |
| `run_temporal_cbam_cmc_comparison.py` | 时序+CBAM vs 时序+CBAM+CMC |
| `run_temporal_cbam_mixup_cmc_comparison.py` | 时序+CBAM+Mixup vs 时序+CBAM+Mixup+CMC |

这些是早期的对比实验脚本，已被 `run_ablation_best_hparams.py` 替代。

---

## 工具脚本

### `check_ckpt.py` — Checkpoint 检查工具

```
入口: python check_ckpt.py [path/to/checkpoint.pth]

功能:
  1. 读取 checkpoint 的 args（如果保存了）
  2. 检查同目录下的 history.json
  3. 从模型结构推断 d_model、关节数等信息
```

---

## 关键 Checkpoint

| 路径 | 模型 | PA-MPJPE | 用途 |
|------|------|----------|------|
| `strict_offline_runs/T1_rgb_only_teacher/best.pth` | RGBOnlyTeacher | 57.6mm | CMC 消融实验的 Teacher |
| `strict_offline_runs/fusion_teacher_20260406_212858/best.pth` | EnhancedFusionTeacher | — | 已弃用 |
| `experiment_results/teacher_002/best.pth` | RGBOnlyTeacher | 68.0mm | run_debug_ablation 训练的，过拟合，不用 |

---

## 结果目录

| 路径 | 说明 |
|------|------|
| `experiment_results/ablation_best_hparams/` | 消融实验汇总（A/B/C/D + summary.json） |
| `experiment_results/debug_ablation/` | 快速消融实验（V0~V4, T0~T2） |
| `experiment_results/bayesian_hparam_tuning/` | HPO 搜索结果 |
| `experiment_results/cmc_mixup_all_splits/` | CMC+Mixup 全 Protocol×Split 实验 |
| `strict_offline_runs/` | 训练输出目录（各脚本自动生成） |

---

## 常用命令

```bash
# 消融实验（全部变体）
python run_ablation_best_hparams.py data_base config.yaml

# 消融实验（单个变体）
python run_ablation_best_hparams.py data_base config.yaml --variant cmc

# 快速消融（只跑 T1 RGB Teacher）
python run_debug_ablation.py data_base config.yaml T1_rgb_only_teacher --prefix=T001_rgb_teacher

# CMC+Mixup 全 3×3 划分实验
python run_cmc_mixup_all_splits.py data_base config.yaml
python run_cmc_mixup_all_splits.py data_base config.yaml --protocol protocol2 --split cross_subject_split

# 检查 checkpoint
python check_ckpt.py strict_offline_runs/T1_rgb_only_teacher/best.pth

# 完整的训练命令示例
python train_baseline.py data_base config.yaml --split cross_subject_split \
    --epochs 30 --window 32 --stride 2 --no_cbam --lr 1.03e-04 --batch_size 64 \
    --mixup_alpha 0.4 --no_root_relative

python train_lupi_rgb_teacher.py data_base config.yaml --split cross_subject_split \
    --epochs 30 --window 32 --stride 2 --no_cbam --lr 1.03e-04 --batch_size 64 \
    --lambda_cmc 0.2 --lambda_distill 0.0 --cmc_stopgrad_rgb \
    --teacher_ckpt strict_offline_runs/T1_rgb_only_teacher/best.pth --no_root_relative
```
