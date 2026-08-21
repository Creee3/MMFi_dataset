# 5090 四模型唯一运行入口

更新时间：2026-08-21

> 本地目录已按同一结构整理。学校 5090 使用 Windows 目录联接，本地 macOS 使用
> 相对符号链接；它们只提供导航，不会复制模型源码。本地额外保留论文目录、
> `资料收集/`、Git 环境和 Python 虚拟环境，这些不属于 5090 训练目录。

## 1. 先看这里

云端唯一项目根目录：

```text
D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset
```

以后区分两件事：

- **运行实验**：只进入根目录的 `launchers\`。
- **查看源码或历史 checkpoint**：进入 `models\`。

不要从 `models\` 中随便挑一个旧脚本启动新实验。`models\` 同时保存官方源码、
历史运行包和 checkpoint 导航；当前统一实验命令以 `launchers\README_中文.md`
为准。

## 2. 四个模型现在放在哪里

| 模型 | 当前正式训练代码 | 正式入口 |
|---|---|---|
| PrivPose（我们的模型） | 项目根目录 | `run_s2p2_3x3_rank2.py` |
| MetaFi++ | `s2p2_equal_samples_mpjpe_code_20260820\` | 统一 baseline 3×3 runner |
| HPE-Li | `s2p2_equal_samples_mpjpe_code_20260820\` | 统一 baseline 3×3 runner |
| DT-Pose | `s2p2_equal_samples_mpjpe_code_20260820\` | 先预训练，再运行统一 baseline 3×3 runner |

三个 baseline 共用等样本数据入口和训练调度框架，但网络源码分别位于：

```text
s2p2_equal_samples_mpjpe_code_20260820\model\metafi\
s2p2_equal_samples_mpjpe_code_20260820\model\hpeli\
s2p2_equal_samples_mpjpe_code_20260820\model\model.py
```

PrivPose 没有放进 baseline runner。它使用独立的 RGB Pose Teacher、WiFi Student、
CMC 和 Mixup 流程，必须继续从项目根目录运行。

## 3. 共享目录

```text
data_base\             四个模型共享的 MM-Fi 数据
strict_offline_runs\   所有新训练结果
rerun_archives\        历史云主机归档和 checkpoint
models\                源码/checkpoint 分类导航，不是统一启动目录
launchers\             当前唯一推荐的命令入口
Zipfiles\              传输压缩包
_root_archive_20260821\ 非当前主线的旧代码、评估工程、历史结果和说明
```

不要移动或删除 `data_base`、`strict_offline_runs`、`rerun_archives` 和当前两套
活动代码。`_python_deps` 仅存在于 5090，本地使用自己的 Python 环境。

根目录在 2026-08-21 做过一次非破坏性整理。归档过程没有删除文件；原路径、
新路径、文件数量和字节数记录在：

```text
_root_archive_20260821\README_中文.md
_root_archive_20260821\archive_manifest.json          5090 清单
_root_archive_20260821\local_archive_manifest.json    本地清单
```

归档目录只用于历史追溯和恢复，不作为当前训练入口。

## 4. 最常用的命令

在 PyCharm 的 **CMD** 终端执行：

```cmd
cd /d D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset\launchers
```

### 检查环境和代码，不训练

```cmd
00_verify_all.cmd
```

### 我们的 PrivPose 完整 3×3

先检查计划：

```cmd
01_privpose_dry_run.cmd
```

正式运行或按原目录自动续跑：

```cmd
02_privpose_run_or_resume_3x3.cmd
```

### PrivPose 的 S2P2 组件消融

```cmd
03_privpose_s2p2_ablation.cmd
```

### MetaFi++ 完整等样本 3×3

```cmd
11_metafi_run_3x3.cmd
```

### HPE-Li 完整等样本 3×3

```cmd
12_hpeli_run_3x3.cmd
```

### DT-Pose 完整等样本 3×3

DT-Pose 必须按顺序执行：

```cmd
20_dtpose_pretrain_verify.cmd
21_dtpose_pretrain_run_or_resume.cmd
22_dtpose_pose_verify.cmd
23_dtpose_run_3x3.cmd
```

### 三个 baseline 放在一个 81-run manifest 中运行

必须先完成 DT-Pose 的 9 格预训练，然后执行：

```cmd
30_all_baselines_run_3x3.cmd
```

单模型分别运行和一次性运行 81 次二选一即可，不要两种方式同时重复启动。

## 5. 结果位置

| 任务 | 默认结果目录 |
|---|---|
| PrivPose 3×3 | `strict_offline_runs\S2P2_3x3_rank2` |
| PrivPose 消融 | `strict_offline_runs\S2P2_temporal_cmc_mixup_ablation` |
| MetaFi++ 等样本 3×3 | `strict_offline_runs\metafi_equal_samples_3x3_mpjpe_20260821_01` |
| HPE-Li 等样本 3×3 | `strict_offline_runs\hpeli_equal_samples_3x3_mpjpe_20260821_01` |
| DT-Pose 等样本预训练 | `strict_offline_runs\dtpose_equal_samples_pretrain_20260821_01` |
| DT-Pose 等样本姿态 3×3 | `strict_offline_runs\dtpose_equal_samples_3x3_mpjpe_20260821_01` |
| 三 baseline 合并运行 | `strict_offline_runs\three_baselines_equal_samples_3x3_mpjpe_20260821_01` |

PrivPose runner 会读取 `last.pth` 并跳过完成任务，因此再次执行同一脚本就是续跑。
baseline runner 则要求对已有输出显式传入 `resume`，具体见 `launchers\README_中文.md`。

## 6. 默认超参数与统一口径

不在命令行增加覆盖项时，姿态训练使用以下参数：

| 模型 | 配置 batch | 单卡 micro-batch | 实际每次更新使用的 batch | 学习率 | epoch |
|---|---:|---:|---:|---:|---:|
| PrivPose Student | 128 | 128 | 128 | `2.0724687211082514e-4` | 30 |
| MetaFi++ | 16 | 16 | 16 | `1.85e-3` | 50 |
| HPE-Li | 16 | 16 | 16 | `1e-3` | 50 |
| DT-Pose 姿态阶段 | 32 | 16 | 16 | `8.49e-4` | 50 |

DT-Pose 姿态阶段的配置文件虽然写着 batch 32、micro-batch 16，但归档训练器
`train_pose.py` 的 `step_count` 没有递增，因此每个 micro-batch 都会执行一次
`optimizer.step()`，实际更新 batch 是 16，不是标准梯度累积后的 32。为忠实复现
历史训练逻辑，源码没有在本次整理中修改。

DT-Pose 的 CSI 自监督预训练是另一套参数：配置 batch 4096、micro-batch 256、
梯度累积 16 次、base LR `1.5e-4`，按 `batch_size / 256` 缩放后的 AdamW 初始
学习率为 `0.0024`，共 400 epoch，seed 42。

三个 baseline 的新实验均按对应验证集的最低 MPJPE 选择 checkpoint。这里的
validation 结果不能写成独立 official test performance。

如需修改 baseline 的 batch、学习率或 epoch，使用底层 runner 的命令行覆盖项；
三个模型需要不同参数时必须分开运行。若希望实际更新 batch 为 32，应同时设置：

```text
--batch_size 32 --max_device_batch_size 32
```

完整示例见 `launchers\README_中文.md`。PrivPose 的正式参数来自
`strict_offline_runs\S2P2_cmc_lr_bs_tuning_w84200_resume_clean2\selected_for_3x3.json`，
不要使用 baseline runner 的覆盖方式修改。

## 7. 当前状态说明

- PrivPose 正式 3×3 已有历史结果目录。
- 新的三个 baseline 等样本代码已经通过 5090 的源码、依赖和九格样本数核验。
- 旧的 `mmfi_3x3_mpjpe_20260819_full_02` 只是 81 项 `dry_run`，已经随历史结果归档，
  不是训练完成结果，也不是当前启动目录。
- 当前尚未生成新的 9 格等样本 DT-Pose 预训练权重；不能把历史权重直接冒充新权重。
- 所有新 baseline checkpoint 都按 validation MPJPE 选择。
