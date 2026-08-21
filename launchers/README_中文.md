# 四模型 CMD 启动脚本说明

更新时间：2026-08-21

本目录与学校 Windows 5090 保持一致。`.cmd` 脚本只在 5090 的 CMD 中执行；
本地 macOS 用于查看、版本管理和核验代码，不直接运行这些 Windows 启动器。

## 使用方式

在 5090 的 PyCharm CMD 中进入：

```cmd
cd /d D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset\launchers
```

先运行：

```cmd
00_verify_all.cmd
```

所有脚本都会调用 `_env.cmd` 设置固定根目录、Python、数据集和 baseline 代码路径。
不要单独运行 `_env.cmd`。脚本不使用 PowerShell 的 `$env:`、`&` 或
`Tee-Object`。

## 脚本与用途

| 脚本 | 用途 | 是否训练 |
|---|---|---|
| `00_verify_all.cmd` | 检查 Python/CUDA、PrivPose 入口、baseline 源码和九格样本数 | 否 |
| `01_privpose_dry_run.cmd` | 生成并检查 PrivPose 3×3 计划 | 否 |
| `02_privpose_run_or_resume_3x3.cmd` | 正式运行或续跑 PrivPose 3×3 | 是 |
| `03_privpose_s2p2_ablation.cmd` | 运行或续跑 PrivPose 组件消融 | 是 |
| `10_baseline_verify_metafi_hpeli.cmd` | 单独核验 MetaFi++/HPE-Li 54 项计划 | 否 |
| `11_metafi_run_3x3.cmd` | MetaFi++ 九格三 seed | 是 |
| `12_hpeli_run_3x3.cmd` | HPE-Li 九格三 seed | 是 |
| `20_dtpose_pretrain_verify.cmd` | 核验 DT-Pose 九格预训练样本 | 否 |
| `21_dtpose_pretrain_run_or_resume.cmd` | 生成 DT-Pose 九格 encoder 权重 | 是 |
| `22_dtpose_pose_verify.cmd` | 核验九份新权重和 27 项姿态计划 | 否 |
| `23_dtpose_run_3x3.cmd` | DT-Pose 九格三 seed 姿态训练 | 是 |
| `30_all_baselines_run_3x3.cmd` | 三 baseline 合并为 81 项顺序训练 | 是 |
| `90_summarize_all_baselines.cmd` | 汇总完成或部分完成的 81 项 manifest | 否 |
| `99_show_status.cmd` | 显示 PrivPose 和 baseline 结果状态 | 否 |

## baseline 的输出目录与续跑

MetaFi++ 和 HPE-Li 脚本的第一个参数可以指定输出目录，第二个参数写 `resume`
表示续跑。例如：

```cmd
11_metafi_run_3x3.cmd "D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset\strict_offline_runs\metafi_equal_samples_3x3_mpjpe_20260821_01" resume
```

DT-Pose 预训练的用法相同：

```cmd
21_dtpose_pretrain_run_or_resume.cmd "D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset\strict_offline_runs\dtpose_equal_samples_pretrain_20260821_01" resume
```

DT-Pose 姿态脚本参数依次为：预训练目录、结果目录、可选的 `resume`：

```cmd
23_dtpose_run_3x3.cmd "D:\...\dtpose_equal_samples_pretrain_20260821_01" "D:\...\dtpose_equal_samples_3x3_mpjpe_20260821_01" resume
```

已有输出目录但没有 `resume` 时，runner 会拒绝覆盖。续跑时必须保持模型、格子、
seed、worker 和超参数与原 manifest 一致。

## 只跑单格或修改超参数

这种情况直接使用 baseline runner，不需要增加新的启动脚本。例如只跑 MetaFi++
的 P2-S2：

```cmd
cd /d D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset\s2p2_equal_samples_mpjpe_code_20260820
set "PYTHONPATH=%CD%;D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset\_python_deps"
C:\Users\CHEN\.conda\envs\bit\python.exe -u cloud_rerun\run_3x3_equal_samples_three_models.py --project_root "%CD%" --dataset_root "D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset\data_base" --output_root "D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset\strict_offline_runs\metafi_p2s2_custom_01" --cells P2-S2 --models metafi --seeds 0 1 2 --num_workers 8 --eval_num_workers 8
```

可在命令末尾增加：

```text
--batch_size 32 --max_device_batch_size 32 --lr 1e-3 --epochs 60
```

同一条命令选择多个模型时，覆盖参数会同时作用于所有模型。三个模型需要不同参数
时，应分别运行。

## 默认姿态训练参数

不写覆盖项时，baseline runner 会分别使用：

| 模型 | 配置 batch | micro-batch | 实际更新 batch | 学习率 | epoch |
|---|---:|---:|---:|---:|---:|
| MetaFi++ | 16 | 16 | 16 | `1.85e-3` | 50 |
| HPE-Li | 16 | 16 | 16 | `1e-3` | 50 |
| DT-Pose | 32 | 16 | 16 | `8.49e-4` | 50 |

DT-Pose 的“实际更新 batch=16”来自历史 `train_pose.py` 的既有行为：
`step_count` 没有递增，所以每个 micro-batch 都执行更新。源码为保持历史可比性
没有修改。上面的自定义示例把 batch 和 micro-batch 都设为 32，是为了让实际更新
batch 确实等于 32；不要再用 `32 / 16` 并把它描述为标准 batch 32。

DT-Pose 预训练与姿态训练不是同一组参数。预训练默认使用 batch 4096、micro-batch
256、base LR `1.5e-4`、缩放后 optimizer LR `0.0024`、400 epoch、seed 42。

PrivPose 不使用这套 baseline 参数。它的正式 Student 参数由
`strict_offline_runs\S2P2_cmc_lr_bs_tuning_w84200_resume_clean2\selected_for_3x3.json`
清单固定为 batch 128、学习率
`2.0724687211082514e-4`，从编号 `01`/`02` 的脚本运行。

## 注意事项

- 所有训练按顺序运行，不会自动在同一张 5090 上并发多个模型。
- `num_workers` 默认 8；Windows 卡顿或 worker 崩溃时改回 4、2 或 0。
- 日志由训练代码自动保存，不需要终端重定向。
- DT-Pose 的严格等样本实验必须先完成新的预训练。
- `models\` 是代码和 checkpoint 导航；当前新实验只使用本目录列出的入口。
- 所有新 baseline checkpoint 按 validation MPJPE 选择；validation 结果不等于
  独立 official test performance。
