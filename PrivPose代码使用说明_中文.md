# PrivPose 代码使用说明

这份文件是当前 PrivPose 工程的中文使用入口。日常修改、同步和运行实验时，优先以本文件及项目根目录中的主线代码为准。

## 1. 当前目录

本地主目录：

```text
/Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset
```

学校 5090 对应目录：

```text
D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset
```

## 2. 主线代码

正式 3×3 和当前消融共同依赖以下 7 个 Python 文件：

```text
run_s2p2_3x3_rank2.py
├── run_s2p2_cmc_hparam_tuning.py
├── train_t1_rgb_teacher_mpjpe.py
├── train_lupi_rgb_teacher_mpjpe.py
├── common.py
├── mmfi_lib/mmfi.py
└── mmfi_lib/evaluate.py
```

各文件用途：

| 文件 | 用途 |
| --- | --- |
| `run_s2p2_3x3_rank2.py` | 正式 Protocol × Split 3×3 实验入口 |
| `run_s2p2_temporal_cmc_mixup_ablation.py` | S2P2 Temporal、CMC、Mixup 组件消融入口 |
| `run_s2p2_cmc_hparam_tuning.py` | 日志、重试、进程监控和调参结果读取工具 |
| `train_t1_rgb_teacher_mpjpe.py` | 训练 2-D pose Teacher |
| `train_lupi_rgb_teacher_mpjpe.py` | 训练 WiFi Student，并按配置启用 CMC/Mixup |
| `common.py` | 模型、增强、损失、数据加载和评估公共实现 |
| `mmfi_lib/mmfi.py` | MM-Fi 数据读取 |
| `mmfi_lib/evaluate.py` | MPJPE、PA-MPJPE 和 PCK 计算 |
| `config.yaml` | MM-Fi 协议、数据划分和模态配置 |

## 3. 5090 环境

不要使用系统默认的 `python`。当前确认可用的解释器是：

```text
C:\Users\CHEN\.conda\envs\bit\python.exe
```

在 PyCharm 的 Windows CMD 终端中先执行：

```bat
cd /d D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset
set "PYTHONPATH=%CD%\_python_deps"
set "PYTHONNOUSERSITE=1"
C:\Users\CHEN\.conda\envs\bit\python.exe -c "import torch, scipy, yaml, numpy; print(torch.cuda.is_available())"
```

输出 `True` 表示 CUDA 环境可用。OpenCV 缺失时可能出现 Depth modality 警告；当前 WiFi CSI、2-D pose 和 3-D ground-truth 主线不使用 depth，可以忽略该警告。

## 4. 正式 3×3 实验

正式入口：

```text
run_s2p2_3x3_rank2.py
```

启动命令：

```bat
C:\Users\CHEN\.conda\envs\bit\python.exe -u .\run_s2p2_3x3_rank2.py .\data_base .\config.yaml --num_workers 8 --eval_num_workers 0 --worker_fallbacks 4 2 1 0 --min_free_gpu_mib 0
```

该任务会运行 3 个 protocol 与 3 个 split 的 9 个实验单元。每个单元先进行 Teacher 内部四折选择，再运行 Student seeds 0/1/2。

默认还需要：

```text
strict_offline_runs\S2P2_cmc_lr_bs_tuning_w84200_resume_clean2\selected_for_3x3.json
```

这个选择文件目前在 5090 上存在；本地若要独立运行正式 3×3，也必须同时保存该 JSON。

正式结果目录：

```text
strict_offline_runs\S2P2_3x3_rank2
```

中断后重新执行同一命令即可。runner 会读取 `last.pth` 继续训练，并跳过已有 `train_complete.json` 的任务。

只检查命令、不启动训练：

```bat
C:\Users\CHEN\.conda\envs\bit\python.exe -u .\run_s2p2_3x3_rank2.py .\data_base .\config.yaml --num_workers 8 --eval_num_workers 0 --worker_fallbacks 4 2 1 0 --min_free_gpu_mib 0 --dry_run
```

## 5. S2P2 组件消融

当前消融包含：

1. Temporal only：CMC=0，Mixup=0。
2. Temporal + CMC：CMC=0.1，Mixup=0。
3. Temporal + Mixup：CMC=0，Mixup=0.2。
4. Full：直接复用正式 S2P2 结果，不重新训练。

启动命令：

```bat
C:\Users\CHEN\.conda\envs\bit\python.exe -u .\run_s2p2_temporal_cmc_mixup_ablation.py .\data_base .\config.yaml --num_workers 8 --eval_num_workers 0 --worker_fallbacks 4 2 1 0 --min_free_gpu_mib 0
```

输出目录：

```text
strict_offline_runs\S2P2_temporal_cmc_mixup_ablation
```

最终汇总：

```text
strict_offline_runs\S2P2_temporal_cmc_mixup_ablation\ablation_result.json
```

训练脚本的标题可能统一显示 `WiFi + RGB-privileged CMC`。判断某一组是否真正启用 Teacher/CMC，应查看：

- `lambda_cmc` 是否大于 0；
- 是否出现 `Loaded RGB teacher`；
- `cmc_w` 和 `cmc` 是否大于 0。

## 6. 独立评估 checkpoint

示例：评估正式 S2P2 seed 0 的 `best.pth`：

```bat
C:\Users\CHEN\.conda\envs\bit\python.exe -u .\evaluate_final_test_mpjpe.py .\data_base .\config.yaml --run_dir .\strict_offline_runs\S2P2_3x3_rank2\protocol2\cross_subject_split\student\seed_00 --model_type lupi --checkpoint best.pth --num_workers 0
```

默认结果写入该 run 目录下的 `final_test.json`。论文中 validation 指标与独立 final-test 指标必须分开描述。

## 7. 数据检查

快速抽样检查 CSI、RGB keypoints 和 ground truth：

```bat
C:\Users\CHEN\.conda\envs\bit\python.exe -u .\check_s2p2_data_integrity.py .\data_base .\config.yaml --mat_check sample --npy_check sample
```

不要移动或删除：

```text
data_base
config.yaml
mmfi_lib
```

当前训练至少需要 WiFi CSI、同步 2-D pose/RGB keypoints 和 3-D ground truth。

## 8. 常见输出文件

| 文件 | 含义 |
| --- | --- |
| `log.txt` | 训练脚本保存的日志 |
| `process_output.log` | runner 捕获的完整子进程输出 |
| `history.json` | 每个 epoch 的 validation 指标 |
| `best.pth` | validation MPJPE 最优 checkpoint |
| `last.pth` | 最近 epoch checkpoint，用于断点续跑 |
| `train_complete.json` | 单个训练任务完成标记 |
| `student_result.json` | 单个 3×3 单元的三 seed 汇总 |
| `ablation_result.json` | 消融四组最终汇总 |

日志已经由代码自动保存，不需要额外使用 `Tee-Object` 或输出重定向。

## 9. 不要混用的实验口径

- 正式 S2P2 保留 CSI noise、frequency masking、temporal masking 和同步 Mixup。
- all-zero 脚本只是回退/对照实验，不能替代正式结果。
- Teacher 只在训练时提供结构监督，WiFi Student 推理时不读取 Teacher 或 2-D pose。
- checkpoint 按 validation MPJPE 选择，不能直接称为 official test performance。
- 新实验使用独立 `results_root`，不要覆盖 `S2P2_3x3_rank2`。

## 10. 代码归档位置

本地完整归档：

```text
PrivPose_本地归档_20260815
```

其中 `mainline/` 保存正式主线副本，`optional_tools/` 保存评估、baseline、数据检查与回退脚本，`legacy_scripts/` 保存历史实验代码。

5090 云端旧 Python 文件归档：

```text
_archived_python_20260816
```

活动代码始终以项目根目录为准，不要直接修改归档副本。
