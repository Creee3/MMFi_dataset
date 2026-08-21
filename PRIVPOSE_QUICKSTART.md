# PrivPose 主线代码快速使用说明

本文件是当前 PrivPose 工程的主线入口说明。除非明确进行历史实验复现，否则优先使用这里列出的代码和脚本。

## 1. 主线文件

| 文件或目录 | 用途 |
| --- | --- |
| `common.py` | 模型、数据加载、数据增强、损失和评估的共享实现 |
| `config.yaml` | MM-Fi 协议、划分和数据配置 |
| `mmfi_lib/` | MM-Fi 数据集读取和 MPJPE、PA-MPJPE、PCK 评估 |
| `train_t1_rgb_teacher_mpjpe.py` | RGB 2-D 姿态 Teacher 训练入口 |
| `train_lupi_rgb_teacher_mpjpe.py` | WiFi 学生模型 + RGB 特权信息 + CMC 训练入口 |
| `run_s2p2_cmc_hparam_tuning.py` | CMC 超参数搜索；被正式 3x3 runner 调用 |
| `run_s2p2_3x3_rank2.py` | 当前正式的 Protocol x Split 3x3 实验入口 |
| `quick_check.sh` | 检查环境、数据目录和主线导入 |
| `run_formal_3x3.sh` | 调用正式 3x3 runner |
| `train_rgb_teacher.sh` | 单独启动 RGB Teacher |
| `evaluate_final_test.sh` | 调用归档中的独立 final-test 工具 |

baseline、zero-augmentation 回退实验、独立测试工具和数据完整性脚本仍保留在本地活动目录，同时在归档目录的 `optional_tools/` 中保存副本。它们不属于正式 3x3 的运行依赖，但没有从原目录删除。

## 2. 环境和数据

推荐使用项目已有环境：

```bash
source .venv/bin/activate
python -c "import torch, scipy, yaml, numpy; print('environment ok')"
```

默认目录约定：

```text
MMFi_Dataset/
├── data_base/       # 数据集；不要移动或删除
├── config.yaml
├── common.py
├── mmfi_lib/
└── strict_offline_runs/
```

当前代码需要 CSI、RGB 2-D keypoints 和 ground truth。只保留 RGB、WiFi CSI、ground truth 的样本可以用于当前方法，但不能再依赖其他模态。

## 3. 训练前检查

```bash
./quick_check.sh
```

这个检查只验证环境、配置、数据目录和主线 Python 模块，不启动训练。

## 4. 正式 3x3 实验

正式版本是 `run_s2p2_3x3_rank2.py`。它会按 Protocol 1/2/3 和 random/cross-scene/cross-subject 三种划分运行，并在已有结果目录中自动跳过已经完成的部分。

```bash
./run_formal_3x3.sh
```

默认参数：

- `num_workers=8`，失败后回退到 `4 -> 2 -> 1 -> 0`；
- `eval_num_workers=0`，减少 Windows/远程环境中的验证阶段卡死风险；
- `min_free_gpu_mib=0`，不等待额外显存；
- 结果写入 `strict_offline_runs/S2P2_3x3_rank2/`；
- 断点续跑依赖该目录内已有的 `history.json`、`student_result.json` 和 checkpoint。

如果选择文件不在默认位置，显式指定：

```bash
SELECTION_FILE=/path/to/selected_for_3x3.json ./run_formal_3x3.sh
```

如果只想先检查命令计划，不启动训练：

```bash
./run_formal_3x3.sh --dry_run
```

## 5. 独立 final test

训练过程中的 validation 指标和独立 final test 要分开。已有 checkpoint 的独立测试示例：

```bash
RUN_DIR=strict_offline_runs/S2P2_3x3_rank2/protocol2/cross_subject_split/student/seed_00 \
MODEL_TYPE=lupi \
./evaluate_final_test.sh
```

脚本默认使用 `best.pth`、`four_way_split`、`num_workers=0`；这些参数都可以通过环境变量或命令行参数覆盖。

独立测试的 Python 实现保留在项目根目录；归档目录中另有一份相同副本。

## 6. 重要口径

- 当前正式学生模型保留 CSI noise/frequency-mask/time-mask 和同步 Mixup，不能用无扰动脚本代替正式结果。
- WiFi-only 推理不读取 RGB；RGB 2-D keypoints 只在训练阶段作为特权信息使用。
- checkpoint 主选择指标是 validation MPJPE；同一 epoch 的 PA-MPJPE 和 PCK 作为配套记录。
- 训练日志中的 `test` 字样不自动等于独立 test；要根据实际 loader 判断是 validation 还是 final test。
- PCK 的当前归一化尺度由 `mmfi_lib/evaluate.py` 中的关节索引 `5` 和 `12` 计算，论文和结果表应保持同一评估代码。
- 新实验必须使用新的 `--results_root`，不要覆盖已有正式结果。

## 7. 可选代码和历史代码在哪里

调参、消融、旧版 job 编排、Windows 云端启动器和 Optuna bookkeeping 已复制到：

```text
rerun_archives/privpose_local_mainline_20260815/
```

原项目目录继续作为活动代码目录，不因归档而停止维护。归档目录内的 `归档说明_中文.md` 记录了每个文件的来源和用途，`mainline/` 保存正式 3x3 的完整代码副本。

其中：

- `optional_tools/evaluation/`：独立 final test、checkpoint 对比和误差分解；
- `optional_tools/fallback/`：无 CSI 扰动的 3x3 回退实验；
- `optional_tools/baseline/`：WiFi-only baseline；
- `optional_tools/data_checks/`：数据完整性检查；
- `legacy_scripts/`：更早的调参、消融和任务编排脚本。
