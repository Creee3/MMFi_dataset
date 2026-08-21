# 本地根目录归档说明

更新时间：2026-08-21

本目录收纳从本地 `MMFi_Dataset` 根目录移出的非当前主线内容。整理采用同盘移动，
没有删除或覆盖模型源码、数据、checkpoint、实验结果和论文资料。

## 当前活动主线

根目录继续保留：

- PrivPose 正式 3x3、CMC 调参、组件消融和 Teacher/Student 训练入口；
- 三个 baseline 的等样本活动代码；
- `launchers/` 与 `models/` 导航；
- `data_base/`、`strict_offline_runs/`、`rerun_archives/` 和 `Zipfiles/`；
- 本地特有的论文目录和 `资料收集/`。

## 归档分类

- `evaluation_projects/`、`evaluation_tools/`：旧 PCK、逐关节与独立评估工程；
- `historical_results/`、`historical_reports/`：旧结果、汇总和阶段性记录；
- `old_privpose_scripts/`：已不是当前正式入口的 PrivPose 脚本；
- `old_code_bundles/`：已被当前等样本包取代的旧 baseline 工程；
- `old_docs/`、`previous_archives/`：旧说明和此前归档；
- `dataset_cleanup_history/`：数据清理与完整性检查工具；
- `transfer_and_incomplete/`：传输包、临时 staging 和旧导航容器；
- `cache_and_metadata/`、`audit_records/`：缓存、临时文件和云端只读审计记录。

## 完整清单与恢复

`local_archive_manifest.json` 记录每个原路径、目标路径、文件数、字节数及单文件
SHA-256。所有记录状态均应为 `moved_verified` 或 `source_absent`。

需要恢复某一项时，按清单把 `destination` 原样移动回 `source`。恢复前必须确认
原路径不存在，禁止覆盖同名的新文件。

`maintenance_tools/organize_local_like_5090.py` 是本次整理脚本，仅用于审计和复现
目录规则，不是模型训练入口。
