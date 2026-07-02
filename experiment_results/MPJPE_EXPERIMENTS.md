# MPJPE 优化实验记录

## 背景

原始消融实验以 **PA-MPJPE** 作为优化目标（scheduler 监控 PA-MPJPE、best checkpoint 按 PA-MPJPE 选取）。为进一步探索不同优化目标的影响，新增 **MPJPE** 作为主指标的实验线。

### 改动方式

从 `train_lupi_rgb_teacher.py` 复制出 `train_lupi_rgb_teacher_mpjpe.py`，仅修改优化指标的选取逻辑：

| 位置 | PA 版 | MPJPE 版 |
|------|-------|----------|
| Scheduler | `scheduler.step(pa)` | `scheduler.step(mpjpe)` |
| Best checkpoint | `if pa < best_metric` | `if mpjpe < best_metric` |
| SWA 比较 | `if swa_pa < best_metric` | `if swa_mpjpe < best_metric` |

模型结构、损失函数、训练流程完全不变。思路：让 ReduceLROnPlateau 和 best model 保存都跟随 MPJPE 而非 PA-MPJPE。

---

## MPJPE Bayesian HPO 结果

**脚本**: `run_bayesian_hparam_tuning_mpjpe.py`  
**模式**: mixup (CMC + Mixup, no CBAM)  
**划分**: cross_subject_split / Protocol 2  
**搜索空间**: lr ∈ [1e-4, 5e-3], bs ∈ {16, 32, 64, 128, 256}  
**试验**: 15 trials × 3 seeds  

| 参数 | 最优值 |
|------|--------|
| lr | **1.14e-04** |
| batch_size | **16** |
| MPJPE | 236.13 ± 2.14 mm |
| PA-MPJPE | 93.00 ± 0.56 mm |
| PCK@20 | 31.08 ± 1.08% |
| PCK@50 | 78.32 ± 0.42% |

> 对比 PA-HPO 最优结果：lr=1.03e-04, bs=64, PA-MPJPE=91.98mm。MPJPE 优化版找到了更小的 lr 和更小的 batch size。

---

## CMC+Mixup 全 9 种划分实验结果（MPJPE 优化版）

**脚本**: `run_cmc_mixup_all_splits_mpjpe.py`  
**配置**: lr=1.14e-04, bs=16, CMC λ=0.2, Mixup α=0.4, no CBAM  
**每种组合**: 2 seeds × 30 epochs  
**结果目录**: `experiment_results/cmc_mixup_all_splits_mpjpe/`

| Protocol | Split | MPJPE (mm) | PA-MPJPE (mm) | PCK@20 (%) | PCK@50 (%) |
|----------|-------|-----------|---------------|------------|------------|
| Daily | Random | 169.43 ± 0.59 | 102.43 ± 0.35 | 57.53 ± 0.25 | 87.47 ± 0.04 |
| Daily | Cross-Scene | 376.27 ± 1.68 | 100.25 ± 1.29 | 3.66 ± 0.60 | 48.56 ± 0.39 |
| Daily | Cross-Subject | 217.16 ± 0.78 | 98.11 ± 1.27 | 41.75 ± 0.37 | 84.33 ± 0.07 |
| Rehab | Random | 197.86 ± 1.23 | 100.77 ± 0.10 | 40.10 ± 0.62 | 82.45 ± 0.01 |
| Rehab | Cross-Scene | 334.56 ± 7.16 | 101.97 ± 1.36 | 6.48 ± 0.25 | 51.54 ± 2.10 |
| Rehab | Cross-Subject | 236.52 ± 1.24 | 94.44 ± 0.06 | 30.21 ± 0.67 | 78.07 ± 0.54 |
| All | Random | 176.18 ± 0.63 | 103.17 ± 0.03 | 52.60 ± 0.44 | 86.19 ± 0.05 |
| All | Cross-Scene | 362.19 ± 6.53 | 105.87 ± 3.76 | 5.32 ± 1.27 | 49.13 ± 1.65 |
| All | Cross-Subject | 230.85 ± 2.05 | 102.83 ± 0.94 | 35.43 ± 0.04 | 80.18 ± 0.37 |

### 关键观察

- **Random split 最容易**：MPJPE 最低（169~198mm），说明同分布下 WiFi 定位能力尚可
- **Cross-Scene 最难**：MPJPE 高达 334~376mm，跨场景泛化是 WiFi 姿态估计的核心难点
- **Cross-Subject 居中**：MPJPE 217~237mm，与 HPO 时的 236mm 一致
- PA-MPJPE 在不同划分之间相对稳定（94~106mm），说明姿态形状估计受划分影响较小，绝对位置才是瓶颈

---

## 新增文件清单

| 文件 | 用途 |
|------|------|
| `train_lupi_rgb_teacher_mpjpe.py` | MPJPE 优化版训练脚本（scheduler + best ckpt 按 MPJPE） |
| `run_bayesian_hparam_tuning_mpjpe.py` | MPJPE 贝叶斯 HPO（15 trials × 3 seeds） |
| `run_cmc_mixup_all_splits_mpjpe.py` | MPJPE 优化版 9 种划分全实验（2 seeds × 30 epochs） |

## 运行命令

```bash
# MPJPE HPO（仅 mixup 模式）
python run_bayesian_hparam_tuning_mpjpe.py data_base config.yaml --modes mixup

# MPJPE 全 9 种划分实验
python run_cmc_mixup_all_splits_mpjpe.py data_base config.yaml
```
