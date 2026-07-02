# Strict MPJPE 资料收集与下一步方法路线

日期：2026-06-26

当前结论：这轮 `alpha/dropout` 小网格没有稳定推进到 235 mm。最新 stage2 最优为
`242.57 mm`，而此前最可信的一次 strict final-test 仍是 `238.93 mm`。这说明瓶颈更像
跨主体/跨环境泛化、根节点/绝对位置迁移、骨架拓扑约束，而不是普通网格搜索。

## 本地已整理 PDF

集中目录：

```text
资料收集/strict_mpjpe_papers/
```

已放入：

- `DT-Pose_2501.09411.pdf`
- `MM-Fi_2305.10345.pdf`
- `AdaPose_cross_site_wifi_pose.pdf`
- `MetaFi_wifi_transformer_pose.pdf`
- `Person-in-WiFi-3D_CVPR2024.pdf`

没有直接 clone 代码仓库，先只保留链接，避免占用 D 盘/本机磁盘空间。需要深入某条路线时再单独下载。

## 核心 WiFi 姿态论文与代码

### DT-Pose: Towards Robust and Realistic Human Pose Estimation via WiFi Signals

- 论文：https://arxiv.org/abs/2501.09411
- 代码：https://github.com/cseeyangchen/DT-Pose
- 本地 PDF：`资料收集/strict_mpjpe_papers/DT-Pose_2501.09411.pdf`
- 和我们最相关的点：
  - 论文明确把问题拆成 cross-domain gap 和 structural fidelity gap。
  - 方法是两阶段：WiFi 表征预训练 + 拓扑约束姿态解码。
  - 这正好对应我们现在的现象：PA-MPJPE 还可以，但 MPJPE 在严格 final test 上卡住。
- 可落到当前工程的改法：
  - 先不完整复刻 DT-Pose；优先做轻量版拓扑解码头。
  - 把当前 `PoseHead(feature -> 17*3)` 改成 `feature -> 17 joint tokens -> graph/transformer decoder -> xyz`。
  - 加骨长一致性/左右肢体软约束，不需要额外存数据。
- 风险：
  - 完整两阶段预训练会增加训练时间；如果 D 盘紧张，不要缓存大规模预处理文件，在线读取/在线 mask 更稳。

### MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset

- 论文：https://arxiv.org/abs/2305.10345
- 项目页：https://ntu-aiot-lab.github.io/mm-fi
- 官方工具箱：https://github.com/ybhbingo/MMFi_dataset
- 本地 PDF：`资料收集/strict_mpjpe_papers/MM-Fi_2305.10345.pdf`
- 和我们最相关的点：
  - 数据集本身支持跨模态监督、WiFi/RGB/mmWave 等多模态任务。
  - 40 个 subject 被分到 4 个环境/域，说明 strict split 下的 subject gap 与 environment gap 会混在一起。
- 可落到当前工程的改法：
  - final-test breakdown 必须按 subject 和 scene 同时看。
  - 训练/验证划分要在论文里说清楚：我们的四路划分比标准 32/8 cross-subject 更严格，因为 teacher/student/model-selection/test 分离。

### AdaPose: Towards Cross-Site Device-Free Human Pose Estimation with Commodity WiFi

- 论文：https://arxiv.org/abs/2309.16964
- 本地 PDF：`资料收集/strict_mpjpe_papers/AdaPose_cross_site_wifi_pose.pdf`
- 代码：暂未找到稳定的官方公开仓库。
- 和我们最相关的点：
  - 重点不是换更大模型，而是处理 WiFi CSI 对环境变化敏感带来的 domain shift。
  - 对我们来说可以作为“为什么 strict final test 难”的论文依据。
- 可落到当前工程的改法：
  - 尝试轻量 domain alignment：CORAL/MMD/MixStyle，不做很重的目标域自适应。
  - 因为 final test 不能用标签，避免任何会泄露 test label 的 adaptation。

### C-MambaPose: Physics-Informed Complex Mamba for Cross-Environment WiFi HPE

- 论文：https://arxiv.org/abs/2606.13700
- 代码：https://github.com/phucngvinuni/cmampose
- 和我们最相关的点：
  - 代码描述里把 WiFi CSI 的 amplitude/phase 复数表征、时空序列建模、GraFormer GCN decoder 放在一起。
  - 对 cross-environment split 很有针对性。
- 可落到当前工程的改法：
  - 不建议第一步直接迁 Mamba，依赖和调试成本偏高。
  - 可先借鉴两个低成本点：amplitude/phase 分支输入、GraFormer/GCN 风格姿态 decoder。
- 风险：
  - 这是较新的预印本/代码，稳定性和可复现性需要谨慎验证。

### GraphPose-Fi / GraFormer 类骨架解码

- GraphPose-Fi 代码：https://github.com/Cirrick/GraphPose-Fi
- GraFormer 代码：https://github.com/Graformer/GraFormer
- 和我们最相关的点：
  - WiFi encoder 输出的是语义向量，本身缺骨架空间先验。
  - 图卷积/图 Transformer decoder 可以把人体关节邻接关系显式注入输出端。
- 可落到当前工程的改法：
  - 在 `common.py` 增加 `GraphPoseHead`。
  - 保留当前 WiFi encoder，不动数据加载，先只替换 pose head 做 A/B。
  - loss 仍用 absolute MPJPE/L1，附加轻量 bone-length consistency。

### MetaFi / MetaFi++ 代码线索

- 代码：https://github.com/pridy999/metafi_pose_estimation
- 本地 PDF：`资料收集/strict_mpjpe_papers/MetaFi_wifi_transformer_pose.pdf`
- 和我们最相关的点：
  - Transformer-based WiFi pose 作为可比较 baseline/写论文背景。
  - DT-Pose 官方 README 也把 MetaFi++ 作为基线来源之一。
- 可落到当前工程的改法：
  - 不建议直接整套替换；当前 baseline 已经能到 239 左右，整套换模型成本大。
  - 可借鉴其输入 token 化/pose decoder 设计。

### Person-in-WiFi-3D

- 项目/数据页：https://aiotgroup.github.io/Person-in-WiFi-3D/
- 代码：https://github.com/aiotgroup/Person-in-WiFi-3D-repo
- 本地 PDF：`资料收集/strict_mpjpe_papers/Person-in-WiFi-3D_CVPR2024.pdf`
- 和我们最相关的点：
  - 是 WiFi 3D pose 的重要近期工作，可用于 related work。
  - 更偏 multi-person / end-to-end WiFi pose，不一定能直接解决我们 strict single-person MM-Fi 的 235 mm 目标。

## 泛化与稳定化方法代码

### DomainBed

- 代码：https://github.com/facebookresearch/DomainBed
- 用途：提供 domain generalization 方法的基准实现思路。
- 当前工程可借鉴：
  - 不需要接入整个 DomainBed。
  - 只借鉴 ERM + CORAL/MMD/GroupDRO 风格的训练项设计。

### MixStyle

- 代码：https://github.com/KaiyangZhou/mixstyle-release
- 用途：通过混合 feature statistics 提升 domain generalization。
- 当前工程可借鉴：
  - 在 WiFi encoder 输出 feature 后加一个可开关的 `MixStyle1D`。
  - 只在训练打开，验证/测试关闭。
  - 先小概率 `p=0.3`、小 alpha，避免破坏绝对位置信息。

### SWAD / Model Soups

- SWAD 代码：https://github.com/khanrc/swad
- Model Soups 代码：https://github.com/mlfoundations/model-soups
- 用途：用更稳定/更平坦的 checkpoint 或权重平均减少 seed/checkpoint 方差。
- 当前工程可借鉴：
  - 普通 late-epoch SWA 已经没有明显收益。
  - 下一步更值得做 top-k validation checkpoint soup：保存验证最好的 3-5 个 checkpoint，只平均这些。
  - 这个不增加数据，不需要新预处理，对磁盘压力小。

## 我建议的下一步优先级

### P0：先做诊断，不再扩大普通网格

先确认 238.93 mm 那个 checkpoint 的错误主要集中在哪里：

```bash
python evaluate_final_test_breakdown_mpjpe.py data_base config.yaml --run_dir "experiment_results/strict_mpjpe_grid_search\baseline_lr7.0e-05_bs64_a0p2_do0p15_ly2_ff1024_head512_an0p02_af0p05_at0p05_rel0p0_root0p0_sub0p0\seed_00" --model_type baseline --checkpoint best.pth --num_workers 0
```

如果 worst subjects 主要集中在某一个 scene，优先做 domain generalization；如果所有 subject 都有 root 偏差，优先做 root/relative architecture。

### P1：Graph/Topology Pose Head

这是当前最像“真实模型改进”的方向：

- 保留现有 WiFi encoder。
- 替换 `PoseHead` 为 `GraphPoseHead`。
- 输入 feature 生成 17 个 joint token。
- 用人体骨架 adjacency 做 1-2 层轻量 GCN/GraphTransformer。
- 输出 17x3。
- 附加 bone-length consistency，不需要额外数据。

预期：主要改善结构稳定性，可能先让 PA-MPJPE 更好；如果和 root head 结合，才更可能推 MPJPE。

### P2：Root-relative Pose Head + Root Head

我们的 PA-MPJPE 约 95-98 mm，但 MPJPE 约 239-252 mm，这说明形状学到了，绝对 root/translation 是大问题之一。

建议改成：

```text
wifi feature -> relative pose head -> 17x3 root-relative skeleton
             -> root head          -> 1x3 root/global position
final pose = relative pose + root
```

这比只加 `lambda_rel_pose` 辅助 loss 更强，因为结构上强迫网络分开解决“形状”和“全局位置”。

### P3：Top-k Checkpoint Soup

这个最省空间，适合快速试：

- 训练时保留验证 MPJPE 最好的 3-5 个 checkpoint。
- 训练后平均这些 checkpoint 的权重。
- final test 比较 `best.pth`、`last.pth`、`topk_soup.pth`。

预期：不一定大幅降均值，但可能减少 238 -> 247 这种复现波动。

### P4：轻量 Domain Generalization

只做小步：

- feature-level CORAL/MMD：按 subject 或 scene 分组约束 feature statistics。
- MixStyle1D：在 WiFi feature 后混合均值/方差。
- 不推荐一开始上强 subject adversarial，因为 WiFi 的绝对位置和环境特征可能对 root prediction 有用，压太狠会伤 MPJPE。

## 不建议立刻做的事

- 不继续扩大 `alpha/dropout/lr` 大网格：当前收益已经不稳定。
- 不做大规模离线预处理缓存：D 盘空间紧，收益也不确定。
- 不直接迁完整 Mamba/DT-Pose 两阶段大工程：调通成本高，先用轻量 ablation 证明方向。
- 不用 final test 参与调参或自适应：论文上会很难解释，且容易泄露。

## 适合论文叙述的主线

可以把方法动机写成：

1. 严格四路划分下，普通 WiFi baseline 的 validation-test gap 显著，说明主要问题是跨主体/跨域泛化。
2. RGB/LUPI/CMC 更容易改善结构对齐，即 PA-MPJPE，但不一定改善绝对 MPJPE。
3. 因此本文把 WiFi pose 分解为骨架结构预测与 root/global position 预测，并引入轻量拓扑解码/稳定 checkpoint。
4. 这样既不依赖 test leakage，也不需要额外大规模预处理，适合 MM-Fi strict split。
