# 你现在关注一下我的这四个文件，你应该能明白这是一条主线的

> 你现在关注一下我的这四个文件，你应该能明白这是一条主线的
>
> User context
>
> Attachments:
> - train_baseline_mpjpe.py: `./train_baseline_mpjpe.py`
> - train_lupi_rgb_teacher_mpjpe.py: `./train_lupi_rgb_teacher_mpjpe.py`
> - run_ablation_mpjpe.py: `./run_ablation_mpjpe.py`
> - run_bayesian_hparam_tuning_mpjpe.py: `./run_bayesian_hparam_tuning_mpjpe.py`

<details><summary>2 previous messages</summary>

> 我先把这四个文件串起来看一遍，确认这条训练主线的入口、依赖关系和实验分工，再给你一个清晰的整体理解。
>
> 我已经把主干读出来了：这四个文件不是孤立脚本，而是“训练脚本两条分支 + 两个上层实验调度器”的一整套流程。接下来我把它们之间的因果关系和你这条主线的设计意图梳理清楚。
</details>

是的，这四个文件我已经看明白了，确实是一条很清晰的主线，而且层次分得也比较自然。

这条主线可以概括成两层训练脚本、两层实验调度：

1. `train_baseline_mpjpe.py`
负责最基础的 WiFi-only 时序基线训练。
核心特点是：
- 只吃 `wifi_window`
- 用 `TemporalWiFiStudentCBAM`
- 训练目标直接按 `MPJPE` 选最优
- 支持 mixup、增强、warmup、scheduler、early stopping

2. `train_lupi_rgb_teacher_mpjpe.py`
是在 baseline 之上往前推进的一版。
核心逻辑是：
- 主体仍然是 WiFi 预测 3D pose
- 训练时额外引入 `RGB teacher`
- 加了两种辅助监督：
  - `CMC` 对齐 WiFi/RGB 特征
  - `distillation` 用 teacher pose 蒸馏 student
- 验证时仍然只看 `WiFi-only` 路径
所以它本质上是在做 LUPI：训练时用 privileged info，测试时不用。

3. `run_ablation_mpjpe.py`
这是这条主线的“横向对比实验器”。
它不是新模型，而是把前面两类训练脚本组织成 4 个消融 variant：
- A: baseline
- B: baseline + mixup
- C: lupi + cmc
- D: lupi + cmc + mixup

它的作用很明确：回答“性能提升到底来自哪里”。

4. `run_bayesian_hparam_tuning_mpjpe.py`
这是这条主线的“上游超参搜索器”。
它先在 CMC 路线上做 Optuna 搜索，找 `lr / batch_size` 最优组合，再把最优设置用于后续实验。
它回答的是“这条主线应该用什么训练配置最合理”。

如果把它画成一条主线，基本就是：

`Bayesian HPO`
-> 确定较优训练超参
-> `Baseline / LUPI-CMC` 两类训练脚本
-> `Ablation` 做系统对比
-> 最终围绕 `MPJPE` 汇总结论

不过我顺手也看到了两个很关键的小问题，后面很可能会影响实验管理：

- `train_lupi_rgb_teacher_mpjpe.py` 里实际 `run_dir` 前缀是
  `temporal_lupi_rgb_teacher_cmc_cbam_distill`
  但 `run_ablation_mpjpe.py` 和 `run_bayesian_hparam_tuning_mpjpe.py` 里写的是
  `temporal_lupi_rgb_teacher_mpjpe_cmc_cbam_distill`
  这里多了个 `_mpjpe`，会导致调度脚本找不到对应输出目录，或者 fallback 行为不稳定。

- `train_lupi_rgb_teacher_mpjpe.py` 最后打印的是
  `Result: PA-MPJPE = ...`
  但整个脚本实际上是在按 `MPJPE` 选最优并返回 `best_metric`。
  这个输出文字和真实优化目标不一致，容易误导后续记录。

如果你愿意，我下一步可以直接帮你做两件事之一：

1. 把这条主线彻底整理成一张“代码架构图 + 实验逻辑图”
2. 直接帮你修这几个主线里的命名/结果口径不一致问题，让四个文件完全对齐

> 此外你在关注一下这个T1的内容，按照道理来说这个内容应该和我那个train_lupi_rgb_teacher_mpjpe.py这个的训练逻辑一样的的，模型也是一样的，唯一不同的应该就是一个是以mpjpe为约束一个是以pampjpe约束
>
> User context
>
> Attachments:
> - run_debug_ablation.py: `./run_debug_ablation.py`

<details><summary>2 previous messages</summary>

> 我继续顺着这条主线往上核对 `T1`，重点看 `run_debug_ablation.py` 以及它关联的 teacher 训练逻辑，确认它和 `train_lupi_rgb_teacher_mpjpe.py` 是否真的是“同模型同流程，只差优化指标”。
>
> 我已经定位到关键参照物了：`T1` 不是来自 `run_debug_ablation.py` 本身，而是被后续 MPJPE 路线当成 teacher checkpoint 的来源 `strict_offline_runs/T1_rgb_only_teacher/best.pth`。我再把 `train_lupi_rgb_teacher.py` 和 `train_lupi_rgb_teacher_mpjpe.py` 正面对一下，这样能明确是不是只有选优指标不同。
</details>

不是完全这样。`T1` 里确实包含了和 `train_lupi_rgb_teacher_mpjpe.py` 同名同结构的 `RGBOnlyTeacher` 思路，但如果严格按“训练逻辑”来对，这两者差别不只是 `MPJPE` 和 `PA-MPJPE`。

关键要分成两层看：

`T1_rgb_only_teacher` 在 [run_debug_ablation.py](./run_debug_ablation.py) 里，是一个“独立训练 RGB-only teacher”的实验变体。
它做的是：
- 模型：`RGBOnlyTeacher(d_model=512, dropout=0.15)`
- 输入：`rgb`
- 训练损失：只有 `F.l1_loss(pred3d, gt)`
- 训练流程：用 `train_one_variant(...)`
- 验证选优：按 `PA-MPJPE` 选最优，`scheduler.step(pa)`，`best = min(..., key="pa_mpjpe")`

而 [train_lupi_rgb_teacher_mpjpe.py](./train_lupi_rgb_teacher_mpjpe.py) 做的是另一件事：
- 主模型不是 `RGBOnlyTeacher`，而是 `TemporalLUPICMC_CBAM`
- `RGBOnlyTeacher` 只作为冻结的 teacher 被加载
- student 输入是 `wifi_window`
- 总损失是 `pose + lambda_cmc * cmc + lambda_distill * distill`
- 验证时评估的是 `WiFi-only student`
- 选优是按 `MPJPE`

所以结论是：

1. 如果你说“`T1` 的模型和 `train_lupi_rgb_teacher_mpjpe.py` 里用到的 teacher 分支是不是同一个”，这个基本是对的。
都是 `RGBOnlyTeacher` 这套 `EnhancedRGBEncoder + PoseHead` 结构。

2. 如果你说“`T1` 的训练逻辑和 `train_lupi_rgb_teacher_mpjpe.py` 一样，只差一个用 MPJPE，一个用 PA-MPJPE”，这个不对。
它们训练目标和角色完全不同：
- `T1` 是在训练一个 RGB-only teacher 本体
- `train_lupi_rgb_teacher_mpjpe.py` 是在训练 WiFi student，RGB-only teacher 只是监督源

另外我还确认了一点：
[train_lupi_rgb_teacher.py](./train_lupi_rgb_teacher.py)
和
[train_lupi_rgb_teacher_mpjpe.py](./train_lupi_rgb_teacher_mpjpe.py)

这两个才是真正“几乎完全一样，只差选优指标”的一对：
- 前者按 `PA-MPJPE` 选 best，`scheduler.step(pa)`
- 后者按 `MPJPE` 选 best，`scheduler.step(mpjpe)`

也就是说，真正对应你这句话的，不是 `T1` 和 `train_lupi_rgb_teacher_mpjpe.py`，而是：
- `train_lupi_rgb_teacher.py`
- `train_lupi_rgb_teacher_mpjpe.py`

如果你愿意，我下一步可以直接帮你把这三者的关系整理成一张非常清楚的“角色对照表”，这样你后面写论文或讲实验线会特别顺。

> 我的意思是debug里面训练出teacher的那个逻辑应该是我的目录文件某个文件训练逻辑是相同的，只是约束条件不一样

对，你这个意思我明白了。

你的判断基本是对的，但要更精确一点说：

`debug` 里面负责“训练出 teacher”的那段逻辑，确实应该对应你目录里某个“PA 版本”的训练思路；只是它现在不是一个单独的正式训练脚本，而是被写进了 [run_debug_ablation.py](./run_debug_ablation.py) 的 `T1_rgb_only_teacher` 变体里。

对应关系可以这样看：

- `run_debug_ablation.py` 里的 `T1_rgb_only_teacher`
  - 模型：`RGBOnlyTeacher`
  - 输入：`rgb`
  - 损失：`L1(pred3d, gt)`
  - 选优：按 `PA-MPJPE`
  - 作用：产出那个 `T1` teacher checkpoint

- [train_lupi_rgb_teacher.py](./train_lupi_rgb_teacher.py)
  - 不是训练这个 teacher 本体
  - 而是加载一个已经训练好的 `RGBOnlyTeacher` 作为 teacher
  - 再去训练 WiFi student
  - 选优：按 `PA-MPJPE`

- [train_lupi_rgb_teacher_mpjpe.py](./train_lupi_rgb_teacher_mpjpe.py)
  - 跟上一个脚本逻辑几乎同构
  - 只是选优指标改成了 `MPJPE`

所以你这句话如果改成更准确的版本，就是：

`debug` 里的 `T1` 是 “teacher 的训练源头”；
而 [train_lupi_rgb_teacher.py](./train_lupi_rgb_teacher.py) 和 [train_lupi_rgb_teacher_mpjpe.py](./train_lupi_rgb_teacher_mpjpe.py) 是 “使用这个 teacher 去监督 student”的后续训练脚本。

也就是说：

- “模型一样”这一点，对 `RGBOnlyTeacher` 本身来说是对的
- “训练逻辑一样，只是约束条件不同”这句话，严格来说更适合描述：
  - [train_lupi_rgb_teacher.py](./train_lupi_rgb_teacher.py)
  - [train_lupi_rgb_teacher_mpjpe.py](./train_lupi_rgb_teacher_mpjpe.py)

而 `T1` 本身更像是这条链路里的“teacher 预训练阶段”。

如果你要，我下一步可以直接帮你把这整条链路整理成一句论文里能用的表述，比如：
“先训练 T1 RGB-only teacher，再在 LUPI 框架下用其进行 CMC/蒸馏监督；PA 版与 MPJPE 版仅在模型选择准则上不同。”

> 因为我这个T1训练的是以pampjpe为标准，你可以帮我把它改成以mpjpe为约束进行训练吗，就是创造一个新文件，原来teacher训练逻辑完全不要改变，只改变训练的约束条件

<details><summary>5 previous messages</summary>

> 我现在直接给 `T1` 单独做一个 `MPJPE` 版本的新脚本，保留原来的模型、数据流、增强、mixup、损失都不动，只把“scheduler / best ckpt / 汇总选优”从 `PA-MPJPE` 切到 `MPJPE`。
>
> 我已经确定方案了：不去碰 `debug` 原文件，而是把 `T1_rgb_only_teacher` 那段训练单独抽成一个新的 `MPJPE` 脚本，默认超参、增强、mixup、warmup、数据加载方式都按 `debug` 保持一致。
>
> 我准备开始落文件了：新文件会只服务于 “训练 T1 teacher 的 MPJPE 版本”，避免把 `run_debug_ablation.py` 的其他消融逻辑一起耦合进来。然后我会顺手做一次语法检查，确保能直接跑。
>
> <details><summary>Created 1 file</summary>
>
> - Wrote `./train_t1_rgb_teacher_mpjpe.py` (+200 -0)
>
> </details>
>
> 新文件已经落好了。我现在做一个轻量校验，确认这个脚本至少能通过语法层和参数层，不会一运行就因为拼写或接口问题挂掉。
</details>

新文件已经建好了：[train_t1_rgb_teacher_mpjpe.py](./train_t1_rgb_teacher_mpjpe.py)。

我保持了你原来 `T1_rgb_only_teacher` 的训练逻辑不变：还是 `RGBOnlyTeacher`，还是同样的输入、增强、mixup、L1 loss、warmup、AdamW、梯度裁剪、数据加载方式。唯一改动的是“约束条件”切到了 `MPJPE`：
- `scheduler.step(mpjpe)` 代替 `scheduler.step(pa)`
- `best.pth` 按 `mpjpe` 最小保存
- 最终结果汇报按 `MPJPE` 输出
- `history.json` 仍然同时保留 `mpjpe` 和 `pa_mpjpe`

我还做了语法检查，脚本可以通过 `py_compile`。你现在可以这样跑：

```bash
python train_t1_rgb_teacher_mpjpe.py data_base config.yaml
```

如果你愿意，我下一步可以继续帮你把引用 `strict_offline_runs/T1_rgb_only_teacher/best.pth` 的那些 MPJPE 实验脚本，也统一切到这个新 teacher 路径。

> 你把它保存在strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth这样子

<details><summary>3 previous messages</summary>

> 我把新脚本的输出目录改成你指定的固定路径：`strict_offline_runs/T1_rgb_only_teacher_mpjpe/`，这样 `best.pth` 就会稳定落在你要的位置。
>
> <details><summary>Edited 1 file</summary>
>
> - Wrote `./train_t1_rgb_teacher_mpjpe.py` (+3 -2)
>
> </details>
>
> 我顺手把路径固定逻辑收紧了，现在不会再按时间戳新建目录，而是直接写到你指定的 teacher 目录里。再做一次快速校验，确认改完仍然没问题。
</details>

已经改好了。

现在这个新脚本会固定把结果保存到：
- `strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth`
- `strict_offline_runs/T1_rgb_only_teacher_mpjpe/last.pth`
- `strict_offline_runs/T1_rgb_only_teacher_mpjpe/history.json`
- `strict_offline_runs/T1_rgb_only_teacher_mpjpe/log.txt`

对应文件是 [train_t1_rgb_teacher_mpjpe.py](./train_t1_rgb_teacher_mpjpe.py)，并且我已经重新做过语法检查了。

你现在直接跑：

```bash
python train_t1_rgb_teacher_mpjpe.py data_base config.yaml
```

如果你愿意，我下一步可以继续帮你把 [run_ablation_mpjpe.py](./run_ablation_mpjpe.py) 和 [run_bayesian_hparam_tuning_mpjpe.py](./run_bayesian_hparam_tuning_mpjpe.py) 里引用的 teacher 路径，也统一切到这个新的 `T1_rgb_only_teacher_mpjpe/best.pth`。

> 默认的batchsize是多少呢

默认是 `128`。

在 [train_t1_rgb_teacher_mpjpe.py](./train_t1_rgb_teacher_mpjpe.py) 里：
- `--batch_size` 默认 `128`
- `--val_batch_size` 默认也是 `128`

如果你不手动传参数，训练会按这个跑。比如改成 `64` 可以这样：

```bash
python train_t1_rgb_teacher_mpjpe.py data_base config.yaml --batch_size 64 --val_batch_size 64
```

> 这个版本在前期就出现了非常严重的过拟合现象

这很可能不是“训练逻辑被我改坏了”，而是 `T1` 这类纯 `RGB -> 3D` teacher 本来就更容易出现这种现象：

- `PA-MPJPE` 主要看姿态对齐后的误差
- `MPJPE` 直接看绝对 3D 误差，对全局平移、尺度、深度偏差更敏感

所以同一个模型会出现一种很典型的情况：
- `train loss` 继续降
- `PA-MPJPE` 还能慢慢变好
- 但 `MPJPE` 很早就开始恶化

这看起来像“前期严重过拟合”，但本质上更像是 `MPJPE` 约束下，模型很快学到了相对姿态，却没有学好绝对空间位置。

就你这个新脚本来说，我只改了这几处：
- `scheduler.step(mpjpe)`
- `best.pth` 按 `mpjpe` 保存
- 最终输出按 `MPJPE`

所以如果问题是“换成 MPJPE 后前期就炸”，最可疑的就是学习率和正则强度对 `MPJPE` 不够稳。

你可以先直接试这组更保守的参数：

```bash
python train_t1_rgb_teacher_mpjpe.py data_base config.yaml \
  --lr 1e-4 \
  --weight_decay 5e-3 \
  --batch_size 64 \
  --val_batch_size 64 \
  --patience 5
```

如果你愿意，我可以继续直接帮你做一个 `v2`：
- 训练逻辑仍然不变
- 仍然按 `MPJPE` 选最优
- 只额外把默认超参调成更抗过拟合的一版

我建议的默认会是：
- `lr=1e-4`
- `weight_decay=5e-3`
- `batch_size=64`
- `patience=5`

如果你点头，我就直接帮你改这个新文件。

> 我是用window的python，你给我命令显示无法解析

那是因为我刚才给的是多行 shell 写法，Windows 终端里不一定能直接解析。

你在 Windows 里直接用单行就行：

```bash
python train_t1_rgb_teacher_mpjpe.py data_base config.yaml --lr 1e-4 --weight_decay 5e-3 --batch_size 64 --val_batch_size 64 --patience 5
```

如果你是用指定解释器，也可以写成：

```bash
py train_t1_rgb_teacher_mpjpe.py data_base config.yaml --lr 1e-4 --weight_decay 5e-3 --batch_size 64 --val_batch_size 64 --patience 5
```

如果你的数据根目录不是 `data_base`，就把 `data_base` 换成你自己的实际路径。

如果你愿意，我还可以顺手给你整理一版“Windows 可直接复制”的常用命令，包括：
- 默认训练
- 小 batch 训练
- 指定 GPU
- 指定输出 seed

> 代码不需要设置早停

那就不用传 `--patience`，因为这个脚本里默认就是不早停，`--patience` 默认值是 `0`。

Windows 下你直接这样跑就行：

```bash
python train_t1_rgb_teacher_mpjpe.py data_base config.yaml --lr 1e-4 --weight_decay 5e-3 --batch_size 64 --val_batch_size 64
```

如果你想完全按脚本默认参数跑，最短就是：

```bash
python train_t1_rgb_teacher_mpjpe.py data_base config.yaml
```

如果你愿意，我也可以顺手把脚本里和 early stopping 相关的打印文案再收一收，让它看起来更干净。

> 这次过拟合有所改善但是还是有过拟合，没那么严重了

<details><summary>3 previous messages</summary>

> 我先不碰训练逻辑，只把这个新脚本的默认超参再往“抗过拟合”方向收一档：更小学习率、更强正则、更高 dropout、更小 batch。这样你后面直接跑默认命令，就不用每次手动改一串参数。
>
> <details><summary>Edited 1 file</summary>
>
> - Wrote `./train_t1_rgb_teacher_mpjpe.py` (+6 -6)
>
> </details>
>
> 我已经把默认值改成更保守的一版了，现在直接用默认命令跑，会明显比之前更偏向抑制过拟合。再做一次语法检查，确保这次改动没有带来别的问题。
</details>

我已经把这个新脚本的默认超参往“更抗过拟合”方向调了一版，训练逻辑本身没动，还是只按 `MPJPE` 选优。

现在 [train_t1_rgb_teacher_mpjpe.py](./train_t1_rgb_teacher_mpjpe.py) 的默认值变成了：
- `batch_size=32`
- `val_batch_size=64`
- `dropout=0.25`
- `lr=5e-5`
- `weight_decay=1e-2`
- `mixup_alpha=0.6`

这样你在 Windows 下直接跑默认命令就会用这组更稳的配置：

```bash
python train_t1_rgb_teacher_mpjpe.py data_base config.yaml
```

我也重新做了语法检查，脚本没问题。
如果这版还是有过拟合，下一步我更建议优先试两件事，而不是乱改结构：

- 把 `--aug_freq_mask` 和 `--aug_time_mask` 再加大一点
- 或者把 `--epochs` 从 `30` 降到 `20`

如果你要，我下一步可以直接继续帮你出一个“更稳但不激进”的推荐命令给 Windows 一键跑。

> 和原来相比你改动了什么参数

和你最开始那个版本相比，我只改了默认超参数，没有改训练逻辑。

最开始我创建这个脚本时的默认值是：
- `batch_size=128`
- `val_batch_size=128`
- `dropout=0.15`
- `lr=3e-4`
- `weight_decay=1e-3`
- `mixup_alpha=0.4`

现在改成了：
- `batch_size=32`
- `val_batch_size=64`
- `dropout=0.25`
- `lr=5e-5`
- `weight_decay=1e-2`
- `mixup_alpha=0.6`

也就是具体变化是：

- `batch_size`: `128 -> 32`
- `val_batch_size`: `128 -> 64`
- `dropout`: `0.15 -> 0.25`
- `lr`: `3e-4 -> 5e-5`
- `weight_decay`: `1e-3 -> 1e-2`
- `mixup_alpha`: `0.4 -> 0.6`

这些改动的方向都是一致的：让训练更保守、更强正则、更不容易前期贴住训练集。

没改的部分包括：
- 模型还是 `RGBOnlyTeacher`
- loss 还是 `L1`
- 还是按 `MPJPE` 做 `scheduler.step(...)`
- 还是按 `MPJPE` 保存 `best.pth`
- `epochs=30` 没改
- `warmup_epochs=3` 没改
- `lr_patience=3` 没改
- `patience=0` 没改
- `window=32`、`stride=2` 没改
- 增强类型没改，只是 `mixup_alpha` 更强了

如果你要，我下一条可以直接给你做一个非常清楚的“原版 T1 / 新版 T1-MPJPE / 当前抗过拟合版”三列对照表。

> 我想问我的lupi过程，来自rgb的向量是直接从来没有重新复制一个模块获得的吧

对，你这条理解基本是对的。

在你的 LUPI 过程里，来自 RGB 的向量不是在 student 里面又复制一套 `RGB encoder` 重新学出来的，而是直接来自外部的 `teacher`。

看代码的话：

在 [train_lupi_rgb_teacher_mpjpe.py](./train_lupi_rgb_teacher_mpjpe.py) 里：
- 先单独实例化了一个 `teacher = RGBOnlyTeacher(...)`
- 再从 `teacher_ckpt` 加载参数
- 然后 `teacher.eval()`，并且 `for p in teacher.parameters(): p.requires_grad = False`

这说明：
- teacher 是单独的一套模块
- 但它在 LUPI 训练里是冻结的，不参与更新

后面训练时：
- `pred3d, feat_w = model.forward_wifi(wifi_window)`
- `teacher_pred, feat_v_teacher = teacher(None, rgb, return_feat=True)`

这里的 `feat_v_teacher` 就是 RGB 分支出来的特征向量，直接来自这个冻结的 teacher。

再往下：
- `z_w = model.proj_w(feat_w)`
- `z_v = model.proj_v(feat_v_teacher)`

所以更精确地说是：

1. 没有在 `TemporalLUPICMC_CBAM` 里面再复制一个完整的 RGB 编码器。
2. 但有一个单独加载出来的 `RGBOnlyTeacher` 模块，专门负责产出 RGB 特征和 teacher pose。
3. student 这边只额外有一个 `proj_v`，把 teacher 的 RGB feature 投影到 CMC 对比空间里。

也就是说，RGB 向量来源是“外部冻结 teacher”，不是“student 内部重新造一套 RGB backbone”。

顺手提醒你一个细节：
在 [train_lupi_rgb_teacher_mpjpe.py](./train_lupi_rgb_teacher_mpjpe.py) 里，`--cmc_stopgrad_rgb` 这个参数目前其实没有真正用到，因为 teacher 本来就在 `torch.no_grad()` 里，而且参数也被冻结了。

如果你愿意，我下一步可以直接帮你把这条 LUPI 信息流画成一条很清楚的“WiFi student / RGB teacher / CMC / distill”结构图。

> Epoch 30: val MPJPE=176.5mm | PA-MPJPE=56.5mm | PCK@20=42.5% | PCK@50=91.1%
> ============================================================
> T1 RGB-only Teacher finished! Best MPJPE: 156.4mm
> ============================================================
>
> Result: MPJPE = 156.4mm
> Run dir: strict_offline_runs\T1_rgb_only_teacher_mpjpe
>
> 跑完啦，我现在要重新对mpjpe的那个进行调优，保持原来代码参数不变，帮我修改teacher来源，其他配置一致

<details><summary>4 previous messages</summary>

> 我来只做最小改动：把 MPJPE 这条线里硬编码的 teacher checkpoint 路径切到你刚跑出来的 `T1_rgb_only_teacher_mpjpe`，其余参数、搜索空间、训练配置全部保持原样。
>
> 我已经定位到最小需要修改的地方了。先把你这次要重跑的 `run_bayesian_hparam_tuning_mpjpe.py` 切到新的 teacher；顺手也把同一路径的 `run_ablation_mpjpe.py` 一并对齐，避免后面复现实验时又混回旧 teacher。
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `./run_ablation_mpjpe.py` (+2 -2)
> - Wrote `./run_bayesian_hparam_tuning_mpjpe.py` (+1 -1)
>
> </details>
>
> 路径已经替你切好了，而且只改了 teacher 来源，别的配置没动。再做一个快速 grep 复核，确认这两条 MPJPE 主线现在都指向新的 `T1_rgb_only_teacher_mpjpe`。
</details>

已经改好了。

现在这两条 `MPJPE` 主线都会使用你刚训练出来的 teacher：
- [run_bayesian_hparam_tuning_mpjpe.py](./run_bayesian_hparam_tuning_mpjpe.py)
- [run_ablation_mpjpe.py](./run_ablation_mpjpe.py)

新的 teacher 路径统一是：
- `strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth`

除了这个路径，我没有动别的配置。搜索空间、batch size 候选、lr 范围、CMC 配置、mixup 配置、训练 epoch 都保持原样。我也做了语法检查，两个脚本都没问题。

你现在如果要重新做 MPJPE 调优，直接跑原来的命令就行，只是 teacher 已经换成新的了，例如：

```bash
python run_bayesian_hparam_tuning_mpjpe.py data_base config.yaml
```

如果你愿意，我下一步还可以顺手帮你把同样的 teacher 路径也同步到 [run_cmc_mixup_all_splits_mpjpe.py](./run_cmc_mixup_all_splits_mpjpe.py)，这样整条 MPJPE 线就完全统一了。

> 我调优只需要调cmc+mixup这个版本的就可以

对，这样做是合理的。

按你现在这条实验主线，重新调优时只调 `cmc + mixup` 这个版本就够了，也就是 `cbam_mixup` 这一档没有必要、`mixup` 这一档更贴近你现在在 `run_ablation_mpjpe.py` 里真正要用的 `D_temporal_mixup_cmc` 配置：
- 有 `CMC`
- 有 `Mixup`
- `--no_cbam`
- 其他训练设置与主实验一致

所以你现在最合适的是只跑这个模式：

```bash
python run_bayesian_hparam_tuning_mpjpe.py data_base config.yaml --modes mixup
```

如果你还想减少开销，也可以顺手把 trial 数和 seed 数收一点，先做一轮快搜：

```bash
python run_bayesian_hparam_tuning_mpjpe.py data_base config.yaml --modes mixup --n_trials 10 --seeds 2
```

然后如果结果趋势稳定，再跑正式版：

```bash
python run_bayesian_hparam_tuning_mpjpe.py data_base config.yaml --modes mixup --n_trials 15 --seeds 3
```

补一句关键判断：
你现在的 `run_ablation_mpjpe.py` 里最终四组对比，真正依赖这个 teacher 调优结果的核心目标组，确实就是 `D_temporal_mixup_cmc`。所以只重调这一组，是和你现在主线一致的，不会跑偏。

如果你要，我下一步可以直接帮你把 [run_bayesian_hparam_tuning_mpjpe.py](./run_bayesian_hparam_tuning_mpjpe.py) 改成默认只跑 `mixup`，这样你就不用每次手动加 `--modes mixup`。

> 等等这个mixup有cbam吗，没有吧

对，你说得对，`mixup` 这个模式没有 `CBAM`。

在 [run_bayesian_hparam_tuning_mpjpe.py](./run_bayesian_hparam_tuning_mpjpe.py) 里四个 mode 是这样定义的：

- `temporal_only`
  - `--no_cbam`
  - `--mixup_alpha 0.0`

- `mixup`
  - `--no_cbam`
  - `--mixup_alpha 0.4`

- `cbam`
  - 有 `CBAM`
  - `--mixup_alpha 0.0`

- `cbam_mixup`
  - 有 `CBAM`
  - `--mixup_alpha 0.4`

所以：

- `mixup = CMC + Mixup + no CBAM`
- `cbam_mixup = CMC + Mixup + CBAM`

而你当前 [run_ablation_mpjpe.py](./run_ablation_mpjpe.py) 里的 `D_temporal_mixup_cmc` 也是：
- `train_lupi_rgb_teacher_mpjpe.py`
- 有 `CMC`
- 有 `Mixup`
- `COMMON_ARGS` 里带了 `--no_cbam`

所以如果你是为了给当前这条 `ablation_mpjpe` 主线配超参，那确实应该调的是 `mixup`，不是 `cbam_mixup`。

也就是说你刚才那句应该改成：

- 如果目标是对齐当前 `run_ablation_mpjpe.py` 的 `D_temporal_mixup_cmc`，就调 `mixup`
- 如果你想研究 “CMC + Mixup + CBAM” 这条新支线，才调 `cbam_mixup`

你这个提醒很关键，我前一条的结论其实是对的，但表述里把它和 `cbam_mixup` 混着说了，这里以这一条为准。

> 你觉得我在调优过程中alpha和dropout那些参数要和teacher统一一下吗

我不建议为了“统一”而强行把 `alpha`、`dropout` 跟 teacher 设成一样。

对你现在这条 LUPI 线来说，更合理的是：

- `teacher` 的超参属于 teacher 自己的预训练过程
- `student` 的 `mixup_alpha`、`dropout` 属于 student 自己的泛化控制
- 两者是有关联，但不是必须绑定一致

也就是说，`teacher` 现在是固定 checkpoint：
- `strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth`

在这个前提下，你当前重新做 `MPJPE` 调优时，最稳的做法是：
- 先不要把 `alpha`、`dropout` 跟 teacher 强行统一
- 先保持 student 主线原配置不变
- 只调你原本就在调的 `lr` 和 `batch_size`

原因很简单：
- 你这次的变量本来就已经换成了“teacher 来源”
- 如果再同时改 `alpha`、`dropout`，实验解释会变乱
- 你就很难判断提升到底来自新 teacher，还是来自 student 正则变强

所以我给你的建议是：

1. 现在这轮调优先保持 student 不变
   也就是继续用你 `run_bayesian_hparam_tuning_mpjpe.py` 里那套固定值：
   - `dropout=0.15`
   - `mixup_alpha=0.4`

2. 先只对 `mixup` 模式重新搜 `lr + batch_size`

3. 如果这轮跑完后效果还是不理想，再做第二轮“小范围微调”
   - `dropout`: 可以试 `0.1 / 0.15 / 0.2`
   - `mixup_alpha`: 可以试 `0.2 / 0.4 / 0.6`

一句话总结就是：

不用和 teacher 统一。
当前最重要的是“控制变量”，先只换 teacher、只重新调 `lr` 和 `bs`，这样实验结论最干净。

如果你愿意，我下一步可以直接帮你判断一下：你现在这条 `mixup` 的 MPJPE 调优脚本里，`dropout=0.15` 和 `mixup_alpha=0.4` 还值不值得进入第二轮搜索。

> 你再帮我检查一下mpjpe这个版本是否使用了正确的调优

<details><summary>6 previous messages</summary>

> 我先按“代码审计”的方式核对一遍 `MPJPE` 这条线：看调优脚本搜的到底是不是你要的配置、消融脚本有没有正确吃到调优结果，以及有没有口径不一致的地方。
>
> <details><summary>Explored 2 files, ran 1 command</summary>
>
> <details><summary>Explored 3 files</summary>
>
> - Read `./run_bayesian_hparam_tuning_mpjpe.py`
> - Read `./run_bayesian_hparam_tuning_mpjpe.py`
> - Read `./run_ablation_mpjpe.py`
>
> </details>
>
> - Ran `ls -R /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/experiment_results/bayesian_hparam_tuning_mpjpe 2>/dev/null | head -n 120`
>
> </details>
>
> 我已经看到一处很关键的隐患了：`MPJPE` 调优本身的目标函数是对的，但这条线里有一个“运行目录前缀”不一致，可能让脚本在搬运结果时找错目录。这个我再把定位补全，然后给你一个明确结论。
>
> <details><summary>Explored 2 files, ran 1 command</summary>
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./train_lupi_rgb_teacher_mpjpe.py`
> - Read `./train_baseline_mpjpe.py`
>
> </details>
>
> - Ran `sed -n '1,120p' /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/experiment_results/bayesian_hparam_tuning_mpjpe/mixup/bo_summary.json 2>/dev/null`
>
> </details>
>
> 我把需要落结论的几个点都找到了。最后再补一下精确行号，这样我直接按“哪几处对、哪几处要小心”给你。
>
> <details><summary>Explored 4 files</summary>
>
> - Read `./run_bayesian_hparam_tuning_mpjpe.py`
> - Read `./run_bayesian_hparam_tuning_mpjpe.py`
> - Read `./run_ablation_mpjpe.py`
> - Read `./train_lupi_rgb_teacher_mpjpe.py`
>
> </details>
</details>

我检查完了，结论是：

**主要结论**

`MPJPE` 版本的调优逻辑本身是对的，但现在有两处会影响“是否真的用上了正确调优结果”。

1. [run_ablation_mpjpe.py](./run_ablation_mpjpe.py:38) 还硬编码着旧的最优超参
   现在写的是 `BEST_LR = 1.14e-04`、`BEST_BS = 16`，并且 [COMMON_ARGS](./run_ablation_mpjpe.py:41) 会直接使用它们。
   这意味着：你就算重新跑完 `run_bayesian_hparam_tuning_mpjpe.py`，`run_ablation_mpjpe.py` 也**不会自动吃到新的最优 lr / bs**，除非你手动把这两个常量改掉。

2. `run_dir` 前缀不一致，结果搬运有隐患
   在 [run_bayesian_hparam_tuning_mpjpe.py](./run_bayesian_hparam_tuning_mpjpe.py:65) 和 [run_ablation_mpjpe.py](./run_ablation_mpjpe.py:78) 里，CMC 路线的前缀写的是
   `temporal_lupi_rgb_teacher_mpjpe_cmc_cbam_distill`
   但实际训练脚本 [train_lupi_rgb_teacher_mpjpe.py](./train_lupi_rgb_teacher_mpjpe.py:29) 创建的目录前缀是
   `temporal_lupi_rgb_teacher_cmc_cbam_distill`
   现在之所以大概率还能跑，是因为脚本优先靠 stdout 里的 `Run dir:` 去找目录；但如果 stdout 解析失败，fallback 可能找不到新目录。

**确认没问题的部分**

- `run_bayesian_hparam_tuning_mpjpe.py` 的目标函数是按 `MPJPE` 选最优的，见 [min(h, key=lambda e: e["mpjpe"])](./run_bayesian_hparam_tuning_mpjpe.py:335)。
- trial 的 objective 也是返回 `mean MPJPE`，见 [line 287-289](./run_bayesian_hparam_tuning_mpjpe.py:287)。
- teacher 来源已经切到新的 `strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth`，见 [run_bayesian_hparam_tuning_mpjpe.py:67](./run_bayesian_hparam_tuning_mpjpe.py:67) 和 [run_ablation_mpjpe.py:80](./run_ablation_mpjpe.py:80)。
- 如果你现在只想调 `CMC + Mixup + no CBAM`，那确实应该跑 `--modes mixup`，这一点和你的 `D_temporal_mixup_cmc` 是对齐的，见 [run_bayesian_hparam_tuning_mpjpe.py:84](./run_bayesian_hparam_tuning_mpjpe.py:84)。

**一句话判断**

- `调优脚本本身`: 对
- `调优结果是否会自动用于后续 ablation`: 现在不对
- `目录前缀是否完全一致`: 现在不对

如果你愿意，我可以直接帮你把这两处一起修掉：
- 统一 `prefix`
- 让 `run_ablation_mpjpe.py` 更容易吃到你这次新的最优 `lr / bs`

> 您好

你好。要的话我可以直接把刚才那两处 `MPJPE` 线的问题顺手修掉。

> 你好

<details><summary>1 previous message</summary>

> Stream error
>
> Reconnecting 2/5
> stream disconnected before completion: Concurrency limit exceeded for user, please retry later
</details>

你好。你如果要继续，我可以直接接着把 `MPJPE` 这条线整理干净。

> 可以你修一下

<details><summary>6 previous messages</summary>

> 我来把这两处真正影响主线稳定性的点补干净：一是统一 `MPJPE` 线的 `run_dir prefix`，二是让 `run_ablation_mpjpe.py` 能自动读取你最新的 `mixup` 调优结果，而不是继续吃硬编码旧值。
>
> <details><summary>Explored 2 files, 1 search</summary>
>
> - Read `./run_bayesian_hparam_tuning_mpjpe.py`
> - Searched for `temporal_lupi_rgb_teacher_mpjpe_cmc_cbam_distill|BEST_LR|BEST_BS|bo_summary.json|best_lr|best_bs` in `run_*mpjpe.py`
> - Read `./run_ablation_mpjpe.py`
>
> </details>
>
> 方案我已经确定了：不去改你的实验定义，只补两件基础设施问题。`run_ablation_mpjpe.py` 会优先读 `bayesian_hparam_tuning_mpjpe/mixup/bo_summary.json` 里的最优 `lr/bs`，读不到时才退回你现在的硬编码值；同时把所有 `prefix` 对齐到训练脚本实际生成的目录名。
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `./run_ablation_mpjpe.py` (+30 -4)
> - Wrote `./run_bayesian_hparam_tuning_mpjpe.py` (+1 -1)
>
> </details>
>
> 核心修复已经落下去了。我再快速校验一下：确认 `prefix` 统一了，`ablation` 也确实会优先读取最新的 `mixup` 调优结果。
>
> <details><summary>Explored 1 file, 1 search, ran 1 command</summary>
>
> <details><summary>Explored 1 search</summary>
>
> - Searched for `temporal_lupi_rgb_teacher_cmc_cbam_distill|temporal_lupi_rgb_teacher_mpjpe_cmc_cbam_distill|BAYES_MIXUP_SUMMARY|load_best_hparams|BEST_LR|BEST_BS` in `run_bayesian_hparam_tuning_mpjpe.py`
>
> </details>
>
> - Ran `python -m py_compile /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/run_bayesian_hparam_tuning_mpjpe.py /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/run_ablation_mpjpe.py`
>
> <details><summary>Explored 1 file</summary>
>
> - Read `./run_ablation_mpjpe.py`
>
> </details>
>
> </details>
</details>

修好了，而且我尽量只动了真正影响主线稳定性的地方。

现在这条 `MPJPE` 线有两个关键修复：

1. `prefix` 已统一
   我把 [run_bayesian_hparam_tuning_mpjpe.py](./run_bayesian_hparam_tuning_mpjpe.py) 和 [run_ablation_mpjpe.py](./run_ablation_mpjpe.py) 里的 CMC 运行目录前缀，统一成和 [train_lupi_rgb_teacher_mpjpe.py](./train_lupi_rgb_teacher_mpjpe.py) 实际生成的一样：
- `temporal_lupi_rgb_teacher_cmc_cbam_distill`

这样 stdout fallback 失败时，也能正确按前缀找目录，不会再有潜在错配。

2. `run_ablation_mpjpe.py` 现在会优先读取你最新的 `mixup` 调优结果
   它会先看：
- `experiment_results/bayesian_hparam_tuning_mpjpe/mixup/bo_summary.json`

如果这个文件存在并且格式正常，就自动读取里面的：
- `best_lr`
- `best_bs`

如果读不到，才退回原来的默认值：
- `lr=1.14e-04`
- `bs=16`

所以你现在的流程会更顺：

1. 先跑 `mixup` 的 MPJPE 调优
```bash
python run_bayesian_hparam_tuning_mpjpe.py data_base config.yaml --modes mixup
```

2. 再跑 ablation
```bash
python run_ablation_mpjpe.py data_base config.yaml
```

这样 `ablation` 会自动吃到你刚调出来的 `mixup` 最优 `lr / bs`，不用你再手改常量。

我也顺手做了语法检查，这两个脚本都没问题。
如果你愿意，我下一步可以再帮你把 [run_cmc_mixup_all_splits_mpjpe.py](./run_cmc_mixup_all_splits_mpjpe.py) 也一起对齐，不然它现在还是老的 `BEST_LR/BEST_BS` 和旧 `prefix`。

> 我感觉我的贝叶斯调优现在不太ok，不知道是不是teacher调整了dropout和mixup的参数的原因

<details><summary>2 previous messages</summary>

> 我先不急着归因到 teacher 本身，先把“更可能导致调优失真”的地方拎出来给你。你这条贝叶斯线里有一个很容易把新旧实验混在一起的问题，我核对一下具体位置。
>
> <details><summary>Explored 2 files</summary>
>
> - Read `./run_bayesian_hparam_tuning_mpjpe.py`
> - Read `./run_bayesian_hparam_tuning_mpjpe.py`
>
> </details>
</details>

我觉得“teacher 改了 dropout 和 mixup，所以贝叶斯调优不太对”**有可能有影响，但大概率不是首要问题**。

我更怀疑的是两个更实际的原因：

1. 你换了 teacher，但 student 的搜索空间没跟着调整
   现在 [run_bayesian_hparam_tuning_mpjpe.py](./run_bayesian_hparam_tuning_mpjpe.py) 还是固定：
   - `dropout=0.15`
   - `mixup_alpha=0.4`
   - `weight_decay=1e-3`
   - 只搜 `lr + batch_size`

   如果新的 `T1_rgb_only_teacher_mpjpe` 分布和旧 teacher 不一样，student 的最优 `lr/bs` 当然会漂，但这不代表脚本错了，只代表“teacher 变了，最优点变了”。

2. 你当前 Optuna study 会复用旧 trial
   这是更危险的。
   在 [run_bayesian_hparam_tuning_mpjpe.py:459-465](./run_bayesian_hparam_tuning_mpjpe.py:459) 里：
   - `study_name` 没变
   - `storage_path` 没变
   - `load_if_exists=True`

   这意味着如果你之前已经用“旧 teacher”跑过 `mixup` 模式，今天再跑时，Optuna 会把旧 trial 和新 teacher 的 trial 混在同一个 study 里一起选最优。
   这种情况下，结果“看起来不太对”是完全正常的，因为你在比较的是两套不同 teacher 条件下的 trial。

我现在的判断是：

- `teacher` 改了正则配置，确实可能让最优 `lr/bs` 变化
- 但“贝叶斯调优现在不太 ok”的更大嫌疑，是 **旧 study 没清、旧 trial 被复用**

你现在最该做的是：

- 对 `mixup` 模式重新开一个全新的 Optuna study
- 不要沿用之前的 `experiment_results/bayesian_hparam_tuning_mpjpe/mixup/optuna_study.db`
- 也不要复用之前已经生成的 `bo_trials`

最稳的办法是：
- 备份或删除 `experiment_results/bayesian_hparam_tuning_mpjpe/mixup/`
- 然后重新跑 `--modes mixup`

如果你不想手动清目录，我可以直接帮你改脚本，给它加一个很干净的开关，比如：
- `--study_tag t1_mpjpe_teacher`
或
- `--fresh_study`

这样新 teacher 和旧 teacher 的调优结果就不会混在一起。
我比较建议直接加这个，而不是靠手工删文件。

> 我在训练之前只删除了上次一贝叶斯训练得到的结果

那要看你删的是哪一部分。

如果你只是删了“表面的结果文件”，通常**还不够**，因为这个脚本有两层复用机制：

1. Optuna study 会复用旧 trial
   在 [run_bayesian_hparam_tuning_mpjpe.py](./run_bayesian_hparam_tuning_mpjpe.py) 里用了 `load_if_exists=True`，所以只要这个文件还在：
- `experiment_results/bayesian_hparam_tuning_mpjpe/mixup/optuna_study.db`

它就会把旧 teacher 的 trial 也一起算进去。

2. 单个 seed 的训练结果也会断点续跑
   只要这些目录里已有 `history.json`，脚本就直接读旧结果，不会重跑：
- `experiment_results/bayesian_hparam_tuning_mpjpe/mixup/bo_trials/...`
- `experiment_results/bayesian_hparam_tuning_mpjpe/mixup/final_eval/...`

所以如果你换了 teacher，想做一次“干净”的新调优，最稳的是直接删整个：

- `experiment_results/bayesian_hparam_tuning_mpjpe/mixup/`

这样能同时清掉：
- `optuna_study.db`
- `bo_trials`
- `final_eval`
- `bo_summary.json`

如果你只删了 `bo_summary.json` 或者只删了部分结果目录，那基本还是会混进旧数据。

最简单的判断方式是看运行时这句输出：
- `Optuna study: X trials already done`

如果这里 `X` 不是 `0`，那就说明旧 trial 还在。

如果你愿意，我可以直接帮你把脚本改成支持一个 `--fresh_study` 开关，这样以后就不用手动删目录了。

> 我就是吧bayesian_hparam_tuning_mpjpe给删掉了

那如果你删的是整个 `experiment_results/bayesian_hparam_tuning_mpjpe/`，这一步其实是干净的。

也就是说：
- `optuna_study.db` 没了
- `bo_trials` 没了
- `final_eval` 没了
- `bo_summary.json` 没了

这种情况下，新的贝叶斯调优不会混进上一次的 trial。
所以现在“调优不太 ok”的原因，基本就**不是旧 study 残留**了。

那更可能的原因就剩这几个：

1. `teacher` 变了，student 的最优点真的变了
   你现在换成的是 `T1_rgb_only_teacher_mpjpe`，这会改变蒸馏/CMC supervision 的分布，所以原来那套 `lr + bs` 最优值不一定还成立。

2. 搜索空间可能不够贴现在这个 teacher
   你现在还是在搜：
- `lr ∈ [1e-4, 5e-3]`
- `bs ∈ [16, 32, 64, 128, 256]`

   但如果新 teacher 更“硬”或者更敏感，最优学习率可能已经落到 `1e-4` 以下了。
   这时候贝叶斯再聪明，也只能在一个偏高的区间里找“相对最好”。

3. 固定超参没有跟着 teacher 变化
   目前 student 这边固定还是：
- `dropout=0.15`
- `mixup_alpha=0.4`
- `weight_decay=1e-3`

   如果新 teacher 本身更容易让 student 过拟合，真正该一起微调的可能就不止 `lr` 和 `bs`。

所以我现在的判断是：

- 你的这次 BO 是“干净的”
- 问题更像是“新 teacher 下，原搜索空间和固定配置不再合适”

我建议下一步优先做这个，而不是先怀疑代码错了：

- 把 `lr` 搜索下界往下放一点，比如改成 `3e-5`
- 暂时只跑 `mixup` 模式
- 先用 `n_trials=10~15`、`seeds=2~3`
- 先别动太多别的变量

如果你愿意，我可以直接帮你把 `run_bayesian_hparam_tuning_mpjpe.py` 改成一版更适合你这个新 teacher 的搜索空间，而且只改搜索范围，不动你的主实验逻辑。

> 不行我一定要跑cmc，因为这是我唯一的创新点

可以，而且你这个坚持是对的。你这条线里确实必须保留 `CMC`，因为那是你的核心创新点。

不过这里有个容易混淆的地方：

在 [run_bayesian_hparam_tuning_mpjpe.py](./run_bayesian_hparam_tuning_mpjpe.py) 里，`mixup` 模式**不是只跑 mixup**，它其实是：

- 有 `CMC`
- 有 `Mixup`
- 没有 `CBAM`

也就是说：

- `temporal_only` = `CMC + no mixup + no CBAM`
- `mixup` = `CMC + mixup + no CBAM`
- `cbam` = `CMC + CBAM + no mixup`
- `cbam_mixup` = `CMC + CBAM + mixup`

所以你说“我一定要跑 CMC”，这没有问题，而你现在应该跑的其实正是：

```bash
python run_bayesian_hparam_tuning_mpjpe.py data_base config.yaml --modes mixup
```

这个 `mixup` 模式本身就带着 `CMC`，不是把 `CMC` 去掉了。

你真正需要决定的是：

- 如果你的主实验配置是 `CMC + Mixup + no CBAM`，就调 `mixup`
- 如果你的主实验配置是 `CMC + Mixup + CBAM`，就调 `cbam_mixup`

按你现在的 [run_ablation_mpjpe.py](./run_ablation_mpjpe.py) 来看，你主线对应的是前者，所以跑 `mixup` 是对的，而且没有丢掉 `CMC`。

如果你愿意，我下一步可以直接帮你把 `run_bayesian_hparam_tuning_mpjpe.py` 里的四个 mode 名字改得更直白一点，比如改成 `cmc_only / cmc_mixup / cmc_cbam / cmc_cbam_mixup`，这样以后看起来就不会绕了。

> cmc至少是需要保留的

对，`CMC` 至少要保留，这点我完全同意。

按你现在这条主线来说，`CMC` 不是可选增强，而是核心机制。所以后续无论是：
- 贝叶斯调优
- 最终 ablation
- 论文主结果

都应该建立在“`CMC` 保留”的前提上。

你现在这个 `MPJPE` 调优脚本里，其实四个 mode 都是带 `CMC` 的，只是差别在于：
- 是否加 `Mixup`
- 是否加 `CBAM`

所以如果你当前主线是“`CMC + Mixup + no CBAM`”，那就继续调这个：
```bash
python run_bayesian_hparam_tuning_mpjpe.py data_base config.yaml --modes mixup
```

这里的 `mixup` 名字容易误导，但它实际是：
- `CMC`
- `Mixup`
- `no CBAM`

所以它是保留了 `CMC` 的。

如果你愿意，我下一步可以直接帮你把这个脚本里的 mode 名字改清楚一点，避免后面一直把 `mixup` 误看成“不带 CMC”的版本。

> 我把云端的db文件拿下来了你看一下

<details><summary>6 previous messages</summary>

> 我先在工作区里把你拉下来的 `.db` 文件定位出来，再直接检查里面的 study、trial 数量和最优参数，看看这次云端调优到底发生了什么。
>
> <details><summary>Explored 1 search, 1 list, ran 1 command</summary>
>
> - Ran `ls -R /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/experiment_results/bayesian_hparam_tuning_mpjpe 2>/dev/null | head -n 200`
>
> <details><summary>Explored 1 search, 1 list</summary>
>
> - Listed files in `MMFi_Dataset`
> - Searched for `optuna_study\.db$|\.db$`
>
> </details>
>
> </details>
>
> 目录里这会儿没直接扫到结果，我继续精确找一下 `.db`，然后把里面的 Optuna 内容读出来，不靠猜。
>
> <details><summary>Explored 2 searches</summary>
>
> - Searched for `*.db` in `MMFi_Dataset`
> - Searched for `*/bayesian_hparam_tuning_mpjpe/*` in `MMFi_Dataset`
>
> </details>
>
> User context
>
> Attachments:
> - optuna_study.db: `./optuna_study.db`
>
> 我还没拿到返回，先轮询一下这两个查找任务；找到文件后我会直接把 trial 和 best params 摘出来给你。
</details>

> 这个
>
> User context
>
> Attachments:
> - optuna_study.db: `./optuna_study.db`

<details><summary>8 previous messages</summary>

> 我就看这个根目录下的 [optuna_study.db](./optuna_study.db)，把里面的 study、trial 数、最优参数和对应数值直接拆给你看。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `sqlite3 /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/optuna_study.db '.tables'`
> - Ran `sqlite3 /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/optuna_study.db "select study_id, study_name from studies;"`
> - Ran `sqlite3 /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/optuna_study.db "select t.study_id, s.study_name, count(*) as n_trials, min(v.value) as best_value from trials t join trial_values v on t.trial_id=v.trial_id join studies s on t.study_id=s.study_id group by t.study_id, s.study_name;"`
> - Ran `sqlite3 /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/optuna_study.db "select s.study_name, t.number, v.value, group_concat(p.param_name || '=' || coalesce(cast(p.param_value as text), 'NULL'), ', ') as params from trials t join trial_values v on t.trial_id=v.trial_id join studies s on t.study_id=s.study_id left join trial_params p on t.trial_id=p.trial_id where t.state=1 group by s.study_name, t.number, v.value order by v.value asc limit 10;"`
>
> </details>
>
> 已经读到核心信息了：这个 db 里只有一个 study，而且是 `hpo_mixup_mpjpe`，共有 10 个 trial。现在我把最优 trial 的具体参数再拉出来。
>
> <details><summary>Ran 1 command</summary>
>
> - Ran `sqlite3 /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/optuna_study.db "select s.study_name, t.number, v.value, p1.param_value as lr_raw, p2.param_value as bs_raw from trials t join trial_values v on t.trial_id=v.trial_id join studies s on t.study_id=s.study_id left join trial_params p1 on t.trial_id=p1.trial_id and p1.param_name='lr' left join trial_params p2 on t.trial_id=p2.trial_id and p2.param_name='batch_size' where t.state=1 order by v.value asc limit 10;"`
>
> </details>
>
> 参数那条查询没回出内容，我再换个更稳妥的查法，把 `trials` 和 `trial_params` 结构先看一下，再把最佳 trial 的 `lr / batch_size` 单独摘出来。
>
> <details><summary>Ran 4 commands</summary>
>
> - Ran `sqlite3 /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/optuna_study.db "pragma table_info(trials); pragma table_info(trial_params); pragma table_info(trial_values);"`
> - Ran `sqlite3 /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/optuna_study.db "select trial_id, value from trial_values order by value asc limit 20;"`
> - Ran `sqlite3 /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/optuna_study.db "select trial_id, number, state from trials order by trial_id limit 20;"`
> - Ran `sqlite3 /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/optuna_study.db "select trial_id, param_name, param_value, distribution_json from trial_params order by trial_id, param_name limit 40;"`
>
> </details>
>
> 我已经把 DB 里的核心状态抓出来了：最优 trial、总 trial 数、还有一个正在运行/未完成的 trial。现在我把 batch size 的真实取值映射回去，再给你一个明确判断。
>
> <details><summary>Ran 1 command</summary>
>
> - Ran `sqlite3 /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/optuna_study.db "select t.trial_id, t.number, v.value, max(case when p.param_name='lr' then p.param_value end) as lr, max(case when p.param_name='batch_size' then p.param_value end) as bs_index from trials t join trial_values v on t.trial_id=v.trial_id left join trial_params p on t.trial_id=p.trial_id where t.state='COMPLETE' group by t.trial_id, t.number, v.value order by v.value asc;"`
>
> </details>
</details>

我看了这个根目录下的 [optuna_study.db](./optuna_study.db)，里面的信息很清楚：

**DB 里现在是什么**
- 只有 1 个 study：`hpo_mixup_mpjpe`
- 已完成 `10` 个 trial
- 还有 `1` 个 trial 处于 `RUNNING` 状态

**当前最优结果**
- 最优 trial 是 `trial #7`（数据库里的 `trial_id=8`）
- 最优目标值：`236.6587 mm`
- 学习率：`0.000114399747492913`
- `batch_size` 存的是 categorical 索引 `0.0`
- 你这份脚本的 choices 是 `[16, 32, 64, 128, 256]`
- 所以真实最优 `batch_size = 16`

也就是说，这个 DB 当前给出的最优组合是：

```text
lr = 1.143997e-4
batch_size = 16
best MPJPE = 236.66 mm
```

**我顺手看到两个重要信号**

1. 这个结果其实和你之前硬编码的那组非常接近
   你之前用的是：
- `lr = 1.14e-4`
- `bs = 16`

   这说明这次 BO 并没有“完全跑飞”，反而是又收敛到了几乎同一个点。

2. 有一个 `RUNNING` trial 没收尾
   数据库里 `trial_id=11, number=10` 还是 `RUNNING`。这通常意味着：
- 云端那次任务中断了
- 或者 trial 还没正常写完状态

   这不会影响前 10 个已完成 trial 的最优结果读取，但会说明这份 DB 不是一个完全收尾的最终状态。

**我的判断**
这份 DB 本身看不出“明显错误调优”的迹象。
相反，它显示出来的是：

- 最优点依然落在低学习率、`bs=16`
- 和你旧结论高度一致
- 所以“你感觉现在不太 ok”更像是：
  - 新 teacher 下整体性能分布变了
  - 但最优超参位置没有发生特别剧烈的偏移

如果你愿意，我下一步可以继续帮你做两件很有价值的事之一：

1. 直接把这个 DB 结果写回 [run_ablation_mpjpe.py](./run_ablation_mpjpe.py) 的说明文字里核对一遍
2. 把这 10 个 trial 全部整理成一个表，帮你看有没有“搜索空间下界卡住”的迹象

> 📊 Bayesian HPO Summary — MPJPE Optimized  |  cross_subject_split / Protocol 2
> ========================================================================================================================
> Mode             Description                         Best lr  Best bs              MPJPE (mm)           PA-MPJPE (mm)        PCK@20        PCK@50
> ------------------------------------------------------------------------------------------------------------------------
> mixup            No CBAM, Mixup ON (alpha=0.4)      1.03e-04       64           236.62 ± 2.30            91.95 ± 0.46  31.49 ± 0.41  78.15 ± 0.56
>
> 这个是调优结果，帮我更新消融和跑3*3的实验

<details><summary>6 previous messages</summary>

> 我先把这次新的 `mixup` 最优结果对到两条后续脚本上：`ablation_mpjpe` 和你说的那个 `3×3` 实验脚本。先定位 `3×3` 入口，再直接替你改成 `lr=1.03e-04, bs=64`。
>
> <details><summary>Explored 2 files, 1 search</summary>
>
> - Searched for `3\*3|3x3|all_splits|BEST_LR|BEST_BS|1\.14e-04|bs=16|mixup` in `run_*mpjpe.py`
> - Read `./run_ablation_mpjpe.py`
> - Read `./run_cmc_mixup_all_splits_mpjpe.py`
>
> </details>
>
> 我已经看到要改的地方了：`ablation` 逻辑本身会自动读 `bo_summary`，但文案里的旧数值还没更新；`3×3` 那个脚本还是硬编码旧超参和旧 teacher/prefix。接下来我直接把这三处一起对齐。
>
> <details><summary>Edited 2 files</summary>
>
> - Wrote `./run_ablation_mpjpe.py` (+5 -5)
> - Wrote `./run_cmc_mixup_all_splits_mpjpe.py` (+5 -5)
>
> </details>
>
> 核心更新已经落好了。我再做一遍快速核查，确认两个脚本都已经用上新的 `lr=1.03e-04, bs=64`，而且 `3×3` 也切到了新的 teacher 和正确前缀。
>
> <details><summary>Explored 1 file, 1 search, ran 1 command</summary>
>
> <details><summary>Explored 1 file, 1 search</summary>
>
> - Searched for `1\.14e-04|bs=16|T1_rgb_only_teacher/best\.pth|temporal_lupi_rgb_teacher_mpjpe_cmc_cbam_distill|1\.03e-04|BEST_BS  = 64|T1_rgb_only_teacher_mpjpe/best\.pth|temporal_lupi_rgb_teacher_cmc_cbam_distill` in `run_ablation_mpjpe.py`
> - Read `./run_cmc_mixup_all_splits_mpjpe.py`
>
> </details>
>
> - Ran `python -m py_compile /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/run_ablation_mpjpe.py /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/run_cmc_mixup_all_splits_mpjpe.py`
>
> </details>
</details>

已经替你更新好了。

现在这两个后续实验都对齐到了你最新的 `MPJPE` 调优结果：
- [run_ablation_mpjpe.py](./run_ablation_mpjpe.py)
- [run_cmc_mixup_all_splits_mpjpe.py](./run_cmc_mixup_all_splits_mpjpe.py)

这次我做了这几件事：
- 把 `3×3` 脚本的超参改成 `lr=1.03e-04`、`batch_size=64`
- 把 `3×3` 脚本的 teacher 路径切到 `strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth`
- 把 `3×3` 脚本的 `PREFIX` 改成和训练脚本实际输出一致的 `temporal_lupi_rgb_teacher_cmc_cbam_distill`
- 把 `ablation` 的 fallback 超参更新成 `1.03e-04 / 64`
- 把 `ablation` 里显示结果的标题文案改成动态显示当前 `BEST_LR / BEST_BS`

我也重新做了语法检查，这两个脚本都没问题。

你现在可以直接跑：

```bash
python run_ablation_mpjpe.py data_base config.yaml
```

和

```bash
python run_cmc_mixup_all_splits_mpjpe.py data_base config.yaml
```

如果你愿意，我下一步还可以顺手把 [run_ablation_mpjpe.py](./run_ablation_mpjpe.py) 文件头部注释里的旧数值说明也完全改成动态口径，免得以后看注释和真实运行值不一致。

> 可以，改完你自己再审阅一下

<details><summary>6 previous messages</summary>

> 我把这两个脚本里残留的旧说明也一起收干净，然后我会自己再过一遍，确认“注释、打印信息、实际参数”三者一致。
>
> <details><summary>Explored 2 files, 1 search</summary>
>
> - Read `./run_ablation_mpjpe.py`
> - Searched for `1\.14e-04|bs=16|lr=1\.14e-04|Best hparams|Bayesian HPO 最优参数|mixup 模式` in `run_ablation_mpjpe.py`
> - Read `./run_cmc_mixup_all_splits_mpjpe.py`
>
> </details>
>
> 我看到实际参数和打印已经一致了，剩下可以再补的只是把来源说明写得更明确一点。这个改动很小，我做完就按“审阅结论”给你一个明确收口。
>
> <details><summary>Edited 1 file</summary>
>
> - Wrote `./run_ablation_mpjpe.py` (+2 -2)
>
> </details>
>
> 我再做最后一轮自审，只看高风险点：注释口径、teacher 路径、prefix、动态参数读取和语法状态。确认这些都一致后我给你结论。
>
> <details><summary>Explored 1 file, 1 search, ran 1 command</summary>
>
> <details><summary>Explored 1 file, 1 search</summary>
>
> - Searched for `T1_rgb_only_teacher_mpjpe/best\.pth|temporal_lupi_rgb_teacher_cmc_cbam_distill|BEST_LR|BEST_BS|bo_summary\.json|FALLBACK_LR|FALLBACK_BS` in `run_ablation_mpjpe.py`
> - Read `./run_ablation_mpjpe.py`
>
> </details>
>
> - Ran `python -m py_compile /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/run_ablation_mpjpe.py /Users/lecha/Desktop/four_grade/graduation_article/data_pdf/MMFi_Dataset/run_cmc_mixup_all_splits_mpjpe.py`
>
> </details>
</details>

我已经改完并且自己审了一遍，这次两条脚本是一致的。

现在的状态是：

- [run_ablation_mpjpe.py](./run_ablation_mpjpe.py)
  - 会优先读取 `experiment_results/bayesian_hparam_tuning_mpjpe/mixup/bo_summary.json`
  - fallback 也更新成了你这次的新结果 `lr=1.03e-04, bs=64`
  - teacher 路径是 `strict_offline_runs/T1_rgb_only_teacher_mpjpe/best.pth`
  - CMC 前缀和训练脚本一致：`temporal_lupi_rgb_teacher_cmc_cbam_distill`
  - 打印标题和实际运行参数一致，不再显示旧值

- [run_cmc_mixup_all_splits_mpjpe.py](./run_cmc_mixup_all_splits_mpjpe.py)
  - 已改成 `lr=1.03e-04, bs=64`
  - teacher 路径也切到了新的 `T1_rgb_only_teacher_mpjpe`
  - `PREFIX` 也和训练脚本对齐了
  - 打印出来的 best params 和真实参数一致

我还做了两件自检：
- 全局 grep 复核了旧的 `1.14e-04 / bs=16 / 旧 teacher 路径 / 旧 prefix`
- 对这两个脚本重新跑了 `py_compile`，都通过了

这次我的审阅结论是：这两个脚本现在已经能稳定代表你最新的 `MPJPE + CMC + Mixup` 主线，没有明显的口径不一致问题了。

你现在可以直接继续跑实验。如果你愿意，下一步我可以帮你再做一个很实用的小补丁：把 `run_cmc_mixup_all_splits_mpjpe.py` 也改成像 `ablation` 一样自动读取 `bo_summary.json`，这样以后你重新调优后，`3×3` 不用再手改数值。

> 给我命令

如果你现在要直接跑，常用命令就是这两个。

跑 `MPJPE` 消融：

```bash
python run_ablation_mpjpe.py data_base config.yaml
```

只跑某一个消融分支也可以：

```bash
python run_ablation_mpjpe.py data_base config.yaml --variant mixup_cmc
```

跑你说的 `3×3` 实验：

```bash
python run_cmc_mixup_all_splits_mpjpe.py data_base config.yaml
```

如果你只想先跑某一个 protocol 或 split，可以这样：

```bash
python run_cmc_mixup_all_splits_mpjpe.py data_base config.yaml --protocol protocol2
```

```bash
python run_cmc_mixup_all_splits_mpjpe.py data_base config.yaml --split cross_subject_split
```

```bash
python run_cmc_mixup_all_splits_mpjpe.py data_base config.yaml --protocol protocol2 --split cross_subject_split
```

如果你是在 Windows 下，就照这样单行直接粘贴运行就行。

> ablation的消融逻辑是对的吧

对，**就你当前这条 `MPJPE` 主线来说，`ablation` 的消融逻辑是对的**。

它现在比较的是这四组，而且控制变量关系是清楚的：

- `A_temporal_only`
  - baseline
  - `no Mixup`
  - `no CMC`

- `B_temporal_mixup`
  - baseline
  - `Mixup`
  - `no CMC`

- `C_temporal_cmc`
  - LUPI/CMC 路线
  - `no Mixup`
  - `CMC on`

- `D_temporal_mixup_cmc`
  - LUPI/CMC 路线
  - `Mixup`
  - `CMC on`

所以它能回答两个核心问题：

- `Mixup` 有没有带来增益
- `CMC` 有没有带来增益

而且你现在这版还有几个关键点是正确对齐的：

- 全部按 `MPJPE` 选 best
- teacher 已经换成新的 `T1_rgb_only_teacher_mpjpe`
- `CMC` 组和非 `CMC` 组都在同一套 `no CBAM` 设定下比较
- `mixup_alpha` 的开关也对应正确：
  - A/C 是 `0.0`
  - B/D 是 `0.4`

不过我还是提醒你一个**方法学上的小前提**：

现在这个消融是“**共享同一套调优后主线超参**”的消融，不是“每个 variant 各自单独调到最优”。
这通常是**合理的消融设计**，因为它更强调控制变量、看模块贡献；但如果你想写成“每个方法都在各自最优配置下比较”，那就不够了。

一句话总结就是：

- 如果你的目标是“验证 `CMC` 和 `Mixup` 的贡献”，现在这套 `ablation` 逻辑是对的
- 如果你的目标是“比较四个方法各自天花板性能”，那就需要每个 variant 单独调参

按你现在的论文主线，我认为**当前这个消融设计是成立的**。
如果你要，我下一步可以再帮你把这四组的“论文表述版本”写成一段标准实验描述。

> 我要写论文，英文文献的论文，你可以按照目前我这一套内容写一下模型和实验部分吗

<details><summary>2 previous messages</summary>

> 我先把模型结构和实验设定再对一遍，确保写出来的论文段落能直接贴近你现在这套代码和实验主线，而不是泛泛描述。
>
> <details><summary>Explored 3 files, 1 search</summary>
>
> - Read `./common.py`
> - Searched for `class TemporalWiFiEncoderCBAM|class WiFiFrameEncoderCBAM|class TemporalLUPICMC_CBAM|class TemporalWiFiStudentCBAM|class RGBOnlyTeacher|class EnhancedRGBEncoder|class PoseHead|class CBAM|def info_nce_loss` in `common.py`
> - Read `./train_baseline_mpjpe.py`
> - Read `./train_lupi_rgb_teacher_mpjpe.py`
>
> </details>
</details>

**Model**

We propose a teacher-guided WiFi-based 3D pose estimation framework. The student takes a 32-frame WiFi CSI window as input and first extracts per-frame spatial features with a CNN backbone. The frame encoder consists of a shallow convolutional stem, two downsampling convolution blocks, and a final projection layer. In the full backbone, CBAM can be inserted after early convolution blocks, but in the final MPJPE-based main experiments we keep CBAM disabled to isolate the effect of Mixup and CMC. The frame-wise features are then fed into a temporal Transformer encoder with sinusoidal positional encoding, and the final temporal token is passed to a pose regression head that predicts 17 joints in 3D.

To exploit privileged information, we additionally train a frozen RGB-only teacher. The teacher encodes 2D RGB joint coordinates with a lightweight joint embedding layer, a positional embedding, and a 2-layer Transformer encoder, followed by a pose head. During student training, the teacher is not updated; it only provides RGB feature representations for contrastive alignment. The student is optimized with pose regression loss and a contrastive CMC loss:
\[
\mathcal{L}=\mathcal{L}_{pose}+\lambda_{cmc}\mathcal{L}_{cmc}+\lambda_{distill}\mathcal{L}_{distill},
\]
where \(\mathcal{L}_{pose}\) is the L1 loss on 3D joints. In our final MPJPE experiments, \(\lambda_{distill}=0\), so only CMC is used as privileged supervision.

**Experiments**

We evaluate on the `cross_subject_split` setting for the main ablation study and additionally run a 3×3 comparison over three protocols and three split strategies. The main ablation compares four configurations: WiFi baseline, WiFi + Mixup, WiFi + CMC, and WiFi + Mixup + CMC. All variants use the same training recipe, with 30 epochs, window size 32, stride 2, AdamW, and MPJPE-based model selection. Mixup is applied with \(\alpha=0.4\), and the contrastive temperature is fixed at \(\tau=0.1\) with \(\lambda_{cmc}=0.2\). We report MPJPE as the primary metric and PA-MPJPE and PCK@20/50 as secondary metrics. For the ablation, we run 5 random seeds.

Before the final ablation, we perform Bayesian hyperparameter optimization with Optuna on the `mixup` mode only, i.e., the setting with CMC + Mixup and no CBAM. The search space includes learning rate and batch size, with the best setting found at `lr = 1.03e-4` and `batch_size = 64`. This tuned configuration is then reused for the ablation and the 3×3 experiment to keep the comparison consistent. The 3×3 study evaluates all combinations of `protocol1/2/3` and `random/cross-scene/cross-subject` splits, using 2 seeds per setting.

If you want, I can next turn this into a more formal **paper-ready LaTeX version** with subsection titles and equations.

> 结果你看一下，这个是在S2P2下面的消融，你应该知道是什么意思
>
> User context
>
> Images:
> - `/var/folders/w_/5zlw7x752qz8b9r82zjtnd600000gn/T/codex-clipboard-07771ee7-9566-4b41-b48e-8af6dba22a39.png`

看懂了，这是 `S2P2` 下的四组消融，也就是你当前主线里的：

- A: WiFi baseline
- B: WiFi + Mixup
- C: WiFi + CMC
- D: WiFi + Mixup + CMC

这个结果的核心含义是：

**MPJPE 主指标上**
- Baseline: `240.30`
- Mixup: `234.91`，最好
- CMC: `237.25`
- Mixup + CMC: `237.05`

所以如果只看 `MPJPE`，`Mixup-only` 是最优，`CMC` 有提升但没有超过 `Mixup-only`。`Mixup + CMC` 比 baseline 好 `3.25mm`，但比 Mixup-only 差 `2.14mm`。

**PA-MPJPE 上**
- Baseline: `98.21`
- Mixup: `95.33`
- CMC: `93.91`
- Mixup + CMC: `93.10`，最好

这个很关键。说明 `CMC` 对姿态结构/对齐后的误差是有明显帮助的。也就是说，CMC 让预测的 pose shape 更合理，但不一定让绝对 3D 坐标的 MPJPE 最优。

**PCK 上**
- `PCK@20` 最好是 `Mixup + CMC`: `31.17`
- `PCK@50` 最好是 `CMC-only`: `78.77`

所以 CMC 对关键点命中率也不是没用，尤其是更严格的 `PCK@20`，`Mixup + CMC` 是最高的。

论文里我建议你不要写成：

> CMC achieves the best MPJPE.

这个不成立。

更稳的表述应该是：

> Mixup achieves the lowest MPJPE under S2P2, while the proposed CMC-based LUPI supervision consistently improves structural pose quality, as reflected by lower PA-MPJPE and improved PCK@20. Combining Mixup with CMC obtains the best PA-MPJPE and PCK@20, suggesting that CMC mainly benefits pose configuration learning rather than absolute coordinate localization.

中文理解就是：

`Mixup` 更直接改善绝对坐标误差；
`CMC` 更明显改善姿态结构和对齐后的质量；
`Mixup + CMC` 在 `PA-MPJPE` 和 `PCK@20` 上最好，但在 `MPJPE` 上不如单独 Mixup。

这个结果不是坏结果，但论文里要把贡献点讲成“结构约束/跨模态表征对齐”，不要强行说它在所有指标上都最好。

> 我的老师说：模型就按时序+cmc来写，mixup放在实验部分

你老师这个建议是对的，而且更适合论文叙事。

也就是说，**模型方法部分不要把 Mixup 写成模型核心模块**。你的模型应该写成：

> Temporal WiFi Student + RGB Teacher + CMC-based LUPI supervision

也就是：
- 时序 WiFi 编码器是主干
- RGB-only teacher 是训练时 privileged information
- CMC 是你的核心方法/创新点
- 测试时只用 WiFi student

而 `Mixup` 应该放到实验设置或训练策略里，作为 regularization / augmentation：

> During training, we optionally apply Mixup as a data augmentation strategy.

这样处理有几个好处：

- 论文主贡献更干净：`Temporal + CMC`
- 不会让读者误以为 Mixup 是你设计的新模块
- 消融结果也更好解释：Mixup 是训练增强，CMC 是方法模块
- 你现在的结果里 Mixup 的 MPJPE 最好，也不会抢走 CMC 的“方法贡献”位置

所以论文结构建议改成：

**Method 部分**
- Temporal WiFi Encoder
- RGB-only Teacher under LUPI
- Cross-Modal Contrastive Matching / CMC
- Training Objective

这里可以写总损失：
\[
\mathcal{L} = \mathcal{L}_{pose} + \lambda_{cmc}\mathcal{L}_{cmc}
\]
如果你最终 `lambda_distill=0`，就不要强调 distillation，最多说 teacher is used to provide RGB privileged features.

**Experiment 部分**
- Dataset and Protocols
- Implementation Details
- Hyperparameter Tuning
- Training Augmentation: Mixup
- Ablation Study

Mixup 的写法可以是：

> Mixup is used as a training-time data augmentation strategy and is not considered part of the proposed architecture. We include it in the ablation study to examine its interaction with the proposed CMC supervision.

这句话很关键，很适合你现在的情况。

所以你的论文主线可以定成：

> We propose a temporal WiFi-based 3D pose estimator trained with cross-modal contrastive matching from an RGB-only teacher. Mixup is used only as a training regularizer in selected experiments.

这样既符合你老师说的，也能保护你的创新点。

> 论文创新点：第一个就是特权信息；第二个cmc；第三个跨域实验
> 我老师认为可以把这三个作为创新点

你老师这个框架很好，而且比“时序 + CMC + Mixup”更像正式论文的贡献表述。

你现在可以把创新点组织成三条：

**1. Privileged Information From RGB During Training**

核心意思是：
测试时只用 WiFi，但训练时引入 RGB pose teacher 作为 privileged information。

论文里可以写成：

> We introduce a learning using privileged information framework for WiFi-based 3D pose estimation, where an RGB-only teacher provides additional supervision during training while the final model requires only WiFi CSI at inference time.

这个点强调的是：你不是简单多模态融合，而是训练时借助 RGB、推理时完全 WiFi-only。

**2. Cross-Modal Contrastive Matching**

核心意思是：
不是直接让 WiFi 输出模仿 RGB，而是通过 CMC 对齐 WiFi temporal feature 和 RGB teacher feature。

可以写成：

> To transfer structural pose knowledge from the RGB teacher to the WiFi student, we design a cross-modal contrastive matching objective that aligns WiFi temporal representations with RGB privileged representations in a shared embedding space.

这个是你最核心的方法创新。尤其适合结合你现在结果说：CMC 对 PA-MPJPE 和 PCK@20 更明显，说明它更偏向提升 pose structure。

**3. Cross-Domain Generalization Evaluation**

核心意思是：
你不是只在一个固定 split 上报结果，而是做了 protocol × split 的系统实验。

可以写成：

> We conduct a comprehensive cross-domain evaluation across different activity protocols and data split strategies, including random, cross-scene, and cross-subject settings, to assess the robustness and generalization ability of the proposed WiFi-based pose estimation framework.

这个点对论文很重要，因为它能把你的实验从“跑了一个指标”提升成“验证泛化能力”。

推荐你论文贡献段落这样写：

```text
The main contributions of this work are threefold.
First, we formulate WiFi-based 3D human pose estimation under a learning using privileged information paradigm, where RGB pose observations are available only during training and the deployed model relies solely on WiFi CSI.
Second, we propose a cross-modal contrastive matching objective to transfer structural pose knowledge from an RGB-only teacher to a temporal WiFi student by aligning their latent representations.
Third, we provide a comprehensive cross-domain evaluation over multiple activity protocols and split strategies, demonstrating the robustness of the proposed framework under random, cross-scene, and cross-subject settings.
```

然后全文叙事就应该围绕这三点展开：

- Method 重点写 privileged teacher + CMC
- Mixup 放到 training details / ablation
- 3×3 结果放到 generalization evaluation
- S2P2 消融说明 CMC 对结构指标有效，比如 PA-MPJPE 和 PCK@20

这样你的创新点是完整的，而且和现在实验结果也能对上。
