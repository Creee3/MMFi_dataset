# models 目录说明

`models\` 只用于查看四个模型的代码来源和历史 checkpoint，不再作为日常启动目录。
所有当前命令请从根目录的 `launchers\` 执行。

本地 macOS 使用相对符号链接，学校 5090 使用 Windows 目录联接。链接形式不同，
但目标关系一致，均不会生成第二份活动源码。

## 当前正式训练入口

| 模型 | 当前入口 |
|---|---|
| PrivPose | 根目录 `run_s2p2_3x3_rank2.py`；使用 `launchers\02_privpose_run_or_resume_3x3.cmd` |
| MetaFi++ | `s2p2_equal_samples_mpjpe_code_20260820\`；使用 `launchers\11_metafi_run_3x3.cmd` |
| HPE-Li | 同一等样本 baseline 包；使用 `launchers\12_hpeli_run_3x3.cmd` |
| DT-Pose | 同一等样本 baseline 包；先用 `21` 预训练，再用 `23` 训练姿态 |

## 本目录中的旧分类入口

- `metafi_pp\code` 指向旧 MetaFi++ S2P2 云端包，用于历史复现。
- `dtpose\official_source` 是 DT-Pose 官方基础工程和历史共享源码。
- `hpeli\code` 指向上述共享工程，HPE-Li 网络位于其中的 `model\hpeli\`。
- `dtpose\checkpoints`、`hpeli\checkpoints` 指向历史 S2P2 checkpoint 归档。
- `baseline_equal_samples\code` 指向当前等样本三 baseline 代码包。
- `privpose\` 只有说明；PrivPose 的活动源码仍保留在项目根目录。

不要在这些联接目标中复制出第二份活动源码，也不要用历史 checkpoint 冒充新的
等样本实验权重。

## 唯一实体源码

- PrivPose：项目根目录的 `common.py`、`mmfi_lib/` 和正式训练脚本；
- 三 baseline 当前等样本代码：`s2p2_equal_samples_mpjpe_code_20260820/`；
- DT-Pose/HPE-Li 历史官方共享工程：`dtpose/official_source/`；
- MetaFi++ 历史 S2P2 包：根目录 `metafi_s2p2_cloud_bundle_20260817/`。

`code` 和 `checkpoints` 均为导航链接。修改源码时应进入上面的实体位置，不要把
链接目录另行复制后再改。
