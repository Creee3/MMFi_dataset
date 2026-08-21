# DT-Pose 当前入口

新的等样本 DT-Pose 必须分两阶段：

```cmd
cd /d D:\study\four_grade\graduation_article\data_pdf\MMFi_Dataset\launchers
20_dtpose_pretrain_verify.cmd
21_dtpose_pretrain_run_or_resume.cmd
22_dtpose_pose_verify.cmd
23_dtpose_run_3x3.cmd
```

`models\dtpose\official_source` 和历史 checkpoint 继续用于官方源码追溯及旧结果复现。
当前严格等样本实验不能直接使用其中的历史预训练权重。

