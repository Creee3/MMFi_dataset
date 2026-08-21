import os
import subprocess
import re
import numpy as np
import yaml

BASE_CONFIG_PATH = '/root/autodl-tmp/DT-Pose-main/config/mmfi/pose_config-Copy2.yaml'
TEMP_CONFIG_PATH = '/root/autodl-tmp/DT-Pose-main/config/mmfi/temp_config_run-Copy2.yaml'
# 设置 5 个随机种子
SEEDS = [0, 1, 2, 3, 4]

def main():
    print(f"{'=' * 60}")
    print(f"🚀 开始 5 次独立实验评估 (Mean ± Std)")
    print(f"{'=' * 60}")

    all_mpjpe = []
    all_pa_mpjpe = []
    all_pck50 = []
    all_pck20 = []
    failed_runs = []

    my_env = os.environ.copy()
    my_env["OMP_NUM_THREADS"] = "4"    
    my_env["PYTHONUNBUFFERED"] = "1"    

    for seed in SEEDS:
        print(f"\n▶ 正在运行第 {seed + 1}/5 次实验 (Seed = {seed})...")

        # 1. 读取基础配置，修改随机种子
        with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        cfg['seed'] = seed
        cfg['init_rand_seed'] = seed

        # 2. 将修改后的配置保存为临时文件
        with open(TEMP_CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # 3. 组装运行命令
        cmd = ["python", "-W", "ignore", "train_pose-Copy2.py", "--config_file", TEMP_CONFIG_PATH]

        # 4. 启动子进程运行，传入修复后的环境变量
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=my_env)

        best_mpjpe, best_pa, best_pck50, best_pck20 = None, None, None, None

        # 5. 更安全的逐行读取方式，避免 readline 阻塞
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
            
            # 使用 end='' 避免打印出空行
            print(line, end='') 

            if line.startswith("Best mpjpe:"):
                best_mpjpe = float(line.split(":")[1].strip()) * 1000
            elif line.startswith("Best pa-mpjpe:"):
                best_pa = float(line.split(":")[1].strip()) * 1000
            elif line.startswith("Best pck50:"):
                best_pck50 = float(line.split(":")[1].strip())
            elif line.startswith("Best pck20:"):
                best_pck20 = float(line.split(":")[1].strip())

        process.wait()

        # 6. 记录本次成绩
        if best_mpjpe and best_pa and best_pck50 and best_pck20:
            all_mpjpe.append(best_mpjpe)
            all_pa_mpjpe.append(best_pa)
            all_pck50.append(best_pck50)
            all_pck20.append(best_pck20)
            print(f"✅ 第 {seed + 1} 次完成! MPJPE: {best_mpjpe:.1f}mm | PA: {best_pa:.1f}mm | PCK@50: {best_pck50:.1f}% | PCK@20: {best_pck20:.1f}%")
        else:
            print(f"❌ 第 {seed + 1} 次运行异常，未抓取到完整数据。")
            failed_runs.append(seed + 1)

    # 7. 计算最终的平均值和标准差
    if len(all_mpjpe) > 0:
        print(f"\n{'=' * 60}")
        print(f"hpeli/protocol2-s2")
        print(f"🎉 5 次实验全部完成！")
        print(f"{'=' * 60}")
        print(f"MPJPE (mm):    {np.mean(all_mpjpe):.1f} ± {np.std(all_mpjpe):.1f}")
        print(f"PA-MPJPE (mm): {np.mean(all_pa_mpjpe):.1f} ± {np.std(all_pa_mpjpe):.1f}")
        print(f"PCK@50 (%):    {np.mean(all_pck50):.1f} ± {np.std(all_pck50):.1f}")
        print(f"PCK@20 (%):    {np.mean(all_pck20):.1f} ± {np.std(all_pck20):.1f}")
        print(f"{'=' * 60}")

        if failed_runs:
            print(f"{'-' * 60}")
            print(f"⚠️ 警告：共有 {len(failed_runs)} 次实验运行异常，未计入最终统计！")
            print(f"异常的实验轮次: 第 {', '.join(map(str, failed_runs))} 次")
            print(f"{'=' * 60}")

    # 删掉临时配置文件
    if os.path.exists(TEMP_CONFIG_PATH):
        os.remove(TEMP_CONFIG_PATH)

if __name__ == "__main__":
    main()