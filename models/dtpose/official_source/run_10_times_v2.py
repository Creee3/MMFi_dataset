import os
import subprocess
import re
import numpy as np
import yaml


BASE_CONFIG_PATH = '/root/autodl-tmp/DT-Pose-main/config/mmfi/pose_config.yaml'
TEMP_CONFIG_PATH = '/root/autodl-tmp/DT-Pose-main/config/mmfi/temp_config_run.yaml'

SEEDS = [4]

# 注意：你之前写的前5次其实是2次的历史数据，为了配合 Seed=2,3,4 凑齐5次
all_mpjpe = [249.5, 236.1, 242.9, 252.1]  
all_pa_mpjpe = [101.6, 101.5, 102.4, 102.4]  
all_pck50 = [76.1, 78.6, 77.0, 74.9]  
all_pck20 = [26.1, 29.2, 28.1, 26.8]  


def main():
    print(f"{'=' * 60}")
    print(f"🚀 继续运行剩余实验，并汇总 5 次总分")
    print(f"当前已载入前 {len(all_mpjpe)} 次的历史成绩。")
    print(f"{'=' * 60}")

    for seed in SEEDS:
        print(f"\n▶ 正在运行实验 (Seed = {seed})...")

        with open(BASE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)

        cfg['seed'] = seed
        cfg['init_rand_seed'] = seed

        with open(TEMP_CONFIG_PATH, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        cmd = ["python", "-W", "ignore", "train_pose.py", "--config", TEMP_CONFIG_PATH]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        best_mpjpe, best_pa, best_pck50, best_pck20 = None, None, None, None

        # 实时读取子进程的输出
        for line in process.stdout:
            # 【关键修复】实时打印原本的输出内容，不再盲等！
            print(line, end="") 
            
            if "Best mpjpe:" in line:
                best_mpjpe = float(re.search(r"Best mpjpe:\s*([\d.]+)", line).group(1)) * 1000
            elif "Best pa-mpjpe:" in line:
                best_pa = float(re.search(r"Best pa-mpjpe:\s*([\d.]+)", line).group(1)) * 1000
            elif "Best pck50:" in line:
                best_pck50 = float(re.search(r"Best pck50:\s*([\d.]+)", line).group(1))
            elif "Best pck20:" in line:
                best_pck20 = float(re.search(r"Best pck20:\s*([\d.]+)", line).group(1))

        process.wait()

        if best_mpjpe and best_pa and best_pck50 and best_pck20:
            all_mpjpe.append(best_mpjpe)
            all_pa_mpjpe.append(best_pa)
            all_pck50.append(best_pck50)
            all_pck20.append(best_pck20)
            print(f"\n✅ Seed {seed} 提取成功! MPJPE: {best_mpjpe:.1f}mm | PA: {best_pa:.1f}mm | PCK@50: {best_pck50:.1f}% | PCK@20: {best_pck20:.1f}%")
        else:
            print(f"\n❌ Seed {seed} 运行异常或未能正确提取到 Best 指标。")

    # 汇总计算
    if len(all_mpjpe) > 0:
        print(f"\n{'=' * 60}")
        print(f"🎉 全部实验完成")
        print(f"{'=' * 60}")
        print(f"基于 {len(all_mpjpe)} 次运行的统计结果：")
        print(f"MPJPE (mm):    {np.mean(all_mpjpe):.1f} ± {np.std(all_mpjpe):.1f}")
        print(f"PA-MPJPE (mm): {np.mean(all_pa_mpjpe):.1f} ± {np.std(all_pa_mpjpe):.1f}")
        print(f"PCK@50 (%):    {np.mean(all_pck50):.1f} ± {np.std(all_pck50):.1f}")
        print(f"PCK@20 (%):    {np.mean(all_pck20):.1f} ± {np.std(all_pck20):.1f}")
        print(f"{'=' * 60}")

    if os.path.exists(TEMP_CONFIG_PATH):
        os.remove(TEMP_CONFIG_PATH)


if __name__ == "__main__":
    main()