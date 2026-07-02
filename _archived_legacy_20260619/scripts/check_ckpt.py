# check_ckpt.py
# 读取 checkpoint 及同目录下的训练记录
import torch
import sys
import os
import json

path = sys.argv[1] if len(sys.argv) > 1 else "strict_offline_runs/T1_rgb_only_teacher/best.pth"
run_dir = os.path.dirname(path)

print(f"Run dir: {run_dir}")
print(f"Files:   {os.listdir(run_dir) if os.path.isdir(run_dir) else 'NOT FOUND'}")
print()

# ── 1. 检查同目录其他 .pth 文件是否有 args ──
for fname in sorted(os.listdir(run_dir)) if os.path.isdir(run_dir) else []:
    if fname.endswith(".pth") and fname != os.path.basename(path):
        fpath = os.path.join(run_dir, fname)
        try:
            ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
            if "args" in ckpt:
                print(f"Found args in: {fname}")
                print("=" * 60)
                for k, v in sorted(ckpt["args"].items()):
                    print(f"  {k:<25s} = {v}")
                print()
                break
        except:
            pass
    if fname == "history.json":
        with open(os.path.join(run_dir, fname)) as f:
            h = json.load(f)
        print(f"history.json: {len(h)} epochs recorded")
        if h:
            print(f"  Keys in entry 0: {list(h[0].keys())}")
        print()

# ── 2. 检查 log.txt 开头（通常有参数打印） ──
log_path = os.path.join(run_dir, "log.txt")
if os.path.isfile(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        head = "".join(f.readlines()[:25])
    print("log.txt (first 25 lines):")
    print("=" * 60)
    print(head)
    print("=" * 60)
    print()

# ── 3. 如果什么都没有，从模型结构推断 ──
print("─" * 60)
ckpt = torch.load(path, map_location="cpu", weights_only=False)
state = ckpt.get("model", ckpt)

total_params = sum(v.numel() for v in state.values() if isinstance(v, torch.Tensor))
print(f"Model: RGBOnlyTeacher | Params: {total_params:,} | d_model: 512 | Joints: 17")
print("Training args (lr/bs/mixup): NOT saved in this checkpoint.")
print("You'll need the original training script or log to know these.")
