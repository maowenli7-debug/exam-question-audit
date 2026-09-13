#!/usr/bin/env python
"""环境自检：依赖版本、计算设备、模型与数据是否就位。

只做只读检查，不安装任何东西。用来在训练前快速回答「这台机器现在能不能跑」。
"""

from __future__ import annotations

import importlib.metadata as md
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (发行包名, 是否必需, 说明)
# 注意这里用的是 pip 的**发行包名**而不是 import 名：yaml 的发行包叫 pyyaml，
# 用 import 名去查 importlib.metadata 会误报「未安装」。
REQUIRED = [
    ("torch", True, "训练与推理"),
    ("transformers", True, "模型加载"),
    ("peft", True, "LoRA/QLoRA"),
    ("trl", True, "SFTTrainer"),
    ("datasets", True, "数据集"),
    ("accelerate", True, "设备与混合精度"),
    ("pydantic", True, "结构化校验"),
    ("pyyaml", True, "配置读取"),
    ("fastapi", True, "推理服务"),
    ("uvicorn", True, "ASGI 服务器"),
    ("modelscope", False, "国内下载模型（可选）"),
    ("bitsandbytes", False, "4-bit 量化，仅 CUDA 需要"),
    ("vllm", False, "高吞吐部署，仅 CUDA"),
]

GREEN, YELLOW, RED, DIM, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m"


def _mark(ok: bool, warn: bool = False) -> str:
    if ok:
        return f"{GREEN}✓{RESET}"
    return f"{YELLOW}—{RESET}" if warn else f"{RED}✗{RESET}"


def main() -> int:
    print("=" * 66)
    print("命题审核 LLM —— 环境自检")
    print("=" * 66)

    # --- Python ---
    print(f"\nPython : {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < (3, 10):
        print(f"  {RED}✗ 需要 Python 3.10 及以上{RESET}")
        return 1

    # --- 依赖 ---
    print("\n依赖:")
    missing_required: list[str] = []
    for pkg, required, why in REQUIRED:
        try:
            version = md.version(pkg)
            print(f"  {_mark(True)} {pkg:16s} {version:12s} {DIM}{why}{RESET}")
        except md.PackageNotFoundError:
            if required:
                missing_required.append(pkg)
            print(f"  {_mark(False, warn=not required)} {pkg:16s} {'未安装':12s} {DIM}{why}{RESET}")

    # --- 计算设备 ---
    print("\n计算设备:")
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  {GREEN}✓{RESET} CUDA 可用：{name}（{vram:.0f} GB）")
            print(f"      → 可运行完整的 4-bit QLoRA 与 vLLM")
        elif torch.backends.mps.is_available():
            device = "mps"
            print(f"  {GREEN}✓{RESET} Apple MPS 可用（无 CUDA）")
            print(f"      {YELLOW}→ bitsandbytes 的 4-bit 不支持 MPS，训练会自动关闭量化{RESET}")
            print(f"      {YELLOW}→ vLLM 不支持 MPS，本机请用 transformers 后端{RESET}")
        else:
            print(f"  {YELLOW}—{RESET} 只有 CPU，训练会非常慢")
    except ImportError:
        print(f"  {RED}✗{RESET} torch 未安装，无法检测设备")

    # --- 模型 ---
    print("\n模型:")
    for name in ("Qwen2.5-0.5B-Instruct", "Qwen2.5-7B-Instruct"):
        path = REPO_ROOT / "models" / name
        weights = list(path.glob("*.safetensors")) if path.exists() else []
        if weights:
            size = sum(w.stat().st_size for w in weights) / 1024**3
            print(f"  {GREEN}✓{RESET} {name:24s} {size:.2f} GB")
        elif path.exists():
            print(f"  {YELLOW}—{RESET} {name:24s} 目录存在但权重不完整（下载中断？）")
        else:
            print(f"  {RED}✗{RESET} {name:24s} 未下载")

    # --- 数据 ---
    print("\n数据:")
    data_dir = REPO_ROOT / "data"
    for split in ("train", "dev", "test"):
        path = data_dir / f"{split}.jsonl"
        if path.exists():
            n = sum(1 for _ in path.open(encoding="utf-8"))
            print(f"  {GREEN}✓{RESET} {split:6s} {n:6d} 条")
        else:
            print(f"  {RED}✗{RESET} {split:6s} 缺失，请先运行 make data")

    # --- 已训练产物 ---
    print("\n训练产物:")
    found = False
    for out in sorted((REPO_ROOT / "outputs").glob("*/adapter_model.safetensors")):
        size = out.stat().st_size / 1024**2
        print(f"  {GREEN}✓{RESET} {out.parent.relative_to(REPO_ROOT)}  ({size:.1f} MB)")
        found = True
    if not found:
        print(f"  {DIM}— 还没有训练好的 adapter，请先运行 make train{RESET}")

    # --- 结论 ---
    print("\n" + "=" * 66)
    if missing_required:
        print(f"{RED}缺少必需依赖：{', '.join(missing_required)}{RESET}")
        print("安装： make install")
        return 1

    print(f"{GREEN}必需依赖齐备。{RESET}当前设备：{device}")
    if device != "cuda":
        print(
            f"{YELLOW}注意：本机没有 NVIDIA 显卡，跑不了「24GB 单卡 + 4-bit QLoRA + vLLM」"
            f"的正式配置。\n      本机可跑通全链路（0.5B 模型 + 纯 LoRA + transformers 后端），"
            f"正式训练见 docs/runbook_cuda.md。{RESET}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
