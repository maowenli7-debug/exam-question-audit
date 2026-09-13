#!/usr/bin/env python
"""把 LoRA 适配器合并进基座权重，导出一个独立的完整模型。

为什么要合并
------------
vLLM 支持运行时挂 LoRA（``--enable-lora``），但每个 token 的计算图上都多一层
低秩分支，吞吐会掉。合并成一份权重后就是普通模型，推理路径最短。

什么时候**不要**合并
--------------------
需要热切换多个适配器、或者要保留随时换 adapter 的能力时，用
``deploy/vllm_serve.sh`` 的 ``--enable-lora`` 路径。合并是不可逆的。

用法::

    python scripts/merge_adapter.py \\
        --base models/Qwen2.5-0.5B-Instruct \\
        --adapter outputs/qwen2.5-0.5b-lora \\
        --out models/merged

注意：合并要在 **fp32/bf16** 下做。用 4-bit 量化模型合并会引入量化误差，
合并出来的权重和「4-bit 基座 + LoRA」的实际推理结果对不上。
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description="合并 LoRA 适配器到基座模型")
    ap.add_argument("--base", required=True, help="基座模型目录")
    ap.add_argument("--adapter", required=True, help="LoRA 适配器目录")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_file = Path(args.adapter) / "adapter_model.safetensors"
    if not adapter_file.exists():
        print(f"✗ 适配器不存在：{adapter_file}")
        return 1

    out_dir = Path(args.out)
    if out_dir.exists():
        # 合并结果可能已经在用，直接覆盖会让正在跑的服务读到半个文件。
        # 让调用方显式决定是否清理。
        print(f"✗ 输出目录已存在：{out_dir}")
        print(f"  如需重新合并，请先删除：rm -rf {out_dir}")
        return 1

    dtype = getattr(torch, args.dtype)
    print(f"加载基座模型 {args.base}（{args.dtype}）…")
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=dtype, trust_remote_code=True)

    print(f"挂载适配器 {args.adapter} …")
    model = PeftModel.from_pretrained(base, args.adapter)

    print("合并权重…")
    model = model.merge_and_unload()

    out_dir.mkdir(parents=True, exist_ok=True)
    # safe_serialization=True：新版 transformers 默认不加载 .bin，
    # 存成 .bin 会导致合并出来的模型在下游加载失败
    model.save_pretrained(out_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    tokenizer.save_pretrained(out_dir)

    # 记下来源，便于事后追溯这个合并模型是从哪个 adapter 训出来的
    (out_dir / "merge_info.json").write_text(
        json.dumps(
            {"base_model": args.base, "adapter": args.adapter, "dtype": args.dtype},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # adapter 目录里的 tokenizer 文件如果有更新版本，以适配器侧的为准
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = Path(args.adapter) / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    size_gb = sum(f.stat().st_size for f in out_dir.glob("*.safetensors")) / 1024**3
    print(f"\n✓ 合并完成：{out_dir}（{size_gb:.2f} GB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
