"""QLoRA 微调。

设备自适应
----------
同一份代码要能在两种环境跑：

  - **CUDA（目标环境，24GB 单卡）**：走完整的 4-bit QLoRA —— bitsandbytes nf4 量化
    基座模型 + bf16 计算 + LoRA 适配器。
  - **Apple MPS / CPU（本机）**：bitsandbytes 的 4-bit 没有 MPS 实现，强行加载会
    直接报错。所以脚本检测到非 CUDA 时会**自动关掉量化**，改用纯 LoRA 训练，
    并打印明确警告。这不是降级兜底，而是唯一能在这台机器上真实跑通的方式。

本机跑的是 0.5B 而非 7B，但 LoRA 结构（r=16 / alpha=32 / 全线性层）与目标配置
完全一致，换到 CUDA 机器上只需改 YAML 里的 model.name_or_path 和 load_in_4bit。

prompt 渲染
-----------
训练序列由 ``prompts.py`` 显式渲染成字符串后交给 trl，**不用** trl 的对话式
模板管道。原因：``prompts.py`` 同时被训练和推理（infer.py）使用，显式渲染能保证
两边看到的 token 序列逐字节一致；让 trl 各自套一遍模板则多一层漂移风险。

completion 的边界靠「整段渲染」与「prompt 渲染」取差集得到，见 ``_render_pair``，
不硬编码 ``<|im_end|>`` 之类的特殊 token。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from .prompts import build_messages
from .schema import AuditResult, Question

__all__ = ["main", "load_config", "build_dataset"]


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# --------------------------------------------------------------------------
# 数据集
# --------------------------------------------------------------------------

def _render_pair(tokenizer, question: Question, audit_json: str) -> tuple[str, str]:
    """渲染出 (prompt, completion) 字符串对。

    先渲染「system + user + assistant」的完整对话，再渲染「system + user」并带
    生成提示，两者取差集即为 completion。这样不依赖任何硬编码的特殊 token，
    换模型（换 chat template）也不用改代码。

    取差集前会断言完整版确实以 prompt 开头——如果某个模型的模板在
    ``add_generation_prompt`` 下产生的不是完整版的前缀，这里会立刻炸出来，
    而不是悄悄训出一个错位的模型。
    """
    messages = build_messages(question)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": audit_json}],
        tokenize=False,
    )
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not full.startswith(prompt):
        raise RuntimeError(
            "prompt 不是完整对话的前缀，chat template 行为与预期不符。\n"
            f"prompt 结尾: {prompt[-60:]!r}\n"
            f"full  对应处: {full[len(prompt)-60:len(prompt)+20]!r}"
        )
    return prompt, full[len(prompt) :]


def build_dataset(path: Path, tokenizer, max_length: int) -> Dataset:
    """读 jsonl，渲染成 trl 的 prompt-completion 数据集。

    只保留 ``prompt`` / ``completion`` 两列。记录里的 ``injected`` 字段是我们
    做数据质检用的元数据，属于答案的一部分，**绝不能**进入训练数据。
    """
    rows: list[dict[str, str]] = []
    too_long = 0

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            question = Question.model_validate(rec["question"])
            # 用 pydantic 重新序列化，保证键顺序和分隔符是规范形式，
            # 而不是依赖 jsonl 里那份可能被手工编辑过的写法
            audit_json = AuditResult.model_validate(rec["audit"]).model_dump_json()

            prompt, completion = _render_pair(tokenizer, question, audit_json)

            n_tokens = len(tokenizer(prompt + completion)["input_ids"])
            if n_tokens > max_length:
                too_long += 1
                continue

            rows.append({"prompt": prompt, "completion": completion})

    if not rows:
        raise RuntimeError(f"{path} 里没有一条样本能在 max_length={max_length} 内装下")
    if too_long:
        print(f"  [warn] {too_long} 条样本超过 max_length={max_length}，已跳过")

    return Dataset.from_list(rows)


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: dict, device: str):
    model_cfg = cfg["model"]
    name_or_path = model_cfg["name_or_path"]
    dtype = getattr(torch, model_cfg.get("dtype", "bfloat16"))

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, trust_remote_code=True)

    wants_4bit = bool(model_cfg.get("load_in_4bit", False))
    use_4bit = wants_4bit and device == "cuda"

    if wants_4bit and not use_4bit:
        print(
            f"  [warn] 配置要求 4-bit 量化，但当前设备是 {device}。\n"
            "         bitsandbytes 的 4-bit 仅支持 CUDA，MPS/CPU 上会直接报错，\n"
            "         因此本次训练**自动关闭量化**，改用纯 LoRA。\n"
            "         这不影响 LoRA 结构本身，但显存占用会显著高于 QLoRA。"
        )

    kwargs: dict = {"trust_remote_code": True, "dtype": dtype}

    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
            bnb_4bit_use_double_quant=bool(model_cfg.get("bnb_4bit_use_double_quant", True)),
        )
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(name_or_path, **kwargs)

    if not use_4bit:
        # 量化模型由 accelerate 负责摆放，非量化模型要手动搬
        model = model.to(device)

    model.config.use_cache = False  # 与 gradient checkpointing 冲突

    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    elif bool(cfg["training"].get("gradient_checkpointing", False)):
        # PEFT + gradient checkpointing 的经典坑：重算时输入张量的 requires_grad
        # 是 False，梯度到不了 LoRA 层，表现为 loss 平着不降、却不报任何错。
        # prepare_model_for_kbit_training 内部会做这件事，非量化路径要手动补上。
        model.enable_input_require_grads()

    return model, tokenizer, use_4bit


def build_lora_config(cfg: dict) -> LoraConfig:
    lora_cfg = cfg["lora"]
    return LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )


def count_trainable(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA 微调命题审核模型")
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="只用前 N 条训练样本，用于快速冒烟验证（不改配置文件）",
    )
    ap.add_argument("--output-dir", type=Path, default=None, help="覆盖配置里的输出目录")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device()
    tcfg = cfg["training"]
    output_dir = args.output_dir or Path(tcfg["output_dir"])

    print(f"配置文件 : {args.config}")
    print(f"设备     : {device}")
    print(f"输出目录 : {output_dir}")

    model, tokenizer, use_4bit = load_model_and_tokenizer(cfg, device)
    print(f"量化     : {'4-bit nf4 (QLoRA)' if use_4bit else '未启用'}")

    model = get_peft_model(model, build_lora_config(cfg))
    trainable, total = count_trainable(model)
    print(f"可训练参数: {trainable:,} / {total:,} ({trainable / total:.2%})\n")

    max_length = int(tcfg.get("max_length", 1024))
    print("构造数据集…")
    train_ds = build_dataset(Path(cfg["data"]["train_file"]), tokenizer, max_length)
    eval_ds = build_dataset(Path(cfg["data"]["dev_file"]), tokenizer, max_length)
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    print(f"  训练集 {len(train_ds)} 条，验证集 {len(eval_ds)} 条\n")

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        # trl 0.24 的字段名是 max_length，不是老版本教程里的 max_seq_length
        max_length=max_length,
        completion_only_loss=True,  # 只在 assistant 的 JSON 上算 loss，不学复述题目
        num_train_epochs=float(tcfg["num_train_epochs"]),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        # 默认 8。不显式设的话，验证阶段会以 8 倍 batch 去算 logits，
        # 在 MPS 上会以同样的方式 OOM——而且是在训练跑了很久之后才炸。
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        learning_rate=float(tcfg["learning_rate"]),
        lr_scheduler_type=tcfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(tcfg.get("warmup_ratio", 0.03)),
        logging_steps=int(tcfg.get("logging_steps", 10)),
        save_steps=int(tcfg.get("save_steps", 200)),
        eval_strategy="steps",
        eval_steps=int(tcfg.get("eval_steps", 200)),
        save_total_limit=int(tcfg.get("save_total_limit", 2)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        # bf16 只在 CUDA 上开：Trainer 的 fp16/bf16 混精走的是 CUDA autocast +
        # GradScaler，MPS 上没有等价实现。MPS 侧靠 fp32 权重本身保证数值稳定。
        bf16=bool(tcfg.get("bf16", False)) and device == "cuda",
        seed=int(tcfg.get("seed", 42)),
        report_to="none",  # 不启用 wandb 等外部上报，数据不出内网
    )
    # 注意：这里**不能**设 dataset_kwargs={"skip_prepare_dataset": True}。
    # 那个开关是给「自带 collator、自己 tokenize」的场景用的（典型是 VLM）。
    # 我们给的是标准 prompt/completion 字符串列，要让 SFTTrainer 自己走
    # _prepare_dataset 做 tokenize —— 它会顺带按 completion_only_loss 生成
    # completion_mask。跳过这一步，Trainer 就会因为数据集里没有 input_ids
    # 而报 "No columns match the model's forward method signature"。

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,  # trl 0.24 改叫 processing_class，不再是 tokenizer
    )

    print("开始训练…")
    trainer.train()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"\nLoRA 适配器已保存到 {output_dir}")

    (output_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "config_file": str(args.config),
                "device": device,
                "used_4bit": use_4bit,
                "trainable_params": trainable,
                "total_params": total,
                "train_samples": len(train_ds),
                "eval_samples": len(eval_ds),
                "max_length": max_length,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
