"""本地推理后端（transformers）。

这是能在 Mac 上跑的推理实现，用于评测和本机演示。
正式部署走 vLLM（见 deploy/），但两者共用 ``prompts.py`` 的模板与
``schema.py`` 的解析器，所以结果口径一致。

与训练的一致性
--------------
prompt 渲染走的是 ``prompts.build_messages`` + ``apply_chat_template``，
和 ``train_qlora._render_pair`` 里的 prompt 部分**完全同一段代码路径**。
任何一侧改了模板而另一侧没改，模型表现都会莫名其妙地掉——这是本项目里
最容易被忽视、也最容易踩的坑。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompts import build_messages
from .schema import AuditResult, Question, parse_audit_output

__all__ = ["AuditModel", "GenerationResult"]


@dataclass
class GenerationResult:
    """一次推理的原始输出与解析结果。"""

    raw: str
    parsed: AuditResult | None
    format_ok: bool
    failure_reason: str = ""
    failure_detail: str = ""
    extracted: bool = False
    """True 表示原文带代码块或前后缀，是靠容错提取才拿到 JSON 的。"""


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AuditModel:
    """加载基座模型（可选叠加 LoRA 适配器）并提供审核推理。"""

    def __init__(
        self,
        base_model: str | Path,
        adapter: str | Path | None = None,
        *,
        device: str | None = None,
        dtype: str = "float32",
        max_new_tokens: int = 256,
    ) -> None:
        self.device = device or resolve_device()
        self.max_new_tokens = max_new_tokens
        self.base_model = str(base_model)
        self.adapter = str(adapter) if adapter else None

        # 先校验适配器路径，再加载模型。
        # 顺序很重要：peft 在目录不存在时抛的错很晦涩（往往只是一句找不到
        # config.json），而「忘了先跑 make train」是这里最常见的失误。
        # 放到加载之后检查的话，还得先白等几秒到几十秒把基座模型读进来。
        if self.adapter:
            adapter_file = Path(self.adapter) / "adapter_model.safetensors"
            if not adapter_file.exists():
                raise FileNotFoundError(
                    f"找不到 LoRA 适配器：{adapter_file}\n"
                    f"  请先运行 make train 产出适配器；\n"
                    f"  或改用基座模型（不传 adapter 参数），但注意那**不是**微调后的模型。"
                )

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            dtype=getattr(torch, dtype),
            trust_remote_code=True,
        )

        if self.adapter:
            # 延迟导入：不装 peft 的环境只用基座模型时不该因此报错
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, self.adapter)

        self.model = self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------

    def _render_prompt(self, question: Question) -> str:
        return self.tokenizer.apply_chat_template(
            build_messages(question),
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.no_grad()
    def generate_raw(self, question: Question, *, greedy: bool = True) -> str:
        """生成原始文本。评测用贪心解码，保证结果可复现。"""
        prompt = self._render_prompt(question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else 0.7,
            top_p=None if greedy else 0.9,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        # 只取新生成的部分，不要把 prompt 一起解码回来
        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def audit(self, question: Question, *, greedy: bool = True) -> GenerationResult:
        """审核一道题，返回原始输出与解析结果。"""
        raw = self.generate_raw(question, greedy=greedy)
        outcome = parse_audit_output(raw)
        return GenerationResult(
            raw=raw,
            parsed=outcome.result,
            format_ok=outcome.ok,
            failure_reason=outcome.reason,
            failure_detail=outcome.error or "",
            extracted=outcome.extracted,
        )
