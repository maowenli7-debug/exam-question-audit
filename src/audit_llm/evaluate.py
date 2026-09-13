"""评测：格式遵循率 与 字段准确率。

两类指标，口径必须分清
----------------------
1. **格式遵循率**（主指标）
   输出能否被解析成合法的 ``AuditResult``。只看结构，不看内容对错。
   这是本项目的主指标——下游是数据库，格式不合法就入不了库，
   此时审核判断对不对没有意义。

2. **字段准确率**
   结构合法**之后**，内容判得对不对：结论、风险等级、问题类型。

   格式失败的样本在字段准确率里一律记为「错」——因为下游系统拿不到可用的
   结构化结果。这一点很重要：如果只在格式合法的子集上算字段准确率，会出现
   「格式越差、字段准确率越高」的假象（模型只对简单样本给出了合法 JSON）。
   报告里会同时给出两个口径，但主结论以「占全量」为准。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .schema import AuditResult, Question

if TYPE_CHECKING:  # pragma: no cover - 只在类型检查时求值
    # AuditModel 只用于下面 evaluate_model 的类型标注，运行时不需要。
    #
    # 这里**必须**用 TYPE_CHECKING 而不是顶层 import：infer 模块顶层有
    # ``import torch``，而 CI 只装 pydantic + pytest（torch 有好几个 GB，
    # 而 report 相关的逻辑一行都用不到它）。一旦顶层导入，test_eval_report.py
    # 会在 collection 阶段就 ModuleNotFoundError，整个测试文件跑不了。
    #
    # 这个坑本地测不出来——本地 conda 环境装着 torch，import 一定会成功。
    # 只有 CI 能暴露它，所以 CI 里专门加了一步守住（见 .github/workflows/ci.yml）。
    from .infer import AuditModel

__all__ = ["eval_result_to_dict"]


@dataclass
class EvalResult:
    """一个模型在一份测试集上的完整评测结果。"""

    name: str
    n: int = 0
    format_ok: int = 0
    """能解析成合法 AuditResult 的样本数——格式遵循率的分子。"""

    bare_json_ok: int = 0
    """未经代码块/前后缀提取就合法的样本数。用来识别「靠容错提取才达标」的情况。"""

    failure_modes: Counter = field(default_factory=Counter)
    """失败归类：empty / no_json / bad_json / schema。"""

    conclusion_correct: int = 0
    risk_correct: int = 0

    issue_tp: int = 0
    issue_fp: int = 0
    issue_fn: int = 0

    predictions: list[dict] = field(default_factory=list)
    """逐条原始输出，写入 reports/predictions/ 供人工复盘。"""

    # -- 派生指标 --------------------------------------------------------

    @property
    def format_rate(self) -> float:
        return self.format_ok / self.n if self.n else 0.0

    @property
    def bare_json_rate(self) -> float:
        return self.bare_json_ok / self.n if self.n else 0.0

    @property
    def conclusion_acc(self) -> float:
        """结论准确率，分母是**全量**样本（格式失败的算错）。"""
        return self.conclusion_correct / self.n if self.n else 0.0

    @property
    def risk_acc(self) -> float:
        return self.risk_correct / self.n if self.n else 0.0

    @property
    def issue_precision(self) -> float:
        denom = self.issue_tp + self.issue_fp
        return self.issue_tp / denom if denom else 0.0

    @property
    def issue_recall(self) -> float:
        denom = self.issue_tp + self.issue_fn
        return self.issue_tp / denom if denom else 0.0

    @property
    def issue_f1(self) -> float:
        p, r = self.issue_precision, self.issue_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate_model(
    model: AuditModel,
    records: list[dict],
    name: str,
    *,
    verbose: bool = True,
    log_every: int = 25,
) -> EvalResult:
    """在测试集上跑一遍，算出全部指标。

    解码用贪心（``greedy=True``），保证同一模型每次跑出的数字完全一致——
    评测必须可复现，否则「提升到 91%」这种结论没法验证。
    """
    result = EvalResult(name=name, n=len(records))

    for idx, rec in enumerate(records, 1):
        question = Question.model_validate(rec["question"])
        gold = AuditResult.model_validate(rec["audit"])

        pred = model.audit(question, greedy=True)

        result.predictions.append(
            {
                "id": rec["id"],
                "subject": question.subject,
                "gold": rec["audit"],
                "raw_output": pred.raw,
                "format_ok": pred.format_ok,
                "failure_reason": pred.failure_reason,
                "failure_detail": pred.failure_detail,
                "extracted": pred.extracted,
                "parsed": pred.parsed.model_dump(mode="json") if pred.parsed else None,
            }
        )

        if pred.format_ok:
            result.format_ok += 1
            if not pred.extracted:
                result.bare_json_ok += 1
        else:
            result.failure_modes[pred.failure_reason or "unknown"] += 1

        # --- 字段准确率：格式失败即内容全错，见模块 docstring ---
        if pred.parsed is not None:
            if pred.parsed.conclusion == gold.conclusion:
                result.conclusion_correct += 1
            if pred.parsed.risk_level == gold.risk_level:
                result.risk_correct += 1

            pred_types = {i.type.value for i in pred.parsed.issues}
            gold_types = {i.type.value for i in gold.issues}
        else:
            pred_types, gold_types = set(), {i.type.value for i in gold.issues}

        result.issue_tp += len(pred_types & gold_types)
        result.issue_fp += len(pred_types - gold_types)
        result.issue_fn += len(gold_types - pred_types)

        if verbose and idx % log_every == 0:
            print(
                f"    [{name}] {idx}/{len(records)}  "
                f"格式遵循 {result.format_rate:.1%}  结论准确 {result.conclusion_acc:.1%}"
            )

    return result


def eval_result_to_dict(r: EvalResult) -> dict:
    """转成可 JSON 序列化的字典（不含逐条预测，那个单独存）。"""
    return {
        "name": r.name,
        "n": r.n,
        "format_rate": round(r.format_rate, 4),
        "format_ok": r.format_ok,
        "bare_json_rate": round(r.bare_json_rate, 4),
        "bare_json_ok": r.bare_json_ok,
        "failure_modes": dict(r.failure_modes),
        "conclusion_acc": round(r.conclusion_acc, 4),
        "risk_acc": round(r.risk_acc, 4),
        "issue_precision": round(r.issue_precision, 4),
        "issue_recall": round(r.issue_recall, 4),
        "issue_f1": round(r.issue_f1, 4),
        "issue_tp": r.issue_tp,
        "issue_fp": r.issue_fp,
        "issue_fn": r.issue_fn,
    }
