"""审核结论的结构化 Schema 与解析器。

这个模块有两个职责，对应项目里的两件事：

1. ``AuditResult`` —— 审核结论的**唯一权威格式**。合成数据按它生成标签、
   模型按它输出、下游审核系统按它消费。
2. ``parse_audit_output`` —— 把模型的一段原始文本解析成 ``AuditResult``。
   评测里的主指标「结构化输出格式遵循率」就是拿它算的：

       格式遵循率 = 能解析成合法 AuditResult 的样本数 / 总样本数

   判定口径刻意严格（见 ``AuditResult`` 的 model_config），因为原流程的痛点
   正是「输出格式不统一、还得人工整理」——一个下游程序解析不了的 JSON，
   哪怕内容对，也等于没解决问题。

注意区分两类指标，不要混为一谈：
  - **格式遵循率**（本模块）   ：输出是不是合法结构，与内容对错无关
  - **字段准确率**（evaluate.py）：结构合法之后，内容判得对不对
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from .taxonomy import Conclusion, IssueType, RiskLevel, Stage, Subject

__all__ = [
    "Question",
    "AuditIssue",
    "AuditResult",
    "ParseOutcome",
    "parse_audit_output",
    "audit_json_schema",
    "AUDIT_RESPONSE_FORMAT",
    "AUDIT_FIELD_VALUES",
]


class Question(BaseModel):
    """一道待审核的题目——模型的**输入**，与 ``AuditResult``（输出）对称。"""

    model_config = ConfigDict(str_strip_whitespace=True, use_attribute_docstrings=True)

    id: str
    """样本唯一标识。"""

    subject: Subject
    """学科。"""

    stage: Stage
    """学段。是否超纲要相对它判断。"""

    stem: str
    """题干。"""

    options: dict[str, str]
    """选项，形如 ``{"A": "...", "B": "..."}``。"""

    answer: str
    """命题人给出的参考答案选项号，如 ``"A"``。"""

    explanation: str
    """命题人给出的解析。审核「答案与解析是否矛盾」要靠它。"""


class AuditIssue(BaseModel):
    """一条审核发现的问题。"""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        use_attribute_docstrings=True,
    )

    type: IssueType
    """问题类型，取值见 taxonomy.IssueType。"""

    detail: str
    """具体问题描述，需指明题目中的哪一处、为什么有问题。"""

    severity: RiskLevel
    """该问题自身的严重程度，独立于整题的 risk_level。"""


class AuditResult(BaseModel):
    """一道题的完整审核结论——本项目的权威输出格式。"""

    model_config = ConfigDict(
        extra="forbid",  # 不接受 schema 之外的字段：下游按固定结构消费
        str_strip_whitespace=True,
        use_attribute_docstrings=True,
    )

    conclusion: Conclusion
    """审核结论：通过 / 需修改 / 不通过。"""

    risk_level: RiskLevel
    """整题风险等级，取所有 issues 中最高的 severity；无问题时为「低」。"""

    issues: list[AuditIssue]
    """发现的问题列表。审核通过时为空列表 ``[]``，而不是省略该字段。"""

    suggestion: str
    """修改建议。无问题时为空字符串。"""


# --------------------------------------------------------------------------
# 解析模型原始输出
# --------------------------------------------------------------------------

# 模型经常把 JSON 包在 markdown 代码块里，或前后带一句解释。
# 这两种都是「格式不规范」但不该直接判死——真实系统里也会做同样的事，
# 所以解析器先尝试容错提取，再交给严格校验。
_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class ParseOutcome:
    """一次解析的结果，比单纯返回 None 多带了失败原因，便于评测时归因。"""

    ok: bool
    result: AuditResult | None
    error: str | None = None
    extracted: bool = False
    """True 表示原文不是裸 JSON，是剥掉代码块/前后缀才拿到的。

    单独记录是为了能报告「裸 JSON 率」——如果一个模型全靠提取才达标，
    说明它并没真正学会输出规范格式。
    """

    reason: str = ""
    """失败归类，用于评测报告里统计失败模式：
    ``empty`` / ``no_json`` / ``bad_json`` / ``schema``。
    """


def _try_load(raw: str) -> tuple[Any | None, str, str, bool]:
    """按「裸 JSON -> 代码块 -> 首尾花括号」三级容错提取并 loads。

    返回 ``(对象, 失败原因, 原因码, 是否经过提取)``。

    失败原因给人看（含 JSON 解析器的原始报错），原因码给程序统计失败模式，
    两者分开，免得统计口径被长文本污染。
    """
    text = raw.strip()
    if not text:
        return None, "输出为空", "empty", False

    # 一级：整段就是 JSON
    try:
        return json.loads(text), "", "", False
    except json.JSONDecodeError:
        pass

    # 二级：markdown 代码块包裹
    if m := _FENCED_JSON.search(text):
        try:
            return json.loads(m.group(1).strip()), "", "", True
        except json.JSONDecodeError:
            pass

    # 三级：截取第一个 { 到最后一个 }，容忍前后夹带的解释性文字
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1]), "", True
        except json.JSONDecodeError as exc:
            # 把 JSON 解析器的报错原样带出去，评测归因时能看出是缺引号还是多了逗号
            return None, f"JSON 语法错误: {exc.msg} (第 {exc.lineno} 行第 {exc.colno} 列)", "bad_json", True

    return None, "输出中找不到 JSON 对象（既没有裸 JSON，也没有代码块或花括号）", "no_json", False


def parse_audit_output(raw: str) -> ParseOutcome:
    """把模型原始输出解析为 ``AuditResult``。

    失败不抛异常，而是返回 ``ok=False`` 的 ``ParseOutcome``——
    评测要统计失败率，不能让个别坏样本中断整个测试集。

    注意：本函数**只判格式**，不判结论对错。
    「conclusion 填了通过但实际该不通过」是格式合法的输出，
    会在 evaluate.py 里被记为格式通过、内容错误。
    """
    obj, error, reason, extracted = _try_load(raw)
    if obj is None:
        return ParseOutcome(ok=False, result=None, error=error, extracted=extracted, reason=reason)

    if not isinstance(obj, dict):
        return ParseOutcome(
            ok=False,
            result=None,
            error=f"顶层应为 JSON 对象，实际是 {type(obj).__name__}",
            extracted=extracted,
            reason="schema",
        )

    try:
        result = AuditResult.model_validate(obj)
    except ValidationError as exc:
        # 只保留前 3 条错误，避免长输出把报告刷屏
        problems = exc.errors()[:3]
        detail = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in problems
        )
        return ParseOutcome(ok=False, result=None, error=detail, extracted=extracted, reason="schema")

    return ParseOutcome(ok=True, result=result, extracted=extracted)


# --------------------------------------------------------------------------
# 供 prompt 与部署使用的 schema 描述
# --------------------------------------------------------------------------

def audit_json_schema() -> dict[str, Any]:
    """导出 JSON Schema，写入 configs/audit_schema.json 供上下游对齐。"""
    return AuditResult.model_json_schema()


# 内置到 system prompt 里的精简格式说明。
#
# 不用 model_json_schema() 直接塞给模型：那玩意带 $defs 和一堆 title，
# 对小模型是纯噪音，反而降低格式遵循率。
#
# 骨架里刻意**只放空字符串**，取值另用文字列出。原因是我们实测踩到的：
# 早期版本把骨架写成
#     "conclusion": "通过 | 需修改 | 不通过"
# 基座模型会把 `通过 | 需修改 | 不通过` 这整串当成字面量照抄下来，
# 于是输出因为枚举值非法而解析失败（评测里表现为 failure_reason="schema"）。
#
# 那要不要换成「填好的示例」？不行，会更糟：那样基座模型只要原样复读示例
# 就能通过 schema 校验拿到 100% 格式遵循率——它根本没在审核，指标却很好看。
# 格式遵循率衡量的应该是「模型能否产出可用的结构化结论」，而不是「模型能否
# 复读 prompt 里的 JSON」。
#
# 空字符串骨架两头都堵住了：它明显是占位符（不会被误当取值），
# 同时照抄它必然校验失败。要拿分只能真的按字段填。
AUDIT_RESPONSE_FORMAT = """{
  "conclusion": "",
  "risk_level": "",
  "issues": [
    {"type": "", "detail": "", "severity": ""}
  ],
  "suggestion": ""
}"""


# 与上面的骨架配套的取值说明，必须一起出现在 prompt 里，
# 否则模型知道结构却不知道合法值域。
AUDIT_FIELD_VALUES = """conclusion        —— 取「通过」/「需修改」/「不通过」之一
risk_level        —— 取「低」/「中」/「高」之一
issues            —— 数组，题目无任何问题时写空数组 []
issues[].type     —— 取「敏感表述」/「学科错误」/「表述歧义」/「超纲」/「格式不规范」之一
issues[].detail   —— 具体指出题目的哪一处有问题、为什么
issues[].severity —— 取「低」/「中」/「高」之一
suggestion        —— 修改建议；题目无任何问题时写空字符串"""
