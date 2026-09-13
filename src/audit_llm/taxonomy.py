"""命题审核的分类体系。

本模块只定义「审核有哪些维度、维度之间如何映射」，不涉及数据生成与序列化：
  - 数据生成     -> synth_data.py
  - 结构化校验   -> schema.py

设计原则
--------
审核结论必须能由题目中注入的缺陷**确定性地**推导出来（见 derive_conclusion）。
这样合成数据的标签才是可靠的 ground truth，而不是另一个模型的输出——
否则「用模型生成的标签去评测模型」会形成循环论证，指标毫无意义。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Iterable, Sequence


class Subject(StrEnum):
    """学科。取 7 个，覆盖初中/高中主流科目。

    注意：本项目的题目是模板合成的，**没有**接入任何公开教育数据集。
    学科划分只是按常见分类选的，不代表与某个具体数据集的 schema 对齐。
    """

    MATH = "数学"
    PHYSICS = "物理"
    CHEMISTRY = "化学"
    BIOLOGY = "生物"
    HISTORY = "历史"
    GEOGRAPHY = "地理"
    POLITICS = "政治"


class Stage(StrEnum):
    """学段。是否「超纲」要相对学段判断，所以它是审核的输入之一。"""

    JUNIOR = "初中"
    SENIOR = "高中"


class IssueType(StrEnum):
    """审核发现的问题类型。

    注意：这里**不含**「无缺陷」——审核通过就是 issues 为空列表，
    把「没有问题」建模成一种「问题」会让枚举语义变脏。
    """

    SENSITIVE = "敏感表述"      # 政治敏感、地域歧视、宗教争议
    SUBJECT_ERROR = "学科错误"  # 答案与解析矛盾、选项重复、知识点错误
    AMBIGUOUS = "表述歧义"      # 题干有歧义、缺少必要条件
    OUT_OF_SCOPE = "超纲"       # 超出本学段课程标准
    FORMAT = "格式不规范"       # 选项标点、单位缺失、排版问题


class RiskLevel(StrEnum):
    """风险等级。顺序敏感：LOW < MEDIUM < HIGH，见 _RISK_ORDER。"""

    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"


class Conclusion(StrEnum):
    """审核结论。由 issues 中最高的风险等级决定，见 derive_conclusion。"""

    PASS = "通过"
    REVISE = "需修改"
    REJECT = "不通过"


# --------------------------------------------------------------------------
# 风险等级排序
# --------------------------------------------------------------------------

_RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
}

# 最高风险等级 -> 审核结论。
#
# 「低风险不阻塞」是有意设计：题目可以有轻微格式问题但仍然通过审核，
# 这样 conclusion 与 risk_level 就不是一一对应，模型必须分别判断两个字段，
# 而不是学会「高风险=>不通过」这种退化解法。
_RISK_TO_CONCLUSION: dict[RiskLevel, Conclusion] = {
    RiskLevel.LOW: Conclusion.PASS,
    RiskLevel.MEDIUM: Conclusion.REVISE,
    RiskLevel.HIGH: Conclusion.REJECT,
}


def max_risk(levels: Iterable[RiskLevel]) -> RiskLevel | None:
    """返回一组风险等级中最高的一个；空集合返回 None。"""
    levels = list(levels)
    if not levels:
        return None
    return max(levels, key=lambda lv: _RISK_ORDER[lv])


def derive_conclusion(severities: Sequence[RiskLevel]) -> tuple[Conclusion, RiskLevel]:
    """由各问题的严重程度推导出 (审核结论, 总体风险等级)。

    这是整个数据集的标签生成规则，也是评测时判断「结论字段是否预测正确」的基准。

    规则：
        无问题              -> (通过,   低)
        最高风险 = 低       -> (通过,   低)     轻微问题不阻塞
        最高风险 = 中       -> (需修改, 中)
        最高风险 = 高       -> (不通过, 高)
    """
    top = max_risk(severities)
    if top is None:
        return Conclusion.PASS, RiskLevel.LOW
    return _RISK_TO_CONCLUSION[top], top


# --------------------------------------------------------------------------
# 问题类型的固有风险倾向
# --------------------------------------------------------------------------
# 只作为生成缺陷时的**采样权重**，不是硬约束：同一个问题类型可以有不同严重程度
# （例如「学科错误」里的选项重复只是中风险，答案与解析矛盾则是高风险）。
# 这样 risk_level 无法仅从 issues.type 推断，模型必须读到 detail 才能判对。

TYPICAL_SEVERITIES: dict[IssueType, tuple[RiskLevel, ...]] = {
    IssueType.SENSITIVE: (RiskLevel.HIGH,),
    IssueType.SUBJECT_ERROR: (RiskLevel.MEDIUM, RiskLevel.HIGH),
    IssueType.AMBIGUOUS: (RiskLevel.MEDIUM, RiskLevel.HIGH),
    IssueType.OUT_OF_SCOPE: (RiskLevel.LOW, RiskLevel.MEDIUM),
    IssueType.FORMAT: (RiskLevel.LOW,),
}

# 合成数据时的问题类型采样权重（相对值）。敏感表述刻意压低占比，
# 因为真实业务里它罕见但后果最严重——占比过高会让数据集失真。
ISSUE_SAMPLING_WEIGHT: dict[IssueType, float] = {
    IssueType.SENSITIVE: 0.06,
    IssueType.SUBJECT_ERROR: 0.30,
    IssueType.AMBIGUOUS: 0.24,
    IssueType.OUT_OF_SCOPE: 0.22,
    IssueType.FORMAT: 0.18,
}

ALL_SUBJECTS: tuple[Subject, ...] = tuple(Subject)
ALL_STAGES: tuple[Stage, ...] = tuple(Stage)
ALL_ISSUE_TYPES: tuple[IssueType, ...] = tuple(IssueType)
