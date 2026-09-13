"""prompt 契约测试。

这些测试守护的是「prompt 长什么样」这件事本身。看起来琐碎，但 prompt 是模型的
输入分布，改动的后果不会在单元测试里暴露，只会在几小时训练之后的评测数字里暴露。
所以把已知的坑固化成断言，比写在注释里可靠。
"""

from __future__ import annotations

import json

import pytest

from audit_llm.prompts import SYSTEM_PROMPT, build_messages, render_question
from audit_llm.schema import (
    AUDIT_FIELD_VALUES,
    AUDIT_RESPONSE_FORMAT,
    AuditResult,
    Question,
    parse_audit_output,
)


@pytest.fixture
def sample_question() -> Question:
    return Question(
        id="q-test",
        subject="数学",
        stage="初中",
        stem="已知一次函数 y = 2x + 1，求 x = 3 时的函数值。",
        options={"A": "5", "B": "6", "C": "7", "D": "8"},
        answer="C",
        explanation="将 x = 3 代入得 y = 2×3 + 1 = 7，故选 C。",
    )


# --------------------------------------------------------------------------
# 格式骨架不能被「照抄」拿分
# --------------------------------------------------------------------------

def test_format_skeleton_is_not_valid_on_its_own():
    """把骨架原样输出，必须**不能**通过 schema 校验。

    这是实测踩过的坑。早期版本骨架写的是
        "conclusion": "通过 | 需修改 | 不通过"
    基座模型直接把这一整串当字面量抄了下来，输出因枚举非法而失败。

    修的方向有两条，测试把两条都钉住：
      - 骨架本身不能含合法枚举值（否则照抄即可通过，指标虚高）
      - 骨架也不能是「照抄必然通过」的填空示例
    """
    outcome = parse_audit_output(AUDIT_RESPONSE_FORMAT)
    assert not outcome.ok, (
        "格式骨架被原样输出时竟然解析成功了——这意味着模型只要复读 prompt "
        "就能拿到格式遵循率，指标失去意义。"
    )
    assert outcome.reason == "schema", (
        f"期望因枚举值非法而失败（schema），实际 reason={outcome.reason!r}"
    )


def test_format_skeleton_has_no_enum_literals():
    """骨架里不该出现任何合法的枚举取值。"""
    from audit_llm.taxonomy import Conclusion, IssueType, RiskLevel

    skeleton = json.loads(AUDIT_RESPONSE_FORMAT)
    valid_values = {
        *(c.value for c in Conclusion),
        *(r.value for r in RiskLevel),
        *(t.value for t in IssueType),
    }
    # 骨架里出现的所有字符串值
    present = {skeleton["conclusion"], skeleton["risk_level"], skeleton["suggestion"]}
    for issue in skeleton["issues"]:
        present.update(issue.values())

    leaked = present & valid_values
    assert not leaked, f"骨架里泄漏了合法枚举值：{leaked}——模型照抄即可通过校验"


def test_field_values_lists_every_enum():
    """取值说明必须覆盖全部枚举，缺一个模型就可能自创取值。"""
    from audit_llm.taxonomy import Conclusion, IssueType, RiskLevel

    for member in (*Conclusion, *RiskLevel, *IssueType):
        assert member.value in AUDIT_FIELD_VALUES, (
            f"取值说明里没有 {member.value!r}，模型无从知道它合法"
        )


def test_system_prompt_contains_both_skeleton_and_values():
    """骨架和取值说明必须成对出现——只有骨架，模型知道结构却不知道值域。"""
    assert AUDIT_RESPONSE_FORMAT in SYSTEM_PROMPT
    assert AUDIT_FIELD_VALUES in SYSTEM_PROMPT


# --------------------------------------------------------------------------
# prompt 内容契约
# --------------------------------------------------------------------------

def test_every_injected_issue_type_is_described_in_prompt():
    """taxonomy 里定义的每一类问题，system prompt 都必须告诉模型怎么判。

    新增一类问题却忘了写进 prompt，模型永远预测不出这个类型——
    表现为该类别的召回率恒为 0，而这是静默的。
    """
    from audit_llm.taxonomy import IssueType

    for issue_type in IssueType:
        assert issue_type.value in SYSTEM_PROMPT, (
            f"问题类型 {issue_type.value!r} 没有在 system prompt 里说明"
        )


def test_question_rendering_includes_all_fields(sample_question: Question):
    """题干、选项、答案、解析都要进 prompt——审核「答案与解析矛盾」必须两边都看到。"""
    text = render_question(sample_question)
    assert sample_question.stem in text
    assert sample_question.answer in text
    assert sample_question.explanation in text
    for key, value in sample_question.options.items():
        assert f"{key}. {value}" in text


def test_messages_shape(sample_question: Question):
    msgs = build_messages(sample_question)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_PROMPT


def test_gold_json_is_parseable_by_the_same_parser():
    """训练用的标签 JSON，必须能被推理侧的解析器接受。

    两边用的是同一个 ``parse_audit_output``。如果标签本身通不过它，
    模型学到的东西和评测口径就错位了。
    """
    gold = AuditResult(
        conclusion="需修改",
        risk_level="中",
        issues=[{"type": "超纲", "detail": "涉及导数", "severity": "中"}],
        suggestion="替换为一次函数情境",
    )
    outcome = parse_audit_output(gold.model_dump_json())
    assert outcome.ok
    assert outcome.result == gold
    assert outcome.extracted is False, "标签应当就是裸 JSON，不该需要容错提取"
