"""合成数据的正确性测试。

这批数据是用来训练和评测模型的，标签错了整个项目就没有意义。
所以这里测的不是「代码不崩」，而是**标签和题目文本是否自洽**。

重点看 ``test_issue_detail_matches_question``：它守的是一个真实踩过的坑——
构造「答案与解析矛盾」的样本时，detail 里写的是注入**前**的答案字母，
而记录的 answer 已被改成错误项，导致「描述说标注为 D、数据里却是 C」。
这种不一致肉眼抽查很难发现，但会让模型学到矛盾的信号。
"""

from __future__ import annotations

import random
import re

import pytest

from audit_llm.schema import Question, parse_audit_output
from audit_llm.synth_data import _UNIT_RE, _inject_format, generate_dataset
from audit_llm.taxonomy import Conclusion, RiskLevel, derive_conclusion
from audit_llm.templates import TEMPLATES, distinct_or_none, validate_templates


@pytest.fixture(scope="module")
def records() -> list[dict]:
    """一个中等规模的样本集，多个测试共用，避免重复生成。"""
    return generate_dataset(400, seed=20240913)


# ==========================================================================
# 模板层
# ==========================================================================

def test_templates_have_no_missing_slots() -> None:
    """每个模板引用的槽位都必须能被它的 slot 函数提供。"""
    assert validate_templates() == []


def test_template_options_are_distinct() -> None:
    """四个选项必须互不相同，否则「选项重复」会被误当成我们注入的缺陷。"""
    rng = random.Random(0)
    for tpl in TEMPLATES:
        built = tpl.build(rng)
        opts = [built["correct"], *built["wrong"]]
        assert distinct_or_none(opts), f"{tpl.subject}/{tpl.stage} 选项撞车: {opts}"


def test_every_subject_stage_has_its_own_templates() -> None:
    """每个「学科 × 学段」都必须有专属模板，不能依赖 templates_for 的回退。

    ``templates_for`` 在找不到精确匹配时会退回该学科的全部模板，
    但调用方仍会把记录的 ``stage`` 写成请求的那个——结果是**用初中模板造出的题
    被打上「高中」标签**。学段是判断「是否超纲」的依据，标错了标签就是错的。

    实测这个回退目前从不触发（14 个组合全都有专属模板），所以它只是一层
    安全网。正因如此更该锁住：有人删掉一个模板时，回退会**静默**生效，
    数据悄悄变脏而没有任何报错。
    """
    from audit_llm.taxonomy import Stage, Subject
    from audit_llm.templates import templates_for

    for subject in Subject:
        for stage in Stage:
            exact = [t for t in TEMPLATES if t.subject == subject and t.stage == stage]
            assert exact, (
                f"{subject}/{stage} 没有专属模板，会触发学段回退，"
                f"造出学段标注与题目不符的脏数据"
            )
            # 回退结果必须与精确结果一致——即回退确实没被用上
            assert set(templates_for(subject, stage)) == set(exact)


# ==========================================================================
# 标签层
# ==========================================================================

def test_audit_label_parses(records: list[dict]) -> None:
    """标签本身必须是合法 AuditResult——它是格式遵循率的判定基准。"""
    import json

    for rec in records:
        outcome = parse_audit_output(json.dumps(rec["audit"], ensure_ascii=False))
        assert outcome.ok, f"{rec['id']} 的标签不合法: {outcome.error}"


def test_conclusion_derived_from_severities(records: list[dict]) -> None:
    """conclusion / risk_level 必须能由 issues 的 severity 唯一推出。"""
    for rec in records:
        audit = rec["audit"]
        expected_conclusion, expected_risk = derive_conclusion(
            [i["severity"] for i in audit["issues"]]
        )
        assert audit["conclusion"] == expected_conclusion, rec["id"]
        assert audit["risk_level"] == expected_risk, rec["id"]


def test_suggestion_present_iff_issues_present(records: list[dict]) -> None:
    """suggestion 为空当且仅当没有问题。

    注意：conclusion 为「通过」时 suggestion 仍可能非空——那是低风险问题
    不阻塞入库的情况（见 taxonomy._RISK_TO_CONCLUSION 的说明）。
    所以这里断言的是 issues，不是 conclusion。
    """
    for rec in records:
        audit = rec["audit"]
        assert (audit["suggestion"] == "") == (len(audit["issues"]) == 0), rec["id"]


def test_issue_detail_is_never_blank(records: list[dict]) -> None:
    """detail 不能是空字符串——空 detail 的样本对训练毫无价值。"""
    for rec in records:
        for issue in rec["audit"]["issues"]:
            assert issue["detail"].strip(), f"{rec['id']} 有空 detail"


# ==========================================================================
# 标签与题目文本的一致性（核心）
# ==========================================================================

def _quoted(detail: str) -> str | None:
    """取出 detail 里「」中引用的那个术语/片段。"""
    start, end = detail.find("「"), detail.find("」")
    if start == -1 or end <= start:
        return None
    return detail[start + 1 : end]


def _quoted_letter(detail: str) -> str | None:
    """取出 detail 里「选项 X」的那个选项号。"""
    match = re.search(r"选项\s*([A-D])", detail)
    return match.group(1) if match else None


def test_injected_defect_is_visible_in_text(records: list[dict]) -> None:
    """注入的缺陷必须真的体现在题目文本里。

    如果只在标签里写「超纲」而题干解析看不出超纲，模型学到的就是
    「凭空猜标签」，线上面对干净题目时会失效。

    这里用最强的一种检查：把 detail 里引号引用的那个片段抠出来，
    断言它**确实出现在题目文本中**。这比检查「detail 里有没有『超纲』二字」
    有意义得多——后者只是措辞问题，前者才是「缺陷可见」的实质。
    """
    checked = 0
    for rec in records:
        for issue in rec["audit"]["issues"]:
            quoted = _quoted(issue["detail"])
            if quoted is None:
                continue
            question = rec["question"]
            haystack = question["stem"] + question["explanation"] + " ".join(question["options"].values())

            if issue["type"] == "超纲":
                # 超纲术语是追加到解析里的
                assert quoted in question["explanation"], (
                    f"{rec['id']}: detail 说解析用了「{quoted}」，但解析里没有"
                )
                checked += 1
            elif issue["type"] == "敏感表述":
                # 敏感内容插在题干末尾
                assert quoted in question["stem"], (
                    f"{rec['id']}: detail 说题干含「{quoted}」，但题干里没有"
                )
                checked += 1
            elif issue["type"] == "表述歧义" and "缺少限定条件" in issue["detail"]:
                # 被删掉的限定词必须真的**不在**题干里了
                assert quoted not in question["stem"], (
                    f"{rec['id']}: detail 说删掉了「{quoted}」，但题干里还在"
                )
                checked += 1
            else:
                # 其余类型至少要求引用的片段能在题目文本里找到
                assert quoted in haystack, (
                    f"{rec['id']}: detail 引用了「{quoted}」，但题目文本里找不到"
                )
                checked += 1

    assert checked > 0, "没有可校验的样本，测试形同虚设"


def test_answer_explanation_conflict_detail_matches_record(records: list[dict]) -> None:
    """「答案与解析矛盾」样本的 detail 必须与记录里的答案字母一致。

    这是回归测试：曾出现 detail 写「参考答案标注为 D」而 answer 字段是 C，
    因为 detail 用了注入前的答案字母。
    """
    checked = 0
    for rec in records:
        for issue in rec["audit"]["issues"]:
            if issue["type"] != "学科错误" or "相互矛盾" not in issue["detail"]:
                continue
            checked += 1
            answer_letter = rec["question"]["answer"]
            assert f"参考答案标注为 {answer_letter}" in issue["detail"], (
                f"{rec['id']}: detail 与 answer 字段不一致\n"
                f"  answer = {answer_letter}\n"
                f"  detail = {issue['detail']}"
            )
    assert checked > 0, "样本集里没有「答案与解析矛盾」的样本，测试形同虚设"


def test_duplicate_option_detail_is_true(records: list[dict]) -> None:
    """声称「选项重复」的样本，题目里必须真的有两个选项内容相同。"""
    checked = 0
    for rec in records:
        for issue in rec["audit"]["issues"]:
            if issue["type"] != "学科错误" or "完全相同" not in issue["detail"]:
                continue
            checked += 1
            values = list(rec["question"]["options"].values())
            assert len(set(values)) < len(values), (
                f"{rec['id']}: 标签说选项重复，但四个选项互不相同"
            )
    assert checked > 0, "样本集里没有「选项重复」的样本"


def _format_question(options: dict[str, str], answer: str = "B") -> Question:
    return Question(
        id="q-fmt",
        subject="物理",
        stage="初中",
        stem="某物体的质量为 308 g，求其体积。",
        options=options,
        answer=answer,
        explanation="由密度公式可得。",
    )


def test_unit_format_injection_declines_when_others_lack_units() -> None:
    """「其余选项均带单位」这句全称断言，不成立时注入器就必须放弃。

    实测踩到的脏数据：

        选项 B「308」缺少物理量单位，而其余选项均带单位，格式不统一

    可同题的选项 C 也是「308」，同样不带单位。前半句是真的，后半句是假的。
    模型会从这种样本里学到「detail 可以随便写」。

    **这条测试是确定性的，不靠随机采样碰运气。** 最初我把它写成「遍历生成的
    样本集，检查每一条格式问题的全称断言」——看起来更全面，实际上没有牙齿：
    这个冲突在 2500 条里才出现 1 次，400 条的测试集根本抽不到，把守卫删掉
    测试照样全绿。所以改成直接构造出冲突场景。
    """
    rng = random.Random(0)

    # 选项 C 也不带单位 —— 不能说「其余选项均带单位」，必须落到标点模式
    q = _format_question({"A": "1.0 g/cm³", "B": "308 g", "C": "308", "D": "2.0 g/cm³"})
    injection = _inject_format(rng, q)
    assert injection is not None
    assert "其余选项均带单位" not in injection.issue.detail, (
        "选项 C 不带单位，却声称「其余选项均带单位」——这就是脏标签"
    )
    assert "末尾使用了逗号" in injection.issue.detail

    # 反例：其余选项都带单位时，应当走缺单位模式
    q2 = _format_question({"A": "1.0 g/cm³", "B": "308 g", "C": "3.0 g/cm³", "D": "2.0 g/cm³"})
    injection2 = _inject_format(rng, q2)
    assert injection2 is not None
    assert "缺少物理量单位" in injection2.issue.detail
    assert _UNIT_RE.search(injection2.options["B"]) is None, "单位没被剥掉"


def test_punct_format_injection_declines_when_others_end_with_comma() -> None:
    """同理：「其余选项均以句号或无标点结尾」也要真成立才能说。"""
    rng = random.Random(0)
    # 选项 C 已经以逗号结尾，此时再说「其余选项都不以逗号结尾」就是假的
    q = _format_question({"A": "甲，", "B": "乙", "C": "丙，", "D": "丁"})
    injection = _inject_format(rng, q)
    if injection is not None:
        assert "其余选项均以句号或无标点结尾" not in injection.issue.detail, (
            "选项 C 以逗号结尾，却声称其余选项都不以逗号结尾"
        )


def test_format_issue_details_match_data(records: list[dict]) -> None:
    """在真实生成的样本上复查一遍全称断言（覆盖面，不作唯一防线）。

    上面两条确定性测试才是主防线；这条是兜底——万一以后新增了别的
    「而其余选项……」句式，这里能扫到。
    """
    checked = 0
    for rec in records:
        options = rec["question"]["options"]
        for issue in rec["audit"]["issues"]:
            if issue["type"] != "格式不规范":
                continue
            detail = issue["detail"]
            letter = _quoted_letter(detail)
            others = {k: v for k, v in options.items() if k != letter}

            if "其余选项均带单位" in detail:
                checked += 1
                offenders = {k: v for k, v in others.items() if not _UNIT_RE.search(v)}
                assert not offenders, f"{rec['id']}: detail 与题面不符：{offenders}"
            elif "其余选项均以句号或无标点结尾" in detail:
                checked += 1
                offenders = {k: v for k, v in others.items() if v.endswith("，")}
                assert not offenders, f"{rec['id']}: detail 与题面不符：{offenders}"
    assert checked > 0, "样本集里没有「格式不规范」的样本"


def test_unit_regex_does_not_mangle_compound_units() -> None:
    """剥离单位的正则必须锚定在末尾，且能识别复合单位。

    没有末尾锚定时，`m/s²` 会被更短的 `m/s` 抢先匹配，留下一个孤零零的 `²`——
    造出「选项内容是 `5 ²`」这种现实中不存在的题面。模板库目前没有复合单位，
    但只要以后加一个，这种错就会**静默**发生：没有异常，只是数据变脏了。
    """
    for text, want in [
        ("5 m/s²", "5"),
        ("9.8 m/s²", "9.8"),
        ("308 m", "308"),
        ("3.0×10⁸ m/s", "3.0×10⁸"),
        ("1.5 g/cm³", "1.5"),
        ("18 g/mol", "18"),
        ("220 V", "220"),
        ("308", "308"),  # 本来就带单位的不该被动
    ]:
        assert _UNIT_RE.sub("", text).strip() == want, f"{text!r} 剥离结果不对"

    # 锚定在末尾：单位出现在中间时不应匹配，这样注入器会自然地落到标点模式，
    # 而不是硬剥出一个错误的标签
    assert _UNIT_RE.search("以 5 m 的速度前进") is None


def test_clean_samples_have_no_issues(records: list[dict]) -> None:
    """没注入缺陷的样本，issues 必须为空且结论为通过。"""
    clean = [r for r in records if not r["injected"]]
    assert clean, "样本集里没有干净样本，正负比例失衡"
    for rec in clean:
        assert rec["audit"]["issues"] == [], rec["id"]
        assert rec["audit"]["conclusion"] == Conclusion.PASS
        assert rec["audit"]["risk_level"] == RiskLevel.LOW


# ==========================================================================
# 数据集整体性质
# ==========================================================================

def test_dataset_has_both_positive_and_negative(records: list[dict]) -> None:
    """正负样本都要有，且都不能太少，否则模型会退化成「一律通过/一律不通过」。"""
    pass_rate = sum(r["audit"]["conclusion"] == "通过" for r in records) / len(records)
    assert 0.25 <= pass_rate <= 0.75, f"通过率 {pass_rate:.2%} 过于极端"


def test_all_subjects_and_stages_covered(records: list[dict]) -> None:
    """7 个学科都要出现，否则模型对缺席学科毫无能力。"""
    subjects = {r["question"]["subject"] for r in records}
    assert len(subjects) == 7, f"学科覆盖不全: {sorted(subjects)}"


def test_dataset_is_reproducible() -> None:
    """同一个 seed 必须产出完全相同的数据——否则评测结果无法复现。"""
    a = generate_dataset(50, seed=7)
    b = generate_dataset(50, seed=7)
    assert a == b


def test_injected_field_not_leaked_into_question() -> None:
    """injected 是我们的质检元数据，不能出现在模型能看到的问题文本里。"""
    records = generate_dataset(120, seed=99)
    for rec in records:
        blob = rec["question"]["stem"] + rec["question"]["explanation"]
        assert "injected" not in blob
        # 缺陷类型名本身也不该出现在题干里（否则等于把答案写在题面上）
        for itype in ("敏感表述", "学科错误", "表述歧义", "格式不规范"):
            assert itype not in blob, f"{rec['id']} 题干泄露了缺陷类型名 {itype}"


def test_split_isolation_by_base_question() -> None:
    """训练集和测试集不能共用同一道底题。

    这是评测有效性的底线：底题相同、只是坏法不同，测试指标就没有意义。
    比的是 ``base_key``（注入前的干净题目）而不是最终题干——同一底题注入
    不同缺陷后题干会不同，但底题还是那一道。
    """
    train = generate_dataset(600, seed=11)
    train_keys = {r["base_key"] for r in train}

    test = generate_dataset(
        80,
        seed=12,
        forbidden_keys=train_keys,
        require_unique_keys=True,
    )

    assert len(test) == 80, "隔离之后凑不够样本数"
    assert {r["base_key"] for r in test} & train_keys == set(), "测试集里混入了训练集的底题"
    assert len({r["base_key"] for r in test}) == 80, "测试集内部底题不唯一"


def test_dataset_reports_base_key() -> None:
    """每条样本都要带 base_key，否则切分隔离无从谈起。"""
    records = generate_dataset(50, seed=5)
    for rec in records:
        assert rec.get("base_key"), f"{rec['id']} 缺 base_key"
