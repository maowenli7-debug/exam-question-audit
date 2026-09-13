"""合成「题目 + 审核结论」指令数据。

核心思路：**先注入缺陷，再反推结论**
----------------------------------
如果让另一个大模型去给题目写审核意见，那份「标签」本身就可能是错的，
用它去训练/评测等于循环论证。所以这里反过来做：

    1. 从模板生成一道**干净**的题目
    2. 按概率**真实地改动题目文本**，注入已知类型的缺陷
    3. 由「实际注入了什么缺陷」**确定性地**推导出审核结论

这样标签是构造出来的 ground truth，不是模型猜的。代价是题目本身是模板产物，
语言真实度有限——这个局限在 README 里明确写了，不藏着。

注入的缺陷必须**可见**
----------------------
注入不能只是给题目挂一个「有超纲问题」的标签，而必须真的把超纲内容写进题目
文本。否则模型学到的是「看标签猜标签」，线上面对干净题目时会失效。

自带校验闭包（``Injection.verify``）
----------------------------------
多个注入器叠加时**会互相干扰**。真实踩过的坑：先注入「选项重复」制造了两个
相同的选项，紧接着「格式不规范」给其中一个加了逗号，两个选项就不再相同了，
可标签里还写着「内容完全相同」。

靠逐个给注入器打补丁堵不住这类问题，所以改成：每个注入器**自带一个校验闭包**，
声明「我注入完之后，题目应该满足什么条件」。全部注入完成后统一跑一遍，
任何一条不成立就丢弃整条样本重新生成。

校验闭包捕获的是**注入前**的变量（如被删掉的限定词），所以它检查的是
「当时的改动到最终题目里还成立吗」，这正是我们需要的语义。
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .schema import AuditIssue, AuditResult, Question
from .taxonomy import (
    ALL_STAGES,
    ALL_SUBJECTS,
    ISSUE_SAMPLING_WEIGHT,
    TYPICAL_SEVERITIES,
    IssueType,
    RiskLevel,
    Stage,
    Subject,
    derive_conclusion,
)
from .templates import distinct_or_none, templates_for

__all__ = ["generate_dataset", "write_dataset", "Sample", "main"]

# 生成一条样本的最大重试次数。注入器互相干扰、槽位撞车都会触发重试，
# 正常情况下几百次里碰不到一次，给 20 次已经很宽裕。
_MAX_ATTEMPTS = 20


# ==========================================================================
# 缺陷注入器
# ==========================================================================

@dataclass(frozen=True)
class Injection:
    """一次成功的缺陷注入：**标签** + **改动后的题目文本** + **自校验**。

    ``verify`` 接收最终题目（可能已被后续注入器继续修改），返回这次注入
    声称的缺陷是否依然成立。
    """

    issue: AuditIssue
    stem: str
    options: dict[str, str]
    answer: str
    explanation: str
    verify: Callable[[Question], bool]


# --- 敏感表述 ---------------------------------------------------------------
# 说明：这里刻意使用**温和且模板化**的表述，覆盖的是教育命题审核里真实存在的
# 类别（商业推广、不适宜场景、地域刻板印象、宗教内容），而不是真实的政治敏感
# 或歧视性文本。目的是让模型学会识别这类**模式**，不是生成有害内容。
_SENSITIVE_BANK: tuple[tuple[str, str], ...] = (
    (
        "（本题由××教育科技有限公司独家提供，版权所有）",
        "题干中夹带商业推广信息，违反命题不得植入广告的规范",
    ),
    (
        "（某同学用本题答案与同桌打赌，赌注为一包香烟）",
        "题干情境涉及赌博与烟草，不适宜作为学生试题背景",
    ),
    (
        "（据称××地区的学生普遍做不出这道题）",
        "题干包含地域刻板印象表述，可能引发地域歧视",
    ),
    (
        "（本题依据××宗教教义改编，非该教信仰者难以作答）",
        "题干引入宗教内容并暗示信仰差异，不适宜作为通用试题",
    ),
)


def _inject_sensitive(rng: random.Random, q: Question) -> Injection | None:
    insert, detail = rng.choice(_SENSITIVE_BANK)
    return Injection(
        issue=AuditIssue(type=IssueType.SENSITIVE, detail=detail, severity=RiskLevel.HIGH),
        stem=q.stem + insert,
        options=dict(q.options),
        answer=q.answer,
        explanation=q.explanation,
        verify=lambda final, text=insert: text in final.stem,
    )


# --- 学科错误 ---------------------------------------------------------------
# 两种模式，各自对应不同的风险等级：
#   答案与解析矛盾 -> 高风险（学生会被教错）
#   选项重复       -> 中风险（无法唯一作答，但至少没教错）
# 「答案与解析矛盾」最能体现审核的价值——题目单看没错，必须把答案和解析
# 对读才能发现，这正是人工审核最容易漏掉、也最该自动化的一类。

def _inject_subject_error(rng: random.Random, q: Question) -> Injection | None:
    wrong_letters = sorted(set(q.options) - {q.answer})
    if not wrong_letters:
        return None

    if rng.random() < 0.5:
        # 把参考答案改成错误项，解析原样保留 -> 两者对不上。
        # detail 里必须写**改动后**的答案字母 bad，以及解析真正支持的那个选项
        # （q.answer，即正确项）。曾经这里误用了 q.answer 当「标注答案」，
        # 导致 detail 说「标注为 D」而记录里 answer 是 C，标签与题目自相矛盾。
        bad = rng.choice(wrong_letters)
        return Injection(
            issue=AuditIssue(
                type=IssueType.SUBJECT_ERROR,
                detail=(
                    f"参考答案标注为 {bad}，但解析推导出的结论是「{q.options[q.answer]}」"
                    f"（选项 {q.answer}），答案与解析相互矛盾"
                ),
                severity=RiskLevel.HIGH,
            ),
            stem=q.stem,
            options=dict(q.options),
            answer=bad,
            explanation=q.explanation,
            # 最终题目里答案必须仍是 bad，且解析支持的那个选项文本没被改掉
            verify=lambda final, bad=bad, correct=q.answer, text=q.options[q.answer]: (
                final.answer == bad and final.options.get(correct) == text
            ),
        )

    dup_from = rng.choice(wrong_letters)
    options = dict(q.options)
    options[dup_from] = q.options[q.answer]
    return Injection(
        issue=AuditIssue(
            type=IssueType.SUBJECT_ERROR,
            detail=f"选项 {dup_from} 与选项 {q.answer} 内容完全相同，正确答案不唯一",
            severity=RiskLevel.MEDIUM,
        ),
        stem=q.stem,
        options=options,
        answer=q.answer,
        explanation=q.explanation,
        # 最终题目里必须**真的**存在两个完全相同的选项
        verify=lambda final: len(set(final.options.values())) < len(final.options),
    )


# --- 表述歧义 ---------------------------------------------------------------
# 两种模式，都是**删掉解题所必需的信息**，而不是另加一句模糊的话：
#   模式一：删掉限定条件（「匀速」「完全」…）——题目变得无法唯一作答
#   模式二：把具体数值换成「若干」——缺少必要条件，无法计算
# 第二种是兜底，因为不是每道题的题干里都有可删的限定词；
# 两者都对应真实命题事故：作者忘了写条件、模板占位符没替换掉。

_QUALIFIERS = ("匀速", "完全", "充分", "标准状况下", "在常温常压下", "密闭", "纯合")
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _inject_ambiguous(rng: random.Random, q: Question) -> Injection | None:
    severity = rng.choice(TYPICAL_SEVERITIES[IssueType.AMBIGUOUS])

    # 模式一：删掉限定条件。detail 描述的是**被删掉的那个词**，
    # 所以它注入后不应该再出现在题干里——校验闭包检查的正是这一点。
    if present := [w for w in _QUALIFIERS if w in q.stem]:
        word = rng.choice(present)
        return Injection(
            issue=AuditIssue(
                type=IssueType.AMBIGUOUS,
                detail=f"题干缺少限定条件「{word}」，导致存在多种理解、无法唯一作答",
                severity=severity,
            ),
            stem=q.stem.replace(word, "", 1),
            options=dict(q.options),
            answer=q.answer,
            explanation=q.explanation,
            verify=lambda final, w=word: w not in final.stem,
        )

    # 模式二：把题干里的具体数值替换成「若干」，制造「缺少必要条件」。
    # 比起编一个「指代不明」的标签，这种改法在文本上可见、可验证。
    if _NUMBER_RE.search(q.stem):
        return Injection(
            issue=AuditIssue(
                type=IssueType.AMBIGUOUS,
                detail="题干未给出具体数值，仅以「若干」等模糊表述代替，缺少解题的必要条件，无法唯一作答",
                severity=severity,
            ),
            stem=_NUMBER_RE.sub("若干", q.stem),
            options=dict(q.options),
            answer=q.answer,
            explanation=q.explanation,
            verify=lambda final: "若干" in final.stem,
        )

    # 既没有可删的限定词、也没有具体数值：本题不适合注入歧义缺陷，
    # 交给生成器改选其他类型，而不是硬造一个站不住的标签。
    return None


# --- 超纲 -------------------------------------------------------------------
# 只在初中学段注入：初中题引用「导数」是无歧义的超纲。
# 高中题目的超纲边界模糊（各地考纲差异大），强行注入会造出有争议的标签，
# 所以高中阶段这个注入器直接不适用（返回 None），由生成器改选其他缺陷。

_OUT_OF_SCOPE_TERMS: dict[Subject, tuple[str, ...]] = {
    Subject.MATH: ("导数", "极限", "矩阵运算", "泰勒展开", "偏导数"),
    Subject.PHYSICS: ("麦克斯韦方程组", "薛定谔方程", "洛伦兹变换", "拉格朗日量"),
    Subject.CHEMISTRY: ("分子轨道理论", "能带理论", "配位场理论", "吉布斯自由能变"),
    Subject.BIOLOGY: ("基因编辑技术", "蛋白质组学", "表观遗传调控", "全基因组关联分析"),
    Subject.HISTORY: ("年鉴学派", "计量史学方法", "布罗代尔长时段理论"),
    Subject.GEOGRAPHY: ("遥感反演模型", "地理信息系统空间分析", "大气环流数值模拟"),
    Subject.POLITICS: ("边际效用递减规律", "博弈论纳什均衡", "公共选择理论"),
}


def _inject_out_of_scope(rng: random.Random, q: Question) -> Injection | None:
    if q.stage is not Stage.JUNIOR:
        return None  # 见上文说明：高中超纲边界模糊，不注入
    terms = _OUT_OF_SCOPE_TERMS.get(q.subject)
    if not terms:
        return None
    term = rng.choice(terms)
    severity = rng.choice(TYPICAL_SEVERITIES[IssueType.OUT_OF_SCOPE])
    return Injection(
        issue=AuditIssue(
            type=IssueType.OUT_OF_SCOPE,
            detail=f"解析中使用了「{term}」，超出初中学段课程标准要求，学生无法据此理解",
            severity=severity,
        ),
        stem=q.stem,
        options=dict(q.options),
        answer=q.answer,
        explanation=f"{q.explanation}（也可用{term}的相关结论直接验证。）",
        verify=lambda final, t=term: t in final.explanation,
    )


# --- 格式不规范 -------------------------------------------------------------
# 两类：物理量缺单位、选项标点不统一。都是一眼可见但真实高频的命题事故。

# 单位表按「长的排前面」排列，并且**锚定在末尾**。
# 不锚定会出事：`m/s²` 会被 `m/s` 抢先匹配，剥离后留下一个孤零零的 `²`，
# 造出「选项内容是 `5 ²`」这种不存在的题面。模板库现在没有复合单位，
# 但只要以后有人加一个，这种错就会静默发生——所以锚定和顺序都是必须的。
# 顺带一提，锚定还让「选项不是以单位结尾」的情况自然地落到标点模式，
# 而不是硬剥出一个错误的标签。
_UNIT_RE = re.compile(
    r"\s*(cm/s²|m/s²|m/s|km/h|cm³|cm²|m·s⁻²|g/cm³|g/mol|kg|km|mL|L|m|s|g|Ω|V|A|%)$"
)


def _inject_format(rng: random.Random, q: Question) -> Injection | None:
    correct_opt = q.options[q.answer]
    stripped = _UNIT_RE.sub("", correct_opt).strip()

    # 只有「其余选项都带单位」时才能说「格式不统一」。
    # 不加这个条件就会写出与题面不符的 detail——实测曾产出过
    # 「选项 B「308」缺少单位，而其余选项均带单位」，可同题的选项 C 也是「308」。
    # 标签说了文本里没有的事，就是脏数据。
    others = [v for k, v in q.options.items() if k != q.answer]
    if stripped and stripped != correct_opt and all(_UNIT_RE.search(v) for v in others):
        options = dict(q.options)
        options[q.answer] = stripped
        return Injection(
            issue=AuditIssue(
                type=IssueType.FORMAT,
                detail=f"选项 {q.answer}「{stripped}」缺少物理量单位，而其余选项均带单位，格式不统一",
                severity=RiskLevel.LOW,
            ),
            stem=q.stem,
            options=options,
            answer=q.answer,
            explanation=q.explanation,
            verify=lambda final, letter=q.answer, text=stripped: (
                final.options.get(letter) == text
                and all(_UNIT_RE.search(v) for k, v in final.options.items() if k != letter)
            ),
        )

    # 标点模式：避开已经重复的选项组。
    # 若某个选项的内容与另一个选项完全相同（可能来自「选项重复」注入），
    # 给它加标点会让两者不再相同，直接毁掉另一条标签的成立条件。
    counts = Counter(q.options.values())
    unique_letters = [l for l in sorted(q.options) if counts[q.options[l]] == 1]
    if not unique_letters:
        return None
    victim = unique_letters[-1]

    # 同理，「其余选项均以句号或无标点结尾」也得真成立，
    # 否则同题里已经有个逗号结尾的选项时，这句话就是假的。
    others = [v for k, v in q.options.items() if k != victim]
    if any(v.endswith("，") for v in others):
        return None

    options = dict(q.options)
    options[victim] = options[victim].rstrip("。.") + "，"
    return Injection(
        issue=AuditIssue(
            type=IssueType.FORMAT,
            detail=f"选项 {victim} 末尾使用了逗号，其余选项均以句号或无标点结尾，标点使用不统一",
            severity=RiskLevel.LOW,
        ),
        stem=q.stem,
        options=options,
        answer=q.answer,
        explanation=q.explanation,
        verify=lambda final, v=victim: (
            final.options.get(v, "").endswith("，")
            and not any(
                t.endswith("，") for k, t in final.options.items() if k != v
            )
        ),
    )


_INJECTORS = {
    IssueType.SENSITIVE: _inject_sensitive,
    IssueType.SUBJECT_ERROR: _inject_subject_error,
    IssueType.AMBIGUOUS: _inject_ambiguous,
    IssueType.OUT_OF_SCOPE: _inject_out_of_scope,
    IssueType.FORMAT: _inject_format,
}

# 每种缺陷类型对应的修改建议。注入器只负责「发现」，建议统一在这里给，
# 避免同一类问题在不同注入路径下写出措辞不一的建议。
_SUGGESTIONS: dict[IssueType, str] = {
    IssueType.SENSITIVE: "删除题干中的不适宜内容，重新表述题目情境",
    IssueType.SUBJECT_ERROR: "核对参考答案与解析，确保两者推导一致且各选项互不重复",
    IssueType.AMBIGUOUS: "补全题干中的限定条件，消除歧义后再入库",
    IssueType.OUT_OF_SCOPE: "改用本学段课程标准内的知识点表述，或将题目调整到相应学段",
    IssueType.FORMAT: "统一各选项的标点与单位格式",
}


# ==========================================================================
# 生成流程
# ==========================================================================

@dataclass
class Sample:
    question: Question
    audit: AuditResult
    injected: list[IssueType] = field(default_factory=list)
    """注入了哪些缺陷。**仅供数据质检与统计，绝不进入 prompt**——
    它属于答案的一部分，泄露给模型就等于作弊。"""

    base_key: str = ""
    """**注入之前**那道干净题目的指纹。

    划分训练/测试集必须按它来，不能按最终题干：同一道底题注入不同缺陷后
    题干会变得各不相同，但底题本身是同一个。如果不拦住，测试集里的题
    可能只是训练集某道题的「另一种坏法」，评测就失去意义了。
    """


def _clean_question(rng: random.Random) -> Question | None:
    """生成一道干净题目。槽位导致选项撞车时返回 None，由调用方重试。"""
    subject = rng.choice(ALL_SUBJECTS)
    stage = rng.choice(ALL_STAGES)
    tpl = rng.choice(templates_for(subject, stage))

    built = tpl.build(rng)
    correct = built["correct"]
    wrong = built["wrong"]
    if not distinct_or_none([correct, *wrong]):
        return None

    # 洗牌决定正确选项落在 A/B/C/D 哪一位
    letters = ["A", "B", "C", "D"]
    texts = [correct, *wrong]
    rng.shuffle(texts)
    options = dict(zip(letters, texts))
    answer = letters[texts.index(correct)]

    return Question(
        id="",  # 留空，由 generate_dataset 按最终序号补
        subject=subject,
        stage=stage,
        stem=built["stem"],
        options=options,
        answer=answer,
        explanation=built["explanation"],
    )


def _apply(question: Question, injection: Injection) -> Question:
    """把注入结果写回题目。"""
    return question.model_copy(
        update={
            "stem": injection.stem,
            "options": injection.options,
            "answer": injection.answer,
            "explanation": injection.explanation,
        }
    )


def _base_key(question: Question) -> str:
    """干净题目的指纹，用于切分时判重。

    用注入**前**的题干、选项、答案和解析共同计算——只看题干不够，
    数学类模板同一题干配不同选项也是不同的题。
    """
    payload = "\x1f".join(
        [question.stem, question.answer, question.explanation, *sorted(question.options.values())]
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _try_make_sample(rng: random.Random, defect_rate: float) -> Sample | None:
    """尝试生成一条样本。校验不通过时返回 None，由调用方重试。"""
    question = _clean_question(rng)
    if question is None:
        return None

    key = _base_key(question)  # 必须在注入之前算，注入会改掉题干和选项
    injections: list[Injection] = []

    if rng.random() < defect_rate:
        # 注入 1 个缺陷（多数），偶尔叠加第 2 个——多缺陷样本让模型学会
        # 在一条 issues 里列出多个问题，而不是只报最常见的那一个。
        n_defects = 2 if rng.random() < 0.25 else 1
        remaining = list(_INJECTORS)

        for _ in range(n_defects):
            # 按 ISSUE_SAMPLING_WEIGHT 加权抽类型。某个注入器对本条题目不适用
            # （返回 None，例如「超纲」只对初中生效）时，把它从本轮候选里剔除再抽，
            # 而不是顺序遍历——顺序遍历会让排在前面的类型被过度采样，
            # 权重就形同虚设了。
            tried: set[IssueType] = set()
            for _attempt in range(len(remaining)):
                pool = [t for t in remaining if t not in tried]
                if not pool:
                    break
                itype = rng.choices(pool, weights=[ISSUE_SAMPLING_WEIGHT[t] for t in pool], k=1)[0]
                tried.add(itype)

                injection = _INJECTORS[itype](rng, question)
                if injection is None:
                    continue
                question = _apply(question, injection)
                injections.append(injection)
                remaining.remove(itype)  # 同一种缺陷不重复注入
                break

    # 所有注入器都跑完后统一校验：每个注入器声称的缺陷，在最终题目里是否仍成立。
    # 这是防注入器互相干扰的关键一步——不通过就整条丢弃重来。
    if not all(inj.verify(question) for inj in injections):
        return None

    issues = [inj.issue for inj in injections]
    conclusion, risk = derive_conclusion([i.severity for i in issues])
    suggestion = "；".join(_SUGGESTIONS[i.type] for i in issues) if issues else ""

    audit = AuditResult(
        conclusion=conclusion,
        risk_level=risk,
        issues=issues,
        suggestion=suggestion,
    )
    return Sample(question=question, audit=audit, injected=[i.type for i in issues], base_key=key)


def _make_sample(rng: random.Random, defect_rate: float) -> Sample:
    """反复尝试直到产出一条自洽的样本。"""
    for _ in range(_MAX_ATTEMPTS):
        sample = _try_make_sample(rng, defect_rate)
        if sample is not None:
            return sample
    raise RuntimeError(
        f"连续 {_MAX_ATTEMPTS} 次都没生成出自洽样本，"
        "通常是新增的注入器与已有注入器冲突，或模板槽位范围过窄"
    )


def generate_dataset(
    n: int,
    *,
    seed: int,
    defect_rate: float = 0.60,
    forbidden_keys: set[str] | None = None,
    require_unique_keys: bool = False,
) -> list[dict]:
    """生成 n 条样本，返回可直接写 jsonl 的字典列表。

    ``defect_rate`` 是「至少有一个缺陷」的比例，默认 0.60——
    正负样本大致 4:6。真实业务里通过率其实更高，但审核任务的价值恰恰在
    找出有问题的题，负样本太少会让模型退化成「一律通过」。

    ``forbidden_keys`` / ``require_unique_keys`` 用来做**底题级**的切分隔离：
    测试集必须传 ``forbidden_keys=训练集的全部 base_key``，否则同一道底题
    换个坏法就溜进测试集，指标会虚高。这道防线加在生成阶段而不是事后过滤，
    是因为事后过滤会让测试集样本数达不到预期。

    注意：模板库里有 13 个纯记忆型模板题干固定，一旦要求唯一，它们很快耗尽，
    生成器随后只能反复抽到槽位丰富的数理化模板。所以
    ``require_unique_keys`` 只适合开在测试/验证集（数量小），
    训练集开它会严重牺牲历史/地理/政治的学科覆盖。
    """
    rng = random.Random(seed)
    forbidden = set(forbidden_keys or ())
    seen: set[str] = set()
    records: list[dict] = []

    attempts = 0
    # 上限给得宽，因为固定题干模板被禁掉后会大量空转
    max_attempts = max(n * 300, 5000)
    while len(records) < n:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"生成 {n} 条样本时尝试 {attempts} 次仍未凑够。"
                "通常是 forbidden_keys 里塞了太多底题，或模板库的题干空间本身不足。"
            )

        sample = _make_sample(rng, defect_rate)
        key = sample.base_key

        if key in forbidden:
            continue
        if require_unique_keys and key in seen:
            continue

        seen.add(key)
        sample.question.id = f"q{len(records):06d}"
        records.append(
            {
                "id": sample.question.id,
                "question": json.loads(sample.question.model_dump_json()),
                "audit": json.loads(sample.audit.model_dump_json()),
                # 仅供数据质检统计；训练脚本读数据时会显式丢弃
                "injected": [t.value for t in sample.injected],
                "base_key": key,
            }
        )

    return records


def write_dataset(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:  # pragma: no cover - CLI 入口
    import argparse

    ap = argparse.ArgumentParser(description="合成命题审核指令数据")
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--train", type=int, default=2500)
    ap.add_argument("--dev", type=int, default=200)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--defect-rate", type=float, default=0.60)
    args = ap.parse_args()

    # 三个集用不同 seed，避免同一模板同一槽位组合同时出现在训练集和测试集里
    splits = {
        "train": (args.train, args.seed),
        "dev": (args.dev, args.seed + 10_000),
        "test": (args.test, args.seed + 20_000),
    }
    for name, (count, seed) in splits.items():
        records = generate_dataset(count, seed=seed, defect_rate=args.defect_rate)
        path = args.out_dir / f"{name}.jsonl"
        write_dataset(records, path)
        print(f"{path}: {len(records)} 条")


if __name__ == "__main__":  # pragma: no cover
    main()
