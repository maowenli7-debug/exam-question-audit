"""题目模板库 —— 合成数据的「题干 + 选项 + 答案 + 解析」来源。

关于这批数据的诚实说明
----------------------
这些题目是**模板 + 槽位随机**生成的，不是真实题库。
它们的用途是让模型学会「审核这件事的输出格式与判定逻辑」，
**不能**用来评估模型在真实题目上的审核能力——真实题目的语言复杂度远高于此。
README 里对此有明确标注，不要把它当成 EduData 或任何真实题库的替代品。

设计要点
--------
1. **正确选项不写死位置**。模板只给「正确选项文本」和三个干扰项文本，位置由
   生成器洗牌决定。否则模型会学到「答案总是 A」这种与审核无关的伪规律——
   审核要判的是「答案与解析是否一致」，答案位置必须随机。

2. **槽位必须自洽**。``make_slots`` 一次算出所有数值（如面积 = 长 × 宽），
   保证题干与选项里的数字对得上，不会出现「题干说长 5 宽 3、选项面积写 20」。
   ``validate_templates()`` 会检查每个模板引用的槽位都能被它的 slot 函数提供，
   把 ``{x_plsu}`` 这种拼写错误在测试阶段就拦下来，而不是等生成时 KeyError。

3. **干扰项要「像错的」而不是「随便一个数」**。干扰项刻意取自常见错解
   （周长当面积、漏除系数、除以而非乘以），这样审核模型面对的是有区分度的
   选项，而不是一眼可辨的荒谬值。
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from typing import Any, Callable

from .taxonomy import Stage, Subject

__all__ = [
    "Template",
    "TEMPLATES",
    "templates_for",
    "render_options_block",
    "validate_templates",
    "distinct_or_none",
]


@dataclass(frozen=True)
class Template:
    subject: Subject
    stage: Stage
    make_slots: Callable[[random.Random], dict[str, Any]]
    stem: str
    correct: str
    distractors: tuple[str, str, str]
    explanation: str

    @property
    def format_strings(self) -> tuple[str, ...]:
        """本模板所有需要填充的文本，供 validate_templates 自检。"""
        return (self.stem, self.correct, *self.distractors, self.explanation)

    def build(self, rng: random.Random) -> dict[str, str]:
        """填充槽位，返回题干/四个选项文本/解析。

        选项**不在这里定序**——洗牌交给 synth_data.py 统一负责，
        模板作者不必操心答案位置。
        """
        slots = self.make_slots(rng)
        return {
            "stem": self.stem.format(**slots),
            "correct": self.correct.format(**slots),
            "wrong": [d.format(**slots) for d in self.distractors],
            "explanation": self.explanation.format(**slots),
        }


def _no_slots(_rng: random.Random) -> dict[str, Any]:
    """给纯记忆型题目（历史/地理/政治）用：无需随机数值。"""
    return {}


# ==========================================================================
# 数学
# ==========================================================================

def _slots_linear_eq(rng: random.Random) -> dict[str, Any]:
    a = rng.randint(2, 9)
    x = rng.randint(2, 12)
    b = rng.randint(1, 20)
    return {
        "a": a, "b": b, "x": x, "c": a * x + b,
        "x_plus": x + 1,          # 常见错解：移项后忘了变号
        "x_minus": x - 1,
        "a_times_x": a * x,       # 常见错解：移项后没有再除以系数
    }


def _slots_rect(rng: random.Random) -> dict[str, Any]:
    w = rng.randint(3, 15)
    h = rng.randint(2, 12)
    if w == h:
        h += 1  # 避开 w=h=4 时周长与面积数值相等导致的选项撞车
    return {
        "w": w, "h": h,
        "area": w * h,
        "peri": 2 * (w + h),      # 常见错解：把周长当面积
        "area_plus_w": w * h + w,
        "w_plus_h": w + h,
    }


def _slots_quadratic_disc(rng: random.Random) -> dict[str, Any]:
    b = rng.randint(2, 9)
    c = rng.randint(1, 8)
    disc = b * b - 4 * c
    return {
        "b": b, "c": c,
        "b_sq": b * b,
        "four_c": 4 * c,
        "disc": disc,                          # 正确答案：b² − 4ac
        "disc_alt1": b * b + 4 * c,            # 常见错解：符号弄反
        "disc_alt2": b * b - 4 * c + 1,
        "disc_neg": -disc if disc != 0 else disc - 2,
    }


def _slots_arith_seq(rng: random.Random) -> dict[str, Any]:
    a1 = rng.randint(1, 9)
    d = rng.randint(2, 6)
    n = rng.randint(5, 15)
    an = a1 + (n - 1) * d
    return {
        "a1": a1, "d": d, "n": n,
        "subscript_n": "ₙ",
        "an": an,
        "an_off1": an + d,        # 常见错解：项数多算了一项
        "an_off2": an - d,        # 常见错解：项数少算了一项
        "a1_plus_d": a1 + d,      # 常见错解：把 a₂ 当成了 aₙ
    }


MATH_TEMPLATES: tuple[Template, ...] = (
    Template(
        subject=Subject.MATH, stage=Stage.JUNIOR,
        make_slots=_slots_linear_eq,
        stem="解方程 {a}x + {b} = {c}，则 x 的值为（　　）",
        correct="{x}",
        distractors=("{x_plus}", "{x_minus}", "{a_times_x}"),
        explanation="移项得 {a}x = {c} − {b} = {a_times_x}，两边同除以 {a}，得 x = {x}。",
    ),
    Template(
        subject=Subject.MATH, stage=Stage.JUNIOR,
        make_slots=_slots_rect,
        stem="一个长方形的长为 {w} cm，宽为 {h} cm，则它的面积是（　　）",
        correct="{area} cm²",
        distractors=("{peri} cm²", "{area_plus_w} cm²", "{w_plus_h} cm²"),
        explanation="长方形面积 = 长 × 宽 = {w} × {h} = {area}（cm²）。注意 {peri} 是周长而不是面积。",
    ),
    Template(
        subject=Subject.MATH, stage=Stage.JUNIOR,
        make_slots=_slots_quadratic_disc,
        stem="关于 x 的一元二次方程 x² − {b}x + {c} = 0 的根的判别式 b² − 4ac 的值是（　　）",
        correct="{disc}",
        distractors=("{disc_alt1}", "{disc_alt2}", "{disc_neg}"),
        explanation="其中 a = 1，b = −{b}，c = {c}，故判别式为 (−{b})² − 4×1×{c} = {b_sq} − {four_c} = {disc}。",
    ),
    Template(
        subject=Subject.MATH, stage=Stage.SENIOR,
        make_slots=_slots_arith_seq,
        stem="已知等差数列 {{aₙ}} 的首项 a₁ = {a1}，公差 d = {d}，则 a{subscript_n} 的值为（　　）",
        correct="{an}",
        distractors=("{an_off1}", "{an_off2}", "{a1_plus_d}"),
        explanation="由等差数列通项公式 aₙ = a₁ + (n − 1)d，得 a{subscript_n} = {a1} + ({n} − 1) × {d} = {an}。",
    ),
)


# ==========================================================================
# 物理
# ==========================================================================

def _slots_speed(rng: random.Random) -> dict[str, Any]:
    v = rng.randint(2, 20)
    t = rng.randint(2, 12)
    return {
        "v": v, "t": t,
        "s": v * t,               # 正确答案：s = vt
        "v_plus_t": v + t,        # 常见错解：把速度与时间相加
        "v_over_t": v // t if t and v >= t else v,   # 常见错解：除反了
        "s_alt": v * t + v,
    }


def _slots_density(rng: random.Random) -> dict[str, Any]:
    v = rng.randint(4, 20)
    rho = rng.randint(1, 12)
    m = rho * v  # 反推质量，保证密度是整数，选项里不出现长小数
    return {
        "m": m, "v": v,
        "rho": rho,               # 正确答案：ρ = m / V
        "rho_alt1": rho + 1,
        "rho_alt2": rho - 1 if rho > 1 else rho + 2,
        "m_times_v": m * v,       # 常见错解：该除却乘
    }


def _slots_ohm(rng: random.Random) -> dict[str, Any]:
    i = rng.randint(1, 6)
    r = rng.randint(3, 20)
    u = i * r
    return {
        "i": i, "r": r,
        "u": u,
        "u_times_i": u * i,       # 常见错解：R = UI
        "r_alt": r + i,
        "i_over_u": i,            # 常见错解：把电流当电阻
    }


PHYSICS_TEMPLATES: tuple[Template, ...] = (
    Template(
        subject=Subject.PHYSICS, stage=Stage.JUNIOR,
        make_slots=_slots_speed,
        stem="一辆汽车以 {v} m/s 的速度匀速行驶了 {t} s，则它通过的路程是（　　）",
        correct="{s} m",
        distractors=("{v_plus_t} m", "{v_over_t} m", "{s_alt} m"),
        explanation="由 s = vt，得 s = {v} m/s × {t} s = {s} m。",
    ),
    Template(
        subject=Subject.PHYSICS, stage=Stage.JUNIOR,
        make_slots=_slots_density,
        stem="某物体的质量为 {m} g，体积为 {v} cm³，则它的密度是（　　）",
        correct="{rho} g/cm³",
        distractors=("{m_times_v} g/cm³", "{rho_alt1} g/cm³", "{rho_alt2} g/cm³"),
        explanation="由 ρ = m / V，得 ρ = {m} g ÷ {v} cm³ = {rho} g/cm³。",
    ),
    Template(
        subject=Subject.PHYSICS, stage=Stage.SENIOR,
        make_slots=_slots_ohm,
        stem="一段导体两端的电压为 {u} V 时，通过它的电流为 {i} A，则该导体的电阻为（　　）",
        correct="{r} Ω",
        distractors=("{u_times_i} Ω", "{r_alt} Ω", "{i_over_u} Ω"),
        explanation="由欧姆定律 I = U / R 变形得 R = U / I = {u} V ÷ {i} A = {r} Ω。",
    ),
)


# ==========================================================================
# 化学
# ==========================================================================

def _slots_mass_fraction(rng: random.Random) -> dict[str, Any]:
    solute = rng.randint(5, 40)
    solvent = rng.randint(60, 200)
    total = solute + solvent
    pct = round(solute / total * 100, 1)
    return {
        "solute": solute, "solvent": solvent, "total": total,
        "pct": pct,
        "pct_alt1": round(solute / solvent * 100, 1),   # 常见错解：分母漏了溶质
        "pct_alt2": round(pct + 10, 1),
        "pct_alt3": round(max(pct - 10, 0.1), 1),
    }


def _slots_mole(rng: random.Random) -> dict[str, Any]:
    # 取常见物质的摩尔质量，让题目看起来像真的
    mm = rng.choice([18, 32, 40, 44, 58.5, 98])
    n = rng.randint(1, 8)
    return {
        "mm": mm, "n": n,
        "mass": n * mm,           # 正确答案：m = n·M
        "mass_alt": mm + n,       # 常见错解：把相加当相乘
    }


CHEMISTRY_TEMPLATES: tuple[Template, ...] = (
    Template(
        subject=Subject.CHEMISTRY, stage=Stage.JUNIOR,
        make_slots=_slots_mass_fraction,
        stem="将 {solute} g 氯化钠完全溶解在 {solvent} g 水中，所得溶液中溶质的质量分数约为（　　）",
        correct="{pct}%",
        distractors=("{pct_alt1}%", "{pct_alt2}%", "{pct_alt3}%"),
        explanation="溶液总质量 = {solute} g + {solvent} g = {total} g，溶质质量分数 = {solute} ÷ {total} × 100% ≈ {pct}%。",
    ),
    Template(
        subject=Subject.CHEMISTRY, stage=Stage.SENIOR,
        make_slots=_slots_mole,
        stem="已知某物质的摩尔质量为 {mm} g/mol，则 {n} mol 该物质的质量是（　　）",
        correct="{mass} g",
        distractors=("{mm} g", "{n} g", "{mass_alt} g"),
        explanation="由 m = n·M，得 m = {n} mol × {mm} g/mol = {mass} g。",
    ),
)


# ==========================================================================
# 生物
# ==========================================================================

BIOLOGY_TEMPLATES: tuple[Template, ...] = (
    Template(
        subject=Subject.BIOLOGY, stage=Stage.JUNIOR,
        make_slots=_no_slots,
        stem="植物细胞中进行光合作用的场所是（　　）",
        correct="叶绿体",
        distractors=("线粒体", "细胞核", "液泡"),
        explanation="叶绿体含有叶绿素，是光合作用的场所；线粒体是呼吸作用的主要场所。",
    ),
    Template(
        subject=Subject.BIOLOGY, stage=Stage.JUNIOR,
        make_slots=_no_slots,
        stem="人体消化和吸收营养物质的主要器官是（　　）",
        correct="小肠",
        distractors=("胃", "大肠", "食道"),
        explanation="小肠内表面有皱襞和小肠绒毛，大大增加了消化吸收的面积，是消化吸收的主要器官。",
    ),
    Template(
        subject=Subject.BIOLOGY, stage=Stage.SENIOR,
        make_slots=_no_slots,
        stem="两对相对性状的纯合亲本杂交，F₁ 自交所得 F₂ 中，若两对基因独立遗传，则其表现型之比为（　　）",
        correct="9:3:3:1",
        distractors=("3:1", "1:1", "1:2:1"),
        explanation="两对基因独立遗传时，F₂ 表现型比例为 (3:1) × (3:1) = 9:3:3:1。",
    ),
    Template(
        subject=Subject.BIOLOGY, stage=Stage.SENIOR,
        make_slots=_no_slots,
        stem="在生态系统中，能把无机物合成有机物的成分是（　　）",
        correct="生产者",
        distractors=("消费者", "分解者", "非生物的物质和能量"),
        explanation="生产者（主要是绿色植物）通过光合作用把无机物合成有机物，是生态系统的基石。",
    ),
)


# ==========================================================================
# 历史
# ==========================================================================

HISTORY_TEMPLATES: tuple[Template, ...] = (
    Template(
        subject=Subject.HISTORY, stage=Stage.JUNIOR,
        make_slots=_no_slots,
        stem="中国近代史上，标志着中国开始沦为半殖民地半封建社会的事件是（　　）",
        correct="鸦片战争",
        distractors=("甲午中日战争", "八国联军侵华战争", "辛亥革命"),
        explanation="1840 年鸦片战争后签订《南京条约》，中国开始沦为半殖民地半封建社会。",
    ),
    Template(
        subject=Subject.HISTORY, stage=Stage.JUNIOR,
        make_slots=_no_slots,
        stem="我国进入社会主义初级阶段的标志是（　　）",
        correct="三大改造的基本完成",
        distractors=("中华人民共和国成立", "第一部宪法颁布", "第一个五年计划完成"),
        explanation="1956 年底三大改造基本完成，实现了生产资料私有制向社会主义公有制的转变。",
    ),
    Template(
        subject=Subject.HISTORY, stage=Stage.SENIOR,
        make_slots=_no_slots,
        stem="第一次工业革命中，使人类进入「蒸汽时代」的关键发明是（　　）",
        correct="改良蒸汽机",
        distractors=("珍妮纺纱机", "蒸汽机车", "水力织布机"),
        explanation="瓦特改良的蒸汽机提供了稳定动力，使人类进入「蒸汽时代」，是工业革命的核心标志。",
    ),
)


# ==========================================================================
# 地理
# ==========================================================================

GEOGRAPHY_TEMPLATES: tuple[Template, ...] = (
    Template(
        subject=Subject.GEOGRAPHY, stage=Stage.JUNIOR,
        make_slots=_no_slots,
        stem="我国地势的总特征是（　　）",
        correct="西高东低，呈三级阶梯状分布",
        distractors=("东高西低，呈三级阶梯状分布", "南高北低，逐级下降", "四周高、中间低"),
        explanation="我国地势西高东低，呈三级阶梯状分布，使许多大河自西向东流入海洋。",
    ),
    Template(
        subject=Subject.GEOGRAPHY, stage=Stage.JUNIOR,
        make_slots=_no_slots,
        stem="我国面积最大的省级行政区是（　　）",
        correct="新疆维吾尔自治区",
        distractors=("西藏自治区", "内蒙古自治区", "青海省"),
        explanation="新疆维吾尔自治区面积约 166 万平方千米，是我国面积最大的省级行政区。",
    ),
    Template(
        subject=Subject.GEOGRAPHY, stage=Stage.SENIOR,
        make_slots=_no_slots,
        stem="受地转偏向力影响，北半球河流的侵蚀作用主要表现为（　　）",
        correct="右岸侵蚀",
        distractors=("左岸侵蚀", "两岸均匀侵蚀", "只在下游侵蚀"),
        explanation="地转偏向力使北半球运动的物体向右偏转，故河流右岸侵蚀作用更强。",
    ),
)


# ==========================================================================
# 政治
# ==========================================================================

POLITICS_TEMPLATES: tuple[Template, ...] = (
    Template(
        subject=Subject.POLITICS, stage=Stage.JUNIOR,
        make_slots=_no_slots,
        stem="在我国，公民行使国家权力的机关是（　　）",
        correct="人民代表大会",
        distractors=("人民政府", "人民法院", "人民政协"),
        explanation="我国宪法规定，人民行使国家权力的机关是全国人民代表大会和地方各级人民代表大会。",
    ),
    Template(
        subject=Subject.POLITICS, stage=Stage.JUNIOR,
        make_slots=_no_slots,
        stem="法律区别于道德等行为规范的最主要特征是（　　）",
        correct="由国家强制力保证实施",
        distractors=("由国家制定或认可", "对全体社会成员具有普遍约束力", "靠社会舆论维持"),
        explanation="法律靠国家强制力保证实施，这是法律区别于道德等行为规范的最主要特征。",
    ),
    Template(
        subject=Subject.POLITICS, stage=Stage.SENIOR,
        make_slots=_no_slots,
        stem="「实践是检验真理的唯一标准」这一论断体现的哲学道理是（　　）",
        correct="实践是认识的基础",
        distractors=("认识决定实践", "真理是主观的", "意识具有直接现实性"),
        explanation="实践是认识的来源、动力、检验标准和目的，检验真理的唯一标准只能是实践。",
    ),
)


# ==========================================================================
# 汇总与自检
# ==========================================================================

TEMPLATES: tuple[Template, ...] = (
    MATH_TEMPLATES
    + PHYSICS_TEMPLATES
    + CHEMISTRY_TEMPLATES
    + BIOLOGY_TEMPLATES
    + HISTORY_TEMPLATES
    + GEOGRAPHY_TEMPLATES
    + POLITICS_TEMPLATES
)


def templates_for(subject: Subject, stage: Stage) -> tuple[Template, ...]:
    """取出指定 学科 × 学段 的模板。

    历史/地理/政治等学科某一学段可能没有模板，此时回退到该学科的全部模板——
    宁可学段略有出入，也不要让某个学科在数据集里整个缺席。
    """
    exact = tuple(t for t in TEMPLATES if t.subject == subject and t.stage == stage)
    if exact:
        return exact
    fallback = tuple(t for t in TEMPLATES if t.subject == subject)
    if not fallback:
        raise ValueError(f"学科 {subject} 没有任何模板")
    return fallback


def _placeholders(text: str) -> set[str]:
    """取出 format 串里的字段名。

    用 string.Formatter 而不是正则，因为它能正确处理 ``{{aₙ}}`` 这类
    转义花括号——正则会把转义后的内容误判成槽位。
    """
    return {
        name
        for _, name, _, _ in string.Formatter().parse(text)
        if name is not None
    }


def validate_templates() -> list[str]:
    """自检：每个模板引用的槽位，其 slot 函数必须都能提供。

    返回问题描述列表，空列表表示全部通过。测试里会断言它为空。
    没有这层检查，``{x_plsu}`` 这类拼写错误要等到合成数据时才炸出来，
    而那时数据可能已经写了一半。
    """
    problems: list[str] = []
    for idx, tpl in enumerate(TEMPLATES):
        provided = set(tpl.make_slots(random.Random(0)))
        for text in tpl.format_strings:
            missing = _placeholders(text) - provided
            if missing:
                problems.append(
                    f"模板 #{idx}（{tpl.subject}/{tpl.stage}）引用了槽位 "
                    f"{sorted(missing)}，但 make_slots 未提供"
                )
    return problems


def render_options_block(options: dict[str, str]) -> str:
    """把选项字典渲染成文本块，供写进 jsonl 或人工抽样检查。"""
    return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))


def distinct_or_none(values: list[str] | tuple[str, ...]) -> bool:
    """校验四个选项互不相同。

    生成器靠它决定要不要重新采样槽位——数学模板里 w×h 和 2(w+h) 在 w=h=4 时
    会撞车。这种样本必须丢掉，否则「选项重复」会被模型当成我们**注入**的缺陷，
    污染标签（模型学到的和我们要教的就对不上了）。
    """
    return len(set(values)) == len(values)
