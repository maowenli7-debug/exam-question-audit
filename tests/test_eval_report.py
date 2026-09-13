"""评测报告的生成测试。

为什么要测「报告」而不是只测「指标计算」
------------------------------------
``scripts/run_eval.py`` 里有一段根据实际数字写结论的逻辑（``_conclusion_text``），
它在三种情况下走三条不同分支：提升明显、提升有限/持平、**反而下降**。

第三条分支最重要却最难触发——它要等真实评测跑出「微调后更差」才会执行。
如果不专门测，这段代码很可能直到线上真出问题才第一次运行。而它恰恰是
最该正常工作的一段：格式遵循率下降时，报告必须**明确报警**并给出排查方向，
而不是含糊带过或（更糟）把下降说成提升。

所以这里用构造的 EvalResult 把这三种情况都跑一遍，断言措辞与数字一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_eval import _conclusion_text, _delta, _fmt_pct, render_report  # noqa: E402

from audit_llm.evaluate import EvalResult  # noqa: E402


def _make(name: str, n: int, format_ok: int, *, bare: int | None = None, **kw) -> EvalResult:
    r = EvalResult(name=name, n=n, format_ok=format_ok)
    r.bare_json_ok = format_ok if bare is None else bare
    for key, value in kw.items():
        setattr(r, key, value)
    return r


META = {
    "base_model": "models/Qwen2.5-0.5B-Instruct",
    "adapter": "outputs/qwen2.5-0.5b-lora",
    "test_file": "data/test.jsonl",
    "device": "mps",
}


# ==========================================================================
# 告警分支
# ==========================================================================

def test_regression_is_reported_as_regression():
    """微调后变差时必须报警——这是最该正确的一条分支。"""
    base = _make("基座模型", 300, 120)
    ft = _make("微调模型", 300, 60)
    text = _conclusion_text(base, ft)

    assert "下降" in text, "格式遵循率下降却没有在结论里指出"
    assert "⚠️" in text, "下降时没有醒目的告警标记"
    # 不能把下降说成提升
    assert "提升" not in text.replace("提升的", ""), "把下降说成了提升"


def test_improvement_is_reported_as_improvement():
    base = _make("基座模型", 300, 60)
    ft = _make("微调模型", 300, 270)
    text = _conclusion_text(base, ft)
    assert "提升" in text
    assert "⚠️" not in text
    assert "70.0" in text, "没有写具体提升幅度"


def test_flat_result_does_not_claim_improvement():
    """两者一样时，报告不能说「提升」，而应指向「训练可能没生效」。"""
    base = _make("基座模型", 300, 150)
    ft = _make("微调模型", 300, 150)
    text = _conclusion_text(base, ft)
    assert "相同" in text
    assert "adapter" in text or "训练" in text, "持平时应提示排查训练是否生效"


def test_marginal_gain_is_not_oversold():
    """提升 1 个百分点不能说成「确实学会了」。"""
    base = _make("基座模型", 300, 150)
    ft = _make("微调模型", 300, 153)
    text = _conclusion_text(base, ft)
    assert "有限" in text or "幅度" in text
    assert "说明微调确实让模型学会了按固定结构输出" not in text


def test_saturated_format_rate_defers_to_bare_json():
    """主指标饱和时，结论必须以裸 JSON 率为准。

    这是本项目的**真实情形**：基座格式遵循率 99.0%、微调 100.0%，看着毫无差别。
    但基座那 99% 里 266 条是靠容错提取救回来的，真正自己输出干净 JSON 的只有 10.3%。

    不特判这种情况的话，报告会走进「幅度有限……或基座模型本身已能较好遵循指令
    格式」那一支——那句话与同一张表里的「266 条需容错提取」直接矛盾，
    而且是**把唯一的真实结论说反了**：基座恰恰是不会输出干净格式的那个。
    """
    base = _make("基座模型", 300, 297, bare=31)  # 266 条靠提取
    ft = _make("微调模型", 300, 300, bare=300)
    text = _conclusion_text(base, ft)

    assert "266" in text, "没有点出有多少条是靠容错提取救回来的"
    assert "裸 JSON" in text, "没有指向真正体现差异的那个指标"
    assert "基座模型本身已能较好遵循指令格式" not in text, (
        "把「基座靠后处理才达标」说成了「基座本来就会」——结论说反了"
    )
    assert "⚠️" in text, "这种情况该有醒目提示，否则只看主指标会以为微调没效果"


def test_saturation_branch_does_not_hijack_real_format_gain():
    """主指标真的大幅提升时，不能被「裸 JSON」分支抢走。

    新分支的条件是 ``gain <= 0.05 and bare_gap > 0.2``。这里验证前者确实在起作用：
    格式遵循率涨了 70 个百分点时，结论必须走正常的提升分支。
    """
    base = _make("基座模型", 300, 60, bare=20)
    ft = _make("微调模型", 300, 270, bare=260)
    text = _conclusion_text(base, ft)

    assert "提升到" in text, "真实的格式提升被裸 JSON 分支盖掉了"
    assert "已饱和" not in text


def test_rescued_count_matches_between_table_and_conclusion():
    """结论段与表格里的「提取条数」必须是同一个数。

    两处口径若不一致（比如一处用 format_ok - bare_json_ok、另一处用
    n - bare_json_ok），报告会同时出现两个不同的数字，读者无从判断哪个是真的。
    这正是本次修复前真实存在的 bug。"""
    base = _make("基座模型", 300, 297, bare=31)
    ft = _make("微调模型", 300, 300, bare=300)
    md = render_report(base, ft, META)
    text = _conclusion_text(base, ft)

    assert "266 条" in md, "表格没有显示「需容错提取才达标」的条数"
    assert "266 条" in text, "结论段的条数与表格不一致"


# ==========================================================================
# 报告渲染
# ==========================================================================

def test_report_renders_all_metric_rows():
    base = _make("基座模型", 300, 60, bare=40, conclusion_correct=30, risk_correct=35,
                 issue_tp=10, issue_fp=5, issue_fn=20)
    ft = _make("微调模型", 300, 270, bare=250, conclusion_correct=200, risk_correct=210,
               issue_tp=80, issue_fp=10, issue_fn=15)
    md = render_report(base, ft, META)

    for label in ("格式遵循率", "审核结论准确率", "风险等级准确率", "问题类型 F1", "结论"):
        assert label in md, f"报告缺少「{label}」"

    # 两个模型是两行不同数字，不能是同一份结果被打印了两遍
    assert "**20.0%**" in md  # 基座 60/300
    assert "**90.0%**" in md  # 微调 270/300


def test_report_notes_that_metrics_are_from_synthetic_data():
    """报告必须自带「数据是合成的、指标不可外推」的声明。

    这条看着像形式主义，但它防的是一个具体的误读：报告的表格长得和真实
    评测报告一模一样，很容易被单独截图传播，脱离上下文就成了「这个模型
    在真实题目上有 90% 的格式遵循率」。
    """
    base = _make("基座模型", 300, 60)
    ft = _make("微调模型", 300, 270)
    md = render_report(base, ft, META)
    assert "合成" in md
    assert "不能" in md or "无法" in md


def _pred(gold_types: list[str], pred_types: list[str]) -> dict:
    """构造一条逐条预测记录（只保留算分型召回用得上的字段）。"""
    return {
        "gold": {"issues": [{"type": t} for t in gold_types]},
        "parsed": {"issues": [{"type": t} for t in pred_types]},
    }


def test_issue_type_table_reports_per_type_recall():
    """分型召回表必须按类型分别给出，而不是只给一个总 F1。

    总 F1 把「某类几乎查不出」平均掉了——本项目实测 F1=0.830 看着还行，
    拆开才发现「学科错误」召回只有 53.8%，而它恰好是出现最多的一类。
    """
    base = _make("基座模型", 10, 10)
    base.predictions = [_pred(["学科错误"], []) for _ in range(10)]

    ft = _make("微调模型", 10, 10)
    ft.predictions = (
        [_pred(["学科错误"], ["学科错误"]) for _ in range(2)]   # 2/10
        + [_pred(["学科错误"], []) for _ in range(8)]
        + [_pred(["超纲"], ["超纲"]) for _ in range(4)]         # 4/4
    )

    md = render_report(base, ft, META)
    assert "分问题类型的召回率" in md
    assert "| 学科错误 | 10 | 0.0% | 20.0% |" in md
    assert "| 超纲 | 4 | 0.0% | 100.0% |" in md
    # 最常见且最弱的一类必须被点名
    assert "学科错误" in md.split("最弱的一类")[-1]
    assert "100%" in md.split("占了全部漏检的")[-1][:8]


def test_issue_type_table_flags_only_genuinely_weak_type():
    """四类都查得出来时不该硬点一个「最弱」出来。"""
    ft = _make("微调模型", 10, 10)
    ft.predictions = [_pred(["超纲"], ["超纲"]) for _ in range(10)]
    base = _make("基座模型", 10, 10)
    base.predictions = [_pred(["超纲"], []) for _ in range(10)]

    md = render_report(base, ft, META)
    assert "分问题类型的召回率" in md
    assert "最弱的一类" not in md, "召回 100% 却还报「最弱的一类」"


def test_zero_base_recall_is_explained_not_left_as_bare_zero():
    """基座全 0 召回时必须给出解释。

    一列全是 0.0% 很容易被读成「指标算错了」或「模型很差」。真实情况更微妙：
    基座在 300 条里一个问题都没报出来，它不是查不准，是根本没在查。
    不解释的话，读者会对整张表的可信度打折。
    """
    base = _make("基座模型", 10, 10)
    base.predictions = [_pred(["超纲"], []) for _ in range(10)]
    ft = _make("微调模型", 10, 10)
    ft.predictions = [_pred(["超纲"], ["超纲"]) for _ in range(10)]

    md = render_report(base, ft, META)
    assert "一个问题都没报出来" in md


def test_zero_base_note_absent_when_base_detects_something():
    """基座有检出时不该加那句解释——否则是另一个方向的误导。"""
    base = _make("基座模型", 10, 10)
    base.predictions = [_pred(["超纲"], ["超纲"]) for _ in range(10)]
    ft = _make("微调模型", 10, 10)
    ft.predictions = [_pred(["超纲"], ["超纲"]) for _ in range(10)]

    md = render_report(base, ft, META)
    assert "一个问题都没报出来" not in md


def test_issue_type_table_omitted_without_predictions():
    """没有逐条数据时不渲染该节——不能出一张空表或让报告崩掉。"""
    base = _make("基座模型", 300, 60)
    ft = _make("微调模型", 300, 270)
    md = render_report(base, ft, META)
    assert "分问题类型的召回率" not in md


def test_failure_modes_are_rendered_in_chinese():
    """失败原因要翻成中文——报告是给人看的，不是给程序读的。"""
    base = _make("基座模型", 300, 60)
    base.failure_modes["no_json"] = 200
    base.failure_modes["schema"] = 40
    ft = _make("微调模型", 300, 270)
    ft.failure_modes["schema"] = 30

    md = render_report(base, ft, META)
    assert "输出里找不到 JSON" in md
    assert "不符合 schema" in md
    assert "no_json" not in md, "英文码泄漏到了给人看的报告里"


# ==========================================================================
# 小工具
# ==========================================================================

@pytest.mark.parametrize(
    "before,after,arrow",
    [(0.5, 0.9, "↑"), (0.9, 0.5, "↓"), (0.5, 0.5, "→")],
)
def test_delta_arrow_matches_direction(before: float, after: float, arrow: str):
    assert arrow in _delta(before, after)


def test_fmt_pct_uses_one_decimal():
    assert _fmt_pct(0.9166) == "91.7%"
    assert _fmt_pct(0.0) == "0.0%"


def test_bare_json_rate_can_be_lower_than_format_rate():
    """「靠容错提取才达标」的情况必须能被表达出来（两个指标分离）。"""
    r = _make("微调模型", 100, format_ok=90, bare=30)
    assert r.format_rate == 0.9
    assert r.bare_json_rate == 0.3
