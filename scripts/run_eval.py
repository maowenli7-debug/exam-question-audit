#!/usr/bin/env python
"""在测试集上对比「基座模型」与「微调后模型」，输出评测报告。

用法::

    python scripts/run_eval.py                          # 用默认配置
    python scripts/run_eval.py --limit 50               # 只跑前 50 条，快速验证
    python scripts/run_eval.py --adapter outputs/xxx    # 指定某个 adapter

产出：
    reports/eval_base_vs_ft.json   机器可读的指标
    reports/eval_report.md         给人看的对比报告
    reports/predictions/*.jsonl    逐条原始输出，供人工复盘
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from audit_llm.evaluate import EvalResult, eval_result_to_dict, evaluate_model  # noqa: E402

# 注意：这里**不能**在模块顶层 ``from audit_llm.infer import AuditModel``。
# infer 会 ``import torch``，而本模块的报告渲染逻辑（render_report /
# _conclusion_text）被 tests/test_eval_report.py 导入来测，CI 里没有 torch。
# 顶层导入会让那个测试文件在 collection 阶段就挂掉——本地因为装着 torch 而看不见。
# 所以把 import 推迟到 main() 里，只在实际要跑推理时才拉进来。
# （evaluate.py 里 AuditModel 出现在类型标注上，已用 TYPE_CHECKING 同样处理。）

# 格式失败原因的英文码 -> 中文说明，写进报告便于阅读
FAILURE_LABELS = {
    "empty": "输出为空",
    "no_json": "输出里找不到 JSON",
    "bad_json": "JSON 语法错误",
    "schema": "JSON 合法但不符合 schema",
    "unknown": "未归类",
}


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _delta(before: float, after: float) -> str:
    d = (after - before) * 100
    arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
    return f"{arrow} {abs(d):.1f} pp"


def _issue_type_breakdown(r: EvalResult) -> tuple[Counter, Counter]:
    """从逐条预测里按问题类型统计（真值数, 命中数）。

    只看总 F1 会掩盖一个要命的事实：**漏检是集中在某一类上的**。
    本项目实测 F1=0.830 看着还行，拆开才发现「学科错误」召回只有 53.8%，
    而它恰好是出现最多的一类（78 条）；其余四类都在 84% 以上。
    总 F1 把这两种完全不同的表现平均掉了，看报告的人会以为「大致都能查出来」。

    ``r.predictions`` 为空时返回空 Counter——测试里构造的 EvalResult 没有逐条数据，
    这时报告不渲染这一节，而不是渲染一张空表。
    """
    gold: Counter = Counter()
    hit: Counter = Counter()
    for pred in r.predictions:
        g = {i["type"] for i in (pred.get("gold") or {}).get("issues", [])}
        p = {i["type"] for i in (pred.get("parsed") or {}).get("issues", [])}
        gold.update(g)
        hit.update(g & p)
    return gold, hit


def _issue_type_table(base: EvalResult, ft: EvalResult) -> str:
    """分问题类型的召回对比表。按真值条数降序——最常见的类型排最前。"""
    gold_b, hit_b = _issue_type_breakdown(base)
    gold_f, hit_f = _issue_type_breakdown(ft)
    if not gold_f and not gold_b:
        return ""

    L: list[str] = ["\n## 分问题类型的召回率\n"]
    L.append("> 分母是该类型的**真值条数**（测试集固定，两个模型共用）。")
    L.append("> 总 F1 会把「某类几乎查不出」平均掉，这一节专门把它拆出来。\n")
    L.append("| 问题类型 | 真值条数 | 基座召回 | 微调召回 |")
    L.append("|---|---:|---:|---:|")
    for itype, total in gold_f.most_common():
        b = hit_b.get(itype, 0) / total
        f = hit_f.get(itype, 0) / total
        L.append(f"| {itype} | {total} | {_fmt_pct(b)} | {_fmt_pct(f)} |")

    # 基座那一列全是 0.0% 时，必须说明原因，否则读者会以为指标算错了。
    # 真实情况是：基座在 300 条里**一个问题都没报出来**（issues 恒为空数组），
    # 它不是「查得不准」，而是根本没在查。这跟「召回率低」是两回事。
    if not sum(hit_b.values()):
        L.append(
            "\n注意基座那一列的 0.0%：这不是统计口径问题，而是基座模型在全部 "
            f"{base.n} 条里**一个问题都没报出来**（`issues` 恒为空数组），"
            "且结论一律回答「通过」。它学会了输出的「形状」，但没学会「审」。"
            "所谓 54.7% 的结论准确率，是**蒙的**——测试集里判「通过」的样本"
            "恰好占 54.7%，一律答「通过」就能拿到这个数。"
        )

    # 点出最弱的一类。报告是给人看的，「哪类查不出来」比「平均多少」更能指导下一步。
    worst = min(gold_f, key=lambda t: hit_f.get(t, 0) / gold_f[t])
    worst_rate = hit_f.get(worst, 0) / gold_f[worst]
    base_worst = hit_b.get(worst, 0) / gold_f[worst]
    if worst_rate < 0.8:
        L.append(
            f"\n最弱的一类是 **{worst}**（微调后召回 {_fmt_pct(worst_rate)}，"
            f"基座 {_fmt_pct(base_worst)}）。这一类的漏检占了全部漏检的 "
            f"{_gold_share(gold_f, hit_f, worst):.0%}，是对下游影响最大的短板。"
        )
    return "\n".join(L) + "\n"


def _gold_share(gold: Counter, hit: Counter, itype: str) -> float:
    """某一类的漏检数占全部漏检数的比例。"""
    missed = sum(gold.values()) - sum(hit.values())
    return (gold[itype] - hit.get(itype, 0)) / missed if missed else 0.0


def render_report(base: EvalResult, ft: EvalResult, meta: dict) -> str:
    """渲染 markdown 对比报告。"""
    L: list[str] = []
    L.append("# 评测报告：基座模型 vs 微调模型\n")
    L.append(
        f"- 基座模型：`{meta['base_model']}`\n"
        f"- LoRA 适配器：`{meta['adapter']}`\n"
        f"- 测试集：`{meta['test_file']}`（{base.n} 条）\n"
        f"- 推理设备：`{meta['device']}`　解码：贪心（结果可复现）\n"
    )

    L.append("\n## 主指标：结构化输出格式遵循率\n")
    L.append("> 定义：模型输出能被解析成合法 `AuditResult` 的样本占比。")
    L.append("> 只看结构是否合法，与内容对错无关。\n")
    L.append("| 模型 | 格式遵循率 | 其中「裸 JSON」 | 需容错提取才达标 |")
    L.append("|---|---:|---:|---:|")
    for r in (base, ft):
        extracted = r.format_ok - r.bare_json_ok
        L.append(
            f"| {r.name} | **{_fmt_pct(r.format_rate)}** "
            f"| {_fmt_pct(r.bare_json_rate)} | {extracted} 条 |"
        )
    L.append(
        f"\n**变化：{_delta(base.format_rate, ft.format_rate)}**\n"
    )
    L.append(
        "「裸 JSON」这一列值得单独看：如果一个模型全靠剥代码块、截花括号才达标，"
        "说明它并没有真正学会输出规范格式，只是被后处理救了回来。\n"
    )

    L.append("\n## 字段准确率\n")
    L.append(
        "> 分母是**全量**样本，格式失败的样本一律记为错——下游系统拿不到可用的"
        "结构化结果。若只在格式合法的子集上算，会出现「格式越差、字段准确率越高」"
        "的假象。\n"
    )
    L.append("| 指标 | 基座模型 | 微调模型 | 变化 |")
    L.append("|---|---:|---:|---|")
    rows = [
        ("审核结论准确率", base.conclusion_acc, ft.conclusion_acc),
        ("风险等级准确率", base.risk_acc, ft.risk_acc),
        ("问题类型 精确率", base.issue_precision, ft.issue_precision),
        ("问题类型 召回率", base.issue_recall, ft.issue_recall),
        ("问题类型 F1", base.issue_f1, ft.issue_f1),
    ]
    for label, b, f in rows:
        L.append(f"| {label} | {_fmt_pct(b)} | {_fmt_pct(f)} | {_delta(b, f)} |")
    L.append(
        f"\n问题类型按多标签统计（TP={base.issue_tp}/{ft.issue_tp}, "
        f"FP={base.issue_fp}/{ft.issue_fp}, FN={base.issue_fn}/{ft.issue_fn}）。\n"
    )

    type_table = _issue_type_table(base, ft)
    if type_table:
        L.append(type_table)

    L.append("\n## 格式失败原因分布\n")
    modes = sorted(set(base.failure_modes) | set(ft.failure_modes))
    if modes:
        L.append("| 失败原因 | 基座模型 | 微调模型 |")
        L.append("|---|---:|---:|")
        for mode in modes:
            label = FAILURE_LABELS.get(mode, mode)
            L.append(f"| {label} | {base.failure_modes.get(mode, 0)} | {ft.failure_modes.get(mode, 0)} |")
    else:
        L.append("两个模型都没有格式失败样本。\n")

    L.append("\n## 结论\n")
    L.append(_conclusion_text(base, ft))

    L.append("\n---\n")
    L.append(
        "> 本报告由 `scripts/run_eval.py` 自动生成。\n"
        "> 数据集为模板合成的**合成数据**，题目语言复杂度远低于真实命题，\n"
        "> 上述指标**不能**外推到真实题目上的审核能力。详见 `data/STATS.md` 的「已知局限」。\n"
    )
    return "\n".join(L) + "\n"


def _conclusion_text(base: EvalResult, ft: EvalResult) -> str:
    """根据实际数字写结论，不预设「一定提升」。

    两个指标必须一起看——这是本函数踩过的坑。

    主指标「格式遵循率」带三级容错提取兜底：只要输出文本里**能剥出**一个合法
    JSON 就算达标。而基座模型最爱把 JSON 包在 ```json 代码块里，剥一下就达标了。
    于是主指标在基座身上虚高、趋近饱和，看着像「基座已经会了」。

    本项目的实测正是这个陷阱：格式遵循率 99.0% → 100.0%，几乎没差别；但同期
    裸 JSON 率是 10.3% → 100.0%，基座 300 条里 266 条是靠提取救回来的。
    **真实增益完全被主指标盖住了。**

    所以当主指标增益很小、而裸 JSON 率差距很大时，结论必须以裸 JSON 为准，
    绝不能写「基座本身已能较好遵循指令格式」——那句话与同一张表里的
    「266 条需容错提取」直接矛盾。
    """
    lines: list[str] = []
    gain = ft.format_rate - base.format_rate
    bare_gap = ft.bare_json_rate - base.bare_json_rate
    # 口径必须与报告表格的「需容错提取才达标」一致（format_ok - bare_json_ok），
    # 否则结论段和表格会给出两个不同的数字，读者无从判断哪个是真的。
    rescued = base.format_ok - base.bare_json_ok

    if gain < -0.001:
        lines.append(
            f"⚠️ 微调后格式遵循率反而下降（{_fmt_pct(base.format_rate)} → "
            f"{_fmt_pct(ft.format_rate)}）。这是需要排查的信号，常见原因是"
            "训练数据里有格式不一致的样本，或训练轮数过多导致过拟合。"
        )
    elif gain <= 0.05 and bare_gap > 0.2:
        # 主指标饱和、真实差异藏在裸 JSON 率里 —— 不能按「提升有限」草草收尾
        lines.append(
            f"⚠️ **主指标已饱和，但那是假象，不要据此认为微调没效果。**\n\n"
            f"格式遵循率 {_fmt_pct(base.format_rate)} → {_fmt_pct(ft.format_rate)}"
            f"（{gain * 100:+.1f} pp），看似没差别。但这个指标带容错提取兜底："
            f"基座 {base.n} 条里有 **{rescued} 条**是靠剥代码块、截花括号才达标的，"
            f"真正自己就输出干净 JSON 的只有 {_fmt_pct(base.bare_json_rate)}；"
            f"微调后裸 JSON 率是 {_fmt_pct(ft.bare_json_rate)}。\n\n"
            f"真实结论：**微调把「靠后处理救回来」变成了「本来就合法」"
            f"（裸 JSON 率 {bare_gap * 100:+.1f} pp）。** 这个增益比主指标更有实际意义——"
            f"生产环境不该指望下游系统必须做容错提取，那本身就是不稳定的来源。"
        )
    elif gain > 0.05:
        lines.append(
            f"微调后格式遵循率从 {_fmt_pct(base.format_rate)} 提升到 "
            f"{_fmt_pct(ft.format_rate)}（+{gain * 100:.1f} 个百分点），"
            "说明微调确实让模型学会了按固定结构输出。"
        )
    elif gain > 0:
        lines.append(
            f"格式遵循率提升 {gain * 100:.1f} 个百分点，幅度有限。"
            "可能是训练轮数不足、数据量偏小，或基座模型本身已能较好遵循指令格式。"
        )
    else:
        lines.append(
            f"两者格式遵循率相同（{_fmt_pct(base.format_rate)}）。"
            "需要检查：训练是否真的生效、adapter 是否被正确加载。"
        )

    if base.conclusion_acc or ft.conclusion_acc:
        lines.append(
            f"\n内容层面，审核结论准确率从 {_fmt_pct(base.conclusion_acc)} 变为 "
            f"{_fmt_pct(ft.conclusion_acc)}，问题类型 F1 从 {base.issue_f1:.3f} 变为 "
            f"{ft.issue_f1:.3f}。"
        )
        lines.append(
            "\n注意：格式遵循率与内容准确率是两件事。格式达标只说明输出结构可用，"
            "不代表审核判断正确——后者需要真实标注数据才能可靠评估，"
            "而本项目的标签是合成规则生成的。"
        )
    return "\n".join(lines)


def main() -> None:
    from audit_llm.infer import AuditModel  # 见文件顶部的说明：推迟到此处导入

    ap = argparse.ArgumentParser(description="对比评测基座模型与微调模型")
    ap.add_argument("--base-model", default="models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--adapter", default="outputs/qwen2.5-0.5b-lora")
    ap.add_argument("--test-file", default="data/test.jsonl")
    ap.add_argument("--reports-dir", type=Path, default=Path("reports"))
    ap.add_argument("--limit", type=int, default=None, help="只评测前 N 条（快速验证用）")
    ap.add_argument("--skip-base", action="store_true", help="跳过基座模型，只评微调模型")
    ap.add_argument("--dtype", default="float32", help="MPS 上用 float32 最稳")
    args = ap.parse_args()

    adapter_path = Path(args.adapter)
    if not (adapter_path / "adapter_model.safetensors").exists():
        print(f"✗ 找不到 LoRA 适配器：{adapter_path}/adapter_model.safetensors")
        print("  请先运行 make train")
        sys.exit(1)

    test_path = Path(args.test_file)
    records = [json.loads(line) for line in test_path.open(encoding="utf-8")]
    if args.limit:
        records = records[: args.limit]
    print(f"测试集：{test_path}（{len(records)} 条）\n")

    results: list[EvalResult] = []

    if not args.skip_base:
        print(f"加载基座模型 {args.base_model} …")
        base_model = AuditModel(args.base_model, device=None, dtype=args.dtype)
        print(f"  设备 {base_model.device}\n评测中…")
        results.append(evaluate_model(base_model, records, "基座模型"))
        del base_model

    print(f"\n加载微调模型（{args.base_model} + {adapter_path}）…")
    ft_model = AuditModel(args.base_model, adapter_path, device=None, dtype=args.dtype)
    print(f"  设备 {ft_model.device}\n评测中…")
    results.append(evaluate_model(ft_model, records, "微调模型"))

    # --- 落盘 ---
    if len(results) == 2:
        base, ft = results
    else:
        ft = results[0]
        # 只评了微调模型时，用一份空的基座结果占位，报告仍能生成
        base = EvalResult(name="基座模型", n=ft.n)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "base_model": args.base_model,
        "adapter": str(adapter_path),
        "test_file": str(test_path),
        "device": ft_model.device,
        "n_samples": len(records),
        "decoding": "greedy",
    }

    json_path = args.reports_dir / "eval_base_vs_ft.json"
    json_path.write_text(
        json.dumps(
            {"meta": meta, "base": eval_result_to_dict(base), "finetuned": eval_result_to_dict(ft)},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    md_path = args.reports_dir / "eval_report.md"
    md_path.write_text(render_report(base, ft, meta), encoding="utf-8")

    pred_dir = args.reports_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    for r in results:
        safe = "base" if r.name == "基座模型" else "finetuned"
        with (pred_dir / f"{safe}.jsonl").open("w", encoding="utf-8") as fh:
            for row in r.predictions:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 58}")
    print(f"格式遵循率  {base.name}: {_fmt_pct(base.format_rate)}  →  {ft.name}: {_fmt_pct(ft.format_rate)}")
    print(f"结论准确率  {base.name}: {_fmt_pct(base.conclusion_acc)}  →  {ft.name}: {_fmt_pct(ft.conclusion_acc)}")
    print(f"{'=' * 58}")
    print(f"\n报告：{md_path}")
    print(f"指标：{json_path}")
    print(f"逐条：{pred_dir}/")


if __name__ == "__main__":
    main()
