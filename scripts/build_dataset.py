#!/usr/bin/env python
"""生成三个数据集切分，并输出统计报告 data/STATS.md。

用法::

    python scripts/build_dataset.py --out-dir data --seed 42
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

# 允许直接 python scripts/build_dataset.py 运行（不必先 pip install -e .）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_llm.schema import audit_json_schema  # noqa: E402
from audit_llm.synth_data import generate_dataset, write_dataset  # noqa: E402
from audit_llm.taxonomy import ISSUE_SAMPLING_WEIGHT, Subject  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _show(path: Path) -> str:
    """打印用路径：在仓库内就显示相对路径，否则显示绝对路径。

    这里原先硬编码了 ``data/`` 前缀，于是 ``--out-dir /tmp/ci1`` 时打印的
    仍然是 ``data/train.jsonl``——**输出与实际写入位置不符**。CI 里正是用
    ``--out-dir /tmp/...`` 生成两份数据做比对，那个提示让人以为写错了地方
    （我就照着去查了一遍仓库文件有没有被改）。
    """
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _distribution(counter: collections.Counter, total: int) -> list[str]:
    """把 Counter 渲染成 markdown 表格行。"""
    if total == 0:
        return ["| — | 0 | 0% |", "|---|---:|---:|"]
    rows = ["| 取值 | 数量 | 占比 |", "|---|---:|---:|"]
    for key, count in counter.most_common():
        rows.append(f"| {key} | {count} | {count / total * 100:.1f}% |")
    return rows


def render_stats(splits: dict[str, list[dict]], seed: int, defect_rate: float) -> str:
    lines: list[str] = []
    lines.append("# 数据集统计\n")
    lines.append(
        "> 本文件由 `scripts/build_dataset.py` 自动生成，不要手工编辑。\n"
        f"> seed = `{seed}`，defect_rate = `{defect_rate}`。\n"
    )

    lines.append("## 切分规模\n")
    lines.append("| 切分 | 样本数 | 用途 |")
    lines.append("|---|---:|---|")
    purposes = {"train": "QLoRA 微调", "dev": "训练中监控（不参与评测结论）", "test": "base vs 微调 对比评测"}
    for name, recs in splits.items():
        lines.append(f"| {name} | {len(recs)} | {purposes.get(name, '—')} |")

    train = splits["train"]
    total = len(train)

    lines.append("\n## 训练集标签分布\n")
    lines.append("### 审核结论\n")
    lines += _distribution(collections.Counter(r["audit"]["conclusion"] for r in train), total)
    lines.append("\n### 风险等级\n")
    lines += _distribution(collections.Counter(r["audit"]["risk_level"] for r in train), total)

    lines.append("\n### 学科\n")
    lines += _distribution(collections.Counter(r["question"]["subject"] for r in train), total)

    lines.append("\n### 学段\n")
    lines += _distribution(collections.Counter(r["question"]["stage"] for r in train), total)

    lines.append("\n## 注入缺陷的分布\n")
    lines.append(
        "注意：**实际分布与设定权重不一致**，这是设计使然，不是 bug。\n\n"
        "- `超纲` 只对**初中学段**注入（高中超纲边界模糊，强行注入会造出有争议的标签）\n"
        "- `表述歧义` 要求题干里存在可删除的限定词（如「匀速」「完全」）或具体数值；\n"
        "  历史/地理/政治/生物等纯记忆型题目两者都没有，无法注入\n"
        "- `格式不规范` 几乎对所有题目都适用，因此吸收了上述两类落空的配额\n"
    )
    issue_counter: collections.Counter = collections.Counter()
    for rec in train:
        issue_counter.update(rec["injected"])
    n_issues = sum(issue_counter.values())

    wsum = sum(ISSUE_SAMPLING_WEIGHT.values())
    lines.append("| 缺陷类型 | 设定权重 | 实际占比 | 实际条数 |")
    lines.append("|---|---:|---:|---:|")
    for itype, weight in sorted(ISSUE_SAMPLING_WEIGHT.items(), key=lambda kv: -kv[1]):
        actual = issue_counter.get(itype.value, 0)
        pct = actual / n_issues * 100 if n_issues else 0.0
        lines.append(f"| {itype.value} | {weight / wsum * 100:.1f}% | {pct:.1f}% | {actual} |")

    lines.append("\n### 每题的缺陷数量\n")
    lines += _distribution(collections.Counter(len(r["injected"]) for r in train), total)

    lines.append("\n## 正负样本比例\n")
    clean = sum(1 for r in train if not r["injected"])
    lines.append(
        f"- 无缺陷（干净）样本：{clean} 条（{clean / total * 100:.1f}%）\n"
        f"- 含至少一个缺陷：{total - clean} 条（{(total - clean) / total * 100:.1f}%）\n"
    )
    lines.append(
        "\n> 真实业务里通过率通常更高。这里刻意让负样本占到约 40%，"
        "否则模型会退化成「一律通过」——而审核的价值恰恰在于找出有问题的题。\n"
    )

    lines.append("\n## 切分隔离检查（按底题）\n")
    lines.append(
        "隔离的粒度是**底题**（注入缺陷之前的干净题目），不是最终题干。\n"
        "理由：同一道底题注入不同缺陷后题干会变得各不相同，但底题还是那一道，\n"
        "若允许它跨切分出现，测试集里的题就只是训练集的「另一种坏法」，指标会虚高。\n"
    )
    train_keys = {r["base_key"] for r in train}
    dev_keys = {r["base_key"] for r in splits["dev"]}
    test_keys = {r["base_key"] for r in splits["test"]}

    lines.append("\n| 检查项 | 结果 |")
    lines.append("|---|---|")
    lines.append(f"| 训练集底题数（去重） | {len(train_keys)} / {len(train)} 条样本 |")
    lines.append(f"| 验证集底题数（去重） | {len(dev_keys)} / {len(splits['dev'])} 条样本 |")
    lines.append(f"| 测试集底题数（去重） | {len(test_keys)} / {len(splits['test'])} 条样本 |")
    lines.append(f"| 训练 ∩ 测试 | **{len(train_keys & test_keys)}** |")
    lines.append(f"| 训练 ∩ 验证 | **{len(train_keys & dev_keys)}** |")
    lines.append(f"| 验证 ∩ 测试 | **{len(dev_keys & test_keys)}** |")

    leaked = len(train_keys & test_keys) + len(train_keys & dev_keys) + len(dev_keys & test_keys)
    if leaked == 0:
        lines.append(
            "\n✅ 三个切分在底题层面完全隔离。测试指标可以视为在未见过的题目上取得。\n"
        )
    else:
        lines.append(
            f"\n⚠️ 仍存在 {leaked} 处底题重叠，评测结果必须附带说明。\n"
        )

    lines.append("\n## 已知局限\n")
    lines.append(
        "1. **题目是模板生成的，不是真实题库。** 语言复杂度远低于真实命题，"
        "模型在本数据集上的表现**不能**外推到真实题目。\n"
        f"2. **题干重复度高。** 训练集 {len(train)} 条样本只来自 "
        f"{len(train_keys)} 个底题（重复率 "
        f"{1 - len(train_keys) / len(train):.0%}）。根因是模板库里有 13 个纯记忆型"
        "模板（历史/地理/政治/生物）题干固定，却占了约六成采样量。"
        "训练集**没有**强制底题唯一——强制之后这 13 个模板会迅速耗尽，"
        "历史/地理/政治将几乎从数据集中消失，学科覆盖的损失大于收益。\n"
        "3. **审核维度由规则定义，不是真实审核员的判断。** "
        "标签的「正确性」仅限于「与注入规则自洽」，不等于符合真实业务标准。\n"
        "4. **缺陷类型分布与设定权重有偏差**（见上表），根源是适用性限制而非采样错误。\n"
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="生成命题审核合成数据集")
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--train", type=int, default=2500)
    ap.add_argument("--dev", type=int, default=200)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--defect-rate", type=float, default=0.60)
    args = ap.parse_args()

    splits: dict[str, list[dict]] = {}

    # 先训后验再测，逐级传递已用过的底题——保证低优先级切分不会撞上高优先级的。
    # 训练集不强制底题唯一（会牺牲学科覆盖，见 render_stats 的说明），
    # 验证/测试集数量小，强制唯一以换取更干净的评测。
    train = generate_dataset(args.train, seed=args.seed, defect_rate=args.defect_rate)
    splits["train"] = train
    print(f"  {_show(args.out_dir / 'train.jsonl')}  {len(train)} 条")

    used: set[str] = {r["base_key"] for r in train}

    dev = generate_dataset(
        args.dev,
        seed=args.seed + 10_000,
        defect_rate=args.defect_rate,
        forbidden_keys=used,
        require_unique_keys=True,
    )
    splits["dev"] = dev
    used |= {r["base_key"] for r in dev}
    print(f"  {_show(args.out_dir / 'dev.jsonl')}  {len(dev)} 条")

    test = generate_dataset(
        args.test,
        seed=args.seed + 20_000,
        defect_rate=args.defect_rate,
        forbidden_keys=used,
        require_unique_keys=True,
    )
    splits["test"] = test
    print(f"  {_show(args.out_dir / 'test.jsonl')}  {len(test)} 条")

    for name, records in splits.items():
        write_dataset(records, args.out_dir / f"{name}.jsonl")

    stats_path = args.out_dir / "STATS.md"
    stats_path.write_text(render_stats(splits, args.seed, args.defect_rate), encoding="utf-8")
    print(f"  {_show(stats_path)}  统计报告")

    # 抽样文件，便于人工快速检查数据长什么样
    sample_path = args.out_dir / "SAMPLE.json"
    sample_path.write_text(
        json.dumps(splits["train"][:5], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  {_show(sample_path)}  5 条抽样")

    # 导出 JSON Schema 给上下游对齐。
    # 数据是校验规则的产物，schema 变了数据也该跟着变，所以放这里一起产出，
    # 而不是单独写个脚本——避免两者不同步。
    schema_path = REPO_ROOT / "configs" / "audit_schema.json"
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(
        json.dumps(audit_json_schema(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  {_show(schema_path)}  对外 JSON Schema")


if __name__ == "__main__":
    main()
