"""训练与推理**共用**的 prompt 模板。

为什么单独抽一个模块，而不是各写各的
------------------------------------
训练时用的 prompt 和线上推理用的 prompt 只要差一个标点，模型的表现就会漂移
（train/serve skew）。这个项目里训练脚本、评测脚本、FastAPI 服务三处都要构造
prompt，所以模板必须只有一份，谁都不许就地拼字符串。

改动本文件等于改动模型的输入分布——改完必须重跑训练，否则线上效果会和评测对不上。
"""

from __future__ import annotations

from .schema import AUDIT_FIELD_VALUES, AUDIT_RESPONSE_FORMAT, Question

__all__ = ["SYSTEM_PROMPT", "render_question", "build_messages", "build_completion"]


SYSTEM_PROMPT = f"""你是一名教育命题审核专家，负责在题库入库前审核题目的合规性与质量。

## 审核维度

你需要逐项检查以下五类问题，发现问题就记入 issues：

1. **敏感表述** —— 政治敏感、地域歧视、民族宗教争议、不适宜的教育场景。此类问题一经发现即为高风险。
2. **学科错误** —— 参考答案与解析相互矛盾、选项内容重复、知识点表述错误。
3. **表述歧义** —— 题干存在多种理解、缺少解题的必要条件、指代不明。
4. **超纲** —— 涉及的知识点超出该学段课程标准要求。
5. **格式不规范** —— 选项标点不统一、物理量缺单位、排版错乱。

## 判定规则

- 题目没有任何问题：`issues` 为空列表，`conclusion` 为「通过」，`risk_level` 为「低」。
- 有问题时，`risk_level` 取所有 issues 中**最高**的 severity：
  - 最高为「高」 → `conclusion` 为「不通过」
  - 最高为「中」 → `conclusion` 为「需修改」
  - 最高为「低」 → `conclusion` 为「通过」（轻微问题不阻塞入库）

## 输出要求

只输出一个 JSON 对象，不要输出任何解释性文字，不要用 markdown 代码块包裹。

输出结构如下，字段名必须完全一致，不得增删字段。下面是**空骨架**，
每个字段的值都要由你根据实际审核结果填写：

{AUDIT_RESPONSE_FORMAT}

各字段的取值范围：

{AUDIT_FIELD_VALUES}
"""


def render_question(question: Question) -> str:
    """把一道题渲染成 user 消息。

    字段顺序固定为 学科/学段 -> 题干 -> 选项 -> 答案 -> 解析，训练与推理共用，
    保证两边看到的文本逐字节一致。
    """
    options = "\n".join(f"{key}. {text}" for key, text in sorted(question.options.items()))
    return (
        f"【学科】{question.subject}\n"
        f"【学段】{question.stage}\n"
        f"【题干】{question.stem}\n"
        f"【选项】\n{options}\n"
        f"【参考答案】{question.answer}\n"
        f"【解析】{question.explanation}\n\n"
        f"请审核以上题目，按要求输出 JSON。"
    )


def build_messages(question: Question) -> list[dict[str, str]]:
    """构造 chat 格式的消息列表，供 tokenizer.apply_chat_template 使用。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_question(question)},
    ]


def build_completion(result_json: str) -> list[dict[str, str]]:
    """构造 assistant 回复。

    参数是**已经序列化好的 JSON 字符串**而不是对象，因为我们要对键顺序和
    分隔符有完全控制：合成标签时用 ``AuditResult.model_dump_json()`` 输出紧凑
    JSON（无空格），模型学到的就是这种最省 token 的写法，推理时也更不容易截断。
    """
    return [{"role": "assistant", "content": result_json}]
