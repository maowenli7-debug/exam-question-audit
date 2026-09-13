"""命题审核 LLM —— 内网本地化的题目合规审核。

模块地图::

    taxonomy   审核维度定义与「缺陷 -> 结论」的标签推导规则
    schema     审核结论的权威格式与解析器（格式遵循率的判定依据）
    synth_data 合成「题目 + 审核结论」指令数据
    prompts    system / user prompt 模板（训练与推理共用一份）
    train_qlora QLoRA 微调
    evaluate   base vs 微调 的对比评测
    infer      本地 transformers 推理后端
    api        FastAPI 服务（含 OpenAI 兼容接口）
"""

__version__ = "0.1.0"
