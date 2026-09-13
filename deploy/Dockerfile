# ============================================================================
# 生产镜像：vLLM + FastAPI，仅支持 NVIDIA GPU。
#
# 为什么基于 vllm/vllm-openai 而不是 python:3.x
# --------------------------------------------
# vLLM 需要与 CUDA、PyTorch、flash-attn 精确匹配的编译版本，自己在通用基础镜像
# 上装一遍，很大概率卡在编译或运行动态库不匹配上。官方镜像已经把这些对齐好了。
# 代价是镜像大（约 16GB），但这是一次性的分发成本。
#
# 为什么有两个进程（vLLM + FastAPI）而不是一个
# -------------------------------------------
# vLLM 自身就提供 OpenAI 兼容接口，但我们的业务接口 /v1/audit 需要额外做
# 「渲染 prompt → 调模型 → 校验成 AuditResult」这三步。把它塞进 vLLM 需要写
# 自定义 logits processor，耦合太深。分成两层后：
#   - vLLM 层只管高吞吐推理，不关心审核业务
#   - FastAPI 层只管业务编排，不关心推理优化
# 换模型、换量化方式只动 vLLM 层，改审核规则只动 FastAPI 层。
#
# 构建：
#   docker build -f deploy/Dockerfile -t exam-audit:latest .
# 运行：
#   docker run --gpus all -p 8000:8000 -v $PWD/models:/app/models \
#     -v $PWD/outputs:/app/outputs exam-audit:latest
# ============================================================================

FROM vllm/vllm-openai:latest

WORKDIR /app

# 只装 FastAPI 层需要的依赖。torch/transformers/vllm 基础镜像里已经有了，
# 重复安装会覆盖掉镜像里那几个"编译好的正确版本"。
RUN pip install --no-cache-dir \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.30" \
    "pydantic>=2.7" \
    "httpx>=0.27" \
    "pyyaml>=6.0"

# 先拷源码再拷数据：数据/模型是挂载进来的，不进镜像层，改代码时不会触发重新拷贝数据
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY scripts/ /app/scripts/
COPY deploy/ /app/deploy/

ENV PYTHONPATH=/app/src \
    AUDIT_BACKEND=vllm \
    VLLM_BASE_URL=http://localhost:8001/v1 \
    BASE_MODEL=/app/models/Qwen2.5-0.5B-Instruct \
    ADAPTER=/app/outputs/qwen2.5-0.5b-lora \
    HF_HUB_OFFLINE=1

# vLLM 对外不暴露 8001，只在本容器内被 FastAPI 调用；
# 外部只能访问 8000，审核接口的鉴权和限流将来加在这一层。
EXPOSE 8000

# 用 shell 形式启动，因为要先拉起 vLLM 再等它就绪
CMD ["bash", "/app/deploy/entrypoint.sh"]
