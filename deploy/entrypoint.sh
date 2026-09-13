#!/usr/bin/env bash
# 容器入口：先起 vLLM，等它就绪，再起 FastAPI。
#
# 为什么要等：FastAPI 第一个请求会转发给 vLLM，如果 vLLM 还没加载完模型，
# 请求会 502。虽然 api.py 的 /healthz 能反映真实状态，但让启动顺序正确
# 比让调用方去处理竞态要好。
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8001}"
API_PORT="${API_PORT:-8000}"
MODEL_PATH="${BASE_MODEL:-/app/models/Qwen2.5-0.5B-Instruct}"
ADAPTER_PATH="${ADAPTER:-/app/outputs/qwen2.5-0.5b-lora}"
SERVED_NAME="${VLLM_MODEL_NAME:-question-audit}"

echo "=========================================================="
echo " 命题审核 LLM 服务启动"
echo "   基座模型 : ${MODEL_PATH}"
echo "   LoRA     : ${ADAPTER_PATH}"
echo "   服务名   : ${SERVED_NAME}"
echo "=========================================================="

# --- 合并 LoRA 适配器 ------------------------------------------------------
# vLLM 可以直接加载 LoRA（--enable-lora），但把适配器合并进基座权重后，
# 推理时少一层 LoRA 计算，吞吐更高。合并是一次性的，放在启动阶段做。
# 合并失败（比如适配器还没训练出来）不阻塞启动——退回纯基座模型，
# 但会打印醒目警告，避免"以为在用微调模型、其实用的是基座"这种沉默的错。
if [[ -d "${ADAPTER_PATH}" && -f "${ADAPTER_PATH}/adapter_model.safetensors" ]]; then
    echo "[1/3] 合并 LoRA 适配器到基座模型…"
    if python /app/scripts/merge_adapter.py \
        --base "${MODEL_PATH}" \
        --adapter "${ADAPTER_PATH}" \
        --out /app/models/merged; then
        MODEL_PATH=/app/models/merged
        echo "      合并完成 → ${MODEL_PATH}"
    else
        echo "      !! 合并失败，退回使用未微调的基座模型 !!"
    fi
else
    echo "[1/3] 未找到 LoRA 适配器（${ADAPTER_PATH}），使用基座模型"
    echo "      !! 当前服务返回的是未经微调的基座模型输出 !!"
fi

# --- 启动 vLLM ------------------------------------------------------------
# 只监听 127.0.0.1：vLLM 不对外暴露，所有外部请求都走 FastAPI 那一层
echo "[2/3] 启动 vLLM（内部端口 ${VLLM_PORT}）…"
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_NAME}" \
    --host 127.0.0.1 \
    --port "${VLLM_PORT}" \
    --dtype auto \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.90 \
    > /tmp/vllm.log 2>&1 &
VLLM_PID=$!

# vLLM 加载模型要几十秒，轮询 /models 直到它响应
echo "      等待 vLLM 就绪…"
for i in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
        echo "      vLLM 就绪（用时 ${i}s）"
        break
    fi
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "      !! vLLM 进程已退出，日志尾部："
        tail -30 /tmp/vllm.log
        exit 1
    fi
    sleep 1
done

if ! curl -sf "http://127.0.0.1:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
    echo "      !! 等待 vLLM 超时（120s），日志尾部："
    tail -30 /tmp/vllm.log
    exit 1
fi

# --- 启动 FastAPI ---------------------------------------------------------
echo "[3/3] 启动 FastAPI（对外端口 ${API_PORT}）…"
exec uvicorn audit_llm.api:app --host 0.0.0.0 --port "${API_PORT}" --workers 1
