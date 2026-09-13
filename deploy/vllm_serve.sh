#!/usr/bin/env bash
# 在 CUDA 机器上直接启动 vLLM（不用 Docker）。
#
# 适用场景：模型和代码已经在机器上，只是想快速起个服务验证效果，
# 或者调试 vLLM 的参数（--max-model-len / --gpu-memory-utilization 这些
# 需要按实际显存反复试）。
#
# 用法：
#   bash deploy/vllm_serve.sh                    # 用默认路径
#   MODEL=models/Qwen2.5-7B-Instruct bash deploy/vllm_serve.sh
set -euo pipefail

MODEL="${MODEL:-models/Qwen2.5-0.5B-Instruct}"
SERVED_NAME="${SERVED_NAME:-question-audit}"
PORT="${PORT:-8001}"
MAX_LEN="${MAX_LEN:-2048}"
GPU_UTIL="${GPU_UTIL:-0.90}"
ADAPTER="${ADAPTER:-}"

if [[ ! -d "${MODEL}" ]]; then
    echo "✗ 找不到模型目录：${MODEL}"
    echo "  7B 模型下载："
    echo "    modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ${MODEL}"
    exit 1
fi

if ! command -v nvidia-smi > /dev/null 2>&1; then
    echo "✗ 未检测到 nvidia-smi。vLLM 需要 NVIDIA GPU。"
    echo "  本机（Apple Silicon）请改用 transformers 后端：make serve"
    exit 1
fi

echo "GPU 状态："
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo

ARGS=(
    --model "${MODEL}"
    --served-model-name "${SERVED_NAME}"
    --host 0.0.0.0
    --port "${PORT}"
    --dtype auto
    --max-model-len "${MAX_LEN}"
    --gpu-memory-utilization "${GPU_UTIL}"
)

# 可选：直接挂载 LoRA 适配器（不合并权重，多适配器热切换时用这个）
# 单适配器场景下 entrypoint.sh 的「先合并再加载」吞吐更好
if [[ -n "${ADAPTER}" && -f "${ADAPTER}/adapter_model.safetensors" ]]; then
    echo "启用 LoRA 适配器：${ADAPTER}"
    ARGS+=(--enable-lora --lora-modules "${SERVED_NAME}=${ADAPTER}")
fi

echo "启动 vLLM：${MODEL} → :${PORT}"
exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
