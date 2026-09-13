#!/usr/bin/env bash
# 等训练进程结束后自动跑评测。
#
# 为什么写成脚本而不是手动分两步：
# 训练要几小时，人工盯着不现实。而且训练和评测**都在 MPS 上跑**，
# 同时启动会互相抢显存直接 OOM——所以必须严格串行。
#
# 用法：
#   bash scripts/train_then_eval.sh [训练进程名]
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PYTHON="${PYTHON:-/opt/anaconda3/envs/llamafactory/bin/python}"
ADAPTER="outputs/qwen2.5-0.5b-lora"
PATTERN="${1:-audit_llm.train_qlora}"

echo "[chain] 等待训练进程结束（匹配：${PATTERN}）…"
while pgrep -f "${PATTERN}" > /dev/null 2>&1; do
    sleep 30
done
echo "[chain] 训练进程已退出"

# 训练可能是正常结束，也可能是崩了。用产物判断，而不是退出码——
# 后台进程的退出码在这里拿不到，而「有没有 adapter」才是我们真正关心的。
if [[ ! -f "${ADAPTER}/adapter_model.safetensors" ]]; then
    echo "[chain] ✗ 没有产出 ${ADAPTER}/adapter_model.safetensors，训练应该是失败了"
    echo "[chain]   最后 40 行训练日志："
    tail -40 outputs_train.log 2>/dev/null | tr '\r' '\n' | grep -vE '^\s*$' | tail -25
    exit 1
fi

SIZE=$(du -h "${ADAPTER}/adapter_model.safetensors" | cut -f1)
echo "[chain] ✓ 训练产物就绪（${SIZE}），开始评测"

# 评测前确认没有别的推理进程在占 MPS
sleep 10

export PYTHONPATH=src
"${PYTHON}" scripts/run_eval.py --adapter "${ADAPTER}" 2>&1 | tr '\r' '\n' | grep -vE '^\s*$'

echo "[chain] 完成"
