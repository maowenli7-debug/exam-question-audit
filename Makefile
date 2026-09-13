# ============================================================================
# 命题审核 LLM —— 常用命令
#
# PYTHON 指向本项目验证过的 conda 环境。换环境时改这一行即可，或
#   make train PYTHON=/path/to/python
# ============================================================================
PYTHON ?= /opt/anaconda3/envs/llamafactory/bin/python
export PYTHONPATH := src

# 默认用本机可跑的 0.5B 配置；正式训练换成 qlora_qwen2.5-7b.yaml
CONFIG ?= configs/qlora_qwen2.5-0.5b.yaml

.DEFAULT_GOAL := help

.PHONY: help
help:  ## 显示所有可用命令
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# 环境
# ---------------------------------------------------------------------------

.PHONY: check-env
check-env:  ## 检查依赖与设备（不装任何东西）
	$(PYTHON) scripts/check_env.py

.PHONY: install
install:  ## 安装基础依赖（国内走清华镜像）
	$(PYTHON) -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------

.PHONY: data
data:  ## 合成 2500/200/300 条数据，并生成 data/STATS.md
	$(PYTHON) scripts/build_dataset.py

# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

.PHONY: smoke
smoke:  ## 只用 20 条样本跑通训练链路（几分钟，用于验证代码）
	$(PYTHON) -m audit_llm.train_qlora --config $(CONFIG) \
		--max-train-samples 20 --output-dir outputs/smoke

.PHONY: train
train:  ## 完整训练（本机默认 0.5B；正式训练请覆盖 CONFIG）
	$(PYTHON) -m audit_llm.train_qlora --config $(CONFIG)

# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------

.PHONY: eval
eval:  ## 在 300 条测试集上对比 base 与微调模型
	$(PYTHON) scripts/run_eval.py

# ---------------------------------------------------------------------------
# 服务
# ---------------------------------------------------------------------------

.PHONY: serve
serve:  ## 启动 FastAPI 服务（本机用 transformers 后端）
	$(PYTHON) -m uvicorn audit_llm.api:app --host 0.0.0.0 --port 8000

.PHONY: serve-vllm
serve-vllm:  ## 启动生产后端（需要 NVIDIA GPU）
	bash deploy/vllm_serve.sh

# ---------------------------------------------------------------------------
# 交付
# ---------------------------------------------------------------------------

.PHONY: merge
merge:  ## 合并 LoRA 到基座权重，得到可独立部署的模型
	$(PYTHON) scripts/merge_adapter.py \
		--base models/Qwen2.5-0.5B-Instruct \
		--adapter outputs/qwen2.5-0.5b-lora \
		--out models/merged

.PHONY: docker
docker:  ## 构建生产镜像（需要 NVIDIA GPU 才能运行）
	docker build -f deploy/Dockerfile -t exam-audit:latest .

# ---------------------------------------------------------------------------
# 质量
# ---------------------------------------------------------------------------

.PHONY: test
test:  ## 跑测试
	$(PYTHON) -m pytest

.PHONY: clean
clean:  ## 清掉生成物（保留 data/ 与已训练的 adapter）
	rm -rf outputs/smoke .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: clean-all
clean-all: clean  ## 连数据和训练产物一起清掉
	rm -rf data/*.jsonl outputs/
