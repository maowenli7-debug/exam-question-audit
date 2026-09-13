# 24GB 单卡正式训练手册

开发机（Apple Silicon，16GB 统一内存）上跑的是 Qwen2.5-0.5B + 纯 LoRA，目的是验证流水线。**正式配置是 Qwen2.5-7B + 4-bit QLoRA + 单卡 24GB**，那份配置在没有 NVIDIA 显卡的机器上无法真实执行。这份手册说明怎么在目标机器上跑出正式版本。

## 前置条件

| 项 | 要求 |
|---|---|
| GPU | NVIDIA，显存 ≥ 24GB（RTX 3090 / 4090 / A10 等） |
| CUDA | 12.1 及以上（`nvidia-smi` 能看到驱动） |
| Python | 3.10 ~ 3.12 |
| 磁盘 | ≥ 60GB（7B 权重 15.2GB + 优化器状态 + 检查点） |

## 一、环境准备

```bash
conda create -n audit-llm python=3.11 -y
conda activate audit-llm

# PyTorch 必须用 CUDA 版。pip 默认装的是 CPU 版，装完 torch.cuda.is_available() 会是 False
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 其余依赖
pip install -r requirements.txt

# CUDA 专属：4-bit 量化 + 高吞吐推理
pip install bitsandbytes vllm
```

装完先自检，这一步能挡掉后面 90% 的报错：

```bash
python scripts/check_env.py
```

期望看到：

```
✓ CUDA 可用：NVIDIA GeForce RTX 4090（24 GB）
    → 可运行完整的 4-bit QLoRA 与 vLLM
✓ bitsandbytes   0.4x.x
✓ vllm          0.1x.x
```

如果 `bitsandbytes` 那行是 `✗`，不要往下走——训练脚本会检测到非 CUDA 环境并**自动关闭 4-bit 量化**（这是刻意的设计，见下节），你会以为自己在跑 QLoRA，实际跑的是占显存大得多的纯 LoRA，然后 OOM。

## 二、下载 7B 模型

huggingface.co 在国内不通，走 ModelScope：

```bash
modelscope download --model Qwen/Qwen2.5-7B-Instruct \
    --local_dir models/Qwen2.5-7B-Instruct
```

约 15.2GB，视网速 10~40 分钟。想用 HF 镜像的话：

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
    --local-dir models/Qwen2.5-7B-Instruct
```

## 三、数据

数据是本项目自己合成的，不需要下载：

```bash
make data     # 生成 data/train.jsonl (2500) / dev.jsonl (200) / test.jsonl (300)
```

产物是确定性的（固定 seed），任何机器上生成的结果完全一致。

## 四、切换到 7B 配置

`configs/qlora_qwen2.5-7b.yaml` 已经写好，与 0.5B 配置的差异只有两处：

```yaml
model:
  name_or_path: models/Qwen2.5-7B-Instruct   # ← 改模型
  load_in_4bit: true                          # ← 开启 4-bit 量化
  bnb_4bit_quant_type: nf4
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: true
```

**LoRA 超参完全一致**（r=16 / alpha=32 / 全部线性层）。这是刻意的：本机 0.5B 验证的就是这套 LoRA 结构，换到 7B 上不需要重新调参，直接跑。

## 五、先冒烟再全量

这是硬性步骤，不要跳过。7B 全量训练跑几小时，配置错了会在几小时后才暴露。

```bash
# 20 条样本，2 分钟，验证：模型能加载、数据能构造、loss 能反传、adapter 能保存
python -m audit_llm.train_qlora \
    --config configs/qlora_qwen2.5-7b.yaml \
    --max-train-samples 20 \
    --output-dir outputs/smoke-7b
```

要看到 `outputs/smoke-7b/adapter_model.safetensors` 存在且非空，且 loss 不是 NaN。

**显存不够时会报什么**：`torch.cuda.OutOfMemoryError`。对策按优先级：

1. `per_device_train_batch_size` 降到 1，同时把 `gradient_accumulation_steps` 翻倍（保持等效 batch 不变）
2. `gradient_checkpointing: true`（用重算换显存，慢约 30%）
3. `max_length` 从 1024 降到 768（实测本项目样本最长 798 token，降到 768 会丢掉约 2% 的长样本，训练脚本会打印跳过条数）

## 六、全量训练

```bash
python -m audit_llm.train_qlora --config configs/qlora_qwen2.5-7b.yaml
```

24GB 单卡上的预期（QLoRA 4-bit，r=16，batch 1 × accum 16，max_length 1024）：

| 项 | 预期值 |
|---|---|
| 可训练参数 | 约 40M / 7.6B（约 0.5%） |
| 显存占用 | 18 ~ 22 GB |
| 训练速度 | 约 1.5 ~ 3 小时（2 epoch，2500 条） |
| 产物 | `outputs/qwen2.5-7b-lora/adapter_model.safetensors`，约 80 ~ 160 MB |

训练过程中盯这几件事：

```bash
# 另开一个终端看显存
watch -n 5 nvidia-smi

# 看 loss 曲线（trainer 日志里）
grep "'loss'" outputs/qwen2.5-7b-lora/trainer_state.json
```

**loss 不降 = 大概率是 PEFT + gradient checkpointing 的经典坑**：重算时输入张量的 `requires_grad` 是 False，梯度传不到 LoRA 层。脚本里已经在非量化路径调了 `enable_input_require_grads()`，但如果你手动改过训练代码，检查一下这一句还在不在。

## 七、评测

```bash
python scripts/run_eval.py \
    --base-model models/Qwen2.5-7B-Instruct \
    --adapter outputs/qwen2.5-7b-lora \
    --dtype bfloat16
```

7B 在 300 条测试集上跑两遍（基座 + 微调），4090 上约 10 分钟。

产出 `reports/eval_report.md`。数字要如实写进 README，**不要拿本机 0.5B 的结果冒充 7B**。

## 八、合并与部署

```bash
# 合并 LoRA 到基座权重，得到可独立部署的完整模型
python scripts/merge_adapter.py \
    --base models/Qwen2.5-7B-Instruct \
    --adapter outputs/qwen2.5-7b-lora \
    --out models/merged-7b \
    --dtype bfloat16

# 起服务
bash deploy/vllm_serve.sh
```

或走 Docker：

```bash
docker compose -f deploy/docker-compose.yml up -d
curl localhost:8000/healthz
```

## 九、常见问题

**`torch.cuda.is_available()` 是 False**
装成 CPU 版 torch 了。卸掉重装：`pip uninstall torch -y && pip install torch --index-url https://download.pytorch.org/whl/cu121`

**`bitsandbytes` 报找不到 CUDA 库**
版本与 CUDA 不匹配。`pip install bitsandbytes --upgrade`，或按官方文档选对应版本。

**训练脚本打印「自动关闭量化」**
说明 `device != "cuda"`。检查 GPU 是否被别的进程占满、CUDA 是否可用。**不要忽略这条警告**——它意味着你正在用 4 倍以上的显存跑训练。

**vLLM 启动报显存不足**
`--gpu-memory-utilization` 默认 0.90，训练进程没退干净时会冲突。确认 `nvidia-smi` 里没有残留进程，或把它降到 0.85。

**`--max-model-len` 设多大**
本项目样本最长 798 token，prompt + 输出合计不超过 1024。设 2048 留一倍余量，再大只是白占 KV cache 的显存。
