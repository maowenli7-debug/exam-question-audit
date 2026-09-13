"""FastAPI 推理服务。

两个接口，面向两类调用方
------------------------
- ``POST /v1/audit``           业务接口。传入一道题，直接返回结构化审核结论。
                               调用方拿到的一定是合法 ``AuditResult``，不用自己解析。
- ``POST /v1/chat/completions`` OpenAI 兼容接口。给**已有系统**用——
                               原流程是调外部网页 AI，接口形态是 chat 那一套，
                               改成指向内网这个地址即可，业务代码基本不用动。

这正是「数据不出内网」的落地方式：把 base_url 从外部服务换成内网地址。

两种后端
--------
``AUDIT_BACKEND=transformers``（默认）
    在本进程内加载模型。能在 Mac 上跑，适合开发与演示。
``AUDIT_BACKEND=vllm``
    转发到独立的 vLLM OpenAI 服务。生产环境用这个——transformers 后端没有
    连续批处理，并发能力差得远。启动方式见 deploy/。

启动::

    make serve
    # 或
    AUDIT_BACKEND=vllm VLLM_BASE_URL=http://localhost:8001/v1 \
      uvicorn audit_llm.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .infer import AuditModel, resolve_device
from .prompts import SYSTEM_PROMPT, build_messages
from .schema import AuditResult, Question, parse_audit_output

# --------------------------------------------------------------------------
# 配置（全部走环境变量，便于容器化）
# --------------------------------------------------------------------------

BACKEND = os.getenv("AUDIT_BACKEND", "transformers").lower()
BASE_MODEL = os.getenv("BASE_MODEL", "models/Qwen2.5-0.5B-Instruct")
ADAPTER = os.getenv("ADAPTER", "outputs/qwen2.5-0.5b-lora")
DTYPE = os.getenv("DTYPE", "float32")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
VLLM_MODEL_NAME = os.getenv("VLLM_MODEL_NAME", "question-audit")

# 进程内模型句柄。vLLM 后端下为 None。
_model: AuditModel | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时按后端类型初始化。

    transformers 后端加载模型要几秒到几十秒，放在 lifespan 里而不是请求里，
    否则第一个请求会超时。
    """
    global _model

    if BACKEND == "transformers":
        print(f"[startup] 加载 transformers 后端：{BASE_MODEL} + {ADAPTER}")
        _model = AuditModel(BASE_MODEL, ADAPTER or None, device=None, dtype=DTYPE,
                            max_new_tokens=MAX_NEW_TOKENS)
        print(f"[startup] 就绪，设备 {_model.device}")
    elif BACKEND == "vllm":
        print(f"[startup] vLLM 后端，转发到 {VLLM_BASE_URL}")
        async with httpx.AsyncClient() as client:
            try:
                resp = await client.get(f"{VLLM_BASE_URL}/models", timeout=10.0)
                resp.raise_for_status()
                print(f"[startup] vLLM 就绪：{resp.json()}")
            except Exception as exc:  # noqa: BLE001
                # 不阻止启动：vLLM 可能比本服务晚几秒起来，健康检查会反映真实状态
                print(f"[startup] 警告：连不上 vLLM（{exc}），服务仍会启动")
    else:
        raise RuntimeError(f"未知的 AUDIT_BACKEND: {BACKEND}（可选 transformers / vllm）")

    yield
    _model = None


app = FastAPI(
    title="命题审核 LLM 服务",
    description="内网本地化的题目合规审核。数据全程不出内网，不依赖任何外部网页 AI。",
    version="0.1.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# 请求 / 响应模型
# --------------------------------------------------------------------------

class AuditRequest(BaseModel):
    question: Question
    max_new_tokens: int | None = Field(default=None, ge=16, le=1024)


class AuditResponse(BaseModel):
    result: AuditResult
    raw_output: str = Field(description="模型原始输出，便于排查格式问题")
    needs_extraction: bool = Field(
        description="True 表示原始输出带代码块或解释性文字，是靠容错提取才拿到 JSON 的。"
        "这个字段为 True 时格式合规性其实是打折扣的，调用方可据此决定是否告警。"
    )
    latency_ms: int


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "question-audit"
    messages: list[ChatMessage]
    temperature: float = 0.0
    max_tokens: int | None = Field(default=None, ge=16, le=2048)


# --------------------------------------------------------------------------
# 推理分发
# --------------------------------------------------------------------------

def _audit_with_transformers(question: Question, max_new_tokens: int | None) -> tuple[str, AuditResult, bool]:
    if _model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")
    if max_new_tokens:
        _model.max_new_tokens = max_new_tokens
    gen = _model.audit(question, greedy=True)
    if gen.parsed is None:
        # 格式不合规时也要给调用方一个可用的响应，而不是 500：
        # 审核系统需要知道「这条没过格式校验」，而不是整条请求失败。
        raise HTTPException(
            status_code=422,
            detail={
                "message": "模型输出无法解析为结构化审核结论",
                "failure_reason": gen.failure_reason,
                "failure_detail": gen.failure_detail,
                "raw_output": gen.raw,
            },
        )
    return gen.raw, gen.parsed, gen.extracted


async def _audit_with_vllm(question: Question, max_new_tokens: int | None) -> tuple[str, AuditResult, bool]:
    payload = {
        "model": VLLM_MODEL_NAME,
        "messages": build_messages(question),
        "temperature": 0.0,
        "max_tokens": max_new_tokens or MAX_NEW_TOKENS,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{VLLM_BASE_URL}/chat/completions", json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"vLLM 返回 {resp.status_code}: {resp.text[:300]}")
        body = resp.json()

    raw = body["choices"][0]["message"]["content"]
    outcome = parse_audit_output(raw)
    if outcome.result is None:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "模型输出无法解析为结构化审核结论",
                "failure_reason": outcome.reason,
                "failure_detail": outcome.error,
                "raw_output": raw,
            },
        )
    return raw, outcome.result, outcome.extracted


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------

@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """健康检查。K8s / Docker healthcheck 用这个。"""
    info: dict[str, Any] = {"status": "ok", "backend": BACKEND, "device": resolve_device()}

    if BACKEND == "transformers":
        info["model_loaded"] = _model is not None
        if _model is None:
            info["status"] = "loading"
    else:
        info["vllm_base_url"] = VLLM_BASE_URL
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{VLLM_BASE_URL}/models")
                resp.raise_for_status()
            info["vllm_reachable"] = True
        except Exception as exc:  # noqa: BLE001
            info["status"] = "degraded"
            info["vllm_reachable"] = False
            info["vllm_error"] = str(exc)[:200]

    return info


@app.post("/v1/audit", response_model=AuditResponse)
async def audit(req: AuditRequest) -> AuditResponse:
    """审核一道题，返回结构化结论。

    这是审核系统应该调用的接口——返回体已经过 ``AuditResult`` 校验，
    调用方拿到就能入库，不需要自己解析模型输出。
    """
    started = time.perf_counter()

    if BACKEND == "transformers":
        raw, result, extracted = _audit_with_transformers(req.question, req.max_new_tokens)
    else:
        raw, result, extracted = await _audit_with_vllm(req.question, req.max_new_tokens)

    return AuditResponse(
        result=result,
        raw_output=raw,
        needs_extraction=extracted,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest) -> dict[str, Any]:
    """OpenAI 兼容接口。

    给已有审核系统用的：把 base_url 从外部服务改成这个内网地址，
    业务代码几乎不用动，数据也就不再出内网了。
    """
    messages = [m.model_dump() for m in req.messages]

    if BACKEND == "vllm":
        payload = {
            "model": VLLM_MODEL_NAME,
            "messages": messages,
            "temperature": req.temperature,
            "max_tokens": req.max_tokens or MAX_NEW_TOKENS,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{VLLM_BASE_URL}/chat/completions", json=payload)
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"vLLM 返回 {resp.status_code}")
            return resp.json()

    if _model is None:
        raise HTTPException(status_code=503, detail="模型尚未加载完成")

    import torch

    prompt = _model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _model.tokenizer(prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        out = _model.model.generate(
            **inputs,
            max_new_tokens=req.max_tokens or MAX_NEW_TOKENS,
            do_sample=req.temperature > 0,
            temperature=req.temperature if req.temperature > 0 else None,
            pad_token_id=_model.tokenizer.pad_token_id or _model.tokenizer.eos_token_id,
        )
    text = _model.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    # 返回体按 OpenAI 的格式拼，调用方才不用改解析代码
    return {
        "id": "chatcmpl-local",
        "object": "chat.completion",
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
    }


@app.get("/v1/schema")
async def get_schema() -> dict[str, Any]:
    """返回审核结论的 JSON Schema，供调用方生成客户端代码或做本地校验。"""
    return AuditResult.model_json_schema()


@app.get("/v1/prompt")
async def get_prompt() -> dict[str, str]:
    """返回 system prompt。

    暴露出来是为了让调用方知道模型到底被要求做什么——审核标准变了要改
    prompt 并重训，这个接口让「当前线上用的是哪版标准」变得可查。
    """
    return {"system_prompt": SYSTEM_PROMPT}
