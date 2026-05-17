"""Completions API — LLM补全接口，兼容OpenAI API格式"""

import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator

import litellm
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Chat Completions"])


# ─── 请求模型 ─────────────────────────────────────────────────────────


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    stop: str | list[str] | None = None
    n: int | None = Field(default=None, ge=1, le=128)
    user: str | None = None
    seed: int | None = None


# ─── 响应模型（用于文档生成，实际返回litellm的dict） ───────────────────


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# ─── 错误处理工具 ─────────────────────────────────────────────────────


def _handle_litellm_error(exc: Exception) -> HTTPException:
    """将litellm异常转换为HTTPException"""
    exc_type = type(exc).__name__
    exc_str = str(exc)

    logger.warning("LiteLLM调用异常 [%s]: %s", exc_type, exc_str)

    status_code = 500
    if "AuthenticationError" in exc_type or "Invalid API Key" in exc_str:
        status_code = 401
    elif "BadRequestError" in exc_type:
        status_code = 400
    elif "NotFoundError" in exc_type or "model" in exc_str.lower():
        status_code = 404
    elif "RateLimitError" in exc_type:
        status_code = 429
    elif "Timeout" in exc_type:
        status_code = 504
    elif "ServiceUnavailableError" in exc_type:
        status_code = 503

    return HTTPException(status_code=status_code, detail=f"{exc_type}: {exc_str}")


# ─── 流式响应生成器 ───────────────────────────────────────────────────


async def _stream_response(
    model: str,
    messages: list[dict[str, Any]],
    litellm_kwargs: dict[str, Any],
    litellm_model: str | None = None,
) -> AsyncGenerator[str, None]:
    """生成SSE流式响应"""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    try:
        response = await litellm.acompletion(
            model=litellm_model or model,
            messages=messages,
            stream=True,
            **litellm_kwargs,
        )

        async for chunk in response:
            chunk_dict = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
            # 统一id和created
            chunk_dict["id"] = completion_id
            chunk_dict["created"] = created
            chunk_dict["object"] = "chat.completion.chunk"

            yield f"data: {json.dumps(chunk_dict, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.exception("流式生成过程中发生异常")
        error_payload = {
            "error": {
                "message": str(exc),
                "type": type(exc).__name__,
            }
        }
        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"


# ─── 端点 ─────────────────────────────────────────────────────────────


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest, req: Request):
    """OpenAI兼容的Chat Completions接口

    决策流程：
    1. 固化规则拦截（最高优先级，零成本）
    2. 动态路由（model="auto"时通过三层决策选模型）
    3. LiteLLM透传（指定模型时直连）
    """
    logger.info(
        "Chat completion请求: model=%s stream=%s messages=%d",
        request.model,
        request.stream,
        len(request.messages),
    )

    # ── 第0层：固化规则拦截 ──────────────────────────────────────────────
    from smart_router.solidification.rule_matcher import RuleMatcher
    from smart_router.solidification.response_builder import build_fake_response

    rule_store = getattr(req.app.state, "rule_store", None)
    if rule_store is not None:
        matcher = RuleMatcher(rule_store)
        system_prompt = ""
        user_content = ""
        for msg in request.messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            elif msg.get("role") == "user":
                user_content = msg.get("content", "")

        matched_rule = matcher.match(system_prompt, user_content)
        if matched_rule:
            logger.info(
                "固化规则命中: rule_id=%s name=%s",
                matched_rule.id,
                matched_rule.name,
            )
            # 异步更新命中计数
            await rule_store.update_metrics(matched_rule.id, hit=True)
            # 记录调用日志
            call_log_db = getattr(req.app.state, "call_log_db", None)
            if call_log_db:
                from smart_router.logger.call_recorder import CallRecorder
                recorder = CallRecorder(call_log_db)
                await recorder.record_call(
                    messages=request.messages,
                    model_used="smart-router-rule",
                    routing_reason=f"固化规则命中: {matched_rule.id}",
                    response_content=matched_rule.output.content,
                    tokens_input=0,
                    tokens_output=0,
                    cost=0.0,
                    latency_ms=0,
                    was_rule_hit=True,
                    rule_id=matched_rule.id,
                )
            return build_fake_response(matched_rule)

    # ── 构建litellm参数 ────────────────────────────────────────────────
    litellm_kwargs: dict[str, Any] = {}
    if request.temperature is not None:
        litellm_kwargs["temperature"] = request.temperature
    if request.max_tokens is not None:
        litellm_kwargs["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        litellm_kwargs["top_p"] = request.top_p
    if request.frequency_penalty is not None:
        litellm_kwargs["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        litellm_kwargs["presence_penalty"] = request.presence_penalty
    if request.stop is not None:
        litellm_kwargs["stop"] = request.stop
    if request.n is not None:
        litellm_kwargs["n"] = request.n
    if request.user is not None:
        litellm_kwargs["user"] = request.user
    if request.seed is not None:
        litellm_kwargs["seed"] = request.seed

    # ── 从模型注册表查找 LiteLLM 模型标识 ───────────────────────────────
    config = req.app.state.config
    registry = config.models_registry.models

    # model="auto" 时，使用路由引擎选择最佳模型
    if request.model == "auto":
        from smart_router.routing.router import route
        routing_result = await route({"messages": request.messages}, config)
        selected_model = routing_result.primary_model
        model_entry = registry.get(selected_model)
        litellm_model = model_entry.litellm_model if model_entry else selected_model
        logger.info("路由决策: %s → %s", request.model, selected_model)
    else:
        model_entry = registry.get(request.model)
        litellm_model = model_entry.litellm_model if model_entry else None

    if request.stream:
        return StreamingResponse(
            _stream_response(request.model, request.messages, litellm_kwargs, litellm_model),
            media_type="text/event-stream",
        )

    # 非流式响应
    try:
        response = await litellm.acompletion(
            model=litellm_model or request.model,
            messages=request.messages,
            stream=False,
            **litellm_kwargs,
        )

        resp_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)

        # 确保必要字段存在
        if "id" not in resp_dict:
            resp_dict["id"] = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        if "created" not in resp_dict:
            resp_dict["created"] = int(time.time())
        if "object" not in resp_dict:
            resp_dict["object"] = "chat.completion"

        # 记录调用日志
        call_log_db = getattr(req.app.state, "call_log_db", None)
        if call_log_db:
            from smart_router.logger.call_recorder import CallRecorder
            recorder = CallRecorder(call_log_db)
            usage = resp_dict.get("usage", {})
            content = ""
            choices = resp_dict.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")
            await recorder.record_call(
                messages=request.messages,
                model_used=litellm_model or request.model,
                routing_reason="直接调用",
                response_content=content,
                tokens_input=usage.get("prompt_tokens", 0),
                tokens_output=usage.get("completion_tokens", 0),
                cost=0.0,
                latency_ms=0,
            )

        logger.info(
            "Chat completion完成: id=%s usage=%s",
            resp_dict.get("id"),
            resp_dict.get("usage"),
        )
        return resp_dict

    except Exception as exc:
        raise _handle_litellm_error(exc)
