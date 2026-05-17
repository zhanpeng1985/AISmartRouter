"""Embeddings API — 向量嵌入接口，兼容OpenAI API格式"""

import logging
from typing import Any

import litellm
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Embeddings"])


# ─── 请求模型 ─────────────────────────────────────────────────────────


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str
    encoding_format: str = "float"
    dimensions: int | None = None
    user: str | None = None


# ─── 错误处理工具 ─────────────────────────────────────────────────────


def _handle_litellm_error(exc: Exception) -> HTTPException:
    """将litellm异常转换为HTTPException"""
    exc_type = type(exc).__name__
    exc_str = str(exc)

    logger.warning("LiteLLM embedding异常 [%s]: %s", exc_type, exc_str)

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


# ─── 端点 ─────────────────────────────────────────────────────────────


@router.post("/embeddings")
async def create_embeddings(request: EmbeddingRequest, req: Request):
    """OpenAI兼容的Embeddings接口

    当前阶段直接透传到LiteLLM执行。
    """
    input_len = len(request.input) if isinstance(request.input, str) else len(request.input)
    logger.info("Embeddings请求: model=%s input_len=%s", request.model, input_len)

    litellm_kwargs: dict[str, Any] = {}
    if request.encoding_format != "float":
        litellm_kwargs["encoding_format"] = request.encoding_format
    if request.dimensions is not None:
        litellm_kwargs["dimensions"] = request.dimensions
    if request.user is not None:
        litellm_kwargs["user"] = request.user

    try:
        response = await litellm.aembedding(
            model=request.model,
            input=request.input,
            **litellm_kwargs,
        )

        resp_dict = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        logger.info("Embeddings完成: model=%s", request.model)
        return resp_dict

    except Exception as exc:
        raise _handle_litellm_error(exc)
