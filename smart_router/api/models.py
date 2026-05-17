"""Models API — 模型列表与信息查询接口"""

import logging
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["Models"])


# ─── 响应模型 ─────────────────────────────────────────────────────────


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ─── 端点 ─────────────────────────────────────────────────────────────


@router.get("/models")
async def list_models(request: Request):
    """返回已注册模型列表，兼容OpenAI API格式

    从 models_registry.yaml 读取已注册模型，返回包含 id、object、created、owned_by 的列表。
    """
    config = request.app.state.config
    models_registry = config.models_registry.models

    now = int(time.time())
    data = [
        ModelInfo(
            id=model_id,
            created=now,
            owned_by=entry.provider,
        )
        for model_id, entry in models_registry.items()
    ]

    logger.info("返回模型列表: %d个模型", len(data))
    return ModelListResponse(data=data)
