"""Response Builder — 固化响应构建器

将规则匹配结果封装为与 OpenAI chat completion 完全兼容的响应格式，
并注入 SmartRouter 元数据标记（命中来源、规则 ID、节省成本等）。
"""

import time
import uuid

from smart_router.solidification.rule_store import SolidificationRule


def build_fake_response(
    rule: SolidificationRule,
    model: str = "smart-router-rule",
) -> dict:
    """构造与 OpenAI chat completion 完全兼容的伪响应

    Args:
        rule: 命中的固化规则
        model: 模型标识（默认 smart-router-rule）

    Returns:
        符合 OpenAI /v1/chat/completions 响应格式的字典
    """
    return {
        "id": f"chatcmpl-rule-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": rule.output.role,
                    "content": rule.output.content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "_smart_router": {
            "source": "solidification_rule",
            "rule_id": rule.id,
            "rule_name": rule.name,
            "cost": 0.0,
        },
    }
