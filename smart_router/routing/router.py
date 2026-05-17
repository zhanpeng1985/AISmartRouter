"""Router — AI调用动态路由引擎"""

import hashlib
import json

from smart_router.config.loader import AppConfig
from smart_router.routing.capability_matcher import identify_required_capabilities, match_models
from smart_router.routing.cost_ranker import calculate_estimated_cost, rank_by_cost_effectiveness
from smart_router.routing.decision_cache import DecisionCache, RoutingResult
from smart_router.routing.preference_filter import filter_by_preferences

# 全局缓存实例
_decision_cache = DecisionCache()


def _get_cache_key(
    system_prompt: str, required_capabilities: dict[str, int], quality_tier: str
) -> str:
    """生成缓存key = hash(system_prompt + required_capabilities + quality_tier)"""
    content = (
        f"{system_prompt}|{json.dumps(required_capabilities, sort_keys=True)}|{quality_tier}"
    )
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _extract_messages(request_context: dict) -> list[dict]:
    """从请求上下文中提取 messages"""
    return request_context.get("messages", [])


def _extract_system_prompt(messages: list[dict]) -> str:
    """提取 system prompt 文本"""
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
    return ""


def _get_quality_tier(request_context: dict, config: AppConfig) -> str:
    """确定 quality_tier"""
    # 优先使用请求中的覆盖
    if "quality_tier" in request_context:
        return request_context["quality_tier"]

    # 检查 overrides（基于消息内容关键词匹配）
    messages = _extract_messages(request_context)
    all_text = ""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            all_text += content

    for keyword, tier in config.user_preferences.quality.overrides.items():
        if keyword in all_text:
            return tier

    return config.user_preferences.quality.default_tier


async def route(request_context: dict, config: AppConfig) -> RoutingResult:
    """
    三层决策流程:
    1. 用户偏好过滤 → 剔除不符合偏好的模型
    2. 能力匹配筛选 → 识别任务所需能力维度，筛选达标模型
    3. 成本排序 → 按性价比排序，返回primary + fallback链
    """
    messages = _extract_messages(request_context)
    system_prompt = _extract_system_prompt(messages)
    quality_tier = _get_quality_tier(request_context, config)

    # 识别所需能力
    required_capabilities = identify_required_capabilities(messages)

    # 生成缓存key
    cache_key = _get_cache_key(system_prompt, required_capabilities, quality_tier)

    # 尝试缓存命中
    cached = _decision_cache.get(cache_key)
    if cached is not None:
        return cached

    models = config.models_registry.models

    # 第1层：用户偏好过滤
    pref_filtered = filter_by_preferences(models, config.user_preferences)

    # 第2层：能力匹配筛选
    capability_matched = match_models(models, required_capabilities)

    # 取交集（既满足偏好又满足能力的模型）
    candidates = [mid for mid in pref_filtered if mid in capability_matched]

    if not candidates:
        # 如果能力匹配没有结果，回退到偏好过滤的结果
        candidates = pref_filtered

    if not candidates:
        # 如果连偏好过滤也没有结果，使用所有模型（极端情况）
        candidates = list(models.keys())

    # 第3层：成本排序
    ranked = rank_by_cost_effectiveness(models, candidates, quality_tier)

    if not ranked:
        return RoutingResult(
            primary_model="",
            fallback_models=[],
            reason="无可用模型",
            estimated_cost=0.0,
        )

    primary_model = ranked[0]
    fallback_models = ranked[1:3]  # 最多2个fallback

    primary_entry = models[primary_model]
    estimated_cost = calculate_estimated_cost(primary_entry)

    reason = (
        f"三层决策完成：偏好过滤后{len(pref_filtered)}个模型，"
        f"能力匹配后{len(capability_matched)}个模型，"
        f"按{quality_tier}排序选择{primary_model}为主模型"
    )
    if fallback_models:
        reason += f"，{', '.join(fallback_models)}为fallback"

    result = RoutingResult(
        primary_model=primary_model,
        fallback_models=fallback_models,
        reason=reason,
        estimated_cost=estimated_cost,
    )

    # 写入缓存
    _decision_cache.put(cache_key, result)

    return result
