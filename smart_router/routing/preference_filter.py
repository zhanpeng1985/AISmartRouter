"""Preference Filter — 用户偏好过滤器"""

from smart_router.config.loader import ModelEntry, UserPreferencesConfig


def _calculate_estimated_cost(model: ModelEntry) -> float:
    """计算单次标准调用预估成本（4000 input + 2000 output tokens）"""
    return model.pricing.input_per_million * 0.004 + model.pricing.output_per_million * 0.002


def filter_by_preferences(
    models: dict[str, ModelEntry], preferences: UserPreferencesConfig
) -> list[str]:
    """
    过滤规则:
    - 排除 blocked_providers 中的供应商
    - 如果 region=prefer_cn，优先保留中国区模型
    - 排除单次成本超过 budget.max_per_call 的模型
    - 按 provider_priority 排序（优先级高的排前面）
    """
    candidates = []

    for model_id, model in models.items():
        # 排除 blocked_providers
        if model.provider in preferences.blocked_providers:
            continue

        # 排除超预算的
        estimated_cost = _calculate_estimated_cost(model)
        if estimated_cost > preferences.budget.max_per_call:
            continue

        candidates.append(model_id)

    def _sort_key(model_id: str) -> tuple:
        model = models[model_id]
        provider = model.provider

        # provider_priority 索引，不在列表中的排最后
        try:
            priority_idx = preferences.provider_priority.index(provider)
        except ValueError:
            priority_idx = len(preferences.provider_priority)

        # region 偏好：prefer_cn 时 cn 模型优先
        if preferences.region == "prefer_cn":
            region_score = 0 if model.region == "cn" else 1
        else:
            region_score = 0

        return (priority_idx, region_score)

    candidates.sort(key=_sort_key)
    return candidates
