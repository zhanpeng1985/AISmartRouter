"""Cost Ranker — 成本排序器"""

from smart_router.config.loader import ModelEntry


def calculate_estimated_cost(model: ModelEntry) -> float:
    """计算标准单次调用预估成本（4000 input + 2000 output tokens）"""
    return model.pricing.input_per_million * 0.004 + model.pricing.output_per_million * 0.002


def _calculate_total_capability(model: ModelEntry) -> float:
    """计算能力总分"""
    caps = model.capabilities
    return (
        caps.chinese_understanding
        + caps.instruction_following
        + caps.logical_reasoning
        + caps.information_extraction
        + caps.code_generation
        + caps.creative_writing
        + caps.long_context
        + caps.structured_output
        + caps.multimodal
    )


def rank_by_cost_effectiveness(
    models: dict[str, ModelEntry], model_ids: list[str], quality_tier: str
) -> list[str]:
    """
    排序策略根据quality_tier:
    - cost_first: 纯按成本从低到高
    - balanced: 能力总分/成本 的性价比排序
    - quality_first: 纯按能力总分从高到低

    返回排序后的模型ID列表
    """
    valid_ids = [mid for mid in model_ids if mid in models]

    if quality_tier == "cost_first":
        return sorted(valid_ids, key=lambda mid: calculate_estimated_cost(models[mid]))

    if quality_tier == "quality_first":
        return sorted(
            valid_ids,
            key=lambda mid: _calculate_total_capability(models[mid]),
            reverse=True,
        )

    # balanced (default)
    def _score(mid: str) -> float:
        model = models[mid]
        cost = calculate_estimated_cost(model)
        total_cap = _calculate_total_capability(model)
        if cost <= 0:
            return float("inf")
        return total_cap / cost

    return sorted(valid_ids, key=_score, reverse=True)
