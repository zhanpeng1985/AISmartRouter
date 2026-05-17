"""Decision Cache — 路由决策缓存"""

import time

from pydantic import BaseModel


class RoutingResult(BaseModel):
    """路由决策结果"""

    primary_model: str
    fallback_models: list[str]
    reason: str
    estimated_cost: float


class DecisionCache:
    """
    缓存同类任务的路由决策，避免重复计算

    缓存key = hash(system_prompt + required_capabilities + quality_tier)
    TTL = 3600秒（1小时）
    最大缓存条目 = 1000
    """

    def __init__(self, ttl: int = 3600, max_size: int = 1000) -> None:
        self._ttl = ttl
        self._max_size = max_size
        self._cache: dict[str, tuple[RoutingResult, float]] = {}

    def get(self, cache_key: str) -> RoutingResult | None:
        """获取缓存结果，如果过期则返回 None"""
        if cache_key not in self._cache:
            return None

        result, timestamp = self._cache[cache_key]
        if time.time() - timestamp > self._ttl:
            del self._cache[cache_key]
            return None

        return result

    def put(self, cache_key: str, result: RoutingResult) -> None:
        """放入缓存，如果超过最大条目数则清理最旧的"""
        now = time.time()

        # 如果已存在，更新
        if cache_key in self._cache:
            self._cache[cache_key] = (result, now)
            return

        # 清理过期条目
        self._cleanup_expired(now)

        # 如果仍超过最大大小，清理最旧的
        if len(self._cache) >= self._max_size:
            self._cleanup_oldest(1)

        self._cache[cache_key] = (result, now)

    def invalidate_all(self) -> None:
        """清空所有缓存"""
        self._cache.clear()

    def _cleanup_expired(self, now: float) -> None:
        """清理过期条目"""
        expired = [
            key
            for key, (_, timestamp) in self._cache.items()
            if now - timestamp > self._ttl
        ]
        for key in expired:
            del self._cache[key]

    def _cleanup_oldest(self, count: int) -> None:
        """清理最旧的条目"""
        sorted_items = sorted(self._cache.items(), key=lambda x: x[1][1])
        for key, _ in sorted_items[:count]:
            del self._cache[key]
