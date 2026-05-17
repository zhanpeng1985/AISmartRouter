"""Call Recorder — AI调用异步记录器

提供fire-and-forget模式的异步调用日志记录，不阻塞主请求流程。
使用方式：
1. 直接调用 record_call() 记录完整调用
2. 或分步调用：start_record → update_routing → finish_record
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from smart_router.logger.feature_extractor import FeatureExtractor
from smart_router.logger.models import CallLog, CallLogDB

logger = logging.getLogger(__name__)


class CallRecorder:
    """异步记录调用日志，不阻塞主请求流程

    核心设计：
    - fire-and-forget模式：日志写入失败只打印warning，不影响主流程
    - 异步写入：使用asyncio.create_task将写入操作放入后台
    - 特征提取：自动从请求中提取结构化特征
    """

    def __init__(self, db: CallLogDB, extractor: FeatureExtractor | None = None) -> None:
        self._db = db
        self._extractor = extractor or FeatureExtractor()
        # 进行中的记录（分步模式用）
        self._pending_records: dict[str, CallLog] = {}

    async def record_call(
        self,
        messages: list[dict],
        model_used: str,
        routing_reason: str,
        response_content: str,
        tokens_input: int,
        tokens_output: int,
        cost: float,
        latency_ms: int,
        was_cached: bool = False,
        was_rule_hit: bool = False,
        rule_id: str | None = None,
    ) -> str:
        """记录一次完整的调用（异步写入，fire-and-forget模式）

        Args:
            messages: OpenAI格式的消息列表
            model_used: 使用的模型名称
            routing_reason: 路由决策原因
            response_content: AI响应内容
            tokens_input: 输入token数
            tokens_output: 输出token数
            cost: 本次调用成本
            latency_ms: 延迟毫秒数
            was_cached: 是否缓存命中
            was_rule_hit: 是否规则命中
            rule_id: 命中的规则ID

        Returns:
            记录ID
        """
        # 生成记录ID
        record_id = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()

        # 提取特征
        system_prompt_hash = self._extractor.extract_system_prompt_hash(messages)
        system_prompt_text = self._extractor.extract_system_prompt_text(messages)
        user_content = self._extractor.extract_user_content(messages)
        task_type = self._extractor.extract_task_type(system_prompt_text)
        input_keywords = self._extractor.extract_keywords(user_content)
        input_structure = self._extractor.detect_input_structure(user_content)
        output_type = self._extractor.detect_output_type(response_content)

        # 构建CallLog
        log = CallLog(
            id=record_id,
            timestamp=now,
            task_type=task_type,
            system_prompt_hash=system_prompt_hash,
            system_prompt_text=system_prompt_text,
            user_content=user_content,
            input_length=len(user_content),
            input_keywords=json.dumps(input_keywords, ensure_ascii=False),
            input_structure=input_structure,
            model_used=model_used,
            routing_reason=routing_reason,
            was_cached=1 if was_cached else 0,
            was_rule_hit=1 if was_rule_hit else 0,
            rule_id=rule_id or "",
            output_content=response_content,
            output_type=output_type,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost=cost,
            latency_ms=latency_ms,
        )

        # fire-and-forget异步写入
        asyncio.create_task(self._async_insert(log))

        return record_id

    async def start_record(self, messages: list[dict]) -> str:
        """分步模式：请求进入时调用，获取record_id

        Args:
            messages: OpenAI格式的消息列表

        Returns:
            记录ID，后续用于update_routing和finish_record
        """
        record_id = uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()

        # 提取输入侧特征
        system_prompt_hash = self._extractor.extract_system_prompt_hash(messages)
        system_prompt_text = self._extractor.extract_system_prompt_text(messages)
        user_content = self._extractor.extract_user_content(messages)
        task_type = self._extractor.extract_task_type(system_prompt_text)
        input_keywords = self._extractor.extract_keywords(user_content)
        input_structure = self._extractor.detect_input_structure(user_content)

        log = CallLog(
            id=record_id,
            timestamp=now,
            task_type=task_type,
            system_prompt_hash=system_prompt_hash,
            system_prompt_text=system_prompt_text,
            user_content=user_content,
            input_length=len(user_content),
            input_keywords=json.dumps(input_keywords, ensure_ascii=False),
            input_structure=input_structure,
        )

        self._pending_records[record_id] = log
        return record_id

    async def update_routing(
        self,
        record_id: str,
        model_used: str,
        routing_reason: str,
        was_cached: bool = False,
        was_rule_hit: bool = False,
        rule_id: str | None = None,
    ) -> None:
        """分步模式：路由决策后调用

        Args:
            record_id: start_record返回的ID
            model_used: 使用的模型名称
            routing_reason: 路由决策原因
            was_cached: 是否缓存命中
            was_rule_hit: 是否规则命中
            rule_id: 命中的规则ID
        """
        log = self._pending_records.get(record_id)
        if log is None:
            logger.warning("update_routing: 未找到记录 %s", record_id)
            return

        log.model_used = model_used
        log.routing_reason = routing_reason
        log.was_cached = 1 if was_cached else 0
        log.was_rule_hit = 1 if was_rule_hit else 0
        log.rule_id = rule_id or ""

    async def finish_record(
        self,
        record_id: str,
        response_content: str,
        tokens_input: int,
        tokens_output: int,
        cost: float,
        latency_ms: int,
    ) -> None:
        """分步模式：AI响应后调用，异步写入数据库

        Args:
            record_id: start_record返回的ID
            response_content: AI响应内容
            tokens_input: 输入token数
            tokens_output: 输出token数
            cost: 本次调用成本
            latency_ms: 延迟毫秒数
        """
        log = self._pending_records.pop(record_id, None)
        if log is None:
            logger.warning("finish_record: 未找到记录 %s", record_id)
            return

        # 提取输出侧特征
        output_type = self._extractor.detect_output_type(response_content)

        log.output_content = response_content
        log.output_type = output_type
        log.tokens_input = tokens_input
        log.tokens_output = tokens_output
        log.cost = cost
        log.latency_ms = latency_ms

        # fire-and-forget异步写入
        asyncio.create_task(self._async_insert(log))

    async def update_feedback(
        self,
        record_id: str,
        user_accepted: bool = True,
        was_corrected: bool = False,
    ) -> None:
        """更新用户反馈信息（质量侧）

        注意：此方法需要同步写入（非fire-and-forget），因为反馈是后置的。

        Args:
            record_id: 记录ID
            user_accepted: 用户是否接受
            was_corrected: 是否被用户修正
        """
        try:
            await self._db.db.execute(
                """
                UPDATE call_logs
                SET user_accepted = ?, was_corrected = ?
                WHERE id = ?
                """,
                (1 if user_accepted else 0, 1 if was_corrected else 0, record_id),
            )
            await self._db.db.commit()
        except Exception:
            logger.warning("更新反馈信息失败: %s", record_id, exc_info=True)

    async def get_stats(self) -> dict[str, Any]:
        """返回统计数据

        Returns:
            统计字典，包含：
            - total_calls: 总调用次数
            - total_cost: 总成本
            - total_rule_hits: 规则命中次数
            - cost_saved: 规则节省的成本
            - top_models: 按使用频率排序的模型
            - daily_trend: 近7天每日调用量
        """
        try:
            base_stats = await self._db.get_stats()
            top_models = await self._db.get_top_models(limit=5)
            daily_trend = await self._db.get_daily_trend(days=7)

            return {
                "total_calls": base_stats.get("total_calls", 0),
                "total_cost": base_stats.get("total_cost", 0.0),
                "total_rule_hits": base_stats.get("total_rule_hits", 0),
                "total_cache_hits": base_stats.get("total_cache_hits", 0),
                "cost_saved": base_stats.get("cost_saved_by_rules", 0.0),
                "avg_latency_ms": base_stats.get("avg_latency_ms", 0.0),
                "avg_tokens_input": base_stats.get("avg_tokens_input", 0.0),
                "avg_tokens_output": base_stats.get("avg_tokens_output", 0.0),
                "top_models": top_models,
                "daily_trend": daily_trend,
            }
        except Exception:
            logger.warning("获取统计数据失败", exc_info=True)
            return {
                "total_calls": 0,
                "total_cost": 0.0,
                "total_rule_hits": 0,
                "total_cache_hits": 0,
                "cost_saved": 0.0,
                "avg_latency_ms": 0.0,
                "avg_tokens_input": 0.0,
                "avg_tokens_output": 0.0,
                "top_models": [],
                "daily_trend": [],
            }

    async def get_clusters(self) -> list[dict[str, Any]]:
        """获取prompt聚类数据

        Returns:
            聚类列表
        """
        try:
            return await self._db.get_clusters()
        except Exception:
            logger.warning("获取聚类数据失败", exc_info=True)
            return []

    # ─── 内部方法 ──────────────────────────────────────────────────────

    async def _async_insert(self, log: CallLog) -> None:
        """异步写入日志（内部使用，失败只打印warning）"""
        try:
            await self._db.insert(log)
        except Exception:
            logger.warning(
                "异步写入调用日志失败: id=%s, model=%s",
                log.id,
                log.model_used,
                exc_info=True,
            )
