"""Logger Models — 调用日志数据模型与SQLite ORM

基于aiosqlite实现异步数据库操作，定义call_logs表结构及CRUD方法。
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger(__name__)

# ─── DDL ─────────────────────────────────────────────────────────────

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS call_logs (
    id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,

    -- 输入侧
    task_type TEXT,
    system_prompt_hash TEXT,
    system_prompt_text TEXT,
    user_content TEXT,
    input_length INTEGER,
    input_keywords TEXT,
    input_structure TEXT,

    -- 决策侧
    model_used TEXT,
    routing_reason TEXT,
    was_cached INTEGER DEFAULT 0,
    was_rule_hit INTEGER DEFAULT 0,
    rule_id TEXT,

    -- 输出侧
    output_content TEXT,
    output_type TEXT,
    tokens_input INTEGER DEFAULT 0,
    tokens_output INTEGER DEFAULT 0,
    cost REAL DEFAULT 0.0,
    latency_ms INTEGER DEFAULT 0,

    -- 质量侧
    user_accepted INTEGER DEFAULT 1,
    was_corrected INTEGER DEFAULT 0
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_system_prompt_hash ON call_logs(system_prompt_hash);",
    "CREATE INDEX IF NOT EXISTS idx_task_type ON call_logs(task_type);",
    "CREATE INDEX IF NOT EXISTS idx_timestamp ON call_logs(timestamp);",
]


# ─── 数据模型 ────────────────────────────────────────────────────────


@dataclass
class CallLog:
    """单次调用日志记录"""

    id: str = ""
    timestamp: str = ""

    # 输入侧
    task_type: str = ""
    system_prompt_hash: str = ""
    system_prompt_text: str = ""
    user_content: str = ""
    input_length: int = 0
    input_keywords: str = "[]"  # JSON array
    input_structure: str = ""

    # 决策侧
    model_used: str = ""
    routing_reason: str = ""
    was_cached: int = 0
    was_rule_hit: int = 0
    rule_id: str = ""

    # 输出侧
    output_content: str = ""
    output_type: str = ""
    tokens_input: int = 0
    tokens_output: int = 0
    cost: float = 0.0
    latency_ms: int = 0

    # 质量侧
    user_accepted: int = 1
    was_corrected: int = 0

    def to_dict(self) -> dict[str, Any]:
        """转为字典（用于JSON序列化等）"""
        return asdict(self)

    def get_keywords_list(self) -> list[str]:
        """解析input_keywords JSON为列表"""
        if not self.input_keywords:
            return []
        try:
            return json.loads(self.input_keywords)
        except (json.JSONDecodeError, TypeError):
            return []


# ─── 数据库操作类 ─────────────────────────────────────────────────────

# 表的所有列名（按建表顺序）
_COLUMNS = [
    "id", "timestamp",
    "task_type", "system_prompt_hash", "system_prompt_text",
    "user_content", "input_length", "input_keywords", "input_structure",
    "model_used", "routing_reason", "was_cached", "was_rule_hit", "rule_id",
    "output_content", "output_type", "tokens_input", "tokens_output",
    "cost", "latency_ms",
    "user_accepted", "was_corrected",
]

_INSERT_SQL = f"""
INSERT INTO call_logs ({', '.join(_COLUMNS)})
VALUES ({', '.join('?' for _ in _COLUMNS)})
"""


def _row_to_calllog(row: aiosqlite.Row) -> CallLog:
    """将数据库行转为CallLog对象"""
    return CallLog(**dict(zip(_COLUMNS, row)))


class CallLogDB:
    """异步SQLite数据库操作类

    使用aiosqlite实现，所有方法均为async，不阻塞事件循环。
    """

    def __init__(self) -> None:
        self._db: aiosqlite.Connection | None = None
        self._db_path: str = ""

    async def init(self, db_path: str) -> None:
        """初始化数据库连接并创建表

        Args:
            db_path: 数据库文件路径，如 "data/call_logs.db"
        """
        # 确保data/目录存在
        db_dir = Path(db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = db_path
        self._db = await aiosqlite.connect(db_path)
        self._db.row_factory = aiosqlite.Row

        # 启用WAL模式提升并发读写性能
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")

        # 创建表和索引
        await self._db.executescript(_CREATE_TABLE_SQL)
        for idx_sql in _CREATE_INDEXES_SQL:
            await self._db.execute(idx_sql)
        await self._db.commit()

        logger.info("数据库初始化完成: %s", db_path)

    async def close(self) -> None:
        """关闭数据库连接"""
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("数据库连接已关闭")

    @property
    def db(self) -> aiosqlite.Connection:
        """获取数据库连接（内部使用）"""
        if self._db is None:
            raise RuntimeError("数据库未初始化，请先调用 init()")
        return self._db

    async def insert(self, log: CallLog) -> None:
        """插入一条调用日志

        Args:
            log: CallLog对象
        """
        values = tuple(getattr(log, col) for col in _COLUMNS)
        try:
            await self.db.execute(_INSERT_SQL, values)
            await self.db.commit()
        except Exception:
            logger.warning("写入调用日志失败", exc_info=True)

    async def query_by_prompt_hash(self, prompt_hash: str) -> list[CallLog]:
        """按system_prompt_hash查询调用日志

        Args:
            prompt_hash: system prompt的hash值

        Returns:
            匹配的CallLog列表
        """
        cursor = await self.db.execute(
            "SELECT * FROM call_logs WHERE system_prompt_hash = ? ORDER BY timestamp DESC",
            (prompt_hash,),
        )
        rows = await cursor.fetchall()
        return [_row_to_calllog(row) for row in rows]

    async def get_stats(self) -> dict[str, Any]:
        """返回全局统计数据

        Returns:
            包含总调用次数、总成本、规则命中率等统计信息
        """
        stats: dict[str, Any] = {}

        # 总调用次数
        cursor = await self.db.execute("SELECT COUNT(*) FROM call_logs")
        row = await cursor.fetchone()
        stats["total_calls"] = row[0] if row else 0

        # 总成本
        cursor = await self.db.execute("SELECT COALESCE(SUM(cost), 0.0) FROM call_logs")
        row = await cursor.fetchone()
        stats["total_cost"] = round(row[0], 6) if row else 0.0

        # 规则命中次数
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM call_logs WHERE was_rule_hit = 1"
        )
        row = await cursor.fetchone()
        stats["total_rule_hits"] = row[0] if row else 0

        # 缓存命中次数
        cursor = await self.db.execute(
            "SELECT COUNT(*) FROM call_logs WHERE was_cached = 1"
        )
        row = await cursor.fetchone()
        stats["total_cache_hits"] = row[0] if row else 0

        # 规则节省的成本（规则命中时不产生AI调用成本）
        cursor = await self.db.execute(
            "SELECT COALESCE(SUM(cost), 0.0) FROM call_logs WHERE was_rule_hit = 1"
        )
        row = await cursor.fetchone()
        stats["cost_saved_by_rules"] = round(row[0], 6) if row else 0.0

        # 平均延迟
        cursor = await self.db.execute(
            "SELECT COALESCE(AVG(latency_ms), 0) FROM call_logs"
        )
        row = await cursor.fetchone()
        stats["avg_latency_ms"] = round(row[0], 1) if row else 0.0

        # 平均token消耗
        cursor = await self.db.execute(
            "SELECT COALESCE(AVG(tokens_input), 0), COALESCE(AVG(tokens_output), 0) FROM call_logs"
        )
        row = await cursor.fetchone()
        stats["avg_tokens_input"] = round(row[0], 1) if row else 0.0
        stats["avg_tokens_output"] = round(row[1], 1) if row else 0.0

        return stats

    async def get_clusters(self) -> list[dict[str, Any]]:
        """按system_prompt_hash聚合，返回聚类信息

        每个聚类包含：
        - prompt_hash: hash值
        - task_type: 最常见的任务类型
        - call_count: 调用次数
        - model_distribution: 模型使用分布
        - avg_cost: 平均成本
        - avg_latency_ms: 平均延迟
        - rule_hit_rate: 规则命中率
        - consistency: 输出一致性（相同输入→相同输出的比例）

        Returns:
            聚类列表，按调用次数降序排列
        """
        cursor = await self.db.execute(
            """
            SELECT
                system_prompt_hash,
                COUNT(*) as call_count
            FROM call_logs
            WHERE system_prompt_hash IS NOT NULL AND system_prompt_hash != ''
            GROUP BY system_prompt_hash
            ORDER BY call_count DESC
            """
        )
        rows = await cursor.fetchall()

        clusters: list[dict[str, Any]] = []
        for row in rows:
            prompt_hash = row[0]
            call_count = row[1]

            # 获取该聚类下的聚合统计
            agg_cursor = await self.db.execute(
                """
                SELECT
                    AVG(cost),
                    AVG(latency_ms),
                    SUM(CASE WHEN was_rule_hit = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN was_cached = 1 THEN 1 ELSE 0 END)
                FROM call_logs
                WHERE system_prompt_hash = ?
                """,
                (prompt_hash,),
            )
            agg_row = await agg_cursor.fetchone()
            avg_cost = agg_row[0] or 0.0
            avg_latency = agg_row[1] or 0.0
            rule_hits = agg_row[2] or 0
            cache_hits = agg_row[3] or 0

            # 获取该聚类下的模型分布
            model_cursor = await self.db.execute(
                """
                SELECT model_used, COUNT(*) as cnt
                FROM call_logs
                WHERE system_prompt_hash = ? AND model_used IS NOT NULL
                GROUP BY model_used
                ORDER BY cnt DESC
                """,
                (prompt_hash,),
            )
            model_rows = await model_cursor.fetchall()
            model_distribution = {r[0]: r[1] for r in model_rows if r[0]}

            # 计算规则命中率
            rule_hit_rate = round(rule_hits / call_count, 4) if call_count > 0 else 0.0

            # 获取task_type（最常见的任务类型）
            task_type_cursor = await self.db.execute(
                """
                SELECT task_type, COUNT(*) as cnt
                FROM call_logs
                WHERE system_prompt_hash = ? AND task_type IS NOT NULL
                GROUP BY task_type
                ORDER BY cnt DESC
                LIMIT 1
                """,
                (prompt_hash,),
            )
            task_row = await task_type_cursor.fetchone()
            task_type = task_row[0] if task_row else ""

            # 获取代表性system prompt文本
            sample_cursor = await self.db.execute(
                """
                SELECT system_prompt_text FROM call_logs
                WHERE system_prompt_hash = ? AND system_prompt_text IS NOT NULL
                LIMIT 1
                """,
                (prompt_hash,),
            )
            sample_row = await sample_cursor.fetchone()
            sample_prompt = sample_row[0] if sample_row else ""

            clusters.append({
                "prompt_hash": prompt_hash,
                "task_type": task_type,
                "call_count": call_count,
                "model_distribution": model_distribution,
                "avg_cost": round(avg_cost, 6),
                "avg_latency_ms": round(avg_latency, 1),
                "rule_hit_rate": rule_hit_rate,
                "cache_hit_count": cache_hits,
                "sample_prompt": sample_prompt[:200] if sample_prompt else "",
            })

        return clusters

    async def get_daily_trend(self, days: int = 7) -> list[dict[str, Any]]:
        """获取近N天每日调用量趋势

        Args:
            days: 天数

        Returns:
            每日调用量列表
        """
        cursor = await self.db.execute(
            """
            SELECT
                DATE(timestamp) as day,
                COUNT(*) as call_count,
                COALESCE(SUM(cost), 0.0) as total_cost,
                SUM(CASE WHEN was_rule_hit = 1 THEN 1 ELSE 0 END) as rule_hits
            FROM call_logs
            WHERE timestamp >= DATE('now', ?)
            GROUP BY DATE(timestamp)
            ORDER BY day ASC
            """,
            (f"-{days} days",),
        )
        rows = await cursor.fetchall()
        return [
            {
                "date": row[0],
                "call_count": row[1],
                "total_cost": round(row[2], 6),
                "rule_hits": row[3],
            }
            for row in rows
        ]

    async def get_top_models(self, limit: int = 10) -> list[dict[str, Any]]:
        """获取使用频率最高的模型

        Args:
            limit: 返回数量

        Returns:
            模型使用频率列表
        """
        cursor = await self.db.execute(
            """
            SELECT
                model_used,
                COUNT(*) as call_count,
                AVG(cost) as avg_cost,
                AVG(latency_ms) as avg_latency_ms
            FROM call_logs
            WHERE model_used IS NOT NULL AND model_used != ''
            GROUP BY model_used
            ORDER BY call_count DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "model": row[0],
                "call_count": row[1],
                "avg_cost": round(row[2], 6) if row[2] else 0.0,
                "avg_latency_ms": round(row[3], 1) if row[3] else 0.0,
            }
            for row in rows
        ]
