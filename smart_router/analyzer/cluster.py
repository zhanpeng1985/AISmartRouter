"""Cluster — 调用聚类分析

从调用日志中按system_prompt_hash聚合，生成CallCluster对象，
供后续模式检测和规则生成使用。
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from smart_router.config.loader import AnalyzerConfig
from smart_router.logger.models import CallLogDB

logger = logging.getLogger(__name__)


@dataclass
class CallCluster:
    """一个调用聚类的数据结构"""

    cluster_id: str
    system_prompt_hash: str
    system_prompt_text: str
    task_type: str
    call_count: int
    unique_inputs: int
    output_distribution: dict[str, int] = field(default_factory=dict)
    avg_cost: float = 0.0
    total_cost: float = 0.0
    input_output_pairs: list[tuple[str, str]] = field(default_factory=list)


class ClusterAnalyzer:
    """从调用日志中聚类分析"""

    def __init__(self, config: AnalyzerConfig | None = None) -> None:
        self._config = config or AnalyzerConfig()
        self._min_call_count = self._config.min_call_count

    async def analyze(self, db: CallLogDB) -> list[CallCluster]:
        """执行聚类分析

        聚类步骤:
        1. 从db.get_clusters()获取按system_prompt_hash聚合的数据
        2. 对每个hash，查询其所有日志
        3. 统计output_distribution, unique_inputs, input_output_pairs
        4. 过滤: call_count >= min_call_count
        """
        raw_clusters = await db.get_clusters()
        logger.info("从数据库获取到 %d 个原始聚类", len(raw_clusters))

        clusters: list[CallCluster] = []
        for raw in raw_clusters:
            prompt_hash: str = raw.get("prompt_hash", "")
            call_count: int = raw.get("call_count", 0)

            # 过滤低频聚类
            if call_count < self._min_call_count:
                continue

            # 查询该hash下的所有日志
            logs = await db.query_by_prompt_hash(prompt_hash)

            # 统计输出分布
            output_distribution: dict[str, int] = {}
            input_set: set[str] = set()
            input_output_pairs: list[tuple[str, str]] = []
            total_cost: float = 0.0

            for log in logs:
                # 输出分布
                output_key = log.output_content.strip() if log.output_content else ""
                if output_key:
                    output_distribution[output_key] = output_distribution.get(output_key, 0) + 1

                # 唯一输入
                input_key = log.user_content.strip() if log.user_content else ""
                if input_key:
                    input_set.add(input_key)
                    if output_key:
                        input_output_pairs.append((input_key, output_key))

                total_cost += log.cost

            avg_cost = total_cost / call_count if call_count > 0 else 0.0

            # 生成 cluster_id
            cluster_id = hashlib.md5(prompt_hash.encode("utf-8")).hexdigest()[:12]

            cluster = CallCluster(
                cluster_id=cluster_id,
                system_prompt_hash=prompt_hash,
                system_prompt_text=raw.get("sample_prompt", ""),
                task_type=raw.get("task_type", ""),
                call_count=call_count,
                unique_inputs=len(input_set),
                output_distribution=output_distribution,
                avg_cost=round(avg_cost, 6),
                total_cost=round(total_cost, 6),
                input_output_pairs=input_output_pairs,
            )
            clusters.append(cluster)

        logger.info(
            "聚类分析完成: %d 个聚类通过最小调用次数过滤(>=%d)",
            len(clusters),
            self._min_call_count,
        )
        return clusters
