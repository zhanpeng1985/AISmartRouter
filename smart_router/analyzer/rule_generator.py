"""Rule Generator — 候选规则生成器

根据检测到的模式生成候选固化规则:
- EXACT_MATCH → 为每个稳定的(input_hash, output)对生成exact规则
- KEYWORD_RULE → 为每个output类别提取区分性关键词，生成keyword规则
"""

import hashlib
import logging
import uuid
from collections import Counter, defaultdict
from typing import Any

from smart_router.analyzer.cluster import CallCluster
from smart_router.analyzer.pattern_detector import DetectedPattern, PatternType
from smart_router.solidification.rule_store import (
    RuleMatch,
    RuleMetrics,
    RuleOutput,
    SolidificationRule,
)

logger = logging.getLogger(__name__)


class RuleGenerator:
    """根据检测到的模式生成候选规则"""

    def generate(
        self, cluster: CallCluster, pattern: DetectedPattern
    ) -> list[SolidificationRule] | None:
        """根据检测到的模式生成候选规则

        Args:
            cluster: 调用聚类
            pattern: 检测到的模式

        Returns:
            候选规则列表，如果模式不可固化则返回None
        """
        if pattern.pattern_type == PatternType.EXACT_MATCH:
            return self._generate_exact_rules(cluster, pattern)
        elif pattern.pattern_type == PatternType.KEYWORD_RULE:
            return self._generate_keyword_rules(cluster, pattern)
        else:
            logger.debug(
                "聚类 %s 不可固化，跳过规则生成", cluster.cluster_id
            )
            return None

    def _generate_exact_rules(
        self, cluster: CallCluster, pattern: DetectedPattern
    ) -> list[SolidificationRule]:
        """为EXACT_MATCH模式生成exact规则

        对每个稳定的(input, output)对，生成一条exact规则。
        稳定 = 该input对应的output完全一致（无冲突）。
        """
        # 按 input 分组，找出稳定的 input→output 映射
        input_to_outputs: dict[str, Counter] = defaultdict(Counter)
        for inp, outp in cluster.input_output_pairs:
            input_to_outputs[inp][outp] += 1

        rules: list[SolidificationRule] = []
        for inp, output_counter in input_to_outputs.items():
            # 只取完全一致的映射（该input的所有输出相同）
            if len(output_counter) != 1:
                continue

            output = output_counter.most_common(1)[0][0]
            input_hash = hashlib.md5(inp.encode("utf-8")).hexdigest()[:16]

            # 只为有足够代表性的pair生成规则（至少出现2次）
            count = sum(output_counter.values())
            if count < 2:
                continue

            rule_id = f"auto-exact-{cluster.cluster_id}-{input_hash[:8]}"
            rule_name = (
                f"精确匹配规则: {cluster.task_type} - "
                f"{inp[:50]}..." if len(inp) > 50 else f"精确匹配规则: {cluster.task_type} - {inp}"
            )

            rule = SolidificationRule(
                id=rule_id,
                name=rule_name,
                status="draft",
                match=RuleMatch(
                    type="exact",
                    system_prompt_contains=cluster.system_prompt_hash,
                    input_hash=input_hash,
                ),
                output=RuleOutput(
                    content=output,
                    role="assistant",
                ),
                metrics=RuleMetrics(
                    accuracy=1.0,
                    hit_count=0,
                    cost_saved=0.0,
                ),
            )
            rules.append(rule)

        logger.info(
            "聚类 %s EXACT_MATCH 生成 %d 条候选规则",
            cluster.cluster_id,
            len(rules),
        )
        return rules

    def _generate_keyword_rules(
        self, cluster: CallCluster, pattern: DetectedPattern
    ) -> list[SolidificationRule]:
        """为KEYWORD_RULE模式生成keyword规则

        从模式的details中获取区分性关键词，为每个output类别生成一条规则。
        """
        discriminative_keywords: dict[str, list[str]] = pattern.details.get(
            "discriminative_keywords", {}
        )

        if not discriminative_keywords:
            logger.warning(
                "聚类 %s KEYWORD_RULE 模式缺少区分性关键词", cluster.cluster_id
            )
            return []

        rules: list[SolidificationRule] = []
        for output_content, keywords in discriminative_keywords.items():
            if not keywords:
                continue

            # 取前5个最有区分性的关键词
            top_keywords = keywords[:5]

            rule_id = (
                f"auto-kw-{cluster.cluster_id}-"
                f"{hashlib.md5(output_content.encode('utf-8')).hexdigest()[:8]}"
            )
            keyword_summary = ", ".join(top_keywords)
            rule_name = (
                f"关键词规则: {cluster.task_type} - [{keyword_summary}]"
            )

            rule = SolidificationRule(
                id=rule_id,
                name=rule_name,
                status="draft",
                match=RuleMatch(
                    type="keyword",
                    system_prompt_contains=cluster.system_prompt_hash,
                    input_keywords_any=top_keywords,
                ),
                output=RuleOutput(
                    content=output_content,
                    role="assistant",
                ),
                metrics=RuleMetrics(
                    accuracy=0.0,
                    hit_count=0,
                    cost_saved=0.0,
                ),
            )
            rules.append(rule)

        logger.info(
            "聚类 %s KEYWORD_RULE 生成 %d 条候选规则",
            cluster.cluster_id,
            len(rules),
        )
        return rules
