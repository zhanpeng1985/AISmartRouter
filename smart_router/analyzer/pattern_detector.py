"""Pattern Detector — 模式检测器

对聚类数据进行可固化模式检测：
- EXACT_MATCH: 相同输入→相同输出（一致性 >= 95%）
- KEYWORD_RULE: 含特定关键词→固定输出（区分性关键词）
- NOT_SOLIDIFIABLE: 无法固化
"""

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from smart_router.analyzer.cluster import CallCluster

logger = logging.getLogger(__name__)


class PatternType(str, Enum):
    """模式类型枚举"""

    EXACT_MATCH = "exact_match"
    KEYWORD_RULE = "keyword_rule"
    NOT_SOLIDIFIABLE = "not_solidifiable"


@dataclass
class DetectedPattern:
    """检测到的模式"""

    pattern_type: PatternType
    confidence: float
    details: dict = field(default_factory=dict)


# 关键词提取: 匹配中文词组和英文单词
_KEYWORD_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,4}|[a-zA-Z_]\w*", re.UNICODE)

# 停用词
_STOP_WORDS = frozenset({
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人",
    "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去",
    "你", "会", "着", "没有", "看", "好", "自己", "这", "他", "她",
    "the", "is", "a", "an", "and", "or", "of", "to", "in", "for",
    "on", "with", "at", "by", "from", "it", "this", "that", "be",
})


def _extract_keywords(text: str) -> list[str]:
    """从文本中提取关键词（中文2-4字词组 + 英文单词），过滤停用词"""
    if not text:
        return []
    matches = _KEYWORD_PATTERN.findall(text)
    return [w for w in matches if w.lower() not in _STOP_WORDS]


class PatternDetector:
    """检测可固化的重复调用模式"""

    # 阈值常量
    EXACT_MATCH_THRESHOLD = 0.95
    KEYWORD_IN_CLASS_THRESHOLD = 0.80
    KEYWORD_OUT_CLASS_THRESHOLD = 0.20

    def detect(self, cluster: CallCluster) -> DetectedPattern:
        """检测可固化模式

        检测优先级:
        1. 精确匹配检测: 相同input→相同output一致性 >= 95%
        2. 关键词规则检测: 区分性关键词
        3. 都不满足 → NOT_SOLIDIFIABLE
        """
        # 优先尝试精确匹配
        exact = self._detect_exact_match(cluster)
        if exact.pattern_type == PatternType.EXACT_MATCH:
            return exact

        # 尝试关键词规则
        keyword = self._detect_keyword_rule(cluster)
        if keyword.pattern_type == PatternType.KEYWORD_RULE:
            return keyword

        return DetectedPattern(
            pattern_type=PatternType.NOT_SOLIDIFIABLE,
            confidence=0.0,
            details={"reason": "不满足精确匹配或关键词规则阈值"},
        )

    def _detect_exact_match(self, cluster: CallCluster) -> DetectedPattern:
        """精确匹配检测: 相同输入是否总对应相同输出"""
        if not cluster.input_output_pairs:
            return DetectedPattern(
                pattern_type=PatternType.NOT_SOLIDIFIABLE,
                confidence=0.0,
                details={"reason": "无输入输出对"},
            )

        # 按 input 分组，统计每个 input 对应的 output 分布
        input_to_outputs: dict[str, Counter] = defaultdict(Counter)
        for inp, outp in cluster.input_output_pairs:
            input_to_outputs[inp][outp] += 1

        if not input_to_outputs:
            return DetectedPattern(
                pattern_type=PatternType.NOT_SOLIDIFIABLE,
                confidence=0.0,
                details={"reason": "无输入输出对"},
            )

        # 计算一致性：每个input中，最常见output占比的加权平均
        total_pairs = 0
        consistent_pairs = 0
        for inp, output_counter in input_to_outputs.items():
            most_common_count = output_counter.most_common(1)[0][1]
            total_for_input = sum(output_counter.values())
            consistent_pairs += most_common_count
            total_pairs += total_for_input

        consistency = consistent_pairs / total_pairs if total_pairs > 0 else 0.0

        if consistency >= self.EXACT_MATCH_THRESHOLD:
            return DetectedPattern(
                pattern_type=PatternType.EXACT_MATCH,
                confidence=round(consistency, 4),
                details={
                    "consistency": round(consistency, 4),
                    "unique_inputs": len(input_to_outputs),
                    "total_pairs": total_pairs,
                    "consistent_pairs": consistent_pairs,
                },
            )

        return DetectedPattern(
            pattern_type=PatternType.NOT_SOLIDIFIABLE,
            confidence=round(consistency, 4),
            details={
                "consistency": round(consistency, 4),
                "reason": f"一致性 {consistency:.2%} 低于阈值 {self.EXACT_MATCH_THRESHOLD:.0%}",
            },
        )

    def _detect_keyword_rule(self, cluster: CallCluster) -> DetectedPattern:
        """关键词规则检测: 区分性关键词指向特定输出"""
        if not cluster.input_output_pairs:
            return DetectedPattern(
                pattern_type=PatternType.NOT_SOLIDIFIABLE,
                confidence=0.0,
                details={"reason": "无输入输出对"},
            )

        # 如果输出类别只有1个，无法做关键词区分
        if len(cluster.output_distribution) <= 1:
            return DetectedPattern(
                pattern_type=PatternType.NOT_SOLIDIFIABLE,
                confidence=0.0,
                details={"reason": "输出类别仅1个，无需关键词区分"},
            )

        # 按 output 分组，收集每组的输入关键词
        output_to_keywords: dict[str, list[str]] = defaultdict(list)
        output_counts: dict[str, int] = defaultdict(int)

        for inp, outp in cluster.input_output_pairs:
            keywords = _extract_keywords(inp)
            output_to_keywords[outp].extend(keywords)
            output_counts[outp] += 1

        # 对每个output类别，计算关键词的区分性
        # 区分性关键词: 在该类别中出现率 > 80%，在其他类别中 < 20%
        total_inputs = sum(output_counts.values())
        discriminative_keywords: dict[str, list[str]] = {}

        for outp, keywords in output_to_keywords.items():
            if output_counts[outp] == 0:
                continue

            keyword_counter = Counter(keywords)
            class_size = output_counts[outp]

            # 该类别中每个关键词的出现率
            discriminating = []
            for kw, count in keyword_counter.items():
                in_class_rate = count / class_size

                # 在其他类别中的出现率
                other_count = 0
                other_total = 0
                for other_outp, other_kws in output_to_keywords.items():
                    if other_outp == outp:
                        continue
                    other_counter = Counter(other_kws)
                    other_count += other_counter.get(kw, 0)
                    other_total += output_counts[other_outp]

                out_class_rate = other_count / other_total if other_total > 0 else 0.0

                if (
                    in_class_rate >= self.KEYWORD_IN_CLASS_THRESHOLD
                    and out_class_rate < self.KEYWORD_OUT_CLASS_THRESHOLD
                ):
                    discriminating.append(kw)

            if discriminating:
                discriminative_keywords[outp] = discriminating

        if not discriminative_keywords:
            return DetectedPattern(
                pattern_type=PatternType.NOT_SOLIDIFIABLE,
                confidence=0.0,
                details={"reason": "未找到区分性关键词"},
            )

        # 计算置信度: 被区分性关键词覆盖的输入占比
        covered_inputs = 0
        for inp, outp in cluster.input_output_pairs:
            input_keywords = set(_extract_keywords(inp))
            if outp in discriminative_keywords:
                if input_keywords & set(discriminative_keywords[outp]):
                    covered_inputs += 1

        coverage = covered_inputs / len(cluster.input_output_pairs)
        confidence = round(min(coverage, 1.0), 4)

        return DetectedPattern(
            pattern_type=PatternType.KEYWORD_RULE,
            confidence=confidence,
            details={
                "discriminative_keywords": discriminative_keywords,
                "coverage": round(coverage, 4),
                "output_categories": len(cluster.output_distribution),
            },
        )
