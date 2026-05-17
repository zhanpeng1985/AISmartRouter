"""Validator — 规则回测验证器

用聚类的历史数据回测候选规则，计算准确率和覆盖率，
判断是否达到可部署标准（accuracy >= 0.95, coverage >= 0.7）。
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

from smart_router.analyzer.cluster import CallCluster
from smart_router.analyzer.pattern_detector import _KEYWORD_PATTERN, _STOP_WORDS
from smart_router.solidification.rule_store import SolidificationRule

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """规则验证结果"""

    rule_id: str
    accuracy: float
    coverage: float
    ready_to_deploy: bool
    hits: int = 0
    correct: int = 0
    misses: int = 0
    total_samples: int = 0


def _extract_keywords(text: str) -> list[str]:
    """从文本中提取关键词（复用pattern_detector的逻辑）"""
    if not text:
        return []
    matches = _KEYWORD_PATTERN.findall(text)
    return [w for w in matches if w.lower() not in _STOP_WORDS]


class RuleValidator:
    """用聚类历史数据回测候选规则"""

    # 部署阈值
    ACCURACY_THRESHOLD = 0.95
    COVERAGE_THRESHOLD = 0.70

    def validate(
        self, rule: SolidificationRule, cluster: CallCluster
    ) -> ValidationResult:
        """用聚类的历史数据回测候选规则

        步骤:
        1. 对每条历史记录，执行规则匹配
        2. 统计命中数、正确数、未命中数
        3. 计算accuracy和coverage
        4. 判断是否ready_to_deploy
        """
        total = len(cluster.input_output_pairs)
        if total == 0:
            return ValidationResult(
                rule_id=rule.id,
                accuracy=0.0,
                coverage=0.0,
                ready_to_deploy=False,
                hits=0,
                correct=0,
                misses=0,
                total_samples=0,
            )

        hits = 0
        correct = 0

        for inp, expected_out in cluster.input_output_pairs:
            matched = self._match_rule(rule, inp, cluster.system_prompt_hash)
            if matched:
                hits += 1
                if rule.output.content.strip() == expected_out.strip():
                    correct += 1

        misses = total - hits
        accuracy = correct / hits if hits > 0 else 0.0
        coverage = hits / total if total > 0 else 0.0

        ready = (
            accuracy >= self.ACCURACY_THRESHOLD
            and coverage >= self.COVERAGE_THRESHOLD
        )

        result = ValidationResult(
            rule_id=rule.id,
            accuracy=round(accuracy, 4),
            coverage=round(coverage, 4),
            ready_to_deploy=ready,
            hits=hits,
            correct=correct,
            misses=misses,
            total_samples=total,
        )

        logger.debug(
            "规则 %s 验证: accuracy=%.2f%%, coverage=%.2f%%, ready=%s",
            rule.id,
            accuracy * 100,
            coverage * 100,
            ready,
        )
        return result

    def _match_rule(
        self, rule: SolidificationRule, user_input: str, prompt_hash: str
    ) -> bool:
        """模拟规则匹配逻辑

        根据 rule.match.type 执行不同匹配:
        - exact: input_hash 匹配
        - keyword: input_keywords_any 任一匹配
        - regex: input_regex 正则匹配
        """
        match = rule.match

        # 先检查 system_prompt_contains（如果规则指定了）
        if match.system_prompt_contains and match.system_prompt_contains != prompt_hash:
            return False

        if match.type == "exact":
            if not match.input_hash:
                return False
            input_hash = hashlib.md5(user_input.encode("utf-8")).hexdigest()[:16]
            return input_hash == match.input_hash

        elif match.type == "keyword":
            if not match.input_keywords_any:
                return False
            input_keywords = set(k.lower() for k in _extract_keywords(user_input))
            rule_keywords = set(k.lower() for k in match.input_keywords_any)
            return bool(input_keywords & rule_keywords)

        elif match.type == "regex":
            if not match.input_regex:
                return False
            try:
                return bool(re.search(match.input_regex, user_input))
            except re.error:
                logger.warning("规则 %s 正则表达式无效: %s", rule.id, match.input_regex)
                return False

        return False
