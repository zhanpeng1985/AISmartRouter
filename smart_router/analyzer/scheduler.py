"""Scheduler — 固化分析主流程编排

协调完整的分析流程:
聚类分析 → 模式检测 → 规则生成 → 回测验证 → 分析报告
"""

import logging
from dataclasses import asdict
from typing import Any

from smart_router.analyzer.cluster import CallCluster, ClusterAnalyzer
from smart_router.analyzer.pattern_detector import DetectedPattern, PatternType, PatternDetector
from smart_router.analyzer.rule_generator import RuleGenerator
from smart_router.analyzer.validator import RuleValidator, ValidationResult
from smart_router.config.loader import AnalyzerConfig
from smart_router.logger.models import CallLogDB
from smart_router.solidification.rule_store import RuleStore, SolidificationRule

logger = logging.getLogger(__name__)


class SolidificationAnalyzer:
    """固化分析主流程编排"""

    def __init__(
        self,
        db: CallLogDB,
        rule_store: RuleStore,
        config: AnalyzerConfig | None = None,
    ) -> None:
        self._db = db
        self._rule_store = rule_store
        self._config = config or AnalyzerConfig()
        self._cluster_analyzer = ClusterAnalyzer(self._config)
        self._pattern_detector = PatternDetector()
        self._rule_generator = RuleGenerator()
        self._validator = RuleValidator()

    async def run_analysis(self) -> dict[str, Any]:
        """执行完整分析流程

        流程:
        1. 聚类分析 → 获取所有聚类
        2. 模式检测 → 对每个聚类检测模式
        3. 规则生成 → 对可固化模式生成候选规则
        4. 回测验证 → 验证准确率和覆盖率
        5. 返回分析报告

        Returns:
            分析报告字典
        """
        logger.info("=" * 50)
        logger.info("固化分析流程启动")
        logger.info("=" * 50)

        # Step 1: 聚类分析
        clusters = await self._cluster_analyzer.analyze(self._db)
        logger.info("Step 1 聚类分析完成: %d 个聚类", len(clusters))

        if not clusters:
            return self._empty_report()

        # Step 2-4: 对每个聚类执行模式检测→规则生成→回测验证
        all_candidate_rules: list[SolidificationRule] = []
        all_validation_results: list[ValidationResult] = []
        solidifiable_count = 0
        not_solidifiable_count = 0

        for cluster in clusters:
            # Step 2: 模式检测
            pattern = self._pattern_detector.detect(cluster)
            logger.debug(
                "聚类 %s 模式: %s (confidence=%.2f)",
                cluster.cluster_id,
                pattern.pattern_type.value,
                pattern.confidence,
            )

            if pattern.pattern_type == PatternType.NOT_SOLIDIFIABLE:
                not_solidifiable_count += 1
                continue

            solidifiable_count += 1

            # Step 3: 规则生成
            rules = self._rule_generator.generate(cluster, pattern)
            if not rules:
                continue

            # Step 4: 回测验证
            for rule in rules:
                validation = self._validator.validate(rule, cluster)
                # 更新规则的metrics
                rule.metrics.accuracy = validation.accuracy
                rule.metrics.cost_saved = round(
                    validation.hits * cluster.avg_cost, 6
                )
                all_candidate_rules.append(rule)
                all_validation_results.append(validation)

        # 统计可部署规则
        ready_rule_ids = {v.rule_id for v in all_validation_results if v.ready_to_deploy}
        ready_rules = [r for r in all_candidate_rules if r.id in ready_rule_ids]

        # 估算月度节省
        estimated_monthly_savings = self._estimate_savings(
            ready_rules, clusters, all_validation_results
        )

        report = {
            "analyzed_clusters": len(clusters),
            "solidifiable_clusters": solidifiable_count,
            "not_solidifiable": not_solidifiable_count,
            "candidate_rules": [r.model_dump() for r in all_candidate_rules],
            "candidate_rule_count": len(all_candidate_rules),
            "ready_to_deploy": [r.model_dump() for r in ready_rules],
            "ready_to_deploy_count": len(ready_rules),
            "validation_results": [asdict(v) for v in all_validation_results],
            "estimated_monthly_savings": round(estimated_monthly_savings, 2),
        }

        logger.info(
            "固化分析完成: 分析%d个聚类, 可固化%d个, "
            "候选规则%d条, 可部署%d条, 预估月节省$%.2f",
            report["analyzed_clusters"],
            report["solidifiable_clusters"],
            report["candidate_rule_count"],
            report["ready_to_deploy_count"],
            report["estimated_monthly_savings"],
        )
        return report

    async def deploy_rule(self, rule: SolidificationRule) -> SolidificationRule:
        """将验证通过的规则添加到rule_store

        Args:
            rule: 要部署的规则（status会被设为active）

        Returns:
            部署后的规则
        """
        # 部署时将状态从draft改为active
        rule.status = "active"
        deployed = await self._rule_store.add_rule(rule)
        logger.info("规则已部署: %s (%s)", rule.id, rule.name)
        return deployed

    def _empty_report(self) -> dict[str, Any]:
        """返回空报告"""
        return {
            "analyzed_clusters": 0,
            "solidifiable_clusters": 0,
            "not_solidifiable": 0,
            "candidate_rules": [],
            "candidate_rule_count": 0,
            "ready_to_deploy": [],
            "ready_to_deploy_count": 0,
            "validation_results": [],
            "estimated_monthly_savings": 0.0,
        }

    def _estimate_savings(
        self,
        ready_rules: list[SolidificationRule],
        clusters: list[CallCluster],
        validation_results: list[ValidationResult],
    ) -> float:
        """估算月度节省成本

        逻辑:
        - 对每条可部署规则，根据验证结果中的命中数和对应聚类的avg_cost
        - 假设每天命中量 ≈ 历史日均命中量
        - 月度 = 日均 × 30
        """
        # 构建 rule_id → validation 映射
        validation_map = {v.rule_id: v for v in validation_results}

        # 构建 cluster_id → cluster 映射
        # 需要从规则ID反推cluster
        total_monthly_savings = 0.0
        for rule in ready_rules:
            v = validation_map.get(rule.id)
            if not v or v.hits == 0:
                continue

            # 找到规则对应的cluster（通过system_prompt_contains匹配）
            prompt_hash = rule.match.system_prompt_contains
            matched_cluster = None
            for c in clusters:
                if c.system_prompt_hash == prompt_hash:
                    matched_cluster = c
                    break

            if not matched_cluster:
                continue

            # 日均命中 × 平均成本 × 30天
            # 用历史数据的命中比例 × 聚类总调用 × avg_cost
            hit_rate = v.hits / v.total_samples if v.total_samples > 0 else 0.0
            daily_calls = matched_cluster.call_count / 30.0  # 粗估日均
            daily_savings = daily_calls * hit_rate * matched_cluster.avg_cost
            monthly_savings = daily_savings * 30.0
            total_monthly_savings += monthly_savings

        return total_monthly_savings
