"""Solidification — 固化规则引擎

提供规则存储、匹配、响应构建三大核心能力。
"""

from smart_router.solidification.rule_store import RuleStore

# 全局 RuleStore 单例（与 config/rules.yaml 绑定）
rule_store = RuleStore()

__all__ = ["rule_store", "RuleStore"]
