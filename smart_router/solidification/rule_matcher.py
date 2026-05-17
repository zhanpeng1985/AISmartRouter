"""Rule Matcher — 固化规则匹配引擎

最高优先级拦截器，在路由决策之前执行。
支持三种匹配策略（按优先级排序）：
1. exact — input_hash SHA-256 精确匹配（最快）
2. keyword — 关键词包含匹配（any/all + system_prompt_contains 前置条件）
3. regex — 正则表达式匹配

匹配流程：
- 遍历活跃规则列表（RuleStore 已按优先级排序）
- 若规则配置了 system_prompt_contains，先验证前置条件
- 再按 match.type 执行对应匹配逻辑
- 命中即返回，不再检查后续规则
"""

import hashlib
import logging
import re

from smart_router.solidification.rule_store import RuleStore, SolidificationRule

logger = logging.getLogger(__name__)


class RuleMatcher:
    """规则匹配器

    接收 system_prompt 和 user_content，在活跃规则中按优先级逐一匹配，
    命中则返回对应 SolidificationRule，否则返回 None。
    """

    def __init__(self, rule_store: RuleStore) -> None:
        self._rule_store = rule_store

    def match(self, system_prompt: str, user_content: str) -> SolidificationRule | None:
        """执行规则匹配

        Args:
            system_prompt: 系统提示词（用于 system_prompt_contains 前置条件）
            user_content: 用户输入内容

        Returns:
            命中的 SolidificationRule 或 None
        """
        rules = self._rule_store.get_active_rules()
        for rule in rules:
            if self._try_match(rule, system_prompt, user_content):
                return rule
        return None

    def _try_match(
        self,
        rule: SolidificationRule,
        system_prompt: str,
        user_content: str,
    ) -> bool:
        """对单条规则尝试匹配"""
        match_cfg = rule.match
        mtype = match_cfg.type

        # ── 前置条件：system_prompt_contains ────────────────────────────
        if match_cfg.system_prompt_contains:
            if match_cfg.system_prompt_contains not in system_prompt:
                return False

        # ── 1. exact: SHA-256 精确匹配 ──────────────────────────────────
        if mtype == "exact":
            if match_cfg.input_hash is None:
                return False
            content_hash = hashlib.sha256(user_content.encode("utf-8")).hexdigest()
            return content_hash == match_cfg.input_hash

        # ── 2. keyword: 关键词匹配 ──────────────────────────────────────
        if mtype == "keyword":
            matched = False

            # keywords_any: 任一命中即匹配
            if match_cfg.input_keywords_any:
                if any(kw in user_content for kw in match_cfg.input_keywords_any):
                    matched = True

            # keywords_all: 全部命中才匹配
            if not matched and match_cfg.input_keywords_all:
                if all(kw in user_content for kw in match_cfg.input_keywords_all):
                    matched = True

            return matched

        # ── 3. regex: 正则表达式匹配 ────────────────────────────────────
        if mtype == "regex":
            if match_cfg.input_regex is None:
                return False
            try:
                return bool(re.search(match_cfg.input_regex, user_content))
            except re.error:
                logger.warning(
                    "规则 %s 正则表达式错误: %s", rule.id, match_cfg.input_regex
                )
                return False

        # 未知的匹配类型，默认不匹配
        logger.warning("规则 %s 包含未知的匹配类型: %s", rule.id, mtype)
        return False
