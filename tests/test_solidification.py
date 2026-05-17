"""Tests for the solidification rule engine."""

import hashlib
import os
import tempfile
from pathlib import Path

import pytest

from smart_router.solidification.rule_matcher import RuleMatcher
from smart_router.solidification.rule_store import (
    RuleMatch,
    RuleOutput,
    RuleStore,
    SolidificationRule,
)
from smart_router.solidification.response_builder import build_fake_response


@pytest.fixture
def temp_rules_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        f.write("rules: []")
        path = f.name
    yield Path(path)
    os.unlink(path)


@pytest.fixture
def store(temp_rules_file):
    return RuleStore(temp_rules_file)


@pytest.fixture
def matcher(store):
    return RuleMatcher(store)


class TestRuleStore:
    def test_initial_empty(self, store):
        assert len(store.get_all_rules()) == 0
        assert len(store.get_active_rules()) == 0

    @pytest.mark.asyncio
    async def test_add_rule(self, store):
        rule = SolidificationRule(
            id="kw_001",
            name="关键词测试",
            match=RuleMatch(type="keyword", input_keywords_any=["保险"]),
            output=RuleOutput(content="命中"),
        )
        await store.add_rule(rule)
        assert len(store.get_all_rules()) == 1
        assert store.get_rule("kw_001") is not None

    @pytest.mark.asyncio
    async def test_disable_rule(self, store):
        rule = SolidificationRule(
            id="kw_001",
            name="关键词测试",
            match=RuleMatch(type="keyword", input_keywords_any=["保险"]),
            output=RuleOutput(content="命中"),
        )
        await store.add_rule(rule)
        await store.disable_rule("kw_001")
        assert store.get_rule("kw_001").status == "disabled"
        assert len(store.get_active_rules()) == 0

    @pytest.mark.asyncio
    async def test_update_metrics(self, store):
        rule = SolidificationRule(
            id="ex_001",
            name="精确匹配",
            match=RuleMatch(type="exact", input_hash="a" * 64),
            output=RuleOutput(content="命中"),
        )
        await store.add_rule(rule)
        await store.update_metrics("ex_001", hit=True)
        assert store.get_rule("ex_001").metrics.hit_count == 1

    @pytest.mark.asyncio
    async def test_reload_no_change(self, store):
        assert await store.reload() is False

    def test_priority_order(self, store):
        # exact should be first in active rules
        import asyncio

        async def setup():
            await store.add_rule(
                SolidificationRule(
                    id="kw_001",
                    name="kw",
                    match=RuleMatch(type="keyword", input_keywords_any=["x"]),
                    output=RuleOutput(content="kw"),
                )
            )
            await store.add_rule(
                SolidificationRule(
                    id="ex_001",
                    name="ex",
                    match=RuleMatch(type="exact", input_hash="a" * 64),
                    output=RuleOutput(content="ex"),
                )
            )

        asyncio.run(setup())
        active = store.get_active_rules()
        assert active[0].id == "ex_001"
        assert active[1].id == "kw_001"


class TestRuleMatcher:
    @pytest.mark.asyncio
    async def test_exact_match(self, store, matcher):
        text = "精确文本"
        h = hashlib.sha256(text.encode()).hexdigest()
        await store.add_rule(
            SolidificationRule(
                id="ex_001",
                name="精确匹配",
                match=RuleMatch(type="exact", input_hash=h),
                output=RuleOutput(content="命中"),
            )
        )
        result = matcher.match("", text)
        assert result is not None
        assert result.id == "ex_001"

    @pytest.mark.asyncio
    async def test_keyword_any(self, store, matcher):
        await store.add_rule(
            SolidificationRule(
                id="kw_001",
                name="关键词",
                match=RuleMatch(type="keyword", input_keywords_any=["保险", "条款"]),
                output=RuleOutput(content="命中"),
            )
        )
        assert matcher.match("", "我想了解保险") is not None
        assert matcher.match("", "我想了解条款") is not None
        assert matcher.match("", "无关内容") is None

    @pytest.mark.asyncio
    async def test_keyword_all(self, store, matcher):
        await store.add_rule(
            SolidificationRule(
                id="kw_002",
                name="全部关键词",
                match=RuleMatch(type="keyword", input_keywords_all=["重疾", "赔付"]),
                output=RuleOutput(content="命中"),
            )
        )
        assert matcher.match("", "重疾险的赔付标准") is not None
        assert matcher.match("", "重疾险") is None

    @pytest.mark.asyncio
    async def test_system_prompt_filter(self, store, matcher):
        await store.add_rule(
            SolidificationRule(
                id="kw_003",
                name="系统提示词过滤",
                match=RuleMatch(
                    type="keyword",
                    system_prompt_contains="分类",
                    input_keywords_any=["测试"],
                ),
                output=RuleOutput(content="命中"),
            )
        )
        assert matcher.match("请进行分类", "这是一个测试") is not None
        assert matcher.match("请进行提取", "这是一个测试") is None

    @pytest.mark.asyncio
    async def test_regex_match(self, store, matcher):
        await store.add_rule(
            SolidificationRule(
                id="re_001",
                name="正则测试",
                match=RuleMatch(type="regex", input_regex=r"\d{11}"),
                output=RuleOutput(content="命中"),
            )
        )
        assert matcher.match("", "我的电话是13800138000") is not None
        assert matcher.match("", "没有号码") is None

    def test_no_rules_returns_none(self, matcher):
        assert matcher.match("", "任意内容") is None


class TestResponseBuilder:
    def test_build_fake_response(self):
        rule = SolidificationRule(
            id="r_001",
            name="测试规则",
            match=RuleMatch(type="keyword"),
            output=RuleOutput(content="响应内容", role="assistant"),
        )
        resp = build_fake_response(rule)
        assert resp["object"] == "chat.completion"
        assert resp["model"] == "smart-router-rule"
        assert resp["choices"][0]["message"]["content"] == "响应内容"
        assert resp["_smart_router"]["source"] == "solidification_rule"
        assert resp["_smart_router"]["rule_id"] == "r_001"
        assert resp["usage"]["total_tokens"] == 0
