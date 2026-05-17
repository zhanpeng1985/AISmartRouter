"""Rule Store — 固化规则存储与管理

从 rules.yaml 加载和管理固化规则，支持：
- 规则的增删改查与状态管理
- 热更新检测（文件变更自动重载）
- 线程安全的 YAML 读写（asyncio.Lock）
- 内存缓存，匹配性能 < 1ms
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "rules.yaml"


# ─── Pydantic 模型 ────────────────────────────────────────────────────


class RuleMatch(BaseModel):
    """规则匹配条件"""

    type: str = "keyword"  # exact / keyword / regex
    system_prompt_contains: str | None = None
    input_keywords_any: list[str] | None = None
    input_keywords_all: list[str] | None = None
    input_regex: str | None = None
    input_hash: str | None = None


class RuleOutput(BaseModel):
    """规则输出"""

    content: str
    role: str = "assistant"


class RuleMetrics(BaseModel):
    """规则度量指标"""

    accuracy: float = 0.0
    hit_count: int = 0
    cost_saved: float = 0.0


class SolidificationRule(BaseModel):
    """单条固化规则"""

    id: str
    name: str
    status: str = "active"  # active / disabled / draft
    match: RuleMatch = Field(default_factory=RuleMatch)
    output: RuleOutput
    metrics: RuleMetrics = Field(default_factory=RuleMetrics)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ─── RuleStore ────────────────────────────────────────────────────────


class RuleStore:
    """规则存储器

    负责从 YAML 文件加载规则、维护内存缓存、提供线程安全的读写操作。
    所有会修改状态的方法（add / disable / update_metrics / reload）均受 Lock 保护。
    纯读取方法（get_active_rules / get_rule / get_all_rules）直接读取内存引用，无锁。
    """

    # 匹配类型的优先级顺序：exact > keyword > regex
    _TYPE_PRIORITY: dict[str, int] = {"exact": 0, "keyword": 1, "regex": 2}

    def __init__(self, file_path: Path | str | None = None) -> None:
        self._file_path = Path(file_path) if file_path else DEFAULT_RULES_PATH
        self._rules: dict[str, SolidificationRule] = {}
        self._active_rules: list[SolidificationRule] = []
        self._lock = asyncio.Lock()
        self._last_mtime: float = 0.0

        # 同步加载初始规则（模块导入时或实例化时执行一次）
        self._load_sync()

    # ── 内部辅助方法 ───────────────────────────────────────────────────

    def _get_mtime(self) -> float:
        """获取规则文件修改时间"""
        if not self._file_path.exists():
            return 0.0
        return self._file_path.stat().st_mtime

    def _read_yaml(self) -> dict[str, Any]:
        """读取 YAML 文件内容"""
        if not self._file_path.exists():
            logger.warning("规则文件不存在: %s，初始化为空规则", self._file_path)
            return {"rules": []}
        with open(self._file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {"rules": []}

    def _save_to_file(self) -> None:
        """将当前规则持久化到 YAML（无锁，调用方需自行加锁）"""
        payload = {
            "rules": [
                self._rule_to_dict(rule) for rule in self._rules.values()
            ]
        }
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._file_path, "w", encoding="utf-8") as f:
            yaml.dump(
                payload,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )
        self._last_mtime = self._get_mtime()
        logger.info("规则已持久化到 %s，共 %d 条", self._file_path, len(self._rules))

    @staticmethod
    def _rule_to_dict(rule: SolidificationRule) -> dict[str, Any]:
        """将规则序列化为字典，过滤掉默认值以精简 YAML"""
        d = rule.model_dump()
        # 清理 match / output / metrics 中的 None / 空值，保持 YAML 整洁
        for section in ("match", "output", "metrics"):
            if section in d and isinstance(d[section], dict):
                d[section] = {
                    k: v for k, v in d[section].items() if v is not None and v != []
                }
        return d

    def _build_active_rules(self) -> list[SolidificationRule]:
        """构建按匹配优先级排序的活跃规则列表"""
        active = [
            r for r in self._rules.values() if r.status == "active"
        ]
        active.sort(key=lambda r: self._TYPE_PRIORITY.get(r.match.type, 3))
        return active

    def _load_sync(self) -> None:
        """同步加载规则（初始化时使用）"""
        raw = self._read_yaml()
        rule_list = raw.get("rules", [])
        if not isinstance(rule_list, list):
            rule_list = []

        self._rules = {}
        for item in rule_list:
            if not isinstance(item, dict):
                continue
            try:
                rule = SolidificationRule.model_validate(item)
                self._rules[rule.id] = rule
            except Exception as exc:
                logger.warning("规则加载失败，已跳过: %s — %s", item, exc)

        self._active_rules = self._build_active_rules()
        self._last_mtime = self._get_mtime()
        logger.info(
            "RuleStore 初始化完成 — 共 %d 条规则，活跃 %d 条",
            len(self._rules),
            len(self._active_rules),
        )

    # ── 公共读取接口（无锁，直接访问内存）────────────────────────────────

    def get_active_rules(self) -> list[SolidificationRule]:
        """获取所有 status=active 的规则（已按优先级排序）"""
        return self._active_rules

    def get_rule(self, rule_id: str) -> SolidificationRule | None:
        """获取单条规则"""
        return self._rules.get(rule_id)

    def get_all_rules(self) -> list[SolidificationRule]:
        """获取所有规则（含 disabled / draft）"""
        return list(self._rules.values())

    # ── 公共写入接口（受 Lock 保护）──────────────────────────────────────

    async def load_rules(self) -> None:
        """从 YAML 文件重新加载规则（会覆盖内存缓存）"""
        async with self._lock:
            self._load_sync()

    async def add_rule(self, rule: SolidificationRule) -> SolidificationRule:
        """添加新规则并持久化到 YAML

        Raises:
            ValueError: 如果规则 ID 已存在
        """
        async with self._lock:
            if rule.id in self._rules:
                raise ValueError(f"规则 ID 已存在: {rule.id}")
            self._rules[rule.id] = rule
            self._active_rules = self._build_active_rules()
            self._save_to_file()
        logger.info("规则已添加: %s (%s)", rule.id, rule.name)
        return rule

    async def disable_rule(self, rule_id: str) -> SolidificationRule | None:
        """停用规则（状态设为 disabled）"""
        async with self._lock:
            rule = self._rules.get(rule_id)
            if not rule:
                return None
            rule.status = "disabled"
            self._active_rules = self._build_active_rules()
            self._save_to_file()
        logger.info("规则已停用: %s", rule_id)
        return rule

    async def update_metrics(self, rule_id: str, hit: bool = True) -> SolidificationRule | None:
        """更新规则命中统计

        Args:
            rule_id: 规则 ID
            hit: 是否命中（True 时 hit_count +1）
        """
        async with self._lock:
            rule = self._rules.get(rule_id)
            if not rule:
                return None
            if hit:
                rule.metrics.hit_count += 1
            self._save_to_file()
        return rule

    async def reload(self) -> bool:
        """热更新：检测文件变化并重新加载

        Returns:
            True 如果检测到变更并执行了重载
        """
        current_mtime = self._get_mtime()
        if current_mtime > self._last_mtime:
            await self.load_rules()
            logger.info("规则热更新完成")
            return True
        return False
