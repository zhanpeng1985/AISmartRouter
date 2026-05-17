"""Config Loader — YAML配置加载器

支持从config/目录加载所有YAML配置文件，提供：
- 类型安全的Pydantic配置模型
- 文件修改时间检测的热更新
- 统一的配置访问入口
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# 默认配置目录：项目根目录下的 config/
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"


# ─── Pydantic 配置模型 ───────────────────────────────────────────────


class ServerConfig(BaseModel):
    """服务器配置"""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


class DatabaseConfig(BaseModel):
    """数据库配置"""

    path: str = "data/call_logs.db"


class AnalyzerConfig(BaseModel):
    """分析器配置"""

    schedule: str = "daily"
    min_call_count: int = 20
    min_consistency: float = 0.85


class SolidificationConfig(BaseModel):
    """固化策略配置"""

    auto_deploy_threshold: float = 0.98
    manual_review_threshold: float = 0.95


class SettingsConfig(BaseModel):
    """全局设置"""

    server: ServerConfig = Field(default_factory=ServerConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    analyzer: AnalyzerConfig = Field(default_factory=AnalyzerConfig)
    solidification: SolidificationConfig = Field(default_factory=SolidificationConfig)


class BudgetConfig(BaseModel):
    """预算配置"""

    max_per_call: float = 0.5
    monthly_limit: float = 500.0


class QualityConfig(BaseModel):
    """质量偏好配置"""

    default_tier: str = "balanced"
    overrides: dict[str, str] = Field(default_factory=dict)


class UserPreferencesConfig(BaseModel):
    """用户偏好配置"""

    provider_priority: list[str] = Field(
        default_factory=lambda: ["alibaba", "deepseek", "baidu", "openai", "anthropic"]
    )
    blocked_providers: list[str] = Field(default_factory=list)
    region: str = "prefer_cn"
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)


class UserPreferencesWrapper(BaseModel):
    """用户偏好包装"""

    user_preferences: UserPreferencesConfig = Field(default_factory=UserPreferencesConfig)


class ModelPricing(BaseModel):
    """模型定价"""

    input_per_million: float = 0.0
    output_per_million: float = 0.0


class ModelCapabilities(BaseModel):
    """模型能力评分（1-5分）"""

    chinese_understanding: float = 3.0
    instruction_following: float = 3.0
    logical_reasoning: float = 3.0
    information_extraction: float = 3.0
    code_generation: float = 3.0
    creative_writing: float = 3.0
    long_context: float = 3.0
    structured_output: float = 3.0
    multimodal: float = 1.0


class ModelEntry(BaseModel):
    """单个模型注册信息"""

    provider: str
    litellm_model: str
    region: str = "cn"
    context_window: int = 32768
    pricing: ModelPricing = Field(default_factory=ModelPricing)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)


class ModelsRegistryConfig(BaseModel):
    """模型注册表"""

    models: dict[str, ModelEntry] = Field(default_factory=dict)


class RuleMatch(BaseModel):
    """规则匹配条件"""

    type: str = "keyword"
    system_prompt_contains: str | None = None
    input_keywords_any: list[str] = Field(default_factory=list)


class RuleOutput(BaseModel):
    """规则输出"""

    content: str = ""


class RuleMetrics(BaseModel):
    """规则度量指标"""

    accuracy: float = 0.0
    hit_count: int = 0
    cost_saved: float = 0.0


class RuleEntry(BaseModel):
    """单条固化规则"""

    id: str
    name: str = ""
    status: str = "active"
    match: RuleMatch = Field(default_factory=RuleMatch)
    output: RuleOutput = Field(default_factory=RuleOutput)
    metrics: RuleMetrics = Field(default_factory=RuleMetrics)


class RulesConfig(BaseModel):
    """规则配置"""

    rules: list[RuleEntry] = Field(default_factory=list)


class AppConfig(BaseModel):
    """应用总配置 — 聚合所有子配置"""

    settings: SettingsConfig = Field(default_factory=SettingsConfig)
    user_preferences: UserPreferencesConfig = Field(default_factory=UserPreferencesConfig)
    models_registry: ModelsRegistryConfig = Field(default_factory=ModelsRegistryConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)


# ─── 配置加载器 ─────────────────────────────────────────────────────


class ConfigLoader:
    """YAML配置加载器

    从指定目录加载所有YAML文件，支持热更新检测。
    配置文件名 → 配置模型的映射：
        - settings.yaml → SettingsConfig
        - user_preferences.yaml → UserPreferencesConfig
        - models_registry.yaml → ModelsRegistryConfig
        - rules.yaml → RulesConfig
    """

    # 文件名 → (Pydantic模型类, AppConfig中的字段名)
    _FILE_MAP: dict[str, tuple[type[BaseModel], str]] = {
        "settings.yaml": (SettingsConfig, "settings"),
        "user_preferences.yaml": (UserPreferencesWrapper, "user_preferences"),
        "models_registry.yaml": (ModelsRegistryConfig, "models_registry"),
        "rules.yaml": (RulesConfig, "rules"),
    }

    def __init__(self, config_dir: Path | str | None = None) -> None:
        self._config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
        self._config = AppConfig()
        self._mtimes: dict[str, float] = {}
        self._loaded = False

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    @property
    def config(self) -> AppConfig:
        """获取当前配置（如未加载则自动加载）"""
        if not self._loaded:
            self.load_all()
        return self._config

    def _read_yaml(self, filepath: Path) -> dict[str, Any]:
        """读取单个YAML文件"""
        if not filepath.exists():
            logger.warning("配置文件不存在: %s，使用默认值", filepath)
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    def _get_mtime(self, filepath: Path) -> float:
        """获取文件修改时间"""
        if not filepath.exists():
            return 0.0
        return filepath.stat().st_mtime

    def load_all(self) -> AppConfig:
        """加载所有配置文件"""
        logger.info("从 %s 加载配置文件...", self._config_dir)

        config_kwargs: dict[str, Any] = {}

        for filename, (model_cls, field_name) in self._FILE_MAP.items():
            filepath = self._config_dir / filename
            raw = self._read_yaml(filepath)

            # 记录修改时间
            self._mtimes[filename] = self._get_mtime(filepath)

            # user_preferences.yaml 有一层包装
            if filename == "user_preferences.yaml":
                parsed = model_cls.model_validate(raw)
                config_kwargs[field_name] = parsed.user_preferences
            else:
                parsed = model_cls.model_validate(raw)
                config_kwargs[field_name] = parsed

            logger.info("  ✓ %s 已加载", filename)

        self._config = AppConfig(**config_kwargs)
        self._loaded = True
        logger.info("配置加载完成")
        return self._config

    def check_updates(self) -> bool:
        """检查配置文件是否有更新

        Returns:
            True 如果有文件被修改
        """
        updated = False
        for filename in self._FILE_MAP:
            filepath = self._config_dir / filename
            current_mtime = self._get_mtime(filepath)
            previous_mtime = self._mtimes.get(filename, 0.0)
            if current_mtime > previous_mtime:
                logger.info("检测到配置文件变更: %s", filename)
                updated = True
        return updated

    def reload_if_updated(self) -> bool:
        """如果检测到更新则重新加载配置

        Returns:
            True 如果执行了重新加载
        """
        if self.check_updates():
            self.load_all()
            return True
        return False

    def get(self, key: str, default: Any = None) -> Any:
        """便捷方法：获取配置值

        支持点号分隔的嵌套路径，如 "settings.server.port"
        """
        if not self._loaded:
            self.load_all()

        obj = self._config
        for part in key.split("."):
            if isinstance(obj, BaseModel):
                obj = getattr(obj, part, None)
            elif isinstance(obj, dict):
                obj = obj.get(part)
            else:
                return default
            if obj is None:
                return default
        return obj
