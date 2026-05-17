"""SmartRouter — FastAPI应用入口

启动流程:
1. 加载YAML配置
2. 初始化各子模块
3. 启动HTTP服务（兼容OpenAI API格式）
"""

import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from smart_router.config.loader import ConfigLoader
from smart_router.api import admin, completions, models, embeddings

# ─── 加载 .env 并处理 API Key 映射 ──────────────────────────────────

# 加载项目根目录的 .env 文件（如果存在）
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# LiteLLM API Key 映射：将中台格式的 Key 映射为标准环境变量名
if os.getenv("QWEN_API_KEY") and not os.getenv("DASHSCOPE_API_KEY"):
    os.environ["DASHSCOPE_API_KEY"] = os.getenv("QWEN_API_KEY")
    logging.getLogger("smart_router").info("已映射环境变量: QWEN_API_KEY → DASHSCOPE_API_KEY")

if os.getenv("ZHIPU_API_KEY") and not os.getenv("ZHIPUAI_API_KEY"):
    os.environ["ZHIPUAI_API_KEY"] = os.getenv("ZHIPU_API_KEY")
    logging.getLogger("smart_router").info("已映射环境变量: ZHIPU_API_KEY → ZHIPUAI_API_KEY")

# DEEPSEEK_API_KEY 名称与 LiteLLM 一致，无需映射

# ─── 日志配置 ───────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("smart_router")

# ─── 全局配置加载器 ─────────────────────────────────────────────────

config_loader: ConfigLoader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global config_loader

    # ── 启动阶段 ──
    logger.info("=" * 60)
    logger.info("SmartRouter 启动中...")
    logger.info("=" * 60)

    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.load_all()

    # 将配置存储到 app.state 供路由使用
    app.state.config_loader = config_loader
    app.state.config = config
    
    # 初始化 CallLogDB
    from smart_router.logger.models import CallLogDB
    db = CallLogDB()
    await db.init(config.settings.database.path)
    app.state.call_log_db = db
    logger.info("CallLogDB 已初始化")
    
    # 初始化 RuleStore
    from smart_router.solidification import rule_store as global_rs
    app.state.rule_store = global_rs
    logger.info("RuleStore 已绑定到 app.state")
    
    server_cfg = config.settings.server
    logger.info("服务地址: http://%s:%d", server_cfg.host, server_cfg.port)
    logger.info("调试模式: %s", server_cfg.debug)
    logger.info("数据库路径: %s", config.settings.database.path)
    logger.info("分析调度: %s", config.settings.analyzer.schedule)
    logger.info("已注册模型数: %d", len(config.models_registry.models))
    logger.info("固化规则数: %d", len(config.rules.rules))
    logger.info("SmartRouter 启动完成 ✓")
    logger.info("=" * 60)

    yield

    # ── 关闭阶段 ──
    # 关闭数据库连接
    db = getattr(app.state, "call_log_db", None)
    if db:
        await db.close()
    logger.info("SmartRouter 正在关闭...")


# ─── FastAPI 应用实例 ───────────────────────────────────────────────

app = FastAPI(
    title="SmartRouter",
    description="智能AI调用路由器 — 渐进式固化策略驱动的混合路由决策引擎",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── API路由注册 ────────────────────────────────────────────────────

app.include_router(admin.router)
app.include_router(completions.router)
app.include_router(models.router)
app.include_router(embeddings.router)


# ─── 健康检查 ───────────────────────────────────────────────────────


@app.get("/health", tags=["系统"])
async def health_check():
    """健康检查接口"""
    return {
        "status": "ok",
        "service": "smart-router",
        "version": "0.1.0",
    }


@app.get("/", tags=["系统"])
async def root():
    """根路径，返回服务基本信息"""
    return {
        "service": "SmartRouter",
        "description": "智能AI调用路由器",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


# ─── 主入口 ─────────────────────────────────────────────────────────


def main():
    """命令行入口"""
    # 先加载配置获取端口信息
    loader = ConfigLoader()
    config = loader.load_all()

    uvicorn.run(
        "smart_router.main:app",
        host=config.settings.server.host,
        port=config.settings.server.port,
        reload=config.settings.server.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
