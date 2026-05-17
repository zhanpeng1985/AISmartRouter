"""Admin API — 管理后台接口

提供固化规则的 CRUD 与管理端点：
- GET  /admin/rules            — 查看所有规则
- GET  /admin/rules/{rule_id}  — 查看单个规则详情
- POST /admin/rules            — 添加新规则
- PUT  /admin/rules/{rule_id}/disable — 停用规则
- POST /admin/analyze          — 手工触发固化分析
- POST /admin/analyze/deploy/{rule_id} — 部署候选规则
- GET  /admin/stats            — 调用日志统计
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from smart_router.solidification import rule_store as global_rule_store
from smart_router.solidification.rule_store import (
    RuleMatch,
    RuleMetrics,
    RuleOutput,
    SolidificationRule,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin")


# ─── 请求体模型 ───────────────────────────────────────────────────────


class RuleCreateRequest(BaseModel):
    """创建规则请求体"""

    id: str
    name: str
    status: str = "active"
    match: RuleMatch
    output: RuleOutput
    metrics: RuleMetrics = Field(default_factory=RuleMetrics)


# ─── 辅助函数 ─────────────────────────────────────────────────────────


def _get_rule_store(request: Request):
    """获取 RuleStore 实例（优先从 app.state 获取，否则回退到全局单例）"""
    store = getattr(request.app.state, "rule_store", None)
    if store is None:
        store = global_rule_store.rule_store
    return store


def _get_call_log_db(request: Request):
    """获取 CallLogDB 实例（从 app.state 获取）"""
    db = getattr(request.app.state, "call_log_db", None)
    return db


def _get_config(request: Request):
    """获取应用配置"""
    config = getattr(request.app.state, "config", None)
    return config


# ─── 端点：规则 CRUD ──────────────────────────────────────────────────


@router.get("/rules")
async def list_rules(request: Request):
    """查看所有规则（含 active / disabled / draft）"""
    store = _get_rule_store(request)
    rules = store.get_all_rules()
    return {
        "count": len(rules),
        "rules": [r.model_dump() for r in rules],
    }


@router.get("/rules/{rule_id}")
async def get_rule(rule_id: str, request: Request):
    """查看单个规则详情"""
    store = _get_rule_store(request)
    rule = store.get_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="规则不存在")
    return rule.model_dump()


@router.post("/rules")
async def create_rule(req: RuleCreateRequest, request: Request):
    """添加新规则"""
    store = _get_rule_store(request)

    if store.get_rule(req.id):
        raise HTTPException(status_code=409, detail="规则 ID 已存在")

    rule = SolidificationRule(
        id=req.id,
        name=req.name,
        status=req.status,
        match=req.match,
        output=req.output,
        metrics=req.metrics,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    await store.add_rule(rule)
    return rule.model_dump()


@router.put("/rules/{rule_id}/disable")
async def disable_rule(rule_id: str, request: Request):
    """停用规则"""
    store = _get_rule_store(request)
    rule = await store.disable_rule(rule_id)
    if not rule:
        raise HTTPException(status_code=404, detail="规则不存在")
    return rule.model_dump()


# ─── 端点：固化分析 ───────────────────────────────────────────────────


@router.post("/analyze")
async def run_analysis(request: Request):
    """手工触发固化分析，返回分析报告

    流程: 聚类分析 → 模式检测 → 规则生成 → 回测验证
    """
    from smart_router.analyzer.scheduler import SolidificationAnalyzer
    from smart_router.logger.models import CallLogDB

    db = _get_call_log_db(request)
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="CallLogDB 未初始化，请检查数据库配置",
        )

    config = _get_config(request)
    analyzer_config = config.settings.analyzer if config else None

    rule_store = _get_rule_store(request)
    analyzer = SolidificationAnalyzer(db, rule_store, analyzer_config)

    try:
        report = await analyzer.run_analysis()
        return report
    except Exception as e:
        logger.error("固化分析执行失败: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"固化分析执行失败: {str(e)}",
        )


@router.post("/analyze/deploy/{rule_id}")
async def deploy_rule(rule_id: str, request: Request):
    """部署候选规则（将draft规则变为active）

    先触发分析获取候选规则，然后根据rule_id部署指定规则。
    """
    from smart_router.analyzer.scheduler import SolidificationAnalyzer
    from smart_router.logger.models import CallLogDB

    db = _get_call_log_db(request)
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="CallLogDB 未初始化，请检查数据库配置",
        )

    config = _get_config(request)
    analyzer_config = config.settings.analyzer if config else None

    rule_store = _get_rule_store(request)

    # 先检查规则是否已存在于store中（可能是之前分析生成的draft规则）
    existing_rule = rule_store.get_rule(rule_id)
    if existing_rule is not None:
        if existing_rule.status == "active":
            return {
                "status": "already_active",
                "rule": existing_rule.model_dump(),
                "message": f"规则 {rule_id} 已经是活跃状态",
            }

        # 将draft规则改为active
        existing_rule.status = "active"
        # 重新添加会因ID冲突失败，所以直接持久化
        from smart_router.solidification.rule_store import RuleStore

        async with rule_store._lock:
            rule_store._active_rules = rule_store._build_active_rules()
            rule_store._save_to_file()
        logger.info("规则已部署: %s (%s)", rule_id, existing_rule.name)
        return {
            "status": "deployed",
            "rule": existing_rule.model_dump(),
        }

    # 如果规则不在store中，需要先运行分析获取候选规则
    analyzer = SolidificationAnalyzer(db, rule_store, analyzer_config)
    try:
        report = await analyzer.run_analysis()
    except Exception as e:
        logger.error("固化分析执行失败: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"固化分析执行失败: {str(e)}",
        )

    # 在候选规则中查找目标
    from pydantic import TypeAdapter

    candidate_rules = [
        SolidificationRule.model_validate(r)
        for r in report.get("candidate_rules", [])
    ]
    target_rule = None
    for r in candidate_rules:
        if r.id == rule_id:
            target_rule = r
            break

    if target_rule is None:
        raise HTTPException(
            status_code=404,
            detail=f"未找到规则 {rule_id}，请先运行分析确认规则存在",
        )

    # 检查验证结果
    ready_ids = {
        v["rule_id"]
        for v in report.get("validation_results", [])
        if v.get("ready_to_deploy")
    }
    if rule_id not in ready_ids:
        raise HTTPException(
            status_code=400,
            detail=f"规则 {rule_id} 未通过验证标准（accuracy>=95%, coverage>=70%），"
            "不建议部署",
        )

    # 部署规则
    try:
        deployed = await analyzer.deploy_rule(target_rule)
        return {
            "status": "deployed",
            "rule": deployed.model_dump(),
        }
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/stats")
async def get_stats(request: Request):
    """调用日志统计

    返回全局统计数据，包含总调用次数、总成本、规则命中率等。
    """
    from smart_router.logger.call_recorder import CallRecorder

    db = _get_call_log_db(request)
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="CallLogDB 未初始化，请检查数据库配置",
        )

    recorder = CallRecorder(db)
    try:
        stats = await recorder.get_stats()
        return stats
    except Exception as e:
        logger.error("获取统计数据失败: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"获取统计数据失败: {str(e)}",
        )
