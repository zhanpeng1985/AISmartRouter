"""Logger — 调用日志采集模块

提供AI调用的异步日志记录、特征提取和统计查询功能。
"""

from smart_router.logger.call_recorder import CallRecorder
from smart_router.logger.feature_extractor import FeatureExtractor
from smart_router.logger.models import CallLog, CallLogDB

__all__ = [
    "CallLog",
    "CallLogDB",
    "CallRecorder",
    "FeatureExtractor",
]


async def get_stats(recorder: CallRecorder) -> dict:
    """获取统计数据（供后续 api/admin.py 使用）

    Args:
        recorder: CallRecorder实例

    Returns:
        统计数据字典
    """
    return await recorder.get_stats()


async def get_clusters(recorder: CallRecorder) -> list[dict]:
    """获取prompt聚类数据（供后续 api/admin.py 使用）

    Args:
        recorder: CallRecorder实例

    Returns:
        聚类列表
    """
    return await recorder.get_clusters()
