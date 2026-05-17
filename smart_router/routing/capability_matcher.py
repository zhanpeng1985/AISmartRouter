"""Capability Matcher — 模型能力匹配器"""

import re

from smart_router.config.loader import ModelEntry


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文约1 token/字，其他约0.3 token/字符）"""
    count = 0.0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            count += 1.0
        else:
            count += 0.3
    return int(count)


def _extract_all_text(messages: list[dict]) -> str:
    """从消息列表中提取所有文本内容"""
    parts = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            # 处理多模态 content 列表
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if isinstance(text, str):
                        parts.append(text)
    return "\n".join(parts)


def identify_required_capabilities(messages: list[dict]) -> dict[str, int]:
    """
    从请求中推断所需能力维度及最低评分要求

    推断逻辑（基于system_prompt和user_content的关键词/特征）:
    - 含中文内容 → chinese_understanding >= 3
    - 含"分类"/"归类"/"识别" → information_extraction >= 3
    - 含代码块或编程相关 → code_generation >= 3
    - 含"分析"/"推理"/"为什么" → logical_reasoning >= 3
    - 含"写"/"创作"/"生成文案" → creative_writing >= 3
    - 含JSON/结构化输出要求 → structured_output >= 3
    - 含图片/image → multimodal >= 3
    - 输入token数 > 8000 → long_context >= 3
    """
    required: dict[str, int] = {}

    all_text = _extract_all_text(messages)
    all_text_lower = all_text.lower()

    # 含中文内容
    if re.search(r"[\u4e00-\u9fff]", all_text):
        required["chinese_understanding"] = 3

    # 含"分类"/"归类"/"识别"
    if any(kw in all_text for kw in ["分类", "归类", "识别"]):
        required["information_extraction"] = 3

    # 含代码块或编程相关
    code_keywords = [
        "python",
        "javascript",
        "java",
        "c++",
        "code",
        "编程",
        "代码",
        "函数",
        "class ",
        "def ",
    ]
    if "```" in all_text or any(kw in all_text_lower for kw in code_keywords):
        required["code_generation"] = 3

    # 含"分析"/"推理"/"为什么"
    if any(kw in all_text for kw in ["分析", "推理", "为什么", "原因", "逻辑"]):
        required["logical_reasoning"] = 3

    # 含"写"/"创作"/"生成文案"
    if any(kw in all_text for kw in ["写", "创作", "生成文案", "文案", "故事", "文章"]):
        required["creative_writing"] = 3

    # 含JSON/结构化输出要求
    structured_keywords = ["json", "结构化", "schema", "格式", "output format"]
    if any(kw in all_text_lower for kw in structured_keywords):
        required["structured_output"] = 3

    # 含图片/image
    has_image_keyword = any(
        kw in all_text_lower for kw in ["image", "图片", "图像", "photo", "vision"]
    )
    has_multimodal_content = any(
        isinstance(msg.get("content"), list) for msg in messages
    )
    if has_image_keyword or has_multimodal_content:
        required["multimodal"] = 3

    # 输入token数 > 8000
    estimated_tokens = _estimate_tokens(all_text)
    if estimated_tokens > 8000:
        required["long_context"] = 3

    return required


def match_models(models: dict[str, ModelEntry], required: dict[str, int]) -> list[str]:
    """筛选所有能力评分达标的模型"""
    matched = []

    for model_id, model in models.items():
        caps = model.capabilities
        meets_all = True

        for cap_name, min_score in required.items():
            actual_score = getattr(caps, cap_name, 0)
            if actual_score < min_score:
                meets_all = False
                break

        if meets_all:
            matched.append(model_id)

    return matched
