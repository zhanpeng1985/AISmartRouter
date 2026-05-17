"""Feature Extractor — 调用特征提取器

从AI调用请求/响应中提取结构化特征，用于：
- 日志记录的维度丰富
- 后续固化分析的输入
- 聚类与模式发现
"""

import hashlib
import json
import re
from collections import Counter

# ─── 中文常见词汇（用于简单分词） ────────────────────────────────────

_CN_COMMON_WORDS: set[str] = {
    # 保险领域
    "保险", "保单", "保费", "保额", "投保", "承保", "理赔", "续保",
    "豁免", "受益人", "被保人", "投保人", "受益", "责任", "免赔",
    "条款", "合同", "险种", "年金", "寿险", "重疾", "医疗", "意外",
    "健康", "养老", "教育", "分红", "万能", "投资", "账户", "现金价值",
    # 通用领域
    "分析", "总结", "提取", "生成", "翻译", "分类", "评估", "建议",
    "比较", "判断", "识别", "转换", "计算", "描述", "解释", "说明",
    "查询", "搜索", "匹配", "排序", "筛选", "统计", "预测", "推荐",
    "审核", "校验", "验证", "检查", "填写", "补全", "修正", "优化",
    # 数据与格式
    "数据", "信息", "内容", "格式", "结构", "字段", "属性", "类型",
    "文档", "文本", "表格", "列表", "对象", "数组", "配置", "参数",
}

# 英文停用词
_EN_STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "either", "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "if", "when", "where", "how",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
}

# 任务类型关键词映射
_TASK_TYPE_PATTERNS: dict[str, list[str]] = {
    "classification": [
        "分类", "归类", "类别", "category", "classify", "label",
        "判断类型", "属于哪类", "类型判断",
    ],
    "extraction": [
        "提取", "抽取", "解析", "extract", "parse", "抽取",
        "获取字段", "读取", "识别并提取",
    ],
    "generation": [
        "生成", "创作", "编写", "generate", "create", "write",
        "compose", "起草", "撰写",
    ],
    "analysis": [
        "分析", "评估", "诊断", "analyze", "evaluate", "assess",
        "比较", "对比", "审查", "审核",
    ],
    "translation": [
        "翻译", "转换语言", "translate", "translation",
        "中译英", "英译中", "转换为",
    ],
}


class FeatureExtractor:
    """从请求中提取特征，用于日志记录和后续固化分析

    所有方法均为静态方法，无状态，线程安全。
    """

    @staticmethod
    def extract_system_prompt_hash(messages: list[dict]) -> str:
        """提取system prompt并计算SHA256 hash

        从消息列表中找到role为system的消息，拼接后计算hash。

        Args:
            messages: OpenAI格式的消息列表

        Returns:
            SHA256 hash字符串（前16位），如无system prompt返回空字符串
        """
        system_parts: list[str] = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    system_parts.append(content)

        if not system_parts:
            return ""

        combined = "\n---\n".join(system_parts)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def extract_system_prompt_text(messages: list[dict]) -> str:
        """提取system prompt原文

        Args:
            messages: OpenAI格式的消息列表

        Returns:
            拼接后的system prompt文本
        """
        system_parts: list[str] = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if content:
                    system_parts.append(content)
        return "\n---\n".join(system_parts)

    @staticmethod
    def extract_user_content(messages: list[dict]) -> str:
        """提取用户消息内容

        Args:
            messages: OpenAI格式的消息列表

        Returns:
            拼接后的用户消息文本
        """
        user_parts: list[str] = []
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    if isinstance(content, str):
                        user_parts.append(content)
                    elif isinstance(content, list):
                        # 多模态消息格式
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                user_parts.append(part.get("text", ""))
                            elif isinstance(part, str):
                                user_parts.append(part)
        return "\n".join(user_parts)

    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> list[str]:
        """提取文本关键词（简单的中文/英文分词+频率）

        中文: 按常见词汇匹配（正向最大匹配）
        英文: 按空格分词 + stopwords过滤

        Args:
            text: 输入文本
            top_n: 返回前N个关键词

        Returns:
            关键词列表（按频率降序）
        """
        if not text:
            return []

        word_counter: Counter[str] = Counter()

        # 中文词汇匹配：正向最大匹配
        # 按词汇长度降序匹配，优先匹配长词
        sorted_cn_words = sorted(_CN_COMMON_WORDS, key=len, reverse=True)
        remaining = text
        while remaining:
            matched = False
            for word in sorted_cn_words:
                if remaining.startswith(word):
                    word_counter[word] += 1
                    remaining = remaining[len(word):]
                    matched = True
                    break
            if not matched:
                remaining = remaining[1:]  # 跳过非匹配字符

        # 英文分词
        en_words = re.findall(r"[a-zA-Z]{2,}", text.lower())
        for w in en_words:
            if w not in _EN_STOPWORDS and len(w) >= 2:
                word_counter[w] += 1

        return [word for word, _ in word_counter.most_common(top_n)]

    @staticmethod
    def detect_input_structure(content: str) -> str:
        """检测输入结构类型

        Args:
            content: 输入内容文本

        Returns:
            结构类型: plain_text / json / code / markdown / table
        """
        if not content or not content.strip():
            return "plain_text"

        stripped = content.strip()

        # JSON检测
        if stripped.startswith(("{", "[")):
            try:
                json.loads(stripped)
                return "json"
            except (json.JSONDecodeError, ValueError):
                pass

        # Markdown检测
        md_indicators = ["```", "##", "###", "- ", "* ", "1. ", "| ", "---"]
        md_hits = sum(1 for indicator in md_indicators if indicator in content)
        if md_hits >= 2:
            return "markdown"

        # 表格检测（竖线分隔的表格行）
        table_lines = [line for line in content.split("\n") if "|" in line and line.strip().startswith("|")]
        if len(table_lines) >= 2:
            return "table"

        # 代码检测
        code_indicators = [
            r"\bdef\s+\w+\s*\(",      # Python function
            r"\bfunction\s+\w+\s*\(",  # JS function
            r"\bclass\s+\w+",         # class definition
            r"\bimport\s+\w+",        # import statement
            r"\bfrom\s+\w+\s+import", # from import
            r"#[^\n]*$",              # single line comment
            r"//",                    # C-style comment
            r"\bif\s*\(.+\):",       # if statement
            r"\bfor\s*\(.+\)",       # for loop
            r"\breturn\s+",          # return statement
        ]
        code_hits = sum(1 for p in code_indicators if re.search(p, content, re.MULTILINE))
        if code_hits >= 2:
            return "code"

        return "plain_text"

    @staticmethod
    def detect_output_type(content: str) -> str:
        """检测输出类型

        Args:
            content: AI输出的内容文本

        Returns:
            输出类型: enum / json / text / code / number
        """
        if not content or not content.strip():
            return "text"

        stripped = content.strip()

        # 纯数字
        if re.match(r"^[+-]?\d+(\.\d+)?$", stripped):
            return "number"

        # 枚举型（短文本，通常为分类标签）
        if len(stripped) <= 20 and "\n" not in stripped:
            # 常见枚举值模式
            enum_patterns = [
                r"^(是|否|Yes|No|True|False|true|false)$",
                r"^(正面|负面|中性|积极|消极)$",
                r"^(高|中|低|严重|一般|轻微)$",
                r"^[A-Z][a-z]+(?:[-_][A-Z]?[a-z]+)*$",  # PascalCase或kebab-case
            ]
            for pattern in enum_patterns:
                if re.match(pattern, stripped):
                    return "enum"

        # JSON输出
        if stripped.startswith(("{", "[")):
            try:
                json.loads(stripped)
                return "json"
            except (json.JSONDecodeError, ValueError):
                pass

        # 代码输出
        if stripped.startswith("```"):
            return "code"
        code_indicators = ["def ", "class ", "import ", "function ", "const ", "let ", "var "]
        if any(stripped.startswith(ind) for ind in code_indicators):
            return "code"

        return "text"

    @staticmethod
    def extract_task_type(system_prompt: str) -> str:
        """从system prompt推断任务类型

        Args:
            system_prompt: system prompt文本

        Returns:
            任务类型: classification / extraction / generation / analysis / translation / other
        """
        if not system_prompt:
            return "other"

        prompt_lower = system_prompt.lower()

        # 按优先级匹配
        type_scores: dict[str, int] = {}
        for task_type, keywords in _TASK_TYPE_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in prompt_lower or kw in system_prompt)
            if score > 0:
                type_scores[task_type] = score

        if not type_scores:
            return "other"

        # 返回得分最高的类型
        best_type = max(type_scores, key=type_scores.get)  # type: ignore[arg-type]
        return best_type
