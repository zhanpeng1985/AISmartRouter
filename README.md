# AISmartRouter

面向个人用户的开源AI智能路由器。核心理念：**能用规则实现的就不用AI**，通过渐进式固化引擎，持续降低AI使用成本。

> 一句话定位：把你对AI的调用历史变成零成本的固化规则，越用越省。

---

## 核心特性

- **OpenAI兼容API** — 对调用方完全透明，只需修改`base_url`，零代码迁移成本
- **三层路由决策** — 用户偏好过滤 → 能力维度匹配 → 成本性价比排序，为每次调用选择最优模型
- **渐进式固化引擎** — 自动发现高频高一致性调用模式，将AI调用转化为零成本规则命中
- **调用日志与分析** — 完整记录每次调用的输入特征、路由决策、响应内容与成本，离线分析固化潜力
- **100+模型支持** — 基于LiteLLM，开箱支持DeepSeek / 阿里通义千问 / 智谱GLM / OpenAI / Anthropic等

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                        调用方                                │
│         (任意OpenAI兼容客户端: curl / Python / LangChain)     │
└─────────────────────────┬───────────────────────────────────┘
                          │  OpenAI API格式
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  API层 (FastAPI)                                            │
│  ├── /v1/chat/completions   对话接口                          │
│  ├── /v1/embeddings         Embedding接口                    │
│  ├── /v1/models             可用模型列表                      │
│  └── /health                健康检查                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────────────┐
│  固化规则引擎         │       │  动态路由引擎                 │
│  (RuleMatcher)      │       │  (三层决策)                   │
│  · 规则匹配          │       │  1. 用户偏好过滤              │
│  · 零成本响应        │       │  2. 能力维度匹配              │
│  · 命中统计          │       │  3. 成本性价比排序             │
└─────────┬───────────┘       └─────────────┬───────────────┘
          │                                 │
          │  未命中                         │ 路由决策
          ▼                                 ▼
┌─────────────────────┐       ┌─────────────────────────────┐
│  固化分析器          │       │  LiteLLM执行层               │
│  (Analyzer)         │◄──────│  · 统一调用100+模型           │
│  · 聚类分析          │  调用日志 │  · 自动处理provider差异       │
│  · 模式检测          │       │  · 流式/非流式响应            │
│  · 规则生成          │       │                             │
│  · 回测验证          │       │                             │
└─────────────────────┘       └─────────────────────────────┘
```

---

## 快速启动

### 安装

```bash
# 克隆仓库
git clone https://github.com/zhanpeng1985/AISmartRouter.git
cd AISmartRouter

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -e ".[dev]"
```

### 配置API Key

复制示例环境变量文件并填入你的API Key：

```bash
cp .env.example .env
```

编辑`.env`文件，填入至少一个提供商的Key：

```bash
# DeepSeek（推荐，性价比高）
DEEPSEEK_API_KEY=sk-xxxxxxxx
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# 阿里通义千问
QWEN_API_KEY=sk-xxxxxxxx
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# 智谱GLM
ZHIPU_API_KEY=xxxxxxxx
ZHIPU_API_BASE=https://open.bigmodel.cn/api/paas/v4

# OpenAI（可选）
# OPENAI_API_KEY=sk-xxxxxxxx

# Anthropic（可选）
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

> 内置映射：项目自动将`QWEN_API_KEY`映射为LiteLLM所需的`DASHSCOPE_API_KEY`，`ZHIPU_API_KEY`映射为`ZHIPUAI_API_KEY`，使用更便捷。

### 启动服务

```bash
# 方式一：使用命令行入口
smart-router

# 方式二：直接运行模块
python -m smart_router.main

# 方式三：使用uvicorn（可自定义参数）
uvicorn smart_router.main:app --host 0.0.0.0 --port 8000 --reload
```

默认服务地址：`http://localhost:8000`

### 验证

```bash
# 健康检查
curl http://localhost:8000/health

# 查看可用模型列表
curl http://localhost:8000/v1/models

# 发起对话（将原本对OpenAI的请求指向SmartRouter）
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}]
  }'
```

> `model: "auto"` 会触发三层路由决策，自动为你选择最优模型。也可以指定具体模型如`deepseek-chat`、`qwen-plus`等。

---

## 配置说明

配置文件位于 `config/` 目录，支持热更新（修改后调用刷新接口即可生效，无需重启）：

### 模型注册表 (`config/models_registry.yaml`)

定义所有可用的AI模型，包含Provider、LiteLLM标识、区域、上下文窗口、定价、能力评分等。

默认已注册6款模型：

| 模型 | Provider | 上下文窗口 | 输入$/M | 输出$/M | 中文理解 |
|------|----------|-----------|--------|--------|----------|
| deepseek-chat | DeepSeek | 64K | 1.0 | 2.0 | 4.5 |
| qwen-plus | 阿里 | 128K | 0.8 | 2.0 | 4.5 |
| qwen-turbo | 阿里 | 128K | 0.3 | 0.6 | 4.0 |
| glm-4-flash | 智谱 | 128K | 0.1 | 0.1 | 4.0 |
| gpt-4o-mini | OpenAI | 128K | 0.15 | 0.6 | 3.5 |
| claude-3-haiku | Anthropic | 200K | 0.25 | 1.25 | 3.0 |

能力评分维度：`chinese_understanding` / `instruction_following` / `logical_reasoning` / `information_extraction` / `code_generation` / `creative_writing` / `long_context` / `structured_output` / `multimodal`

### 用户偏好 (`config/user_preferences.yaml`)

配置Provider优先级、预算上限、质量层级偏好等，影响第一层路由决策。

### 固化规则 (`config/rules.yaml`)

零成本规则库。当请求特征匹配规则时，直接返回预设响应，无需调用AI。

### 全局设置 (`config/settings.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  debug: false

database:
  path: "data/call_logs.db"    # SQLite调用日志路径

analyzer:
  schedule: "daily"            # 分析调度周期: daily / hourly / manual
  min_call_count: 20           # 触发分析的最小调用次数
  min_consistency: 0.85        # 模式一致性阈值

solidification:
  auto_deploy_threshold: 0.98  # 自动部署准确率阈值
  manual_review_threshold: 0.95 # 人工审核准确率阈值
```

---

## 固化引擎工作原理

SmartRouter的差异化能力在于**渐进式固化**，将高频AI调用逐步转化为零成本规则：

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  采集    │ → │  聚类    │ → │ 模式识别 │ → │ 规则生成 │ → │ 回测验证 │ → │  上线    │
│(CallLog)│   │(Cluster)│   │(Pattern)│   │(Generate│   │(Validate│   │(Deploy) │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

1. **采集** — 每次AI调用都被记录到SQLite，包括输入特征、路由决策、响应内容、token消耗、成本
2. **聚类** — 按输入内容特征（system prompt + user prompt）进行聚类，发现相似调用组
3. **模式识别** — 识别高频且响应高度一致的调用模式（一致性≥85%）
4. **规则生成** — 自动生成固化规则，将匹配条件与标准响应关联
5. **回测验证** — 在历史数据上验证规则准确率（accuracy≥95%且coverage≥70%才建议部署）
6. **上线** — 通过管理接口部署规则，后续同类请求直接零成本命中

> 随着时间推移，固化的规则越来越多，AI调用成本持续下降，真正做到**越用越省**。

### 手动触发固化分析

```bash
curl -X POST http://localhost:8000/admin/analyze
```

### 部署候选规则

```bash
curl -X POST http://localhost:8000/admin/analyze/deploy/{rule_id}
```

---

## API端点

### 对话与模型

| 端点 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | OpenAI兼容对话接口，支持流式/非流式 |
| `/v1/embeddings` | POST | Embedding接口 |
| `/v1/models` | GET | 可用模型列表 |

### 管理后台

| 端点 | 方法 | 说明 |
|------|------|------|
| `/admin/rules` | GET | 查看所有固化规则 |
| `/admin/rules/{rule_id}` | GET | 查看单个规则详情 |
| `/admin/rules` | POST | 添加新规则 |
| `/admin/rules/{rule_id}/disable` | PUT | 停用规则 |
| `/admin/stats` | GET | 调用日志统计（总次数、总成本、规则命中率等） |
| `/admin/analyze` | POST | 触发固化分析 |
| `/admin/analyze/deploy/{rule_id}` | POST | 部署候选规则 |

### 系统

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/docs` | GET | 自动生成的Swagger API文档 |

---

## 技术栈

- **Python 3.11+** / **FastAPI** — 高性能异步Web框架
- **LiteLLM** — AI执行底座，统一调用100+模型
- **SQLite** + **aiosqlite** — 异步调用日志存储
- **YAML** — 配置管理（支持热更新）
- **APScheduler** — 定时任务调度（固化分析）
- **Pydantic v2** — 数据校验与序列化

---

## 项目结构

```
AISmartRouter/
├── smart_router/
│   ├── api/              # OpenAI兼容API端点
│   │   ├── completions.py   # /v1/chat/completions
│   │   ├── embeddings.py    # /v1/embeddings
│   │   ├── models.py        # /v1/models
│   │   └── admin.py         # 管理后台接口
│   ├── routing/          # 动态路由引擎
│   │   └── router.py        # 三层决策实现
│   ├── solidification/   # 固化规则引擎
│   │   ├── rule_matcher.py  # 规则匹配
│   │   ├── rule_store.py    # 规则存储
│   │   └── response_builder.py # 响应构建
│   ├── analyzer/         # 模式分析与规则自动生成
│   │   └── scheduler.py     # 固化分析调度器
│   ├── logger/           # 调用记录与特征提取
│   │   ├── models.py        # CallLogDB数据模型
│   │   └── call_recorder.py # 调用记录器
│   ├── config/           # 配置加载器
│   │   └── loader.py        # YAML配置热加载
│   └── main.py           # FastAPI应用入口
├── config/
│   ├── models_registry.yaml   # 模型注册表
│   ├── user_preferences.yaml  # 用户偏好
│   ├── rules.yaml             # 固化规则库
│   └── settings.yaml          # 全局设置
├── tests/                # 测试用例
├── pyproject.toml        # 项目配置与依赖
├── .env.example          # 环境变量示例
└── README.md             # 本文档
```

---

## 核心理念

### 渐进式固化

高频且高一致性的AI调用模式，经过**聚类分析 → 模式检测 → 规则生成 → 回测验证**，逐步固化为零成本的规则匹配。

不是替代AI，而是在AI已经证明可以稳定处理的场景上，用规则消除重复成本。这是一个**持续演进**的过程：

- 第1天：100% AI调用，收集数据
- 第7天：10% 请求命中固化规则
- 第30天：40% 请求命中固化规则
- 第90天：70% 请求命中固化规则

**越用越省**，成本曲线持续下降。

---

## License

[MIT](LICENSE)
