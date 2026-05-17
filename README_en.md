English | [中文](./README.md)

> **AISmartRouter** — An open-source AI smart router that progressively solidifies your AI call patterns into zero-cost rules. If rules can handle it, don't call AI.

# AISmartRouter

An open-source AI smart router for individual users. Core philosophy: **If rules can handle it, don't call AI** — progressively solidify your AI usage patterns into zero-cost rules through a gradual solidification engine.

> One-liner: Turn your AI call history into zero-cost solidified rules — the more you use it, the more you save.

---

## Key Features

- **OpenAI-Compatible API** — Completely transparent to callers. Just change `base_url`, zero code migration cost.
- **Three-Layer Routing Decision** — User preference filtering → Capability dimension matching → Cost-performance ranking, selecting the optimal model for every call.
- **Progressive Solidification Engine** — Automatically discovers high-frequency, high-consistency call patterns and converts AI calls into zero-cost rule matches.
- **Call Logging & Analytics** — Fully records input features, routing decisions, response content, and costs for each call; offline analysis of solidification potential.
- **100+ Model Support** — Built on LiteLLM, supporting DeepSeek / Alibaba Qwen / Zhipu GLM / OpenAI / Anthropic out of the box.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Caller                                │
│         (Any OpenAI-compatible client: curl / Python / LangChain)  │
└─────────────────────────┬───────────────────────────────────┘
                          │  OpenAI API format
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  API Layer (FastAPI)                                        │
│  ├── /v1/chat/completions   Chat endpoint                   │
│  ├── /v1/embeddings         Embedding endpoint               │
│  ├── /v1/models             Available models list            │
│  └── /health                Health check                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────────────┐
│  Solidified Rule     │       │  Dynamic Routing Engine     │
│  Engine              │       │  (Three-layer Decision)     │
│  (RuleMatcher)      │       │  1. User preference filter   │
│  · Rule matching    │       │  2. Capability matching      │
│  · Zero-cost response│      │  3. Cost-performance ranking │
│  · Hit statistics   │       │                             │
└─────────┬───────────┘       └─────────────┬───────────────┘
          │                                 │
          │  No match                       │ Routing decision
          ▼                                 ▼
┌─────────────────────┐       ┌─────────────────────────────┐
│  Solidification     │       │  LiteLLM Execution Layer    │
│  Analyzer           │◄──────│  · Unified access to 100+   │
│  (Analyzer)         │ Call  │    models                   │
│  · Cluster analysis │ logs  │  · Auto-handle provider     │
│  · Pattern detection│       │    differences              │
│  · Rule generation  │       │  · Streaming / non-streaming│
│  · Backtest validation│     │    response                 │
└─────────────────────┘       └─────────────────────────────┘
```

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/zhanpeng1985/AISmartRouter.git
cd AISmartRouter

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Configure API Keys

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit the `.env` file and add at least one provider's key:

```bash
# DeepSeek (recommended, great cost-performance)
DEEPSEEK_API_KEY=sk-xxxxxxxx
DEEPSEEK_API_BASE=https://api.deepseek.com/v1

# Alibaba Qwen
QWEN_API_KEY=sk-xxxxxxxx
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

# Zhipu GLM
ZHIPU_API_KEY=xxxxxxxx
ZHIPU_API_BASE=https://open.bigmodel.cn/api/paas/v4

# OpenAI (optional)
# OPENAI_API_KEY=sk-xxxxxxxx

# Anthropic (optional)
# ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

> Built-in mapping: The project automatically maps `QWEN_API_KEY` to `DASHSCOPE_API_KEY` and `ZHIPU_API_KEY` to `ZHIPUAI_API_KEY` as required by LiteLLM, making configuration more convenient.

### Start the Service

```bash
# Option 1: Use the CLI entry point
smart-router

# Option 2: Run as a module
python -m smart_router.main

# Option 3: Use uvicorn (with custom parameters)
uvicorn smart_router.main:app --host 0.0.0.0 --port 8000 --reload
```

Default service URL: `http://localhost:8000`

### Verify

```bash
# Health check
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/v1/models

# Send a chat request (point your original OpenAI request to SmartRouter)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello, please introduce yourself"}]
  }'
```

> `model: "auto"` triggers the three-layer routing decision, automatically selecting the optimal model for you. You can also specify a concrete model like `deepseek-chat`, `qwen-plus`, etc.

---

## Configuration

Configuration files are located in the `config/` directory and support hot-reloading (call the refresh endpoint after modification — no restart required):

### Model Registry (`config/models_registry.yaml`)

Defines all available AI models, including provider, LiteLLM identifier, region, context window, pricing, and capability scores.

6 models are registered by default:

| Model | Provider | Context Window | Input $/M | Output $/M | Chinese Understanding |
|-------|----------|----------------|-----------|------------|----------------------|
| deepseek-chat | DeepSeek | 64K | 1.0 | 2.0 | 4.5 |
| qwen-plus | Alibaba | 128K | 0.8 | 2.0 | 4.5 |
| qwen-turbo | Alibaba | 128K | 0.3 | 0.6 | 4.0 |
| glm-4-flash | Zhipu | 128K | 0.1 | 0.1 | 4.0 |
| gpt-4o-mini | OpenAI | 128K | 0.15 | 0.6 | 3.5 |
| claude-3-haiku | Anthropic | 200K | 0.25 | 1.25 | 3.0 |

Capability score dimensions: `chinese_understanding` / `instruction_following` / `logical_reasoning` / `information_extraction` / `code_generation` / `creative_writing` / `long_context` / `structured_output` / `multimodal`

### User Preferences (`config/user_preferences.yaml`)

Configure provider priority, budget limits, quality tier preferences, etc. — influences the first routing decision layer.

### Solidification Rules (`config/rules.yaml`)

The zero-cost rule library. When request features match a rule, a preset response is returned directly — no AI call needed.

### Global Settings (`config/settings.yaml`)

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  debug: false

database:
  path: "data/call_logs.db"    # SQLite call log path

analyzer:
  schedule: "daily"            # Analysis schedule: daily / hourly / manual
  min_call_count: 20           # Minimum call count to trigger analysis
  min_consistency: 0.85        # Pattern consistency threshold

solidification:
  auto_deploy_threshold: 0.98  # Auto-deploy accuracy threshold
  manual_review_threshold: 0.95 # Manual review accuracy threshold
```

---

## How the Solidification Engine Works

SmartRouter's differentiating capability is **progressive solidification** — gradually converting high-frequency AI calls into zero-cost rules:

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  Collect │ → │ Cluster │ → │ Pattern │ → │ Generate│ → │ Backtest│ → │  Deploy │
│(CallLog)│   │(Cluster)│   │(Pattern)│   │(Rule)   │   │(Validate│   │(Deploy) │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

1. **Collect** — Every AI call is logged to SQLite, including input features, routing decisions, response content, token consumption, and cost.
2. **Cluster** — Group calls by input content features (system prompt + user prompt) to discover similar call groups.
3. **Pattern Detection** — Identify high-frequency, highly consistent call patterns (consistency ≥ 85%).
4. **Rule Generation** — Automatically generate solidification rules linking match conditions to standard responses.
5. **Backtest Validation** — Verify rule accuracy on historical data (accuracy ≥ 95% and coverage ≥ 70% before deployment is recommended).
6. **Deploy** — Deploy rules via the admin interface; subsequent matching requests are served at zero cost.

> Over time, more and more rules get solidified, AI call costs keep dropping — truly achieving **the more you use it, the more you save**.

### Trigger Solidification Analysis Manually

```bash
curl -X POST http://localhost:8000/admin/analyze
```

### Deploy a Candidate Rule

```bash
curl -X POST http://localhost:8000/admin/analyze/deploy/{rule_id}
```

---

## API Endpoints

### Chat & Models

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | OpenAI-compatible chat endpoint, supports streaming/non-streaming |
| `/v1/embeddings` | POST | Embedding endpoint |
| `/v1/models` | GET | Available models list |

### Admin

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/rules` | GET | View all solidification rules |
| `/admin/rules/{rule_id}` | GET | View a single rule's details |
| `/admin/rules` | POST | Add a new rule |
| `/admin/rules/{rule_id}/disable` | PUT | Disable a rule |
| `/admin/stats` | GET | Call log statistics (total calls, total cost, rule hit rate, etc.) |
| `/admin/analyze` | POST | Trigger solidification analysis |
| `/admin/analyze/deploy/{rule_id}` | POST | Deploy a candidate rule |

### System

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/docs` | GET | Auto-generated Swagger API documentation |

---

## Relationship with LiteLLM

AISmartRouter is **not** a replacement for LiteLLM — it's a **decision layer** built on top of it.

| Aspect | AISmartRouter | LiteLLM |
|--------|--------------|---------|
| Responsibility | Decision: Should AI be called? Which model? Can a rule replace it? | Execution: Unified interface to 100+ models, load balancing, failover |
| Core Value | Cost optimization + Progressive solidification | Provider unification + Reliability |
| Analogy | The brain (judgment) | The hands (execution) |

**Design Philosophy**:
- LiteLLM solves "how to call AI models" (unified interface, load balancing, failover)
- AISmartRouter solves "whether AI needs to be called" and "which model offers the best value"
- Together = guaranteed call reliability **and** continuously reduced call necessity

**Integration**: AISmartRouter imports LiteLLM as a pip dependency (`import litellm`), never modifying LiteLLM source code — only calling through standard APIs. LiteLLM is MIT-licensed, fully compliant.

---

## Tech Stack

- **Python 3.11+** / **FastAPI** — High-performance async web framework
- **LiteLLM** — AI execution layer, unified access to 100+ models
- **SQLite** + **aiosqlite** — Async call log storage
- **YAML** — Configuration management (with hot-reload support)
- **APScheduler** — Scheduled task execution (solidification analysis)
- **Pydantic v2** — Data validation and serialization

---

## Project Structure

```
AISmartRouter/
├── smart_router/
│   ├── api/              # OpenAI-compatible API endpoints
│   │   ├── completions.py   # /v1/chat/completions
│   │   ├── embeddings.py    # /v1/embeddings
│   │   ├── models.py        # /v1/models
│   │   └── admin.py         # Admin API endpoints
│   ├── routing/          # Dynamic routing engine
│   │   └── router.py        # Three-layer decision implementation
│   ├── solidification/   # Solidification rule engine
│   │   ├── rule_matcher.py  # Rule matching
│   │   ├── rule_store.py    # Rule storage
│   │   └── response_builder.py # Response building
│   ├── analyzer/         # Pattern analysis & automatic rule generation
│   │   └── scheduler.py     # Solidification analysis scheduler
│   ├── logger/           # Call recording & feature extraction
│   │   ├── models.py        # CallLogDB data model
│   │   └── call_recorder.py # Call recorder
│   ├── config/           # Configuration loader
│   │   └── loader.py        # YAML config hot-reload
│   └── main.py           # FastAPI application entry point
├── config/
│   ├── models_registry.yaml   # Model registry
│   ├── user_preferences.yaml  # User preferences
│   ├── rules.yaml             # Solidification rule library
│   └── settings.yaml          # Global settings
├── tests/                # Test cases
├── pyproject.toml        # Project configuration & dependencies
├── .env.example          # Environment variable example
└── README.md             # This document
```

---

## Core Philosophy

### Progressive Solidification

High-frequency, high-consistency AI call patterns go through **cluster analysis → pattern detection → rule generation → backtest validation**, gradually solidifying into zero-cost rule matches.

This isn't about replacing AI — it's about using rules to eliminate repetitive costs in scenarios where AI has already proven it can handle things stably. This is a **continuously evolving** process:

- Day 1: 100% AI calls, collecting data
- Day 7: 10% of requests hit solidified rules
- Day 30: 40% of requests hit solidified rules
- Day 90: 70% of requests hit solidified rules

**The more you use it, the more you save** — the cost curve keeps dropping.

---

## License

[MIT](LICENSE)
