# LangGraph + Ollama for CYSMIC

LangGraph is the orchestration framework. It provides:

1. **State Management** - Track conversation/context across agents
2. **Graph Structure** - Define workflow as nodes and edges
3. **Cyclic Execution** - Loops for reflection/retry (unlike DAGs)
4. **Human-in-the-loop** - Pause for approval at critical steps

## Architecture with Ollama

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph State                          │
│  { messages, context, current_step, data }                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Data Agent  │───▶│  ML Agent    │───▶│  LLM Agent   │ │
│  │ (ingest/     │    │ (forecast/   │    │ (reasoning/ │ │
│  │  validate)   │    │  predict)    │    │  explain)    │ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                   │                   │          │
│         └───────────────────┴───────────────────┘          │
│                         │                                  │
│                 ┌───────▼───────┐                          │
│                 │  Controller   │                          │
│                 │ (orchestrate) │                          │
│                 └───────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

## Why LangGraph?

| Feature | LangGraph | Celery | Airflow |
|---------|-----------|--------|---------|
| Native LLM support | ✅ | ❌ | ❌ |
| Cyclic workflows | ✅ | ❌ | ❌ |
| State persistence | ✅ | Partial | Partial |
| Human-in-loop | ✅ | ❌ | ✅ |
| Local/Ollama | ✅ | ✅ | ✅ |

## Setup

```bash
pip install langgraph langchain langchain-community
```
