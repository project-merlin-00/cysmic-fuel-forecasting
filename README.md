# CYSMIC Fuel Forecasting

Agentic AI framework for fuel demand forecasting in Kenya's oil & gas retail sector.

## Overview

This project implements the framework described in the EPRA 2026 Research Paper: *"Fuel Demand Forecasting in Kenya's Oil & Gas Retail Sector: A Framework for Agentic Orchestration"*

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                     │
│                    (Controller Agent - LangGraph)           │
├─────────────────────────────────────────────────────────────┤
│  Data Agent  │  ML Ensemble (Prophet, LSTM, XGBoost)  │  LLM Engine (Ollama) │
├─────────────────────────────────────────────────────────────┤
│                    DATA LAYER                               │
└─────────────────────────────────────────────────────────────┘
```

For detailed diagrams, see [docs/architecture.md](docs/architecture.md) (Mermaid.js)

### Generate Professional Diagrams

**Option 1: Mermaid.js** (in docs/architecture.md)
- Renders in GitHub, VS Code, Obsidian
- Copy to [mermaid.live](https://mermaid.live) for PNG/SVG export

**Option 2: Excalidraw** 
- Open https://excalidraw.com
- Use prompts from `skills/diagram-generator/prompts.md`

**Option 3: draw.io**
- Open https://app.diagrams.net
- Professional diagrams with extensive icon library

- `src/data/` - Data ingestion and preprocessing
- `src/models/` - ML model implementations
- `src/agents/` - Agentic AI components
- `src/orchestration/` - Workflow orchestration
- `docs/` - Architecture documentation
- `notebooks/` - Jupyter notebooks for analysis

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run data ingestion
python -m src.data.ingest

# Train models
python -m src.models.train

# Start orchestration
python -m src.orchestration.run
```

## Paper

https://drive.google.com/file/d/1AMzUFT4JXMtWpxvqfY_igFk--91kn_V6/view

## License

MIT

## Author

Arthur Kisaka - CYSMIC Initiative
