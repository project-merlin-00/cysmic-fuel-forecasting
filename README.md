# CYSMIC Fuel Forecasting

Agentic AI framework for fuel demand forecasting in Kenya's oil & gas retail sector.

## Overview

This project implements the framework described in the EPRA 2026 Research Paper: *"Fuel Demand Forecasting in Kenya's Oil & Gas Retail Sector: A Framework for Agentic Orchestration"*

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                     │
│                    (Controller Agent)                       │
├─────────────────────────────────────────────────────────────┤
│  ML Ensemble (Prophet, LSTM, XGBoost)  │  LLM Engine      │
├─────────────────────────────────────────────────────────────┤
│                    DATA LAYER                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

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
