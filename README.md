# CYSMIC Fuel Forecasting

Agentic AI framework for fuel demand forecasting in Kenya's oil & gas retail sector.

## Overview

This project implements the framework described in the EPRA 2026 Research Paper: *"Fuel Demand Forecasting in Kenya's Oil & Gas Retail Sector: A Framework for Agentic Orchestration"*

## Quick Start (Demo Mode)

```bash
# Clone the repo
git clone https://github.com/project-merlin-00/cysmic-fuel-forecasting.git
cd cysmic-fuel-forecasting

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run demo (synthetic data)
python3 demo.py
```

## Demo Features

The demo showcases:
- **Synthetic data generation** - 4 stations, 3 products, 90 days
- **Demand forecasting** - 7-day predictions with confidence intervals
- **Inventory management** - Stock level monitoring with alerts
- **Natural language queries** - Simulated LLM responses
- **Export** - CSV/JSON export of forecasts

### Demo Commands

```bash
# Generate sample data
python3 demo.py data

# Run forecast
python3 demo.py forecast NBO001 diesel

# Check inventory
python3 demo.py inventory NBO001

# Ask question
python3 demo.py ask "Should I order more stock?"

# Export
python3 demo.py export csv
```

## Full System (With ML + LLM)

For the full agentic system with real ML models and LLM integration:

```bash
# Install full dependencies
pip install -r requirements-full.txt

# Start Ollama (for local LLM)
ollama serve
ollama pull llama2

# Run the full system
python3 -m src.orchestration.langgraph_ollama
```

### Full System Requirements

- Python 3.10+
- Ollama (https://ollama.ai)
- 8GB+ RAM for ML models

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

## Project Structure

- `src/data/` - Data ingestion and preprocessing
- `src/models/` - ML model implementations (Prophet, LSTM, XGBoost)
- `src/agents/` - Agentic AI components (LLM agents)
- `src/orchestration/` - Workflow orchestration (LangGraph)
- `docs/` - Architecture documentation
- `notebooks/` - Jupyter notebooks for analysis
- `demo.py` - Demo with synthetic data
- `requirements.txt` - Demo dependencies
- `requirements-full.txt` - Full system dependencies

## Demo Output Example

```
╔════════════════════════════════════════════════════════════╗
║     🔮 CYSMIC Fuel Forecasting - CLI Demo                 ║
║        Agentic AI for Kenya Oil & Gas Retail              ║
╚════════════════════════════════════════════════════════════╝

📊 Forecast: NBO001 - Diesel

| Date       | Day       | Volume (L) | Confidence |
|------------|-----------|------------|------------|
| 2026-02-16 | Monday    |     12,344 |        87% |
| 2026-02-17 | Tuesday   |     12,567 |        84% |
| ...

📦 Inventory: NBO001

| Product   | Level (L)  | Status   |
|-----------|-------------|----------|
| Super     |      22,470 | 🟢 OK     |
| Diesel    |      16,533 | 🟢 OK     |
| Kerosene  |       8,824 | 🟡 LOW    |

🤖 Analysis: Recommend ordering diesel within 48 hours
```

## Paper

**EPRA 2026 Research Paper:**
https://drive.google.com/file/d/1pT2cxN8z1U4zjMzfoX2CO7103j2QONmw

**GitHub:**
https://github.com/project-merlin-00/cysmic-fuel-forecasting

## License

MIT

## Author

Arthur Kisaka - CYSMIC Initiative
