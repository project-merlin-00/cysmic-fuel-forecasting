# CYSMIC Fuel Forecasting - Architecture Diagrams

> These diagrams use Mermaid.js. Render in GitHub, VS Code, or [Mermaid Live Editor](https://mermaid.live)

---

## System Architecture Overview

```mermaid
flowchart TB
    subgraph "Presentation Layer"
        CLI[CLI/TUI Interface]
        API[REST API]
        Dashboard[Dashboard]
    end
    
    subgraph "Orchestration Layer"
        Controller[Controller Agent<br/>LangGraph]
        Scheduler[Workflow Scheduler]
    end
    
    subgraph "Intelligence Layer"
        ML[ML Ensemble<br/>Prophet, LSTM, XGBoost]
        LLM[LLM Agent<br/>Ollama/LangChain]
    end
    
    subgraph "Data Layer"
        POS[POS Data]
        Weather[Weather API]
        Events[Events/Calendar]
        Prices[Fuel Prices]
        Stations[Station Metadata]
    end
    
    CLI --> Controller
    API --> Controller
    Dashboard --> Controller
    
    Controller --> ML
    Controller --> LLM
    Controller --> Scheduler
    
    ML --> POS
    ML --> Weather
    ML --> Prices
    
    LLM --> Events
    LLM --> Stations
```

---

## Agentic Workflow (LangGraph)

```mermaid
flowchart LR
    subgraph "LangGraph State"
        State[messages, context,<br/>task, result]
    end
    
    A[Data Agent] --> B[ML Agent]
    B --> C[LLM Agent]
    C --> D[End]
    
    A -.->|updates| State
    B -.->|updates| State
    C -.->|updates| State
```

---

## Data Pipeline

```mermaid
flowchart TB
    subgraph "Ingestion"
        CSV[CSV Files]
        API[External APIs]
        DB[(Database)]
    end
    
    subgraph "Processing"
        Clean[Data Cleaning]
        Validate[Validation]
        Feature[Feature Engineering]
    end
    
    subgraph "Storage"
        Raw[(Raw Data)]
        Processed[(Processed)]
        Model[(Model Artifacts)]
    end
    
    CSV --> Clean
    API --> Clean
    DB --> Clean
    
    Clean --> Validate
    Validate --> Feature
    
    Raw <--> Clean
    Processed <--> Feature
    Model <--> Feature
```

---

## ML Ensemble Model

```mermaid
flowchart TB
    Input[Historical Sales Data]
    
    subgraph "Model Ensemble"
        Prophet[Prophet<br/>Seasonality]
        LSTM[LSTM<br/>Temporal Patterns]
        XGB[XGBoost<br/>Feature-Based]
    end
    
    Ensemble[Weighted Ensemble]
    Output[Final Forecast]
    
    Input --> Prophet
    Input --> LSTM
    Input --> XGB
    
    Prophet --> Ensemble
    LSTM --> Ensemble
    XGB --> Ensemble
    
    Ensemble --> Output
```

---

## CLI Interface

```mermaid
flowchart TB
    User[(User)] --> CLI[CLI Input]
    
    subgraph "Commands"
        FC[forecast]
        INV[inventory]
        ASK[ask]
        EXP[export]
    end
    
    subgraph "Functions"
        Data[Data Retrieval]
        ML[ML Inference]
        LLM[LLM Reasoning]
        EXP_F[Export Functions]
    end
    
    CLI --> FC
    CLI --> INV
    CLI --> ASK
    CLI --> EXP
    
    FC --> Data
    FC --> ML
    INV --> Data
    ASK --> LLM
    EXP --> EXP_F
    
    Data --> Out[Output]
    ML --> Out
    LLM --> Out
    EXP_F --> Out
```

---

## Deployment Architecture

```mermaid
flowchart LR
    subgraph "Local Development"
        Ollama[Ollama<br/>LLM]
        Python[Python<br/>LangGraph]
    end
    
    subgraph "Production (Future)"
        Cloud[Cloud GPU]
        K8s[Kubernetes]
        HF[HuggingFace<br/>Endpoints]
    end
    
    Ollama -.->|推理| Python
    
    style Ollama fill:#90EE90
    style Python fill:#ADD8E6
    style Cloud fill:#FFB6C1
    style K8s fill:#FFB6C1
    style HF fill:#FFB6C1
```

---

## Data Flow Example

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant LangGraph
    participant ML
    participant LLM
    
    User->>CLI: forecast NBO001
    CLI->>LangGraph: Run workflow
    
    LangGraph->>ML: Get historical data
    ML-->>LangGraph: Sales data
    
    LangGraph->>ML: Run forecast
    ML-->>LangGraph: Predictions
    
    LangGraph->>LLM: Analyze context
    LLM-->>LangGraph: Insights
    
    LangGraph->>CLI: Results + Analysis
    CLI-->>User: Markdown table
```

---

## Export Pipeline

```mermaid
flowchart LR
    Data[Forecast Data] --> CSV[CSV]
    Data --> XLSX[Excel]
    Data --> PDF[PDF Report]
    Data --> API[API Response]
    
    CSV --> Share[Share/Email]
    XLSX --> Share
    PDF --> Share
    API --> Integration
```
