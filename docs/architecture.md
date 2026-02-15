# CYSMIC Fuel Forecasting - Architecture Diagrams

> Rendered at [mermaid.live](https://mermaid.live) - export as PNG/SVG

---

## System Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a90d9', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#f5f5f5'}}}%%
flowchart TB
    subgraph Presentation["📱 Presentation Layer"]
        direction LR
        CLI[CLI/TUI]
        API[REST API]
        Dashboard[Dashboard]
    end
    
    subgraph Orchestration["⚙️ Orchestration Layer"]
        Controller[Controller<br/>LangGraph]
        Scheduler[Workflow<br/>Scheduler]
    end
    
    subgraph Intelligence["🧠 Intelligence Layer"]
        direction LR
        ML[ML Ensemble<br/>Prophet<br/>LSTM<br/>XGBoost]
        LLM[LLM Agent<br/>Ollama<br/>LangChain]
    end
    
    subgraph Data["💾 Data Layer"]
        direction LR
        POS[POS Sales]
        Weather[Weather API]
        Events[Events]
        Prices[Fuel Prices]
        Stations[Station Meta]
    end
    
    Presentation --> Controller
    Controller --> Orchestration
    Orchestration --> Intelligence
    Intelligence --> Data
    
    Controller --> Scheduler
    
    style Presentation fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    style Orchestration fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style Intelligence fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style Data fill:#fce4ec,stroke:#c2185b,stroke-width:2px
```

---

## Agentic Workflow

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    Start([Start]) --> DataAgent[📥<br/>Data Agent]
    DataAgent --> MLAgent[🤖<br/>ML Agent]
    MLAgent --> LLMAgent[💡<br/>LLM Agent]
    LLMAgent --> End([End])
    
    subgraph State["State Management"]
        MS[(messages,<br/>context,<br/>task)]
    end
    
    DataAgent -.->|reads| MS
    MLAgent -.->|reads| MS
    LLMAgent -.->|writes| MS
    
    style Start fill:#4caf50,stroke:#2e7d32,color:#fff
    style End fill:#f44336,stroke:#c62828,color:#fff
    style DataAgent fill:#2196f3,stroke:#1565c0,color:#fff
    style MLAgent fill:#9c27b0,stroke:#6a1b9a,color:#fff
    style LLMAgent fill:#ff9800,stroke:#ef6c00,color:#fff
```

---

## ML Ensemble Pipeline

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    Input[📊 Historical<br/>Sales Data] --> Split[Train/Test<br/>Split]
    
    subgraph Models["🤖 Model Ensemble"]
        direction LR
        Prophet[📈 Prophet<br/>Seasonality]
        LSTM[🧠 LSTM<br/>Temporal]
        XGB[🎯 XGBoost<br/>Features]
    end
    
    Split --> Prophet
    Split --> LSTM
    Split --> XGB
    
    subgraph Ensemble["⚖️ Weighted Ensemble"]
        Weights[ weights]
        Combine[Combine]
    end
    
    Prophet --> Combine
    LSTM --> Combine
    XGB --> Combine
    Weights -.-> Combine
    
    Combine --> Output[📤<br/>Forecast<br/>Output]
    
    style Input fill:#e3f2fd,stroke:#1976d2
    style Models fill:#fff3e0,stroke:#f57c00
    style Ensemble fill:#e8f5e9,stroke:#388e3c
    style Output fill:#fce4ec,stroke:#c2185b
```

---

## CLI Commands Flow

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    User[👤 User] --> CLI[❯ CLI]
    
    subgraph Commands["📋 Commands"]
        FC[forecast]
        INV[inventory]
        ASK[ask]
        EXP[export]
    end
    
    subgraph Execution["⚡ Execution"]
        direction LR
        Data[Data<br/>Fetch]
        ML[ML<br/>Predict]
        LLM[LLM<br/>Reason]
        EXP_F[Export<br/>CSV/XLSX]
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
    
    ML --> Output[📊 Output]
    Data --> Output
    LLM --> Output
    EXP_F --> File[💾 File]
    
    style User fill:#4caf50,color:#fff
    style CLI fill:#2196f3,color:#fff
    style Commands fill:#ff9800,color:#fff
    style Execution fill:#9c27b0,color:#fff
```

---

## Data Pipeline

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    subgraph Ingest["📥 Ingestion"]
        CSV[CSV Files]
        API[External APIs]
        DB[(Database)]
    end
    
    subgraph Process["🔧 Processing"]
        Clean[Data<br/>Cleaning]
        Validate[Validation]
        Feature[Feature<br/>Engineering]
    end
    
    subgraph Store["💾 Storage"]
        Raw[(Raw)]
        Processed[(Processed)]
        Models[(Model<br/>Artifacts)]
    end
    
    subgraph Serve["📤 Serving"]
        API_S[REST API]
        CLI_S[CLI]
        Dashboard_S[Dashboard]
    end
    
    CSV --> Clean
    API --> Clean
    DB --> Clean
    
    Clean --> Validate
    Validate --> Feature
    
    Raw <--> Clean
    Processed <--> Feature
    Models <--> Feature
    
    Feature --> API_S
    Feature --> CLI_S
    Feature --> Dashboard_S
    
    style Ingest fill:#e3f2fd,stroke:#1976d2
    style Process fill:#fff3e0,stroke:#f57c00
    style Store fill:#e8f5e9,stroke:#388e3c
    style Serve fill:#fce4ec,stroke:#c2185b
```
