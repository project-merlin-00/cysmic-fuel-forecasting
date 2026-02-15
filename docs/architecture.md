# CYSMIC Fuel Forecasting - Architecture Diagrams

> Render at [mermaid.live](https://mermaid.live) - Kenya Oil & Gas Retail AI System

---

## System Architecture - Fuel Demand Forecasting

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#4a90d9', 'edgeLabelBackground':'#ffffff'}}}%%
flowchart TB
    subgraph Client["📱 Client Interface"]
        direction LR
        CLI[CLI/TUI]
        API[REST API]
    end
    
    subgraph Orchestration["⚙️ LangGraph Controller"]
        CA[Controller<br/>Agent]
        WS[Workflow<br/>Scheduler]
    end
    
    subgraph Intelligence["🧠 Intelligence Layer"]
        direction TB
        subgraph ML["ML Ensemble"]
            Prophet[📈 Prophet<br/>Seasonality]
            LSTM[🧠 LSTM<br/>Temporal]
            XGB[🎯 XGBoost<br/>Features]
        end
        subgraph LLM["LLM Agent"]
            Ollama[Ollama<br/>LLM]
            Chain[LangChain<br/>Tools]
        end
    end
    
    subgraph Data["💾 Data Layer - Kenya O&G Retail"]
        direction LR
        POS[POS Sales<br/>Data]
        Weather[🌤️ Kenya Met<br/>API]
        Events[📅 Calendar<br/>Holidays]
        Prices[⛽ EPRA<br/>Prices]
        Stations[⛽ Station<br/>Metadata]
    end
    
    Client --> CA
    CA --> WS
    CA --> ML
    CA --> LLM
    
    ML --> POS
    ML --> Weather
    ML --> Prices
    
    LLM --> Events
    LLM --> Stations
    
    style Client fill:#e3f2fd,stroke:#1976d2
    style Orchestration fill:#fff3e0,stroke:#f57c00
    style ML fill:#e8f5e9,stroke:#388e3c
    style LLM fill:#f3e5f5,stroke:#7b1fa2
    style Data fill:#fce4ec,stroke:#c2185b
```

---

## Agentic Workflow - Demand Forecasting

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    Start([🚀 Start]) --> DataIngest[📥<br/>Data Agent<br/>Gathers sales<br/>weather<br/>events]
    
    DataIngest --> MLPredict[🤖<br/>ML Agent<br/>Prophet/LSTM<br/>XGBoost]
    
    MLPredict --> LLMReason[💡<br/>LLM Agent<br/>Analyzes context<br/>Adjusts forecast]
    
    LLMReason --> Alert[⚠️<br/>Generate Alerts<br/>Stockout risk<br/>Overstock]
    
    Alert --> Output[📤<br/>Forecast Output<br/>7-day prediction<br/>Markdown/CSV]
    
    Output --> End([✅ End])
    
    style Start fill:#4caf50,color:#fff
    style DataIngest fill:#2196f3,color:#fff
    style MLPredict fill:#9c27b0,color:#fff
    style LLMReason fill:#ff9800,color:#fff
    style Alert fill:#f44336,color:#fff
    style Output fill:#00bcd4,color:#fff
    style End fill:#4caf50,color:#fff
```

---

## Kenya Fuel Retail Data Flow

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    subgraph Sources["📥 Data Sources"]
        direction LR
        EPRA[EPRA<br/>Prices]
        Met[Kenya Met<br/>Weather]
        POS[Station POS<br/>Systems]
        Events[Gov/News<br/>Events]
    end
    
    subgraph Processing["🔧 Processing"]
        Clean[Data<br/>Cleaning]
        Validate[Validation]
        Feature[Feature<br/>Engineering]
    end
    
    subgraph Models["🤖 ML Models"]
        direction LR
        Prophet[Prophet<br/>Seasonality]
        LSTM[LSTM<br/>Pattern]
        XGB[XGBoost<br/>Features]
    end
    
    subgraph Output["📤 Outputs"]
        direction LR
        Forecast[Demand<br/>Forecast]
        Alerts[Stock<br/>Alerts]
        API[REST<br/>API]
    end
    
    Sources --> Processing
    Processing --> Models
    Models --> Output
    
    style Sources fill:#e3f2fd,stroke:#1976d2
    style Processing fill:#fff3e0,stroke:#f57c00
    style Models fill:#e8f5e9,stroke:#388e3c
    style Output fill:#fce4ec,stroke:#c2185b
```

---

## CLI Workflow - Fuel Forecasting Commands

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart TB
    User[👤 Station<br/>Manager] --> CLI[❯ cysmic]
    
    subgraph Commands["📋 Commands"]
        FC[forecast<br/>NBO001]
        INV[inventory<br/>NBO001]
        ASK[ask<br/>question]
        EXP[export<br/>csv/xlsx]
    end
    
    subgraph Actions["⚡ Actions"]
        Data1[Fetch<br/>Data]
        Predict[Run ML<br/>Forecast]
        LLM1[LLM<br/>Analysis]
        Format[Format<br/>Output]
    end
    
    subgraph Results["📤 Results"]
        Table[Markdown<br/>Table]
        Chart[Forecast<br/>Chart]
        File[CSV/XLSX<br/>File]
    end
    
    CLI --> FC
    CLI --> INV
    CLI --> ASK
    CLI --> EXP
    
    FC --> Data1
    FC --> Predict
    INV --> Data1
    ASK --> LLM1
    EXP --> Format
    
    Predict --> Table
    Data1 --> Chart
    LLM1 --> Table
    Format --> File
    
    style User fill:#4caf50,color:#fff
    style CLI fill:#2196f3,color:#fff
    style Commands fill:#ff9800,color:#fff
    style Actions fill:#9c27b0,color:#fff
    style Results fill:#00bcd4,color:#fff
```

---

## Inventory Management Flow

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    Start([Start]) --> Check[📦 Check<br/>Inventory]
    
    Check --> Level{Stock<br/>Level?}
    
    Level -->|OK (>30%)| Monitor[👀 Monitor]
    Level -->|LOW (15-30%)| Alert1[⚠️ Alert<br/>Reorder]
    Level -->|CRITICAL (<15%)| Alert2[🚨 Urgent<br/>Restock]
    
    Monitor --> Predict[📈 Predict<br/>7-day demand]
    Predict --> Enough{Enough<br/>stock?}
    
    Enough -->|Yes| Wait[⏰ Wait]
    Enough -->|No| Recommend[📝 Recommend<br/>order qty]
    
    Alert1 --> Recommend
    Alert2 --> Recommend
    
    Recommend --> Order[🛒 Create<br/>Order]
    Order --> End([✅ End])
    Wait --> End
    
    style Start fill:#4caf50,color:#fff
    style Check fill:#2196f3,color:#fff
    style Level fill:#ff9800,color:#fff
    style Alert1 fill:#ff9800,color:#fff
    style Alert2 fill:#f44336,color:#fff
    style Predict fill:#9c27b0,color:#fff
    style Recommend fill:#00bcd4,color:#fff
    style Order fill:#4caf50,color:#fff
    style End fill:#4caf50,color:#fff
```

---

## Kenya O&G Retail Value Chain

```mermaid
%%{init: {'theme': 'base'}}%%
flowchart LR
    Upstream[🏭 Upstream<br/>Refineries] -->|Bulk Fuel| Midstream[🚛 Midstream<br/>Transportation]
    
    Midstream -->|Deliver| Retail[⛽ Retail<br/>Stations]
    
    Retail -->|Sales Data| AI[🤖 CYSMIC AI<br/>Forecasting]
    
    AI -->|Forecast| Inventory[📦 Inventory<br/>Management]
    AI -->|Predict| Demand[📈 Demand<br/>Planning]
    AI -->|Alert| Pricing[💰 Dynamic<br/>Pricing]
    
    Inventory --> Retail
    Demand --> Procurement[🛒 Procurement]
    Pricing --> Retail
    
    style Upstream fill:#e3f2fd,stroke:#1976d2
    style Midstream fill:#e3f2fd,stroke:#1976d2
    style Retail fill:#fce4ec,stroke:#c2185b
    style AI fill:#fff3e0,stroke:#f57c00
    style Inventory fill:#e8f5e9,stroke:#388e3c
    style Demand fill:#e8f5e9,stroke:#388e3c
    style Pricing fill:#e8f5e9,stroke:#388e3c
    style Procurement fill:#f3e5f5,stroke:#7b1fa2
```
