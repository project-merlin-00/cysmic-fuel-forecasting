# EPRA Research Conference 2026 - Full Paper

**Theme:** Advancing Energy Affordability and Security in Sustainable Development

---

## Title

**"Fuel Demand Forecasting in Kenya's Oil & Gas Retail Sector: A Framework for Agentic Orchestration"**

---

## Abstract

Kenya's oil and gas retail sector operates with significant inefficiencies in demand forecasting, relying predominantly on manual, experience-based predictions that lead to stockouts or overstocking. This paper presents a novel agentic AI architecture designed specifically for fuel demand forecasting, leveraging the orchestration of traditional machine learning models with large language models (LLMs) to create a responsive, self-improving forecasting system.

The proposed framework integrates:
1. **Time-series ML models** (ARIMA, LSTM, Prophet) for quantitative demand prediction
2. **LLM agents** for qualitative reasoning - processing news, weather, events, policy changes
3. **Orchestration layer** - agentic workflow that coordinates data ingestion, model inference, and decision triggers

Unlike conventional forecasting systems, this architecture enables:
- Real-time incorporation of external factors (holidays, elections, infrastructure projects)
- Natural language interfaces for store managers to query forecasts
- Automated alert generation for anomalous demand patterns

This paper contributes a practical framework for downstream petroleum retailers in emerging markets, with specific recommendations for Kenya's context including data infrastructure requirements and implementation phasing.

**Keywords:** Agentic AI, Demand Forecasting, Large Language Models, Machine Learning, Oil & Gas Retail, Kenya

---

## 1. Introduction

### 1.1 Background
The downstream petroleum sector in Kenya serves as a critical infrastructure for economic activity, with fuel availability directly impacting transportation, manufacturing, and household welfare. Kenya's petroleum retail market consists of approximately 4,000 registered fuel stations, operated by major oil marketing companies (OMCs) including KenolKobil, TotalEnergies, Shell, and independent operators. The sector contributes approximately 8% of Kenya's GDP and provides employment to over 100,000 people directly and indirectly.

Unlike mature markets, Kenya's fuel retail landscape exhibits unique characteristics that complicate demand forecasting:

- **High Volatility:** Consumer demand fluctuates dramatically based on economic cycles, currency movements, and regional economic activity. A sudden fuel price adjustment—whether from international crude oil markets or EPRA-regulated pump prices—can shift purchasing patterns within days. The 2022-2023 fuel price crisis demonstrated how quickly demand patterns can change, with some stations reporting 30% swings in daily volume within two weeks.

- **Seasonal Patterns:** Agricultural cycles (planting/harvest seasons in maize, coffee, and tea regions), school terms, holiday travel, and religious festivals (Ramadan, Easter, Christmas) create distinct demand signatures that vary across regions. Coastal tourism seasons, Northern Kenya's drought cycles, and Rift Valley agricultural activities each influence local demand differently.

- **Regulatory & Compliance Challenges:** Kenya's fuel retail sector faces challenges from fuel adulteration and illicit trade. EPRA has closed 17 stations for selling adulterated or export-bound fuel locally, and the country loses approximately KES 34 billion annually to fuel adulteration (Business Daily, 2024). These illicit practices create unpredictable demand shifts that formal stations must anticipate, as contaminated fuel drives customers to seek alternatives.

- **Infrastructure Variability:** Road network changes, new mall openings, traffic pattern shifts, and competitor station launches can suddenly alter catchment area dynamics. Nairobi's rapid urbanization means station catchments can change significantly within 12-24 months.

Despite this complexity, retail operations continue to rely heavily on manual forecasting methods—often based on historical thumb rules or manager intuition—leading to:

- **Stockouts** during peak demand periods, causing revenue loss and customer dissatisfaction
- **Overstocking** resulting in working capital tied up in inventory
- **Inefficient logistics** due to reactive rather than predictive supply chain planning
- **Revenue leakage** estimated at 5-10% of potential margins due to poor demand visibility

### 1.2 Problem Statement
Existing literature on AI in petroleum forecasting focuses predominantly on upstream operations (reservoir prediction, drilling optimization) or crude oil price forecasting. Downstream retail demand forecasting—particularly for fuel stations—remains under-researched in the African context.

Furthermore, conventional ML approaches treat demand forecasting as a pure prediction problem, ignoring the wealth of qualitative information (market news, regulatory announcements, weather patterns, local events) that experienced managers incorporate intuitively. A station manager in Mombasa may intuitively know that "the Eid holiday weekend will increase diesel demand by 20%," but this knowledge remains tacit and difficult to encode in traditional statistical models.

The research problem addressed by this paper is: **How can an agentic AI system combine quantitative machine learning with qualitative LLM-based reasoning to achieve superior fuel demand forecasting accuracy for Kenya's retail sector?**

### 1.3 Research Objective
This paper proposes an **agentic AI architecture** that combines:
- Quantitative ML models for historical pattern recognition
- Large Language Models for qualitative reasoning
- An orchestration layer enabling autonomous workflow execution

The specific objectives are:
1. To design a four-layer architecture for fuel demand forecasting in emerging markets
2. To implement an ML ensemble combining Prophet, LSTM, and XGBoost
3. To integrate LLM agents for qualitative factor processing
4. To develop a LangGraph-based orchestration system
5. To propose a phased implementation roadmap for Kenya's context

### 1.4 Hypothesis
**"An agentic AI system combining time-series ML models with LLM-based reasoning can achieve 20-25% improvement in fuel demand forecasting accuracy for Kenyan retail stations, specifically by capturing volatile economic signals and seasonal patterns that manual methods miss, within 12 months of deployment."**

---

## 2. Literature Review

### 2.1 Demand Forecasting in Oil & Gas Retail
Demand forecasting in petroleum retail has evolved through several phases:

**Traditional Methods (1980s-2000s):**
- Moving averages and exponential smoothing for simple trend capture
- Croston's method for intermittent demand
- Judgmental forecasting based on manager experience

**Statistical Models (2000s-2015):**
- ARIMA and SARIMA for seasonal patterns (Weldon et al., 2022)
- Vector Autoregression (VAR) for multivariate time series
- State-space models and Kalman filtering

**Machine Learning (2015-present):**
- Random Forests and Gradient Boosting for feature-based prediction
- Support Vector Machines (SVM) for non-linear relationships
- Deep learning approaches including LSTM and Transformer networks

Research from Nigeria (2025) demonstrates ML-based demand prediction achieving 15-20% accuracy improvements. However, these studies focus purely on quantitative models without integrating qualitative intelligence. The seminal work by Weldon et al. (2022) reviewed data mining techniques for petroleum price prediction in Kenya, establishing a foundation for quantitative approaches.

### 2.2 Agentic AI in Energy Sector
Agentic AI represents a paradigm shift from passive prediction to active orchestration. BDO (2024) identifies agentic AI as transformative for natural resources, with applications spanning:

- **Predictive Maintenance:** Autonomous agents tracking equipment health through sensor data, predicting failures before they occur, and automatically scheduling maintenance
- **Real-time Logistics:** Intelligent agents adjusting transportation routes based on changing conditions
- **Market Intelligence:** Agents processing news, social media, and regulatory announcements to anticipate market shifts

Salesforce and SupplyChainBrain identify retail as a key beneficiary of agentic AI, where agents monitor demand patterns and adjust replenishment autonomously. McKinsey (2024) projects that gen AI could unlock $6.6-10.4 billion in economic value in Africa's consumer-facing retail sector.

### 2.3 Gap Analysis
The literature reveals several gaps that this paper addresses:

| Aspect | Current Research | This Paper |
|--------|-----------------|------------|
| Geographic focus | Global/North America | Kenya/East Africa |
| Value chain | Upstream/Midstream | Downstream Retail |
| AI approach | Pure ML | Agentic ML + LLM |
| Implementation | Theoretical | Framework + Code + Roadmap |
| Qualitative factors | Not addressed | LLM-based processing |
| Seasonality | Basic | Region-specific models |
| Infrastructure | Assumed mature | Kenya-specific gaps |

### 2.4 Kenya-Specific Context
Kenya's petroleum sector operates under unique regulatory and market conditions:

- **EPRA Regulation:** The Energy and Petroleum Regulatory Authority sets maximum pump prices weekly, based on international benchmarks and exchange rates
- **Import Dependency:** Kenya imports 90%+ of refined petroleum products, making prices vulnerable to international crude movements and shipping costs
- **Currency Risk:** The Kenya Shilling's volatility against the USD directly impacts landed costs
- **Infrastructure Gaps:** Limited pipeline infrastructure means road transport dominates product movement, adding logistics complexity

---

## 3. Proposed Agentic AI Architecture

### 3.1 System Overview

The proposed architecture consists of four layers, each serving distinct functions:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│
│  │ Dashboard   │  │ Chat Interface│  │ Alert Notifications    ││
│  │ (Forecasts)│  │ (Query LLM)  │  │ (Stockout/Overstock)  ││
│  └─────────────┘  └─────────────┘  └─────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                    ORCHESTRATION LAYER (Agentic Core)           │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    CONTROLLER AGENT (LangGraph)              ││
│  │  - Task decomposition    - Multi-agent coordination         ││
│  │  - Workflow execution   - Feedback loop management          ││
│  │  - Error handling       - Human-in-the-loop checkpoints     ││
│  └─────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                    INTELLIGENCE LAYER                           │
│  ┌──────────────────────┐    ┌────────────────────────────────┐│
│  │  ML Model Ensemble   │    │  LLM Reasoning Engine          ││
│  │  ┌────────────────┐  │    │  ┌──────────────────────────┐  ││
│  │  │ Prophet        │  │    │  │ News/Event Processing   │  ││
│  │  │ (seasonality)  │  │    │  │ (qualitative factors)   │  ││
│  │  └────────────────┘  │    │  └──────────────────────────┘  ││
│  │  ┌────────────────┐  │    │  ┌──────────────────────────┐  ││
│  │  │ LSTM           │  │    │  │ Natural Language Query   │  ││
│  │  │ (temporal)     │  │    │  │ Interface              │  ││
│  │  └────────────────┘  │    │  └──────────────────────────┘  ││
│  │  ┌────────────────┐  │    │  ┌──────────────────────────┐  ││
│  │  │ XGBoost       │  │    │  │ Anomaly Detection       │  ││
│  │  │ (features)    │  │    │  │ (pattern breaks)        │  ││
│  │  └────────────────┘  │    │  └──────────────────────────┘  ││
│  └──────────────────────┘    └────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                    DATA LAYER                                   │
│  ┌───────────────┐  ┌──────────────┐  ┌───────────────────────┐│
│  │ POS Data      │  │ External API │  │ Station Metadata      ││
│  │ (sales history)│ │ (weather,    │  │ (location, capacity, ││
│  │               │  │  fuel prices)│  │  competitor proximity)││
│  └───────────────┘  └──────────────┘  └───────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Layer Descriptions

**3.2.1 Presentation Layer**
The presentation layer provides multiple interfaces for users:
- **Dashboard:** Web-based visualization of forecasts, historical accuracy, and alerts
- **Chat Interface:** Natural language query system for station managers
- **Alert System:** Automated notifications via SMS, email, or WhatsApp

**3.2.2 Orchestration Layer (LangGraph)**
The controller agent coordinates the entire workflow:
- Receives requests (forecast, inventory check, query)
- Decomposes tasks into subtasks
- Coordinates data, ML, and LLM agents
- Manages state across workflow execution
- Handles errors and retries
- Supports human-in-the-loop for critical decisions

LangGraph was chosen over alternatives (Celery, Airflow) because:
- Native support for LLM integration
- Cyclic workflows (essential for reflection/retry)
- State persistence across workflow steps
- Built-in support for human-in-the-loop

**3.2.3 Intelligence Layer**
The intelligence layer combines two complementary approaches:

*ML Ensemble:*
- **Prophet:** Facebook's time-series model excels at capturing seasonality (daily, weekly, yearly) and holiday effects
- **LSTM:** Long Short-Term Memory networks model complex temporal dependencies
- **XGBoost:** Gradient boosting handles structured features (price, weather, events) effectively

*LLM Agent:*
- Processes qualitative information (news, policy changes)
- Generates natural language explanations
- Provides contextual adjustments to ML predictions
- Answers ad-hoc queries from managers

**3.2.4 Data Layer**
The data layer manages all data sources:
- **POS Data:** Sales transactions from station systems
- **Weather Data:** Kenya Meteorological Society API
- **Event Calendar:** Holidays, elections, major events
- **EPRA Prices:** Regulated fuel prices
- **Station Metadata:** Location, capacity, competitor information

### 3.3 Agentic Workflow

The controller agent executes the following workflow:

```
1. DATA INGESTION
   ├── Pull historical sales data (daily, by product grade)
   ├── Fetch external data (weather API, competitor prices, calendar)
   └── Validate data quality → flag anomalies

2. ML INFERENCE
   ├── Run Prophet for baseline + seasonality
   ├── Run LSTM for temporal patterns
   ├── Run XGBoost for feature-based predictions
   └── Ensemble predictions (weighted average)

3. LLM REASONING
   ├── Fetch recent news (government policy, infrastructure projects)
   ├── Process local event calendar (holidays, elections)
   └── Generate qualitative adjustment factors

4. ORCHESTRATION
   ├── Combine ML predictions with LLM adjustments
   ├── Calculate confidence intervals
   └── Generate forecasts per station, per product

5. ACTION
   ├── Push forecasts to dashboard
   ├── Trigger reorder alerts if threshold breached
   ├── Update inventory recommendations
   └── Log feedback for model retraining
```

### 3.4 Key Innovations

1. **Hybrid Prediction Engine**
   - ML models handle quantitative patterns (trend, seasonality, autocorrelation)
   - LLM handles qualitative reasoning (news sentiment, event impact)

2. **Conversational Interface**
   - Store managers can ask: "What's our diesel demand for next week?"
   - LLM interprets query, routes to appropriate model, returns natural language response
   - Example: "Should I order extra stock for Eid?" → LLM analyzes event impact + ML forecast

3. **Autonomous Learning**
   - System tracks actual vs. predicted
   - Retrains models on quarterly basis with new data
   - Continuously refines LLM prompts based on accuracy

4. **Kenya-Specific Adaptation**
   - Incorporates EPRA price regulation cycles
   - Models agricultural seasonality (tea, coffee, maize harvests)
   - Accounts for informal market competition

---

## 4. Implementation Framework

### 4.1 Phased Implementation

| Phase | Timeline | Focus | Deliverables |
|-------|----------|-------|--------------|
| **Phase 1** | Months 1-3 | Foundation | Data pipeline, baseline ML model, POS integration |
| **Phase 2** | Months 4-6 | Intelligence | LLM integration, external data sources |
| **Phase 3** | Months 7-9 | Automation | Agentic workflow, automated alerts |
| **Phase 4** | Months 10-12 | Optimization | Retraining, performance tuning, scaling |

### 4.2 Data Requirements

| Data Type | Source | Criticality | Current Availability |
|-----------|--------|--------------|---------------------|
| Daily sales by product | POS Systems | Essential | Fragmented |
| Station metadata | Operations | Essential | Partial |
| Historical prices | EPRA/Regulator | Important | Available |
| Weather data | Kenya Met | Important | Partial API |
| News/Events | Web + APIs | Nice to have | Manual |
| Competitor locations | Manual mapping | Important | Limited |

### 4.3 Infrastructure Requirements

- **Cloud:** AWS/GCP for model hosting (~$500-1000/month for MVP)
- **Edge:** Local caching for offline capability at stations with poor connectivity
- **Security:** VPN for station connectivity, encrypted data transmission
- **LLM:** Local Ollama deployment for data privacy (or cloud API for scale)

### 4.4 Kenya-Specific Considerations

1. **Connectivity:** Limited internet at some stations → batch processing fallback with local caching
2. **Power:** Unreliable electricity → UPS/backup for critical equipment
3. **Skills:** Limited ML talent → partner with local universities, use managed ML services initially
4. **Data Quality:** Inconsistent POS data → invest in data cleaning pipeline, establish data governance

### 4.5 GitHub Implementation

The full implementation is available at: https://github.com/project-merlin-00/cysmic-fuel-forecasting

The repository includes:
- LangGraph orchestration with Ollama integration
- ML model implementations (Prophet, LSTM, XGBoost)
- CLI tool for demand forecasting
- Export capabilities (CSV, Excel)
- Architecture diagrams (Mermaid.js)

---

## 5. Expected Outcomes

### 5.1 Quantifiable Benefits

| Metric | Current (Est.) | Target | Improvement |
|--------|---------------|--------|-------------|
| Forecast Accuracy | 60-70% | 85%+ | +20-25% |
| Stockout Rate | 15% | 5% | -67% |
| Overstock Cost | 10% of COGS | 3% | -70% |
| Manager Time on Forecasting | 10 hrs/week | 2 hrs/week | -80% |
| Lead Time for Orders | 24-48 hours | Real-time | Significant |

### 5.2 Qualitative Benefits

- **Data-driven culture** transformation within organizations
- **Competitive advantage** through superior supply chain planning
- **Customer satisfaction** through consistent fuel availability
- **Scalability** for multi-station operators seeking growth
- **Knowledge preservation** through systematic capture of manager expertise

### 5.3 Economic Impact

For a typical station dispensing 50,000 liters/day:
- 20% accuracy improvement → ~KES 3.6M annual savings (avoided stockouts + reduced overstock)
- 80% time reduction → ~KES 480K annual labor savings
- Total potential value: ~KES 4-5M per station annually

For Kenya's ~4,000 stations:
- Aggregate potential value: KES 16-20 billion annually

---

## 6. Challenges and Mitigations

| Challenge | Mitigation |
|-----------|------------|
| Data quality issues | Invest in data cleaning pipeline; establish data governance; start with stations having better data |
| Infrastructure costs | Start with cloud MVP; scale as ROI proves; consider hybrid cloud-edge |
| Talent gap | Partner with local universities (UoN, KU); use managed ML services initially; train existing staff |
| Adoption resistance | Demonstrate quick wins; involve managers in design; provide training |
| Regulatory compliance | Engage EPRA early; ensure data privacy compliance; document AI decision-making |
| Model drift | Implement continuous monitoring; quarterly retraining; human oversight for major decisions |

---

## 7. Conclusion

This paper presents a novel agentic AI architecture for fuel demand forecasting that addresses a critical gap in Kenya's downstream petroleum sector. By combining traditional machine learning models with large language models in an orchestrated workflow using LangGraph, the framework enables:

1. **Improved accuracy** through hybrid quantitative-qualitative reasoning
2. **Operational efficiency** through automation of routine forecasting tasks
3. **Scalability** through standardized agentic workflows
4. **Accessibility** through natural language interfaces for non-technical users

The proposed phased implementation approach ensures manageable risk while building toward a comprehensive solution. With projected 20-25% accuracy improvements and significant cost savings, this framework offers a practical path forward for Kenyan fuel retailers.

The key contribution is the **agentic orchestration layer** that combines:
- Multiple ML models (Prophet, LSTM, XGBoost) in an ensemble
- LLM-based qualitative reasoning for news, events, and context
- LangGraph workflow automation with human-in-the-loop capabilities
- Kenya-specific adaptations for the local market context

### 7.1 Recommendations

1. **For Retailers:** Begin pilot with 3-5 stations; measure baseline metrics; expand based on ROI; involve station managers in design
2. **For EPRA:** Develop regulatory sandbox for AI pilots; establish data sharing frameworks; create standards for AI in energy
3. **For Policymakers:** Support AI skills development through universities; create enabling environment for innovation; consider tax incentives for AI adoption

### 7.2 Future Research

- Integration with smart fuel storage (IoT sensors)
- Real-time dynamic pricing optimization
- Multi-station network optimization
- Integration with electric vehicle charging demand forecasting

---

## References

1. Weldon, K. et al. (2022). Petroleum prices prediction using data mining techniques. arXiv:2211.12964
2. Contini, A. et al. (2025). Data-driven jet fuel demand forecasting: A case study of Copenhagen Airport. arXiv
3. Nigeria Petroleum Demand ML Study (2025). ResearchGate
4. BDO (2024). Agentic AI Use Cases for Natural Resources Firms
5. Salesforce (2024). Agentic AI in Retail: Benefits & Use Cases
6. McKinsey (2024). Leading, not lagging: Africa's gen AI opportunity
7. SLB (2024). What will it take to scale AI in Africa's O&G sector?
8. ESI Africa (2024). AI potential to transform energy, oil & gas sectors in Africa
9. EPRA Kenya (2025). Petroleum Pricing Guidelines
10. Kenya National Bureau of Statistics (2025). Economic Survey - Energy Sector
11. Business Daily (2024). Kenya loses Sh34bn annually to adulterated fuel. https://www.businessdailyafrica.com/bd/news/kenya-loses-sh34bn-annually-to-adulterated-fuel-2213016
12. EPRA (2025). Public Notice: Crackdown on Adulterated Fuel. https://www.facebook.com/KaraKenya/posts/epra-public-notice-crackdown-on-adulterated-fuel
13. IESE Business School. Kenya's Imperfect Fuel Market. https://media.iese.edu/research/pdfs/WP-1200-E.pdf

---

## Appendices

### Appendix A: Sample API Specifications

```
GET /api/v1/forecast
{
  "station_id": "NBO001",
  "product": "diesel",
  "horizon_days": 7,
  "include_confidence": true
}

Response:
{
  "forecasts": [
    {"date": "2026-02-15", "volume": 12500, "confidence": 0.87},
    {"date": "2026-02-16", "volume": 13200, "confidence": 0.85}
  ],
  "model_version": "1.0.0",
  "generated_at": "2026-02-14T10:00:00Z"
}
```

### Appendix B: Data Schema Requirements

```
stations:
  - station_id (string, PK)
  - name (string)
  - region (string)
  - latitude (float)
  - longitude (float)
  - capacity_liters (int)
  - pumps (int)
  - competitors_nearby (int)

sales:
  - date (date)
  - station_id (string, FK)
  - product (enum: super, diesel, kerosene)
  - volume_liters (int)
  - revenue (int)

events:
  - date (date)
  - event_name (string)
  - event_type (enum: holiday, election, sports, other)
  - impact_level (enum: low, medium, high)
```

### Appendix C: Cost Estimation Details

| Component | MVP (3 months) | Production (12 months) |
|-----------|----------------|----------------------|
| Cloud Infrastructure | $500/month | $1,500/month |
| Development | $3,000 | $12,000 |
| Training | $500 | $2,000 |
| Integration | $1,000 | $3,000 |
| **Total Year 1** | **$11,500** | **$35,500** |
| ROI (per station) | KES 4-5M | KES 4-5M |

---

**Word Count:** ~4,200 words (excluding references)

**Submission Date:** February 15, 2026

**Authors:** 
- Arthur Kisaka, Technical Consultant, Cysma Consulting
- Ian Njuguna, Energy Advisor, COG Kenya; Technical Consultant, Cysma Consulting

**Contact:** research@epra.go.ke
**GitHub:** https://github.com/project-merlin-00/cysmic-fuel-forecasting
