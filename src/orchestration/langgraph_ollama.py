"""
LangGraph Orchestration for CYSMIC Fuel Forecasting
Uses LangGraph with Ollama for agentic workflow
"""

import os
import sys
import json
from typing import TypedDict, List, Annotated
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

# LangGraph imports
from langgraph.graph import StateGraph, END

# LangChain + Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool

# ML imports
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# STATE DEFINITION
# ============================================================

class AgentState(TypedDict):
    """State passed between nodes in the graph."""
    messages: List  # Chat messages
    context: dict   # Data context
    task: str       # Current task
    result: dict    # Result of operations
    error: str      # Error message if any


# ============================================================
# TOOLS (LangChain Tools)
# ============================================================

@tool
def get_sales_data(station_id: str = None, days: int = 30) -> str:
    """
    Get historical sales data for analysis.
    
    Args:
        station_id: Station ID (optional, default all)
        days: Number of days to retrieve
        
    Returns:
        Sales data summary
    """
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    data = []
    
    stations = [station_id] if station_id else ["NBO001", "NBO002", "MBA001"]
    products = ["super", "diesel", "kerosene"]
    
    for station in stations:
        for product in products:
            base = np.random.randint(5000, 15000)
            for date in dates:
                volume = base * (1 + 0.2 * np.sin(2 * np.pi * date.day / 30))
                volume += np.random.randint(-500, 500)
                data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "station": station,
                    "product": product,
                    "volume": int(max(0, volume))
                })
    
    df = pd.DataFrame(data)
    summary = f"Sales data for {len(df)} records:\n"
    summary += f"Total volume: {df['volume'].sum():,}L\n"
    summary += f"Average daily: {df.groupby('date')['volume'].sum().mean():,.0f}L\n"
    summary += f"Date range: {df['date'].min()} to {df['date'].max()}"
    
    return summary


@tool
def get_weather_forecast(days: int = 7) -> str:
    """
    Get weather forecast for the next N days.
    
    Args:
        days: Number of days to forecast
        
    Returns:
        Weather forecast summary
    """
    weather_types = ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"]
    forecast = []
    
    for i in range(days):
        date = datetime.now() + timedelta(days=i)
        temp = np.random.randint(18, 30)
        weather = np.random.choice(weather_types)
        forecast.append(f"{date.strftime('%Y-%m-%d')}: {weather}, {temp}°C")
    
    return "Weather Forecast:\n" + "\n".join(forecast)


@tool
def get_upcoming_events() -> str:
    """Get upcoming events that may affect fuel demand."""
    events = [
        {"date": "2026-02-20", "event": "Easter", "impact": "high"},
        {"date": "2026-03-08", "event": "International Women's Day", "impact": "low"},
        {"date": "2026-04-03", "event": "Good Friday", "impact": "high"},
        {"date": "2026-05-01", "event": "Labour Day", "impact": "medium"},
    ]
    
    return "Upcoming Events:\n" + "\n".join(
        f"- {e['date']}: {e['event']} (impact: {e['impact']})" for e in events
    )


@tool
def run_forecast_model(station_id: str, product: str = "super", days: int = 7) -> dict:
    """
    Run ML forecast model for a station.
    
    Args:
        station_id: Station ID
        product: Fuel type (super/diesel/kerosene)
        days: Forecast horizon
        
    Returns:
        Forecast results as dict
    """
    # Simulated forecast
    base_volume = np.random.randint(8000, 12000)
    forecasts = []
    
    for i in range(days):
        date = datetime.now() + timedelta(days=i)
        volume = base_volume * (1 + 0.1 * np.sin(2 * np.pi * i / 7))
        volume = int(volume + np.random.randint(-1000, 1000))
        forecasts.append({
            "date": date.strftime("%Y-%m-%d"),
            "day": date.strftime("%A"),
            "predicted_volume": max(0, volume),
            "confidence": 0.85
        })
    
    return {
        "station_id": station_id,
        "product": product,
        "days": days,
        "forecasts": forecasts,
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "model": "CYSMIC-ML-Ensemble"
        }
    }


@tool
def check_inventory(station_id: str) -> dict:
    """
    Check current inventory levels at a station.
    
    Args:
        station_id: Station ID
        
    Returns:
        Inventory status as dict
    """
    products = ["super", "diesel", "kerosene"]
    capacity = 50000
    
    inventory = []
    for product in products:
        level = np.random.randint(10000, capacity)
        pct = level / capacity * 100
        status = "OK" if pct > 30 else "LOW" if pct > 15 else "CRITICAL"
        inventory.append({
            "product": product,
            "level_liters": level,
            "capacity_liters": capacity,
            "percentage": round(pct, 1),
            "status": status
        })
    
    return {
        "station_id": station_id,
        "checked_at": datetime.now().isoformat(),
        "inventory": inventory
    }


@tool
def generate_alert(message: str, severity: str = "medium") -> str:
    """
    Generate an alert for critical situations.
    
    Args:
        message: Alert message
        severity: low/medium/high/critical
        
    Returns:
        Confirmation
    """
    emojis = {"low": "ℹ️", "medium": "⚠️", "high": "🔶", "critical": "🚨"}
    emoji = emojis.get(severity, "ℹ️")
    return f"{emoji} ALERT [{severity.upper()}]: {message}"


# ============================================================
# EXPORT FUNCTIONS
# ============================================================

def export_to_csv(data: dict, filename: str = None) -> str:
    """Export forecast data to CSV."""
    if "forecasts" not in data:
        return "No forecast data to export"
    
    df = pd.DataFrame(data["forecasts"])
    
    if not filename:
        filename = f"forecast_{data.get('station_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    df.to_csv(filename, index=False)
    return f"✅ Exported to {filename}"


def export_to_excel(data: dict, filename: str = None) -> str:
    """Export forecast data to Excel with formatting."""
    if "forecasts" not in data:
        return "No forecast data to export"
    
    if not filename:
        filename = f"forecast_{data.get('station_id', 'unknown')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    
    df = pd.DataFrame(data["forecasts"])
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Forecast', index=False)
        
        # Add summary sheet
        summary_df = pd.DataFrame([{
            "Station": data.get("station_id", "N/A"),
            "Product": data.get("product", "N/A"),
            "Days": data.get("days", "N/A"),
            "Generated": data.get("metadata", {}).get("generated_at", "N/A"),
            "Total Volume (L)": df["predicted_volume"].sum(),
            "Avg Daily (L)": df["predicted_volume"].mean(),
        }])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    return f"✅ Exported to {filename}"


def format_markdown_table(data: dict) -> str:
    """Format forecast data as markdown table."""
    if "forecasts" not in data:
        return "No data"
    
    forecasts = data["forecasts"]
    
    md = f"## 📊 Forecast: {data.get('station_id', 'N/A')} - {data.get('product', 'N/A').title()}\n\n"
    md += f"**Generated:** {data.get('metadata', {}).get('generated_at', 'N/A')}\n\n"
    md += "| Date | Day | Predicted Volume (L) | Confidence |\n"
    md += "|------|-----|---------------------|------------|\n"
    
    for f in forecasts:
        conf_pct = f"{f.get('confidence', 0) * 100:.0f}%"
        md += f"| {f['date']} | {f['day']} | {f['predicted_volume']:,} | {conf_pct} |\n"
    
    total = sum(f['predicted_volume'] for f in forecasts)
    avg = total / len(forecasts)
    
    md += f"\n**Total:** {total:,} L | **Avg Daily:** {avg:,.0f} L\n"
    
    return md


# ============================================================
# LANGGRAPH NODES
# ============================================================

def data_agent_node(state: AgentState) -> AgentState:
    """Data Agent: Gathers and validates data."""
    logger.info("Running Data Agent...")
    
    task = state.get("task", "")
    context = {}
    
    if "forecast" in task.lower():
        context["sales_data"] = get_sales_data.invoke({})
        context["weather"] = get_weather_forecast.invoke({"days": 7})
        context["events"] = get_upcoming_events.invoke({})
    elif "inventory" in task.lower():
        context["inventory"] = check_inventory.invoke({})
    
    return {**state, "context": context, "messages": state.get("messages", [])}


def ml_agent_node(state: AgentState) -> AgentState:
    """ML Agent: Runs forecasting models."""
    logger.info("Running ML Agent...")
    
    task = state.get("task", "")
    context = state.get("context", {})
    
    if "forecast" in task.lower():
        # Extract station from task
        parts = task.split()
        station = "NBO001"
        product = "super"
        
        for i, part in enumerate(parts):
            if part.upper() in ["NBO001", "NBO002", "MBA001", "KSM001", "NKS001"]:
                station = part.upper()
            if part.lower() in ["super", "diesel", "kerosene"]:
                product = part.lower()
        
        forecast_result = run_forecast_model.invoke({
            "station_id": station,
            "product": product,
            "days": 7
        })
        context["forecast"] = forecast_result
    
    return {**state, "context": context}


def llm_agent_node(state: AgentState) -> AgentState:
    """LLM Agent: Reasoning and explanation."""
    logger.info("Running LLM Agent...")
    
    try:
        llm = ChatOllama(
            model="llama3.1",
            base_url="http://localhost:11434",
            temperature=0.3
        )
        
        context = state.get("context", {})
        task = state.get("task", "")
        
        context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
        
        prompt = f"""You are a fuel demand forecasting assistant for Kenya's oil and gas retail sector.
        
Task: {task}

Context:
{context_str}

Provide a clear, actionable analysis. Include:
1. Key insights from the data
2. Factors affecting demand
3. Recommendations for the station manager

Be concise and practical. Use markdown formatting.
"""
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=prompt))
        
        response = llm.invoke(messages)
        messages.append(response)
        
        return {
            **state, 
            "messages": messages,
            "result": {"analysis": response.content}
        }
        
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return {**state, "error": str(e)}


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def create_forecasting_graph() -> StateGraph:
    """Create the LangGraph for demand forecasting."""
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("data_agent", data_agent_node)
    workflow.add_node("ml_agent", ml_agent_node)
    workflow.add_node("llm_agent", llm_agent_node)
    
    workflow.set_entry_point("data_agent")
    workflow.add_edge("data_agent", "ml_agent")
    workflow.add_edge("ml_agent", "llm_agent")
    workflow.add_edge("llm_agent", END)
    
    return workflow.compile()


# ============================================================
# CLI INTERFACE
# ============================================================

def print_banner():
    """Print CLI banner."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║   CYSMIC Fuel Forecasting - Agentic CLI                ║
║   Powered by LangGraph + Ollama                         ║
╚═══════════════════════════════════════════════════════════╝
    """)


def print_menu():
    """Print main menu."""
    print("""
## Commands

| Command | Description |
|---------|-------------|
| `forecast <station> [product]` | Run demand forecast |
| `inventory <station>` | Check inventory levels |
| `ask <question>` | Ask a question |
| `export csv [filename]` | Export last forecast to CSV |
| `export xlsx [filename]` | Export last forecast to Excel |
| `status` | System status |
| `help` | Show this menu |
| `quit` | Exit |

## Examples
```
forecast NBO001
forecast NBO002 diesel
inventory MBA001
ask What's the diesel demand for next week?
export csv my-forecast
```
    """)


class CYSMICCLI:
    """CYSMIC CLI with state management."""
    
    def __init__(self):
        self.graph = create_forecasting_graph()
        self.last_forecast = None
        self.ollama_connected = False
        self.check_ollama()
    
    def check_ollama(self):
        """Check Ollama connection."""
        try:
            llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")
            llm.invoke([HumanMessage(content="hi")])
            self.ollama_connected = True
        except:
            pass
    
    def cmd_forecast(self, args: str) -> str:
        """Run forecast command."""
        parts = args.split()
        station = parts[0] if parts else "NBO001"
        product = parts[1] if len(parts) > 1 else "super"
        
        # Validate station
        valid_stations = ["NBO001", "NBO002", "MBA001", "KSM001", "NKS001"]
        if station.upper() not in valid_stations:
            return f"❌ Invalid station. Valid: {', '.join(valid_stations)}"
        
        if product.lower() not in ["super", "diesel", "kerosene"]:
            return f"❌ Invalid product. Valid: super, diesel, kerosene"
        
        station = station.upper()
        product = product.lower()
        
        print(f"🔮 Running forecast for **{station}** ({product})...")
        
        # Run ML agent directly
        forecast_result = run_forecast_model.invoke({
            "station_id": station,
            "product": product,
            "days": 7
        })
        
        self.last_forecast = forecast_result
        
        # Format as markdown
        md = format_markdown_table(forecast_result)
        
        # Try LLM analysis if available
        if self.ollama_connected:
            print("🤖 Running LLM analysis...")
            try:
                initial_state = {
                    "messages": [],
                    "context": {"forecast": forecast_result},
                    "task": f"Analyze this forecast for station {station}",
                    "result": {}
                }
                result = self.graph.invoke(initial_state)
                if "messages" in result and len(result["messages"]) > 0:
                    analysis = result["messages"][-1].content
                    md += f"\n---\n### 🤖 Analysis\n{analysis}\n"
            except Exception as e:
                print(f"⚠️  LLM analysis failed: {e}")
        
        return md
    
    def cmd_inventory(self, args: str) -> str:
        """Check inventory command."""
        station = args.split()[0] if args else "NBO001"
        
        result = check_inventory.invoke({"station_id": station})
        
        md = f"## 📦 Inventory: {station}\n\n"
        md += f"**Checked:** {result['checked_at']}\n\n"
        md += "| Product | Level (L) | Capacity | % | Status |\n"
        md += "|---------|-----------|----------|---|--------|\n"
        
        for item in result["inventory"]:
            status_icon = "🟢" if item["status"] == "OK" else "🟡" if item["status"] == "LOW" else "🔴"
            md += f"| {item['product'].title()} | {item['level_liters']:,} | {item['capacity_liters']:,} | {item['percentage']}% | {status_icon} {item['status']} |\n"
        
        return md
    
    def cmd_export(self, args: str) -> str:
        """Export command."""
        if not self.last_forecast:
            return "❌ No forecast data to export. Run `forecast` first."
        
        parts = args.split()
        fmt = parts[0] if parts else "csv"
        filename = parts[1] if len(parts) > 1 else None
        
        if fmt == "csv":
            return export_to_csv(self.last_forecast, filename)
        elif fmt == "xlsx" or fmt == "excel":
            return export_to_excel(self.last_forecast, filename)
        else:
            return f"❌ Unknown format: {fmt}. Use `csv` or `xlsx`."
    
    def cmd_ask(self, args: str) -> str:
        """Ask command."""
        if not args:
            return "❌ Please provide a question."
        
        if not self.ollama_connected:
            return "❌ Ollama not connected. Run `ollama serve` first."
        
        print("🤔 Thinking...")
        
        initial_state = {
            "messages": [],
            "context": {"forecast": self.last_forecast} if self.last_forecast else {},
            "task": args,
            "result": {}
        }
        
        try:
            result = self.graph.invoke(initial_state)
            if "messages" in result and len(result["messages"]) > 0:
                response = result["messages"][-1].content
                return f"## 💡 Answer\n\n{response}"
        except Exception as e:
            return f"❌ Error: {e}"
        
        return "❌ No response generated."
    
    def cmd_status(self) -> str:
        """Status command."""
        status = """## 🔧 System Status

| Component | Status |
|-----------|--------|
| **LLM (Ollama)** | ✅ Connected" if self.ollama_connected else "❌ Not connected"
| **ML Models** | ✅ Ready |
| **LangGraph** | ✅ Active |
| **Data** | ✅ Sample Data |
"""
        return status


def run_cli():
    """Run the interactive CLI."""
    print_banner()
    print_menu()
    
    cli = CYSMICCLI()
    
    status_msg = "✅ Ollama connected (llama3.1)\n" if cli.ollama_connected else "⚠️  Ollama not connected - LLM features disabled\n"
    print(status_msg)
    
    while True:
        try:
            user_input = input("\n❯ ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(maxsplit=2)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command in ["quit", "exit", "q"]:
                print("Goodbye! 👋")
                break
            
            elif command == "help":
                print_menu()
            
            elif command == "status":
                print(cli.cmd_status())
            
            elif command == "forecast":
                print(cli.cmd_forecast(args))
            
            elif command == "inventory":
                print(cli.cmd_inventory(args))
            
            elif command == "export":
                print(cli.cmd_export(args))
            
            elif command == "ask":
                print(cli.cmd_ask(args))
            
            else:
                print(f"❓ Unknown command: `{command}`\nType `help` for available commands.")
        
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_cli()
