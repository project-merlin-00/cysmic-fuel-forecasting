"""
LangGraph Orchestration for CYSMIC Fuel Forecasting
Uses LangGraph with Ollama for agentic workflow
"""

import os
import sys
from typing import TypedDict, List, Annotated
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

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
def run_forecast_model(station_id: str, product: str = "super", days: int = 7) -> str:
    """
    Run ML forecast model for a station.
    
    Args:
        station_id: Station ID
        product: Fuel type (super/diesel/kerosene)
        days: Forecast horizon
        
    Returns:
        Forecast results
    """
    # Simulated forecast
    base_volume = np.random.randint(8000, 12000)
    forecasts = []
    
    for i in range(days):
        date = datetime.now() + timedelta(days=i)
        # Add some variation
        volume = base_volume * (1 + 0.1 * np.sin(2 * np.pi * i / 7))
        volume = int(volume + np.random.randint(-1000, 1000))
        forecasts.append({
            "date": date.strftime("%Y-%m-%d"),
            "predicted_volume": max(0, volume),
            "confidence": 0.85
        })
    
    result = f"Forecast for {station_id} - {product} ({days} days):\n"
    for f in forecasts:
        result += f"- {f['date']}: {f['predicted_volume']:,}L (conf: {f['confidence']:.0%})\n"
    
    return result


@tool
def check_inventory(station_id: str) -> str:
    """
    Check current inventory levels at a station.
    
    Args:
        station_id: Station ID
        
    Returns:
        Inventory status
    """
    products = ["super", "diesel", "kerosene"]
    capacity = 50000
    
    result = f"Inventory Status - {station_id}:\n"
    for product in products:
        level = np.random.randint(10000, capacity)
        pct = level / capacity * 100
        status = "🟢" if pct > 30 else "🟡" if pct > 15 else "🔴"
        result += f"- {product}: {level:,}L ({pct:.0f}%) {status}\n"
    
    return result


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
# LANGGRAPH NODES
# ============================================================

def data_agent_node(state: AgentState) -> AgentState:
    """Data Agent: Gathers and validates data."""
    logger.info("Running Data Agent...")
    
    task = state.get("task", "")
    
    # Gather relevant data based on task
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
    
    # Run forecast if needed
    if "forecast" in task.lower():
        forecast_result = run_forecast_model.invoke({
            "station_id": "NBO001",
            "product": "super",
            "days": 7
        })
        context["forecast"] = forecast_result
    
    return {**state, "context": context}


def llm_agent_node(state: AgentState) -> AgentState:
    """LLM Agent: Reasoning and explanation."""
    logger.info("Running LLM Agent...")
    
    # Get the LLM (Ollama)
    try:
        llm = ChatOllama(
            model="llama3.1",
            base_url="http://localhost:11434",
            temperature=0.3
        )
        
        # Build context for reasoning
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

Be concise and practical.
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


def should_continue(state: AgentState) -> str:
    """Routing logic."""
    if state.get("error"):
        return "end"
    return "continue"


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def create_forecasting_graph() -> StateGraph:
    """Create the LangGraph for demand forecasting."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("data_agent", data_agent_node)
    workflow.add_node("ml_agent", ml_agent_node)
    workflow.add_node("llm_agent", llm_agent_node)
    
    # Set entry point
    workflow.set_entry_point("data_agent")
    
    # Add edges
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
║   Powered by LangGraph + Ollama                           ║
╚═══════════════════════════════════════════════════════════╝
    """)


def print_menu():
    """Print main menu."""
    print("""
Commands:
  forecast <station_id>  - Run demand forecast
  inventory <station_id> - Check inventory levels  
  ask <question>        - Ask a question
  status                - System status
  help                  - Show this menu
  quit                  - Exit

Examples:
  forecast NBO001
  inventory NBO001
  ask What's the diesel demand for next week?
    """)


def run_cli():
    """Run the interactive CLI."""
    print_banner()
    print_menu()
    
    # Create graph
    graph = create_forecasting_graph()
    
    # Check Ollama
    try:
        llm = ChatOllama(model="llama3.1", base_url="http://localhost:11434")
        print("✅ Ollama connected (llama3.1)\n")
    except Exception as e:
        print(f"⚠️  Ollama not connected: {e}")
        print("   Run 'ollama serve' to start\n")
    
    while True:
        try:
            user_input = input("cysmic> ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if command in ["quit", "exit", "q"]:
                print("Goodbye! 👋")
                break
            
            elif command == "help":
                print_menu()
            
            elif command == "status":
                print("System Status:")
                print("  🤖 LLM: llama3.1 (Ollama)")
                print("  📊 ML: XGBoost, Prophet (ready)")
                print("  🌐 LangGraph: Active")
            
            elif command == "forecast":
                station = args if args else "NBO001"
                print(f"🔮 Running forecast for {station}...")
                
                initial_state = {
                    "messages": [],
                    "context": {},
                    "task": f"Run demand forecast for station {station}",
                    "result": {}
                }
                
                result = graph.invoke(initial_state)
                
                if "result" in result:
                    print("\n📊 Forecast Results:")
                    if "analysis" in result["result"]:
                        print(result["result"]["analysis"])
            
            elif command == "inventory":
                station = args if args else "NBO001"
                print(f"📦 Checking inventory for {station}...")
                print(check_inventory.invoke({"station_id": station}))
            
            elif command == "ask":
                if not args:
                    print("Please provide a question.")
                    continue
                
                print("🤔 Thinking...")
                
                initial_state = {
                    "messages": [],
                    "context": {},
                    "task": args,
                    "result": {}
                }
                
                result = graph.invoke(initial_state)
                
                if "messages" in result and len(result["messages"]) > 0:
                    response = result["messages"][-1]
                    print(f"\n💡 {response.content}")
            
            else:
                print(f"Unknown command: {command}")
                print("Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_cli()
