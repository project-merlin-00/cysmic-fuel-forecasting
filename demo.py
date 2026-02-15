#!/usr/bin/env python3
"""
CYSMIC CLI Demo with Synthetic Data
Run this to test the fuel forecasting system with synthetic data
"""

import json
import random
from datetime import datetime, timedelta
from typing import Dict, List

# ============================================================
# SYNTHETIC DATA GENERATORS
# ============================================================

def generate_sales_data(stations: List[str], products: List[str], days: int = 90) -> List[Dict]:
    """Generate synthetic sales data."""
    data = []
    base_date = datetime.now() - timedelta(days=days)
    
    for station in stations:
        for product in products:
            # Base volume varies by product type
            base = {
                "super": 8000,
                "diesel": 12000,
                "kerosene": 3000
            }.get(product, 5000)
            
            for day in range(days):
                date = base_date + timedelta(days=day)
                
                # Add patterns
                volume = base
                
                # Weekend effect
                if date.weekday() >= 5:
                    volume *= 1.15
                
                # Monthly pay cycle effect
                if date.day in [1, 2, 3, 28, 29, 30]:
                    volume *= 1.1
                
                # Random noise
                volume += random.randint(-1000, 1000)
                
                # Seasonal adjustment
                volume *= (1 + 0.1 * (1 if date.month in [6, 7, 8, 12] else 0))
                
                data.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "station_id": station,
                    "product": product,
                    "volume_liters": max(0, int(volume)),
                    "price_kes": 180.50 if product == "super" else 170.25 if product == "diesel" else 150.00
                })
    
    return data


def generate_inventory(stations: List[str], products: List[str]) -> List[Dict]:
    """Generate inventory data."""
    inventory = []
    
    for station in stations:
        for product in products:
            capacity = 50000 if product == "diesel" else 30000
            level = random.randint(5000, capacity - 5000)
            
            pct = (level / capacity) * 100
            status = "OK" if pct > 30 else "LOW" if pct > 15 else "CRITICAL"
            
            inventory.append({
                "station_id": station,
                "product": product,
                "level_liters": level,
                "capacity_liters": capacity,
                "percentage": int(pct),
                "status": status,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M")
            })
    
    return inventory


# ============================================================
# FORECAST ENGINE (Simplified)
# ============================================================

def generate_forecast(station: str, product: str, days: int = 7) -> Dict:
    """Generate forecast using synthetic model."""
    
    # Base forecasts (simulating different model outputs)
    base_volume = {
        "super": 8500,
        "diesel": 12500,
        "kerosene": 3200
    }.get(product, 5000)
    
    forecasts = []
    for day in range(1, days + 1):
        date = datetime.now() + timedelta(days=day)
        
        # Add variation
        volume = base_volume
        if date.weekday() >= 5:
            volume *= 1.15
        
        # Confidence decreases with forecast horizon
        confidence = 0.90 - (day * 0.03)
        
        forecasts.append({
            "date": date.strftime("%Y-%m-%d"),
            "day_of_week": date.strftime("%A"),
            "predicted_volume": int(volume * random.uniform(0.95, 1.05)),
            "confidence": round(confidence, 2),
            "model": random.choice(["Prophet", "LSTM", "XGBoost", "Ensemble"])
        })
    
    return {
        "station_id": station,
        "product": product,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_horizon": days,
        "forecasts": forecasts,
        "summary": {
            "avg_daily_volume": int(sum(f["predicted_volume"] for f in forecasts) / days),
            "total_volume": sum(f["predicted_volume"] for f in forecasts),
            "peak_day": max(forecasts, key=lambda x: x["predicted_volume"])["date"]
        }
    }


# ============================================================
# CLI INTERFACE
# ============================================================

def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def cmd_forecast(args: str):
    """Run forecast command."""
    parts = args.split()
    station = parts[0] if parts else "NBO001"
    product = parts[1] if len(parts) > 1 else "diesel"
    
    print(f"🔮 Running forecast for {station} ({product})...")
    
    result = generate_forecast(station, product, days=7)
    
    print_header(f"📊 Forecast: {station} - {product.title()}")
    print(f"Generated: {result['generated_at']}\n")
    
    print("| Date       | Day       | Volume (L) | Confidence | Model     |")
    print("|------------|-----------|------------|------------|-----------|")
    for f in result["forecasts"]:
        conf_pct = f"{f['confidence']*100:.0f}%"
        print(f"| {f['date']} | {f['day_of_week']:<9} | {f['predicted_volume']:>10,} | {conf_pct:>10} | {f['model']:<9} |")
    
    print(f"\n📈 Summary:")
    print(f"   Average: {result['summary']['avg_daily_volume']:,} L/day")
    print(f"   Total (7 days): {result['summary']['total_volume']:,} L")
    print(f"   Peak: {result['summary']['peak_day']}")
    
    return result


def cmd_inventory(args: str):
    """Check inventory command."""
    parts = args.split()
    station = parts[0] if parts else "NBO001"
    
    print(f"📦 Checking inventory for {station}...")
    
    inventory = generate_inventory([station], ["super", "diesel", "kerosene"])
    
    print_header(f"📦 Inventory: {station}")
    print(f"Checked: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    
    print("| Product   | Level (L)  | Capacity (L) | %    | Status   |")
    print("|-----------|-------------|---------------|------|----------|")
    for item in inventory:
        status_icon = "🟢" if item["status"] == "OK" else "🟡" if item["status"] == "LOW" else "🔴"
        print(f"| {item['product'].title():<9} | {item['level_liters']:>11,} | {item['capacity_liters']:>13,} | {item['percentage']:>4}% | {status_icon} {item['status']:<6} |")
    
    return {"inventory": inventory}


def cmd_ask(args: str):
    """LLM-style query response (simulated)."""
    if not args:
        return {"error": "Please provide a question"}
    
    print(f"🤔 Question: {args}")
    print("💡 Thinking...\n")
    
    # Simulated LLM responses
    responses = {
        "stock": "Based on current inventory levels and predicted demand, I recommend ordering additional diesel within the next 48 hours. Current stock will cover ~3 days of predicted demand.",
        "trend": "The 7-day forecast shows stable demand with a slight uptick on weekends. Diesel demand averages 12,500L/day with 85% confidence.",
        "default": "Based on historical patterns and current forecasts, I recommend monitoring inventory levels closely. The system predicts normal demand patterns for the next 7 days."
    }
    
    response = responses.get("default", responses["default"])
    
    # Check for keywords
    if "stock" in args.lower() or "order" in args.lower():
        response = responses["stock"]
    elif "trend" in args.lower() or "demand" in args.lower():
        response = responses["trend"]
    
    print(f"🤖 Analysis:\n{response}\n")
    
    return {"question": args, "answer": response}


def cmd_export(data: Dict, fmt: str = "json"):
    """Export data."""
    if fmt == "json":
        filename = "forecast_export.json"
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Exported to {filename}")
    elif fmt == "csv":
        filename = "forecast_export.csv"
        # Simple CSV export
        if "forecasts" in data:
            with open(filename, "w") as f:
                f.write("date,day_of_week,predicted_volume,confidence,model\n")
                for fcast in data["forecasts"]:
                    f.write(f"{fcast['date']},{fcast['day_of_week']},{fcast['predicted_volume']},{fcast['confidence']},{fcast['model']}\n")
        print(f"✅ Exported to {filename}")
    
    return {"filename": filename}


def cmd_data(args: str):
    """Generate and show sample data."""
    print("📥 Generating synthetic sales data...")
    
    stations = ["NBO001", "NBO002", "MBA001", "KIS001"]
    products = ["super", "diesel", "kerosene"]
    
    data = generate_sales_data(stations, products, days=30)
    
    print_header(f"📊 Sample Data: {len(data)} records")
    print(f"Stations: {', '.join(stations)}")
    print(f"Products: {', '.join(products)}")
    print(f"Period: 30 days\n")
    
    # Show sample
    print("Sample records:")
    print("| Date       | Station | Product   | Volume (L) |")
    print("|------------|---------|-----------|------------|")
    for row in data[:10]:
        print(f"| {row['date']} | {row['station_id']} | {row['product']:<9} | {row['volume_liters']:>10,} |")
    
    if len(data) > 10:
        print(f"... and {len(data) - 10} more records")
    
    return {"records": len(data), "sample": data[:5]}


# ============================================================
# MAIN CLI
# ============================================================

def main():
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║     🔮 CYSMIC Fuel Forecasting - CLI Demo                 ║
║        Agentic AI for Kenya Oil & Gas Retail              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    print("Available commands:")
    print("  forecast [station] [product]  - Run demand forecast")
    print("  inventory [station]            - Check inventory status")
    print("  ask <question>                - Query with natural language")
    print("  export [json|csv]             - Export last forecast")
    print("  data                          - Generate sample data")
    print("  quit                          - Exit")
    print()
    
    # Run demo automatically
    print_header("DEMO: Running all commands")
    
    # 1. Generate data
    cmd_data("")
    print()
    
    # 2. Run forecast
    cmd_forecast("NBO001 diesel")
    print()
    
    # 3. Check inventory
    cmd_inventory("NBO001")
    print()
    
    # 4. Ask question
    cmd_ask("Should I order more stock this week?")
    print()
    
    # 5. Export
    forecast_data = generate_forecast("NBO001", "diesel")
    cmd_export(forecast_data, "csv")
    
    print_header("✅ Demo Complete!")
    print("""
To run the full system with real ML models:
1. Install dependencies: pip install -r requirements.txt
2. Start Ollama: ollama serve
3. Run: python -m src.orchestration.langgraph_ollama

Or use the LangGraph CLI directly:
4. python -m src.orchestration.controller
    """)


if __name__ == "__main__":
    main()
