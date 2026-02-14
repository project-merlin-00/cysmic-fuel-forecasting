"""
LLM Agent Module
Implements the reasoning layer using Large Language Models
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LLMAgent:
    """
    LLM-based agent for qualitative reasoning in demand forecasting.
    Processes news, events, and provides contextual adjustments.
    """
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        """
        Initialize LLM Agent.
        
        Args:
            provider: LLM provider ('openai', 'anthropic', 'ollama')
            model: Model name
        """
        self.provider = provider
        self.model = model
        self.client = self._init_client()
    
    def _init_client(self):
        """Initialize LLM client based on provider."""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OPENAI_API_KEY not set")
                return OpenAI(api_key=api_key) if api_key else None
            except ImportError:
                logger.warning("OpenAI not installed")
                return None
        
        elif self.provider == "anthropic":
            try:
                from anthropic import Anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    logger.warning("ANTHROPIC_API_KEY not set")
                return Anthropic(api_key=api_key) if api_key else None
            except ImportError:
                logger.warning("Anthropic not installed")
                return None
        
        elif self.provider == "ollama":
            # Use local Ollama instance
            return {"provider": "ollama", "url": "http://localhost:11434"}
        
        return None
    
    def analyze_news_impact(self, news_items: List[Dict]) -> Dict[str, Any]:
        """
        Analyze news items and predict demand impact.
        
        Args:
            news_items: List of news articles with title, date, source
            
        Returns:
            Dictionary with impact analysis
        """
        if not self.client:
            logger.warning("No LLM client available, returning default analysis")
            return self._default_analysis()
        
        # Prepare news summary
        news_summary = "\n".join([
            f"- {item.get('title', 'No title')} ({item.get('date', '')})"
            for item in news_items[:10]
        ])
        
        prompt = f"""You are a fuel demand analyst for Kenya. Given the following recent news, 
analyze how each item might impact fuel demand at retail stations in the next 1-2 weeks.

News:
{news_summary}

For each significant news item, provide:
1. Demand impact: -3 (major decrease) to +3 (major increase)
2. Affected fuel types: 'super', 'diesel', 'kerosene', or 'all'
3. Duration: 'short-term' (days), 'medium-term' (weeks), 'long-term' (months)
4. Reasoning (1 sentence)

Return your analysis as JSON with this structure:
{{
  "overall_impact": "positive" | "negative" | "neutral",
  "confidence": 0.0-1.0,
  "adjustments": [
    {{"factor": "news_item", "impact": -3 to +3, "fuel_type": "all", "duration": "short-term", "reasoning": "..."}}
  ]
}}
"""
        
        try:
            response = self._call_llm(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error analyzing news: {e}")
            return self._default_analysis()
    
    def analyze_event_impact(self, events: List[Dict]) -> Dict[str, Any]:
        """
        Analyze upcoming events and predict demand impact.
        
        Args:
            events: List of events with date, name, type, expected_impact
            
        Returns:
            Event impact analysis
        """
        prompt = f"""You are a fuel demand analyst for Kenya. Analyze how these upcoming 
events/holidays might affect fuel demand at retail stations.

Events:
{json.dumps(events, indent=2)}

For each event, provide:
1. Expected demand change: percentage (-50% to +100%)
2. Peak timing: 'before', 'during', 'after'
3. Affected regions (if specific): list of regions
4. Reasoning

Return as JSON:
{{
  "overall_event_impact": "positive" | "negative" | "mixed",
  "confidence": 0.0-1.0,
  "event_impacts": [
    {{"event": "...", "demand_change_pct": -20, "peak_timing": "before", "reasoning": "..."}}
  ]
}}
"""
        
        try:
            response = self._call_llm(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"Error analyzing events: {e}")
            return {"confidence": 0, "event_impacts": []}
    
    def process_manager_query(self, query: str, context: Dict) -> str:
        """
        Process natural language query from station manager.
        
        Args:
            query: Manager's question
            context: Current forecast data
            
        Returns:
            Natural language response
        """
        prompt = f"""You are a helpful assistant for fuel station managers. 
Answer their question based on the forecast data provided.

Manager's Question: {query}

Forecast Data:
- 7-day forecast: {context.get('forecast_7d', 'N/A')}
- 30-day forecast: {context.get('forecast_30d', 'N/A')}
- Current stock level: {context.get('stock_level', 'N/A')}
- Reorder point: {context.get('reorder_point', 'N/A')}
- Weather outlook: {context.get('weather', 'N/A')}

Provide a clear, actionable answer in 2-3 sentences. If recommending action, be specific.
"""
        
        try:
            response = self._call_llm(prompt)
            return response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I'm sorry, I couldn't process your query at this time. Please try again."
    
    def detect_anomalies(self, historical_data: List[float], 
                        recent_data: List[float]) -> Dict[str, Any]:
        """
        Detect anomalies in recent demand patterns.
        
        Args:
            historical_data: Historical demand values
            recent_data: Recent demand values
            
        Returns:
            Anomaly detection results
        """
        import numpy as np
        
        if len(historical_data) < 7:
            return {"anomaly_detected": False, "confidence": 0}
        
        # Simple statistical detection
        hist_mean = np.mean(historical_data)
        hist_std = np.std(historical_data)
        recent_mean = np.mean(recent_data)
        
        z_score = (recent_mean - hist_mean) / hist_std if hist_std > 0 else 0
        
        if abs(z_score) > 2:
            anomaly = True
            direction = "above" if z_score > 0 else "below"
            explanation = f"Recent demand is {abs(z_score):.1f} standard deviations {direction} historical average."
        else:
            anomaly = False
            explanation = "No significant anomalies detected."
        
        return {
            "anomaly_detected": anomaly,
            "z_score": z_score,
            "explanation": explanation,
            "confidence": min(abs(z_score) / 3, 1.0)
        }
    
    def _call_llm(self, prompt: str) -> str:
        """Make API call to LLM."""
        if self.provider == "openai" and self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        
        elif self.provider == "anthropic" and self.client:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        
        elif self.provider == "ollama":
            # Fallback to local Ollama
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False
                }
            )
            return response.json().get("response", "")
        
        return "{}"
    
    def _default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when LLM unavailable."""
        return {
            "overall_impact": "neutral",
            "confidence": 0.0,
            "adjustments": []
        }


class ReasoningOrchestrator:
    """
    Orchestrates LLM reasoning with ML predictions.
    Combines quantitative (ML) and qualitative (LLM) inputs.
    """
    
    def __init__(self):
        self.llm_agent = LLMAgent()
    
    def generate_forecast_adjustment(self, 
                                     ml_prediction: float,
                                     news_items: List[Dict],
                                     events: List[Dict],
                                     weather_outlook: str = "normal") -> Dict[str, Any]:
        """
        Generate final forecast with LLM adjustments.
        
        Args:
            ml_prediction: Base ML prediction
            news_items: Recent news
            events: Upcoming events
            weather_outlook: Weather forecast
            
        Returns:
            Adjusted forecast with reasoning
        """
        # Get news impact
        news_impact = self.llm_agent.analyze_news_impact(news_items)
        
        # Get event impact
        event_impact = self.llm_agent.analyze_event_impact(events)
        
        # Calculate adjustment factor
        news_factor = 1 + (news_impact.get("confidence", 0) * 
                          sum(a.get("impact", 0) for a in news_impact.get("adjustments", [])) / 10)
        
        event_factor = 1 + (event_impact.get("confidence", 0) * 
                           sum(e.get("demand_change_pct", 0) for e in event_impact.get("event_impacts", [])) / 100)
        
        # Weather adjustment
        weather_factors = {
            "rainy": 0.95,  # Less travel
            "sunny": 1.05,  # More travel
            "normal": 1.00
        }
        weather_factor = weather_factors.get(weather_outlook, 1.0)
        
        # Combined adjustment
        final_factor = news_factor * event_factor * weather_factor
        adjusted_prediction = ml_prediction * final_factor
        
        return {
            "base_prediction": ml_prediction,
            "adjusted_prediction": adjusted_prediction,
            "adjustment_factor": final_factor,
            "news_impact": news_impact,
            "event_impact": event_impact,
            "weather_factor": weather_factor,
            "reasoning": f"ML: {ml_prediction:.0f} → Adjusted: {adjusted_prediction:.0f} "
                         f"(news: {news_factor:.2f}, events: {event_factor:.2f}, weather: {weather_factor:.2f})"
        }


if __name__ == "__main__":
    agent = LLMAgent()
    print(f"LLM Agent initialized with provider: {agent.provider}")
