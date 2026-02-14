"""
Orchestration Module
Implements the agentic controller that coordinates ML models and LLM agents
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import json

from ..data.ingest import DataIngestor
from ..models.forecast import DemandForecaster, FeatureEngineer
from ..agents.llm_agent import LLMAgent, ReasoningOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ForecastRequest:
    """Represents a forecast request."""
    station_id: str
    product_type: str
    forecast_horizon: int = 7  # days
    include_llm: bool = True
    include_alerts: bool = True


@dataclass
class ForecastResult:
    """Represents forecast results."""
    station_id: str
    product_type: str
    predictions: List[Dict[str, Any]]
    confidence: float
    alerts: List[str] = field(default_factory=list)
    llm_insights: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ControllerAgent:
    """
    Main controller agent that orchestrates the demand forecasting workflow.
    
    Coordinates:
    - Data ingestion
    - ML model inference
    - LLM reasoning
    - Alert generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize controller agent.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.ingestor = DataIngestor()
        self.forecaster = DemandForecaster()
        self.llm_agent = LLMAgent()
        self.reasoning_orchestrator = ReasoningOrchestrator()
        
        # State
        self.is_initialized = False
        self.last_run = None
        
    def initialize(self):
        """Initialize all components."""
        logger.info("Initializing Controller Agent...")
        
        # Load data
        self.data = self.ingestor.run_full_pipeline()
        
        # Prepare features
        self.feature_engineer = FeatureEngineer()
        
        # Initialize ML models
        pos_data = self.data['pos_data']
        
        X, y = self.forecaster.prepare_data(pos_data)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train models
        self.forecaster.train_xgboost(X_train.values, y_train.values)
        
        # Evaluate
        predictions = self.forecaster.predict_ensemble(X_test.values)
        ensemble_pred = self.forecaster.weighted_ensemble(predictions)
        metrics = self.forecaster.evaluate(y_test.values, ensemble_pred)
        
        logger.info(f"Model metrics: {metrics}")
        
        self.is_initialized = True
        logger.info("Controller Agent initialized successfully!")
    
    def process_request(self, request: ForecastRequest) -> ForecastResult:
        """
        Process a forecast request.
        
        Args:
            request: Forecast request
            
        Returns:
            Forecast results
        """
        logger.info(f"Processing forecast request for {request.station_id} - {request.product_type}")
        
        # Filter data for station and product
        df = self.data['pos_data']
        df = df[(df['station_id'] == request.station_id) & 
                (df['product_type'] == request.product_type)]
        
        if len(df) == 0:
            return ForecastResult(
                station_id=request.station_id,
                product_type=request.product_type,
                predictions=[],
                confidence=0.0,
                alerts=["No data available for this station/product"]
            )
        
        # Generate ML predictions
        X, _ = self.forecaster.prepare_data(df)
        ml_predictions = self.forecaster.predict_ensemble(X.values)
        ml_forecast = self.forecaster.weighted_ensemble(ml_predictions)
        
        # Get LLM insights if requested
        llm_insights = None
        if request.include_llm:
            llm_insights = self._get_llm_insights(df)
        
        # Generate predictions list
        predictions = []
        dates = df['date'].tail(request.forecast_horizon).tolist()
        
        for i, date in enumerate(dates):
            pred_value = ml_forecast[i] if i < len(ml_forecast) else ml_forecast[-1]
            
            # Apply LLM adjustment if available
            if llm_insights and 'adjusted_prediction' in llm_insights:
                pred_value = llm_insights['adjusted_prediction']
            
            predictions.append({
                'date': str(date),
                'predicted_volume': int(pred_value),
                'confidence': 0.85 if self.forecaster.is_fitted else 0.5
            })
        
        # Generate alerts
        alerts = []
        if request.include_alerts:
            alerts = self._generate_alerts(predictions, df)
        
        # Calculate overall confidence
        confidence = 0.85 if self.forecaster.is_fitted else 0.5
        
        return ForecastResult(
            station_id=request.station_id,
            product_type=request.product_type,
            predictions=predictions,
            confidence=confidence,
            alerts=alerts,
            llm_insights=llm_insights,
            metadata={
                'generated_at': str(datetime.now()),
                'model_metrics': self.forecaster.evaluate(
                    df['volume_liters'].values[-len(predictions):],
                    [p['predicted_volume'] for p in predictions]
                ) if self.forecaster.is_fitted else {}
            }
        )
    
    def _get_llm_insights(self, df) -> Dict:
        """Get LLM-based insights."""
        # Get recent data for context
        recent_data = df.tail(30)['volume_liters'].tolist()
        
        # Get upcoming events
        events = self.data['events'].to_dict('records')
        
        # Detect anomalies
        anomaly_result = self.llm_agent.detect_anomalies(
            historical_data=recent_data[:20],
            recent_data=recent_data[-7:]
        )
        
        return {
            'anomaly_detection': anomaly_result,
            'upcoming_events': events[:3]
        }
    
    def _generate_alerts(self, predictions: List[Dict], historical_df) -> List[str]:
        """Generate alerts based on predictions."""
        alerts = []
        
        # Check for stockout risk
        avg_daily = historical_df['volume_liters'].mean()
        for pred in predictions:
            if pred['predicted_volume'] > avg_daily * 1.5:
                alerts.append(f"⚠️ High demand expected on {pred['date']}: {pred['predicted_volume']}L (50% above average)")
        
        # Check for overstock risk
        for pred in predictions:
            if pred['predicted_volume'] < avg_daily * 0.5:
                alerts.append(f"📉 Low demand expected on {pred['date']}: {pred['predicted_volume']}L (50% below average)")
        
        return alerts
    
    def run_scheduled_forecast(self, station_ids: List[str] = None):
        """
        Run scheduled forecast for all stations.
        
        Args:
            station_ids: Optional list of station IDs (default: all)
        """
        logger.info("Running scheduled forecast...")
        
        if station_ids is None:
            station_ids = self.data['stations']['station_id'].tolist()
        
        products = ['super', 'diesel', 'kerosene']
        results = []
        
        for station_id in station_ids:
            for product in products:
                request = ForecastRequest(
                    station_id=station_id,
                    product_type=product,
                    forecast_horizon=7,
                    include_llm=True,
                    include_alerts=True
                )
                result = self.process_request(request)
                results.append(result)
                
                # Log significant alerts
                for alert in result.alerts:
                    logger.warning(f"[{station_id}/{product}] {alert}")
        
        self.last_run = datetime.now()
        logger.info(f"Scheduled forecast complete. Processed {len(results)} station-product combinations")
        
        return results
    
    def handle_manager_query(self, station_id: str, query: str) -> str:
        """
        Handle natural language query from station manager.
        
        Args:
            station_id: Station ID
            query: Manager's question
            
        Returns:
            Natural language response
        """
        # Get forecast context
        request = ForecastRequest(
            station_id=station_id,
            product_type='super',  # Default
            forecast_horizon=7,
            include_llm=False,
            include_alerts=False
        )
        result = self.process_request(request)
        
        context = {
            'forecast_7d': f"{result.predictions[0]['predicted_volume']}L/day" if result.predictions else "N/A",
            'forecast_30d': "See 7-day extended",
            'stock_level': "Unknown",
            'reorder_point': "Unknown",
            'weather': "Normal"
        }
        
        return self.llm_agent.process_manager_query(query, context)


class WorkflowScheduler:
    """Schedules and manages forecast workflows."""
    
    def __init__(self, controller: ControllerAgent):
        self.controller = controller
        self.schedule = {}
    
    def add_daily_forecast(self, time: str = "06:00"):
        """
        Add daily forecast job.
        
        Args:
            time: Time to run daily forecast (HH:MM format)
        """
        self.schedule['daily'] = {
            'time': time,
            'action': self.controller.run_scheduled_forecast
        }
        logger.info(f"Daily forecast scheduled at {time}")
    
    def add_weekly_report(self, day: str = "monday", time: str = "09:00"):
        """
        Add weekly report job.
        
        Args:
            day: Day of week
            time: Time to run
        """
        self.schedule['weekly'] = {
            'day': day,
            'time': time,
            'action': self.controller.run_scheduled_forecast
        }
        logger.info(f"Weekly report scheduled for {day} at {time}")
    
    def run(self):
        """Execute all scheduled jobs."""
        logger.info("Running scheduled jobs...")
        
        for job_name, job_config in self.schedule.items():
            try:
                action = job_config.get('action')
                if action:
                    action()
            except Exception as e:
                logger.error(f"Error running {job_name}: {e}")


if __name__ == "__main__":
    # Initialize and run
    controller = ControllerAgent()
    controller.initialize()
    
    # Test forecast
    request = ForecastRequest(
        station_id="NBO001",
        product_type="super",
        forecast_horizon=7
    )
    result = controller.process_request(request)
    
    print(f"\nForecast for {result.station_id} - {result.product_type}")
    print(f"Confidence: {result.confidence:.0%}")
    print("\nPredictions:")
    for pred in result.predictions:
        print(f"  {pred['date']}: {pred['predicted_volume']}L")
    
    if result.alerts:
        print("\nAlerts:")
        for alert in result.alerts:
            print(f"  {alert}")
