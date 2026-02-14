"""
Data Ingestion Module
Handles pulling data from POS systems, external APIs, and station metadata.
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIngestor:
    """Main data ingestion class for fuel demand forecasting."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config {config_path} not found, using defaults")
            return {}
    
    def ingest_pos_data(self, source: str = "csv") -> pd.DataFrame:
        """
        Ingest Point of Sale data.
        
        Args:
            source: Data source type ('csv', 'api', 'database')
            
        Returns:
            DataFrame with columns: date, station_id, product_type, volume, revenue
        """
        logger.info("Ingesting POS data...")
        
        if source == "csv":
            # Placeholder for actual CSV loading
            # df = pd.read_csv(self.data_dir / "sales_data.csv")
            df = self._generate_sample_data()
        elif source == "api":
            # Placeholder for API call
            raise NotImplementedError("API source not implemented")
        else:
            raise ValueError(f"Unknown source: {source}")
        
        logger.info(f"Loaded {len(df)} records from POS")
        return df
    
    def ingest_weather_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch weather data from Kenya Met API.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            
        Returns:
            DataFrame with weather data
        """
        logger.info("Ingesting weather data...")
        
        # Placeholder for actual API call
        # response = requests.get(
        #     "https://api.weather.go.ke/",
        #     params={"start": start_date, "end": end_date}
        # )
        
        # Generate sample data for now
        dates = pd.date_range(start_date, end_date, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'temperature': 20 + 5 * pd.np.random.randn(len(dates)),
            'precipitation': pd.np.random.exponential(2, len(dates)),
            'humidity': 60 + 20 * pd.np.random.randn(len(dates))
        })
        
        return df
    
    def ingest_external_events(self) -> pd.DataFrame:
        """
        Ingest calendar of external events (holidays, elections, etc.)
        
        Returns:
            DataFrame with event information
        """
        logger.info("Ingesting external events...")
        
        # Sample events - in production, scrape from web
        events = [
            {"date": "2026-04-03", "event": "Good Friday", "type": "holiday", "impact": "high"},
            {"date": "2026-04-05", "event": "Easter Sunday", "type": "holiday", "impact": "high"},
            {"date": "2026-06-01", "event": "Madaraka Day", "type": "holiday", "impact": "medium"},
            {"date": "2026-08-10", "event": "Elections", "type": "election", "impact": "very_high"},
            {"date": "2026-10-20", "event": "Diwali", "type": "festival", "impact": "medium"},
            {"date": "2026-12-25", "event": "Christmas", "type": "holiday", "impact": "high"},
        ]
        
        return pd.DataFrame(events)
    
    def ingest_station_metadata(self) -> pd.DataFrame:
        """
        Ingest station information (location, capacity, competitors)
        
        Returns:
            DataFrame with station metadata
        """
        logger.info("Ingesting station metadata...")
        
        # Sample metadata
        stations = [
            {"station_id": "NBO001", "name": "Nairobi Central", "region": "Nairobi", 
             "capacity_liters": 50000, "pumps": 8, "competitors_nearby": 2},
            {"station_id": "NBO002", "name": "Nairobi West", "region": "Nairobi",
             "capacity_liters": 40000, "pumps": 6, "competitors_nearby": 3},
            {"station_id": "MBA001", "name": "Mombasa Road", "region": "Coast",
             "capacity_liters": 60000, "pumps": 10, "competitors_nearby": 1},
        ]
        
        return pd.DataFrame(stations)
    
    def ingest_prices(self) -> pd.DataFrame:
        """Ingest historical fuel prices from EPRA."""
        logger.info("Ingesting fuel prices...")
        
        dates = pd.date_range("2025-01-01", "2026-02-14", freq='W')
        df = pd.DataFrame({
            'date': dates,
            'super': 180 + 10 * pd.np.sin(pd.np.arange(len(dates)) * 0.1) + pd.np.random.randn(len(dates)) * 2,
            'diesel': 170 + 8 * pd.np.sin(pd.np.arange(len(dates)) * 0.1) + pd.np.random.randn(len(dates)) * 2,
            'kerosene': 150 + 5 * pd.np.sin(pd.np.arange(len(dates)) * 0.1) + pd.np.random.randn(len(dates)) * 2,
        })
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> dict:
        """
        Validate data quality and flag anomalies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        logger.info("Validating data quality...")
        
        metrics = {
            "total_records": len(df),
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": df.duplicated().sum(),
            "outliers": self._detect_outliers(df)
        }
        
        return metrics
    
    def _detect_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> dict:
        """Detect outliers using z-score method."""
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        outliers = {}
        
        for col in numeric_cols:
            z_scores = (df[col] - df[col].mean()) / df[col].std()
            outliers[col] = int((z_scores.abs() > threshold).sum())
        
        return outliers
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate sample POS data for development."""
        np = pd.np
        dates = pd.date_range("2025-01-01", "2026-02-14", freq='D')
        stations = ["NBO001", "NBO002", "MBA001", "KSM001", "NKS001"]
        products = ["super", "diesel", "kerosene"]
        
        data = []
        for station in stations:
            for product in products:
                base_volume = np.random.randint(5000, 20000)
                for date in dates:
                    # Add seasonality
                    month_factor = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)
                    day_factor = 1 + 0.1 * np.sin(2 * np.pi * date.day / 30)
                    weekday_factor = 1.2 if date.weekday() < 5 else 1.0
                    
                    volume = base_volume * month_factor * day_factor * weekday_factor
                    volume += np.random.randint(-500, 500)
                    volume = max(0, volume)
                    
                    data.append({
                        'date': date,
                        'station_id': station,
                        'product_type': product,
                        'volume_liters': int(volume),
                        'revenue': int(volume * np.random.uniform(150, 180))
                    })
        
        return pd.DataFrame(data)
    
    def run_full_pipeline(self) -> dict:
        """Execute complete data ingestion pipeline."""
        logger.info("Running full data ingestion pipeline...")
        
        results = {
            "pos_data": self.ingest_pos_data(),
            "weather": self.ingest_weather_data(
                datetime(2025, 1, 1), 
                datetime(2026, 2, 14)
            ),
            "events": self.ingest_external_events(),
            "stations": self.ingest_station_metadata(),
            "prices": self.ingest_prices()
        }
        
        logger.info("Data ingestion complete!")
        return results


if __name__ == "__main__":
    ingestor = DataIngestor()
    data = ingestor.run_full_pipeline()
    print(f"Loaded {len(data['pos_data'])} POS records")
