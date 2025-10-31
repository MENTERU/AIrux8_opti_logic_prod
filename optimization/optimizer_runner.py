"""
Simplified optimization runner that orchestrates zone-based optimization.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict

import pandas as pd

from config.utils import get_data_path
from processing.utilities.master_data_loader import master_data_loader_runner
from processing.utilities.weatherapi_client import VisualCrossingWeatherAPIDataFetcher

from .optimizer import Optimizer


class OptimizerRunner:
    """
    Main runner that coordinates zone-based optimization.
    Uses historical data and weather forecasts to determine optimal HVAC operation.
    """

    def __init__(self, store_name: str = "Clea"):
        """
        Initialize the optimization runner.

        Args:
            store_name: Store name (default: Clea)
        """
        self.store_name = store_name
        self.master_data: Dict[str, Any] = {}
        self.weather_data: pd.DataFrame = pd.DataFrame()
        self.results: Dict[str, Any] = {}

    def load_weather_data(self, start_date: str, end_date: str = None):
        """
        Load master data and fetch forecast weather data using weather API.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to start_date if not provided)
        """
        from datetime import datetime

        # Set end_date to start_date if not provided
        if end_date is None:
            end_date = start_date

        # Validate date range
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if start_dt > end_dt:
                raise ValueError(
                    f"Start date {start_date} cannot be after end date {end_date}"
                )
        except ValueError as e:
            if "time data" in str(e):
                raise ValueError(
                    f"Invalid date format. Use YYYY-MM-DD format. Error: {e}"
                )
            else:
                raise e

        # Load master data
        self.master_data = master_data_loader_runner(self.store_name)
        if not self.master_data:
            raise ValueError(f"Failed to load master data for store {self.store_name}")

        # Load forecast weather data using weather API
        coordinates = self.master_data["store_info"]["coordinates"]
        if not coordinates:
            raise ValueError(f"No coordinates found for store {self.store_name}")

        # Import API key from config
        from config.private_information import WEATHER_API_KEY

        api_key = WEATHER_API_KEY

        # Check for cached weather forecast first
        output_data_path = get_data_path("output_data_path")
        plan_dir = os.path.join(output_data_path, self.store_name)

        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")
        cached_filename = f"weather_forecast_{start_clean}_{end_clean}.csv"
        cached_path = os.path.join(plan_dir, cached_filename)

        if os.path.exists(cached_path):
            logging.info(f"Loading cached weather forecast: {cached_path}")
            try:
                self.weather_data = pd.read_csv(cached_path)
                if "datetime" in self.weather_data.columns:
                    self.weather_data["datetime"] = pd.to_datetime(
                        self.weather_data["datetime"]
                    )
                logging.info(
                    f"Cached weather data loaded. Shape: {self.weather_data.shape}"
                )
                return
            except Exception as e:
                logging.warning(f"Error loading cached weather data: {e}")

        logging.info(f"Fetching weather forecast from {start_date} to {end_date}")
        logging.info(f"Using coordinates: {coordinates}")

        # Create weather API client and fetch forecast data
        weather_fetcher = VisualCrossingWeatherAPIDataFetcher(
            coordinates=coordinates,
            start_date=start_date,
            end_date=end_date,
            unit="metric",
            api_key=api_key,
            temperature_col_name="Outdoor Temp.",
            humidity_col_name="Outdoor Humidity",
            solar_col_name="Solar Radiation",
            batch_size_months=1,
            delay_between_requests=1.0,
        )

        # Fetch weather data
        self.weather_data = weather_fetcher.fetch()

        if self.weather_data is None or self.weather_data.empty:
            raise ValueError(
                f"Failed to fetch weather data for period {start_date} to {end_date}"
            )

        # Save to cache
        try:
            os.makedirs(plan_dir, exist_ok=True)
            self.weather_data.to_csv(cached_path, index=False, encoding="utf-8-sig")
            logging.info(f"Weather forecast cached to: {cached_path}")
        except Exception as e:
            logging.warning(f"Error saving weather forecast to cache: {e}")

        logging.info(f"Loaded master data for store {self.store_name}")
        logging.info(
            f"Fetched weather forecast for {start_date} to {end_date}: {len(self.weather_data)} hours"
        )

    def run_optimization(
        self, start_date: str = None, end_date: str = None, strategy: str = "hourly"
    ) -> Dict[str, Any]:
        """
        Run the zone-based optimization for the specified date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format (defaults to start_date if not provided)

        Returns:
            Dictionary containing optimization results
        """
        # load processed features CSV
        processed_data_path = get_data_path("processed_data_path")
        self.features_csv_path = os.path.join(
            processed_data_path, self.store_name, "features_processed_Clea.csv"
        )
        if not os.path.exists(self.features_csv_path):
            raise FileNotFoundError(f"Features CSV not found: {self.features_csv_path}")

        # Set default dates if not provided (today + 6 days = one week period)
        if start_date is None:
            today = datetime.now().date()
            start_date = today.strftime("%Y-%m-%d")
            end_date = (today + timedelta(days=6)).strftime("%Y-%m-%d")
            logging.info(
                f"No dates provided, using default period: {start_date} to {end_date}"
            )
        elif end_date is None:
            end_date = start_date

        # Validate date range
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            if start_dt > end_dt:
                raise ValueError(
                    f"Start date {start_date} cannot be after end date {end_date}"
                )
        except ValueError as e:
            if "time data" in str(e):
                raise ValueError(
                    f"Invalid date format. Use YYYY-MM-DD format. Error: {e}"
                )
            else:
                raise e

        logging.info(
            f"Running zone-based optimization for {start_date} to {end_date}..."
        )
        try:
            # Load data (master data and weather forecast)
            self.load_weather_data(start_date, end_date)

            # Run optimization
            self.optimizer = Optimizer(use_operating_hours=False, strategy=strategy)
            result_df = self.optimizer.optimize_all_zones(
                forecast_df=self.weather_data,
                features_csv_path=self.features_csv_path,
                master_data=self.master_data,
            )

            if result_df.empty:
                logging.warning("Zone optimization returned no data")
                return {
                    "start_date": start_date,
                    "end_date": end_date,
                    "error": "No optimization data",
                }

            self.results["optimization_result"] = result_df
            self.results["status"] = "success"
            self.results["start_date"] = start_date
            self.results["end_date"] = end_date

            logging.info(
                f"Zone optimization completed successfully: {len(result_df)} results"
            )
            return self.results

        except Exception as e:
            logging.error(f"Zone optimization failed: {e}")
            return {
                "start_date": start_date,
                "end_date": end_date,
                "error": f"Optimization failed: {str(e)}",
            }

    def save_results_to_csv(
        self, start_date: str = None, end_date: str = None, filename: str = None
    ) -> str:
        """
        Save optimization results to CSV.

        Args:
            start_date: Start date in YYYY-MM-DD format (uses result data if None)
            end_date: End date in YYYY-MM-DD format (defaults to start_date if not provided)
            filename: Optional custom CSV filename

        Returns:
            Path to the saved CSV file
        """
        if "optimization_result" not in self.results:
            raise ValueError("No optimization results to save. Run optimization first.")

        # Use dates from results if not provided
        if start_date is None:
            start_date = self.results.get("start_date", "unknown")
        if end_date is None:
            end_date = self.results.get("end_date", start_date)

        # Create output directory in Clea folder
        output_data_path = get_data_path("output_data_path")
        output_dir = os.path.join(output_data_path, self.store_name)
        os.makedirs(output_dir, exist_ok=True)

        # Generate filename with start and end dates
        if filename is None:
            # Convert dates to YYYYMMDD format
            start_date_formatted = start_date.replace("-", "")
            end_date_formatted = end_date.replace("-", "")
            filename = (
                f"optimized_results_{start_date_formatted}_{end_date_formatted}.csv"
            )

        output_path = os.path.join(output_dir, filename)

        # Save results
        self.results["optimization_result"].to_csv(output_path, index=False)

        logging.info(f"Optimization results saved to: {output_path}")
        return output_path
