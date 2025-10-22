"""
Simplified optimization runner that orchestrates both optimization phases.
Delegates CSV generation to CSVGenerator.
"""

import logging
from typing import Any, Dict

import pandas as pd
from module_preprocess import PreprocessingRunner
from utils.weather import get_weather_manager

from config.config_common import PREPROCESSED_DATA_DIR

from .csv_generator import CSVGenerator
from .daytime import DaytimeOptimizer
from .precooling import PreCoolingOptimizer


class OptimizationRunner:
    """
    Main runner that coordinates both pre-cooling and daytime optimization.
    Uses historical data and weather forecasts to determine optimal HVAC operation.
    """

    def __init__(self, data_dir: str = PREPROCESSED_DATA_DIR):
        """
        Initialize the optimization runner.

        Args:
            data_dir: Directory for preprocessed data (default from config)
        """
        self.preprocessing_runner = PreprocessingRunner(data_dir)
        self.precooling_optimizer: PreCoolingOptimizer = None
        self.daytime_optimizer: DaytimeOptimizer = None
        self.csv_generator = CSVGenerator()
        self.results: Dict[str, Any] = {}

    def load_all_data(self, target_date: str):
        """
        Load historical patterns, past operations, and forecast weather data.

        Args:
            target_date: Target date in YYYY-MM-DD format
        """
        # Preprocess historical data
        data_dict, past_data = self.preprocessing_runner.run_preprocessing()

        # Load forecast weather data
        weather_manager = get_weather_manager()
        weather_data = weather_manager.get_weather_data(target_date)

        # Initialize optimizers with data
        self.precooling_optimizer = PreCoolingOptimizer(data_dict, weather_data)
        self.daytime_optimizer = DaytimeOptimizer(data_dict, past_data, weather_data)

        logging.info(
            f"Pre-cooling: {len(self.precooling_optimizer.data_dict)} patterns loaded"
        )
        logging.info(
            f"Daytime: {len(self.daytime_optimizer.data_dict)} patterns loaded"
        )

    def run_daytime_optimization(self, target_date: str) -> Dict[str, Any]:
        """
        Run daytime optimization for the target date.

        Args:
            target_date: Target date in YYYY-MM-DD format

        Returns:
            Dictionary with daytime optimization results
        """
        logging.info(f"Running daytime optimization for {target_date}...")
        try:
            result_df = self.daytime_optimizer.optimize_operation(target_date)

            if result_df is None or (hasattr(result_df, "empty") and result_df.empty):
                logging.warning("Daytime optimization returned no data")
                return {"target_date": target_date, "error": "No optimization data"}

            optimal_units = None
            if "最適台数" in result_df.columns and not result_df["最適台数"].empty:
                optimal_units = int(result_df["最適台数"].mode().iloc[0])

            return {
                "daytime_result": result_df,
                "optimal_units": optimal_units,
                "status": "success",
            }

        except Exception as e:
            logging.error(f"Daytime optimization failed: {e}")
            return {
                "target_date": target_date,
                "error": f"Optimization failed: {str(e)}",
            }

    def run_optimization(self, target_date: str) -> Dict[str, Any]:
        """
        Run the optimization workflow for a target date.

        Args:
            target_date: Target date in YYYY-MM-DD format

        Returns:
            Dictionary containing all optimization results
        """
        logging.info(f"Running optimization for {target_date}...")

        # Run daytime optimization
        daytime_result = self.run_daytime_optimization(target_date)
        self.results["daytime"] = daytime_result

        # Generate output CSV
        output_df = self.generate_output_csv(target_date)
        self.results["output_df"] = output_df

        logging.info("Complete optimization finished successfully")
        return self.results

    def generate_output_csv(self, target_date: str) -> pd.DataFrame:
        """
        Generate the output CSV from optimization results.

        Args:
            target_date: Target date in YYYY-MM-DD format

        Returns:
            DataFrame containing combined pre-cooling and daytime results
        """
        return self.csv_generator.generate_output_csv(target_date, self.results)

    def save_results_to_csv(self, target_date: str, filename: str = None) -> str:
        """
        Save optimization results to CSV.

        Args:
            target_date: Target date in YYYY-MM-DD format
            filename: Optional custom CSV filename

        Returns:
            Path to the saved CSV file
        """
        return self.csv_generator.save_results_to_csv(
            target_date, self.results, filename
        )
