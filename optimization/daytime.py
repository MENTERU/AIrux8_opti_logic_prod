"""
Daytime optimization module for HVAC system.
Extracted from AirCon_daytime_optimizer.ipynb.
Optimizes the number of active units and power usage during daytime (08:30–19:30),
using historical and forecast weather data.
"""

import logging
from datetime import datetime
from typing import Dict

import pandas as pd

from config.config_common import TARGET_INDOOR_TEMP, TIME_WEIGHTS

from .base_optimizer import BaseOptimizer


class DaytimeOptimizer(BaseOptimizer):
    """
    Daytime optimization for HVAC system.
    Inherits common functionality from BaseOptimizer.
    Operates during the window 08:30–19:30.
    """

    def __init__(
        self, data_dict: Dict, past_data: pd.DataFrame, weather_data: pd.DataFrame
    ):
        """
        Initialize DaytimeOptimizer with data sources.

        Args:
            data_dict: Dictionary of historical simulation data
            past_data: Past data DataFrame used for similarity comparisons
            weather_data: Forecast weather DataFrame
        """
        self.data_dict = data_dict
        self.past_data = past_data
        self.weather_data = weather_data

    def optimize_operation(self, target_date: str) -> pd.DataFrame:
        """
        Optimize HVAC operation for the given target date.

        Ensures a 30-min interval grid exists and applies the best pattern for 08:30 slot
        or the nearest available slot if missing.

        Args:
            target_date: Date to optimize in YYYY-MM-DD format

        Returns:
            DataFrame with optimal daytime operation for the date
        """
        logging.info("DaytimeOptimizer.optimize_operation start")

        # Find the optimal operation pattern using past data and forecast
        optim_result = self.find_optimal_operation(
            self.past_data, self.weather_data, time_weights=TIME_WEIGHTS
        )

        # Define daytime operation window
        daytime_start = datetime.strptime(f"{target_date} 08:30", "%Y-%m-%d %H:%M")
        daytime_end = datetime.strptime(f"{target_date} 19:30", "%Y-%m-%d %H:%M")

        # Filter results to daytime window
        mask = optim_result["datetime"].between(daytime_start, daytime_end)
        output_daytime = optim_result[mask]

        return output_daytime

    def find_optimal_operation(
        self,
        past_data: pd.DataFrame,
        forecast_data: pd.DataFrame,
        time_weights: dict = None,
        temp_margin: float = 0.5,
    ) -> pd.DataFrame:
        """
        Find the optimal HVAC operation based on historical data similarity,
        forecasted temperature, and weighted scoring by time-of-day.

        Args:
            past_data: Historical DataFrame containing past HVAC simulations
            forecast_data: Forecast weather DataFrame
            time_weights: Optional dictionary specifying importance of temperature vs power
                          for morning/afternoon/evening periods
            temp_margin: Temperature tolerance for matching historical data

        Returns:
            DataFrame with optimal unit counts, power, indoor temperature, and scores
        """
        # Default time-of-day weights if not provided
        if time_weights is None:
            time_weights = {
                "morning": {
                    "temp": 0.8,
                    "power": 0.2,
                },  # 08:00–11:30, prioritize temperature
                "afternoon": {
                    "temp": 0.5,
                    "power": 0.5,
                },  # 12:00–16:30, balance temp and power
                "evening": {
                    "temp": 0.3,
                    "power": 0.7,
                },  # 17:00–19:00, prioritize energy saving
            }

        def get_time_period(hour: int) -> str:
            """Determine time-of-day period based on hour."""
            if 8 <= hour <= 11:
                return "morning"
            elif 12 <= hour <= 16:
                return "afternoon"
            elif 17 <= hour <= 19:
                return "evening"
            else:
                return "morning"  # Default fallback

        result_rows = []

        # Iterate over forecasted timestamps
        for _, row in forecast_data.iterrows():
            forecast_time = pd.to_datetime(row["datetime"])
            forecast_temp = row["temperature"]
            hour = forecast_time.hour

            # Determine period and corresponding weights
            period = get_time_period(hour)
            weights = time_weights[period]

            # Filter historical data matching the forecast time and temperature
            target_time_str = forecast_time.strftime("%H:%M")
            past_subset = past_data[
                past_data["datetime"].dt.strftime("%H:%M") == target_time_str
            ].copy()
            past_subset = past_subset[
                abs(past_subset["外気温"] - forecast_temp) <= temp_margin
            ]
            past_subset = past_subset[
                past_subset["電力量"] > 0
            ]  # Exclude zero-power entries

            if past_subset.empty:
                continue

            # Compute indoor temperature difference relative to target
            past_subset["temp_diff"] = abs(past_subset["室内温度"] - TARGET_INDOOR_TEMP)

            # Compute statistics for Z-score normalization
            temp_mean, temp_std = (
                past_subset["temp_diff"].mean(),
                past_subset["temp_diff"].std() or 1,
            )
            power_mean, power_std = (
                past_subset["電力量"].mean(),
                past_subset["電力量"].std() or 1,
            )

            # Calculate weighted score for each historical record
            past_subset["score"] = (
                weights["temp"] * (past_subset["temp_diff"] - temp_mean) / temp_std
                + weights["power"] * (past_subset["電力量"] - power_mean) / power_std
            )

            # Select the best historical record (minimum score)
            best_row = past_subset.loc[past_subset["score"].idxmin()]

            # Append result with relevant metrics
            result_rows.append(
                {
                    "datetime": forecast_time,
                    "予報気温": forecast_temp,
                    "外気温": best_row["外気温"],
                    "最適台数": best_row["台数"],
                    "予冷時間": best_row["予冷時間"],
                    "室内温度": best_row["室内温度"],
                    "最適電力量": best_row["電力量"],
                    "スコア": best_row["score"],
                    "時間帯": period,
                    "温度重み": weights["temp"],
                    "電力重み": weights["power"],
                    "diffuse_solar_radiation": row["diffuse_solar_radiation"],
                }
            )

        return pd.DataFrame(result_rows)
