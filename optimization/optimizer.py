"""
Zone-based HVAC optimization module.
Implements the algorithm from the image to find optimal AC settings per zone
by matching similar historical weather patterns and selecting the best-performing
settings that minimize power consumption while maintaining comfort.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.utils import get_data_path
from processing.utilities.master_data_loader import (
    get_comfort_range,
    get_zone_operating_hours,
)


class Optimizer:
    """
    Zone-based HVAC optimization class.

    This class implements the optimization algorithm to find optimal AC settings
    per zone by matching similar historical weather patterns and selecting the
    best-performing settings that minimize power consumption while maintaining comfort.
    """

    def __init__(self):
        """
        Initialize the Optimizer with configuration.

        Args:
            config: Optional configuration dictionary to override defaults
        """
        # Configuration Constants
        self.TARGET_INDOOR_TEMP = 26.0  # default in case there is no gategory mapping
        self.WEATHER_WEIGHTS = {"temperature": 0.7, "solar_radiation": 0.3}
        self.TIME_WEIGHTS = {
            "morning": {"temp": 0.8, "power": 0.2},
            "afternoon": {"temp": 0.5, "power": 0.5},
            "evening": {"temp": 0.3, "power": 0.7},
        }
        self.TEMP_TOLERANCE = 0.5  # ±0.5°C for weather matching
        # self.SOLAR_TOLERANCE = 100.0  # ±100 W/m² for solar radiation matching

        # Load category mappings
        self.category_mappings = self._load_category_mappings()

    def _load_category_mappings(self) -> Dict:
        """Load category mappings from config file."""
        try:
            # Get the project root directory and navigate to config folder
            # __file__ is optimization/zone_optimizer.py, so go up 2 levels to project root
            project_root = os.path.dirname(os.path.dirname(__file__))
            config_path = os.path.join(project_root, "config", "category_mapping.json")
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Could not load category mappings: {e}")
            return {}

    def _map_ac_mode(self, mode_value: int) -> str:
        """Map AC mode numeric value to string."""
        if not self.category_mappings or "A/C Mode" not in self.category_mappings:
            return str(mode_value)

        mode_mapping = self.category_mappings["A/C Mode"]
        # Reverse mapping: find key by value
        for mode_str, mode_num in mode_mapping.items():
            if mode_num == mode_value:
                return mode_str
        return str(mode_value)

    def _map_fan_speed(self, fan_speed_value: int) -> str:
        """Map fan speed numeric value to string."""
        if not self.category_mappings or "A/C Fan Speed" not in self.category_mappings:
            return str(fan_speed_value)

        fan_mapping = self.category_mappings["A/C Fan Speed"]
        # Reverse mapping: find key by value
        for fan_str, fan_num in fan_mapping.items():
            if fan_num == fan_speed_value:
                return fan_str
        return str(fan_speed_value)

    def _get_zone_target_temp(self, zone: str, month: int, master_data: dict) -> float:
        """
        Get zone-specific target temperature from master data.
        Uses the average of 目標室内温度下限 and 目標室内温度上限.

        Args:
            zone: Zone name
            month: Month number (1-12)
            master_data: Master data dictionary

        Returns:
            Target temperature for the zone and month
        """
        try:
            # Convert month to Japanese format
            month_japanese = f"{month}月"

            # Check if zone exists in master data
            if zone in master_data.get("zones", {}):
                zone_data = master_data["zones"][zone]
                if month_japanese in zone_data:
                    target_temp = zone_data[month_japanese].get("target_room_temp")
                    if target_temp is not None:
                        logging.info(
                            f"Using zone-specific target temp for {zone} in {month_japanese}: {target_temp}°C"
                        )
                        return float(target_temp)

            # Fallback to default
            logging.info(
                f"Using default target temp for {zone} in {month_japanese}: {self.TARGET_INDOOR_TEMP}°C"
            )
            return self.TARGET_INDOOR_TEMP

        except Exception as e:
            logging.warning(f"Error getting zone target temp for {zone}: {e}")
            return self.TARGET_INDOOR_TEMP

    def _map_ac_on_off(self, units_count: float) -> str:
        """Map number of units to ON/OFF string."""
        return "ON" if units_count > 0 else "OFF"

    def load_historical_patterns(self, features_csv_path: str) -> pd.DataFrame:
        """
        Load features_processed_Clea.csv and filter valid records.

        Args:
            features_csv_path: Path to the features CSV file

        Returns:
            DataFrame ready for pattern matching
        """
        logging.info(f"Loading historical patterns from {features_csv_path}")

        # Load the CSV file
        df = pd.read_csv(features_csv_path)

        # Convert datetime column
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        # Filter valid records: non-null Indoor Temp., positive adjusted_power
        valid_mask = (
            df["Indoor Temp."].notna()
            & (df["adjusted_power"] > 0)
            & df["Outdoor Temp."].notna()
            & df["Solar Radiation"].notna()
        )

        df_filtered = df[valid_mask].copy()

        logging.info(
            f"Loaded {len(df_filtered)} valid historical patterns from {len(df)} total records"
        )
        logging.info(
            f"Date range: {df_filtered['Datetime'].min()} to {df_filtered['Datetime'].max()}"
        )
        logging.info(f"Zones: {sorted(df_filtered['zone'].unique())}")

        return df_filtered

    def _find_similar_patterns(
        self,
        historical_df: pd.DataFrame,
        forecast_row: pd.Series,
        zone: str,
        n_top: int = 10,
    ) -> pd.DataFrame:
        """
        Find similar historical weather patterns for a given forecast.

        Args:
            historical_df: Historical data DataFrame
            forecast_row: Single forecast row with datetime, outdoor_temp, solar_radiation
            zone: Zone name to filter by
            n_top: Number of top similar patterns to return

        Returns:
            DataFrame with top N most similar historical records for that zone
        """
        # Filter by zone
        zone_data = historical_df[historical_df["zone"] == zone].copy()

        if len(zone_data) == 0:
            logging.warning(f"No historical data found for zone {zone}")
            return pd.DataFrame()

        # Extract forecast values
        forecast_datetime = pd.to_datetime(forecast_row["datetime"])
        forecast_hour = forecast_datetime.hour
        forecast_temp = forecast_row["Outdoor Temp."]
        forecast_solar = forecast_row["Solar Radiation"]

        # Filter by same hour of day
        zone_data["hour"] = zone_data["Datetime"].dt.hour
        same_hour_data = zone_data[zone_data["hour"] == forecast_hour].copy()

        if len(same_hour_data) == 0:
            logging.warning(
                f"No historical data found for zone {zone} at hour {forecast_hour}"
            )
            return pd.DataFrame()

        # Filter by similar weather conditions
        temp_diff = abs(same_hour_data["Outdoor Temp."] - forecast_temp)
        solar_diff = abs(same_hour_data["Solar Radiation"] - forecast_solar)

        # TODO: Keep for now and discuss with team if we need to add solar radiation tolerance
        # weather_similar_mask = (temp_diff <= self.TEMP_TOLERANCE) & (
        #     solar_diff <= self.SOLAR_TOLERANCE
        # )
        weather_similar_mask = temp_diff <= self.TEMP_TOLERANCE

        similar_hist_data = same_hour_data[weather_similar_mask].copy()

        if len(similar_hist_data) == 0:
            logging.warning(
                f"No similar weather patterns found for zone {zone} at hour {forecast_hour}"
            )
            return pd.DataFrame()

        # Calculate similarity score using z-score normalization
        # Get all historical data for normalization
        all_temp_data = zone_data["Outdoor Temp."].dropna()
        all_solar_data = zone_data["Solar Radiation"].dropna()

        temp_mean, temp_std = all_temp_data.mean(), all_temp_data.std()
        solar_mean, solar_std = all_solar_data.mean(), all_solar_data.std()

        # Calculate z-scores for forecast
        forecast_temp_z = (forecast_temp - temp_mean) / temp_std if temp_std > 0 else 0
        forecast_solar_z = (
            (forecast_solar - solar_mean) / solar_std if solar_std > 0 else 0
        )

        # Calculate z-scores for historical data
        similar_hist_data["temp_z"] = (
            similar_hist_data["Outdoor Temp."] - temp_mean
        ) / temp_std
        similar_hist_data["solar_z"] = (
            similar_hist_data["Solar Radiation"] - solar_mean
        ) / solar_std

        # Calculate similarity score (lower is better)
        similar_hist_data["similarity_score"] = self.WEATHER_WEIGHTS[
            "temperature"
        ] * abs(similar_hist_data["temp_z"] - forecast_temp_z) + self.WEATHER_WEIGHTS[
            "solar_radiation"
        ] * abs(
            similar_hist_data["solar_z"] - forecast_solar_z
        )

        # Sort by similarity score and return top N
        top_similar = similar_hist_data.nsmallest(n_top, "similarity_score")

        logging.info(
            f"Found {len(top_similar)} similar patterns for zone {zone} at hour {forecast_hour}"
        )

        return top_similar

    def _filter_by_comfort(
        self, patterns_df: pd.DataFrame, zone: str, month: int, master_data: dict
    ) -> pd.DataFrame:
        """
        Filter patterns by indoor temperature comfort range.

        Args:
            patterns_df: DataFrame with historical patterns
            zone: Zone name
            month: Month number (1-12)
            master_data: Master data dictionary

        Returns:
            Filtered DataFrame with patterns within comfort range
        """
        if len(patterns_df) == 0:
            return patterns_df

        # Get comfort range from master data
        min_temp, max_temp = get_comfort_range(master_data, zone, month)

        # Filter by comfort range
        comfort_mask = (patterns_df["Indoor Temp."] >= min_temp) & (
            patterns_df["Indoor Temp."] <= max_temp
        )

        filtered_patterns = patterns_df[comfort_mask].copy()

        logging.info(
            f"Filtered {len(filtered_patterns)}/{len(patterns_df)} patterns for zone {zone} within comfort range {min_temp}-{max_temp}°C"
        )

        return filtered_patterns

    def _score_and_select_best(
        self,
        patterns_df: pd.DataFrame,
        hour: int,
        zone: str,
        month: int,
        master_data: dict,
    ) -> Optional[pd.Series]:
        """
        Score patterns and select the best one based on power and temperature.

        Args:
            patterns_df: DataFrame with historical patterns
            hour: Hour of day (0-23)
            zone: Zone name
            month: Month number (1-12)
            master_data: Master data dictionary

        Returns:
            Best pattern row (Series) or None if no patterns
        """
        if len(patterns_df) == 0:
            return None

        # Determine time period
        if 5 <= hour <= 11:
            period = "morning"
        elif 12 <= hour <= 16:
            period = "afternoon"
        elif 17 <= hour <= 23:
            period = "evening"
        else:
            period = "morning"  # Default fallback

        weights = self.TIME_WEIGHTS[period]

        # Get zone-specific target temperature from master data
        target_temp = self._get_zone_target_temp(zone, month, master_data)

        # Calculate temperature deviation from target
        patterns_df["temp_diff"] = abs(patterns_df["Indoor Temp."] - target_temp)

        # Calculate z-score normalized scores
        power_mean, power_std = (
            patterns_df["adjusted_power"].mean(),
            patterns_df["adjusted_power"].std(),
        )
        temp_diff_mean, temp_diff_std = (
            patterns_df["temp_diff"].mean(),
            patterns_df["temp_diff"].std(),
        )

        # Normalize scores
        patterns_df["power_score"] = (
            (patterns_df["adjusted_power"] - power_mean) / power_std
            if power_std > 0
            else 0
        )
        patterns_df["temp_score"] = (
            (patterns_df["temp_diff"] - temp_diff_mean) / temp_diff_std
            if temp_diff_std > 0
            else 0
        )

        # Combined score (lower is better)
        patterns_df["combined_score"] = (
            weights["power"] * patterns_df["power_score"]
            + weights["temp"] * patterns_df["temp_score"]
        )

        # Select best pattern (minimum combined score)
        best_pattern = patterns_df.loc[patterns_df["combined_score"].idxmin()]

        logging.info(
            f"Selected best pattern for hour {hour} ({period}): power={best_pattern['adjusted_power']:.0f}W, temp={best_pattern['Indoor Temp.']:.1f}°C, score={best_pattern['combined_score']:.3f}"
        )

        return best_pattern

    def _optimize_zone_for_forecast(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        zone: str,
        master_data: dict,
    ) -> pd.DataFrame:
        """
        Main optimization function for a single zone.

        Args:
            historical_df: Historical patterns DataFrame
            forecast_df: Forecast weather DataFrame
            zone: Zone name to optimize
            master_data: Master data dictionary

        Returns:
            DataFrame with optimization results for the zone
        """
        # Get zone operating hours from master data
        start_hour, end_hour = get_zone_operating_hours(master_data, zone)

        logging.info(
            f"Optimizing zone {zone} for {len(forecast_df)} forecast hours (operating hours: {start_hour}:00-{end_hour}:00)"
        )

        results = []

        for _, forecast_row in forecast_df.iterrows():
            forecast_datetime = pd.to_datetime(forecast_row["datetime"])
            month = forecast_datetime.month
            hour = forecast_datetime.hour

            # Check if current hour is within operating hours for the zone
            if not (start_hour <= hour < end_hour):
                logging.debug(
                    f"Skipping hour {hour} for zone {zone} (outside operating hours {start_hour}:00-{end_hour}:00)"
                )
                continue

            # Step 1-3: Find similar historical patterns for the zone
            similar_patterns = self._find_similar_patterns(
                historical_df, forecast_row, zone, n_top=20
            )

            if len(similar_patterns) == 0:
                logging.warning(
                    f"No similar patterns found for zone {zone} at {forecast_datetime}"
                )
                continue

            # Step 4: Filter by comfort range for the zone
            comfort_patterns = self._filter_by_comfort(
                similar_patterns, zone, month, master_data
            )

            if len(comfort_patterns) == 0:
                logging.warning(
                    f"No comfort patterns found for zone {zone} at {forecast_datetime}"
                )
                continue

            # Step 5: Score and select best pattern
            best_pattern = self._score_and_select_best(
                comfort_patterns, hour, zone, month, master_data
            )

            if best_pattern is None:
                logging.warning(
                    f"No best pattern selected for zone {zone} at {forecast_datetime}"
                )
                continue

            # Extract recommended settings
            units_count = best_pattern["A/C ON/OFF"]
            result = {
                "datetime": forecast_datetime,
                "zone": zone,
                "set_temp": best_pattern["A/C Set Temperature"],
                "mode": self._map_ac_mode(int(best_pattern["A/C Mode"])),
                "fan_speed": self._map_fan_speed(int(best_pattern["A/C Fan Speed"])),
                "numb_units_on": units_count,
                "ac_on_off": self._map_ac_on_off(units_count),
                "power": best_pattern["adjusted_power"],
                "indoor_temp": best_pattern["Indoor Temp."],
                "similarity_score": round(best_pattern["similarity_score"], 2),
                "combined_score": round(best_pattern["combined_score"], 2),
                "hist_datetime_used": best_pattern["Datetime"],
                "forecast_outdoor_temp": forecast_row["Outdoor Temp."],
                "forecast_solar_radiation": forecast_row["Solar Radiation"],
                "hist_outdoor_temp": best_pattern["Outdoor Temp."],
                "hist_solar_radiation": best_pattern["Solar Radiation"],
                "hist_indoor_temp": best_pattern["Indoor Temp."],
            }

            results.append(result)

        if not results:
            logging.warning(
                f"No optimization results for zone {zone} (operating hours: {start_hour}:00-{end_hour}:00)"
            )
            return pd.DataFrame()

        result_df = pd.DataFrame(results)
        logging.info(
            f"Optimization completed for zone {zone}: {len(result_df)} hours optimized (operating hours: {start_hour}:00-{end_hour}:00)"
        )

        return result_df

    def optimize_all_zones(
        self, forecast_df: pd.DataFrame, features_csv_path: str, master_data: dict
    ) -> pd.DataFrame:
        """
        Optimize all zones for the forecast period.

        Args:
            forecast_df: Forecast weather DataFrame
            features_csv_path: Path to features CSV file
            master_data: Master data dictionary

        Returns:
            Combined DataFrame with optimization results for all zones
        """
        logging.info("Starting optimization for all zones")

        # Load historical patterns
        historical_df = self.load_historical_patterns(features_csv_path)

        # Get list of all zones from historical data
        zones = sorted(historical_df["zone"].unique())
        logging.info(f"Found zones to optimize: {zones}")

        all_results = []

        # Optimize each zone
        for zone in zones:
            zone_results = self._optimize_zone_for_forecast(
                historical_df, forecast_df, zone, master_data
            )
            if len(zone_results) > 0:
                all_results.append(zone_results)

        if not all_results:
            logging.warning("No optimization results generated for any zone")
            return pd.DataFrame()

        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)

        # Sort by datetime and zone
        combined_results = combined_results.sort_values(
            ["datetime", "zone"]
        ).reset_index(drop=True)

        logging.info(
            f"Optimization completed for all zones: {len(combined_results)} total results"
        )
        logging.info(f"Results summary by zone:")
        for zone in zones:
            zone_count = len(combined_results[combined_results["zone"] == zone])
            logging.info(f"  {zone}: {zone_count} hours")

        # Convert to wide format
        wide_df = self._convert_to_wide_format(combined_results)

        return wide_df

    def _convert_to_wide_format(self, long_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert long format optimization results to wide format.

        Args:
            long_df: Long format DataFrame with zone column

        Returns:
            Wide format DataFrame with zone-specific columns
        """
        if long_df.empty:
            return pd.DataFrame()

        # Create a base DataFrame with datetime and forecast data
        base_df = (
            long_df[["datetime", "forecast_outdoor_temp", "forecast_solar_radiation"]]
            .drop_duplicates()
            .copy()
        )
        base_df = base_df.sort_values("datetime").reset_index(drop=True)

        # Get unique zones
        zones = sorted(long_df["zone"].unique())

        # Create zone-specific columns for each AC setting
        ac_settings = [
            "set_temp",
            "mode",
            "fan_speed",
            "numb_units_on",
            "ac_on_off",
            "power",
            "indoor_temp",
            "similarity_score",
            "combined_score",
            "hist_outdoor_temp",
            "hist_solar_radiation",
            "hist_indoor_temp",
            "hist_datetime_used",
        ]

        for zone in zones:
            zone_data = long_df[long_df["zone"] == zone].copy()

            # Merge zone data with base dataframe
            zone_data = zone_data[["datetime"] + ac_settings].copy()

            # Rename columns to include zone name
            rename_dict = {col: f"{zone}_{col}" for col in ac_settings}
            zone_data = zone_data.rename(columns=rename_dict)

            # Merge with base dataframe
            base_df = base_df.merge(zone_data, on="datetime", how="left")

        # Sort by datetime
        base_df = base_df.sort_values("datetime").reset_index(drop=True)

        logging.info(
            f"Converted to wide format: {len(base_df)} rows, {len(base_df.columns)} columns"
        )
        logging.info(f"Columns: {list(base_df.columns)}")

        return base_df


def optimize_all_zones(
    forecast_df: pd.DataFrame, features_csv_path: str, master_data: dict
) -> pd.DataFrame:
    """
    Convenience function for backward compatibility.
    Creates a Optimizer instance and runs optimization.
    """
    optimizer = Optimizer()
    return optimizer.optimize_all_zones(forecast_df, features_csv_path, master_data)
