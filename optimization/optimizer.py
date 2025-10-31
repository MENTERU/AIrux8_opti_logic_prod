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

    def __init__(
        self,
        use_operating_hours: bool = False,
        strategy: str = "hourly",
        similar_days_k: int = 15,
    ):
        """
        Initialize the Optimizer with configuration.

        Args:
            use_operating_hours: If True, filter by zone operating hours.
                                 If False, optimize for all 24 hours.
        """
        # Configuration Constants
        self.TARGET_INDOOR_TEMP = 26.0  # default in case there is no gategory mapping
        self.WEATHER_WEIGHTS = {"temperature": 0.7, "solar_radiation": 0.3}
        self.TEMP_TOLERANCE = 0.5  # ±0.5°C for weather matching
        # self.SOLAR_TOLERANCE = 100.0  # ±100 W/m² for solar radiation matching

        # Operating hours flag (default: ignore operating hours)
        self.use_operating_hours = use_operating_hours

        # Strategy: "hourly" (default) or "similar_day"
        self.strategy = strategy
        self.similar_days_k = max(1, int(similar_days_k))

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
            return pd.DataFrame()

        # Extract forecast values
        forecast_datetime = pd.to_datetime(forecast_row["datetime"])
        forecast_hour = forecast_datetime.hour
        forecast_temp = forecast_row["Outdoor Temp."]
        forecast_solar = forecast_row["Solar Radiation"]

        # Filter by same hour of day
        zone_data["hour"] = zone_data["Datetime"].dt.hour
        same_hour_data = zone_data[zone_data["hour"] == forecast_hour].copy()

        # Similar-day strategy: restrict candidates to historically similar days (by daily mean weather)
        if self.strategy == "similar_day" and "Date" in same_hour_data.columns:
            f_date = forecast_datetime.date()
            # Compute forecast day's mean weather
            # Expect caller to pass full forecast_df externally for a day-level calc; fallback to row values
            f_temp = forecast_row["Outdoor Temp."]
            f_solar = forecast_row["Solar Radiation"]
            # Historical daily means for the zone
            daily_hist = (
                zone_data.groupby("Date")[["Outdoor Temp.", "Solar Radiation"]]
                .mean()
                .reset_index()
            )
            if not daily_hist.empty:
                daily_hist["temp_diff"] = (daily_hist["Outdoor Temp."] - f_temp).abs()
                daily_hist["solar_diff"] = (
                    daily_hist["Solar Radiation"] - f_solar
                ).abs()
                daily_hist["score"] = (
                    self.WEATHER_WEIGHTS["temperature"] * daily_hist["temp_diff"]
                    + self.WEATHER_WEIGHTS["solar_radiation"] * daily_hist["solar_diff"]
                )
                top_days = daily_hist.nsmallest(self.similar_days_k, "score")[
                    "Date"
                ].tolist()
                same_hour_data = same_hour_data[same_hour_data["Date"].isin(top_days)]

        if len(same_hour_data) == 0:
            return pd.DataFrame()

        # Filter by similar weather conditions
        temp_diff = abs(same_hour_data["Outdoor Temp."] - forecast_temp)
        solar_diff = abs(same_hour_data["Solar Radiation"] - forecast_solar)

        weather_similar_mask = temp_diff <= self.TEMP_TOLERANCE
        similar_hist_data = same_hour_data[weather_similar_mask].copy()

        if len(similar_hist_data) == 0:
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

        # Calculate statistics for logging (silently track, show in summary)
        if len(filtered_patterns) == 0 and len(patterns_df) > 0:
            pass  # Details shown in summary

        # return filtered_patterns
        return filtered_patterns

    def _select_best_pattern(
        self,
        patterns_df: pd.DataFrame,
        hour: int,
        zone: str,
        month: int,
        master_data: dict,
    ) -> Optional[pd.Series]:
        """
        Select the pattern with the lowest adjusted power consumption.

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

        # Select pattern with lowest adjusted_power
        best_pattern = patterns_df.nsmallest(1, "adjusted_power").iloc[0]

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
        if self.use_operating_hours:
            start_hour, end_hour = get_zone_operating_hours(master_data, zone)
        else:
            start_hour, end_hour = 0, 24  # All 24 hours

        # Print comfort range once per zone (get from first forecast row)
        if len(forecast_df) > 0:
            first_forecast_month = pd.to_datetime(forecast_df.iloc[0]["datetime"]).month
            comfort_range = get_comfort_range(master_data, zone, first_forecast_month)
            print(
                f"[Optimizer] {zone}: 快適範囲 {comfort_range[0]}-{comfort_range[1]}°C"
            )

        results = []
        stats = {
            "total_hours": 0,
            "outside_op_hours": 0,
            "no_similar_patterns": 0,
            "no_comfort_patterns": 0,
            "no_best_pattern": 0,
            "success": 0,
        }

        # Track detailed failures with timestamps and reasons
        detailed_failures = {
            "outside_op": [],
            "no_patterns": [],
            "no_comfort": [],
            "no_best": [],
        }

        # Track comfort filter details for showing why filter was too strict
        comfort_filter_details = []

        # Track pattern counts for each hour for detailed reporting
        hourly_stats = []

        for _, forecast_row in forecast_df.iterrows():
            forecast_datetime = pd.to_datetime(forecast_row["datetime"])
            month = forecast_datetime.month
            hour = forecast_datetime.hour
            stats["total_hours"] += 1

            # Check if current hour is within operating hours for the zone (only if flag is enabled)
            if self.use_operating_hours and not (start_hour <= hour < end_hour):
                stats["outside_op_hours"] += 1
                detailed_failures["outside_op"].append(
                    f"{forecast_datetime.strftime('%m/%d %H:00')}"
                )
                continue

            # Step 1-3: Find similar historical patterns for the zone
            similar_patterns = self._find_similar_patterns(
                historical_df, forecast_row, zone, n_top=20
            )
            similar_count = len(similar_patterns)

            if len(similar_patterns) == 0:
                stats["no_similar_patterns"] += 1
                detailed_failures["no_patterns"].append(
                    f"{forecast_datetime.strftime('%m/%d %H:00')}"
                )
                continue

            # Step 4: Filter by comfort range for the zone
            min_temp, max_temp = get_comfort_range(master_data, zone, month)
            comfort_patterns = self._filter_by_comfort(
                similar_patterns, zone, month, master_data
            )
            comfort_count = len(comfort_patterns)

            if len(comfort_patterns) == 0:
                stats["no_comfort_patterns"] += 1
                timestamp_str = forecast_datetime.strftime("%m/%d %H:00")
                detailed_failures["no_comfort"].append(timestamp_str)

                # Capture why filter failed
                if len(similar_patterns) > 0:
                    hist_min = similar_patterns["Indoor Temp."].min()
                    hist_max = similar_patterns["Indoor Temp."].max()
                    comfort_filter_details.append(
                        {
                            "timestamp": timestamp_str,
                            "required": f"{min_temp}-{max_temp}°C",
                            "actual": f"{hist_min:.1f}-{hist_max:.1f}°C",
                            "similar": similar_count,
                            "comfort": comfort_count,
                        }
                    )
                continue

            # Step 5: Score and select best pattern
            best_pattern = self._select_best_pattern(
                comfort_patterns, hour, zone, month, master_data
            )

            if best_pattern is None:
                stats["no_best_pattern"] += 1
                detailed_failures["no_best"].append(
                    f"{forecast_datetime.strftime('%m/%d %H:00')}"
                )
                continue

            stats["success"] += 1

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
                "hist_datetime_used": best_pattern["Datetime"],
                "forecast_outdoor_temp": forecast_row["Outdoor Temp."],
                "forecast_solar_radiation": forecast_row["Solar Radiation"],
                "hist_outdoor_temp": best_pattern["Outdoor Temp."],
                "hist_solar_radiation": best_pattern["Solar Radiation"],
                "hist_indoor_temp": best_pattern["Indoor Temp."],
            }

            results.append(result)

        result_df = pd.DataFrame(results)

        # Calculate success rate
        valid_hours = stats["total_hours"] - stats["outside_op_hours"]
        success_rate = (stats["success"] / valid_hours * 100) if valid_hours > 0 else 0

        # Print detailed summary
        print(f"\n━━━━━ [{zone}] Summary ━━━━━")
        print(f"✅ 成功: {len(result_df)}時間 ({success_rate:.0f}%)")

        # Show summary of pattern filtering for failed hours
        if len(comfort_filter_details) > 0:
            total_similar = sum(d["similar"] for d in comfort_filter_details)
            avg_similar = total_similar / len(comfort_filter_details)
            print(
                f"📊 失敗時間の平均: 類似パターン {avg_similar:.1f}件 → 快適範囲通過 0件"
            )

        if stats["outside_op_hours"] > 0:
            print(f"⏰ 営業時間外: {stats['outside_op_hours']}時間")
            if len(detailed_failures["outside_op"]) <= 10:
                print(f"   {', '.join(detailed_failures['outside_op'])}")

        if stats["no_similar_patterns"] > 0:
            print(f"🔍 パターンなし: {stats['no_similar_patterns']}時間")
            if len(detailed_failures["no_patterns"]) <= 10:
                print(f"   {', '.join(detailed_failures['no_patterns'])}")

        if stats["no_comfort_patterns"] > 0:
            print(f"🌡️ 快適性なし: {stats['no_comfort_patterns']}時間")
            if len(detailed_failures["no_comfort"]) <= 10:
                # Show detailed reasons with counts
                for detail in comfort_filter_details[:10]:
                    print(
                        f"   {detail['timestamp']}: 類似パターン={detail['similar']}件 → 快適範囲通過=0件 (必要={detail['required']}, 実際={detail['actual']})"
                    )
            elif len(detailed_failures["no_comfort"]) > 10:
                # Show first few with details, then summary
                for detail in comfort_filter_details[:5]:
                    print(
                        f"   {detail['timestamp']}: 類似パターン={detail['similar']}件 → 快適範囲通過=0件 (必要={detail['required']}, 実際={detail['actual']})"
                    )
                print(f"   ... and {len(detailed_failures['no_comfort']) - 5} more")

        if stats["no_best_pattern"] > 0:
            print(f"❌ 選択不可: {stats['no_best_pattern']}時間")

        print("━" * 30)

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
        # Load historical patterns
        historical_df = self.load_historical_patterns(features_csv_path)

        # Get list of all zones from historical data
        zones = sorted(historical_df["zone"].unique())

        all_results = []

        # Optimize each zone
        for zone in zones:
            zone_results = self._optimize_zone_for_forecast(
                historical_df, forecast_df, zone, master_data
            )
            if len(zone_results) > 0:
                all_results.append(zone_results)

        if not all_results:
            print("いずれのゾーンでも最適化結果は生成されませんでした")
            return pd.DataFrame()

        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)

        # Sort by datetime and zone
        combined_results = combined_results.sort_values(
            ["datetime", "zone"]
        ).reset_index(drop=True)

        print(f"\n=== 最適化サマリー ===")
        print(f"合計結果: {len(combined_results)}時間")
        for zone in zones:
            zone_count = len(combined_results[combined_results["zone"] == zone])
            print(f"  {zone}: {zone_count}時間")

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

        return base_df
