"""
Zone-based HVAC optimization module.
Implements the algorithm from the image to find optimal AC settings per zone
by matching similar historical weather patterns and selecting the best-performing
settings that minimize power consumption while maintaining comfort.
"""

import json
import logging
import os
from datetime import date
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

    Supports two optimization modes:
    - Whole day mode (hour_block_size=None): Selects complete 24-hour historical days
    - Hour block mode (hour_block_size is integer >= 2): Selects best N-hour blocks
      from candidate historical days, where each block can come from different days

    Both modes filter for AC ON status only and support optional forecast hour range filtering.
    """

    def __init__(
        self,
        use_operating_hours: bool = False,
        hour_block_size: Optional[int] = 3,
        forecast_hour_range: Optional[Tuple[int, int]] = (8, 19),
    ):
        """
        Initialize the Optimizer with configuration.

        Args:
            use_operating_hours: If True, filter by zone operating hours (default: False)
            hour_block_size: Number of consecutive hours to select (default 3).
                Use None for whole day mode (24 hours). Can be any positive integer >= 2.
            forecast_hour_range: Optional tuple (start_hour, end_hour) for forecast filtering
                (e.g., (8, 19) for 8 AM to 7 PM). If None, processes all 24 hours.
        """
        # Weather weights for block distance calculation
        self.WEATHER_WEIGHTS = {"temperature": 0.7, "solar_radiation": 0.3}
        # Whether to use zone operating hours for optimization (default: False)
        self.use_operating_hours = use_operating_hours

        # Validate hour_block_size
        if hour_block_size is not None and (
            not isinstance(hour_block_size, int) or hour_block_size < 2
        ):
            raise ValueError(
                f"hour_block_size must be None (whole day mode) or an integer >= 2, "
                f"got {hour_block_size}"
            )
        self.hour_block_size = hour_block_size

        # Validate forecast_hour_range
        if forecast_hour_range is not None:
            if (
                not isinstance(forecast_hour_range, tuple)
                or len(forecast_hour_range) != 2
            ):
                raise ValueError(
                    f"forecast_hour_range must be a tuple of (start_hour, end_hour), "
                    f"got {forecast_hour_range}"
                )
            start_hour, end_hour = forecast_hour_range
            if not (
                0 <= start_hour < 24 and 0 < end_hour <= 24 and start_hour < end_hour
            ):
                raise ValueError(
                    f"forecast_hour_range must have start_hour in [0, 23] and end_hour in [1, 24] "
                    f"with start_hour < end_hour, got {forecast_hour_range}"
                )
        self.forecast_hour_range = forecast_hour_range

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

    def _map_ac_on_off(self, units_count: float) -> str:
        """Map number of units to ON/OFF string."""
        return "ON" if units_count > 0 else "OFF"

    def load_historical_patterns(self, features_csv_path: str) -> pd.DataFrame:
        """
        Load features_processed_Clea.csv and filter valid records.

        Filters for records where:
        - Indoor Temp. is not null
        - adjusted_power > 0
        - Outdoor Temp. and Solar Radiation are not null
        - A/C ON/OFF > 0 (AC must be ON)

        Args:
            features_csv_path: Path to the features CSV file

        Returns:
            DataFrame ready for pattern matching (only AC ON records)
        """
        logging.info(f"Loading historical patterns from {features_csv_path}")

        # Load the CSV file
        df = pd.read_csv(features_csv_path)

        # Convert datetime column
        df["Datetime"] = pd.to_datetime(df["Datetime"])

        # Add Date column for day-level grouping
        df["Date"] = df["Datetime"].dt.date

        # Filter valid records: non-null Indoor Temp., positive adjusted_power, AC ON only
        valid_mask = (
            df["Indoor Temp."].notna()
            & (df["adjusted_power"] > 0)
            & df["Outdoor Temp."].notna()
            & df["Solar Radiation"].notna()
            & (df["A/C ON/OFF"] > 0)  # Only AC ON status
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

    def _find_similar_days(
        self,
        historical_df: pd.DataFrame,
        forecast_day_data: pd.DataFrame,
        zone: str,
        n_top: int = 20,
    ) -> List:
        """
        Find similar historical days for a given forecast day.

        Args:
            historical_df: Historical data DataFrame
            forecast_day_data: Forecast data for a single day
            zone: Zone name to filter by
            n_top: Number of top similar days to return

        Returns:
            List of Date objects for top N most similar historical days
        """
        # Filter by zone
        zone_data = historical_df[historical_df["zone"] == zone].copy()

        if len(zone_data) == 0 or "Date" not in zone_data.columns:
            return []

        # Calculate forecast day's mean weather
        f_temp_mean = forecast_day_data["Outdoor Temp."].mean()
        f_solar_mean = forecast_day_data["Solar Radiation"].mean()

        # Historical daily means for the zone
        daily_hist = (
            zone_data.groupby("Date")[["Outdoor Temp.", "Solar Radiation"]]
            .mean()
            .reset_index()
        )

        if daily_hist.empty:
            return []

        # Use z-score normalization for better day-level comparison
        daily_hist_temp = daily_hist["Outdoor Temp."].dropna()
        daily_hist_solar = daily_hist["Solar Radiation"].dropna()

        hist_temp_mean, hist_temp_std = daily_hist_temp.mean(), daily_hist_temp.std()
        hist_solar_mean, hist_solar_std = (
            daily_hist_solar.mean(),
            daily_hist_solar.std(),
        )

        # Calculate z-scores for forecast day
        forecast_temp_z = (
            (f_temp_mean - hist_temp_mean) / hist_temp_std if hist_temp_std > 0 else 0
        )
        forecast_solar_z = (
            (f_solar_mean - hist_solar_mean) / hist_solar_std
            if hist_solar_std > 0
            else 0
        )

        # Calculate z-scores for historical days
        daily_hist["temp_z"] = (
            (daily_hist["Outdoor Temp."] - hist_temp_mean) / hist_temp_std
            if hist_temp_std > 0
            else 0
        )
        daily_hist["solar_z"] = (
            (daily_hist["Solar Radiation"] - hist_solar_mean) / hist_solar_std
            if hist_solar_std > 0
            else 0
        )

        # Calculate day-level distance score (lower is better)
        daily_hist["score"] = self.WEATHER_WEIGHTS["temperature"] * abs(
            daily_hist["temp_z"] - forecast_temp_z
        ) + self.WEATHER_WEIGHTS["solar_radiation"] * abs(
            daily_hist["solar_z"] - forecast_solar_z
        )

        # Select top N days based on day-level similarity
        top_days = daily_hist.nsmallest(n_top, "score")["Date"].tolist()

        return top_days

    def _select_best_complete_day(
        self,
        historical_df: pd.DataFrame,
        zone: str,
        top_days: List,
        forecast_day_data: pd.DataFrame,
        master_data: dict,
    ) -> Tuple[Optional[date], pd.DataFrame]:
        """
        Select the best complete historical day from top similar days and return its patterns.

        Uses a three-tier priority system:
        1. First priority: Select from complete days (all forecast hours available) - choose lowest power
        2. Second priority: If no complete days, select day with least missing hours (if tie, lowest power)

        The returned patterns DataFrame is reduced to one row per hour (lowest power pattern for each hour),
        ensuring efficient lookup and consistent pattern selection.

        Args:
            historical_df: Historical data DataFrame
            zone: Zone name to filter by
            top_days: List of Date objects for similar days
            forecast_day_data: Forecast data for the day (to match hours)
            master_data: Master data dictionary (currently unused, reserved for future comfort filtering)

        Returns:
            Tuple of (best_day Date object, patterns DataFrame with one row per hour) or (None, empty DataFrame) if no valid days
        """
        # Filter by zone
        zone_data = historical_df[historical_df["zone"] == zone].copy()

        if len(zone_data) == 0 or "Date" not in zone_data.columns:
            return None, pd.DataFrame()

        # Get forecast hours to match
        forecast_hours = pd.to_datetime(forecast_day_data["datetime"]).dt.hour.unique()
        required_hour_count = len(forecast_hours)

        # Evaluate each candidate day
        complete_days = []  # Days with all required hours
        incomplete_days = []  # Days with missing hours

        for day_date in top_days:
            day_data = zone_data[zone_data["Date"] == day_date].copy()
            if len(day_data) == 0:
                continue

            # Filter to only forecast hours (in case forecast doesn't have all 24 hours)
            day_data["hour"] = day_data["Datetime"].dt.hour
            day_data = day_data[day_data["hour"].isin(forecast_hours)]

            if len(day_data) == 0:
                continue

            # Calculate average power for the day (lower is better)
            avg_power = day_data["adjusted_power"].mean()
            available_hour_count = len(day_data["hour"].unique())
            missing_count = required_hour_count - available_hour_count
            coverage_rate = (
                available_hour_count / required_hour_count
                if required_hour_count > 0
                else 0.0
            )

            day_info = {
                "Date": day_date,
                "avg_power": avg_power,
                "available_hour_count": available_hour_count,
                "missing_count": missing_count,
                "coverage_rate": coverage_rate,
            }

            if missing_count == 0:
                # Complete day - all required hours available
                complete_days.append(day_info)
            else:
                # Incomplete day - some hours missing
                incomplete_days.append(day_info)

        # Select best day using priority system
        best_day = None

        if len(complete_days) > 0:
            # 第1優先：完全なデータがある日から最小電力の日を選択
            best_day = min(complete_days, key=lambda x: x["avg_power"])["Date"]
            logging.info(
                f"Zone {zone}: Selected complete day {best_day} "
                f"(power: {min(x['avg_power'] for x in complete_days):.0f}W)"
            )
        elif len(incomplete_days) > 0:
            # 第2優先：完全な日がない場合、欠損時間が最も少ない日を選択
            # 同じ欠損数の場合、最小電力の日を優先
            best_day_info = min(
                incomplete_days, key=lambda x: (x["missing_count"], x["avg_power"])
            )
            best_day = best_day_info["Date"]

            logging.warning(
                f"Zone {zone}: No complete days found. "
                f"第2優先適用: Selected day {best_day} "
                f"with {best_day_info['missing_count']} missing hours "
                f"(第3優先確認: coverage rate {best_day_info['coverage_rate']*100:.1f}%, "
                f"power: {best_day_info['avg_power']:.0f}W)"
            )
        else:
            # No valid days found
            logging.warning(
                f"Zone {zone}: No valid days found in top_days for forecast period"
            )
            return None, pd.DataFrame()

        # Get patterns for the best day (avoid redundant filtering)
        best_day_patterns = zone_data[zone_data["Date"] == best_day].copy()
        best_day_patterns["hour"] = best_day_patterns["Datetime"].dt.hour
        best_day_patterns = best_day_patterns[
            best_day_patterns["hour"].isin(forecast_hours)
        ].copy()

        # Reduce to one pattern per hour (lowest power pattern for each hour)
        # This ensures the returned DataFrame has exactly one row per hour
        if len(best_day_patterns) == 0:
            return best_day, pd.DataFrame()

        # Group by hour and select the row with lowest adjusted_power for each hour
        best_day_patterns = (
            best_day_patterns.sort_values("adjusted_power")
            .groupby("hour", as_index=False)
            .first()
        )

        return best_day, best_day_patterns

    def _calculate_hour_block_distance(
        self,
        forecast_block: pd.DataFrame,
        historical_block: pd.DataFrame,
    ) -> float:
        """
        Calculate distance between forecast hour block and historical hour block.

        Args:
            forecast_block: DataFrame with forecast weather for N hours
            historical_block: DataFrame with historical weather for N hours

        Returns:
            Weather distance (lower is better, represents how similar the blocks are)
        """
        # Calculate mean weather values for each block
        forecast_temp_mean = forecast_block["Outdoor Temp."].mean()
        forecast_solar_mean = forecast_block["Solar Radiation"].mean()

        historical_temp_mean = historical_block["Outdoor Temp."].mean()
        historical_solar_mean = historical_block["Solar Radiation"].mean()

        # Calculate z-scores for normalization (using historical data statistics)
        # Get historical data statistics for normalization
        hist_temp_std = historical_block["Outdoor Temp."].std()
        hist_solar_std = historical_block["Solar Radiation"].std()

        # Use simple difference if standard deviation is 0 (all values same)
        if hist_temp_std > 0:
            temp_diff = abs(forecast_temp_mean - historical_temp_mean) / hist_temp_std
        else:
            temp_diff = abs(forecast_temp_mean - historical_temp_mean)

        if hist_solar_std > 0:
            solar_diff = (
                abs(forecast_solar_mean - historical_solar_mean) / hist_solar_std
            )
        else:
            solar_diff = abs(forecast_solar_mean - historical_solar_mean)

        # Calculate weighted weather distance (lower is better)
        weather_distance = (
            self.WEATHER_WEIGHTS["temperature"] * temp_diff
            + self.WEATHER_WEIGHTS["solar_radiation"] * solar_diff
        )

        return weather_distance

    def _select_best_hour_blocks(
        self,
        historical_df: pd.DataFrame,
        zone: str,
        top_days: List,
        forecast_day_data: pd.DataFrame,
        master_data: dict,
    ) -> Dict[int, pd.Series]:
        """
        Select best N-hour blocks from candidate days for hour block mode.

        For each forecast hour block (consecutive N hours), finds the best matching
        N-hour block from all candidate historical days where AC is ON.

        Args:
            historical_df: Historical data DataFrame (already filtered for AC ON)
            zone: Zone name to filter by
            top_days: List of Date objects for similar days
            forecast_day_data: Forecast data for the day
            master_data: Master data dictionary (unused, reserved for future use)

        Returns:
            Dictionary mapping forecast hour → best historical pattern row (pd.Series)
        """
        # Filter by zone (AC ON filtering already done in load_historical_patterns)
        zone_data = historical_df[historical_df["zone"] == zone].copy()

        if len(zone_data) == 0 or "Date" not in zone_data.columns:
            return {}

        # Optimize: Compute hour column once for all zone data
        zone_data["hour"] = zone_data["Datetime"].dt.hour

        # Extract forecast date for display
        forecast_date = pd.to_datetime(forecast_day_data["datetime"].iloc[0]).date()

        # Get forecast hours in order (preserve order from forecast_day_data)
        forecast_hours_ordered = pd.to_datetime(
            forecast_day_data["datetime"]
        ).dt.hour.tolist()
        # Get unique hours while preserving order
        seen = set()
        unique_forecast_hours = []
        for hour in forecast_hours_ordered:
            if hour not in seen:
                seen.add(hour)
                unique_forecast_hours.append(hour)

        print(
            f"\n[Zone: {zone}] Forecast Date: {forecast_date} | Processing {len(unique_forecast_hours)} forecast hours: {unique_forecast_hours}"
        )

        # Group forecast hours into consecutive non-overlapping blocks of hour_block_size
        hour_blocks = []
        block_size = self.hour_block_size
        current_block = []

        for hour in unique_forecast_hours:
            if len(current_block) == 0:
                # Start new block
                current_block = [hour]
            elif hour == current_block[-1] + 1:  # consecutive hour
                # Consecutive hour, add to current block
                current_block.append(hour)
                if len(current_block) == block_size:
                    # Block is complete, add it and start new block
                    hour_blocks.append(current_block)
                    current_block = []
            else:
                # Not consecutive, save current block if it has any hours, then start new block
                if len(current_block) > 0:
                    hour_blocks.append(current_block)
                current_block = [hour]

        # Add remaining block if any
        if len(current_block) > 0:
            hour_blocks.append(current_block)

        print(
            f"[Zone: {zone}] Forecast Date: {forecast_date} | Grouped into {len(hour_blocks)} hour block(s) (block_size={block_size}): {hour_blocks}"
        )

        # Dictionary to store best pattern for each hour
        patterns_by_hour = {}

        # Process each forecast hour block
        for block_idx, hour_block in enumerate(hour_blocks, 1):
            # Extract forecast weather for this hour block
            forecast_block = forecast_day_data[
                pd.to_datetime(forecast_day_data["datetime"]).dt.hour.isin(hour_block)
            ].copy()

            if len(forecast_block) == 0:
                continue

            actual_block_size = len(
                hour_block
            )  # May be smaller than block_size for last incomplete block

            # Skip if block size is 0 (shouldn't happen, but defensive check)
            if actual_block_size == 0:
                continue

            print(
                f"\n[Zone: {zone}] Forecast Date: {forecast_date} | Block {block_idx}/{len(hour_blocks)}: Forecast hours {hour_block} (size={actual_block_size})"
            )

            best_block_candidates = []

            # Evaluate each candidate day
            for day_date in top_days:
                day_data = zone_data[zone_data["Date"] == day_date].copy()
                if len(day_data) == 0:
                    continue

                # Find all possible consecutive blocks in this day
                # that match the forecast hour block size
                day_hours = sorted(day_data["hour"].unique())

                # Need at least actual_block_size hours to form a block
                if len(day_hours) < actual_block_size:
                    continue

                # Try all possible consecutive blocks of actual_block_size in this day
                for start_idx in range(len(day_hours) - actual_block_size + 1):
                    candidate_hours = day_hours[
                        start_idx : start_idx + actual_block_size
                    ]

                    # Fix 1: Validate that candidate hours are actually consecutive
                    expected_consecutive = list(
                        range(
                            candidate_hours[0], candidate_hours[0] + actual_block_size
                        )
                    )
                    if candidate_hours != expected_consecutive:
                        # Skip non-consecutive blocks
                        continue

                    candidate_block = day_data[
                        day_data["hour"].isin(candidate_hours)
                    ].copy()

                    if len(candidate_block) == 0:
                        continue

                    # Calculate weather distance for this candidate block
                    weather_distance = self._calculate_hour_block_distance(
                        forecast_block, candidate_block
                    )

                    # Calculate average power for this block
                    avg_power = candidate_block["adjusted_power"].mean()

                    best_block_candidates.append(
                        {
                            "weather_distance": weather_distance,
                            "avg_power": avg_power,
                            "block_data": candidate_block,
                            "hours": candidate_hours,
                        }
                    )

            # Select best block (lowest weather distance, then lowest power)
            if len(best_block_candidates) > 0:
                best_candidate = min(
                    best_block_candidates,
                    key=lambda x: (x["weather_distance"], x["avg_power"]),
                )

                print(
                    f"  ✓ Selected best block: hours {best_candidate['hours']} "
                    f"(weather_distance={best_candidate['weather_distance']:.3f}, "
                    f"avg_power={best_candidate['avg_power']:.0f}W) "
                    f"from {len(best_block_candidates)} candidates"
                )

                # Map each hour in forecast block to corresponding hour in historical block
                # Use positional mapping: forecast block position i maps to historical block position i
                historical_block_data = best_candidate["block_data"]
                historical_hours_sorted = sorted(historical_block_data["hour"].unique())
                forecast_hours_sorted = sorted(hour_block)

                print(
                    f"  → Mapping forecast hours {forecast_hours_sorted} → historical hours {historical_hours_sorted} (positional)"
                )

                # Map by position: forecast hour at position i → historical hour at position i
                for idx, forecast_hour in enumerate(forecast_hours_sorted):
                    if idx < len(historical_hours_sorted):
                        hist_hour = historical_hours_sorted[idx]
                    else:
                        # Fallback: use last historical hour if forecast block is larger
                        hist_hour = historical_hours_sorted[-1]

                    # Get the row(s) for this historical hour
                    hist_rows = historical_block_data[
                        historical_block_data["hour"] == hist_hour
                    ]

                    if len(hist_rows) == 0:
                        # Should not happen, but defensive check
                        continue

                    # If multiple rows for same hour, select lowest power
                    if len(hist_rows) > 1:
                        hist_row = hist_rows.sort_values("adjusted_power").iloc[0]
                    else:
                        hist_row = hist_rows.iloc[0]

                    patterns_by_hour[forecast_hour] = hist_row
            else:
                print(f"  ✗ No valid candidates found for forecast hours {hour_block}")

        print(
            f"\n[Zone: {zone}] Forecast Date: {forecast_date} | Successfully mapped {len(patterns_by_hour)}/{len(unique_forecast_hours)} hours"
        )

        return patterns_by_hour

    def _optimize_zone_for_forecast(
        self,
        historical_df: pd.DataFrame,
        forecast_df: pd.DataFrame,
        zone: str,
        master_data: dict,
    ) -> pd.DataFrame:
        """
        Main optimization function for a single zone.

        Supports two modes:
        - Whole day mode (hour_block_size=None): Selects complete historical days
        - Hour block mode (hour_block_size is integer): Selects best N-hour blocks from candidate days

        Args:
            historical_df: Historical patterns DataFrame (already filtered for AC ON)
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

        # Apply forecast_hour_range filter if specified
        forecast_df_original = forecast_df.copy()
        forecast_df_working = forecast_df.copy()

        if self.forecast_hour_range is not None:
            range_start, range_end = self.forecast_hour_range
            forecast_df_working["hour"] = pd.to_datetime(
                forecast_df_working["datetime"]
            ).dt.hour
            forecast_df_working = forecast_df_working[
                (forecast_df_working["hour"] >= range_start)
                & (forecast_df_working["hour"] < range_end)
            ].copy()
            forecast_df_working = forecast_df_working.drop(columns=["hour"])

        results = []
        stats = {
            "total_hours": 0,
            "outside_op_hours": 0,
            "no_similar_patterns": 0,
            "success": 0,
        }

        # Track detailed failures with timestamps and reasons
        detailed_failures = {
            "outside_op": [],
            "no_patterns": [],
        }

        # Group forecast by day for daily-level processing
        forecast_df_working["Date"] = pd.to_datetime(
            forecast_df_working["datetime"]
        ).dt.date

        forecast_days = forecast_df_working.groupby("Date")

        # Process each forecast day
        for forecast_date, forecast_day_data in forecast_days:
            # Find similar historical days for this forecast day
            top_days = self._find_similar_days(
                historical_df, forecast_day_data, zone, n_top=20
            )

            # Select patterns based on mode
            if self.hour_block_size is None:
                # Whole day mode: Select best complete day
                best_day, day_patterns = self._select_best_complete_day(
                    historical_df, zone, top_days, forecast_day_data, master_data
                )

                # Create a lookup dictionary from DataFrame for quick access by hour
                patterns_by_hour = {}
                if not day_patterns.empty and "hour" in day_patterns.columns:
                    for _, row in day_patterns.iterrows():
                        patterns_by_hour[row["hour"]] = row
            else:
                # Hour block mode: Select best N-hour blocks
                patterns_by_hour = self._select_best_hour_blocks(
                    historical_df, zone, top_days, forecast_day_data, master_data
                )

            # Process each hour in the forecast day
            for _, forecast_row in forecast_day_data.iterrows():
                forecast_datetime = pd.to_datetime(forecast_row["datetime"])
                hour = forecast_datetime.hour
                stats["total_hours"] += 1

                # Check if hour is within operating hours
                if self.use_operating_hours and not (start_hour <= hour < end_hour):
                    stats["outside_op_hours"] += 1
                    detailed_failures["outside_op"].append(
                        f"{forecast_datetime.strftime('%m/%d %H:00')}"
                    )
                    continue

                # Get pattern for this hour
                if hour not in patterns_by_hour:
                    stats["no_similar_patterns"] += 1
                    detailed_failures["no_patterns"].append(
                        f"{forecast_datetime.strftime('%m/%d %H:00')}"
                    )
                    continue

                # Use the pattern
                best_pattern = patterns_by_hour[hour]

                stats["success"] += 1

                # Extract recommended settings from the pattern
                units_count = best_pattern["A/C ON/OFF"]
                result = {
                    "datetime": forecast_datetime,
                    "zone": zone,
                    "set_temp": best_pattern["A/C Set Temperature"],
                    "mode": self._map_ac_mode(int(best_pattern["A/C Mode"])),
                    "fan_speed": self._map_fan_speed(
                        int(best_pattern["A/C Fan Speed"])
                    ),
                    "numb_units_on": units_count,
                    "ac_on_off": self._map_ac_on_off(units_count),
                    "power": best_pattern["adjusted_power"],
                    "indoor_temp": best_pattern["Indoor Temp."],
                    "hist_datetime_used": best_pattern["Datetime"],
                    "forecast_outdoor_temp": forecast_row["Outdoor Temp."],
                    "forecast_solar_radiation": forecast_row["Solar Radiation"],
                    "hist_outdoor_temp": best_pattern["Outdoor Temp."],
                    "hist_solar_radiation": best_pattern["Solar Radiation"],
                    "hist_indoor_temp": best_pattern["Indoor Temp."],
                }

                results.append(result)

        # If forecast_hour_range is specified, add empty results for hours outside range
        if self.forecast_hour_range is not None:
            range_start, range_end = self.forecast_hour_range
            forecast_df_original["hour"] = pd.to_datetime(
                forecast_df_original["datetime"]
            ).dt.hour
            outside_range_df = forecast_df_original[
                (forecast_df_original["hour"] < range_start)
                | (forecast_df_original["hour"] >= range_end)
            ].copy()

            for _, forecast_row in outside_range_df.iterrows():
                forecast_datetime = pd.to_datetime(forecast_row["datetime"])
                result = {
                    "datetime": forecast_datetime,
                    "zone": zone,
                    "set_temp": None,
                    "mode": None,
                    "fan_speed": None,
                    "numb_units_on": None,
                    "ac_on_off": None,
                    "power": None,
                    "indoor_temp": None,
                    "hist_datetime_used": None,
                    "forecast_outdoor_temp": forecast_row["Outdoor Temp."],
                    "forecast_solar_radiation": forecast_row["Solar Radiation"],
                    "hist_outdoor_temp": None,
                    "hist_solar_radiation": None,
                    "hist_indoor_temp": None,
                }
                results.append(result)

        result_df = pd.DataFrame(results)

        # ---------------------------------------
        # Calculate success rate and print detailed summary
        # ---------------------------------------
        valid_hours = stats["total_hours"] - stats["outside_op_hours"]
        success_rate = (stats["success"] / valid_hours * 100) if valid_hours > 0 else 0

        # Print detailed summary
        print(f"\n━━━━━ [{zone}] Summary ━━━━━")
        print(f"✅ 成功: {len(result_df)}時間 ({success_rate:.0f}%)")

        if stats["outside_op_hours"] > 0:
            print(f"⏰ 営業時間外: {stats['outside_op_hours']}時間")
            if len(detailed_failures["outside_op"]) <= 10:
                print(f"   {', '.join(detailed_failures['outside_op'])}")

        if stats["no_similar_patterns"] > 0:
            print(f"🔍 パターンなし: {stats['no_similar_patterns']}時間")
            if len(detailed_failures["no_patterns"]) <= 10:
                print(f"   {', '.join(detailed_failures['no_patterns'])}")

        print("━" * 30)

        return result_df

    def optimize_all_zones(
        self, forecast_df: pd.DataFrame, features_csv_path: str, master_data: dict
    ) -> pd.DataFrame:
        """
        Optimize all zones for the forecast period.

        Uses the configured optimization mode (whole day or hour block) and
        applies forecast hour range filtering if specified.

        Args:
            forecast_df: Forecast weather DataFrame
            features_csv_path: Path to features CSV file
            master_data: Master data dictionary

        Returns:
            Combined DataFrame with optimization results for all zones in wide format
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
            # "similarity_score",  # Set to None for day-level selection
            "hist_outdoor_temp",
            "hist_solar_radiation",
            "hist_indoor_temp",
            "hist_datetime_used",
            # "hist_day_used",  # Track which historical day was used (same for all hours in forecast day)
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
