"""
Data processor for creating additional features before model training
"""

from typing import Optional

import numpy as np
import pandas as pd

try:
    import jpholiday

    JPHOLIDAY_AVAILABLE = True
except ImportError:
    JPHOLIDAY_AVAILABLE = False
    print(
        "[DataProcessor] Warning: jpholiday not available. Holiday features will be set to 0."
    )


class DataProcessor:
    """
    Data processor for creating additional features before model training
    """

    def __init__(self):
        self.jpholiday_available = JPHOLIDAY_AVAILABLE

    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for model training

        Args:
            df: Input DataFrame with basic features

        Returns:
            DataFrame with additional features added
        """
        df = df.copy()

        # CRITICAL: Sort data by datetime and zone before creating any features
        # This ensures proper time-series ordering for lag features and other time-dependent calculations
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            datetime_col = "Datetime"
            # Sort by zone (if exists) and datetime to ensure proper ordering
            if "zone" in df.columns:
                df = df.sort_values(["zone", "Datetime"])
                print("[DataProcessor] Sorted data by zone and Datetime")
            else:
                df = df.sort_values("Datetime")
                print("[DataProcessor] Sorted data by Datetime")
        elif "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            datetime_col = "datetime"
            # Sort by zone (if exists) and datetime to ensure proper ordering
            if "zone" in df.columns:
                df = df.sort_values(["zone", "datetime"])
                print("[DataProcessor] Sorted data by zone and datetime")
            else:
                df = df.sort_values("datetime")
                print("[DataProcessor] Sorted data by datetime")
        else:
            print(
                "[DataProcessor] Warning: No datetime column found. Using index as datetime."
            )
            datetime_col = None

        # Set datetime as index for easier processing
        if datetime_col:
            df = df.set_index(datetime_col)

        # 1. Wet Bulb Temperature calculation
        if "Outdoor Temp." in df.columns and "Outdoor Humidity" in df.columns:
            df["Wet Bulb Temp"] = (
                df["Outdoor Temp."]
                * np.arctan(0.151977 * (df["Outdoor Humidity"] + 8.313659) ** 0.5)
                + np.arctan(df["Outdoor Temp."] + df["Outdoor Humidity"])
                - np.arctan(df["Outdoor Humidity"] - 1.676331)
                + 0.00391838
                * (df["Outdoor Humidity"] ** 1.5)
                * np.arctan(0.023101 * df["Outdoor Humidity"])
                - 4.686035
            )
            print("[DataProcessor] Added Wet Bulb Temperature feature")

        # 2. Temperature difference features
        if "Outdoor Temp." in df.columns and "Indoor Temp. Lag1" in df.columns:
            df["Temp Diff (Outdoor - Indoor Lag1)"] = (
                df["Outdoor Temp."] - df["Indoor Temp. Lag1"]
            )
            print(
                "[DataProcessor] Added temperature difference (Outdoor - Indoor Lag1)"
            )

        if "Indoor Temp. Lag1" in df.columns and "A/C Set Temperature" in df.columns:
            df["Temp Diff (Indoor Lag1 - Setpoint)"] = (
                df["Indoor Temp. Lag1"] - df["A/C Set Temperature"]
            )
            print(
                "[DataProcessor] Added temperature difference (Indoor Lag1 - Setpoint)"
            )

        # 3. Time-based features (using existing columns if available, otherwise create new ones)
        if "Hour" not in df.columns:
            df["hour"] = df.index.hour
            print("[DataProcessor] Added hour feature")
        else:
            df["hour"] = df["Hour"]  # Use existing Hour column

        if "Month" not in df.columns:
            df["month"] = df.index.month
            print("[DataProcessor] Added month feature")
        else:
            df["month"] = df["Month"]  # Use existing Month column

        if "DayOfWeek" not in df.columns:
            df["weekday"] = df.index.weekday
            print("[DataProcessor] Added weekday feature")
        else:
            df["weekday"] = df["DayOfWeek"]  # Use existing DayOfWeek column

        # Holiday feature - always use IsHoliday name to match model_builder.py
        if "IsHoliday" not in df.columns:
            if self.jpholiday_available:
                df["IsHoliday"] = (
                    df.index.to_series().apply(jpholiday.is_holiday).astype(int)
                )
                print("[DataProcessor] Added IsHoliday feature using jpholiday")
            else:
                df["IsHoliday"] = 0
                print(
                    "[DataProcessor] Added IsHoliday feature (set to 0 - jpholiday not available)"
                )
        else:
            print("[DataProcessor] IsHoliday feature already exists")

        # Week number feature
        df["n_week"] = df.index.isocalendar().week
        print("[DataProcessor] Added week number feature")

        # HourOfWeek feature (moved from model_builder.py)
        # Use existing DayOfWeek and Hour columns if available, otherwise use index
        if "DayOfWeek" in df.columns and "Hour" in df.columns:
            df["HourOfWeek"] = df["DayOfWeek"] * 24 + df["Hour"]
            print(
                "[DataProcessor] Added HourOfWeek feature using DayOfWeek and Hour columns"
            )
        else:
            # Fallback to using index if columns not available
            df["HourOfWeek"] = df.index.weekday * 24 + df.index.hour
            print("[DataProcessor] Added HourOfWeek feature using datetime index")

        # OnRunLength feature (moved from model_builder.py) - cumulative run length when AC is ON
        if "A/C ON/OFF" in df.columns:
            # Process by zone if zone column exists, otherwise process entire dataset
            if "zone" in df.columns:
                df["OnRunLength"] = 0
                for zone in df["zone"].unique():
                    zone_mask = df["zone"] == zone
                    zone_data = df.loc[zone_mask, "A/C ON/OFF"].fillna(0).astype(int)
                    run_length = 0
                    on_run_values = []
                    for is_on in zone_data:
                        if is_on > 0:  # AC is ON
                            run_length += 1
                        else:  # AC is OFF
                            run_length = 0
                        on_run_values.append(run_length)
                    df.loc[zone_mask, "OnRunLength"] = on_run_values
                print("[DataProcessor] Added OnRunLength feature (processed by zone)")
            else:
                # No zone column, process entire dataset
                zone_data = df["A/C ON/OFF"].fillna(0).astype(int)
                run_length = 0
                on_run_values = []
                for is_on in zone_data:
                    if is_on > 0:  # AC is ON
                        run_length += 1
                    else:  # AC is OFF
                        run_length = 0
                    on_run_values.append(run_length)
                df["OnRunLength"] = on_run_values
                print(
                    "[DataProcessor] Added OnRunLength feature (processed entire dataset)"
                )
        else:
            print(
                "[DataProcessor] Warning: A/C ON/OFF not found, skipping OnRunLength feature"
            )

        # 4. Lag features (now properly sorted by zone and datetime)
        lag_columns = ["Outdoor Temp.", "Outdoor Humidity", "adjusted_power"]
        lag_periods = [1, 24]

        for col in lag_columns:
            if col in df.columns:
                for lag in lag_periods:
                    lag_col_name = f"{col} lag{lag}"
                    if "zone" in df.columns:
                        # Calculate lag within each zone to ensure proper time-series ordering
                        df[lag_col_name] = df.groupby("zone")[col].shift(lag)
                        print(
                            f"[DataProcessor] Added {lag_col_name} feature (grouped by zone)"
                        )
                    else:
                        # No zone column, calculate lag for entire dataset
                        df[lag_col_name] = df[col].shift(lag)
                        print(f"[DataProcessor] Added {lag_col_name} feature")
            else:
                print(f"[DataProcessor] Warning: {col} not found for lag features")

        # 5. Rolling window features (now properly sorted by zone and datetime)
        rolling_columns = ["Outdoor Temp.", "Outdoor Humidity", "adjusted_power"]
        rolling_windows = [3, 24]

        for col in rolling_columns:
            if col in df.columns:
                for window in rolling_windows:
                    rolling_col_name = f"{col} rolling_mean{window}"
                    if "zone" in df.columns:
                        # Calculate rolling mean within each zone to ensure proper time-series ordering
                        df[rolling_col_name] = (
                            df.groupby("zone")[col]
                            .rolling(window=window)
                            .mean()
                            .reset_index(level=0, drop=True)
                        )
                        print(
                            f"[DataProcessor] Added {rolling_col_name} feature (grouped by zone)"
                        )
                    else:
                        # No zone column, calculate rolling mean for entire dataset
                        df[rolling_col_name] = df[col].rolling(window=window).mean()
                        print(f"[DataProcessor] Added {rolling_col_name} feature")
            else:
                print(f"[DataProcessor] Warning: {col} not found for rolling features")

        # 6. Fill missing A/C Fan Speed with default value (Low = 1)
        if "A/C Fan Speed" in df.columns:
            missing_count = df["A/C Fan Speed"].isna().sum()
            if missing_count > 0:
                print(
                    f"[DataProcessor] Found {missing_count} missing A/C Fan Speed values"
                )
                # Fill with default value: Low (1) from category_mapping.json
                df["A/C Fan Speed"] = df["A/C Fan Speed"].fillna(1).astype(int)
                print(
                    f"[DataProcessor] Filled missing A/C Fan Speed with default value: 1 (Low)"
                )
            else:
                # Ensure it's integer type even if no missing values
                df["A/C Fan Speed"] = df["A/C Fan Speed"].astype(int)
        else:
            print("[DataProcessor] Warning: A/C Fan Speed column not found")

        # 7. Fill missing A/C Status with default value (OFF = 0)
        if "A/C Status" in df.columns:
            missing_count = df["A/C Status"].isna().sum()
            if missing_count > 0:
                print(
                    f"[DataProcessor] Found {missing_count} missing A/C Status values"
                )
                # Fill with default value: OFF (0) from category_mapping.json
                df["A/C Status"] = df["A/C Status"].fillna(0).astype(int)
                print(
                    f"[DataProcessor] Filled missing A/C Status with default value: 0 (OFF)"
                )
            else:
                # Ensure it's integer type even if no missing values
                df["A/C Status"] = df["A/C Status"].astype(int)
        else:
            print("[DataProcessor] Warning: A/C Status column not found")

        # Reset index to restore Datetime column
        if datetime_col:
            df = df.reset_index()
            # Rename back to original column name
            if datetime_col == "Datetime":
                df = df.rename(columns={"Datetime": "Datetime"})
            elif datetime_col == "datetime":
                df = df.rename(columns={"datetime": "datetime"})

        return df

    def process_features(self, area_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to process features for model training

        Args:
            area_df: Input DataFrame with basic features

        Returns:
            DataFrame with all additional features added
        """
        if area_df is None or area_df.empty:
            print("[DataProcessor] Warning: Input DataFrame is None or empty")
            return area_df

        print(
            f"[DataProcessor] Starting feature processing. Input shape: {area_df.shape}"
        )

        # Process features
        processed_df = self._preprocess_features(area_df)

        print(
            f"[DataProcessor] Feature processing completed. Ou shape: {processed_df.shape}"
        )

        # Print summary of new features added
        new_features = [
            col for col in processed_df.columns if col not in area_df.columns
        ]
        if new_features:
            print(f"[DataProcessor] Added {len(new_features)} new features:")
            for feature in new_features:
                print(f"  - {feature}")
        else:
            print("[DataProcessor] No new features were added")

        return processed_df

    def get_available_features(self, df: pd.DataFrame) -> dict:
        """
        Check which features are available in the DataFrame

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with feature availability status
        """
        from config.config_train import NEW_FEATURES

        feature_status = {}
        for feature in NEW_FEATURES:
            feature_status[feature] = feature in df.columns

        return feature_status

    def print_feature_summary(self, df: pd.DataFrame) -> None:
        """
        Print a summary of feature availability

        Args:
            df: Input DataFrame
        """
        from config.config_train import NEW_FEATURES

        print(f"\n{'='*60}")
        print("📊 Feature Availability Summary")
        print(f"{'='*60}")

        available_features = []
        missing_features = []

        for feature in NEW_FEATURES:
            if feature in df.columns:
                available_features.append(feature)
            else:
                missing_features.append(feature)

        print(f"✅ Available features ({len(available_features)}):")
        for feature in available_features:
            print(f"  - {feature}")

        if missing_features:
            print(f"\n⚠️ Missing features ({len(missing_features)}):")
            for feature in missing_features:
                print(f"  - {feature}")

        print(f"{'='*60}")
