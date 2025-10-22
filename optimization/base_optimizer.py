"""
Base optimizer class containing common functionality for both pre-cooling and daytime optimizers.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


class BaseOptimizer:
    """
    Base class containing shared optimization logic for HVAC scheduling.
    Provides methods for:
      - Finding similar historical weather patterns
      - Calculating optimization scores
      - Normalizing scores
    """

    def find_similar_sets(
        self,
        target_data: pd.DataFrame,
        data_dict: Dict,
        n: int = 5,
        features: List[str] = None,
        data_start_hour: int = 0.0,
        data_end_hour: int = 23.0,
        weights: Dict[str, float] = {"temperature": 0.7, "Solar Radiation": 0.3},
    ) -> List[Dict[str, Any]]:
        """
        Find similar weather patterns from historical data to inform optimization.

        Args:
            target_data: DataFrame containing columns like datetime, Outdoor Temp., Solar Radiation
            data_dict: Historical data dictionary keyed by (date, number_of_units, pre_cool_hours)
            n: Number of top similar sets to return
            features: Features to use for similarity calculation (Outdoor Temp., radiation, etc.)
            data_start_hour: Start hour for filtering historical and target data
            data_end_hour: End hour for filtering historical and target data
            weights: Importance weights for each feature when computing similarity

        Returns:
            List of dictionaries containing top n similar sets with their similarity scores.
        """
        # Map column names between notebook-style and historical data-style
        column_mapping = {"Outdoor Temp.": "外気温", "Solar Radiation": "日射量"}

        # Determine which column style is used in the target data
        if (
            "Outdoor Temp." in target_data.columns
            and "Solar Radiation" in target_data.columns
        ):
            features = ["Outdoor Temp.", "Solar Radiation"]
            weights = {"Outdoor Temp.": 0.7, "Solar Radiation": 0.3}
        else:
            features = ["外気温", "日射量"]
            weights = {"外気温": 0.7, "日射量": 0.3}
            column_mapping = {v: k for k, v in column_mapping.items()}

        if features is None:
            features = ["Outdoor Temp.", "Solar Radiation"]

        # Compute z-score normalization parameters (mean, std) for each feature
        feature_mean = {}
        feature_std = {}
        for feat in features:
            vals = []
            for records in data_dict.values():
                df = pd.DataFrame(records)
                hist_feat = column_mapping.get(feat, feat)
                vals.extend(df[hist_feat].dropna().tolist())
            vals.extend(target_data[feat].dropna().tolist())
            feature_mean[feat] = np.mean(vals)
            feature_std[feat] = np.std(vals) if np.std(vals) > 0 else 1.0

        # Determine main date for comparison (use newest date in target_data)
        target_dates = target_data["datetime"].dt.date.unique()
        main_date = (
            sorted(target_dates)[-1] if len(target_dates) > 1 else target_dates[0]
        )
        prev_date = pd.Timestamp(main_date) - pd.Timedelta(days=1)

        # Filter target data: previous day from data_start_hour to 23:59
        mask_prev = (target_data["datetime"].dt.date == prev_date.date()) & (
            target_data["datetime"].dt.hour >= data_start_hour
        )
        # Filter target data: current day from 00:00 to data_end_hour
        mask_today = (target_data["datetime"].dt.date == main_date) & (
            target_data["datetime"].dt.hour <= data_end_hour
        )
        target_data_filtered = (
            target_data[mask_prev | mask_today]
            .sort_values("datetime")
            .reset_index(drop=True)
        )

        results = []
        skip_nan = 0
        total = 0

        for key, records in data_dict.items():
            total += 1
            df_rec = pd.DataFrame(records)
            rec_dates = df_rec["datetime"].dt.date.unique()
            if len(rec_dates) == 0:
                continue
            rec_main_date = sorted(rec_dates)[-1]
            rec_prev_date = pd.Timestamp(rec_main_date) - pd.Timedelta(days=1)

            # Apply same filtering to historical records
            mask_prev = (df_rec["datetime"].dt.date == rec_prev_date.date()) & (
                df_rec["datetime"].dt.hour >= data_start_hour
            )
            mask_today = (df_rec["datetime"].dt.date == rec_main_date) & (
                df_rec["datetime"].dt.hour <= data_end_hour
            )
            df_rec_filtered = (
                df_rec[mask_prev | mask_today]
                .sort_values("datetime")
                .reset_index(drop=True)
            )

            min_len = min(len(df_rec_filtered), len(target_data_filtered))
            if min_len == 0:
                continue

            diff = 0
            skip = False
            for feat in features:
                hist_feat = column_mapping.get(feat, feat)
                target_feat = feat

                arr1 = df_rec_filtered[hist_feat].to_numpy(dtype=float)[:min_len]
                arr2 = target_data_filtered[target_feat].to_numpy(dtype=float)[:min_len]

                # Ensure at least 50% of the data points are valid
                valid_both = ~np.isnan(arr1) & ~np.isnan(arr2)
                if valid_both.sum() < min_len * 0.5:
                    skip_nan += 1
                    skip = True
                    break

                # Compute z-score difference for valid points only
                if valid_both.sum() > 0:
                    arr1_z = (arr1[valid_both] - feature_mean[feat]) / feature_std[feat]
                    arr2_z = (arr2[valid_both] - feature_mean[feat]) / feature_std[feat]
                    diff += weights[feat] * np.mean(np.abs(arr1_z - arr2_z))

            if not skip:
                # Append similarity score for this historical set
                results.append(
                    {"date": key[0], "予冷時間": key[2], "台数": key[1], "diff": diff}
                )

        # Sort results: lower diff is better, higher number of units preferred
        results = sorted(results, key=lambda x: (x["diff"], -int(x["台数"])))

        logging.info(f"Total historical sets processed: {total}")
        logging.info(f"Skipped sets due to insufficient valid data: {skip_nan}")
        logging.info(f"Number of resulting similar sets: {len(results)}")

        return results[:n]

    def calculate_optimization_scores(
        self,
        filtered_sets: List[Dict],
        data_dict: Dict,
        target_temp: float = 26.0,
        power_start_time: str = "00:00",
        power_end_time: str = "09:30",
    ) -> Tuple[List[Dict], List[float], List[float]]:
        """
        Calculate optimization scores based on predicted indoor temperature and energy consumption.

        Args:
            filtered_sets: List of top similar sets
            data_dict: Historical data dictionary
            target_temp: Target indoor temperature to achieve
            power_start_time: Start time for total power calculation
            power_end_time: End time for total power calculation

        Returns:
            Tuple containing:
                - List of dictionaries with each set, power_sum, and temp_diff
                - List of total power consumption values
                - List of temperature differences from target
        """
        scores = []
        power_sums = []
        temp_diffs = []

        for s in filtered_sets:
            key = (s["date"], s["台数"], s["予冷時間"])
            records = data_dict.get(key)
            if not records:
                continue
            df = pd.DataFrame(records)

            # Get indoor temperature at 07:00
            temp_row = df[df["datetime"].dt.strftime("%H:%M") == "07:00"]
            if temp_row.empty or pd.isna(temp_row.iloc[0]["室内温度"]):
                continue
            temp_diff = abs(temp_row.iloc[0]["室内温度"] - target_temp)

            # Calculate total power consumption over the specified time range
            mask = (df["datetime"].dt.strftime("%H:%M") >= power_start_time) & (
                df["datetime"].dt.strftime("%H:%M") <= power_end_time
            )
            power_sum = df.loc[mask, "電力量"].sum()

            # Skip sets with zero power consumption (likely invalid or missing data)
            if power_sum == 0:
                continue

            power_sums.append(power_sum)
            temp_diffs.append(temp_diff)
            scores.append({"set": s, "power_sum": power_sum, "temp_diff": temp_diff})

        return scores, power_sums, temp_diffs

    def normalize_scores(
        self, scores: List[Dict], power_sums: List[float], temp_diffs: List[float]
    ) -> List[Dict]:
        """
        Normalize scores using z-score normalization to combine power and temperature metrics.

        Args:
            scores: List of score dictionaries containing power_sum and temp_diff
            power_sums: List of total power consumption values
            temp_diffs: List of temperature differences from target

        Returns:
            List of dictionaries with additional keys:
                - power_score: normalized power consumption
                - temp_score: normalized temperature difference
                - total_score: sum of normalized scores (lower is better)
        """
        if not power_sums or not temp_diffs:
            return scores

        mean_power = np.mean(power_sums)
        std_power = np.std(power_sums)
        mean_temp = np.mean(temp_diffs)
        std_temp = np.std(temp_diffs)

        for sc in scores:
            sc["power_score"] = (
                (sc["power_sum"] - mean_power) / std_power if std_power > 0 else 0
            )
            sc["temp_score"] = (
                (sc["temp_diff"] - mean_temp) / std_temp if std_temp > 0 else 0
            )
            # Combine power and temperature scores for overall ranking
            sc["total_score"] = sc["power_score"] + sc["temp_score"]

        return scores
