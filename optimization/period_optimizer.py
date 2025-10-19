# -*- coding: utf-8 -*-
"""
期間最適化システム（簡略化版）
============================
各時刻で独立して最適化するシステム
- 電力: 各時刻で最小消費量を選択
- 室温: 快適範囲内を優先、範囲外は最も涼しい温度を選択
- ビームサーチを廃止し、シンプルなフィルタ+選択方式を採用
"""

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from optimization.feature_builder import OptimizationFeatureBuilder
from processing.utilities.category_mapping_loader import normalize_candidate_values
from training.model_builder import EnvPowerModels


def filter_allowed_modes(previous_mode, mode_candidates: List) -> List:
    """
    モード選択ルールに基づいて許可されるモードをフィルタリング

    Mode Selection Rules:
    - If previous mode was anything other than "HEAT":
      → Next mode can only be "OFF", "FAN", or "COOL"
    - If previous mode was "HEAT":
      → Next mode can only be "OFF" or "FAN"

    Args:
        previous_mode: 前回のモード (例: "COOL", "HEAT", "FAN", "OFF" or numeric code)
        mode_candidates: 候補モードのリスト

    Returns:
        許可されたモードのリスト
    """
    if previous_mode is None:
        # 初回は制約なし
        return mode_candidates

    # Convert previous mode to string and normalize to uppercase for comparison
    prev_mode_str = str(previous_mode).upper()

    # Mode codes: COOL=1, HEAT=2, FAN=3, OFF=0
    # Define allowed transitions based on previous mode
    if prev_mode_str == "HEAT" or prev_mode_str == "2":  # HEAT mode (code 2)
        # From HEAT: only OFF or FAN allowed
        allowed = {"OFF", "FAN", "0", "3"}  # OFF=0, FAN=3
    else:
        # From any other mode (COOL=1, FAN=3, OFF=0): OFF, FAN, or COOL allowed
        allowed = {"OFF", "FAN", "COOL", "0", "1", "3"}  # OFF=0, COOL=1, FAN=3

    # Filter candidates to only include allowed modes
    filtered_modes = [mode for mode in mode_candidates if str(mode).upper() in allowed]

    # If no modes are allowed (shouldn't happen with proper setup), return all candidates
    if not filtered_modes:
        return mode_candidates

    return filtered_modes


def optimize_zone_period(
    zone_name: str,
    zone_data: dict,
    models: EnvPowerModels,
    date_range: pd.DatetimeIndex,
    weather_df: pd.DataFrame,
) -> Tuple[str, Dict[pd.Timestamp, dict]]:
    """
    簡略化された期間最適化システム
    - 営業時間内のみ最適化を実行（start_time-end_time）
    - 営業時間外は自動的にOFFモードに設定
    - 営業時間内では快適温度範囲内の組み合わせをフィルタ
    - その中から電力消費量が最小の組み合わせを選択
    - 快適範囲外の場合は最も涼しい（快適範囲に近い）組み合わせを選択

    戻り値: (ゾーン名, スケジュール辞書)
    """
    print(f"[PeriodOptimizer] Starting period optimization for zone: {zone_name}")

    # Initialize feature builder
    feature_builder = OptimizationFeatureBuilder()

    # ゾーン設定の取得
    start_time_str = str(zone_data.get("start_time", "07:00"))
    end_time_str = str(zone_data.get("end_time", "20:00"))
    start_h = int(start_time_str.split(":")[0])
    start_m = int(start_time_str.split(":")[1]) if ":" in start_time_str else 0
    end_h = int(end_time_str.split(":")[0])
    end_m = int(end_time_str.split(":")[1]) if ":" in end_time_str else 0
    comfort_min = float(zone_data.get("comfort_min", 22.0))
    comfort_max = float(zone_data.get("comfort_max", 25.0))

    # 室内機数の計算
    unit_count = 0
    for _, ou in zone_data.get("outdoor_units", {}).items():
        unit_count += len(ou.get("indoor_units", []))
    unit_count = max(unit_count, 1)

    # 候補の生成
    sp_min = float(zone_data.get("setpoint_min"))
    sp_max = float(zone_data.get("setpoint_max"))
    # Generate temperature candidates with 0.5-degree steps
    sp_list = []
    temp = sp_min
    while temp <= sp_max:
        sp_list.append(temp)
        temp += 0.5

    mode_candidates = zone_data.get("mode_candidates")
    if mode_candidates is not None and not isinstance(
        mode_candidates, (list, tuple, set)
    ):
        mode_candidates = [mode_candidates]
    mode_list = normalize_candidate_values(
        "A/C Mode", mode_candidates, ("COOL", "HEAT", "FAN")
    )

    fan_candidates = zone_data.get("fan_candidates")
    if fan_candidates is not None and not isinstance(
        fan_candidates, (list, tuple, set)
    ):
        fan_candidates = [fan_candidates]
    fan_list = normalize_candidate_values(
        "A/C Fan Speed", fan_candidates, ("Low", "Medium", "High")
    )

    print(
        f"[PeriodOptimizer] Zone {zone_name}: Business hours {start_time_str}-{end_time_str}, "
        f"Comfort range {comfort_min}-{comfort_max}°C, "
        f"Set temp range {sp_min:.1f}-{sp_max:.1f}°C, "
        f"candidates={len(sp_list)}×{len(mode_list)}×{len(fan_list)}"
    )
    print(
        f"[PeriodOptimizer] Zone {zone_name}: Fan options from 制御マスタ = {fan_candidates} → Normalized = {fan_list}"
    )

    # 天候データの準備
    if weather_df is None or weather_df.empty:
        raise ValueError(
            "天候データが提供されていません。APIからデータを取得してください。"
        )

    weather_dict = {}
    for _, row in weather_df.iterrows():
        weather_dict[row["datetime"]] = {
            "outdoor_temp": (
                row["Outdoor Temp."]
                if "Outdoor Temp." in row
                else row.get("Outdoor Temp.", np.nan)
            ),
            "outdoor_humidity": (
                row["Outdoor Humidity"]
                if "Outdoor Humidity" in row
                else row.get("Outdoor Humidity", np.nan)
            ),
            "solar_radiation": row.get("Solar Radiation", 0),
        }

    # 初期温度
    initial_temp = float(zone_data.get("target_room_temp", 25.0))
    last_temp = initial_temp
    last_set_temp = None  # Track previous hour's set temperature for constraint
    last_mode = None  # Track previous hour's mode for mode transition constraints

    # スケジュール辞書
    schedule = {}
    total_power = 0.0
    comfort_violations = 0
    temp_change_violations = (
        0  # Track how many times temp change constraint is violated
    )

    # 各時刻で最適化
    for timestamp in date_range:
        # Check if current hour overlaps with business hours
        # Business hours logic: If the hour contains any part of business hours, optimize for that hour
        current_hour = timestamp.hour
        current_minute = timestamp.minute
        current_time_minutes = current_hour * 60 + current_minute
        start_time_minutes = start_h * 60 + start_m
        end_time_minutes = end_h * 60 + end_m

        # Hour boundaries: current hour starts at current_hour:00 and ends at (current_hour+1):00
        hour_start_minutes = current_hour * 60
        hour_end_minutes = (current_hour + 1) * 60

        # Check if this hour overlaps with business hours
        # Hour overlaps if: hour_start < business_end AND hour_end > business_start
        is_biz = (
            hour_start_minutes < end_time_minutes
            and hour_end_minutes > start_time_minutes
        )

        weather = weather_dict.get(
            timestamp,
            {
                "outdoor_temp": 25.0,
                "outdoor_humidity": 60.0,
                "solar_radiation": 0,
            },
        )

        if not is_biz:
            # Non-business hours: Only set essential fields, others remain empty
            best_combination =             {
                "set_temp": None,  # No set temperature needed outside business hours
                "mode": None,  # No mode setting needed outside business hours
                "fan": None,  # No fan setting needed outside business hours
                "pred_temp": 0.0,  # No temperature prediction needed outside business hours
                "pred_power": 0.0,  # No power consumption when OFF (needed for power calculation)
                "outside_temp": weather[
                    "outdoor_temp"
                ],  # Weather data needed for context
            }
        else:
            # Business hours: Run full optimization
            # 全ての組み合わせを評価
            all_combinations = []
            valid_combinations = []

            # Apply mode transition constraints
            allowed_modes = filter_allowed_modes(last_mode, mode_list)

            # Count violations if modes were filtered
            if len(allowed_modes) < len(mode_list) and last_mode is not None:
                # This is informational - we're applying the constraint, not violating it
                pass

            for sp in sp_list:
                # Apply temperature change constraint: max ±1°C change from previous hour
                if last_set_temp is not None:
                    temp_change = abs(sp - last_set_temp)
                    if temp_change > 1.0:
                        continue  # Skip this temperature if change exceeds 1°C

                for md in allowed_modes:
                    for fs in fan_list:
                        # 特徴量の作成
                        base_features = {
                            "A/C Set Temperature": sp,
                            "Indoor Temp. Lag1": last_temp,
                            "A/C ON/OFF": 1,  # Always 1 during business hours
                            "A/C Mode": md,
                            "A/C Fan Speed": fs,
                            "Outdoor Temp.": weather["outdoor_temp"],
                            "Outdoor Humidity": weather["outdoor_humidity"],
                            "Solar Radiation": weather["solar_radiation"],
                        }

                        # Build complete feature set using feature builder
                        features_df = feature_builder.build_features(
                            base_features=base_features,
                            timestamp=timestamp,
                            zone_name=zone_name,
                            weather_history=None,
                            power_history=None,
                        )

                        # Select only the features the model expects
                        features = features_df[models.feature_cols]

                        # 予測（マルチアウトプットモデルが利用可能な場合は使用）
                        if models.multi_output_model is not None:
                            multi_pred = models.multi_output_model.predict(features)
                            temp_pred = float(multi_pred[0][0])
                            power_pred = float(multi_pred[0][1]) * unit_count
                        else:
                            temp_pred = float(models.temp_model.predict(features)[0])
                            # 電力予測：ビジネス時間中は通常の予測
                            base_power_pred = float(
                                models.power_model.predict(features)[0]
                            )
                            power_pred = base_power_pred * unit_count

                        combination = {
                            "set_temp": sp,
                            "mode": md,
                            "fan": fs,
                            "pred_temp": temp_pred,
                            "pred_power": power_pred,
                            "outside_temp": weather["outdoor_temp"],
                        }

                        # 負の電力予測値でない組み合わせのみ追加
                        if power_pred > 0:
                            all_combinations.append(combination)

                            # 快適範囲内の組み合わせをフィルタ
                            if comfort_min <= temp_pred <= comfort_max:
                                valid_combinations.append(combination)

            # 最適な組み合わせを選択
            if valid_combinations:
                # 快適範囲内で電力消費量が最小の組み合わせ
                best_combination = min(
                    valid_combinations, key=lambda x: x["pred_power"]
                )
            else:
                # 快適範囲外の場合は最も涼しい（快適範囲に近い）組み合わせ
                # 夏場は涼しい方を優先（comfort_minに近い方）
                if all_combinations:
                    best_combination = min(
                        all_combinations,
                        key=lambda x: abs(x["pred_temp"] - comfort_min),
                    )
                else:
                    # No valid combinations due to temperature change constraint
                    # Relax constraint and pick closest valid temperature
                    temp_change_violations += 1
                    # Find the closest temperature to last_set_temp within ±1°C
                    if last_set_temp is not None:
                        closest_temp = max(sp_min, min(sp_max, last_set_temp))
                    else:
                        closest_temp = sp_min

                    # Re-evaluate with the closest temperature (still respecting mode constraints)
                    for md in allowed_modes:
                        for fs in fan_list:
                            base_features = {
                                "A/C Set Temperature": closest_temp,
                                "Indoor Temp. Lag1": last_temp,
                                "A/C ON/OFF": 1,  # Always 1 during business hours
                                "A/C Mode": md,
                                "A/C Fan Speed": fs,
                                "Outdoor Temp.": weather["outdoor_temp"],
                                "Outdoor Humidity": weather["outdoor_humidity"],
                                "Solar Radiation": weather["solar_radiation"],
                            }
                            features_df = feature_builder.build_features(
                                base_features=base_features,
                                timestamp=timestamp,
                                zone_name=zone_name,
                                weather_history=None,
                                power_history=None,
                            )
                            features = features_df[models.feature_cols]

                            if models.multi_output_model is not None:
                                multi_pred = models.multi_output_model.predict(features)
                                temp_pred = float(multi_pred[0][0])
                                power_pred = float(multi_pred[0][1]) * unit_count
                            else:
                                temp_pred = float(
                                    models.temp_model.predict(features)[0]
                                )
                                base_power_pred = float(
                                    models.power_model.predict(features)[0]
                                )
                                power_pred = base_power_pred * unit_count

                            # 負の電力予測値でない組み合わせのみ追加
                            if power_pred > 0:
                                all_combinations.append(
                                    {
                                        "set_temp": closest_temp,
                                        "mode": md,
                                        "fan": fs,
                                        "pred_temp": temp_pred,
                                        "pred_power": power_pred,
                                        "outside_temp": weather["outdoor_temp"],
                                    }
                                )

                    best_combination = min(
                        all_combinations,
                        key=lambda x: abs(x["pred_temp"] - comfort_min),
                    )
                comfort_violations += 1

        # スケジュールに追加
        schedule[timestamp] = best_combination

        # Track values for next hour (only if not None/0.0)
        if (
            best_combination["pred_temp"] is not None
            and best_combination["pred_temp"] != 0.0
        ):
            last_temp = best_combination["pred_temp"]
        if best_combination["set_temp"] is not None:
            last_set_temp = best_combination["set_temp"]
        if best_combination["mode"] is not None:
            last_mode = best_combination["mode"]

        total_power += best_combination["pred_power"]

    # 結果の表示
    business_hours_count = sum(
        1
        for ts in date_range
        if (ts.hour * 60) < end_time_minutes
        and ((ts.hour + 1) * 60) > start_time_minutes
    )
    print(
        f"[PeriodOptimizer] Zone {zone_name} completed - "
        f"Business hours: {business_hours_count}/{len(date_range)} hours, "
        f"Total power: {total_power:.1f} kWh, "
        f"Comfort violations: {comfort_violations}/{business_hours_count} business hours, "
        f"Temp change violations: {temp_change_violations}/{business_hours_count} business hours, "
        f"Mode transition constraints: Applied"
    )

    return zone_name, schedule


class PeriodOptimizer:
    """期間最適化クラス（簡略化版）

    営業時間ベースの最適化：
    1. 営業時間内のみ最適化を実行（master dataのstart_time-end_time）
    2. 営業時間外は自動的にOFFモードに設定
    3. 営業時間内では快適温度範囲内の組み合わせをフィルタ
    4. その中から電力消費量が最小の組み合わせを選択
    5. 快適範囲外の場合は最も涼しい組み合わせを選択
    """

    def __init__(
        self, master: dict, models: Dict[str, EnvPowerModels], max_workers: int = None
    ):
        self.master = master
        self.models = models
        self.max_workers = max_workers or min(len(models), mp.cpu_count())

    def optimize_period(
        self,
        date_range: pd.DatetimeIndex,
        weather_df: pd.DataFrame,
    ) -> Dict[str, Dict[pd.Timestamp, dict]]:
        """期間最適化を実行"""

        print(
            f"[PeriodOptimizer] Starting period optimization for {len(date_range)} hours"
        )
        print(f"[PeriodOptimizer] Date range: {date_range[0]} to {date_range[-1]}")
        print(f"[PeriodOptimizer] Max workers: {self.max_workers}")
        print(f"[PeriodOptimizer] Available zones: {list(self.models.keys())}")

        # 並列処理の準備
        zone_tasks = []
        for zone_name, models in self.models.items():
            zone_data = self.master.get("zones", {}).get(zone_name, {})
            zone_tasks.append(
                (
                    zone_name,
                    zone_data,
                    models,
                    date_range,
                    weather_df,
                )
            )

        # 並列実行
        results = {}
        start_time = time.perf_counter()

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # タスクの投入（簡略化版を使用）
            future_to_zone = {
                executor.submit(optimize_zone_period, *task): task[0]
                for task in zone_tasks
            }

            # 結果の収集
            completed_count = 0
            for future in as_completed(future_to_zone):
                zone_name = future_to_zone[future]
                try:
                    zone_name, zone_schedule = future.result()
                    results[zone_name] = zone_schedule
                    completed_count += 1
                    print(
                        f"[PeriodOptimizer] Completed {completed_count}/"
                        f"{len(zone_tasks)} zones: {zone_name}"
                    )
                except Exception as exc:
                    print(
                        f"[PeriodOptimizer] Zone {zone_name} generated an exception: {exc}"
                    )

        end_time = time.perf_counter()
        print(
            f"[PeriodOptimizer] Period optimization completed in "
            f"{end_time - start_time:.2f} seconds"
        )
        print(f"[PeriodOptimizer] Optimized {len(results)} zones")

        return results
