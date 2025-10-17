# -*- coding: utf-8 -*-
"""
エアコン最適化システム（STEP1〜STEP4一貫版 / 再考リファクタ）
============================================================
要点（ご要望を反映）
- IF（インタフェース）と前処理、天気予報取得、マスタ情報の活用を最優先で尊重
- 1時間粒度をデフォルトに、予測粒度は可変
- マスタJSONから制御エリア（zones）、室外機/室内機、負荷率(load_share)、目標温度や範囲を取得
- 実績（空調・電力・天候）を整備して、制御エリア単位へ集約
  - 空調: 設定条件は最頻値、室内温度は平均
  - 電力: 室外機の消費電力×負荷率の合計
  - 天候: エリア間で共通
- 予測モデル: 制御エリア別に環境（室温/湿度）と電力を学習
- 最適化: 制御区分毎、パターン探索→環境は平均、電力は合計（電力は室内機数で補正）
- 出力: 制御区分別 & 室内機別（制御区分の運転条件を室内機に展開）

ディレクトリ前提（config.utils.get_data_path を使用）
- raw_data_path/<store> に ac-control-*.csv, ac-power-meter-*.csv
- processed_data_path/<store> に前処理済みCSVを出力
- models_path/<store> に学習済みモデルを保存
- master/MASTER_<store>.json にマスタ
- planning/<store> に制御スケジュールCSV
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from optimization.period_optimizer import PeriodOptimizer
from planning.planner import Planner
from processing.aggregator import AreaAggregator
from processing.preprocessor import DataPreprocessor
from processing.utilities.helper_functions import analyze_feature_correlations
from processing.utilities.weatherapi_client import VisualCrossingWeatherAPIDataFetcher
from training.data_processor import DataProcessor
from training.model_builder import ModelBuilder

warnings.filterwarnings("ignore")


# =============================
# 統合ランナー
# =============================
class AirconOptimizer:
    def __init__(
        self,
        store_name: str,
        enable_preprocessing: bool = True,
        skip_aggregation: bool = False,
        excel_master_data: dict = None,
    ):
        self.store_name = store_name
        self.enable_preprocessing = enable_preprocessing
        self.skip_aggregation = skip_aggregation

        # Use Excel master data if provided, otherwise fall back to JSON
        if excel_master_data is not None:
            self.master = excel_master_data
            print(
                f"[AirconOptimizer] Using consolidated Excel master data for {store_name}"
            )
        else:
            print(f"[AirconOptimizer] No Excel master data provided for {store_name}")

        from config.utils import get_data_path

        self.proc_dir = os.path.join(get_data_path("processed_data_path"), store_name)
        self.plan_dir = os.path.join(get_data_path("output_data_path"), store_name)
        os.makedirs(self.plan_dir, exist_ok=True)

    def _load_processed(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        ac_p = os.path.join(
            self.proc_dir, f"ac_control_processed_{self.store_name}.csv"
        )
        pm_p = os.path.join(
            self.proc_dir, f"power_meter_processed_{self.store_name}.csv"
        )
        weather_p = os.path.join(
            self.proc_dir, f"weather_processed_{self.store_name}.csv"
        )
        ac = pd.read_csv(ac_p) if os.path.exists(ac_p) else None
        pm = pd.read_csv(pm_p) if os.path.exists(pm_p) else None
        weather = pd.read_csv(weather_p) if os.path.exists(weather_p) else None
        return ac, pm, weather

    def _get_weather_forecast_path(self, start_date: str, end_date: str) -> str:
        """
        Generate weather forecast file path with date range in filename

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Full path to the weather forecast file
        """
        # Format dates for filename (remove dashes)
        start_clean = start_date.replace("-", "")
        end_clean = end_date.replace("-", "")
        filename = f"weather_forecast_{start_clean}_{end_clean}.csv"
        return os.path.join(self.plan_dir, filename)

    def _load_weather_forecast(
        self,
        start_date: str,
        end_date: str,
        weather_api_key: Optional[str] = None,
        coordinates: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load weather forecast from cached file if it exists, otherwise fetch from API

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            weather_api_key: Weather API key for fetching data if cache is missing
            coordinates: Coordinates for weather API

        Returns:
            Weather DataFrame if file exists or can be fetched from API, None otherwise
        """
        forecast_path = self._get_weather_forecast_path(start_date, end_date)

        if os.path.exists(forecast_path):
            print(f"[Run] Loading cached weather forecast: {forecast_path}")
            try:
                weather_df = pd.read_csv(forecast_path)

                # Convert datetime column to datetime type if it exists
                if "datetime" in weather_df.columns:
                    weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])
                    print(f"[Run] Converted datetime column to datetime type")

                print(f"[Run] Cached weather data loaded. Shape: {weather_df.shape}")
                return weather_df
            except Exception as e:
                print(f"[Run] Error loading cached weather data: {e}")
                return None
        else:
            print(f"[Run] No cached weather forecast found: {forecast_path}")

            # Try to fetch weather data from API if credentials are provided
            if weather_api_key and coordinates:
                print("[Run] APIから天候データを取得...")
                try:
                    from processing.utilities.weatherapi_client import (
                        VisualCrossingWeatherAPIDataFetcher,
                    )

                    weather_df = VisualCrossingWeatherAPIDataFetcher(
                        coordinates=coordinates,
                        start_date=start_date,
                        end_date=end_date,
                        unit="metric",
                        api_key=weather_api_key,
                    ).fetch()
                    if weather_df is not None:
                        self._save_weather_forecast(weather_df, start_date, end_date)
                        print(
                            "[Run] 天候データをAPIから取得し、キャッシュに保存しました"
                        )
                        return weather_df
                    else:
                        print("[Run] APIから天候データを取得できませんでした")
                        return None
                except Exception as e:
                    print(f"[Run] 天候データ取得エラー: {e}")
                    return None
            else:
                print("[Run] 天候データが見つかりません（APIキーまたは座標が未設定）")
                return None

    def _save_weather_forecast(
        self, weather_df: pd.DataFrame, start_date: str, end_date: str
    ) -> None:
        """
        Save weather forecast to cached file with date range in filename

        Args:
            weather_df: Weather DataFrame to save
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        forecast_path = self._get_weather_forecast_path(start_date, end_date)

        try:
            os.makedirs(os.path.dirname(forecast_path), exist_ok=True)
            weather_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")
            print(f"[Run] Weather forecast cached to: {forecast_path}")
        except Exception as e:
            print(f"[Run] Error saving weather forecast: {e}")

    def _load_features_directly(self) -> Optional[pd.DataFrame]:
        """
        Load features directly from the processed CSV file, skipping aggregation
        """
        features_path = os.path.join(
            self.proc_dir, f"features_processed_{self.store_name}.csv"
        )

        if os.path.exists(features_path):
            print(f"[Run] Loading features directly from: {features_path}")
            try:
                area_df = pd.read_csv(features_path)
                print(f"[Run] Features loaded successfully. Shape: {area_df.shape}")

                if "zone" in area_df.columns:
                    zones = area_df["zone"].unique()
                    print(f"[Run] Zones found: {zones}")
                else:
                    print("[Run] Warning: No 'zone' column found in features data")

                return area_df
            except Exception as e:
                print(f"[Run] Error loading features: {e}")
                return None
        else:
            print(f"[Run] Features file not found: {features_path}")
            return None

    def run(
        self,
        weather_api_key: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        freq: str = "1H",
        temperature_std_multiplier: float = 5.0,
        power_std_multiplier: float = 5.0,
    ):
        """全ての処理を実行
        実行順序: 前処理 -> 集約 -> モデル学習 -> 最適化

        Args:
            weather_api_key: Weather API キー
            start_date: 開始日
            end_date: 終了日
            freq: 時間粒度
            temperature_std_multiplier: 温度データの外れ値判定係数
            power_std_multiplier: 電力データの外れ値判定係数

        Returns:
            dict: 最適化結果 (モデル, スケジュール)
        """
        if self.master is None:
            print("[Run] マスタ未読込")
            return None

        # 処理時間計測開始
        total_start_time = time.perf_counter()
        processing_times = {}

        # Get coordinates from self.master
        coordinates = self.master.get("store_info", {}).get("coordinates")
        if coordinates is None:
            print(f"[Run] ERROR: No coordinates found in master data")
            return None
        else:
            print(f"[Run] Using coordinates from master data: {coordinates}")

        # STEP1: 前処理
        if self.enable_preprocessing:
            preprocessing_start_time = time.perf_counter()
            print("[Run] Starting preprocessing...")
            print(f"[Run] Temperature std multiplier: {temperature_std_multiplier}")
            print(f"[Run] Power std multiplier: {power_std_multiplier}")
            preprocessor = DataPreprocessor(self.store_name)
            print("[Run] DataPreprocessor created, loading raw data...")
            ac_raw_data, pm_raw_data = preprocessor.load_raw()
            print("[Run] Raw data loaded, preprocessing AC data...")
            ac_processed_data = preprocessor.preprocess_ac(
                ac_raw_data, temperature_std_multiplier
            )
            print("[Run] AC data preprocessed, preprocessing PM data...")
            pm_processed_data = preprocessor.preprocess_pm(
                pm_raw_data, power_std_multiplier
            )
            print("[Run] PM data preprocessed, saving...")
            print("[Run] PM data preprocessed, checking for cached weather data...")

            # Check if weather_processed file exists
            weather_file = os.path.join(
                self.proc_dir, f"weather_processed_{self.store_name}.csv"
            )
            if os.path.exists(weather_file):
                print(f"[Run] Found cached weather data: {weather_file}")
                historical_weather_data = pd.read_csv(weather_file)
                # Convert datetime column if it exists
                if "datetime" in historical_weather_data.columns:
                    historical_weather_data["datetime"] = pd.to_datetime(
                        historical_weather_data["datetime"]
                    )
                print(
                    f"[Run] Loaded cached weather data: {len(historical_weather_data)} records"
                )
            else:
                print("[Run] No cached weather data found, fetching from API...")
                historical_weather_data = preprocessor._fetch_historical_weather(
                    ac_processed_data, pm_processed_data, weather_api_key, coordinates
                )
                print("[Run] Historical weather data fetched from API, saving...")
            print("[Run] Saving processed data...")
            preprocessor.save(
                ac_processed_data,
                pm_processed_data,
                historical_weather_data,
                export_temp_range_stats=False,
            )
            preprocessing_end_time = time.perf_counter()
            processing_times["前処理"] = (
                preprocessing_end_time - preprocessing_start_time
            )
            print(
                f"[Run] Preprocessing completed - 処理時間: {processing_times['前処理']:.2f}秒"
            )
        else:
            print("[Run] Loading processed data...")
            ac_processed_data, pm_processed_data, historical_weather_data = (
                self._load_processed()
            )
            print("[Run] Processed data loaded")
        # 天候データ取得（実績期間＋最適化期間をカバー）
        weather_start_time = time.perf_counter()

        # 実績期間の推定（前処理済みデータから）
        actual_start_dt = None
        actual_end_dt = None
        try:
            if ac_processed_data is not None and not ac_processed_data.empty:
                ac_dt = pd.to_datetime(ac_processed_data.get("datetime"))
                actual_start_dt = (
                    ac_dt.min()
                    if actual_start_dt is None
                    else min(actual_start_dt, ac_dt.min())
                )
                actual_end_dt = (
                    ac_dt.max()
                    if actual_end_dt is None
                    else max(actual_end_dt, ac_dt.max())
                )
            if pm_processed_data is not None and not pm_processed_data.empty:
                pm_dt = pd.to_datetime(pm_processed_data.get("datetime"))
                actual_start_dt = (
                    pm_dt.min()
                    if actual_start_dt is None
                    else min(actual_start_dt, pm_dt.min())
                )
                actual_end_dt = (
                    pm_dt.max()
                    if actual_end_dt is None
                    else max(actual_end_dt, pm_dt.max())
                )
        except Exception:
            # 無視（実績期間が取れない場合は最適化期間のみ）
            pass

        # 最適化期間の既定
        if start_date is None or end_date is None:
            tomorrow = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
            start_date = tomorrow.strftime("%Y-%m-%d")
            end_date = (tomorrow + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

        # 実績期間と最適化期間を統合した取得レンジ
        combined_start_dt = pd.to_datetime(start_date)
        combined_end_dt = pd.to_datetime(end_date)
        if actual_start_dt is not None:
            combined_start_dt = min(combined_start_dt, actual_start_dt.normalize())
        if actual_end_dt is not None:
            combined_end_dt = max(combined_end_dt, actual_end_dt.normalize())
        combined_start_date = combined_start_dt.strftime("%Y-%m-%d")
        combined_end_date = combined_end_dt.strftime("%Y-%m-%d")

        print(f"[Run] Weather API Key provided: {weather_api_key is not None}")
        if weather_api_key:
            print(f"[Run] Weather API Key: {weather_api_key[:10]}...")
        else:
            print("[Run] No Weather API Key provided")

        print(f"[Run] Date range (optimization): {start_date} to {end_date}")
        print(
            f"[Run] Date range (weather fetch combined): {combined_start_date} to {combined_end_date}"
        )
        print(f"[Run] Coordinates: {coordinates}")

        # ----------------------------

        # 天候データ取得 (Cached or API)
        # ----------------------------
        weather_df = None

        # Load weather data (from cache or API)
        weather_df = self._load_weather_forecast(
            start_date, end_date, weather_api_key, coordinates
        )

        # If no weather data found
        if weather_df is None:
            print("[Run] 天候データが見つかりません")

        # 天候データの統合（履歴 + 未来）
        combined_weather_df = None
        if historical_weather_data is not None and not historical_weather_data.empty:
            print("[Run] Combining historical and future weather data...")
            if weather_df is not None and not weather_df.empty:
                # 重複を避けるため、未来の天候データから履歴期間を除外
                historical_max_date = pd.to_datetime(
                    historical_weather_data["datetime"]
                ).max()
                weather_df_filtered = weather_df[
                    pd.to_datetime(weather_df["datetime"]) > historical_max_date
                ]
                if not weather_df_filtered.empty:
                    combined_weather_df = pd.concat(
                        [historical_weather_data, weather_df_filtered],
                        ignore_index=True,
                    )
                    print(
                        f"[Run] Combined weather data: {len(historical_weather_data)} historical + {len(weather_df_filtered)} future records"
                    )
                else:
                    combined_weather_df = historical_weather_data
                    print(
                        "[Run] Using only historical weather data (no future data needed)"
                    )
            else:
                combined_weather_df = historical_weather_data
                print("[Run] Using only historical weather data")
        else:
            combined_weather_df = weather_df
            print("[Run] Using only future weather data")

        weather_end_time = time.perf_counter()
        processing_times["天候データ取得"] = weather_end_time - weather_start_time
        print(
            f"[Run] Weather data processing completed - 処理時間: {processing_times['天候データ取得']:.2f}秒"
        )

        # 制御エリア集約
        aggregation_start_time = time.perf_counter()

        if self.skip_aggregation:
            print("[Run] Skipping aggregation, loading features directly...")
            area_df = self._load_features_directly()
        else:
            print("[Run] Starting area aggregation...")
            # Use master data from constructor
            if self.master is None:
                print("[Run] ERROR: Master data not available for aggregator")
                return None

            # Extract zones data for aggregator
            aggregator_data = {
                "store_name": self.master.get("store_info", {}).get(
                    "name", self.store_name
                ),
                "zones": self.master.get("zones", {}),
            }
            aggregator = AreaAggregator(aggregator_data)
            area_df = aggregator.build(
                ac_processed_data, pm_processed_data, combined_weather_df, freq=freq
            )
            area_out = os.path.join(
                self.proc_dir, f"features_processed_{self.store_name}.csv"
            )
            os.makedirs(self.proc_dir, exist_ok=True)
            area_df.to_csv(area_out, index=False, encoding="utf-8-sig")
            print(f"[Run] Area data saved to: {area_out}")
            print(
                f"[Run] Area aggregation completed. Shape: {area_df.shape if area_df is not None else 'None'}"
            )

        if area_df is not None and not area_df.empty:
            print(f"[Run] Area data columns: {list(area_df.columns)}")
            print(
                f"[Run] Zones found: {area_df['zone'].unique() if 'zone' in area_df.columns else 'No zone column'}"
            )
            print(
                f"[Run] Adjusted power data: {area_df['adjusted_power'].notna().sum() if 'adjusted_power' in area_df.columns else 'No adjusted_power column'}"
            )

        # 特徴量の確認と相関の簡易レポート
        if area_df is not None:
            # Use helper function for correlation analysis
            correlation_results = analyze_feature_correlations(area_df)

            # Process features for model training
            print("[Run] Processing features for model training...")
            data_processor = DataProcessor()
            area_df = data_processor.process_features(area_df)

            # Print feature summary
            data_processor.print_feature_summary(area_df)

        # 天気予報データの出力（最適化期間のみ）
        if weather_df is not None:
            forecast_df = weather_df[
                (weather_df["datetime"] >= pd.to_datetime(start_date))
                & (weather_df["datetime"] <= pd.to_datetime(end_date))
            ].copy()

            # Save forecast with date-based filename (already cached above, but save filtered version too)
            forecast_path = self._get_weather_forecast_path(start_date, end_date)
            os.makedirs(os.path.dirname(forecast_path), exist_ok=True)
            forecast_df.to_csv(forecast_path, index=False, encoding="utf-8-sig")
            print(f"[Run] Weather forecast saved to: {forecast_path}")

        aggregation_end_time = time.perf_counter()
        processing_times["エリア集約"] = aggregation_end_time - aggregation_start_time
        print(
            f"[Run] Area aggregation completed - 処理時間: {processing_times['エリア集約']:.2f}秒"
        )

        # STEP2: 予測モデル
        model_training_start_time = time.perf_counter()
        print("[Run] Starting model training...")
        builder = ModelBuilder(self.store_name)
        models = builder.train_by_zone(area_df)
        model_training_end_time = time.perf_counter()
        processing_times["モデル学習"] = (
            model_training_end_time - model_training_start_time
        )
        print(
            f"[Run] Model training completed. Models created: {len(models)} - 処理時間: {processing_times['モデル学習']:.2f}秒"
        )
        if not models:
            print("[Run] モデル作成不可（データ不足）")
            return None

        # STEP3: 最適化（並列処理版）
        optimization_start_time = time.perf_counter()
        date_range = pd.date_range(
            start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq=freq
        )
        date_range = date_range[(date_range.hour >= 0) & (date_range.hour <= 23)]

        # 期間最適化版を使用（簡略化版）
        opt = PeriodOptimizer(self.master, models, max_workers=6)  # 6ゾーン分
        schedule = opt.optimize_period(date_range, weather_df)
        optimization_end_time = time.perf_counter()
        processing_times["最適化"] = optimization_end_time - optimization_start_time
        print(
            f"[Run] Optimization completed - 処理時間: {processing_times['最適化']:.2f}秒"
        )

        # STEP4: 出力
        output_start_time = time.perf_counter()
        Planner(self.store_name, self.master).export(schedule, self.plan_dir)
        output_end_time = time.perf_counter()
        processing_times["計画出力"] = output_end_time - output_start_time
        print(
            f"[Run] Planning output completed - 処理時間: {processing_times['計画出力']:.2f}秒"
        )

        # 総処理時間の表示
        total_end_time = time.perf_counter()
        processing_times["総処理時間"] = total_end_time - total_start_time

        print(f"\n{'='*60}")
        print("📊 処理時間サマリー")
        print(f"{'='*60}")
        for process_name, duration in processing_times.items():
            if process_name != "総処理時間":
                percentage = (duration / processing_times["総処理時間"]) * 100
                print(f"{process_name:12}: {duration:6.2f}秒 ({percentage:5.1f}%)")
        print(f"{'='*60}")
        print(f"{'総処理時間':12}: {processing_times['総処理時間']:6.2f}秒 (100.0%)")
        print(f"{'='*60}")
        # schedule = None
        return schedule

    def run_preprocessing_only(
        self,
        weather_api_key: Optional[str] = None,
        temperature_std_multiplier: float = 5.0,
        power_std_multiplier: float = 5.0,
    ):
        """
        前処理のみを実行

        Args:
            weather_api_key: Weather API キー
            temperature_std_multiplier: 温度データの外れ値判定係数
            power_std_multiplier: 電力データの外れ値判定係数

        Returns:
            bool: 成功した場合True
        """
        if self.master is None:
            print("[Preprocess] マスタ未読込")
            return False

        print("[Preprocess] 前処理のみ実行開始...")

        # Get coordinates from self.master
        coordinates = self.master.get("store_info", {}).get("coordinates")
        if coordinates is None:
            print(f"[Preprocess] ERROR: No coordinates provided")
            return False
        else:
            print(f"[Preprocess] Using provided coordinates: {coordinates}")

        # 前処理の実行
        preprocessor = DataPreprocessor(self.store_name)
        ac_raw_data, pm_raw_data = preprocessor.load_raw()
        ac_processed_data = preprocessor.preprocess_ac(
            ac_raw_data, temperature_std_multiplier
        )
        pm_processed_data = preprocessor.preprocess_pm(
            pm_raw_data, power_std_multiplier
        )

        # 天候データの処理
        weather_file = os.path.join(
            self.proc_dir, f"weather_processed_{self.store_name}.csv"
        )
        if os.path.exists(weather_file):
            print(f"[Preprocess] キャッシュされた天候データを使用: {weather_file}")
            historical_weather_data = pd.read_csv(weather_file)
            if "datetime" in historical_weather_data.columns:
                historical_weather_data["datetime"] = pd.to_datetime(
                    historical_weather_data["datetime"]
                )
        else:
            print("[Preprocess] APIから天候データを取得...")
            historical_weather_data = preprocessor._fetch_historical_weather(
                ac_processed_data, pm_processed_data, weather_api_key, coordinates
            )

        # データの保存
        preprocessor.save(
            ac_processed_data,
            pm_processed_data,
            historical_weather_data,
            export_temp_range_stats=False,
        )
        print("[Preprocess] 前処理完了")
        return True

    def run_aggregation_only(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        weather_api_key: Optional[str] = None,
        freq: str = "1H",
    ):
        """
        集約のみを実行

        Args:
            start_date: 開始日
            end_date: 終了日
            weather_api_key: Weather API キー
            freq: 時間粒度

        Returns:
            pd.DataFrame: 集約されたデータ
        """
        if self.master is None:
            print("[Aggregate] マスタ未読込")
            return None

        print("[Aggregate] 集約のみ実行開始...")

        # Get coordinates from self.master
        coordinates = self.master.get("store_info", {}).get("coordinates")
        if coordinates is None:
            print(f"[Aggregate] ERROR: No coordinates found in master data")
            return None
        else:
            print(f"[Aggregate] Using coordinates from master data: {coordinates}")

        # 処理済みデータの読み込み
        ac_processed_data, pm_processed_data, historical_weather_data = (
            self._load_processed()
        )

        if ac_processed_data is None or pm_processed_data is None:
            print("[Aggregate] 処理済みデータが見つかりません")
            return None

        # 座標情報をマスタから取得
        if coordinates is None:
            coordinates = self.master.get("store_info", {}).get("coordinates")

        # 天候データの取得
        weather_df = None
        if start_date and end_date:
            weather_df = self._load_weather_forecast(
                start_date, end_date, weather_api_key, coordinates
            )

        # 天候データの統合
        combined_weather_df = None
        if historical_weather_data is not None and not historical_weather_data.empty:
            if weather_df is not None and not weather_df.empty:
                historical_max_date = pd.to_datetime(
                    historical_weather_data["datetime"]
                ).max()
                weather_df_filtered = weather_df[
                    pd.to_datetime(weather_df["datetime"]) > historical_max_date
                ]
                if not weather_df_filtered.empty:
                    combined_weather_df = pd.concat(
                        [historical_weather_data, weather_df_filtered],
                        ignore_index=True,
                    )
                else:
                    combined_weather_df = historical_weather_data
            else:
                combined_weather_df = historical_weather_data
        else:
            combined_weather_df = weather_df

        # 集約の実行
        # Use master data from constructor
        if self.master is None:
            print("[Aggregate] ERROR: Master data not available for aggregator")
            return None

        # Extract zones data for aggregator
        aggregator_data = {
            "store_name": self.master.get("store_info", {}).get(
                "name", self.store_name
            ),
            "zones": self.master.get("zones", {}),
        }
        aggregator = AreaAggregator(aggregator_data)
        area_df = aggregator.build(
            ac_processed_data, pm_processed_data, combined_weather_df, freq=freq
        )

        # 特徴量の処理
        if area_df is not None:
            from processing.utilities.helper_functions import (
                analyze_feature_correlations,
            )
            from training.data_processor import DataProcessor

            correlation_results = analyze_feature_correlations(area_df)
            data_processor = DataProcessor()
            area_df = data_processor.process_features(area_df)
            data_processor.print_feature_summary(area_df)

        # データの保存
        if area_df is not None:
            area_out = os.path.join(
                self.proc_dir, f"features_processed_{self.store_name}.csv"
            )
            os.makedirs(self.proc_dir, exist_ok=True)
            area_df.to_csv(area_out, index=False, encoding="utf-8-sig")
            print(f"[Aggregate] 集約データを保存: {area_out}")

        print("[Aggregate] 集約完了")
        return area_df

    def run_training_only(self):
        """
        モデル学習のみを実行

        Returns:
            dict: 学習済みモデル
        """
        if self.master is None:
            print("[Train] マスタ未読込")
            return None

        print("[Train] モデル学習のみ実行開始...")

        # 特徴量データの読み込み
        area_df = self._load_features_directly()
        if area_df is None:
            print("[Train] 特徴量データが見つかりません")
            return None

        # モデル学習の実行
        from training.model_builder import ModelBuilder

        builder = ModelBuilder(self.store_name)
        models = builder.train_by_zone(area_df)

        print(f"[Train] モデル学習完了. 作成されたモデル数: {len(models)}")
        return models

    def run_optimization_only(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        weather_api_key: Optional[str] = None,
        coordinates: Optional[str] = None,
        freq: str = "1H",
    ):
        """
        最適化のみを実行

        Args:
            start_date: 開始日
            end_date: 終了日
            weather_api_key: Weather API キー
            coordinates: 座標
            freq: 時間粒度

        Returns:
            dict: 最適化結果
        """
        if self.master is None:
            print("[Optimize] マスタ未読込")
            return None

        print("[Optimize] 最適化のみ実行開始...")

        # 日付の設定
        if start_date is None or end_date is None:
            tomorrow = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
            start_date = tomorrow.strftime("%Y-%m-%d")
            end_date = (tomorrow + pd.Timedelta(days=3)).strftime("%Y-%m-%d")

        # 座標の設定（マスタデータから取得）
        if coordinates is None:
            coordinates = self.master.get("store_info", {}).get("coordinates")

        # モデルの読み込み
        from training.model_builder import ModelBuilder

        builder = ModelBuilder(self.store_name)
        models = builder.load_models()

        if not models:
            print("[Optimize] モデルが見つかりません")
            return None

        # 天候データの読み込み
        weather_df = self._load_weather_forecast(
            start_date, end_date, weather_api_key, coordinates
        )
        if weather_df is None:
            print("[Optimize] 天候データが見つかりません")
            return None

        # 最適化の実行
        date_range = pd.date_range(
            start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq=freq
        )
        date_range = date_range[(date_range.hour >= 0) & (date_range.hour <= 23)]

        opt = PeriodOptimizer(self.master, models, max_workers=6)
        schedule = opt.optimize_period(date_range, weather_df)

        # 出力の生成
        Planner(self.store_name, self.master).export(schedule, self.plan_dir)

        print("[Optimize] 最適化完了")
        return schedule


if __name__ == "__main__":
    # 例:
    pass
