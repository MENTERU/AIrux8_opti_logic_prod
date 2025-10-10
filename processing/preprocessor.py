import glob
import os
from typing import List, Optional, Tuple

import pandas as pd

from processing.utilities.category_mapping_loader import (
    get_default_category_value,
    map_category_series,
)
from processing.utilities.temp_range_export import export_temp_range_stats
from processing.utilities.weatherapi_client import VisualCrossingWeatherAPIDataFetcher


# =============================
# STEP1: 前処理
# =============================
class DataPreprocessor:
    def __init__(self, store_name: str):
        self.store_name = store_name
        from config.utils import get_data_path

        self.data_dir = os.path.join(get_data_path("raw_data_path"), store_name)
        self.output_dir = os.path.join(get_data_path("processed_data_path"), store_name)
        os.makedirs(self.output_dir, exist_ok=True)

    # 共通
    @staticmethod
    def _unify_datetime(
        df: pd.DataFrame,
    ) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        cols = [c for c in df.columns if "datetime" in c.lower() or "日時" in c]
        if not cols:
            return None, None
        col = cols[0]
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None).dt.floor("T")
        return df, col

    @staticmethod
    def _rm_dup(df: pd.DataFrame, dt_col: str) -> pd.DataFrame:
        dev_col = (
            "A/C Name"
            if "A/C Name" in df.columns
            else ("Mesh ID" if "Mesh ID" in df.columns else None)
        )
        return df.drop_duplicates(subset=[dt_col, dev_col]) if dev_col else df

    @staticmethod
    def _rm_outliers(
        df: pd.DataFrame, columns: List[str], standard_deviation_multiplier: float = 3.0
    ) -> pd.DataFrame:
        for column in columns:
            if column in df.columns:
                mean_value, standard_deviation = df[column].mean(), df[column].std()
                df = df[
                    (df[column] - mean_value).abs()
                    <= standard_deviation_multiplier * standard_deviation
                ]
        return df

    def load_raw(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        print(f"[DataPreprocessor] Loading raw data from: {self.data_dir}")
        print(f"[DataPreprocessor] Directory exists: {os.path.exists(self.data_dir)}")

        ac_files = glob.glob(f"{self.data_dir}/**/ac-control-*.csv", recursive=True)
        pm_files = glob.glob(f"{self.data_dir}/**/ac-power-meter-*.csv", recursive=True)

        print(f"[DataPreprocessor] Found {len(ac_files)} AC control files")
        print(f"[DataPreprocessor] Found {len(pm_files)} power meter files")

        if ac_files:
            print(
                f"[DataPreprocessor] AC files: {ac_files[:1]}..."
            )  # Show first 3 files
        if pm_files:
            print(
                f"[DataPreprocessor] PM files: {pm_files[:1]}..."
            )  # Show first 3 files

        ac = (
            pd.concat([pd.read_csv(f) for f in ac_files], ignore_index=True)
            if ac_files
            else None
        )
        pm = (
            pd.concat([pd.read_csv(f) for f in pm_files], ignore_index=True)
            if pm_files
            else None
        )

        print(
            f"[DataPreprocessor] AC data shape: {ac.shape if ac is not None else 'None'}"
        )
        print(
            f"[DataPreprocessor] PM data shape: {pm.shape if pm is not None else 'None'}"
        )

        return ac, pm

    def _apply_categorical_mapping(
        self, dataframe: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        """共通のカテゴリカル変数マッピングを適用"""
        if column not in dataframe.columns:
            return dataframe

        # マッピング前の値の確認
        original_values = dataframe[column].value_counts()
        print(
            f"[DataPreprocessor] {column} マッピング前の値: "
            f"{original_values.head().to_dict()}"
        )

        original_series = dataframe[column]
        mapped_series, applied_mapping, unmapped_values = map_category_series(
            original_series, column
        )
        dataframe[column] = mapped_series

        if unmapped_values:
            print(
                f"[DataPreprocessor] {column} マッピングされなかった値: "
                f"{unmapped_values}"
            )
            unmapped_mask = mapped_series.isna() & original_series.notna()
            default_value = get_default_category_value(column)
            if default_value is not None:
                dataframe.loc[unmapped_mask, column] = default_value
                print(
                    f"[DataPreprocessor] {column} デフォルト値({default_value})で置換: "
                    f"{int(unmapped_mask.sum())}件"
                )
        else:
            print(f"[DataPreprocessor] {column} 全ての値が正常にマッピングされました")

        dataframe[column] = dataframe[column].astype(pd.Int64Dtype())

        return dataframe

    def _apply_zone_specific_mapping(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """エリア別のカテゴリカル変数マッピングを適用"""
        import json
        import os
        from datetime import datetime

        # ログファイルの準備
        log_dir = f"logs/preprocessing/{self.store_name}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, f"zone_mapping_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # エリア別のマッピングログ
        zone_mapping_log = {
            "store_name": self.store_name,
            "timestamp": datetime.now().isoformat(),
            "zones": {},
        }

        # エリア別に処理
        if "A/C Name" in dataframe.columns:
            # A/C Nameからエリアを推定（命名規則に基づく）
            dataframe["zone"] = dataframe["A/C Name"].str.extract(
                r"([A-Za-z]+(?:\s+[A-Za-z]+)*)"
            )[0]
            dataframe["zone"] = dataframe["zone"].fillna("Unknown")
        else:
            dataframe["zone"] = "Unknown"

        for zone in dataframe["zone"].unique():
            if zone == "Unknown":
                continue

            print(f"\n[DataPreprocessor] エリア '{zone}' のカテゴリカル変数処理開始")
            zone_data = dataframe[dataframe["zone"] == zone].copy()
            zone_log = {
                "zone_name": zone,
                "total_records": len(zone_data),
                "categorical_mappings": {},
            }

            # 各カテゴリカル変数を処理
            for column in ["A/C ON/OFF", "A/C Mode", "A/C Fan Speed"]:
                if column in zone_data.columns:
                    print(f"[DataPreprocessor] {zone} - {column} 処理中...")

                    # エリア固有の値の分析
                    unique_values = zone_data[column].value_counts()
                    print(
                        f"[DataPreprocessor] {zone} - {column} ユニーク値: {unique_values.to_dict()}"
                    )

                    original_zone_series = zone_data[column]
                    mapped_series, applied_mapping, unmapped_values = (
                        map_category_series(original_zone_series, column)
                    )
                    zone_data[column] = mapped_series

                    zone_log_entry = {
                        "original_values": unique_values.to_dict(),
                        "mapping": applied_mapping,
                        "mapped_count": len(applied_mapping),
                        "unmapped_count": int(sum(unmapped_values.values())),
                    }
                    if unmapped_values:
                        zone_log_entry["unmapped_values"] = unmapped_values
                    zone_log["categorical_mappings"][column] = zone_log_entry

                    if unmapped_values:
                        print(
                            f"[DataPreprocessor] {zone} - {column} マッピングされなかった値: {unmapped_values}"
                        )
                        unmapped_mask = (
                            mapped_series.isna() & original_zone_series.notna()
                        )
                        default_value = get_default_category_value(column)
                        if default_value is not None:
                            zone_data.loc[unmapped_mask, column] = default_value
                            zone_log["categorical_mappings"][column][
                                "default_value"
                            ] = default_value
                            print(
                                f"[DataPreprocessor] {zone} - {column} デフォルト値({default_value})で置換: {int(unmapped_mask.sum())}件"
                            )

                    zone_data[column] = zone_data[column].astype(pd.Int64Dtype())

                    # 元のデータフレームを更新
                    dataframe.loc[dataframe["zone"] == zone, column] = zone_data[column]

            zone_mapping_log["zones"][zone] = zone_log

        # ログファイルに保存
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(zone_mapping_log, f, ensure_ascii=False, indent=2)

        print(f"\n[DataPreprocessor] エリア別マッピングログ保存: {log_file}")

        # エリア列を削除（元のデータ構造を維持）
        dataframe = dataframe.drop("zone", axis=1)

        return dataframe

    def preprocess_ac(
        self,
        dataframe: Optional[pd.DataFrame],
        standard_deviation_multiplier: float = 5.0,
        category_mapping: Optional[dict] = None,
        zone_specific_mapping: bool = True,
    ) -> Optional[pd.DataFrame]:
        if dataframe is None or dataframe.empty:
            return None
        dataframe, datetime_column = self._unify_datetime(dataframe)
        if dataframe is None:
            return None
        dataframe = self._rm_dup(dataframe, datetime_column)
        dataframe = self._rm_outliers(
            dataframe,
            ["Indoor Temp.", "Outdoor Temp."],
            standard_deviation_multiplier,
        )
        for column in ["Indoor Temp.", "Outdoor Temp."]:
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].interpolate("linear")

        # カテゴリ変換は前処理段階では行わない
        # 後段階（集約時）でエリア別マッピングを実行
        print("[DataPreprocessor] カテゴリカル変数のマッピングは後段階で実行します")

        # 列名を統一（Datetime, Date）
        dataframe["Datetime"] = dataframe[datetime_column]
        dataframe["Date"] = dataframe[datetime_column].dt.date

        # 列の並び順を調整（Datetime, Dateを最初に配置）
        cols = list(dataframe.columns)
        if "Datetime" in cols:
            cols.remove("Datetime")
        if "Date" in cols:
            cols.remove("Date")

        # Datetime, Dateを最初に配置
        dataframe = dataframe[["Datetime", "Date"] + cols]

        return dataframe

    def preprocess_pm(
        self,
        dataframe: Optional[pd.DataFrame],
        standard_deviation_multiplier: float = 5.0,
    ) -> Optional[pd.DataFrame]:
        if dataframe is None or dataframe.empty:
            return None
        dataframe, datetime_column = self._unify_datetime(dataframe)
        if dataframe is None:
            return None
        dataframe = self._rm_dup(dataframe, datetime_column)
        # Total_kWh列を先に作成してから外れ値除去
        phase_columns = [col for col in dataframe.columns if col.startswith("Phase")]
        if phase_columns:
            dataframe["Total_kWh"] = dataframe[phase_columns].sum(axis=1)
        else:
            dataframe["Total_kWh"] = dataframe["Phase A"]

        dataframe = self._rm_outliers(
            dataframe, ["Total_kWh"], standard_deviation_multiplier
        )
        dataframe["Phase A"] = dataframe["Phase A"].fillna(0)
        dataframe["Total_kWh"] = dataframe["Total_kWh"].fillna(0)

        # 列名を統一（Datetime, Date）
        dataframe["Datetime"] = dataframe[datetime_column]
        dataframe["Date"] = dataframe[datetime_column].dt.date

        # 列の並び順を調整（Datetime, Dateを最初に配置）
        cols = list(dataframe.columns)
        if "Datetime" in cols:
            cols.remove("Datetime")
        if "Date" in cols:
            cols.remove("Date")

        # Datetime, Dateを最初に配置
        dataframe = dataframe[["Datetime", "Date"] + cols]

        return dataframe

    def _fetch_historical_weather(
        self,
        ac_data: Optional[pd.DataFrame],
        pm_data: Optional[pd.DataFrame],
        weather_api_key: str,
        coordinates: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch historical weather data based on the date range of AC/PM data"""
        if not weather_api_key or not coordinates:
            print(
                "[DataPreprocessor] No weather API key or coordinates provided, skipping historical weather fetch"
            )
            return None

        # Determine date range from the data
        date_ranges = []
        if ac_data is not None and not ac_data.empty:
            ac_dates = pd.to_datetime(ac_data["Datetime"])
            date_ranges.append((ac_dates.min(), ac_dates.max()))
        if pm_data is not None and not pm_data.empty:
            pm_dates = pd.to_datetime(pm_data["Datetime"])
            date_ranges.append((pm_dates.min(), pm_dates.max()))

        if not date_ranges:
            print("[DataPreprocessor] No data available to determine date range")
            return None

        # Get the overall date range
        min_date = min(dt[0] for dt in date_ranges)
        max_date = max(dt[1] for dt in date_ranges)

        start_date = min_date.strftime("%Y-%m-%d")
        end_date = max_date.strftime("%Y-%m-%d")

        print(
            f"[DataPreprocessor] Fetching historical weather data from {start_date} to {end_date}"
        )

        try:
            weather_df = VisualCrossingWeatherAPIDataFetcher(
                coordinates=coordinates,
                start_date=start_date,
                end_date=end_date,
                unit="metric",
                api_key=weather_api_key,
                batch_size_months=1,  # Process 1 month at a time
                delay_between_requests=1.0,  # 1 second delay between requests
            ).fetch()

            if weather_df is not None:
                print(
                    f"[DataPreprocessor] Historical weather data fetched successfully. Shape: {weather_df.shape}"
                )
                return weather_df
            else:
                print("[DataPreprocessor] Failed to fetch historical weather data")
                return None

        except Exception as e:
            print(f"[DataPreprocessor] Error fetching historical weather data: {e}")
            return None

    def save(
        self,
        ac_control_data: Optional[pd.DataFrame],
        power_meter_data: Optional[pd.DataFrame],
        weather_data: Optional[pd.DataFrame],
    ):
        if ac_control_data is not None:
            # Sort by Datetime column in descending order (latest first)
            if "Datetime" in ac_control_data.columns:
                ac_sorted = ac_control_data.sort_values("Datetime", ascending=False)
            else:
                print(
                    f"⚠️ No 'Datetime' column found in AC control data. Available columns: {list(ac_control_data.columns)}"
                )
                ac_sorted = ac_control_data
            ac_sorted.to_csv(
                os.path.join(
                    self.output_dir, f"ac_control_processed_{self.store_name}.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )
        if power_meter_data is not None:
            # Sort by Datetime column in descending order (latest first)
            if "Datetime" in power_meter_data.columns:
                pm_sorted = power_meter_data.sort_values("Datetime", ascending=False)
            else:
                print(
                    f"⚠️ No 'Datetime' column found in power meter data. Available columns: {list(power_meter_data.columns)}"
                )
                pm_sorted = power_meter_data
            pm_sorted.to_csv(
                os.path.join(
                    self.output_dir, f"power_meter_processed_{self.store_name}.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )
        if weather_data is not None:
            # Find the datetime column (could be "Datetime" or "datetime")
            datetime_col = None
            for col in ["Datetime", "datetime"]:
                if col in weather_data.columns:
                    datetime_col = col
                    break

            if datetime_col:
                # Sort by datetime column in descending order (latest first)
                weather_sorted = weather_data.sort_values(datetime_col, ascending=False)
            else:
                # If no datetime column found, use original data
                print(
                    f"⚠️ No datetime column found in weather data. Available columns: {list(weather_data.columns)}"
                )
                weather_sorted = weather_data

            weather_sorted.to_csv(
                os.path.join(
                    self.output_dir, f"weather_processed_{self.store_name}.csv"
                ),
                index=False,
                encoding="utf-8-sig",
            )

        # 自動Excel出力処理 (AC_setvalue_range_analysis_*.xlsx)
        try:
            print("[DataPreprocessor] 自動Excel出力を開始します...")
            export_temp_range_stats(
                ac_df=ac_control_data,
                store_name=self.store_name,
                output_dir=self.output_dir,
            )
        except Exception as e:
            print(f"[DataPreprocessor] Failed to export monthly range Excel: {e}")
