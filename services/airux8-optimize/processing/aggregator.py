import os
from typing import Dict, List, Optional
import unicodedata
import numpy as np
import pandas as pd

from config.utils import get_weather_historical_path
from processing.utilities.category_mapping_loader import (
    get_default_category_value,
    map_category_series,
)
from service.storage import get_storage_client


# =============================
# STEP1: 集約（制御エリア単位テーブル）
# =============================
class AreaAggregator:
    """制御エリア単位に、空調・電力・天候を1時間単位で統合"""

    def __init__(self, master_info: dict):
        self.m = master_info
        # Normalize Area2 unit mapping 
        self.AREA2_UNIT_MAPPING = {
            area: [self._normalize_ac_name(x) for x in units]
            for area, units in self.AREA2_UNIT_MAPPING_RAW.items()
        }

    @staticmethod
    def _normalize_ac_name(s):
        """Normalizing the ac names and creating the mapping for Area2"""
        if pd.isna(s):
            return s
        s = str(s)
        s = unicodedata.normalize("NFKC", s)
        s = s.replace(" ", "").replace("\u3000", "")
        s = s.replace("–", "-").replace("—", "-").replace("−", "-").replace("ー", "-")
        return s
    
    AREA2_UNIT_MAPPING_RAW = {
        "Area2_1": ["D-8北2", "D-6北1", "D-7南2", "D-5南1"],
        "Area2_2": ["D-4北2", "D-2北1"],
    }
    
    @staticmethod
    def _most_frequent(s: pd.Series):
        return s.mode().iloc[0] if not s.mode().empty else np.nan

    def build(
        self,
        ac: Optional[pd.DataFrame],
        pm: Optional[pd.DataFrame],
        weather: Optional[pd.DataFrame],
        freq: str = "1H",
        apply_zone_mapping: bool = True,
    ) -> pd.DataFrame:
        if self.m is None or "zones" not in self.m:
            raise ValueError("マスタに zones がありません")
        zones = self.m["zones"].copy()

        # Extracting Area 2 of the other zones/rooms
        # splitting Area 2 into Area2_1 and Area2_2 as independent zones
        if "Area 2" in zones:
            base_area2 = zones.pop("Area 2")

            zones["Area2_1"] = base_area2
            zones["Area2_2"] = base_area2

        # 天候（共通）
        weather = weather.copy() if weather is not None else pd.DataFrame()
        if not weather.empty:
            # 天気データの列名を統一（datetime -> Datetime）
            if "datetime" in weather.columns:
                weather["Datetime"] = pd.to_datetime(weather["datetime"]).dt.floor(
                    freq.replace("H", "h")
                )
            elif "Datetime" in weather.columns:
                weather["Datetime"] = pd.to_datetime(weather["Datetime"]).dt.floor(freq)
            else:
                print(
                    f"⚠️ 天気データにDatetime列が見つかりません。利用可能な列: {list(weather.columns)}"
                )
                return pd.DataFrame()
            wcols = [
                c
                for c in [
                    "Outdoor Temp.",
                    "Outdoor Humidity",
                    "Solar Radiation",
                    "temperature C",
                    "humidity",
                ]
                if c in weather.columns
            ]
            weather = (
                weather[["Datetime"] + wcols]
                .groupby("Datetime")
                .agg("mean")
                .reset_index()
            )
            # 列名統一
            if (
                "temperature C" in weather.columns
                and "Outdoor Temp." not in weather.columns
            ):
                weather.rename(columns={"temperature C": "Outdoor Temp."}, inplace=True)
            if (
                "humidity" in weather.columns
                and "Outdoor Humidity" not in weather.columns
            ):
                weather.rename(columns={"humidity": "Outdoor Humidity"}, inplace=True)

        # 制御エリアごとにテーブル構築
        area_rows = []
        for zone_name, zinfo in zones.items():
            # 室内機一覧
            indoor_units: List[str] = []
            # 室外機: {id: {load_share: x}}
            outdoor_units: Dict[str, dict] = zinfo.get("outdoor_units", {})
            for _, ou in outdoor_units.items():
                indoor_units.extend(ou.get("indoor_units", []))
            indoor_units = list(dict.fromkeys(indoor_units))  # unique & keep order

            # 空調（室内機）: 1時間ごと 最頻値/平均
            if ac is not None and not ac.empty and indoor_units:
                # converting into strings and normalizing 
                ac["A/C Name"] = ac["A/C Name"].astype(str).apply(self._normalize_ac_name)
                indoor_units = [self._normalize_ac_name(x) for x in indoor_units]
                
                ac_sub = ac[ac["A/C Name"].isin(indoor_units)].copy()
                
                # setting zone only if AC belongs to this sub area
                if zone_name in ("Area2_1", "Area2_2"):
                    allowed_units = self.AREA2_UNIT_MAPPING.get(zone_name, [])

                    ac_sub = ac_sub[ac_sub["A/C Name"].isin(allowed_units)].copy()
                    ac_sub["zone"] = zone_name

                if zone_name not in ("Area2_1", "Area2_2"):
                    ac_sub["zone"] = zone_name
                    
                if not ac_sub.empty:
                    # エリア別カテゴリカル変数マッピングを適用
                    if apply_zone_mapping:
                        ac_sub = self._apply_zone_categorical_mapping(ac_sub, zone_name)

                    ac_sub["Datetime"] = pd.to_datetime(ac_sub["Datetime"]).dt.floor(
                        freq.replace("H", "h")
                    )
                    # Enforce ON/OFF coding to 0=OFF, 1=ON
                    if "A/C ON/OFF" in ac_sub.columns:
                        ac_sub["A/C ON/OFF"] = pd.to_numeric(
                            ac_sub["A/C ON/OFF"], errors="coerce"
                        ).fillna(0)
                        ac_sub["A/C ON/OFF"] = (ac_sub["A/C ON/OFF"] > 0).astype(
                            "int64"
                        )
                    # After categorical mapping, A/C ON/OFF is already numeric (0=OFF, 1=ON)
                    # So we can use it directly for counting units ON

                    # Debug: check how many samples per unit per hour exist (inflates raw sums)
                    dup_check = (
                        ac_sub.groupby(["Datetime", "A/C Name"])
                        .size()
                        .reset_index(name="samples_per_hour")
                    )
                    if not dup_check.empty:
                        print(
                            f"[AreaAggregator] Zone {zone_name}: Max samples per unit per hour before normalization = {dup_check['samples_per_hour'].max()}"
                        )

                    # 1) Normalize to per-unit-per-hour first so each unit contributes at most 0/1 per hour
                    # Every unit gets normalized per hour and per zone
                    group_cols = ["Datetime", "A/C Name", "zone"]
                    
                    unit_hour = (
                        ac_sub.groupby([group_cols])
                        .agg(
                            {
                                "A/C Set Temperature": AreaAggregator._most_frequent,
                                "Indoor Temp.": "mean",
                                # After categorical mapping ON/OFF is 0/1; use max to collapse within-hour samples
                                "A/C ON/OFF": "max",
                                "A/C Mode": AreaAggregator._most_frequent,
                                "A/C Fan Speed": AreaAggregator._most_frequent,
                            }
                        )
                        .reset_index()
                    )

                    # 2) Aggregate to zone per hour
                    # Because multiple zones are possible, one row per hour per zone
                    # instead of just one row per hour
                    # the mode is handled sparately 
                    group_cols = ["Datetime", "zone"]
                    g = (
                        unit_hour.groupby(group_cols)
                        .agg(
                            {
                                "A/C Set Temperature": AreaAggregator._most_frequent,
                                "Indoor Temp.": "mean",
                                # Sum across units now equals number of units ON (bounded by physical units)
                                "A/C ON/OFF": "sum",
                                "A/C Fan Speed": AreaAggregator._most_frequent,
                            }
                        )
                        .reset_index()
                    )
                    
                    # set zone column
                    if zone_name not in ("Area2_1", "Area2_2"):
                        g["zone"] = zone_name


                    # add A/C Mode as metadata
                    if "A/C Mode" in unit_hour.columns:
                        g = g.merge(
                            unit_hour[group_cols + ["A/C Mode"]]
                            .drop_duplicates(subset=group_cols),
                            on=group_cols,
                            how="left"
                        )

                    # Debug: verify counts are reasonable after normalization
                    if not g.empty:
                        max_units_on = (
                            int(g["A/C ON/OFF"].max())
                            if "A/C ON/OFF" in g.columns
                            else 0
                        )
                        print(
                            f"[AreaAggregator] Zone {zone_name}: Max units ON per hour after normalization = {max_units_on} (total indoor units = {len(indoor_units)})"
                        )

                    # Create A/C Status column based on ON/OFF count and Mode
                    # Status mapping: OFF=0, COOL=1, HEAT=2, FAN=3
                    g["A/C Status"] = 0

                    if "A/C ON/OFF" in g.columns and "A/C Mode" in g.columns:
                        # if not numeric already, for safety 
                        g["A/C Mode"] = pd.to_numeric(g["A/C Mode"], errors="coerce")

                        # assigning status when at least one unit is ON
                        on_mask = g["A/C ON/OFF"] > 0
                        
                        g.loc[on_mask & (g["A/C Mode"] == 1), "A/C Status"] = 1
                        g.loc[on_mask & (g["A/C Mode"] == 2), "A/C Status"] = 2
                        # fan as fallback
                        g.loc[
                            on_mask & ~g["A/C Mode"].isin([1, 2]),
                            "A/C Status"
                        ] = 3
                        g["Operation Status"] = g["A/C Status"]
                    
                    else:
                        # even without ac data, zone should not disappear
                        if ac is not None and not ac.empty and "Datetime" in ac.columns:
                            base_dt = (
                                pd.to_datetime(ac["Datetime"])
                                .dt.floor(freq.replace("H", "h"))
                                .unique()
                            )
                        # if power data exists, still show timestamps
                        elif pm is not None and not pm.empty and "Datetime" in pm.columns:
                            base_dt = (
                                pd.to_datetime(pm["Datetime"])
                                .dt.floor(freq.replace("H", "h"))
                                .unique()
                            )
                        else:
                            base_dt = []
                        
                        # create g
                        g = pd.DataFrame({"Datetime": base_dt})
                        
                        # set default values
                        g["A/C Set Temperature"] = np.nan
                        g["Indoor Temp."] = np.nan
                        g["A/C ON/OFF"] = 0
                        g["A/C Mode"] = 0
                        g["A/C Fan Speed"] = 0
                        g["A/C Status"] = 0
                        # mirror oper. status and ac status
                        g["Operation Status"] = g["A/C Status"]

            # 電力（室外機×負荷率の合計）
            p_list = []
            if pm is not None and not pm.empty and outdoor_units:
                print(
                    f"[AreaAggregator] Zone {zone_name}: Processing {len(outdoor_units)} outdoor units"
                )

                for ou_id, ou in outdoor_units.items():
                    share = float(ou.get("load_share", 1.0))

                    # Try exact match first
                    sub = pm[pm["Mesh ID"] == ou_id].copy()

                    # If no exact match, try extracting the base number (e.g., "49-1" -> 49)
                    if sub.empty and "-" in str(ou_id):
                        base_id = int(str(ou_id).split("-")[0])
                        sub = pm[pm["Mesh ID"] == base_id].copy()

                    if sub.empty:
                        continue

                    print(
                        f"[AreaAggregator] Found {len(sub)} records for Mesh ID: {ou_id}"
                    )

                    # Total_kWh列の存在確認
                    if "Total_kWh" not in sub.columns:
                        print(
                            f"⚠️ Total_kWh列が存在しません。利用可能な列: {list(sub.columns)}"
                        )
                        if "Phase A" in sub.columns:
                            print(f"  Phase A列を使用します")
                            sub["Total_kWh"] = sub["Phase A"]
                        else:
                            print(f"  ❌ 電力データが見つかりません")
                            continue

                    sub["Datetime"] = pd.to_datetime(sub["Datetime"]).dt.floor(
                        freq.replace("H", "h")
                    )
                    sub = sub.groupby("Datetime")["Total_kWh"].sum().reset_index()
                    sub["adjusted_power"] = sub["Total_kWh"] * share

                    print(
                        f"  Total_kWh統計: 平均={sub['Total_kWh'].mean():.2f}, 最大={sub['Total_kWh'].max():.2f}"
                    )
                    print(
                        f"  adjusted_power統計: 平均={sub['adjusted_power'].mean():.2f}, 最大={sub['adjusted_power'].max():.2f}"
                    )

                    p_list.append(sub[["Datetime", "adjusted_power"]])
            if p_list:
                p = (
                    pd.concat(p_list, ignore_index=True)
                    .groupby("Datetime")["adjusted_power"]
                    .sum()
                    .reset_index()
                )

                print(f"[AreaAggregator] 電力データ統合結果:")
                print(f"  統合前レコード数: {len(p_list)}")
                print(f"  統合後レコード数: {len(p)}")
                print(f"  adjusted_power欠損値: {p['adjusted_power'].isnull().sum()}件")
                print(
                    f"  adjusted_power統計: 平均={p['adjusted_power'].mean():.2f}, 最大={p['adjusted_power'].max():.2f}"
                )
            else:
                p = pd.DataFrame(columns=["Datetime", "adjusted_power"])
                print(f"[AreaAggregator] 電力データがありません")

            # マージ
            # left join to prevent artificial timestamps 
            df = g.merge(p, on="Datetime", how="left")

            print(f"[AreaAggregator] マージ後:")
            print(f"  dfレコード数: {len(df)}")
            print(f"  adjusted_power欠損値: {df['adjusted_power'].isnull().sum()}件")
            if not weather.empty:
                df = df.merge(weather, on="Datetime", how="left")

            # adjusted_powerの欠損値分析
            missing_power = df["adjusted_power"].isnull().sum()
            if missing_power > 0:
                print(f"⚠️ adjusted_powerに欠損値が{missing_power}件あります")

                # 欠損値の原因分析
                missing_df = df[df["adjusted_power"].isnull()].copy()
                print(f"  欠損値の詳細分析:")
                print(f"    欠損レコード数: {len(missing_df)}")

                # 時間範囲の確認
                if not missing_df.empty:
                    print(
                        f"    欠損期間: {missing_df['Datetime'].min()} ～ {missing_df['Datetime'].max()}"
                    )

                    # 電力データが存在するかチェック
                    has_power_data = (
                        "adjusted_power" in df.columns
                        and not df["adjusted_power"].isnull().all()
                    )
                    if has_power_data:
                        non_missing_count = df["adjusted_power"].notnull().sum()
                        print(f"    電力データ存在: {non_missing_count}件")
                        print(
                            f"    電力データ欠損率: {missing_power / len(df) * 100:.1f}%"
                        )
                    else:
                        print(f"    ❌ 電力データが全く存在しません")

                    # 空調データとの比較
                    if "Indoor Temp." in df.columns:
                        temp_missing = df["Indoor Temp."].isnull().sum()
                        print(f"    室温データ欠損: {temp_missing}件")
                        if temp_missing == 0:
                            print(f"    ⚠️ 室温データは存在するが電力データが欠損")
                        else:
                            print(f"    ⚠️ 室温データも欠損している可能性")

                # 電力データの統合前後の状況確認
                if p_list:
                    print(f"  電力データ統合前の状況:")
                    print(f"    統合前レコード数: {len(p_list)}")
                    print(f"    統合後レコード数: {len(p)}")
                    print(f"    統合後欠損値: {p['adjusted_power'].isnull().sum()}件")
                else:
                    print(f"    ❌ 電力データが統合されていません（p_listが空）")

                # マージの状況確認
                print(f"  マージ状況:")
                print(f"    空調データレコード数: {len(g)}")
                print(f"    電力データレコード数: {len(p)}")
                print(f"    マージ後レコード数: {len(df)}")

                # 時間範囲の重複確認
                if not g.empty and not p.empty:
                    g_time_range = (g["Datetime"].min(), g["Datetime"].max())
                    p_time_range = (p["Datetime"].min(), p["Datetime"].max())
                    print(
                        f"    空調データ時間範囲: {g_time_range[0]} ～ {g_time_range[1]}"
                    )
                    print(
                        f"    電力データ時間範囲: {p_time_range[0]} ～ {p_time_range[1]}"
                    )

                    # 時間範囲の重複チェック
                    overlap_start = max(g_time_range[0], p_time_range[0])
                    overlap_end = min(g_time_range[1], p_time_range[1])
                    if overlap_start <= overlap_end:
                        print(
                            f"    ✅ 時間範囲に重複があります: {overlap_start} ～ {overlap_end}"
                        )
                    else:
                        print(f"    ❌ 時間範囲に重複がありません")
            else:
                print(f"✅ adjusted_powerに欠損値はありません")
                print(
                    f"  adjusted_power統計: 平均={df['adjusted_power'].mean():.2f}, 最大={df['adjusted_power'].max():.2f}"
                )

            df["zone"] = zone_name
            df.sort_values(
                "Datetime", ascending=False, inplace=True
            )  # Sort by latest first (newest to oldest)
            area_rows.append(df)

        area_df = (
            pd.concat(area_rows, ignore_index=True) if area_rows else pd.DataFrame()
        )
        # ラグ（前時刻室温）
        if not area_df.empty:
            # Sort the final concatenated dataframe by Datetime in descending order (newest to oldest)
            area_df.sort_values("Datetime", ascending=False, inplace=True)
            # 時間特徴量の付与（曜日・時刻・月・週末）
            area_df["Datetime"] = pd.to_datetime(area_df["Datetime"])  # 安全化
            area_df["Date"] = area_df["Datetime"].dt.date
            area_df["DayOfWeek"] = area_df["Datetime"].dt.dayofweek.astype(int)
            area_df["Hour"] = area_df["Datetime"].dt.hour.astype(int)
            area_df["Month"] = area_df["Datetime"].dt.month.astype(int)
            area_df["IsWeekend"] = area_df["DayOfWeek"].isin([5, 6]).astype(int)
            # 祝日フラグ（jpholidayが利用可能なら使用、なければ0）
            try:
                import jpholiday  # type: ignore

                area_df["IsHoliday"] = (
                    area_df["Datetime"]
                    .dt.date.map(lambda d: 1 if jpholiday.is_holiday(d) else 0)
                    .astype(int)
                )
            except Exception:
                area_df["IsHoliday"] = 0
            area_df["Indoor Temp. Lag1"] = (
                area_df.sort_values(
                    ["zone", "Datetime"], ascending=[True, True]
                )  # Sort zones ascending, datetime ascending for lag calculation
                .groupby("zone")["Indoor Temp."]
                .shift(1)
            )
            area_df["Indoor Temp. Lag1"] = area_df["Indoor Temp. Lag1"].fillna(
                area_df["Indoor Temp."]
            )

            # 温度を小数点第1位に丸める
            if "Indoor Temp." in area_df.columns:
                area_df["Indoor Temp."] = area_df["Indoor Temp."].round(1)
            if "Indoor Temp. Lag1" in area_df.columns:
                area_df["Indoor Temp. Lag1"] = area_df["Indoor Temp. Lag1"].round(1)
            if "Outdoor Temp." in area_df.columns:
                area_df["Outdoor Temp."] = area_df["Outdoor Temp."].round(1)

            # 列の並び順を調整（Datetime, Dateを最初に配置）
            cols = list(area_df.columns)
            if "Datetime" in cols:
                cols.remove("Datetime")
            if "Date" in cols:
                cols.remove("Date")

            # Datetime, Dateを最初に配置
            area_df = area_df[["Datetime", "Date"] + cols]

            # Final sort: zone 昇順 → Datetime 降順（各ゾーン内で新しい順）
            # explicit zone order, so area 2_1 and 2_2 are in the right place
            ZONE_ORDER = [
                "Area 1",
                "Area2_1",
                "Area2_2",
                "Area 3",
                "Area 4",
                "Meeting Room",
                "Break Room",
            ]

            area_df["zone"] = pd.Categorical(
                area_df["zone"],
                categories=ZONE_ORDER,
                ordered=True
            )
            area_df.sort_values(
                ["zone", "Datetime"], ascending=[True, False], inplace=True
            )

        return area_df

    def _apply_zone_categorical_mapping(
        self, dataframe: pd.DataFrame, zone_name: str
    ) -> pd.DataFrame:
        """エリア別のカテゴリカル変数マッピングを適用"""
        import json
        import os
        from datetime import datetime

        # ログファイルの準備
        log_dir = f"logs/preprocessing/{self.m.get('store_name', 'unknown')}"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, f"zone_mapping_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # エリア別のマッピングログ
        zone_mapping_log = {
            "store_name": self.m.get("store_name", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "zones": {},
        }

        print(f"\n[AreaAggregator] エリア '{zone_name}' のカテゴリカル変数処理開始")
        zone_log = {
            "zone_name": zone_name,
            "total_records": len(dataframe),
            "categorical_mappings": {},
        }

        # 各カテゴリカル変数を処理
        for column in ["A/C ON/OFF", "A/C Mode", "A/C Fan Speed"]:
            if column in dataframe.columns:
                print(f"[AreaAggregator] {zone_name} - {column} 処理中...")

                # エリア固有の値の分析
                unique_values = dataframe[column].value_counts()
                print(
                    f"[AreaAggregator] {zone_name} - {column} ユニーク値: {unique_values.to_dict()}"
                )

                original_series = dataframe[column]
                mapped_series, applied_mapping, unmapped_values = map_category_series(
                    original_series, column
                )
                dataframe[column] = mapped_series

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
                        f"[AreaAggregator] {zone_name} - {column} マッピングされなかった値: {unmapped_values}"
                    )
                    unmapped_mask = mapped_series.isna() & original_series.notna()
                    default_value = get_default_category_value(column)
                    if default_value is not None:
                        dataframe.loc[unmapped_mask, column] = default_value
                        zone_log_entry["default_value"] = default_value
                        print(
                            f"[AreaAggregator] {zone_name} - {column} デフォルト値({default_value})で置換: {int(unmapped_mask.sum())}件"
                        )

                # TODO : need to revisit later
                # Ensure all NA values are handled before converting to integer
                if dataframe[column].isna().any():
                    default_value = get_default_category_value(column)
                    if default_value is not None:
                        dataframe[column] = dataframe[column].fillna(default_value)
                        print(
                            f"[AreaAggregator] {zone_name} - {column} 残りのNA値をデフォルト値({default_value})で置換"
                        )
                    else:
                        # If no default value, use 0 as fallback
                        dataframe[column] = dataframe[column].fillna(0)
                        print(
                            f"[AreaAggregator] {zone_name} - {column} 残りのNA値を0で置換"
                        )

                dataframe[column] = dataframe[column].astype(pd.Int64Dtype())

        zone_mapping_log["zones"][zone_name] = zone_log

        # ログファイルに保存（安全な書き込み）
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(zone_mapping_log, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ ログファイル保存エラー: {e}")
            # バックアップファイルに保存
            backup_file = log_file.replace(".json", "_backup.json")
            try:
                with open(backup_file, "w", encoding="utf-8") as f:
                    json.dump(zone_mapping_log, f, ensure_ascii=False, indent=2)
                print(f"📋 バックアップファイルに保存: {backup_file}")
            except Exception as backup_e:
                print(f"❌ バックアップ保存も失敗: {backup_e}")

        print(f"\n[AreaAggregator] エリア別マッピングログ保存: {log_file}")

        return dataframe


def aggregation_runner(
    store_name: str,
    store_master_file: dict,
    freq: str = "1H",
):
    """
    集約のみを実行
    Weather data is automatically determined from preprocessed data.

    Args:
        store_name: 店舗名
        store_master_file: マスターデータ
        freq: 時間粒度

    Returns:
        pd.DataFrame: 集約されたデータ
    """
    if store_master_file is None:
        print("[Aggregate] マスタ未読込")
        return None

    print("[Aggregate] 集約のみ実行開始...")

    # Get coordinates from store_master_file
    coordinates = store_master_file.get("store_info", {}).get("coordinates")
    if coordinates is None:
        print(f"[Aggregate] ERROR: No coordinates found in master data")
        return None
    else:
        print(f"[Aggregate] Using coordinates from master data: {coordinates}")

    # 処理済みデータの読み込み（storage 経由）
    storage = get_storage_client()
    plan_dir = f"04_PlanningData/{store_name}"
    try:
        ac_processed_data = storage.read_csv(
            f"02_PreprocessedData/{store_name}/ac_control_processed_{store_name}.csv"
        )
    except Exception:
        ac_processed_data = None
    try:
        pm_processed_data = storage.read_csv(
            f"02_PreprocessedData/{store_name}/power_meter_processed_{store_name}.csv"
        )
    except Exception:
        pm_processed_data = None
    try:
        weather_historical_path = get_weather_historical_path(store_name)
        historical_weather_data = storage.read_csv(weather_historical_path)
    except Exception:
        historical_weather_data = None

    if ac_processed_data is None or pm_processed_data is None:
        print("[Aggregate] 処理済みデータが見つかりません")
        return None

    # Determine date range from preprocessed data (not from optimization parameters)
    # Use AC data datetime as primary, fallback to power meter data
    if not ac_processed_data.empty and "Datetime" in ac_processed_data.columns:
        ac_processed_data["Datetime"] = pd.to_datetime(ac_processed_data["Datetime"])
        data_start_date = ac_processed_data["Datetime"].min()
        data_end_date = ac_processed_data["Datetime"].max()
    elif not pm_processed_data.empty and "Datetime" in pm_processed_data.columns:
        pm_processed_data["Datetime"] = pd.to_datetime(pm_processed_data["Datetime"])
        data_start_date = pm_processed_data["Datetime"].min()
        data_end_date = pm_processed_data["Datetime"].max()
    else:
        print("[Aggregate] ERROR: No datetime data found in preprocessed files")
        return None

    print(
        f"[Aggregate] Data date range: {data_start_date.date()} to {data_end_date.date()}"
    )

    # If historical weather data exists, use it and ignore any API calls
    if historical_weather_data is not None and not historical_weather_data.empty:
        print("[Aggregate] Using existing historical weather data from CSV")
        combined_weather_df = historical_weather_data
    else:
        # No historical weather exists - this is an error since preprocessing should have created it
        print(
            "[Aggregate] ERROR: Historical weather data not found. "
            "Please run the preprocessor module first to generate weather data."
        )
        return None

    # 集約の実行
    # Use master data from constructor
    if store_master_file is None:
        print("[Aggregate] ERROR: Master data not available for aggregator")
        return None

    # Extract zones data for aggregator
    aggregator_data = {
        "store_name": store_master_file.get("store_info", {}).get("name", store_name),
        "zones": store_master_file.get("zones", {}),
        "master_original": store_master_file.get("sheets", {}).get("original"),
    }
    aggregator = AreaAggregator(aggregator_data)
    area_df = aggregator.build(
        ac_processed_data, pm_processed_data, combined_weather_df, freq=freq
    )

    # データの保存
    if area_df is not None:
        storage.write_csv(
            area_df,
            f"02_PreprocessedData/{store_name}/features_processed_{store_name}.csv",
        )
        print(
            f"[Aggregate] 集約データを保存: 02_PreprocessedData/{store_name}/features_processed_{store_name}.csv"
        )

    print("[Aggregate] 集約完了")
    return area_df
