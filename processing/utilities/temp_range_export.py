"""
processing/temp_range_export.py

Generates a monthly temperature and setpoint range analysis Excel file.

Sheets:
- Indoortemp平均
- 設定温度_平均値
- Indoortemp標準偏差
- 設定温度_標準偏差
- 室内機別_サンプル数
- FanSpeed頻度
"""

import logging
import os
import time
from typing import Optional

import pandas as pd


# =============================
# 月別温度レンジ分析エクスポート
# =============================
def export_temp_range_stats(
    ac_df: Optional[pd.DataFrame],
    store_name: str,
    output_dir: str,
) -> None:
    """
    Generate and export the same Excel summary as Kim’s notebook version,
    using the preprocessed AC control data.

    Parameters
    ----------
    ac_df : pd.DataFrame or None
        Cleaned AC control data (after preprocessing).
    store_name : str
        Store identifier (e.g., 'Clea', 'IsetanMitsukoshi').
    output_dir : str
        Directory to save the output Excel file.
    """

    if ac_df is None or ac_df.empty:
        logging.warning(
            "[export_temp_range_stats] Empty DataFrame received, skipping export."
        )
        return

    # Start timing the entire process
    start_time = time.time()
    print(f"[export_temp_range_stats] 📤 Excelファイルを自動出力中...")
    print(
        f"[export_temp_range_stats] ⏱️  データサイズ: {ac_df.shape[0]:,} 行 × {ac_df.shape[1]} 列"
    )

    # 出力先ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"AC_setvalue_range_analysis_{store_name}.xlsx"
    output_path = os.path.join(output_dir, output_filename)

    # OPTIMIZED: Use view instead of full copy to save memory
    df = ac_df

    # =============================
    # STEP1: データ前処理
    # =============================
    step1_start = time.time()

    # A/C ONデータのみ抽出
    if "A/C ON/OFF" in df.columns:
        # OPTIMIZED: Use boolean mask instead of copy operation
        ac_on_mask = df["A/C ON/OFF"] == "ON"
        df = df[ac_on_mask]

    # month列の生成
    if "month" not in df.columns:
        dt_col = next((c for c in df.columns if "datetime" in c.lower()), None)
        if dt_col is None:
            raise ValueError(
                "[export_temp_range_stats] datetime列が見つかりません（month生成に必要）"
            )
        # OPTIMIZED: Avoid unnecessary datetime conversion if already datetime
        if df[dt_col].dtype == "object":
            # Convert from string to datetime, then extract month
            df["month"] = pd.to_datetime(df[dt_col]).dt.month
        else:
            # Already datetime, just extract month (saves conversion time)
            df["month"] = df[dt_col].dt.month

    # FanSpeed列の補完
    if "A/C Fan Speed" in df.columns:
        df["A/C Fan Speed"] = df["A/C Fan Speed"].fillna("Unknown")

    step1_time = time.time() - step1_start
    print(f"[export_temp_range_stats] ⏱️  STEP1 (データ前処理): {step1_time:.2f}秒")

    # =============================
    # STEP2: 月別テーブル生成
    # =============================
    step2_start = time.time()

    months_jp = [f"{i}月" for i in range(1, 13)]
    months_num = list(range(1, 13))
    # OPTIMIZED: Keep as pandas Series for better performance
    ac_names = df["A/C Name"].dropna().unique()
    print(f"[export_temp_range_stats] ⏱️  ACユニット数: {len(ac_names)}個")

    def _mk_monthly_table(value_col: str, agg: str = "mean") -> pd.DataFrame:
        """
        Generate monthly statistics table for temperature data.

        Creates a pivot table with months as rows and AC units as columns,
        calculating specified aggregation (mean/std) for the given value column.

        Args:
            value_col: Column name to aggregate (e.g., 'Indoor Temp.', 'A/C Set Temperature')
            agg: Aggregation function ('mean' or 'std')

        Returns:
            DataFrame with months as rows, AC units as columns, and aggregated values
        """
        # OPTIMIZED: Single groupby operation instead of looping through each AC unit
        grouped = getattr(df.groupby(["A/C Name", "month"])[value_col], agg)()

        # Pivot to get AC names as columns and months as rows
        out = pd.DataFrame(index=months_jp)
        for ac in ac_names:
            if ac in grouped.index.get_level_values(0):
                ac_data = grouped[ac]
                col = [
                    ac_data.get(m, pd.NA) if m in ac_data.index else pd.NA
                    for m in months_num
                ]
            else:
                col = [pd.NA] * len(months_num)
            out[ac] = col

        out.insert(0, "Unnamed: 0", months_jp)
        # Round only temperature-related columns to 1 decimal place
        if "温度" in value_col or "Temp" in value_col:
            # Round only the AC unit columns (skip the "Unnamed: 0" column)
            ac_columns = [col for col in out.columns if col != "Unnamed: 0"]
            # Apply rounding to each AC column individually to ensure it works
            for col in ac_columns:
                # Convert to numeric first, handling any non-numeric values
                out[col] = pd.to_numeric(out[col], errors="coerce").round(1)
        return out.reset_index(drop=True)

    def _mk_sample_count_table() -> pd.DataFrame:
        """
        Generate monthly sample count table for each AC unit.

        Creates a table showing how many data points exist for each AC unit
        in each month, useful for data quality assessment.

        Returns:
            DataFrame with months as rows, AC units as columns, and sample counts
        """
        # OPTIMIZED: Single groupby operation instead of looping through each AC unit
        grouped = df.groupby(["A/C Name", "month"]).size()

        # Pivot to get AC names as columns and months as rows
        out = pd.DataFrame(index=months_jp)
        for ac in ac_names:
            if ac in grouped.index.get_level_values(0):
                ac_data = grouped[ac]
                col = [
                    ac_data.get(m, 0) if m in ac_data.index else 0 for m in months_num
                ]
            else:
                col = [0] * len(months_num)
            out[ac] = col

        out.insert(0, "Unnamed: 0", months_jp)
        # No rounding for sample counts - they should remain as integers
        return out.reset_index(drop=True)

    def _mk_fanspeed_freq_table() -> pd.DataFrame:
        """
        Generate fan speed frequency table for each AC unit by month.

        Creates a detailed table showing how often each AC unit used each fan speed
        setting in each month, with both individual unit counts and total frequencies.

        Returns:
            DataFrame with month/fan_speed combinations as rows, AC units as columns,
            and frequency counts as values
        """
        if "A/C Fan Speed" not in df.columns:
            return pd.DataFrame(
                columns=["Unnamed: 0", "Unnamed: 1", "frequency"] + ac_names.tolist()
            )

        # OPTIMIZED: Use groupby instead of triple nested loop
        fan_speeds = df["A/C Fan Speed"].dropna().unique()

        # Single groupby operation instead of 1,500+ boolean filtering operations
        # Group by month, fan_speed, and AC name, then count occurrences
        grouped = df.groupby(["month", "A/C Fan Speed", "A/C Name"]).size()

        # Calculate total frequency per month/fan_speed combination
        freq_grouped = df.groupby(["month", "A/C Fan Speed"]).size()

        # Build rows efficiently using the pre-grouped data
        rows = []
        for m, m_label in zip(months_num, months_jp):
            for fs in fan_speeds:
                # Get frequency from pre-grouped data
                try:
                    frequency = int(freq_grouped.get((m, fs), 0))
                except (KeyError, TypeError):
                    frequency = 0

                row = {
                    "Unnamed: 0": m_label,
                    "Unnamed: 1": fs,
                    "frequency": frequency,
                }

                # Get AC counts from pre-grouped data
                for ac in ac_names:
                    try:
                        count = int(grouped.get((m, fs, ac), 0))
                    except (KeyError, TypeError):
                        count = 0
                    row[ac] = count
                rows.append(row)

        rows = pd.DataFrame(rows)

        # No rounding for FanSpeed frequency - they should remain as integers
        return rows

    # =============================
    # STEP3: テーブル作成
    # =============================
    step3_start = time.time()

    # Generate all tables
    indoortemp_mean = _mk_monthly_table("Indoor Temp.", agg="mean")
    settemp_mean = _mk_monthly_table("A/C Set Temperature", agg="mean")
    indoortemp_std = _mk_monthly_table("Indoor Temp.", agg="std")
    settemp_std = _mk_monthly_table("A/C Set Temperature", agg="std")
    sample_counts = _mk_sample_count_table()
    fanspeed_freq = _mk_fanspeed_freq_table()

    step3_time = time.time() - step3_start
    print(f"[export_temp_range_stats] ⏱️  STEP3 (テーブル作成): {step3_time:.2f}秒")

    # =============================
    # STEP4: Excel出力
    # =============================
    step4_start = time.time()

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            indoortemp_mean.to_excel(writer, sheet_name="Indoortemp平均", index=False)
            settemp_mean.to_excel(writer, sheet_name="設定温度_平均値", index=False)
            indoortemp_std.to_excel(
                writer, sheet_name="Indoortemp標準偏差", index=False
            )
            settemp_std.to_excel(writer, sheet_name="設定温度_標準偏差", index=False)
            sample_counts.to_excel(
                writer, sheet_name="室内機別_サンプル数", index=False
            )
            fanspeed_freq.to_excel(writer, sheet_name="FanSpeed頻度", index=False)

        step4_time = time.time() - step4_start
        print(f"[export_temp_range_stats] ⏱️  STEP4 (Excel出力): {step4_time:.2f}秒")
        print(f"[export_temp_range_stats] ✅ Excel出力完了: {output_path}")

    except Exception as e:
        logging.error(f"[export_temp_range_stats] Excel出力エラー: {e}")

    # Final timing summary
    total_time = time.time() - start_time
    print(f"[export_temp_range_stats] ⏱️  総実行時間: {total_time:.2f}秒")


def _get_most_frequent_fan_speed(fan_speed_list: list) -> str:
    """
    Find the most frequent fan speed from a list of fan speed candidates.

    Args:
        fan_speed_list: List of fan speed strings (e.g., ["Low,High,Medium", "Low", "High"])

    Returns:
        Most frequent individual fan speed (e.g., "Low")
    """
    # Count frequency of each individual fan speed
    fan_speed_counts = {}

    for fan_speed_string in fan_speed_list:
        if fan_speed_string == "Unknown":
            continue

        # Split comma-separated fan speeds and count each one
        individual_speeds = [speed.strip() for speed in fan_speed_string.split(",")]
        for speed in individual_speeds:
            if speed and speed != "Unknown":
                fan_speed_counts[speed] = fan_speed_counts.get(speed, 0) + 1

    # Return the most frequent fan speed, or "Low" as default
    if fan_speed_counts:
        return max(fan_speed_counts, key=fan_speed_counts.get)
    else:
        return "Low"


def _round_to_half_increment(value):
    """
    Round temperature value to the nearest 0.5°C increment.

    Args:
        value (float): Temperature value to round

    Returns:
        float: Temperature rounded to nearest 0.5°C increment
    """
    if pd.isna(value):
        return value

    # Round to nearest 0.5 increment
    # Multiply by 2, round to nearest integer, then divide by 2
    return round(value * 2) / 2


def update_master_from_analysis(store_name: str, processed_dir: str) -> None:
    """
    Update MASTER_{store_name}_integrated.xlsx
    based on AC_setvalue_range_analysis_{store_name}.xlsx.

    LOGIC:
        1. 各エアコンの月別平均を計算 → mean_settemp, mean_indoor
        2. 各エアコンの標準偏差を計算 → std_settemp, std_indoor
        3. 平均±標準偏差で制御限界を設定:
            上限 = 平均 + 標準偏差
            下限 = 平均 - 標準偏差
        4. エリア別に集計して最終的な制御値を決定

    統計結果（AC_setvalue_range_analysis_◯◯.xlsx）を読み込み、
    MASTER_◯◯.xlsx の関連カラムを自動更新する関数。
    """

    print(
        f"🔄 Updating MASTER file for {store_name} ... / {store_name} のマスタファイルを更新中..."
    )

    # ==============================================================
    # 1. Define file paths / ファイルパスの定義
    # ==============================================================
    analysis_path = os.path.join(
        processed_dir, f"AC_setvalue_range_analysis_{store_name}.xlsx"
    )
    master_path = os.path.join(processed_dir, f"MASTER_{store_name}.xlsx")

    if not os.path.exists(analysis_path):
        print(
            f" Analysis file not found: {analysis_path} / 統計ファイルが見つかりません。"
        )
        return
    if not os.path.exists(master_path):
        print(
            f" MASTER file not found: {master_path} / マスタファイルが見つかりません。"
        )
        return

    # ==============================================================
    # 2. Read analysis Excel sheets / 統計結果Excelを読み込み
    # ==============================================================
    sheets = pd.read_excel(analysis_path, sheet_name=None)
    indoortemp_mean = sheets.get("Indoortemp平均")
    indoortemp_std = sheets.get("Indoortemp標準偏差")
    settemp_mean = sheets.get("設定温度_平均値")
    settemp_std = sheets.get("設定温度_標準偏差")
    fanspeed_freq = sheets.get("FanSpeed頻度")

    # ==============================================================
    # 3. Load MASTER file / マスタファイルを読み込み
    # ==============================================================
    master = pd.read_excel(master_path, sheet_name="制御マスタ")

    # Add missing columns if they don't exist
    target_columns = [
        "目標室内温度下限",
        "目標室内温度上限",
        "設定温度上限",
        "設定温度下限",
        "風量候補",
    ]
    for col in target_columns:
        if col not in master.columns:
            master[col] = pd.NA

    # ==============================================================
    # 4. Create AC unit to area mapping and compute control limits by area
    #    ACユニットからエリアへのマッピングを作成し、エリア別に制御値を算出
    # ==============================================================

    # Load the MASTER sheet to get AC unit to area mapping
    master_mapping = pd.read_excel(master_path, sheet_name="MASTER")
    ac_to_area = dict(zip(master_mapping["環境予測区分"], master_mapping["制御区分"]))

    ac_names = [c for c in settemp_mean.columns if c not in ["Unnamed: 0", "index"]]

    # Use monthly-specific data (not area-averaged)
    # Keep the monthly data structure for monthly-specific calculations
    settemp_monthly = settemp_mean.set_index("Unnamed: 0")
    indoortemp_monthly = indoortemp_mean.set_index("Unnamed: 0")
    settemp_std_monthly = settemp_std.set_index("Unnamed: 0")
    indoortemp_std_monthly = indoortemp_std.set_index("Unnamed: 0")

    # Calculate monthly-specific values for each area
    area_updates = {}
    months_jp = [f"{i}月" for i in range(1, 13)]

    print(
        f"\n🔍 [CALCULATION] Starting monthly-specific calculations for {len(ac_names)} AC units across {len(months_jp)} months"
    )
    print(f"📊 [DATA] AC units: {ac_names[:5]}... (showing first 5)")
    print(f"📅 [DATA] Months: {months_jp}")

    for month in months_jp:
        area_updates[month] = {}
        print(f"\n📅 [MONTH] Processing {month}...")

        for ac in ac_names:
            if ac not in ac_to_area:
                continue  # Skip AC units that don't have area mapping

            area = ac_to_area[ac]
            if pd.isna(area):
                continue  # Skip AC units with NaN area

            # Get monthly-specific values
            monthly_mean_settemp = (
                settemp_monthly.loc[month, ac]
                if month in settemp_monthly.index
                else pd.NA
            )
            monthly_std_settemp = (
                settemp_std_monthly.loc[month, ac]
                if month in settemp_std_monthly.index
                else 0
            )
            monthly_mean_indoor = (
                indoortemp_monthly.loc[month, ac]
                if month in indoortemp_monthly.index
                else pd.NA
            )
            monthly_std_indoor = (
                indoortemp_std_monthly.loc[month, ac]
                if month in indoortemp_std_monthly.index
                else 0
            )

            # Log detailed calculation for first few AC units
            if ac in ac_names[:3]:  # Log first 3 AC units for each month
                print(f"  🔧 [AC] {ac} → {area}:")
                print(
                    f"    📊 Set temp: mean={monthly_mean_settemp:.1f}°C, std={monthly_std_settemp:.1f}°C"
                )
                print(
                    f"    📊 Indoor temp: mean={monthly_mean_indoor:.1f}°C, std={monthly_std_indoor:.1f}°C"
                )

            # Calculate monthly-specific limits
            upper_settemp = (
                _round_to_half_increment(monthly_mean_settemp + monthly_std_settemp)
                if pd.notna(monthly_mean_settemp)
                else pd.NA
            )
            lower_settemp = (
                _round_to_half_increment(monthly_mean_settemp - monthly_std_settemp)
                if pd.notna(monthly_mean_settemp)
                else pd.NA
            )
            upper_indoortemp = (
                _round_to_half_increment(monthly_mean_indoor + monthly_std_indoor)
                if pd.notna(monthly_mean_indoor)
                else pd.NA
            )
            lower_indoortemp = (
                _round_to_half_increment(monthly_mean_indoor - monthly_std_indoor)
                if pd.notna(monthly_mean_indoor)
                else pd.NA
            )

            # Log calculated limits for first few AC units
            if ac in ac_names[:3]:  # Log first 3 AC units for each month
                print(f"    🎯 Calculated limits:")
                print(f"      Set temp: {lower_settemp:.1f}°C to {upper_settemp:.1f}°C")
                print(
                    f"      Indoor temp: {lower_indoortemp:.1f}°C to {upper_indoortemp:.1f}°C"
                )

            # --------------------------------------------------------------
            # Determine most frequent fan speed(s) for this month
            # 風量頻度データから上位カテゴリを抽出
            # --------------------------------------------------------------
            fansspeed_df = fanspeed_freq[fanspeed_freq["Unnamed: 1"].notna()]
            fansspeeds_counts = (
                fansspeed_df.groupby("Unnamed: 1")[ac]
                .sum()
                .sort_values(ascending=False)
                .index.tolist()
            )
            fanspeeds_candidates = (
                ",".join(fansspeeds_counts[:3])
                if len(fansspeeds_counts) > 0
                else "Unknown"
            )

            # Initialize area if not exists
            if area not in area_updates[month]:
                area_updates[month][area] = {
                    "目標室内温度下限": [],
                    "目標室内温度上限": [],
                    "設定温度上限": [],
                    "設定温度下限": [],
                    "風量候補": [],
                }

            # Collect values for this area and month
            area_updates[month][area]["目標室内温度下限"].append(lower_indoortemp)
            area_updates[month][area]["目標室内温度上限"].append(upper_indoortemp)
            area_updates[month][area]["設定温度上限"].append(upper_settemp)
            area_updates[month][area]["設定温度下限"].append(lower_settemp)
            area_updates[month][area]["風量候補"].append(fanspeeds_candidates)

    # Calculate monthly-specific values for each area
    print(f"\n🔄 [AGGREGATION] Aggregating AC units by area for each month...")
    updates = {}
    for month in months_jp:
        updates[month] = {}
        print(f"\n📅 [AGGREGATION] Processing {month}...")

        for area, values in area_updates[month].items():
            # Calculate area averages and round to 0.5 increments
            avg_lower_indoor = _round_to_half_increment(
                pd.Series(values["目標室内温度下限"]).mean()
            )
            avg_upper_indoor = _round_to_half_increment(
                pd.Series(values["目標室内温度上限"]).mean()
            )
            avg_upper_set = _round_to_half_increment(
                pd.Series(values["設定温度上限"]).mean()
            )
            avg_lower_set = _round_to_half_increment(
                pd.Series(values["設定温度下限"]).mean()
            )
            most_frequent_fan = _get_most_frequent_fan_speed(values["風量候補"])

            updates[month][area] = {
                "目標室内温度下限": avg_lower_indoor,
                "目標室内温度上限": avg_upper_indoor,
                "設定温度上限": avg_upper_set,
                "設定温度下限": avg_lower_set,
                "風量候補": most_frequent_fan,
            }

            # Log area aggregation results
            ac_count = len(values["設定温度上限"])
            print(f"  🏢 [AREA] {area}: {ac_count} AC units →")
            print(f"    📊 Set temp: {avg_lower_set:.1f}°C to {avg_upper_set:.1f}°C")
            print(
                f"    📊 Indoor temp: {avg_lower_indoor:.1f}°C to {avg_upper_indoor:.1f}°C"
            )
            print(f"    🌪️ Fan speed: {most_frequent_fan}")

    # ==============================================================
    # 5. Update MASTER rows / マスタの該当行を更新
    # ==============================================================
    print(f"\n💾 [UPDATE] Updating MASTER file with calculated values...")
    updated_rows = 0
    for i, row in master.iterrows():
        area_name = row.get("制御区分")  # Area key column
        month_name = row.get("月")  # Month key column

        if month_name in updates and area_name in updates[month_name]:
            # Log first few updates
            if updated_rows < 5:
                print(f"  📝 [UPDATE] {month_name} - {area_name}:")
                for col, val in updates[month_name][area_name].items():
                    if col in master.columns:
                        print(f"    {col}: {val}")
                        master.at[i, col] = val
            else:
                # Update without logging for remaining rows
                for col, val in updates[month_name][area_name].items():
                    if col in master.columns:
                        master.at[i, col] = val
            updated_rows += 1

    print(f"\n✅ [SUMMARY] Updated {updated_rows} rows with monthly-specific values")

    # ==============================================================
    # 6. Save updated MASTER / 更新後のマスタを保存
    # ==============================================================
    # Read all existing sheets first to preserve them
    all_sheets = pd.read_excel(master_path, sheet_name=None)

    # Update only the 制御マスタ sheet while preserving all other sheets
    all_sheets["制御マスタ"] = master

    # Write all sheets back to the file
    with pd.ExcelWriter(master_path, engine="openpyxl") as writer:
        for sheet_name, sheet_data in all_sheets.items():
            sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)

    print(
        f" MASTER file updated successfully ({updated_rows} rows). / "
        f" MASTERファイルを更新しました（{updated_rows} 行）。"
    )


if __name__ == "__main__":
    store_name = "Clea"
    processed_dir = "/Users/hussain/Menteru-Github/AIrux8_opti_logic/data/01_MasterData"
    update_master_from_analysis(store_name, processed_dir)
