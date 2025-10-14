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
