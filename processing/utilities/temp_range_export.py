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

import os
import logging
import pandas as pd
from typing import Optional


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
        logging.warning("[export_temp_range_stats] Empty DataFrame received, skipping export.")
        return

    print("[export_temp_range_stats] 📤 Excelファイルを自動出力中...")

    # 出力先ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"AC_setvalue_range_analysis_{store_name}.xlsx"
    output_path = os.path.join(output_dir, output_filename)

    df = ac_df.copy()

    # =============================
    # STEP1: データ前処理
    # =============================

    # A/C ONデータのみ抽出
    if "A/C ON/OFF" in df.columns:
        df = df[df["A/C ON/OFF"] == "ON"].copy()

    # month列の生成
    if "month" not in df.columns:
        dt_col = next((c for c in df.columns if "datetime" in c.lower()), None)
        if dt_col is None:
            raise ValueError("[export_temp_range_stats] datetime列が見つかりません（month生成に必要）")
        df["month"] = pd.to_datetime(df[dt_col]).dt.month

    # FanSpeed列の補完
    if "A/C Fan Speed" in df.columns:
        df["A/C Fan Speed"] = df["A/C Fan Speed"].fillna("Unknown")

    # =============================
    # STEP2: 月別テーブル生成
    # =============================

    months_jp = [f"{i}月" for i in range(1, 13)]
    months_num = list(range(1, 13))
    ac_names = df["A/C Name"].dropna().unique().tolist()

    def _mk_monthly_table(value_col: str, agg: str = "mean") -> pd.DataFrame:
        """月別平均・標準偏差テーブル生成"""
        out = pd.DataFrame(index=months_jp)
        for ac in ac_names:
            s = getattr(df.loc[df["A/C Name"] == ac].groupby("month")[value_col], agg)()
            col = [s.get(m, pd.NA) for m in months_num]
            out[ac] = col
        out.insert(0, "Unnamed: 0", months_jp)
        return out.reset_index(drop=True)

    def _mk_sample_count_table() -> pd.DataFrame:
        """室内機別のサンプル数テーブル生成"""
        out = pd.DataFrame(index=months_jp)
        for ac in ac_names:
            s = df.loc[df["A/C Name"] == ac].groupby("month")["A/C Name"].count()
            col = [s.get(m, 0) for m in months_num]
            out[ac] = col
        out.insert(0, "Unnamed: 0", months_jp)
        return out.reset_index(drop=True)

    def _mk_fanspeed_freq_table() -> pd.DataFrame:
        """FanSpeed頻度テーブル生成"""
        if "A/C Fan Speed" not in df.columns:
            return pd.DataFrame(columns=["Unnamed: 0", "Unnamed: 1", "frequency"] + ac_names)

        fan_speeds = df["A/C Fan Speed"].dropna().unique().tolist()
        rows = []
        for m, m_label in zip(months_num, months_jp):
            for fs in fan_speeds:
                row = {
                    "Unnamed: 0": m_label,
                    "Unnamed: 1": fs,
                    "frequency": int(((df["month"] == m) & (df["A/C Fan Speed"] == fs)).sum()),
                }
                for ac in ac_names:
                    cnt = int(
                        (
                            (df["month"] == m)
                            & (df["A/C Name"] == ac)
                            & (df["A/C Fan Speed"] == fs)
                        ).sum()
                    )
                    row[ac] = cnt
                rows.append(row)
        return pd.DataFrame(rows)

    # =============================
    # STEP3: テーブル作成
    # =============================

    indoortemp_mean = _mk_monthly_table("Indoor Temp.", agg="mean")
    settemp_mean = _mk_monthly_table("A/C Set Temperature", agg="mean")
    indoortemp_std = _mk_monthly_table("Indoor Temp.", agg="std")
    settemp_std = _mk_monthly_table("A/C Set Temperature", agg="std")
    sample_counts = _mk_sample_count_table()
    fanspeed_freq = _mk_fanspeed_freq_table()

    # =============================
    # STEP4: Excel出力
    # =============================

    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            indoortemp_mean.to_excel(writer, sheet_name="Indoortemp平均", index=False)
            settemp_mean.to_excel(writer, sheet_name="設定温度_平均値", index=False)
            indoortemp_std.to_excel(writer, sheet_name="Indoortemp標準偏差", index=False)
            settemp_std.to_excel(writer, sheet_name="設定温度_標準偏差", index=False)
            sample_counts.to_excel(writer, sheet_name="室内機別_サンプル数", index=False)
            fanspeed_freq.to_excel(writer, sheet_name="FanSpeed頻度", index=False)

        print(f"[export_temp_range_stats] ✅ Excel出力完了: {output_path}")

    except Exception as e:
        logging.error(f"[export_temp_range_stats] Excel出力エラー: {e}")
