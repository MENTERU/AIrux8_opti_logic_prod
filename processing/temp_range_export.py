import os
import pandas as pd

def update_master_from_analysis(store_name: str, processed_dir: str) -> None:
    """
    Update MASTER_{store_name}_integrated.xlsx
    based on AC_setvalue_range_analysis_{store_name}.xlsx.

    統計結果（AC_setvalue_range_analysis_◯◯.xlsx）を読み込み、
    MASTER_◯◯_integrated.xlsx の関連カラムを自動更新する関数。
    """

    print(f"🔄 Updating MASTER file for {store_name} ... / {store_name} のマスタファイルを更新中...")

    # ==============================================================
    # 1. Define file paths / ファイルパスの定義
    # ==============================================================
    analysis_path = os.path.join(processed_dir, f"AC_setvalue_range_analysis_{store_name}.xlsx")
    master_path = os.path.join(processed_dir, f"MASTER_{store_name}_integrated.xlsx")

    if not os.path.exists(analysis_path):
        print(f" Analysis file not found: {analysis_path} / 統計ファイルが見つかりません。")
        return
    if not os.path.exists(master_path):
        print(f" MASTER file not found: {master_path} / マスタファイルが見つかりません。")
        return

    # ==============================================================
    # 2. Read analysis Excel sheets / 統計結果Excelを読み込み
    # ==============================================================
    sheets = pd.read_excel(analysis_path, sheet_name=None)
    indoortemp_mean = sheets.get("Indoortemp平均")
    settemp_mean = sheets.get("設定温度_平均値")
    settemp_std = sheets.get("設定温度_標準偏差")
    fanspeed_freq = sheets.get("FanSpeed頻度")

    # ==============================================================
    # 3. Load MASTER file / マスタファイルを読み込み
    # ==============================================================
    master = pd.read_excel(master_path)

    # ==============================================================
    # 4. Compute new control limits for each A/C unit
    #    各室内機の平均値・標準偏差を基に制御値を算出
    # ==============================================================
    ac_names = [c for c in settemp_mean.columns if c not in ["Unnamed: 0", "index"]]

    mean_set = settemp_mean.set_index("Unnamed: 0").mean(axis=0, numeric_only=True)
    std_set = settemp_std.set_index("Unnamed: 0").mean(axis=0, numeric_only=True)
    mean_indoor = indoortemp_mean.set_index("Unnamed: 0").mean(axis=0, numeric_only=True)

    updates = {}
    for ac in ac_names:
        upper = mean_set.get(ac, pd.NA) + std_set.get(ac, 0)
        lower = mean_set.get(ac, pd.NA) - std_set.get(ac, 0)
        indoor_target = mean_indoor.get(ac, pd.NA)

        # --------------------------------------------------------------
        # Determine most frequent fan speed(s)
        # 風量頻度データから上位カテゴリを抽出
        # --------------------------------------------------------------
        fs_df = fanspeed_freq[fanspeed_freq["Unnamed: 1"].notna()]
        fs_counts = (
            fs_df.groupby("Unnamed: 1")[ac]
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        fan_candidates = ",".join(fs_counts[:3]) if len(fs_counts) > 0 else "Unknown"

        updates[ac] = {
            "目標室内温度": indoor_target,
            "設定温度上限": upper,
            "設定温度下限": lower,
            "風量候補": fan_candidates,
        }

    # ==============================================================
    # 5. Update MASTER rows / マスタの該当行を更新
    # ==============================================================
    updated_rows = 0
    for i, row in master.iterrows():
        ac_name = row.get("環境予測区分")  # A/C Name key column
        if ac_name in updates:
            for col, val in updates[ac_name].items():
                if col in master.columns:
                    master.at[i, col] = val
            updated_rows += 1

    # ==============================================================
    # 6. Save updated MASTER / 更新後のマスタを保存
    # ==============================================================
    master.to_excel(master_path, index=False)
    print(f" MASTER file updated successfully ({updated_rows} rows). / "
          f" MASTERファイルを更新しました（{updated_rows} 行）。")
