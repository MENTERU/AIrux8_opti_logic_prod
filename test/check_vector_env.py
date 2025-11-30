# check_vector_env.py
import multiprocessing as mp
from datetime import datetime

import pandas as pd
from tianshou.env import SubprocVectorEnv

from deep_reinforcement_learning.environment.prediction.model import load_residual_model
from deep_reinforcement_learning.environment.worker_space import (
    AirControlInit,
    make_env_factory_aircontrol,
)
from input_info.crea_building_information import CreaBuilding


def split_df(df: pd.DataFrame, term: pd.Timestamp):
    """term で過去と将来に分割"""
    _df = df.copy()
    historical_df = _df[_df.index < term]
    validate_df = _df[_df.index >= term]
    return historical_df, validate_df


def extract_set_and_target_from_cols(unit_df: pd.DataFrame, room_names):
    """Trainer と同じ: (設定温度下限, 上限), 目標温度 を部屋ごとに取得"""
    unit_temp_range_list = []
    target_temp_list = []
    for room in room_names:
        set_low = unit_df.loc[unit_df["環境予測区分"] == room, "設定温度下限"].iloc[0]
        set_high = unit_df.loc[unit_df["環境予測区分"] == room, "設定温度上限"].iloc[0]
        target = unit_df.loc[unit_df["環境予測区分"] == room, "目標室内温度"].iloc[0]
        unit_temp_range_list.append((set_low, set_high))
        target_temp_list.append(target)
    return unit_temp_range_list, target_temp_list


def main():
    # ====== 前提設定 ======
    area_name = "Area1"
    base_path = "data/base/hourly_filled.csv"

    # ====== MASTER & 学習済みモデルの読み込み ======
    hvac_master = pd.read_excel("./data/master/MASTER_Clea.xlsx", sheet_name=None)
    master = hvac_master.get("MASTER").set_index(keys="制御区分", drop=True)
    master.index = master.index.str.replace(r"\s+", "", regex=True)
    area_master = master.loc[area_name]

    elec_model_dir = "./models/{area}__kwh.joblib"
    temp_model_dir = "./models/{area}__temp.joblib"

    elec_model = load_residual_model(elec_model_dir.format(area=area_name))
    temp_model = load_residual_model(temp_model_dir.format(area=area_name))

    # ====== ベースデータ読み込み ======
    full_df = pd.read_csv(base_path)
    full_df["Datetime_hour"] = pd.to_datetime(full_df["Datetime_hour"])
    full_df = full_df.set_index("Datetime_hour", drop=True)

    # --- このエリアの室内機・室外機などの列だけ抽出 ---
    use_col = CreaBuilding.get_columns_by_area_units(full_df.columns, area_name)
    base_df_area = full_df[use_col]

    # ====== 室内機の順番をモデルに合わせる ======
    indoor_mode_columns = CreaBuilding.pick_cols(elec_model.x_cols, "A/C Mode")
    room_list = [c.split("__")[-1] for c in indoor_mode_columns]

    unit_temp_range_list, target_temp_list = extract_set_and_target_from_cols(
        area_master, room_list
    )

    ordered_cols = []
    for room in room_list:
        suffix = f"__{room}"
        ordered_cols.extend([c for c in base_df_area.columns if c.endswith(suffix)])
    # 余りの列（電力など）は後ろに付ける
    rest_cols = [c for c in base_df_area.columns if c not in ordered_cols]
    ordered_cols.extend(rest_cols)
    base_df_area = base_df_area.reindex(columns=ordered_cols)

    # ====== 天気データの付与（Trainer._prepare_data と同じイメージ） ======
    # full_df から天気列だけ抜き出して join
    weather_cols = CreaBuilding().weather_colimns  # CreaBuilding 側で定義されている想定
    weather_df = full_df[weather_cols]
    combined_df = base_df_area.join(weather_df, how="left")

    # ====== 期間分割 ======
    t0 = pd.Timestamp("2025-09-10 07:00:00")
    tend = pd.Timestamp("2025-09-11 07:00:00")
    historical_df, validate_df = split_df(combined_df, t0)

    # ====== AirControlInit の構築（新シグネチャに合わせる） ======
    init_args = AirControlInit(
        model_temp=temp_model,
        model_elec=elec_model,
        base_df=historical_df,
        start_term=t0,
        end_term=tend,
        weather_forecast=validate_df,
        reward_params=None,
        unit_temp_range_list=unit_temp_range_list,
        target_temp_list=target_temp_list,
    )

    # ====== ベクター環境の構築 ======
    n_envs = 2
    env_fns = [
        make_env_factory_aircontrol(init_args, seed=42 + i) for i in range(n_envs)
    ]

    # 本番: サブプロセス
    vec_env = SubprocVectorEnv(env_fns)

    # reset して形状確認
    obs, info = vec_env.reset()
    print("obs type:", type(obs))
    print("obs shape:", getattr(obs, "shape", None))
    print("n_envs OK? ->", (getattr(obs, "shape", None) or [None])[0] == n_envs)


if __name__ == "__main__":
    # macOS / Windows では spawn を明示
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 既に設定済みなら無視
        pass

    main()
