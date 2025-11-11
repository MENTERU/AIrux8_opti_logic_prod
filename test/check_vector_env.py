# check_vector_env.py
from datetime import datetime

import pandas as pd
from tianshou.env import SubprocVectorEnv

from deep_reinforcement_learning.environment.worker_space import (
    AirControlInit,
    make_env_factory_aircontrol,
)


def split_df(df, term):
    _df = df.copy()
    historical_df = _df[_df.index < term]
    validate_df = _df[_df.index >= term]
    return historical_df, validate_df


def main():
    base_df = pd.read_csv("data/base/hourly_filled.csv")
    base_df["Datetime_hour"] = pd.to_datetime(base_df["Datetime_hour"])
    base_df = base_df.set_index("Datetime_hour", drop=True)

    t0 = pd.Timestamp("2025-09-10 07:00:00")
    tend = pd.Timestamp("2025-09-11 07:00:00")
    historical_df, validate_df = split_df(base_df, datetime(2025, 9, 10, 7))

    init_args = AirControlInit(
        model_or_path="models/xgb_weight.joblib",
        base_df=historical_df,
        start_term=t0,
        end_term=tend,
        weather_forecast=validate_df,
        reward_params=None,
    )

    n_envs = 2
    env_fns = [
        make_env_factory_aircontrol(init_args, seed=42 + i) for i in range(n_envs)
    ]
    # 本番: サブプロセス
    vec_env = SubprocVectorEnv(env_fns)

    obs, info = vec_env.reset()
    print(type(obs), getattr(obs, "shape", None))
    print("n_envs OK? ->", (getattr(obs, "shape", None) or [None])[0] == n_envs)


if __name__ == "__main__":
    import multiprocessing as mp

    # macOS / Windows では spawn を明示
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # 既に設定済みなら無視
        pass
    main()
