# run_aircontrol_vec_test.py
from __future__ import annotations

import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
from tianshou.env import SubprocVectorEnv

from deep_reinforcement_learning.environment.worker_space import (
    AirControlInit,
    make_env_factory_aircontrol,
)


# ===== ユーティリティ =====
def split_df(df: pd.DataFrame, term: pd.Timestamp):
    _df = df.copy()
    hist = _df[_df.index < term]
    valid = _df[_df.index >= term]
    return hist, valid


def main():
    # macOS / Windows では spawn を明示
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # ===== ベースデータ読み込み =====
    base_df = pd.read_csv("data/base/hourly_filled.csv")
    base_df["Datetime_hour"] = pd.to_datetime(base_df["Datetime_hour"])
    base_df = base_df.set_index("Datetime_hour", drop=True)

    # ===== 期間設定 =====
    t0 = pd.Timestamp("2025-09-10 07:00:00")
    tend = pd.Timestamp("2025-09-11 07:00:00")
    historical_df, validate_df = split_df(base_df, datetime(2025, 9, 10, 7))

    # ===== 環境初期化パラメータ =====
    init_args = AirControlInit(
        model_or_path="models/xgb_weight.joblib",  # もしくはロード済みモデル
        base_df=historical_df,
        start_term=t0,
        end_term=tend,
        weather_forecast=validate_df,
        reward_params=None,  # None → RewardParams() が内部で適用
    )

    # ===== 並列環境作成 =====
    n_envs = 2
    env_fns = [
        make_env_factory_aircontrol(init_args, seed=42 + i) for i in range(n_envs)
    ]
    envs = SubprocVectorEnv(env_fns)

    # ===== リセット =====
    obs, info = envs.reset()
    print(f"[reset] obs.shape={getattr(obs, 'shape', None)}  n_envs={envs.env_num}")

    # ===== ランダム行動で数ステップ進める =====
    n_loops = 25
    for t in range(n_loops):
        # 各環境に 1 アクションずつ（action_space は単一 env の空間）
        samples = [envs.action_space[i].sample() for i in range(envs.env_num)]
        try:
            # MultiDiscrete 等でベクトルアクションの場合はこちらが綺麗に揃う
            acts = np.stack(samples)
        except Exception:
            # スカラーアクション（Discrete 等）の場合はこちらでOK
            acts = np.asarray(samples, dtype=np.int64)

        obs, rew, terminated, truncated, info = envs.step(acts)
        done = np.logical_or(terminated, truncated)

        # info は env ごとの dict のリスト
        kwh = [d.get("kwh", np.nan) for d in info]
        disc = [d.get("discomfort", np.nan) for d in info]
        cterm = [d.get("cost_term", np.nan) for d in info]
        mterm = [d.get("comfort_term", np.nan) for d in info]

        print(f"[t={t}] reward={rew}  done={done}")
        print(f"        kwh={kwh}")
        print(f"        discomfort={disc}")
        print(f"        cost_term={cterm}")
        print(f"        comfort_term={mterm}")

        # 終了した環境だけ個別リセット
        ids_done = np.nonzero(done)[0].tolist()
        if ids_done:
            envs.reset(ids_done)

    envs.close()
    print("OK: vector env test finished.")


if __name__ == "__main__":
    main()
