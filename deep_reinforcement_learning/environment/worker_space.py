# deep_reinforcement_learning/environment/worker_space.py
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import pandas as pd

# 依存：あなたの既存クラス/関数
from deep_reinforcement_learning.environment.control_env import (
    AirControlEnv,
    RewardParams,
)
from deep_reinforcement_learning.environment.prediction.model import load_residual_model


# ============ 監視用: 観測のランニング統計（任意） ============
class _ObsRMS:
    def __init__(self, shape):
        if isinstance(shape, int):
            n = shape
        else:
            shape = tuple(shape)
            n = int(np.prod(shape))
        self.mean = np.zeros(n, dtype=np.float64)
        self.var = np.ones(n, dtype=np.float64)
        self.count = 0.0

    def update(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        # もし (1, obs_dim) なら 1D に潰す
        if x.ndim >= 2:
            # 先頭がバッチで一個だけなら squeeze
            if x.shape[0] == 1:
                x = np.squeeze(x, axis=0)
            else:
                # 万一バッチが複数なら平均して1サンプルに畳む（観測正規化の安定化）
                x = x.reshape(x.shape[0], -1).mean(axis=0)
        x = x.reshape(self.mean.shape)  # (obs_dim,)

        self.count += 1.0
        d = x - self.mean
        self.mean += d / self.count
        d2 = x - self.mean
        # Welford 互換の逐次分散更新（分散は不偏にしたければ最後に補正も可）
        self.var += (d * d2 - self.var) / max(self.count - 1.0, 1.0)


# ============ Env ラッパ（挙動は不変、統計だけ追加） ============
class AirControlEnvIsolated(AirControlEnv):
    """
    - 既存の AirControlEnv をそのまま使い、SubprocVectorEnv に入れても安全なように
      ・モデルのロードをthunk側で完結
      ・観測のランニング統計を任意で保持
    - 元クラスの挙動（reset/step の戻り値など）は変更しません
    """

    def __init__(self, *args, enable_obs_rms: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._obs_rms = (
            _ObsRMS(self.observation_space.shape) if enable_obs_rms else None
        )
        self._workdir = getattr(self, "_workdir", None) or str(Path.cwd())

    @property
    def obs_rms(self):
        return self._obs_rms

    def get_obs_rms(self):
        """SubprocVectorEnv.call_env_method で集計回収する用"""
        return (
            None
            if self._obs_rms is None
            else dict(
                mean=self._obs_rms.mean.copy(),
                var=self._obs_rms.var.copy(),
                count=float(self._obs_rms.count),
            )
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = super().reset(seed=seed, options=options)
        if self._obs_rms is not None:
            self._obs_rms.update(obs)
        return obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = super().step(action)
        if self._obs_rms is not None:
            self._obs_rms.update(obs)
        return obs, rew, terminated, truncated, info


# ============ SubprocVectorEnv 用ファクトリ ============


@dataclass
class AirControlInit:
    """thunkに渡す初期化素材（プロセス間でpickle可能な形に）"""

    # モデル：Path(=joblib) か、ロード済みモデル（ロード済みは deepcopy で分岐）
    model_or_path: str | Path | object
    base_df: pd.DataFrame
    start_term: pd.Timestamp
    end_term: pd.Timestamp
    weather_forecast: pd.DataFrame
    reward_params: RewardParams | None = None


def _ensure_model(model_or_path):
    """プロセス内でモデルインスタンスを用意（Pathならロード、objなら深いコピー）"""
    if isinstance(model_or_path, (str, Path)):
        return load_residual_model(model_or_path)
    # 既にロード済みならコピー（XGB は基本pickle可だが、分離しておくのが安全）
    try:
        return copy.deepcopy(model_or_path)
    except Exception:
        # deepcopy 失敗時はそのまま返す（共有を許容）
        return model_or_path


def make_env_factory_aircontrol(
    init: AirControlInit,
    *,
    env_cls: type[gym.Env] = AirControlEnvIsolated,
    seed: Optional[int] = None,
    enable_obs_rms: bool = True,
) -> Callable[[], gym.Env]:
    """
    SubprocVectorEnv に渡す `thunk` を返す。各プロセスごとに:
      - モデルロード/複製
      - AirControlEnv 派生クラスの生成
      - 必要なら乱数シード設定
    """
    # 引数は pickle されるので、必要なものは dataclass にまとめておく
    init_copy = copy.copy(init)

    def _thunk():
        # 1) モデルをこのプロセス空間で用意
        model = _ensure_model(init_copy.model_or_path)

        # 2) Env を生成（元クラスの挙動は変更しない）
        env = env_cls(
            model=model,
            base_df=init_copy.base_df,
            start_term=init_copy.start_term,
            end_term=init_copy.end_term,
            weather_forecast=init_copy.weather_forecast,
            reward_params=init_copy.reward_params or RewardParams(),
            enable_obs_rms=enable_obs_rms,
        )
        # 3) 任意のシード設定
        if seed is not None:
            try:
                env.reset(seed=int(seed))
            except Exception:
                pass

        return env

    return _thunk
