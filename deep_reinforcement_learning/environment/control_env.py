import copy
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from tianshou.data import Batch as TBatch

from deep_reinforcement_learning.agent.ppo_agent import actions_to_frame
from deep_reinforcement_learning.const import (
    set_fan_range,
    set_mode_range,
    set_on_off_range,
    set_temp_range,
    weather_cols,
)
from deep_reinforcement_learning.environment.prediction.model import (
    predict_full_period_with_residual_model,
)
from deep_reinforcement_learning.environment.prediction.transform_input_data import (
    InputDataBuilderDP,
    pick_cols,
)


@dataclass
class RewardParams:
    alpha_cost: float = 1.0  # エネルギー（kWh）重み
    beta_comfort: float = 4.0  # 快適性（室温-設定の超過分）重み
    deadband_c: float = 0.5  # 不感帯 [℃]（固定値）


class AirControlEnv(gym.Env):
    def __init__(
        self,
        model,
        base_df: pd.DataFrame,
        start_term: pd.Timestamp,
        end_term: pd.Timestamp,
        weather_forecast: pd.DataFrame,
        *,
        reward_params: RewardParams = RewardParams(),
    ):
        """注意: start_term = base_dfの最後の時刻 + 1 h であること。"""
        self.base_df = base_df
        self.start_term = start_term
        self.current_time = None
        self.end_term = end_term
        self.weather_forecast = weather_forecast
        self.reward_params = reward_params
        self.kwh_ref, self.disc_ref = self._estimate_reward_scales(self.base_df)
        if not np.isfinite(self.kwh_ref) or self.kwh_ref <= 0:
            self.kwh_ref = 1.0
        if not np.isfinite(self.disc_ref) or self.disc_ref <= 0:
            self.disc_ref = 1.0
        # 予測器の前処理
        self.model = model
        self.builder = InputDataBuilderDP(
            days=7,
            lags_hours=(1,),
            include_weather_raw=True,
            include_original_controls=True,
        )
        self.builder.fit(base_df)
        self.builder.begin_online(base_df)

        # 列グループ
        self.room_temp_cols = pick_cols(base_df, prefix="Indoor Temp")
        self.set_temp_cols = pick_cols(base_df, prefix="A/C Set Temperature")
        self.set_mode_cols = pick_cols(base_df, prefix="A/C Mode")
        self.set_fan_cols = pick_cols(base_df, prefix="A/C Fan Speed")
        self.set_onoff_cols = pick_cols(base_df, prefix="A/C ON/OFF")
        self.kwh_cols = pick_cols(base_df, prefix="total_kwh__")
        self.weather_cols = list(weather_cols)  # ← 定数をenvに保持

        # 観測の列順
        self.current_state = None
        self._refresh_obs_columns(include_time_features=False)

        # デバイス数・次元
        self.n_devices = len(self.set_temp_cols)
        self.obs_dim = len(self.obs_cols_order)

        # 前回操作（平滑化は入れないが、ロギング等で使うなら保持）
        self.prev_controls = None

        """ Tianshowでの環境の初期設定を行う。 """
        # 状態の空間を定義
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        # 行動の空間を定義
        # 行動は (温度, モード, 風量, ON/OFF) × n_devices の MultiDiscrete
        self.action_space = spaces.MultiDiscrete(
            np.array(
                [
                    len(set_temp_range),
                    len(set_mode_range),
                    len(set_fan_range),
                    len(set_on_off_range),
                ]
                * self.n_devices,
                dtype=np.int64,
            )
        )

    # --------- 内部ユーティリティ ---------
    def _row_at_t0_or_ffill(self, df, cols, t0):
        cols = [c for c in cols if c in df.columns]
        if t0 in df.index:
            return df.loc[[t0], cols]
        return df.loc[:t0, cols].tail(1).reindex(columns=cols)

    def _refresh_obs_columns(self, *, include_time_features: bool):
        # obs は「制御・室内温度・天気」のみ
        body = [
            *self.set_temp_cols,
            *self.set_mode_cols,
            *self.set_fan_cols,
            *self.set_onoff_cols,
            *self.room_temp_cols,
            *self.weather_cols,
        ]
        self.obs_cols_order = body
        self.time_feature_names = (
            ["time__hour", "time__weekday", "time__is_weekend"]
            if include_time_features
            else []
        )
        self.obs_dim = (
            len(self.time_feature_names) + len(body)
            if include_time_features
            else len(body)
        )

    def _nearest_settemp_index(self, val: float) -> int | None:
        if pd.isna(val):
            return None
        arr = set_temp_range
        i = int(np.searchsorted(arr, float(val), side="left"))
        if i == 0:
            return 0
        if i == len(arr):
            return len(arr) - 1
        return i if abs(arr[i] - val) < abs(val - arr[i - 1]) else i - 1

    def _vectorize_obs(self, last_row: pd.Series) -> np.ndarray:
        """obs_cols_order にある列を順に取り、足りない室温は indoor_temp__* を代替参照する。"""
        vals = []
        for c in self.obs_cols_order:
            # まずはそのまま
            if c in last_row.index:
                v = last_row[c]
            else:
                # 「Indoor Temp.__X」→「indoor_temp__X」をフォールバック
                v = np.nan
                if c.startswith("Indoor Temp.__"):
                    alt = c.replace("Indoor Temp.__", "indoor_temp__")
                    if alt in last_row.index:
                        v = last_row[alt]

            # 数値化 & 欠損→0
            try:
                v = float(pd.to_numeric(v))
            except Exception:
                v = np.nan
            if pd.isna(v):
                v = 0.0
            vals.append(v)

        return np.asarray(vals, dtype=np.float32)

    def _extract_obs_info(self, base_df):
        """
        - obs : 現在の空調の制御情報 & 室内温度 & 天気情報
        - info: 現在の設定温度インデックスなど
        """
        df = self.base_df if base_df is None else base_df
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("base_df.index は DatetimeIndex 必須です。")
        df = df.sort_index()

        last_row: pd.Series = df.iloc[-1]
        t_curr: pd.Timestamp = last_row.name

        # 観測ベクトル
        obs_vec = self._vectorize_obs(last_row)  # (obs_dim,)
        assert obs_vec.shape == (self.obs_dim,), (obs_vec.shape, self.obs_dim)

        idx_list = [
            self._nearest_settemp_index(last_row[c]) if c in last_row.index else None
            for c in self.set_temp_cols
        ]
        cur_idx = np.array(
            [-1 if v is None else int(v) for v in idx_list], dtype=np.int64
        )[None, :]
        info = {"current_temp_index": cur_idx, "time": t_curr}
        return obs_vec.astype(np.float32, copy=False), info

    def get_batch(self, base_df: pd.DataFrame | None = None) -> TBatch:
        obs, info = self._extract_obs_info(base_df)
        return TBatch(obs=obs, info=TBatch(**info))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.current_time = self.start_term
        super().reset(seed=seed)
        self.dp_builder = copy.deepcopy(self.builder)
        # 直前の操作（ベースの最終行から拾っておく）
        last = self.base_df.sort_index().iloc[-1]
        ctrl_cols = self.set_temp_cols + self.set_onoff_cols
        self.prev_controls = (
            last[ctrl_cols] if set(ctrl_cols).issubset(last.index) else None
        )
        obs, info = self._extract_obs_info(self.base_df)
        return obs, info

    def _estimate_reward_scales(self, df: pd.DataFrame) -> tuple[float, float]:
        """ベース期間から報酬の基準スケール(kWh, 温度超過)の中央値を推定"""
        df = df.sort_index()
        # kWh 合計の尺度
        kwh_cols = [c for c in df.columns if c.startswith("total_kwh__")]
        if kwh_cols:
            # 旧pandas互換: min_countを使わない
            kwh_sum = df[kwh_cols].sum(axis=1, skipna=True)
            kwh_ref = float(np.nanmedian(kwh_sum.to_numpy()))
        else:
            kwh_ref = 1.0

        # 温度超過の尺度（不感帯差し引きの正部分）
        set_cols = [c for c in df.columns if c.startswith("A/C Set Temperature__")]
        indoor_cols = [c for c in df.columns if c.startswith("Indoor Temp.__")]
        ind_map = {c.split("__", 1)[1]: c for c in indoor_cols if "__" in c}

        diffs = []
        p = self.reward_params
        for sc in set_cols:
            suf = sc.split("__", 1)[1] if "__" in sc else None
            ic = ind_map.get(suf)
            if ic is None:
                continue
            a = pd.to_numeric(df[ic], errors="coerce")
            b = pd.to_numeric(df[sc], errors="coerce")
            over = (a - b).abs() - p.deadband_c
            over = over.where(over > 0, 0.0)
            diffs.append(over)

        if diffs:
            disc_df = pd.concat(diffs, axis=1)
            # 旧pandas互換: min_countを使わない。行平均はNaNを無視、全NaN行はNaNのまま。
            disc_series = disc_df.mean(axis=1, skipna=True)
            disc_ref = float(np.nanmedian(disc_series.to_numpy()))
        else:
            disc_ref = 1.0

        return kwh_ref, disc_ref

    # --------- 報酬計算（最小ハイパラ版） ---------
    def calc_reward(self, pred_row: pd.Series) -> tuple[float, dict]:
        p = self.reward_params

        # kWh 合計
        kwh_cols = [c for c in pred_row.index if c.startswith("total_kwh__")]
        kwh = float(pred_row[kwh_cols].sum()) if kwh_cols else 0.0

        # 快適性（不感帯超過の平均）
        set_cols = [c for c in pred_row.index if c.startswith("A/C Set Temperature__")]
        indoor_cols = [c for c in pred_row.index if c.startswith("indoor_temp__")]
        ind_map = {c.split("__", 1)[1]: c for c in indoor_cols if "__" in c}
        diffs = []
        for sc in set_cols:
            suf = sc.split("__", 1)[1] if "__" in sc else None
            ic = ind_map.get(suf)
            if ic is None:
                continue
            setT = float(pred_row[sc])
            Tin = float(pred_row[ic])
            over = abs(Tin - setT) - p.deadband_c
            diffs.append(max(over, 0.0))
        discomfort = float(np.mean(diffs)) if diffs else 0.0

        # === 自己スケーリング + tanh バウンド ===
        kwh_hat = kwh / max(self.kwh_ref, 1e-6)
        disc_hat = discomfort / max(self.disc_ref, 1e-6)
        cost_term = p.alpha_cost * np.tanh(kwh_hat)
        comfort_term = p.beta_comfort * np.tanh(disc_hat)

        reward = -(cost_term + comfort_term)
        details = dict(
            reward=reward,
            kwh=kwh,
            discomfort=discomfort,
            kwh_hat=kwh_hat,
            disc_hat=disc_hat,
            kwh_ref=self.kwh_ref,
            disc_ref=self.disc_ref,
            cost_term=cost_term,
            comfort_term=comfort_term,
        )
        return reward, details

    # --------- 1ステップ進める ---------
    def step(self, action):
        # 入力の用意
        control_df = actions_to_frame(
            act_indices=action,
            current_time=self.current_time,
            set_temp_list=set_temp_range,
            set_mode_list=set_mode_range,
            set_wind_list=set_fan_range,
            set_on_off_list=set_on_off_range,
            n_devices=self.n_devices,
            set_cols=self.set_temp_cols,
            mode_cols=self.set_mode_cols,
            fan_cols=self.set_fan_cols,
            onoff_cols=self.set_onoff_cols,
        )
        weather_df = self._row_at_t0_or_ffill(
            self.weather_forecast, self.weather_cols, self.current_time
        ).copy()

        # 特徴量 → 予測
        X = self.dp_builder.make_input_next(self.current_time, weather_df, control_df)

        res_all = predict_full_period_with_residual_model(
            model=self.model,
            X_full=X,
            model_target_names=list(self.model.y_cols),
            wanted_target_cols=list(self.model.y_cols),
            bl_prefix="bl__",
            add_back_baseline=True,
        )
        pred_df = res_all["y_pred"]  # DataFrame(1, N)
        env_df = pd.concat([control_df, pred_df], axis=1)
        env_df = pd.concat([env_df, weather_df], axis=1)
        pred_row = env_df.iloc[0]  # Series

        # 報酬
        reward, rinfo = self.calc_reward(pred_row)

        # 状態更新（DPに実績として取り込み、時間を+1h）
        self.update_state(pred_df)

        # 次観測の生成（最新 self.base_df は env 側で持っている想定）
        # ここでは dp_builder の内部状態を進めただけなので、obs は pred_df ではなく base_df の末尾から作る
        obs, info = self._extract_obs_info(env_df)

        # 追加情報
        info.update(rinfo)
        info["time"] = self.current_time

        # 終了判定
        terminated = bool(self.current_time >= self.end_term)
        truncated = False

        return obs, reward, terminated, truncated, info

    def update_state(self, pred_df):
        # 予測を“実績”として DP に取り込む（残差ラグ/同時刻BLを前進させる）
        self.dp_builder.accumulate_actuals(pred_df)
        # 前回操作の保存（将来、滑らかさを入れる/ログに使う場合に備え）
        ctrl_cols = self.set_temp_cols + self.set_onoff_cols
        try:
            self.prev_controls = pred_df.iloc[0][ctrl_cols]
        except Exception:
            pass
        # 計画時間を前進
        self.current_time += pd.Timedelta(hours=1)
