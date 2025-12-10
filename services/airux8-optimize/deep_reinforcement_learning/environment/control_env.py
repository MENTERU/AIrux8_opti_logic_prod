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
    """
    unit_temp_range_list : モデルのカラムの順番とConstのunitの順番が対応しているかの確認を忘れずに
    target_temp_list : モデルのカラムの順番とConstのunitの順番が対応しているかの確認を忘れずに
    """

    def __init__(
        self,
        model_temp,
        model_elec,
        base_df: pd.DataFrame,
        start_term: pd.Timestamp,
        end_term: pd.Timestamp,
        weather_forecast: pd.DataFrame,
        *,
        reward_params: RewardParams = RewardParams(),
        unit_temp_range_list: list,  # [(設定温度下限, 設定温度上限) * ユニット台数]
        target_temp_list: list,
    ):
        """注意: start_term = base_dfの最後の時刻 + 1 h であること。"""
        self.base_df = base_df
        self.start_term = start_term
        self.current_time = None
        self.end_term = end_term
        self.weather_forecast = weather_forecast
        self.reward_params = reward_params
        # 予測器の前処理
        self.model_temp = model_temp  # 室内温度を予測するモデル
        self.model_elec = model_elec  # 室外機の消費電力量を予測するモデル
        self.builder = InputDataBuilderDP(
            days=7,
            lags_hours=(1,),
            include_weather_raw=True,
            include_original_controls=True,
        )
        self.builder.fit(base_df)
        self.builder.begin_online(base_df)

        # 制御マスタからの情報を保存
        self.unit_temp_range_list = unit_temp_range_list
        self.target_temp_list = target_temp_list

        # 列グループ
        self.room_temp_cols = pick_cols(base_df, prefix="Indoor Temp")
        self.set_temp_cols = pick_cols(base_df, prefix="A/C Set Temperature")
        self.set_mode_cols = pick_cols(base_df, prefix="A/C Mode")
        self.set_fan_cols = pick_cols(base_df, prefix="A/C Fan Speed")
        self.set_onoff_cols = pick_cols(base_df, prefix="A/C ON/OFF")
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
        df = df.sort_index()
        last_row: pd.Series = df.iloc[-1]

        # 観測は 1D（バッチ次元を付けない）
        obs_vec = self._vectorize_obs(last_row)  # shape: (obs_dim,)

        # current_temp_index も 1D（バッチ次元を付けない）
        idx_list = [
            self._nearest_settemp_index(last_row[c]) if c in last_row.index else None
            for c in self.set_temp_cols
        ]
        cur_idx = np.array(
            [-1 if v is None else int(v) for v in idx_list], dtype=np.int64
        )  # (n_devices,)

        info = {"current_temp_index": cur_idx, "time": last_row.name}
        return obs_vec.astype(np.float32, copy=False), info

    def get_batch(self, base_df: pd.DataFrame | None = None) -> TBatch:
        obs, info = self._extract_obs_info(base_df)
        return TBatch(obs=obs, info=TBatch(**info))

    def _create_unit_control_master_dict(self):
        self.unit_temp_idx_range_list = [
            (self._nearest_settemp_index(low), self._nearest_settemp_index(high))
            for low, high in self.unit_temp_range_list
        ]
        return dict(
            temp_range_idx_list=self.unit_temp_idx_range_list,
            target_temp_list=self.target_temp_list,
        )

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
        havc_master_info = self._create_unit_control_master_dict()
        info.update(havc_master_info)
        return obs, info

    # --------- 報酬計算（最小ハイパラ版） ---------
    def calc_reward(self, pred_row: pd.Series) -> tuple[float, dict]:
        indoor_cols = [c for c in pred_row.index if c.startswith("indoor_temp__")]

        # target_temp_list は [目標温度] * n台 のリスト想定
        # → indoor_cols の並びと対応するようにしておくこと
        diffs = []
        per_unit_diff = {}

        for col, targetT in zip(indoor_cols, self.target_temp_list):
            Tin = float(pred_row[col])
            diff = abs(Tin - float(targetT))
            diffs.append(diff)
            per_unit_diff[col] = diff

        if diffs:
            # 平均絶対誤差
            discomfort = float(np.mean(diffs))
            # 目標温度に近いほど reward が大きく（0に近く）なる
            reward = -discomfort
        else:
            # 室内温度が取れない場合はゼロ報酬
            discomfort = 0.0
            reward = 0.0

        details = dict(
            reward=reward,
            discomfort=discomfort,
            per_unit_diff=per_unit_diff,
            target_temp_list=list(self.target_temp_list),
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
        X_temp = X.loc[:, ~X.columns.str.startswith("lag1h__")]
        indoor_temp_pred = predict_full_period_with_residual_model(
            model=self.model_temp,
            X_full=X_temp,
            model_target_names=list(self.model_temp.y_cols),
            wanted_target_cols=list(self.model_temp.y_cols),
            bl_prefix="bl__",
            add_back_baseline=True,
        )
        indoor_temp_pred_df = indoor_temp_pred["y_pred"]  # DataFrame(1, N)
        elec_consumption_pred = predict_full_period_with_residual_model(
            model=self.model_elec,
            X_full=X,
            model_target_names=list(self.model_elec.y_cols),
            wanted_target_cols=list(self.model_elec.y_cols),
            bl_prefix="bl__",
            add_back_baseline=True,
        )
        elec_consumption_pred_df = elec_consumption_pred["y_pred"]  # DataFrame(1, N)

        pred_df = pd.concat([indoor_temp_pred_df, elec_consumption_pred_df], axis=1)
        env_df = pd.concat([control_df, pred_df], axis=1)
        env_df = pd.concat([env_df, weather_df], axis=1)
        pred_row = env_df.iloc[0]  # Series

        # 報酬
        reward, rinfo = self.calc_reward(pred_row)
        try:
            pred_cols = list(pred_df.columns)
            info_pred = {f"pred__{c}": float(pred_row[c]) for c in pred_cols}
        except Exception:
            info_pred = {}
        # 次観測の生成（最新 self.base_df は env 側で持っている想定）
        # ここでは dp_builder の内部状態を進めただけなので、obs は pred_df ではなく base_df の末尾から作る
        obs, info = self._extract_obs_info(env_df)
        # 追加情報
        info.update(rinfo)
        info.update(
            dict(
                temp_range_idx_list=self.unit_temp_idx_range_list,
                target_temp_list=self.target_temp_list,
            )
        )
        info["time"] = self.current_time
        if info_pred:
            info["predicted"] = info_pred
        # 終了判定
        terminated = bool(self.current_time >= self.end_term)
        truncated = False
        # 状態更新（DPに実績として取り込み、時間を+1h）
        self.update_state(pred_df)
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
