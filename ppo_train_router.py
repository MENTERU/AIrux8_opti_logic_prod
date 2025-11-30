import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Sequence, Union

import numpy as np
import pandas as pd
import torch
from tianshou.data import Batch as TBatch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from deep_reinforcement_learning.agent.ppo_agent import (
    actions_to_frame,
    create_ppo_for_hvac,
)
from deep_reinforcement_learning.const import (
    set_fan_range,
    set_mode_range,
    set_on_off_range,
    set_temp_range,
)
from deep_reinforcement_learning.environment.prediction.model import load_residual_model
from deep_reinforcement_learning.environment.worker_space import (
    AirControlInit,
    make_env_factory_aircontrol,
)
from deep_reinforcement_learning.utils.load import (
    _apply_obs_rms_if_available,
    _pick_obs_rms,
    _pick_policy_path,
)
from deep_reinforcement_learning.utils.record import (
    attach_update_logging_to_tb,
    save_obs_rms_from_vec,
    summarize_hparams_to_tb,
)
from input_info.crea_building_information import CreaBuilding
from service.ppo_train import AircontrolPPOTrainConfig, AircontrolPPOTrainer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_action_columns_from_env(
    env,
) -> tuple[Sequence[str], Sequence[str], Sequence[str], Sequence[str]]:
    """
    train 環境と同じ列名を env から取得。
    必須属性: set_temp_cols, set_mode_cols, set_fan_cols, set_onoff_cols
    """
    try:
        return (
            env.set_temp_cols,
            env.set_mode_cols,
            env.set_fan_cols,
            env.set_onoff_cols,
        )
    except Exception as e:
        raise AttributeError(
            "環境から操作列名を取得できませんでした。env.set_temp_cols / set_mode_cols / "
            "set_fan_cols / set_onoff_cols が公開されている必要があります。"
        ) from e


def extract_set_and_target_from_cols(unit_df, room_names):
    unit_temp_range_list = []
    target_temp_list = []
    for room in room_names:
        set_low = unit_df.loc[unit_df["環境予測区分"] == room, "設定温度下限"].iloc[0]
        set_high = unit_df.loc[unit_df["環境予測区分"] == room, "設定温度上限"].iloc[0]
        target = unit_df.loc[unit_df["環境予測区分"] == room, "目標室内温度"].iloc[0]
        unit_temp_range_list.append((set_low, set_high))
        target_temp_list.append(target)
    return unit_temp_range_list, target_temp_list


class Trainer:
    def __init__(self, train_data_path, train_weather_path, area_name):
        self.train_data_path = train_data_path
        self.train_weather_path = train_weather_path
        self.area_name = area_name
        # modelパスをセット
        self.elec_model_path: str | None = None
        self.temp_model_path: str | None = None

    def setup(self):
        hvac_master = pd.read_excel("./data/master/MASTER_Clea.xlsx", sheet_name=None)
        master = hvac_master.get("MASTER").set_index(keys="制御区分", drop=True)
        master.index = master.index.str.replace(r"\s+", "", regex=True)

        # 学習済みの機械学習モデルから情報を抽出
        elec_model_dir = "./models/{area}__kwh.joblib"
        temp_model_dir = "./models/{area}__temp.joblib"

        # ★ パスを先に組み立てて属性に保存
        self.elec_model_path = elec_model_dir.format(area=self.area_name)
        self.temp_model_path = temp_model_dir.format(area=self.area_name)

        self.elec_model = load_residual_model(self.elec_model_path)
        self.temp_model = load_residual_model(self.temp_model_path)

        area_master = master.loc[self.area_name]
        indoor_mode_columns = CreaBuilding.pick_cols(self.elec_model.x_cols, "A/C Mode")
        self.room_list = [c.split("__")[-1] for c in indoor_mode_columns]
        self.unit_temp_range_list, self.target_temp_list = (
            extract_set_and_target_from_cols(area_master, self.room_list)
        )
        # train dataの読み込み
        self.train_data = pd.read_csv(self.train_data_path)
        self.train_data["Datetime_hour"] = pd.to_datetime(
            self.train_data["Datetime_hour"]
        )
        self.train_data = self.train_data.set_index("Datetime_hour", drop=True)
        # weather dataの読み込み (期間はtraindataと同じである必要があります。)
        self.weather_data = pd.read_csv(self.train_weather_path)
        self.weather_data["Datetime_hour"] = pd.to_datetime(
            self.weather_data["Datetime_hour"]
        )

        self.weather_data = self.weather_data.set_index("Datetime_hour", drop=True)
        use_col = CreaBuilding.get_columns_by_area_units(
            self.train_data.columns, self.area_name
        )
        self.train_data = self.train_data[use_col]
        # 学習済みのモデルに保存されている室内機ユニットの順番と同じ順番に揃える↓
        ordered_cols = []
        for room in self.room_list:
            suffix = f"__{room}"
            ordered_cols.extend(
                [c for c in self.train_data.columns if c.endswith(suffix)]
            )
        rest_cols = [c for c in self.train_data.columns if c not in ordered_cols]
        ordered_cols.extend(rest_cols)
        self.train_data = self.train_data.reindex(columns=ordered_cols)

    def set_area(self, area):
        self.area_name = area

    def load(self, start_term: pd.Timestamp, end_term: pd.Timestamp):
        cfg = AircontrolPPOTrainConfig()
        cfg.log_dir_root = cfg.log_dir_root.format(area=self.area_name)
        self.trainer = AircontrolPPOTrainer(
            cfg=cfg,
            base_df=self.train_data,
            weather_df=self.weather_data,
            start_term=start_term,
            end_term=end_term,
            temp_model_or_path=self.temp_model,
            elec_model_or_path=self.elec_model,
            unit_temp_range_list=self.unit_temp_range_list,
            target_temp_list=self.target_temp_list,
        )

    def train_run(self):
        self.trainer.run(area_name=self.area_name)

    # =========================
    #   重みフォルダの自動推定
    # =========================
    def _infer_weights_root_from_date(self, start_term: pd.Timestamp) -> Path:
        """
        area_name と start_term の日付から、
        ./logs/aircontrol_ppo/{area}/{YYYYMMDD}[_XX] のうち
        一番「後ろ」のフォルダを weights_root として返す。
        """
        cfg = AircontrolPPOTrainConfig()
        base_log_dir = cfg.log_dir_root.format(
            area=self.area_name
        )  # ./logs/aircontrol_ppo/Area1
        date_str = pd.Timestamp(start_term).strftime("%Y%m%d")
        base_log_root = os.path.join(base_log_dir, date_str)

        # まず ./logs/.../AreaX/20250910 を見る
        candidates: list[str] = []
        if os.path.isdir(base_log_root):
            candidates.append(base_log_root)

        # ./logs/.../AreaX/20250910_02, _03, ... もあれば追加
        idx = 2
        while True:
            alt = f"{base_log_root}_{idx:02d}"
            if os.path.isdir(alt):
                candidates.append(alt)
                idx += 1
            else:
                break

        if not candidates:
            raise FileNotFoundError(
                f"学習ログディレクトリが見つかりませんでした: {base_log_root} (area={self.area_name}, date={date_str})"
            )

        # 最後のものを採用（もっとも新しい run と仮定）
        chosen = Path(candidates[-1])
        print(f"[reproduce] use weights_root = {chosen}")
        return chosen

    # =========================
    #   再現 (1 エピソード実行)
    # =========================
    def reproduce(
        self,
        start_term: pd.Timestamp,
        end_term: pd.Timestamp,
        *,
        weights_root: Union[str, Path, None] = None,
        out_dir: Union[str, Path] = "./replay_outputs",
        area_name: str | None = None,
        episode_len: int = 24,
        seed: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        area_name と start_term から PPO の重みを自動で見つけて、
        1 エピソード実行し、旧 reproduction.py と同様の CSV と meta を出力する。
        """
        if area_name is not None:
            self.area_name = area_name

        if self.train_data is None or self.weather_data is None:
            raise RuntimeError("Trainer.setup() を先に呼んでください。")
        if self.temp_model is None or self.elec_model is None:
            raise RuntimeError(
                "学習済み XGBoost モデルがロードされていません。Trainer.setup() を確認してください。"
            )

        # weights_root が指定されていなければ、area + 日付から自動推定
        if weights_root is None:
            weights_root = self._infer_weights_root_from_date(start_term)
        else:
            weights_root = Path(weights_root)

        out_dir = Path(out_dir)
        # Area ごと & 日付ごとにフォルダ分け
        date_str = pd.Timestamp(start_term).strftime("%Y%m%d")
        out_dir = out_dir / self.area_name / date_str
        out_dir.mkdir(parents=True, exist_ok=True)

        # ---- train と同じように weather を結合して historical / validate を作る ----
        base_df = self.train_data.copy()
        weather_cols = CreaBuilding().weather_colimns
        weather_df = self.weather_data.copy()[weather_cols]
        combined_df = base_df.join(weather_df, how="left")

        t0 = start_term
        tend = end_term
        historical_df = combined_df[combined_df.index < t0]
        validate_df = combined_df[combined_df.index >= t0]

        # ---- 環境構築（学習時と同じ AirControlInit 引数） ----
        init_args = AirControlInit(
            model_temp=self.temp_model,
            model_elec=self.elec_model,
            base_df=historical_df,
            start_term=t0,
            end_term=tend,
            weather_forecast=validate_df,
            reward_params=None,
            unit_temp_range_list=self.unit_temp_range_list,
            target_temp_list=self.target_temp_list,
        )
        env = make_env_factory_aircontrol(init_args, seed=seed)()

        # アクション列名（train 環境と同じ）
        set_cols, mode_cols, fan_cols, onoff_cols = _get_action_columns_from_env(env)

        # ---- Policy 構築 & ロード（ハイパラは AircontrolPPOTrainConfig に合わせる） ----
        cfg = AircontrolPPOTrainConfig(area=self.area_name)
        n_devices = cfg.resolve_n_devices()

        policy = create_ppo_for_hvac(
            single_env=env,
            device=DEVICE,
            lr=cfg.lr,
            set_temp_list=set_temp_range,
            set_mode_list=set_mode_range,
            set_wind_list=set_fan_range,
            set_on_off_list=set_on_off_range,
            n_devices=n_devices,
            deterministic_eval=cfg.deterministic_eval,
            discount_factor=cfg.discount_factor,
            gae_lambda=cfg.gae_lambda,
            eps_clip=cfg.eps_clip,
            value_clip=cfg.value_clip,
            vf_coef=cfg.vf_coef,
            ent_coef=cfg.ent_coef,
            max_grad_norm=cfg.max_grad_norm,
            advantage_normalization=cfg.advantage_normalization,
            reward_normalization=cfg.reward_normalization,
        )
        policy.actor.to(DEVICE)
        policy.critic.to(DEVICE)
        policy.eval()

        # weight / obs_rms を読み込み
        pol_path = _pick_policy_path(weights_root)
        state_dict = torch.load(pol_path, map_location=DEVICE)
        policy.load_state_dict(state_dict)
        print(f"[policy] loaded: {pol_path}")

        obs_rms_npz = _pick_obs_rms(weights_root)
        _apply_obs_rms_if_available(env, obs_rms_npz)

        # ---- 1 エピソード実行 ----
        try:
            obs, info = env.reset(seed=seed)
        except TypeError:
            obs, info = env.reset()

        ep_ret: float = 0.0
        rows: List[pd.DataFrame] = []

        for step in range(episode_len):
            obs_tensor = torch.as_tensor(
                obs, dtype=torch.float32, device=DEVICE
            ).unsqueeze(0)

            info_batch = TBatch()
            if isinstance(info, dict):
                # 1) action_mask
                if "action_mask" in info and info["action_mask"] is not None:
                    mask_np = np.asarray(info["action_mask"], dtype=np.float32)
                    if mask_np.ndim == 1:
                        mask_np = mask_np[None, :]
                    info_batch.action_mask = mask_np

                # 2) current_temp_index（±2 マスク用）
                if (
                    "current_temp_index" in info
                    and info["current_temp_index"] is not None
                ):
                    cur_idx = np.asarray(info["current_temp_index"], dtype=np.int64)
                    if cur_idx.ndim == 1:
                        cur_idx = cur_idx[None, :]
                    info_batch.current_temp_index = cur_idx

                # ★ 3) temp_range_idx_list（設定温度の有効範囲）も渡す
                if (
                    "temp_range_idx_list" in info
                    and info["temp_range_idx_list"] is not None
                ):
                    tri = np.asarray(info["temp_range_idx_list"], dtype=np.int64)
                    # 形は [n_devices, 2] / [1, n_devices, 2] どちらでもOK
                    # Actor 側では ndim==2 or 3 両方対応しているのでそのまま突っ込んでよい
                    info_batch.temp_range_idx_list = tri

            with torch.no_grad():
                out = policy(TBatch(obs=obs_tensor, info=info_batch))
                act = out.act.detach().cpu().numpy()[0].astype(int)

            obs, rew, term, trunc, info = env.step(act)
            ep_ret += float(rew)
            done = bool(term or trunc)

            # actions_to_frame で DataFrame 化
            current_time = info.get("time") if isinstance(info, dict) else None
            df_one = actions_to_frame(
                act_indices=act,
                current_time=(
                    current_time
                    if current_time is not None
                    else t0 + pd.Timedelta(hours=step)
                ),
                set_temp_list=set_temp_range,
                set_mode_list=set_mode_range,
                set_wind_list=set_fan_range,
                set_on_off_list=set_on_off_range,
                n_devices=len(set_cols),
                set_cols=set_cols,
                mode_cols=mode_cols,
                fan_cols=fan_cols,
                onoff_cols=onoff_cols,
                index_name="Datetime_hour",
            )

            # env 側で計算された予測値などを付与（あれば）
            pred_map = {}
            if (
                isinstance(info, dict)
                and "predicted" in info
                and isinstance(info["predicted"], dict)
            ):
                pred_map = info["predicted"]

            if pred_map:
                df_pred = pd.DataFrame([pred_map], index=df_one.index)
                df_one = pd.concat([df_one, df_pred], axis=1)

            # step / reward / return を列として付与
            df_one = df_one.assign(
                step=step, reward=float(rew), ret_so_far=float(ep_ret)
            )
            rows.append(df_one)

            if done:
                break

        actions_df = (
            pd.concat(rows, axis=0).set_index("step", append=True).sort_index()
        )  # MultiIndex: (Datetime_hour, step)

        actions_csv = out_dir / "actions_episode.csv"
        actions_df.to_csv(actions_csv, encoding="utf-8")
        print(f"[save] actions -> {actions_csv}")

        rewards_df = (
            actions_df[["reward", "ret_so_far"]]
            .reset_index()
            .set_index("step")[["reward", "ret_so_far"]]
        )
        rewards_csv = out_dir / "rewards_episode.csv"
        rewards_df.to_csv(rewards_csv, encoding="utf-8")
        print(f"[save] rewards -> {rewards_csv}")

        # メタ情報
        meta = {
            "weights_root": str(weights_root),
            "policy_path": str(pol_path),
            "obs_rms": str(obs_rms_npz) if obs_rms_npz else None,
            "base_csv": self.train_data_path,
            "weather_csv": self.train_weather_path,
            "area": self.area_name,
            "temp_model_path": self.temp_model_path,
            "elec_model_path": self.elec_model_path,
            "start_term": str(start_term),
            "end_term": str(end_term),
            "episode_len": int(episode_len),
            "seed": int(seed),
            "n_devices": int(n_devices),
            "action_columns": {
                "set": list(set_cols),
                "mode": list(mode_cols),
                "fan": list(fan_cols),
                "onoff": list(onoff_cols),
            },
        }
        with open(out_dir / "replay_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return actions_df, rewards_df

    def update_train(
        self,
        start_term: pd.Timestamp,
        end_term: pd.Timestamp,
        *,
        weights_root: Union[str, Path, None] = None,
        out_root: Union[str, Path, None] = None,
        area_name: str | None = None,
        epochs: int = 10,
        step_per_epoch: int = 240,
        step_per_collect: int = 120,
        repeat_per_collect: int = 3,
        batch_size: int = 64,
        episode_per_test: int = 1,
        stop_mean_rew: float = 0.0,
        train_envs: int = 2,
        test_envs: int = 1,
        seed: int = 123,
    ):
        """
        既存の PPO 重みから「続き学習（ファインチューニング）」を行う。

        - XGBoost モデル / エリアごとの列情報 / weather 結合 は Trainer.setup() の結果を使用
        - weights_root が未指定なら、self._infer_weights_root_from_date(start_term) で自動推定
        - ログ出力先は:
            out_root/{YYYYMMDD}[_02] ...
          （out_root 未指定なら、学習時と同じ log_dir_root を使う）
        """
        # 前提チェック
        if area_name is not None:
            self.area_name = area_name
        if (
            getattr(self, "train_data", None) is None
            or getattr(self, "weather_data", None) is None
        ):
            raise RuntimeError("Trainer.setup() を先に呼んでください。")
        if (
            getattr(self, "temp_model", None) is None
            or getattr(self, "elec_model", None) is None
        ):
            raise RuntimeError(
                "学習済み XGBoost モデルがロードされていません。Trainer.setup() を確認してください。"
            )

        # ===== weights_root の決定 =====
        if weights_root is None:
            weights_root = self._infer_weights_root_from_date(start_term)
        else:
            weights_root = Path(weights_root)

        # ===== 出力先 out_root の決定 =====
        cfg = AircontrolPPOTrainConfig(area=self.area_name)
        if out_root is None:
            # 学習と同じルート (./logs/aircontrol_ppo/{area}) を再利用
            out_root = cfg.log_dir_root.format(area=self.area_name)
        out_root = Path(out_root)

        date_str = pd.Timestamp(start_term).strftime("%Y%m%d")
        base_log_root = out_root / date_str

        # ★ ここを「既存ならそのまま使う（上書き）」方式に変更 ★
        log_root = base_log_root
        log_root.mkdir(parents=True, exist_ok=True)
        # これで毎回 {out_root}/{YYYYMMDD} に書き込み・上書きされる

        # ===== データ結合（学習時と同じ方式） =====
        t0 = start_term
        tend = end_term

        base_df = self.train_data.copy()
        weather_cols = CreaBuilding().weather_colimns
        weather_df = self.weather_data.copy()[weather_cols]
        combined_df = base_df.join(weather_df, how="left")

        historical_df = combined_df[combined_df.index < t0]
        validate_df = combined_df[combined_df.index >= t0]

        # ===== 環境 init (学習と同じ AirControlInit 引数) =====
        init_args = AirControlInit(
            model_temp=self.temp_model,
            model_elec=self.elec_model,
            base_df=historical_df,
            start_term=t0,
            end_term=tend,
            weather_forecast=validate_df,
            reward_params=None,
            unit_temp_range_list=self.unit_temp_range_list,
            target_temp_list=self.target_temp_list,
        )

        train_fns = [
            make_env_factory_aircontrol(init_args, seed=seed + i)
            for i in range(train_envs)
        ]
        test_fns = [
            make_env_factory_aircontrol(init_args, seed=10000 + i)
            for i in range(test_envs)
        ]
        train_envs_vec = SubprocVectorEnv(train_fns)
        test_envs_vec = SubprocVectorEnv(test_fns)

        # 代表 env
        single_env = make_env_factory_aircontrol(init_args, seed=999)()

        # 既存 obs_rms を代表 env に適用（あれば）
        obs_rms_path = _pick_obs_rms(weights_root)
        _apply_obs_rms_if_available(single_env, obs_rms_path)

        # ===== Policy 構築（ハイパラは AircontrolPPOTrainConfig に合わせる） =====
        cfg_policy = AircontrolPPOTrainConfig(area=self.area_name)
        n_devices = cfg_policy.resolve_n_devices()

        policy = create_ppo_for_hvac(
            single_env=single_env,
            device=DEVICE,
            lr=cfg_policy.lr,
            set_temp_list=set_temp_range,
            set_mode_list=set_mode_range,
            set_wind_list=set_fan_range,
            set_on_off_list=set_on_off_range,
            n_devices=n_devices,
            deterministic_eval=cfg_policy.deterministic_eval,
            discount_factor=cfg_policy.discount_factor,
            gae_lambda=cfg_policy.gae_lambda,
            eps_clip=cfg_policy.eps_clip,
            value_clip=cfg_policy.value_clip,
            vf_coef=cfg_policy.vf_coef,
            ent_coef=cfg_policy.ent_coef,
            max_grad_norm=cfg_policy.max_grad_norm,
            advantage_normalization=cfg_policy.advantage_normalization,
            reward_normalization=cfg_policy.reward_normalization,
        )
        policy.actor.to(DEVICE)
        policy.critic.to(DEVICE)

        # ==== dist を Collector から隠す wrap（update_training.py と同じ）====
        _orig_update = policy.update

        def _update_with_dist_flag(*args, **kwargs):
            setattr(policy, "_return_dist", True)
            try:
                return _orig_update(*args, **kwargs)
            finally:
                setattr(policy, "_return_dist", False)

        policy.update = _update_with_dist_flag  # type: ignore[assignment]

        _orig_forward = policy.forward

        def _forward_conditional(*args, **kwargs):
            ret = _orig_forward(*args, **kwargs)
            # 収集時（_return_dist が False）なら dist を取り除く
            if not getattr(policy, "_return_dist", False):
                if isinstance(ret, TBatch):
                    try:
                        if hasattr(ret, "dist"):
                            delattr(ret, "dist")
                    except Exception:
                        pass
                    try:
                        ret.pop("dist")
                    except Exception:
                        pass
            return ret

        policy.forward = _forward_conditional  # type: ignore[assignment]

        # ===== 既存重みのロード =====
        pol_path = _pick_policy_path(weights_root)
        state_dict = torch.load(pol_path, map_location=DEVICE)
        policy.load_state_dict(state_dict)
        print(f"[update_train] resume weights from: {pol_path}")

        # ===== TensorBoard / Logger =====
        tb_run_dir = log_root / "runs" / "ppo_finetune"
        tb_run_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(tb_run_dir))
        logger = TensorboardLogger(writer)

        trainer_cfg = dict(
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            repeat_per_collect=repeat_per_collect,
            batch_size=batch_size,
            episode_per_test=episode_per_test,
            stop_mean_rew=stop_mean_rew,
            resume_from=str(pol_path),
            resume_obs_rms=str(obs_rms_path or ""),
        )
        summarize_hparams_to_tb(
            writer,
            policy=policy,
            device=DEVICE,
            train_envs=train_envs_vec,
            test_envs=test_envs_vec,
            set_temp_list=set_temp_range,
            set_mode_list=set_mode_range,
            set_wind_list=set_fan_range,
            set_on_off_list=set_on_off_range,
            n_devices=n_devices,
            trainer_cfg=trainer_cfg,
        )

        # 追加の update ロギング
        policy = attach_update_logging_to_tb(policy, writer)

        # ===== Collector =====
        train_collector = Collector(
            policy=policy,
            env=train_envs_vec,
            buffer=VectorReplayBuffer(80000, train_envs_vec.env_num),
        )
        test_collector = Collector(policy=policy, env=test_envs_vec)

        # ===== ベスト保存 =====
        best_path = log_root / "policy_best.pth"

        def save_best_fn(policy_obj):
            torch.save(policy_obj.state_dict(), best_path)
            print(f"[save_best] -> {best_path}")
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            uniq = log_root / f"policy_best_{ts}_{int(time.time())}.pth"
            torch.save(policy_obj.state_dict(), uniq)
            print(f"[save_best] (archived) -> {uniq}")
            try:
                save_obs_rms_from_vec(
                    train_envs_vec, str(log_root / "obs_rms_best.npz"), min_count=10
                )
            except Exception as e:
                print(f"[save_best] obs_rms save failed: {e}")

        # ===== 実行 =====
        train_envs_vec.reset()
        test_envs_vec.reset()
        train_collector.reset()
        test_collector.reset()

        result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=epochs,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            repeat_per_collect=repeat_per_collect,
            batch_size=batch_size,
            episode_per_test=episode_per_test,
            stop_fn=lambda mean_rew: (mean_rew >= stop_mean_rew),
            logger=logger,
            save_best_fn=save_best_fn,
        ).run()

        print("[update_train] Finetune finished:", result)
        print("[update_train] TensorBoard:", tb_run_dir)
        print("[update_train] Best policy:", best_path, best_path.exists())

        try:
            save_obs_rms_from_vec(
                train_envs_vec, str(log_root / "obs_rms_final.npz"), min_count=0
            )
        except Exception as e:
            print(f"[update_train] final obs_rms save failed: {e}")

        return result


if __name__ == "__main__":
    start = pd.Timestamp("2025-09-10 07:00:00")
    end = pd.Timestamp("2025-09-11 07:00:00")
    ppo = Trainer("data/base/hourly_filled.csv", "data/base/hourly_filled.csv", "Area1")
    ppo.setup()
    ppo.load(start_term=start, end_term=end)
    # ppo.train_run()
    ppo.reproduce(start_term=start, end_term=end)
    # ppo.update_train(start_term=start, end_term=end, area_name="Area1")
