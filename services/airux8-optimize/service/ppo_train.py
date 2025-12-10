import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Type, Union

import pandas as pd
import torch
from deep_reinforcement_learning.agent.ppo_agent import create_ppo_for_hvac
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
from deep_reinforcement_learning.utils.record import (
    attach_update_logging_to_tb,
    save_obs_rms_from_vec,
    summarize_hparams_to_tb,
)
from input_info.crea_building_information import (
    CreaBuilding,
)  # ← 実際のパスに合わせて修正
from tianshou.data import Batch as TBatch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class AircontrolPPOTrainConfig:
    # ベクター環境
    num_train_envs: int = 10
    num_test_envs: int = 1

    # ログ
    log_dir_root: str = "./logs/aircontrol_ppo/{area}"

    # ビルのエリア指定（None なら全体）
    area: Optional[str] = None
    building_cls: Type[CreaBuilding] = CreaBuilding

    # PPO ハイパーパラメータ
    lr: float = 3e-4
    discount_factor: float = 0.9
    gae_lambda: float = 0.99
    eps_clip: float = 0.2
    value_clip: bool = True
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    max_grad_norm: float = 0.5
    advantage_normalization: bool = True
    reward_normalization: bool = True
    deterministic_eval: bool = True

    # Trainer ハイパーパラメータ
    max_epoch: int = 50
    step_per_epoch: int = 480
    step_per_collect: int = 240
    repeat_per_collect: int = 5
    batch_size: int = 120
    episode_per_test: int = 1
    stop_mean_rew: float = 0.0

    # デバイス数
    n_devices: Optional[int] = None

    def resolve_n_devices(self) -> int:
        if self.n_devices is not None:
            return self.n_devices
        if self.area is not None:
            unit_info = self.building_cls.Unit_dict[self.area]
            return len(unit_info["idu"])
        # 既存コードのデフォルト（全館）
        return 26


class AircontrolPPOTrainer:
    def __init__(
        self,
        cfg: AircontrolPPOTrainConfig,
        *,
        base_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        start_term: pd.Timestamp,
        end_term: pd.Timestamp,
        temp_model_or_path: Union[str, Any],
        elec_model_or_path: Union[str, Any],
        unit_temp_range_list: list,
        target_temp_list: list,
    ):
        """
        Parameters
        ----------
        cfg:
            ハイパーパラメータ等の設定
        base_df:
            DatetimeIndex を持つ元データ（室内機・室外機などの系列のみ）
        weather_df:
            DatetimeIndex を持つ天気・時間特徴などの DataFrame
        start_term, end_term:
            強化学習エピソードの期間
        temp_model_or_path:
            室内温度を予測するモデル or joblib パス
        elec_model_or_path:
            室外機消費電力を予測するモデル or joblib パス
        """
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.unit_temp_range_list = unit_temp_range_list
        self.target_temp_list = target_temp_list

        # 渡されたデータ・期間・モデルを保持
        if not isinstance(base_df.index, pd.DatetimeIndex):
            raise ValueError("base_df.index は DatetimeIndex 必須です。")
        if not isinstance(weather_df.index, pd.DatetimeIndex):
            raise ValueError("weather_df.index は DatetimeIndex 必須です。")

        # index を揃えてソート
        self.base_df_raw = base_df.sort_index()
        self.weather_df_raw = weather_df.sort_index()

        self.start_term = start_term
        self.end_term = end_term

        if isinstance(temp_model_or_path, str):
            self.temp_model = load_residual_model(temp_model_or_path)
        else:
            self.temp_model = temp_model_or_path

        if isinstance(elec_model_or_path, str):
            self.elec_model = load_residual_model(elec_model_or_path)
        else:
            self.elec_model = elec_model_or_path

        # 後から埋まるもの
        self.base_df: Optional[pd.DataFrame] = None  # area + weather 結合後
        self.historical_df: Optional[pd.DataFrame] = None  # 過去
        self.validate_df: Optional[pd.DataFrame] = None  # 将来

        self.init_args: Optional[AirControlInit] = None

        self.train_envs = None
        self.test_envs = None
        self.train_collector: Optional[Collector] = None
        self.test_collector: Optional[Collector] = None

        self.policy = None
        self.writer: Optional[SummaryWriter] = None
        self.logger: Optional[TensorboardLogger] = None

        self.log_root: Optional[str] = None
        self.best_path: Optional[str] = None

    def set_area(self, area: str):
        self.cfg.area = area

    # -------------------------
    #  データ前処理（天気結合 + 期間分割）
    # -------------------------
    def _prepare_data(self) -> None:
        base_df = self.base_df_raw.copy()
        weather_df = self.weather_df_raw.copy()[CreaBuilding().weather_colimns]
        # --- 天気・時間特徴を結合 ---
        # index で left join（base_df のタイムスタンプ基準）
        # weather_df 側に「Outdoor Temp.」や時間特徴が載っている想定
        combined_df = base_df.join(weather_df, how="left")

        # 期間分割
        t0 = self.start_term
        historical_df = combined_df[combined_df.index < t0]
        validate_df = combined_df[combined_df.index >= t0]

        self.base_df = combined_df
        self.historical_df = historical_df
        self.validate_df = validate_df

    # -------------------------
    #   モデル & 環境 & Policy 構築
    # -------------------------
    def _build_model_env_and_policy(self) -> None:
        cfg = self.cfg
        assert self.historical_df is not None and self.validate_df is not None

        # 環境初期化引数
        init_args = AirControlInit(
            model_temp=self.temp_model,
            model_elec=self.elec_model,
            base_df=self.historical_df,
            start_term=self.start_term,
            end_term=self.end_term,
            weather_forecast=self.validate_df,
            reward_params=None,
            unit_temp_range_list=self.unit_temp_range_list,
            target_temp_list=self.target_temp_list,
        )

        num_train_envs = cfg.num_train_envs
        num_test_envs = cfg.num_test_envs

        train_fns = [
            make_env_factory_aircontrol(init_args, seed=100 + i)
            for i in range(num_train_envs)
        ]
        test_fns = [
            make_env_factory_aircontrol(init_args, seed=200 + i)
            for i in range(num_test_envs)
        ]
        train_envs = SubprocVectorEnv(train_fns)
        test_envs = SubprocVectorEnv(test_fns)

        # 代表 Env
        single_env = make_env_factory_aircontrol(init_args, seed=999)()

        # PPO Policy
        n_devices = cfg.resolve_n_devices()
        policy = create_ppo_for_hvac(
            single_env=single_env,
            device=self.device,
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
        policy.actor.to(self.device)
        policy.critic.to(self.device)

        # === dist を Collector から隠す wrap（元コードと同じ） ===
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
            if not getattr(policy, "_return_dist", False):
                if isinstance(ret, TBatch):
                    if hasattr(ret, "dist"):
                        try:
                            delattr(ret, "dist")
                        except Exception:
                            try:
                                ret.pop("dist")
                            except Exception:
                                pass
            return ret

        policy.forward = _forward_conditional  # type: ignore[assignment]

        # Collector / Buffer
        train_collector = Collector(
            policy=policy,
            env=train_envs,
            buffer=VectorReplayBuffer(80000, train_envs.env_num),
        )
        test_collector = Collector(policy=policy, env=test_envs)

        self.init_args = init_args
        self.train_envs = train_envs
        self.test_envs = test_envs
        self.train_collector = train_collector
        self.test_collector = test_collector
        self.policy = policy

        # 初回 reset
        self.train_envs.reset()
        self.test_envs.reset()
        self.train_collector.reset()
        self.test_collector.reset()

        print("train num envs:", self.train_envs.env_num)
        print("test  num envs:", self.test_envs.env_num)
        print("action_space:", single_env.action_space)
        print("obs_space   :", single_env.observation_space)

    # -------------------------
    #   ロガー構築
    # -------------------------
    def _build_logger(self) -> None:
        assert self.policy is not None
        assert self.train_envs is not None and self.test_envs is not None

        cfg = self.cfg
        date_str = pd.Timestamp(self.start_term).strftime("%Y%m%d")

        base_log_root = os.path.join(cfg.log_dir_root, date_str)
        log_root = base_log_root
        idx = 2
        while os.path.exists(log_root):
            log_root = f"{base_log_root}_{idx:02d}"
            idx += 1
        os.makedirs(log_root, exist_ok=True)

        tb_run_dir = os.path.join(log_root, "runs", "ppo")
        os.makedirs(tb_run_dir, exist_ok=True)

        writer = SummaryWriter(tb_run_dir)
        logger = TensorboardLogger(writer)

        trainer_cfg = dict(
            step_per_epoch=cfg.step_per_epoch,
            step_per_collect=cfg.step_per_collect,
            repeat_per_collect=cfg.repeat_per_collect,
            batch_size=cfg.batch_size,
            episode_per_test=cfg.episode_per_test,
            stop_mean_rew=cfg.stop_mean_rew,
        )

        summarize_hparams_to_tb(
            writer,
            policy=self.policy,
            device=self.device,
            train_envs=self.train_envs,
            test_envs=self.test_envs,
            set_temp_list=set_temp_range,
            set_mode_list=set_mode_range,
            set_wind_list=set_fan_range,
            set_on_off_list=set_on_off_range,
            n_devices=self.cfg.resolve_n_devices(),
            trainer_cfg=trainer_cfg,
        )

        # 追加の update ロギング
        self.policy = attach_update_logging_to_tb(self.policy, writer)

        self.writer = writer
        self.logger = logger
        self.log_root = log_root

    # -------------------------
    #   学習ループ
    # -------------------------
    def _train_loop(self):
        assert self.policy is not None
        assert self.train_envs is not None and self.test_envs is not None
        assert self.train_collector is not None and self.test_collector is not None
        assert self.logger is not None and self.log_root is not None

        cfg = self.cfg
        max_epoch = cfg.max_epoch
        step_per_epoch = cfg.step_per_epoch
        step_per_collect = cfg.step_per_collect
        repeat_per_collect = cfg.repeat_per_collect
        batch_size = cfg.batch_size
        episode_per_test = cfg.episode_per_test
        stop_mean_rew = cfg.stop_mean_rew

        env_snapshots_root = os.path.join(self.log_root, "env_snapshots")
        os.makedirs(env_snapshots_root, exist_ok=True)

        best_path = os.path.join(self.log_root, "policy_best.pth")
        self.best_path = best_path

        def save_best_fn(policy_obj):
            torch.save(policy_obj.state_dict(), best_path)
            print(f"[save_best] -> {best_path}")

            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            uniq = os.path.join(
                self.log_root, f"policy_best_{ts}_{int(time.time())}.pth"
            )
            torch.save(policy_obj.state_dict(), uniq)
            print(f"[save_best] (archived) -> {uniq}")
            try:
                save_obs_rms_from_vec(
                    self.train_envs,
                    os.path.join(self.log_root, "obs_rms_best.npz"),
                    min_count=10,
                )
            except Exception as e:
                print(f"[save_best] obs_rms save failed: {e}")

        result = OnpolicyTrainer(
            policy=self.policy,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
            max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            repeat_per_collect=repeat_per_collect,
            batch_size=batch_size,
            episode_per_test=episode_per_test,
            stop_fn=lambda mean_rew: (mean_rew >= stop_mean_rew),
            logger=self.logger,
            save_best_fn=save_best_fn,
        ).run()

        print("Training finished:", result)
        print("TensorBoard:", self.writer.log_dir if self.writer else "N/A")
        print("Best policy:", best_path, os.path.exists(best_path))

        try:
            save_obs_rms_from_vec(
                self.train_envs,
                os.path.join(self.log_root, "obs_rms_final.npz"),
                min_count=0,
            )
        except Exception as e:
            print(f"[final] obs_rms save failed: {e}")

        return result

    # -------------------------
    #   公開 API
    # -------------------------
    def run(self, area_name=None):
        # spawn モード設定（macOS/Windows 対応）
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        self.set_area(area=area_name)
        self._prepare_data()
        self._build_model_env_and_policy()
        self._build_logger()
        return self._train_loop()
