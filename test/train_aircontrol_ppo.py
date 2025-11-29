# train_aircontrol_ppo.py
from __future__ import annotations

import multiprocessing as mp
import os
import time
from datetime import datetime

import pandas as pd
import torch
from tianshou.data import Batch as TBatch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from deep_reinforcement_learning.agent.ppo_agent import create_ppo_for_hvac
from deep_reinforcement_learning.const import (
    set_fan_range,
    set_mode_range,
    set_on_off_range,
    set_temp_range,
)

# ==== あなたの既存モジュール ====
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


def main():
    # macOS/Windows 対応（spawn）
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # ====== データ読込・期間分割 ======
    base_df = pd.read_csv("data/base/hourly_filled.csv")
    base_df["Datetime_hour"] = pd.to_datetime(base_df["Datetime_hour"])
    base_df = base_df.set_index("Datetime_hour", drop=True)

    t0 = pd.Timestamp("2025-09-10 07:00:00")
    tend = pd.Timestamp("2025-09-11 07:00:00")

    historical_df = base_df[base_df.index < t0]
    validate_df = base_df[base_df.index >= t0]

    # ====== モデル・初期化引数 ======
    model = load_residual_model("models/xgb_weight.joblib")
    init_args = AirControlInit(
        model_or_path=model,  # そのまま渡してOK（パスでも可）
        base_df=historical_df,
        start_term=t0,
        end_term=tend,
        weather_forecast=validate_df,
        reward_params=None,  # 省略でデフォルト
    )

    # ====== 環境 ======
    NUM_TRAIN_ENVS = 1
    NUM_TEST_ENVS = 1
    train_fns = [
        make_env_factory_aircontrol(init_args, seed=100 + i)
        for i in range(NUM_TRAIN_ENVS)
    ]
    test_fns = [
        make_env_factory_aircontrol(init_args, seed=200 + i)
        for i in range(NUM_TEST_ENVS)
    ]
    train_envs = SubprocVectorEnv(train_fns)
    test_envs = SubprocVectorEnv(test_fns)

    # 代表Env（ポリシー構築用のスペック取得）
    single_env = make_env_factory_aircontrol(init_args, seed=999)()

    # ====== PPO Policy (HVAC専用アクタ) ======
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = create_ppo_for_hvac(
        single_env=single_env,
        device=device,
        lr=3e-4,
        set_temp_list=set_temp_range,
        set_mode_list=set_mode_range,
        set_wind_list=set_fan_range,
        set_on_off_list=set_on_off_range,
        n_devices=26,
        deterministic_eval=True,
        # PPOハイパラ（最低限/保守的）
        discount_factor=0.9,
        gae_lambda=0.95,
        eps_clip=0.18,
        value_clip=True,
        vf_coef=0.5,
        ent_coef=0.0,  # まずは0で安定化→あとで探索強化したければ 0.001-0.01
        max_grad_norm=0.5,
        advantage_normalization=True,
        reward_normalization=True,
    )
    policy.actor.to(device)
    policy.critic.to(device)
    _orig_update = policy.update

    def _update_with_dist_flag(*args, **kwargs):
        setattr(policy, "_return_dist", True)
        try:
            return _orig_update(*args, **kwargs)
        finally:
            setattr(policy, "_return_dist", False)

    policy.update = _update_with_dist_flag
    _orig_forward = policy.forward

    def _forward_conditional(*args, **kwargs):
        ret = _orig_forward(*args, **kwargs)
        # ret は Tianshou の Batch 互換（ActBatch）想定
        if not getattr(policy, "_return_dist", False):
            # 収集時: dist を除去（Collector が len() を取っても安全になる）
            if isinstance(ret, TBatch):
                if hasattr(ret, "dist"):
                    try:
                        delattr(ret, "dist")
                    except Exception:
                        # Batch 実装によっては dict として保持されることもある
                        try:
                            ret.pop("dist")
                        except Exception:
                            pass
        return ret

    policy.forward = _forward_conditional

    # ====== Collector / Buffer ======
    train_collector = Collector(
        policy=policy,
        env=train_envs,
        buffer=VectorReplayBuffer(80000, train_envs.env_num),
    )
    test_collector = Collector(policy=policy, env=test_envs)
    date_str = pd.Timestamp(init_args.start_term).strftime("%Y%m%d")
    base_log_root = f"./logs/aircontrol_ppo/{date_str}"
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
        step_per_epoch=24,  # ↓この後で使う定義値と揃える
        step_per_collect=24,
        repeat_per_collect=3,
        batch_size=24,
        episode_per_test=1,
        stop_mean_rew=0.0,
    )
    summarize_hparams_to_tb(
        writer,
        policy=policy,
        device=device,
        train_envs=train_envs,
        test_envs=test_envs,
        set_temp_list=set_temp_range,
        set_mode_list=set_mode_range,
        set_wind_list=set_fan_range,
        set_on_off_list=set_on_off_range,
        n_devices=26,
        trainer_cfg=trainer_cfg,
    )
    # 追加の update ロギング
    policy = attach_update_logging_to_tb(policy, writer)

    # ====== 実体化（初回reset） ======
    train_envs.reset()
    test_envs.reset()
    train_collector.reset()
    test_collector.reset()

    print("train num envs:", train_envs.env_num)
    print("test  num envs:", test_envs.env_num)
    print("action_space:", single_env.action_space)
    print("obs_space   :", single_env.observation_space)

    # ====== 学習設定 ======
    max_epoch = 20
    step_per_epoch = trainer_cfg["step_per_epoch"]
    step_per_collect = trainer_cfg["step_per_collect"]
    repeat_per_collect = trainer_cfg["repeat_per_collect"]
    batch_size = trainer_cfg["batch_size"]
    episode_per_test = trainer_cfg["episode_per_test"]
    stop_mean_rew = trainer_cfg["stop_mean_rew"]
    env_snapshots_root = os.path.join(log_root, "env_snapshots")
    os.makedirs(env_snapshots_root, exist_ok=True)
    best_path = os.path.join(log_root, "policy_best.pth")

    def save_best_fn(policy_obj):
        # 既存の「常に固定名」も保存（運用で一番使いやすい）
        torch.save(policy_obj.state_dict(), best_path)
        print(f"[save_best] -> {best_path}")

        # 併せて履歴用のユニーク名でも保存（例：時刻とUNIX秒）
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        uniq = os.path.join(log_root, f"policy_best_{ts}_{int(time.time())}.pth")
        torch.save(policy_obj.state_dict(), uniq)
        print(f"[save_best] (archived) -> {uniq}")
        try:
            save_obs_rms_from_vec(
                train_envs, os.path.join(log_root, "obs_rms_best.npz"), min_count=10
            )
        except Exception as e:
            print(f"[save_best] obs_rms save failed: {e}")

    # ====== トレーナ起動 ======
    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=max_epoch,
        step_per_epoch=step_per_epoch,
        step_per_collect=step_per_collect,
        repeat_per_collect=repeat_per_collect,
        batch_size=batch_size,
        episode_per_test=episode_per_test,
        stop_fn=lambda mean_rew: (mean_rew >= stop_mean_rew),
        logger=logger,
        save_best_fn=save_best_fn,
    ).run()

    print("Training finished:", result)
    print("TensorBoard:", tb_run_dir)
    print("Best policy:", best_path, os.path.exists(best_path))
    try:
        save_obs_rms_from_vec(
            train_envs, os.path.join(log_root, "obs_rms_final.npz"), min_count=0
        )
    except Exception as e:
        print(f"[final] obs_rms save failed: {e}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    main()
