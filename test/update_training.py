import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tianshou.data import Batch as TBatch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from torch.utils.tensorboard import SummaryWriter

from deep_reinforcement_learning.agent.ppo_agent import create_ppo_for_hvac
from deep_reinforcement_learning.const import (  # weather_cols,
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", type=str, default="data/base/hourly_filled.csv")
    ap.add_argument("--weather-csv", type=str, default="data/base/hourly_filled.csv")
    ap.add_argument("--model-path", type=str, default="models/xgb_weight.joblib")
    ap.add_argument(
        "--start-term",
        type=str,
        default="2025-09-11 07:00:00",
        required=False,
        help="学習対象開始時刻（trainと同じ）",
    )
    ap.add_argument(
        "--end-term",
        type=str,
        default="2025-09-12 07:00:00",
        required=False,
        help="学習対象終了時刻（trainと同じ）",
    )
    ap.add_argument(
        "--weights-root",
        type=str,
        required=False,
        default="./logs/aircontrol_ppo/20250910",
        help="続き学習の元となる重み/obs_rmsを含むフォルダ（例: ./logs/aircontrol_ppo/20250910 または _02）",
    )
    ap.add_argument(
        "--out-root",
        type=str,
        default="./logs/aircontrol_ppo",
        help="再学習結果の保存ルート（start_termの日付配下に作成）",
    )
    # 追加エポック/ステップ設定（デフォルトは元スクリプトに揃える）
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--step-per-epoch", type=int, default=240)
    ap.add_argument("--step-per-collect", type=int, default=120)
    ap.add_argument("--repeat-per-collect", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--episode-per-test", type=int, default=1)
    ap.add_argument("--stop-mean-rew", type=float, default=0.0)
    # 並列環境数
    ap.add_argument("--train-envs", type=int, default=2)
    ap.add_argument("--test-envs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--n-devices", type=int, default=26)
    args = ap.parse_args()

    # ===== データ =====
    base_df = pd.read_csv(args.base_csv)
    base_df["Datetime_hour"] = pd.to_datetime(base_df["Datetime_hour"])
    base_df = base_df.set_index("Datetime_hour", drop=True)

    t0 = pd.Timestamp(args.start_term)
    tend = pd.Timestamp(args.end_term)
    historical_df = base_df[base_df.index < t0]
    weather_df = pd.read_csv(args.weather_csv)
    weather_df["Datetime_hour"] = pd.to_datetime(weather_df["Datetime_hour"])
    weather_df = weather_df.set_index("Datetime_hour", drop=True)

    # ===== モデル・環境 init =====
    model = load_residual_model(args.model_path)
    init_args = AirControlInit(
        model_or_path=model,
        base_df=historical_df,
        start_term=t0,
        end_term=tend,
        weather_forecast=weather_df,
        reward_params=None,
    )

    train_fns = [
        make_env_factory_aircontrol(init_args, seed=args.seed + i)
        for i in range(args.train_envs)
    ]
    test_fns = [
        make_env_factory_aircontrol(init_args, seed=10000 + i)
        for i in range(args.test_envs)
    ]
    train_envs = SubprocVectorEnv(train_fns)
    test_envs = SubprocVectorEnv(test_fns)

    # 代表 env
    single_env = make_env_factory_aircontrol(init_args, seed=999)()
    # 可能なら obs_rms を適用（代表env のみでOK。vector化側は学習中に更新済みの統計を使うこともあります）
    _apply_obs_rms_if_available(single_env, _pick_obs_rms(Path(args.weights_root)))

    # ===== Policy 構築 & 既存重みのロード =====
    policy = create_ppo_for_hvac(
        single_env=single_env,
        device=DEVICE,
        lr=3e-4,
        set_temp_list=set_temp_range,
        set_mode_list=set_mode_range,
        set_wind_list=set_fan_range,
        set_on_off_list=set_on_off_range,
        n_devices=args.n_devices,
        deterministic_eval=True,
        discount_factor=0.9,
        gae_lambda=0.95,
        eps_clip=0.18,
        value_clip=True,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        advantage_normalization=True,
        reward_normalization=True,
    )
    policy.actor.to(DEVICE)
    policy.critic.to(DEVICE)
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
        # 収集時（_return_dist が False）なら dist を取り除く
        if not getattr(policy, "_return_dist", False):
            if isinstance(ret, TBatch):
                # 属性として持っている場合
                try:
                    if hasattr(ret, "dist"):
                        delattr(ret, "dist")
                except Exception:
                    pass
                # dict 的に保持している場合
                try:
                    ret.pop("dist")
                except Exception:
                    pass
        return ret

    policy.forward = _forward_conditional
    pol_path = _pick_policy_path(Path(args.weights_root))
    state_dict = torch.load(pol_path, map_location=DEVICE)
    policy.load_state_dict(state_dict)
    print(f"[resume] loaded weights from: {pol_path}")

    # ===== ログ保存先（start_term の日付でフォルダを切る）=====
    date_str = pd.Timestamp(init_args.start_term).strftime("%Y%m%d")
    base_log_root = Path(args.out_root) / date_str
    log_root = base_log_root
    idx = 2
    while log_root.exists():
        log_root = Path(f"{base_log_root}_{idx:02d}")
        idx += 1
    log_root.mkdir(parents=True, exist_ok=True)

    tb_run_dir = log_root / "runs" / "ppo_finetune"
    tb_run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_run_dir))
    logger = TensorboardLogger(writer)

    trainer_cfg = dict(
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        repeat_per_collect=args.repeat_per_collect,
        batch_size=args.batch_size,
        episode_per_test=args.episode_per_test,
        stop_mean_rew=args.stop_mean_rew,
        resume_from=str(pol_path),
        resume_obs_rms=str(_pick_obs_rms(Path(args.weights_root)) or ""),
    )
    summarize_hparams_to_tb(
        writer,
        policy=policy,
        device=DEVICE,
        train_envs=train_envs,
        test_envs=test_envs,
        set_temp_list=set_temp_range,
        set_mode_list=set_mode_range,
        set_wind_list=set_fan_range,
        set_on_off_list=set_on_off_range,
        n_devices=args.n_devices,
        trainer_cfg=trainer_cfg,
    )

    # 追加の update ロギング
    policy = attach_update_logging_to_tb(policy, writer)

    # ===== Collector 構築 =====
    train_collector = Collector(
        policy=policy,
        env=train_envs,
        buffer=VectorReplayBuffer(80000, train_envs.env_num),
    )
    test_collector = Collector(policy=policy, env=test_envs)

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
                train_envs, str(log_root / "obs_rms_best.npz"), min_count=10
            )
        except Exception as e:
            print(f"[save_best] obs_rms save failed: {e}")

    # ===== 実行 =====
    # 初期 reset
    train_envs.reset()
    test_envs.reset()
    train_collector.reset()
    test_collector.reset()

    result = OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epochs,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        repeat_per_collect=args.repeat_per_collect,
        batch_size=args.batch_size,
        episode_per_test=args.episode_per_test,
        stop_fn=lambda mean_rew: (mean_rew >= args.stop_mean_rew),
        logger=logger,
        save_best_fn=save_best_fn,
    ).run()

    print("Finetune finished:", result)
    print("TensorBoard:", tb_run_dir)
    print("Best policy:", best_path, best_path.exists())
    try:
        save_obs_rms_from_vec(
            train_envs, str(log_root / "obs_rms_final.npz"), min_count=0
        )
    except Exception as e:
        print(f"[final] obs_rms save failed: {e}")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    main()
# python update_training.py \
#   --weights-root ./logs/aircontrol_ppo/20250910 \
#   --base-csv data/base/hourly_filled.csv \
#   --weather-csv data/base/hourly_filled.csv \
#   --model-path models/xgb_weight.joblib \
#   --start-term "2025-09-10 07:00:00" \
#   --end-term   "2025-09-11 07:00:00" \
#   --out-dir ./logs/aircontrol_ppo_finetune \
