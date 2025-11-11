# train_aircontrol_ppo.py
from __future__ import annotations

import math
import multiprocessing as mp
import numbers
import os
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime

import numpy as np
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


# ---------- 便利フック: update() 経由で KL/clip_frac/entropy を確実ロギング ----------
def attach_update_logging_to_tb(policy, writer, tags=None, prefix="update"):
    orig_update = policy.update
    step = {"n": 0}
    once = {"printed": False}

    def _to_float_if_scalar(x):
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return float(x.detach().reshape(()).item())
            return None
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return float(x)
        if isinstance(x, numbers.Number):
            return float(x)
        if isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.reshape(()))
            return None
        return None

    def _flatten(obj, key_prefix=""):
        if is_dataclass(obj):
            obj = asdict(obj)
        elif hasattr(obj, "__dict__") and not isinstance(obj, dict):
            try:
                obj = {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}
            except Exception:
                obj = {}
        out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{key_prefix}{k}"
                if isinstance(v, dict) or is_dataclass(v) or hasattr(v, "__dict__"):
                    out.update(_flatten(v, key + "/"))
                    continue
                if isinstance(v, (list, tuple)):
                    if len(v) == 1:
                        fv = _to_float_if_scalar(v[0])
                        if fv is not None and math.isfinite(fv):
                            out[key] = fv
                    continue
                fv = _to_float_if_scalar(v)
                if fv is not None and math.isfinite(fv):
                    out[key] = fv
            return out
        else:
            fv = _to_float_if_scalar(obj)
            if fv is not None and math.isfinite(fv):
                out[key_prefix.rstrip("/")] = fv
            return out

    def _alias_stats(d: dict) -> dict:
        m = dict(d)
        for k in ["clip_fraction", "clipfrac", "clip_frac", "clipratio", "clip_ratio"]:
            if k in m:
                m["clip_frac"] = float(m[k])
                break
        for k in ["approx_kl", "approxkl", "approx_kl_divergence", "kl", "kl_div"]:
            if k in m:
                m["approx_kl"] = float(m[k])
                break
        if "entropy_loss" in m and "entropy" not in m:
            try:
                m["entropy"] = -float(m["entropy_loss"])
            except Exception:
                pass
        return m

    @torch.no_grad()
    def _compute_mean_entropy_from_batch(batch):
        device = next(policy.parameters()).device
        if not hasattr(batch, "to_torch"):
            return None
        b = batch.to_torch(dtype=torch.float32, device=device)
        obs = b.obs
        state = getattr(b, "state", None)
        info = getattr(b, "info", None)

        out = policy.actor(obs, state=state, info=info)
        logits = out[0] if isinstance(out, (tuple, list)) else out  # ★ここを追加
        try:
            dist = policy.dist_fn(logits)
        except Exception:
            return None

        ent = dist.entropy()
        if isinstance(ent, torch.Tensor) and ent.ndim > 1:
            ent = ent.sum(dim=tuple(range(1, ent.ndim)))
        return float(ent.mean().item())

    def wrapped_update(*args, **kwargs):
        ret = orig_update(*args, **kwargs)
        scalars = _alias_stats(_flatten(ret))
        if not once["printed"]:
            print("[update keys]", sorted(list(scalars.keys())))
            once["printed"] = True

        # 実バッチに基づく entropy 推定（オプション）
        b = kwargs.get("buffer", None)
        # tianshou内部の update 実装では実 Batch は buffer.sample で取り出すため、
        # ここでは戻り値ベースの記録を主とし、補助的に collector 側で entropy を出す前提でもOK。

        # 書き込み
        target_keys = tags if tags else scalars.keys()
        wrote = 0
        for k in target_keys:
            if k in scalars and math.isfinite(scalars[k]):
                writer.add_scalar(f"{prefix}/{k}", scalars[k], step["n"])
                wrote += 1
        if wrote == 0:
            for k in [
                "approx_kl",
                "clip_frac",
                "loss_actor",
                "loss_critic",
                "entropy",
                "entropy_loss",
            ]:
                if k in scalars and math.isfinite(scalars[k]):
                    writer.add_scalar(f"{prefix}/{k}", scalars[k], step["n"])
        step["n"] += 1
        if step["n"] % 20 == 0:
            writer.flush()
        return ret

    policy.update = wrapped_update
    return policy


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
    NUM_TRAIN_ENVS = 10
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
        discount_factor=0.99,
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
    log_root = f"./logs/aircontrol_ppo/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(log_root, exist_ok=True)
    tb_run_dir = os.path.join(log_root, "runs", "ppo")
    os.makedirs(tb_run_dir, exist_ok=True)

    writer = SummaryWriter(tb_run_dir)
    writer.add_text(
        "hparams",
        "algo=PPO(HVAC), lr=3e-4, gamma=0.99, gae_lambda=0.95, eps_clip=0.18, "
        f"vf_coef=0.5, ent_coef=0.0, max_grad_norm=0.5, "
        f"envs={train_envs.env_num}/{test_envs.env_num}, "
        "step_per_epoch=240, step_per_collect=120, repeat_per_collect=5, batch_size=64",
    )
    logger = TensorboardLogger(writer)

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
    max_epoch = 25
    step_per_epoch = 480  # 1 epoch で集める環境ステップ
    step_per_collect = 240  # 1 collect あたりのサンプル数
    repeat_per_collect = 3  # 収集データでの反復学習回数
    batch_size = 256
    episode_per_test = 1
    stop_mean_rew = 0.0  # 適宜変更（AirControl のスケール依存）

    # ベストモデル保存
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


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    main()
