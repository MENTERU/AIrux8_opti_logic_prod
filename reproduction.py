# replay_aircontrol_episode.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tianshou.data import Batch as TBatch

from deep_reinforcement_learning.agent.ppo_agent import (
    actions_to_frame,
    create_ppo_for_hvac,
)
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
            "環境から操作列名を取得できませんでした。env.set_temp_cols / set_mode_cols / set_fan_cols / set_onoff_cols "
            "が公開されている必要があります。"
        ) from e


def run_one_episode_and_dump(
    *,
    base_csv: Path,
    model_path: Path,
    start_term: str,
    end_term: str,
    weights_root: Path,
    out_dir: Path,
    n_devices: int = 26,
    episode_len: int = 24,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ====== データ / 期間 ======
    base_df = pd.read_csv(base_csv)
    base_df["Datetime_hour"] = pd.to_datetime(base_df["Datetime_hour"])
    base_df = base_df.set_index("Datetime_hour", drop=True)
    t0 = pd.Timestamp(start_term)
    tend = pd.Timestamp(end_term)

    historical_df = base_df[base_df.index < t0]
    validate_df = base_df[base_df.index >= t0]

    # ====== モデル & 環境 ======
    model = load_residual_model(str(model_path))
    init_args = AirControlInit(
        model_or_path=model,
        base_df=historical_df,
        start_term=t0,
        end_term=tend,
        weather_forecast=validate_df,
        reward_params=None,
    )
    env = make_env_factory_aircontrol(init_args, seed=seed)()

    # アクション列名（train 環境と同じ）
    set_cols, mode_cols, fan_cols, onoff_cols = _get_action_columns_from_env(env)

    # ====== Policy 構築 & ロード ======
    policy = create_ppo_for_hvac(
        single_env=env,
        device=DEVICE,
        lr=3e-4,
        set_temp_list=set_temp_range,
        set_mode_list=set_mode_range,
        set_wind_list=set_fan_range,
        set_on_off_list=set_on_off_range,
        n_devices=n_devices,
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
    policy.eval()

    pol_path = _pick_policy_path(weights_root)
    state_dict = torch.load(pol_path, map_location=DEVICE)
    policy.load_state_dict(state_dict)
    print(f"[policy] loaded: {pol_path}")

    # obs_rms があれば適用
    obs_rms_npz = _pick_obs_rms(weights_root)
    _apply_obs_rms_if_available(env, obs_rms_npz)

    # ====== 1 エピソード実行 ======
    try:
        obs, info = env.reset(seed=seed)
    except TypeError:
        obs, info = env.reset()

    ep_ret = 0.0
    rows: List[pd.DataFrame] = []

    for step in range(episode_len):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(
            0
        )

        # action mask を info から（あれば）渡す
        info_batch = TBatch()
        if (
            isinstance(info, dict)
            and "action_mask" in info
            and info["action_mask"] is not None
        ):
            mask_np = np.asarray(info["action_mask"], dtype=np.float32)
            if mask_np.ndim == 1:
                mask_np = mask_np[None, :]
            info_batch = TBatch(action_mask=mask_np)

        with torch.no_grad():
            out = policy(TBatch(obs=obs_tensor, info=info_batch))
            act = out.act.detach().cpu().numpy()[0].astype(int)

        # ステップ実行
        obs, rew, term, trunc, info = env.step(act)
        ep_ret += float(rew)
        done = bool(term or trunc)

        # === ここが肝心：actions_to_frame で DataFrame 化 ===
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
        pred_map = {}
        if (
            isinstance(info, dict)
            and "predicted" in info
            and isinstance(info["predicted"], dict)
        ):
            pred_map = info["predicted"]  # 例: {"pred__total_kwh__41-1": 1.23, ...}

        if pred_map:
            df_pred = pd.DataFrame([pred_map], index=df_one.index)
            df_one = pd.concat([df_one, df_pred], axis=1)
        # 追加で step / reward / return を列として付与
        df_one = df_one.assign(step=step, reward=float(rew), ret_so_far=float(ep_ret))
        rows.append(df_one)

        if done:
            break

    # 連結して保存
    actions_df = pd.concat(rows, axis=0).set_index(
        "step", append=True
    )  # MultiIndex: (Datetime_hour, step)
    actions_csv = out_dir / "actions_episode.csv"
    actions_df.to_csv(actions_csv, encoding="utf-8")
    print(f"[save] actions -> {actions_csv}")

    # 報酬だけ（任意）
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
        "base_csv": str(base_csv),
        "model_path": str(model_path),
        "start_term": start_term,
        "end_term": end_term,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-csv", type=str, default="data/base/hourly_filled.csv")
    parser.add_argument("--model-path", type=str, default="models/xgb_weight.joblib")
    parser.add_argument("--start-term", type=str, default="2025-09-11 07:00:00")
    parser.add_argument("--end-term", type=str, default="2025-09-12 07:00:00")
    parser.add_argument(
        "--weights-root",
        type=str,
        required=False,
        default="./logs/aircontrol_ppo/20250911",
        help="train の日付フォルダ（例: ./logs/aircontrol_ppo/20250911）",
    )
    parser.add_argument("--out-dir", type=str, default="./replay_outputs")
    parser.add_argument("--episode-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-devices", type=int, default=26)
    args = parser.parse_args()
    date_str = pd.Timestamp(args.start_term).strftime("%Y%m%d")

    run_one_episode_and_dump(
        base_csv=Path(args.base_csv),
        model_path=Path(args.model_path),
        start_term=args.start_term,
        end_term=args.end_term,
        weights_root=Path(args.weights_root),
        out_dir=Path(args.out_dir) / date_str,
        n_devices=args.n_devices,
        episode_len=args.episode_len,
        seed=args.seed,
    )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    main()
