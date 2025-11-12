# ---------- 便利フック: update() 経由で KL/clip_frac/entropy を確実ロギング ----------
import math
import numbers
from dataclasses import asdict, is_dataclass

import numpy as np
import torch


def _coerce_rms(r):
    try:
        if r is None:
            return None
        if isinstance(r, dict):
            m = np.asarray(r.get("mean", None), dtype=np.float64)
            v = np.asarray(r.get("var", None), dtype=np.float64)
            c = float(r.get("count", 0.0))
            if m is None or v is None:
                return None
            return m, v, c
        m = np.asarray(getattr(r, "mean", None), dtype=np.float64)
        v = np.asarray(getattr(r, "var", None), dtype=np.float64)
        cnt = None
        for k in ("count", "n", "num_steps", "steps", "total_count"):
            if hasattr(r, k):
                cnt = float(getattr(r, k))
                break
        if m is None or v is None or cnt is None:
            return None
        return m, v, cnt
    except Exception:
        return None


def save_obs_rms_from_vec(vec_env, npz_path: str, min_count: float = 0.0) -> bool:
    """
    SubprocVectorEnv から各ワーカーの obs_rms を集め、
    count が最大のものを npz で保存（mean/var/count）。
    """
    cands = []

    # 1) get_obs_rms を直接呼べる場合（推奨）
    if hasattr(vec_env, "call_env_method"):
        try:
            lst = vec_env.call_env_method("get_obs_rms")
            for r in lst or []:
                z = _coerce_rms(r)
                if z is not None:
                    cands.append(z)
        except Exception as e:
            print(f"[obs_rms] call_env_method('get_obs_rms') failed: {e}")

    # 2) フォールバック：属性を覗く
    if not cands and hasattr(vec_env, "get_env_attr"):
        for key in ("obs_rms", "_obs_rms", "rms"):
            try:
                lst = vec_env.get_env_attr(key)
            except Exception:
                lst = []
            for r in lst or []:
                z = _coerce_rms(r)
                if z is not None:
                    cands.append(z)

    if not cands:
        print("[obs_rms] not found on workers; skip")
        return False

    # 3) 最大カウントを選ぶ
    best = max(cands, key=lambda t: t[2])
    mean, var, cnt = best
    if cnt <= min_count:
        print(f"[obs_rms] count={cnt:.1f} <= {min_count}; skip")
        return False

    np.savez(npz_path, mean=mean, var=var, count=np.asarray(cnt, dtype=np.float64))
    print(f"[obs_rms] saved -> {npz_path} (count={cnt:.1f})")
    return True


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
        b = kwargs.get("buffer", None)  # noqa
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
