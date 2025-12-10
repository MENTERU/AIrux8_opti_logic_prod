from pathlib import Path
from typing import Optional

import numpy as np


def _pick_policy_path(log_root: Path) -> Path:
    """log_root 内の policy_best.pth を返す。無ければ最新の policy_best_*.pth を探索。"""
    p = log_root / "policy_best.pth"
    if p.exists():
        return p
    cands = sorted(log_root.glob("policy_best_*.pth"))
    if not cands:
        raise FileNotFoundError(f"policy not found under: {log_root}")
    return cands[-1]


def _pick_obs_rms(log_root: Path) -> Optional[Path]:
    """obs_rms_best.npz / obs_rms_final.npz のどちらか（優先は best）。無ければ None。"""
    best = log_root / "obs_rms_best.npz"
    if best.exists():
        return best
    fin = log_root / "obs_rms_final.npz"
    return fin if fin.exists() else None


def _apply_obs_rms_if_available(env, npz_path: Optional[Path]) -> bool:
    """
    ・normalize wrapper を使っている想定で、env 直下/内部にある obs_rms 相当へ mean/var/count を上書き。
    ・存在しないときは何もしない（False）。
    ＊train 側の save_obs_rms_from_vec() の出力フォーマットに合わせています。
    """
    if npz_path is None or not npz_path.exists():
        print("[obs_rms] not applied (file not found).")
        return False
    try:
        stats = np.load(npz_path)
        mean = np.asarray(stats["mean"], dtype=np.float64)
        var = np.asarray(stats["var"], dtype=np.float64)
        count = float(np.asarray(stats["count"], dtype=np.float64))
    except Exception as e:
        print(f"[obs_rms] failed to load: {e}")
        return False

    # normalize wrapper っぽい場所を雑に探索
    holders = []
    for key in ("obs_rms", "_obs_rms", "rms"):
        if hasattr(env, key):
            holders.append((env, key))
    cur = getattr(env, "env", None)
    while cur is not None:
        for key in ("obs_rms", "_obs_rms", "rms"):
            if hasattr(cur, key):
                holders.append((cur, key))
        cur = getattr(cur, "env", None)

    for holder, key in holders:
        try:
            tgt = getattr(holder, key)
            if tgt is None:
                continue
            # mean / var
            if hasattr(tgt, "mean"):
                try:
                    np.copyto(tgt.mean, mean, casting="unsafe")
                except Exception:
                    setattr(tgt, "mean", mean.copy())
            if hasattr(tgt, "var"):
                try:
                    np.copyto(tgt.var, var, casting="unsafe")
                except Exception:
                    setattr(tgt, "var", var.copy())
            # count 系
            for nm in ("count", "n", "num_steps", "steps", "total_count"):
                if hasattr(tgt, nm):
                    try:
                        setattr(tgt, nm, float(count))
                    except Exception:
                        pass
                    break
            # update を無効化（収集時に更新しない）
            try:
                tgt.update = lambda *a, **k: None
            except Exception:
                pass
            print(
                f"[obs_rms] applied to {type(holder).__name__}.{key}  count={count:.1f}"
            )
            return True
        except Exception as e:
            print(f"[obs_rms] apply failed on {type(holder).__name__}.{key}: {e}")
    print("[obs_rms] holder not found.")
    return False
