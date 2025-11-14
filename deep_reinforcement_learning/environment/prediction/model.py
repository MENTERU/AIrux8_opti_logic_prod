# ==== 安定保存・復元（完全差し替え） =========================================
# ==== 安定保存・復元（完全差し替え） =========================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor


# --- Booster を取り出すヘルパ ---
def _booster_from_any(est: Any) -> xgb.Booster | None:
    if isinstance(est, xgb.Booster):
        return est
    if hasattr(est, "get_booster"):
        try:
            b = est.get_booster()
            if isinstance(b, xgb.Booster):
                return b
        except Exception:
            pass
    for attr in ("booster_", "booster", "_Booster"):
        if hasattr(est, attr) and isinstance(getattr(est, attr), xgb.Booster):
            return getattr(est, attr)
    return None


def _best_iteration_from_booster(b: xgb.Booster | None) -> Optional[int]:
    """Booster から best_iteration を安全に取得。なければ None。"""
    if b is None:
        return None
    # 新しめの xgboost では property がある
    bi = getattr(b, "best_iteration", None)
    if isinstance(bi, int):
        return bi
    # 属性から読む（保存済みrawにも入っていることが多い）
    try:
        s = b.attr("best_iteration")
        if s is not None:
            return int(s)
    except Exception:
        pass
    return None


# --- 正規化: 任意の推定器から {params, raw(bytes), best_iteration} を抜く ---
def _extract_booster_raw_and_params(est: Any) -> Dict[str, Any]:
    # params
    try:
        params = est.get_params()
    except Exception:
        params = {}

    # raw booster
    raw = None
    b = _booster_from_any(est)
    if b is not None:
        raw = b.save_raw()
    elif hasattr(est, "save_raw"):
        raw = est.save_raw()

    if raw is None:
        raise TypeError(f"Unsupported estimator type for serialization: {type(est)}")

    # memoryview / bytearray / bytes → bytes に正規化
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    elif isinstance(raw, bytearray):
        raw = bytes(raw)
    elif not isinstance(raw, (bytes,)):
        raw = bytes(raw)

    # 列メモ（任意）
    x_cols = list(getattr(est, "_x_cols", []))
    y_cols = list(getattr(est, "_y_cols", []))

    best_it = _best_iteration_from_booster(b)

    return {
        "params": params,
        "raw": raw,
        "x_cols": x_cols,
        "y_cols": y_cols,
        "best_iteration": None if best_it is None else int(best_it),
    }


# --- 保存本体 ---
def dump_residual_model(res: Dict[str, Any], path: str):
    """
    fit_predict_eval_xgb_residual_only の戻り値 res を保存。
    res["model"] が単一でも MultiOutputXGB_Compat でもOK。
    """
    model = res["model"]
    x_cols = list(res["X_cols"])
    y_cols = list(res["Y_cols"])

    is_multi_bundle = hasattr(model, "estimators_") and hasattr(model, "y_cols_")
    payload: Dict[str, Any] = {
        "type": "multi" if is_multi_bundle else "single",
        "x_cols": x_cols,
        "y_cols": y_cols,
        "xgboost_version": xgb.__version__,
        "storage": {"raw_encoding": "bytes"},
    }

    if is_multi_bundle:
        payload["estimators"] = [
            _extract_booster_raw_and_params(e) for e in model.estimators_
        ]
    else:
        payload["estimators"] = [_extract_booster_raw_and_params(model)]

    joblib.dump(payload, path, compress=3)


# --- 復元: bytes → bytearray で load_model、best_iteration を再注入 ---
def _rebuild_regressor_from_raw(blob: Dict[str, Any]) -> XGBRegressor:
    params = blob.get("params") or {}
    raw_bytes: bytes = blob["raw"]
    best_it = blob.get("best_iteration", None)

    ba = bytearray(raw_bytes)

    est = XGBRegressor(**params)
    # 直接ロード（新しめの版）
    try:
        est.load_model(ba)
        booster = _booster_from_any(est)
    except TypeError:
        # 古い版: Booster を介して注入
        booster = xgb.Booster()
        booster.load_model(ba)
        est._Booster = booster

    # best_iteration を属性に再注入（property は attr から拾われる）
    if booster is not None and best_it is not None:
        try:
            booster.set_attr(best_iteration=str(int(best_it)))
        except Exception:
            pass

    # 列メモの復元（保存側に入っていれば）
    if blob.get("x_cols"):
        est._x_cols = list(blob["x_cols"])
    if blob.get("y_cols"):
        est._y_cols = list(blob["y_cols"])
    return est


# --- best_iteration までで推論する共通関数 ---
def _predict_with_best_iter(
    est: Any, Xf, feature_names: Optional[List[str]] = None
) -> np.ndarray:
    """Xf は DataFrame 推奨。やむを得ず ndarray の場合は feature_names を必ず渡す。"""
    booster = _booster_from_any(est)
    if booster is None:
        # Booster 取れない場合のフォールバック：
        # Xf が DataFrame なら DataFrame のまま渡す（to_numpyしない！）
        return est.predict(Xf)

    # --- ここから下はそのまま ---
    if isinstance(Xf, pd.DataFrame):
        dmat = xgb.DMatrix(Xf)
    else:
        bn = getattr(booster, "feature_names", None)
        use_names = feature_names or bn
        dmat = xgb.DMatrix(np.asarray(Xf), feature_names=use_names)

    best_it = _best_iteration_from_booster(booster)
    if isinstance(best_it, int):
        return booster.predict(dmat, iteration_range=(0, best_it + 1))
    return booster.predict(dmat)


# --- ランタイム（推論器） ---
@dataclass
class ResidualRuntime:
    estimators: List[XGBRegressor]
    y_cols: List[str]
    x_cols: List[str]

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        # x_cols に厳密に合わせた「DataFrame」を作る（ここで ndarray に落とさない）
        if isinstance(X, pd.DataFrame):
            Xf_df = X.reindex(columns=self.x_cols)
        else:
            # ndarray 入力でも列名つきに変換して DMatrix へ渡す
            Xf_df = pd.DataFrame(np.asarray(X), columns=self.x_cols)

        if len(self.estimators) == 1:
            y = _predict_with_best_iter(self.estimators[0], Xf_df)
            return np.asarray(y).reshape(-1, 1)

        preds = [_predict_with_best_iter(est, Xf_df) for est in self.estimators]
        return np.column_stack(preds)


# --- 読み込み ---
def load_residual_model(path: str) -> ResidualRuntime:
    payload = joblib.load(path)
    x_cols = list(payload["x_cols"])
    y_cols = list(payload["y_cols"])
    ests = [_rebuild_regressor_from_raw(b) for b in payload["estimators"]]
    return ResidualRuntime(estimators=ests, y_cols=y_cols, x_cols=x_cols)


def predict_full_period_with_residual_model(
    model: Any,
    X_full: pd.DataFrame,
    *,
    model_target_names: Sequence[str],  # 学習時 Y.columns の並び（必須）
    wanted_target_cols: Optional[Sequence[str]] = None,  # 返したい列（省略で全列）
    bl_prefix: str = "bl__",  # ← 互換性のために残すが使わない
    add_back_baseline: bool = True,  # ← 無視
    fill_missing_baseline: float = 0.0,  # ← 無視
    coerce_numeric: bool = True,
) -> dict:
    """
    （※今は残差を使わない）通常の X -> Y モデルで全期間を予測する。

    - model_target_names: 学習時ターゲット列名の順序（predict の列順と一致）
    - wanted_target_cols: 返却したい列（例: ["total_kwh__49-9"]）。None なら全列。

    戻り値:
        {
            "y_pred": DataFrame,      # 予測された Y（元スケール）
            "y_res_pred": DataFrame,  # 互換用。中身は y_pred と同じ。
        }
    """
    if not model_target_names:
        raise ValueError("model_target_names を渡してください。")

    # X 整形
    Xf = X_full.copy()
    if coerce_numeric:
        Xf = Xf.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # 予測: 保存読込後のランタイム or 学習直後モデルの両対応
    if isinstance(model, ResidualRuntime):
        # 列順はランタイム内で x_cols に揃う
        y_np = model.predict(Xf)
    else:
        # 学習直後のモデル（_x_cols / feature_names_in_ に合わせる）
        if hasattr(model, "_x_cols"):
            X_used = Xf.reindex(columns=list(model._x_cols), fill_value=0.0)
        elif hasattr(model, "feature_names_in_"):
            X_used = Xf.reindex(columns=list(model.feature_names_in_), fill_value=0.0)
        else:
            X_used = Xf

        y_np = _predict_with_best_iter(model, X_used)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

    # 出力次元チェック
    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)
    if y_np.shape[1] != len(model_target_names):
        raise ValueError(
            f"モデル出力数({y_np.shape[1]})と model_target_names({len(model_target_names)}) が不一致です。"
        )

    # DataFrame 化（全ターゲット）
    y_all = pd.DataFrame(y_np, index=Xf.index, columns=list(model_target_names))

    # 必要列だけ選択
    if wanted_target_cols is None:
        y_pred = y_all
    else:
        missing = [c for c in wanted_target_cols if c not in y_all.columns]
        if missing:
            raise KeyError(f"wanted_target_cols に未知列があります: {missing}")
        y_pred = y_all.loc[:, list(wanted_target_cols)]

    # ここでは baseline を足し戻さない（既に元スケールの Y を直接予測している前提）
    # 互換性のために y_res_pred も同じ中身で返す
    out = {
        "y_pred": y_pred,
    }
    return out
