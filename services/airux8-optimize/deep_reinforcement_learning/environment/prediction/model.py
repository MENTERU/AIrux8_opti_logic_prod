# ==== 安定保存・復元（完全差し替え） =========================================
# ==== 安定保存・復元（完全差し替え） =========================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
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


def _get_Y_res(Y_features) -> pd.DataFrame:
    """Y_features が dict なら Y_res を、DataFrame ならそれを残差として返す。"""
    if isinstance(Y_features, dict):
        return Y_features["Y_res"]
    if isinstance(Y_features, pd.DataFrame):
        return Y_features
    raise TypeError(
        "Y_features は dict（Y_res を含む）か DataFrame（残差のみ）を渡してください。"
    )


def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            out = df.copy()
            out.index = pd.to_datetime(out["datetime"])
            out = out.drop(columns=["datetime"])
        else:
            raise ValueError("DatetimeIndex か 'datetime' 列が必要です。")
    else:
        out = df.copy()
    out = out.sort_index()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    return out


def _timewise_block_split_last_months(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    *,
    test_last_months: int = 2,  # 検証=直近の「完全な」月を M ヶ月
    purge_days: int = 7,  # 検証開始の直前をパージして学習に入れない
) -> Dict[str, Any]:
    X = _ensure_dtindex(X)
    Y = _ensure_dtindex(Y)

    # 共通インデックス（外れ行は落とす）
    idx = X.index.intersection(Y.index).sort_values()
    X = X.reindex(idx)
    Y = Y.reindex(idx)

    if len(idx) == 0:
        raise ValueError("X と Y の共通インデックスが空です。")

    # === 完全な暦月を抽出 ===
    # 各月の最小・最大日付が、その月の月初～月末をすべて含んでいるかをチェック
    # ここでは「各月が1日0:00〜月末の同一頻度で埋まっている」まで厳密には見ず、
    # 「月の中に少なくとも1点ある」月を候補にし、最後に検証境界で欠損が出ないようにします。
    per = pd.PeriodIndex(idx, freq="M")
    months = pd.Index(per.astype(str))
    uniq_months = pd.Index(months.unique())

    # 直近の完全 M ヶ月を取りたいが、一般には「最後の M ヶ月」にデータ穴が混ざっていても
    # そのまま使えることが多いので、シンプルに末尾から M ヶ月をとる。
    if len(uniq_months) < test_last_months + 1:
        raise ValueError(
            "データ月数が不足しています。test_last_months を減らしてください。"
        )

    test_months = list(uniq_months[-test_last_months:].tolist())
    is_test = months.isin(test_months)

    # 検証期間の境界
    if not is_test.any():
        raise ValueError("検証月に該当するデータ点が見つかりません。")
    test_start = idx[is_test].min()
    test_end = idx[is_test].max()

    # パージ境界（検証直前を学習から除外）
    purge_cut = test_start - pd.Timedelta(days=purge_days)

    # 学習＝ purge_cut より前、かつ検証に含まれない
    is_train = (idx < purge_cut) & (~is_test)

    idx_tr = idx[is_train]
    idx_te = idx[is_test]

    if len(idx_tr) == 0:
        raise ValueError(
            "訓練データが空になりました。purge_days を小さくするか、検証月数を減らしてください。"
        )

    X_tr, Y_tr = X.loc[idx_tr], Y.loc[idx_tr]
    X_te, Y_te = X.loc[idx_te], Y.loc[idx_te]

    return dict(
        X_tr=X_tr,
        Y_tr=Y_tr,
        X_te=X_te,
        Y_te=Y_te,
        X_cols=list(X.columns),
        Y_cols=list(Y.columns),
        idx_tr=idx_tr,
        idx_te=idx_te,
        test_months=test_months,
        purge_cut=purge_cut,
        test_start=test_start,
        test_end=test_end,
    )


def _timewise_all_train(X: pd.DataFrame, Y: pd.DataFrame):
    X = _ensure_dtindex(X)
    Y = _ensure_dtindex(Y)
    idx = X.index.intersection(Y.index).sort_values()
    X = X.reindex(idx)
    Y = Y.reindex(idx)
    return dict(
        X_tr=X,
        Y_tr=Y,
        X_te=X.iloc[0:0],
        Y_te=Y.iloc[0:0],  # 空
        X_cols=list(X.columns),
        Y_cols=list(Y.columns),
        idx_tr=idx,
        idx_te=idx[:0],  # 空Index
    )


def train_xgb_multioutput(
    X_tr: pd.DataFrame | np.ndarray,
    Y_tr: pd.DataFrame | np.ndarray,
    *,
    xgb_params: Dict[str, Any],
    eval_set: Optional[
        List[Tuple[pd.DataFrame | np.ndarray, pd.DataFrame | np.ndarray]]
    ] = None,
    early_stopping_rounds: Optional[int] = None,
    verbose: bool = False,
):
    import numpy as np
    import xgboost as xgb

    # ---- 目的が複数か判定 ----
    if isinstance(Y_tr, pd.DataFrame):
        y_cols = list(Y_tr.columns)
        is_multi = len(y_cols) > 1
    else:
        Y_arr = np.asarray(Y_tr)
        is_multi = (Y_arr.ndim == 2) and (Y_arr.shape[1] > 1)
        y_cols = [f"y{i}" for i in range(Y_arr.shape[1])] if is_multi else ["y0"]

    # ---- 便利クラス（Booster を scikit-wind に寄せる）----
    class BoosterRegressorAdapter:
        def __init__(self, booster, feature_names=None):
            self.booster_ = booster
            self.feature_names_ = feature_names

        def predict(self, X):
            d = xgb.DMatrix(X, feature_names=self.feature_names_)
            return self.booster_.predict(
                d,
                iteration_range=(
                    (0, self.booster_.best_iteration + 1)
                    if hasattr(self.booster_, "best_iteration")
                    and self.booster_.best_iteration is not None
                    else None
                ),
            )

    # ---- ラッパーで早期終了を試み、ダメなら xgb.train にフォールバック ----
    def _fit_single(y_tr_vec, X_val=None, y_val_vec=None):
        # 1) try: XGBRegressor（新しい版）
        try:
            reg = xgb.XGBRegressor(**xgb_params)
            fit_kwargs = {"verbose": verbose}
            if X_val is not None and y_val_vec is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val_vec)]
                # try callbacks
                try:
                    cb = []
                    if early_stopping_rounds and early_stopping_rounds > 0:
                        cb.append(
                            xgb.callback.EarlyStopping(
                                rounds=early_stopping_rounds, save_best=True
                            )
                        )
                    if cb:
                        fit_kwargs["callbacks"] = cb
                except Exception:
                    # 古い版だと callbacks 不可 → 後段にフォールバック
                    raise TypeError("callbacks not supported")
            reg.fit(X_tr, y_tr_vec, **fit_kwargs)
            return reg
        except TypeError:
            # 2) fallback: xgb.train（古い版でも可）
            # パラメータ整備
            params = dict(xgb_params)  # shallow copy
            # sklearn ラッパーの名前と train の名前差を調整
            params.setdefault("objective", "reg:squarederror")
            if "eval_metric" not in params:
                params["eval_metric"] = "rmse"
            num_boost_round = int(params.pop("n_estimators", 100))
            # DMatrix
            dtrain = xgb.DMatrix(X_tr, label=y_tr_vec)
            evals = [(dtrain, "train")]
            if X_val is not None and y_val_vec is not None:
                dvalid = xgb.DMatrix(X_val, label=y_val_vec)
                evals.append((dvalid, "valid"))
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=num_boost_round,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds or 0,
                verbose_eval=False if not verbose else True,
            )
            return BoosterRegressorAdapter(booster)

    # ---- 単一目的 ----
    if not is_multi:
        X_val = Y_val = None
        if eval_set:
            X_val, Y_val = eval_set[0]
            # 1D へ
            if isinstance(Y_val, pd.DataFrame):
                Y_val = Y_val.iloc[:, 0].values
            else:
                Y_val = np.asarray(Y_val).reshape(-1)
        y_tr_vec = (
            Y_tr.values.reshape(-1)
            if isinstance(Y_tr, pd.DataFrame)
            else np.asarray(Y_tr).reshape(-1)
        )
        return _fit_single(y_tr_vec, X_val, Y_val)

    # ---- 複数目的（列ごとに学習）----
    X_val, Y_val = eval_set[0] if (eval_set and len(eval_set) > 0) else (None, None)
    estimators = []
    for j, col in enumerate(y_cols):
        y_tr_vec = (
            Y_tr[col].values
            if isinstance(Y_tr, pd.DataFrame)
            else np.asarray(Y_tr)[:, j]
        )
        y_val_vec = None
        if X_val is not None and Y_val is not None:
            y_val_vec = (
                Y_val[col].values
                if isinstance(Y_val, pd.DataFrame)
                else np.asarray(Y_val)[:, j]
            )
        est = _fit_single(y_tr_vec, X_val, y_val_vec)
        estimators.append(est)

    class MultiOutputXGB_Compat:
        def __init__(self, ests, cols):
            self.estimators_ = ests
            self.y_cols_ = list(cols)

        def predict(self, X):
            import numpy as np

            preds = [est.predict(X) for est in self.estimators_]
            return np.column_stack(preds)

    return MultiOutputXGB_Compat(estimators, y_cols)


def evaluate_predictions(Y_true, Y_pred, y_cols):
    """
    列ごとの R2 / MAE を返すユーティリティ。
    - Y_true, Y_pred は DataFrame/ndarray いずれも可
    - 1列の場合も 2D に揃えて評価
    - y_cols と列数がズレた場合は y_cols を推定して補正
    """
    # ndarray 化 & 2D に揃える
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)
    if Y_true.ndim == 1:
        Y_true = Y_true.reshape(-1, 1)
    if Y_pred.ndim == 1:
        Y_pred = Y_pred.reshape(-1, 1)

    # 列名整合
    if y_cols is None or len(y_cols) != Y_pred.shape[1]:
        y_cols = [f"y{i}" for i in range(Y_pred.shape[1])]

    rows = []
    for j, name in enumerate(y_cols):
        r2 = r2_score(Y_true[:, j], Y_pred[:, j])
        mae = mean_absolute_error(Y_true[:, j], Y_pred[:, j])
        rows.append((name, r2, mae))

    return pd.DataFrame(rows, columns=["target", "R2", "MAE"]).set_index("target")


def fit_predict_eval_xgb_residual_only(
    X_features: pd.DataFrame,
    Y_features,
    *,
    xgb_params: Optional[Dict[str, Any]] = None,
    split_mode: str = "last_months",
    test_last_months: int = 2,
    purge_days: int = 7,
    early_stopping_rounds: Optional[int] = 100,
    verbose: bool = False,
):
    xgb_params = xgb_params or {}

    Y_res = _get_Y_res(Y_features)
    X = _ensure_dtindex(X_features)

    if split_mode == "last_months":
        pack = _timewise_block_split_last_months(
            X, Y_res, test_last_months=test_last_months, purge_days=purge_days
        )
        eval_set = [(pack["X_te"], pack["Y_te"])] if len(pack["X_te"]) > 0 else None
    elif split_mode == "all_train":
        pack = _timewise_all_train(X, Y_res)
        eval_set = None
        early_stopping_rounds = None
    else:
        raise ValueError(f"unknown split_mode: {split_mode}")

    # ← eval_metric はここでは渡さない。xgb_params に入れておく。
    model = train_xgb_multioutput(
        pack["X_tr"],
        pack["Y_tr"],
        xgb_params=xgb_params,
        eval_set=eval_set,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )

    Y_res_hat_tr = model.predict(pack["X_tr"])
    metrics_tr = evaluate_predictions(pack["Y_tr"], Y_res_hat_tr, pack["Y_cols"])

    if len(pack["X_te"]) > 0:
        Y_res_hat_te = model.predict(pack["X_te"])
        metrics_te = evaluate_predictions(pack["Y_te"], Y_res_hat_te, pack["Y_cols"])
    else:
        Y_res_hat_te, metrics_te = None, None

    return {
        "model": model,
        "X_cols": pack["X_cols"],
        "Y_cols": pack["Y_cols"],
        "idx_tr": pack["idx_tr"],
        "idx_te": pack["idx_te"],
        "Y_res_hat_tr": Y_res_hat_tr,
        "Y_res_hat_te": Y_res_hat_te,
        "metrics_tr_res": metrics_tr,
        "metrics_te_res": metrics_te,
    }
