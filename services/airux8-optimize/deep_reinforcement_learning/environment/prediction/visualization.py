import re

# === 置き換え：Feature Importance 抽出まわり（これだけ差し替えればOK） ===
from typing import Any, Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

_FIDX_RE = re.compile(r"^f(\d+)$")


def _booster_from_any(est: Any) -> Optional[xgb.Booster]:
    if isinstance(est, xgb.Booster):
        return est
    if hasattr(est, "get_booster"):
        try:
            b = est.get_booster()
            if isinstance(b, xgb.Booster):
                return b
        except Exception:
            pass
    for attr in ("booster_", "booster"):
        if hasattr(est, attr) and isinstance(getattr(est, attr), xgb.Booster):
            return getattr(est, attr)
    return None


def _importance_df_from_estimator(
    est: Any,
    feature_names: List[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    どの形式でも必ず feature_names 順に並べた DF を返す。
    - get_score のキーが f0,f1,... の「番号」形式でも
    - 実際の「列名」形式でもOK
    - どちらも出なければ feature_importances_ をフォールバック
    """
    booster = _booster_from_any(est)

    if booster is not None:
        raw = booster.get_score(importance_type=importance_type) or {}
        if raw:
            keys = list(raw.keys())
            # 1) キーが 'f123' の番号形式
            if all(_FIDX_RE.match(k) for k in keys):
                scores = np.zeros(len(feature_names), dtype=float)
                for k, v in raw.items():
                    i = int(_FIDX_RE.match(k).group(1))
                    if 0 <= i < len(scores):
                        scores[i] = float(v)
                df = pd.DataFrame({"feature": feature_names, "importance": scores})
                df = df.sort_values("importance", ascending=False).reset_index(
                    drop=True
                )
                return df
            # 2) キーが実列名
            else:
                s = pd.Series({str(k): float(v) for k, v in raw.items()})
                s = s.reindex(feature_names, fill_value=0.0)
                df = s.sort_values(ascending=False).reset_index()
                df.columns = ["feature", "importance"]
                return df

    # 3) フォールバック：sklearn API
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_, dtype=float)
        vals = vals[: len(feature_names)]
        df = pd.DataFrame({"feature": feature_names, "importance": vals})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    # 4) それでも出せない場合はゼロで返す（空防止）
    df = pd.DataFrame({"feature": feature_names, "importance": 0.0})
    return df


def _as_name_to_estimator_map(model: Any, y_cols: List[str]) -> Dict[str, Any]:
    if not hasattr(model, "estimators_"):
        name = y_cols[0] if len(y_cols) == 1 else "y0"
        return {name: model}
    ests = getattr(model, "estimators_")
    names = y_cols if len(ests) == len(y_cols) else [f"y{i}" for i in range(len(ests))]
    return dict(zip(names, ests))


def compute_importance_for_targets(
    res: Dict[str, Any],
    X_features: pd.DataFrame,
    targets: Optional[List[str]] = None,
    *,
    importance_type: str = "gain",
    top_n: Optional[int] = None,
) -> Dict[str, pd.DataFrame]:
    model = res["model"]
    y_cols: List[str] = list(res["Y_cols"])
    x_cols: List[str] = list(res.get("X_cols", list(X_features.columns)))

    name_to_est = _as_name_to_estimator_map(model, y_cols)
    tgt_list = targets or list(name_to_est.keys())

    out: Dict[str, pd.DataFrame] = {}
    for t in tgt_list:
        est = name_to_est[t]
        df = _importance_df_from_estimator(est, x_cols, importance_type=importance_type)
        if top_n is not None:
            df = df.head(top_n).copy()
        out[t] = df
    return out


def plot_feature_importance_for_targets_compat(
    res: Dict[str, Any],
    X_features: pd.DataFrame,
    targets: Optional[List[str]] = None,
    *,
    importance_type: str = "gain",
    top_n: int = 20,
    ncols: int = 2,
    figsize_per_panel=(6, 4),
):
    imps = compute_importance_for_targets(
        res, X_features, targets, importance_type=importance_type, top_n=top_n
    )
    names = list(imps.keys())
    n = len(names)
    ncols = max(1, ncols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )
    ax_list = axes.ravel()
    for i, t in enumerate(names):
        ax = ax_list[i]
        df = imps[t]
        ax.barh(df["feature"][::-1], df["importance"][::-1])
        ax.set_title(f"Feature Importance ({importance_type}) — {t}")
        ax.set_xlabel(importance_type)
        ax.set_ylabel("feature")
    for j in range(len(names), len(ax_list)):
        fig.delaxes(ax_list[j])
    fig.tight_layout()
    fig.show()


def plot_residual_metrics(res: dict, sort_by="R2", top_n=None, figsize=(14, 6)):
    """
    res["metrics_tr_res"], res["metrics_te_res"]（ともに index=target）を
    結合して、train/test を並べた棒グラフを描画します。

    sort_by: "R2" か "MAE"（テストがあればテスト側でソート、なければトレイン側）
    top_n : 上位だけ表示したい場合の件数（None なら全件）
    """

    mpl.rcParams["font.family"] = "Hiragino Sans"
    mpl.rcParams["axes.unicode_minus"] = False
    # 1) 取り出し＆結合
    m_tr = res.get("metrics_tr_res").copy()
    m_te = res.get("metrics_te_res")
    if m_te is None:
        # テストなしでも動くように空のDFを用意
        m_te = pd.DataFrame(index=m_tr.index, columns=m_tr.columns)

    # 列名を明確化して結合
    m_tr = m_tr.rename(columns={"R2": "R2_train", "MAE": "MAE_train"})
    m_te = m_te.rename(columns={"R2": "R2_test", "MAE": "MAE_test"})
    M = m_tr.join(m_te, how="outer")

    # 2) ソートキー作成（test優先、なければtrain）
    if sort_by.upper() == "R2":
        key = M["R2_test"].where(~M["R2_test"].isna(), M["R2_train"])
        ascending = False
    elif sort_by.upper() == "MAE":
        key = M["MAE_test"].where(~M["MAE_test"].isna(), M["MAE_train"])
        ascending = True
    else:
        raise ValueError("sort_by は 'R2' または 'MAE' を指定してください。")

    M = M.loc[key.sort_values(ascending=ascending).index]

    # 3) 上位だけ切り出し
    if isinstance(top_n, int) and top_n > 0:
        M = M.head(top_n)

    # 4) 棒グラフ（R2 と MAE を2枚）
    targets = M.index.to_list()
    x = np.arange(len(targets))
    width = 0.35

    # --- (A) R2 ---
    fig1, ax1 = plt.subplots(figsize=figsize)
    ax1.bar(x - width / 2, M["R2_train"], width, label="Train", alpha=0.9)
    ax1.bar(x + width / 2, M["R2_test"], width, label="Test", alpha=0.9)
    ax1.set_title("R2 by target (Train vs Test)")
    ax1.set_ylabel("R2")
    ax1.axhline(0.0, color="gray", linewidth=1, linestyle="--", alpha=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets, rotation=45, ha="right")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()

    # 値ラベル（見やすいように少数3桁）
    for i, v in enumerate(M["R2_train"]):
        if pd.notna(v):
            ax1.text(i - width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(M["R2_test"]):
        if pd.notna(v):
            ax1.text(i + width / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    # --- (B) MAE ---
    fig2, ax2 = plt.subplots(figsize=figsize)
    ax2.bar(x - width / 2, M["MAE_train"], width, label="Train", alpha=0.9)
    ax2.bar(x + width / 2, M["MAE_test"], width, label="Test", alpha=0.9)
    ax2.set_title("MAE by target (Train vs Test)")
    ax2.set_ylabel("MAE")
    ax2.set_xticks(x)
    ax2.set_xticklabels(targets, rotation=45, ha="right")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend()

    for i, v in enumerate(M["MAE_train"]):
        if pd.notna(v):
            ax2.text(i - width / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(M["MAE_test"]):
        if pd.notna(v):
            ax2.text(i + width / 2, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()
