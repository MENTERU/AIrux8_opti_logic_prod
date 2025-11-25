from __future__ import annotations

import re
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ========= 共通ユーティリティ =========


def _ensure_dtindex(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame.index は DatetimeIndex 必須です。")
    out = df.sort_index()
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    return out


def make_time_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    s = index
    return pd.DataFrame(
        {
            "hour": s.hour,
            "month": s.month,
            "weekday": s.weekday,
            "is_weekend": (s.weekday >= 5).astype(int),
        },
        index=index,
    )


def pick_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    return [c for c in df.columns if c.startswith(prefix)]


def add_lagged_features(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    lags_hours=[1],
    use_freq_shift=False,
    prefix_fmt="lag{h}h__",
) -> pd.DataFrame:
    df = _ensure_dtindex(df)
    cols = list(cols)
    parts = []
    for h in lags_hours:
        lagged = (
            df[cols].shift(freq=pd.Timedelta(hours=h))
            if use_freq_shift
            else df[cols].shift(h)
        )
        lagged = lagged.add_prefix(prefix_fmt.format(h=h))
        parts.append(lagged)
    return pd.concat(parts, axis=1)


def _cut_head_for_baseline_and_lag(
    index: pd.DatetimeIndex, days_for_baseline: int, lags_hours: Iterable[int]
) -> pd.DatetimeIndex:
    max_lag_h = max(lags_hours) if lags_hours else 0
    cut_point = (
        index.min()
        + pd.Timedelta(days=days_for_baseline)
        + pd.Timedelta(hours=max_lag_h)
    )
    return index[index >= cut_point]


# ========= 列名 正規化/並べ替え（ビルダー互換のため） =========

_INDOOR_RAW = "Indoor Temp.__"
_INDOOR_OUT = "indoor_temp__"
_KWH_OUT = "total_kwh__"


def _normalize_raw_metric_name(name: str) -> str:
    s = str(name)
    m = re.match(r"^\s*Indoor\s*Temp\.?\s*__(.*)$", s, flags=re.IGNORECASE)
    if m:
        return f"indoor_temp__{m.group(1)}"
    m = re.match(r"^\s*Total\s*Kwh\.?\s*__(.*)$", s, flags=re.IGNORECASE)
    if m:
        return f"total_kwh__{m.group(1)}"
    # 既に正規化済み or 他の列は触らない
    return s


def _normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return None
    new_cols = [_normalize_raw_metric_name(c) for c in df.columns]
    # 衝突チェック（念のため）
    if len(set(new_cols)) != len(new_cols):
        raise ValueError("正規化後の列名が衝突しています。命名を見直してください。")
    out = df.copy()
    out.columns = new_cols
    return out


def _normalize_metric_names(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        new = c
        new = re.sub(r"^(?:bl__)+", "bl__", new)
        new = re.sub(r"^bl__" + re.escape(_INDOOR_RAW), "bl__" + _INDOOR_OUT, new)
        new = re.sub(
            r"^(lag\d+h__res__)" + re.escape(_INDOOR_RAW), r"\1" + _INDOOR_OUT, new
        )
        out.append(new)
    return out


def _natkey_indoor(unit: str) -> Tuple:
    m = re.match(r"^([A-Za-z]+)-(\d+)(.*)$", unit)
    if not m:
        return (unit, 0, "", 0)
    bld, num, tail = m.group(1), int(m.group(2)), m.group(3)
    m2 = re.match(r"^(南|北)?(\d+)?$", tail)
    if m2:
        direc = m2.group(1) or ""
        idx = int(m2.group(2)) if m2.group(2) else 0
    else:
        direc, idx = tail, 0
    direc_order = {"南": 0, "北": 1, "": 2}
    return (bld.upper(), num, direc_order.get(direc, 3), idx, tail)


def _natkey_kwh(unit: str) -> Tuple:
    m = re.match(r"^(\d+)-(\d+)$", unit)
    return (int(m.group(1)), int(m.group(2))) if m else (unit, 0)


def _derive_units_from_columns(
    cols: Iterable[str], *, metric_prefix_out: str
) -> List[str]:
    units = set()
    pat_bl = re.compile(r"^bl__" + re.escape(metric_prefix_out) + r"(.+)$")
    pat_lag = re.compile(r"^lag\d+h__res__" + re.escape(metric_prefix_out) + r"(.+)$")
    for c in cols:
        m = pat_bl.match(c) or pat_lag.match(c)
        if m:
            units.add(m.group(1))
    return list(units)


def _order_columns_like_builder(
    columns: List[str], *, lags_hours: Iterable[int]
) -> List[str]:
    cols = _normalize_metric_names(columns)
    indoor_units = sorted(
        _derive_units_from_columns(cols, metric_prefix_out=_INDOOR_OUT),
        key=_natkey_indoor,
    )
    kwh_units = sorted(
        _derive_units_from_columns(cols, metric_prefix_out=_KWH_OUT), key=_natkey_kwh
    )
    cols_set = set(cols)
    ordered: List[str] = []
    # 室温 BL
    for u in indoor_units:
        c = f"bl__{_INDOOR_OUT}{u}"
        if c in cols_set:
            ordered.append(c)
    # 室温 ラグ
    for h in sorted(set(map(int, lags_hours))):
        for u in indoor_units:
            c = f"lag{h}h__res__{_INDOOR_OUT}{u}"
            if c in cols_set:
                ordered.append(c)
    # kWh BL
    for u in kwh_units:
        c = f"bl__{_KWH_OUT}{u}"
        if c in cols_set:
            ordered.append(c)
    # kWh ラグ
    for h in sorted(set(map(int, lags_hours))):
        for u in kwh_units:
            c = f"lag{h}h__res__{_KWH_OUT}{u}"
            if c in cols_set:
                ordered.append(c)
    tail = [c for c in cols if c not in ordered]
    return ordered + tail


# ========= 制御から Duration / Diff1h を作る =========


def add_control_derived_features(
    base_df: pd.DataFrame,
    *,
    setT_prefix: str = "A/C Set Temperature__",
    mode_prefix: str = "A/C Mode__",
    wind_prefix: str = "A/C Fan Speed__",
    onoff_prefix: str = "A/C ON/OFF__",
) -> pd.DataFrame:
    """
    base_df に以下の特徴量を追加して返す:
      - 各室内機ごとに
        * Duration_ON__{unit}  : ON になってからの連続稼働時間 [ステップ数]
        * Duration_OFF__{unit} : OFF になってからの連続停止時間 [ステップ数]
        * Duration_Mode__{unit}: Mode が現在値のまま続いている時間 [ステップ数]
        * Duration_Fan__{unit} : Fan が現在値のまま続いている時間 [ステップ数]
        * Diff1h_SetT__{unit}  : 設定温度の 1時間差分 (現在 - 1時間前)
    """
    df = base_df.copy()
    new_cols: Dict[str, pd.Series] = {}

    # ---- ON/OFF 由来 (ON/OFF 継続時間) ----
    onoff_cols = pick_cols(df, onoff_prefix)
    for col in onoff_cols:
        unit = col.replace(onoff_prefix, "")  # 例: "A-25" など
        s = df[col]

        # 0/1 の ON フラグに正規化
        if s.dtype == object:
            on = s.astype(str).str.upper().map({"ON": 1, "OFF": 0})
        else:
            on = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
        on = (on > 0).astype(int)
        off = 1 - on

        # --- ON 連続時間 ---
        grp_on = (on.ne(on.shift()) & on.eq(1)).cumsum()
        on_runtime = on.groupby(grp_on).cumsum()
        on_runtime = on_runtime.where(on == 1, 0)
        new_cols[f"Duration_ON__{unit}"] = on_runtime

        # --- OFF 連続時間 ---
        grp_off = (off.ne(off.shift()) & off.eq(1)).cumsum()
        off_runtime = off.groupby(grp_off).cumsum()
        off_runtime = off_runtime.where(off == 1, 0)
        new_cols[f"Duration_OFF__{unit}"] = off_runtime

    # ---- Mode 由来 (同一モード継続時間) ----
    mode_cols = pick_cols(df, mode_prefix)
    for col in mode_cols:
        unit = col.replace(mode_prefix, "")
        val = pd.to_numeric(df[col], errors="coerce").fillna(0)

        grp = val.ne(val.shift()).cumsum()
        duration = pd.Series(1, index=df.index).groupby(grp).cumsum()
        new_cols[f"Duration_Mode__{unit}"] = duration

    # ---- Fan 由来 (同一風量継続時間) ----
    wind_cols = pick_cols(df, wind_prefix)
    for col in wind_cols:
        unit = col.replace(wind_prefix, "")
        val = pd.to_numeric(df[col], errors="coerce").fillna(0)

        grp = val.ne(val.shift()).cumsum()
        duration = pd.Series(1, index=df.index).groupby(grp).cumsum()
        new_cols[f"Duration_Fan__{unit}"] = duration

    # ---- 設定温度の 1時間差分 ----
    setT_cols = pick_cols(df, setT_prefix)
    for col in setT_cols:
        unit = col.replace(setT_prefix, "")
        temp = pd.to_numeric(df[col], errors="coerce")
        diff = (temp - temp.shift(1)).fillna(0.0)
        new_cols[f"Diff1h_SetT__{unit}"] = diff

    # まとめて結合（断片化を避ける）
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    return df


# ========= オフライン用: X を構築 =========
""" 学習時の特徴量を生成する関数 """


def build_features_residualized(
    base_df: pd.DataFrame,
    *,
    indoor_prefix="Indoor Temp.__",
    setT_prefix="A/C Set Temperature__",
    mode_prefix="A/C Mode__",
    wind_prefix="A/C Fan Speed__",
    onoff_prefix="A/C ON/OFF__",
    weather_cols: Optional[
        Iterable[str]
    ] = None,  # 例: ["Outdoor Temp.", "Outdoor Humidity", "Solar Radiation"]
    include_weather_raw: bool = False,  # ★ 天気は“元値のみ”入れるか（デフォルトは入れない）
    days_for_baseline: int = 8,
    baseline_fallback: float = 0.0,
    include_original_controls: bool = True,
    lags_hours=[1],  # “残差のラグ”に使う
    use_freq_shift=False,  # 欠番が多いとき True
    drop_initial_window=True,
) -> pd.DataFrame:
    """
    X = [時間特徴]
        + [室温: BL と 残差ラグ群 (BL は ON/OFF 条件付き)]
        + [室外機kWh: BL と 残差ラグ群 (BL は従来の同時刻平均)]
        + [天気は bl__/res__ を作らない（必要なら元値のみ）]
        + [操作量（原値 + Duration + Diff1h）]

    ※ 行数は base_df と一致させる（original_index を常に維持）
    """
    # index 整備
    df = _ensure_dtindex(base_df)

    # ★ 制御値からの派生特徴量を追加
    df = add_control_derived_features(
        df,
        setT_prefix=setT_prefix,
        mode_prefix=mode_prefix,
        wind_prefix=wind_prefix,
        onoff_prefix=onoff_prefix,
    )

    original_index = df.index
    parts: list[pd.DataFrame] = []

    # 1) 時間特徴
    tf = make_time_features(original_index)
    if not isinstance(tf.index, pd.DatetimeIndex):
        tf = tf.set_index(original_index)
    parts.append(tf.reindex(original_index))

    # 2) 室内温度: BL + 残差ラグ（BL は ON/OFF 条件付き）
    indoor_cols_raw = pick_cols(df, indoor_prefix)
    if indoor_cols_raw:
        # # 室温を "indoor_temp__" プレフィックスに揃える
        tmp_in = df[indoor_cols_raw].copy()
        indoor_out_cols = list(tmp_in.columns)

        # 室温ラグ特徴の作成（例: 1時間前の室温）
        # lags_hours は関数引数の lags_hours をそのまま使う
        lag_temp = add_lagged_features(
            tmp_in,
            indoor_out_cols,
            lags_hours=lags_hours,  # 例: [1] にしておくと 1h 前だけ
            use_freq_shift=use_freq_shift,
            prefix_fmt="lag{h}h__",
        ).reindex(original_index)

        parts.append(lag_temp)
    # 4) 天気: ★ bl__/res__ を作らない。必要なら“元値のみ”追加
    if include_weather_raw:
        if weather_cols is None:
            candidates = ["Outdoor Temp.", "Outdoor Humidity", "Solar Radiation"]
            weather_cols = [c for c in candidates if c in df.columns]
        if weather_cols:
            parts.append(df[list(weather_cols)].copy().reindex(original_index))

    # 5) 操作量（原値 + 派生特徴）
    if include_original_controls:
        control_cols: list[str] = []

        # 元の制御値 prefix
        control_prefixes = [
            setT_prefix,
            mode_prefix,
            wind_prefix,
            onoff_prefix,
        ]
        # 派生特徴の prefix（Duration / Diff1h）
        derived_prefixes = [
            "Duration_ON__",
            "Duration_OFF__",
            "Duration_Mode__",
            "Duration_Fan__",
            "Diff1h_SetT__",
        ]

        for p in control_prefixes + derived_prefixes:
            control_cols += pick_cols(df, p)

        if control_cols:
            parts.append(df[control_cols].copy().reindex(original_index))

    # 結合（行は落とさない）
    X = pd.concat(parts, axis=1)
    X = X.reindex(original_index)
    if drop_initial_window:
        kept_index = _cut_head_for_baseline_and_lag(
            original_index, days_for_baseline, lags_hours
        )
        X = X.loc[kept_index]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


""" 目的変数を前処理するコード """


def build_targets_residualized(
    base_df: pd.DataFrame,
    *,
    odu_prefix: str = "total_kwh__",
    indoor_prefix: str = "Indoor Temp.__",
    onoff_prefix: str = "A/C ON/OFF__",
    days_for_baseline: int = 8,
    baseline_fallback: float = 0.0,
    out_indoor_prefix: str = "indoor_temp__",
    fillna_value: float | None = None,  # 欠損を0などで埋めたいときに指定
    drop_initial_window=True,
) -> pd.DataFrame:
    """
    返り値: Y_res = 元値 - baseline（行数は base_df と一致させる）
      - 室内温度: 同時刻 + 同一 ON/OFF 状態の baseline
      - kWh     : 同時刻平均 baseline
    列: total_kwh__*, indoor_temp__*
    """
    # --- 1) index整備 ---
    if "_ensure_dtindex" in globals():
        try:
            # 新しい引数に対応している実装向けの呼び出し（存在しなければ TypeError）
            df = _ensure_dtindex(base_df, duplicate_strategy="keep")  # type: ignore
        except TypeError:
            df = _ensure_dtindex(base_df)
    else:
        if not isinstance(base_df.index, pd.DatetimeIndex):
            raise ValueError("base_df.index は DatetimeIndex 必須です。")
        df = base_df.sort_index()

    original_index = df.index

    # --- 2) 出力列の収集 ---
    y_odu_cols = pick_cols(df, odu_prefix)
    y_ind_cols_raw = pick_cols(df, indoor_prefix)
    y_ind_cols_map = {
        c: c.replace(indoor_prefix, out_indoor_prefix) for c in y_ind_cols_raw
    }
    indoor_out_cols = list(y_ind_cols_map.values())

    # --- 3) 元ターゲット（出力名で統一） ---
    Y_raw = pd.DataFrame(index=original_index)
    if y_odu_cols:
        Y_raw[y_odu_cols] = df[y_odu_cols].reindex(original_index)
    if y_ind_cols_raw:
        Y_raw[indoor_out_cols] = (
            df[y_ind_cols_raw].rename(columns=y_ind_cols_map).reindex(original_index)
        )

    if not Y_raw.columns.tolist():
        Y_res = pd.DataFrame(index=original_index)
        return Y_res if fillna_value is None else Y_res.fillna(fillna_value)
    return Y_raw


# =========================================
# ラグ専用 DP
# =========================================


class _SameTimeDP:
    """
    - baseline / residual は持たず、
    - cols に対してラグ特徴だけを出すクラス。
      use_freq_shift=False → 行ベースのラグ（shift(n行)）
      use_freq_shift=True  → 時刻ベースで t-h 時間前を raw_by_ts から取る
    """

    def __init__(
        self,
        cols: List[str],
        *,
        days: int,
        fallback: float,
        lags_hours: Iterable[int],
        use_freq_shift: bool,
        lag_prefix_fmt: str,
    ):
        self.cols = list(cols)  # 例: "indoor_temp__A-25", "total_kwh__A-25"
        self.col_index: Dict[str, int] = {c: i for i, c in enumerate(self.cols)}
        self.n = len(self.cols)

        # days / fallback は旧設計との互換のため保持しているが、
        # ここではラグ専用なので実質未使用。
        self.days = int(days)
        self.fallback = float(fallback)

        self.lags = sorted(set(int(h) for h in (lags_hours or [])))
        self.max_lag = max(self.lags) if self.lags else 0
        self.use_freq_shift = bool(use_freq_shift)
        self.lag_prefix_fmt = lag_prefix_fmt

        # 履歴
        self.raw_by_ts: Dict[pd.Timestamp, np.ndarray] = {}
        if not self.use_freq_shift:
            # 行ベースラグ用：単純に直近 max_lag 行分だけ保持
            self.row_hist: Deque[np.ndarray] = deque(
                maxlen=self.max_lag if self.max_lag > 0 else None
            )
        else:
            # 時刻ベースラグ用：row_hist は不要
            self.row_hist = deque(maxlen=0)

    # ---- 内部 util ----

    def _normalize_ts(self, ts: pd.Timestamp) -> pd.Timestamp:
        return ts.tz_convert(None) if ts.tz is not None else ts

    def _make_output_columns(self) -> List[str]:
        cols: List[str] = []
        for h in self.lags:
            for c in self.cols:
                cols.append(f"{self.lag_prefix_fmt.format(h=h)}{c}")
        return cols

    def _compute_lag_vec(self, ts: pd.Timestamp) -> np.ndarray:
        """
        現在の内部履歴から、ts に対するラグベクトルを作る。
        """
        if not self.lags:
            return np.empty(0, float)

        vecs: List[np.ndarray] = []
        if self.use_freq_shift:
            t = self._normalize_ts(ts)
            for h in self.lags:
                prev_ts = t - pd.Timedelta(hours=h)
                v = self.raw_by_ts.get(prev_ts)
                if v is None:
                    v = np.full(self.n, np.nan)
                vecs.append(v)
        else:
            # 行ベース: 直近 max_lag 行だけ hist にある前提
            for h in self.lags:
                if len(self.row_hist) >= h:
                    v = list(self.row_hist)[-h]
                else:
                    v = np.full(self.n, np.nan)
                vecs.append(v)

        return np.concatenate(vecs, axis=0)

    def _update_state(self, ts: pd.Timestamp, row: np.ndarray) -> None:
        t = self._normalize_ts(ts)
        r = np.asarray(row, dtype=float).copy()
        self.raw_by_ts[t] = r
        if not self.use_freq_shift:
            self.row_hist.append(r)

    def grow_columns(self, new_cols: List[str]):
        """
        既存の履歴に対して nan を埋めて列を増やす。
        """
        add = [c for c in new_cols if c not in self.col_index]
        if not add:
            return
        k = len(add)

        # raw_by_ts 内の各行を拡張
        for ts, v in list(self.raw_by_ts.items()):
            self.raw_by_ts[ts] = np.concatenate([v, np.full(k, np.nan)])

        # 行ベース履歴も拡張
        if self.row_hist:
            new_hist: Deque[np.ndarray] = deque(maxlen=self.row_hist.maxlen)
            for v in self.row_hist:
                new_hist.append(np.concatenate([v, np.full(k, np.nan)]))
            self.row_hist = new_hist

        base = self.n
        for i, c in enumerate(add):
            self.col_index[c] = base + i
        self.cols.extend(add)
        self.n += k

    # ---- public API ----

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df に対してラグ特徴を計算しつつ、内部履歴も初期化する。
        """
        df = _ensure_dtindex(df)
        use_cols = [c for c in self.cols if c in df.columns]
        X = df[use_cols].reindex(columns=self.cols, fill_value=np.nan).to_numpy(float)

        out_cols = self._make_output_columns()
        out = np.empty((len(df), len(out_cols)), float)

        for i, (ts, row) in enumerate(zip(df.index, X)):
            lag_vec = self._compute_lag_vec(ts)
            out[i, :] = lag_vec
            # 自分自身を履歴に追加
            self._update_state(ts, row)

        return pd.DataFrame(out, index=df.index, columns=out_cols)

    def transform_new(
        self, df: pd.DataFrame, *, allow_growth: bool = False, mutate: bool = True
    ) -> pd.DataFrame:
        """
        新しい df に対してラグ特徴を計算。
        mutate=True のときは内部履歴も前進させる。
        df が空（列なし）のときは、「過去の履歴だけを使って未来 ts のラグ」を出せる。
        """
        df = _ensure_dtindex(df)

        if allow_growth:
            incoming = list(df.columns)
            self.grow_columns([c for c in incoming if c not in self.col_index])

        if df.columns.size > 0:
            use_cols = [c for c in self.cols if c in df.columns]
            X = (
                df[use_cols]
                .reindex(columns=self.cols, fill_value=np.nan)
                .to_numpy(float)
            )
        else:
            X = np.full((len(df), self.n), np.nan, float)

        out_cols = self._make_output_columns()
        out = np.empty((len(df), len(out_cols)), float)

        for i, (ts, row) in enumerate(zip(df.index, X)):
            lag_vec = self._compute_lag_vec(ts)
            out[i, :] = lag_vec

            if mutate:
                self._update_state(ts, row)

        return pd.DataFrame(out, index=df.index, columns=out_cols)


class ResidualFeatureDP:
    """
    名前はそのままですが、役割は「ラグ DP 管理」に縮退。
      - _collect_cols で DP 対象列を決める
      - fit/transform は DP に投げるだけ
    """

    def __init__(
        self,
        *,
        days: int = 7,
        baseline_fallback: float = 0.0,
        lags_hours: Iterable[int] = (1,),
        use_freq_shift: bool = False,
        lag_prefix_fmt: str = "lag{h}h__",
    ):
        self.days = int(days)
        self.fallback = float(baseline_fallback)
        self.lags = list(lags_hours)
        self.use_freq_shift = bool(use_freq_shift)
        self.lag_prefix_fmt = lag_prefix_fmt
        self.dp: Optional[_SameTimeDP] = None

    def _collect_cols(self, df: pd.DataFrame) -> List[str]:
        # ここで DP 対象列を制御
        indoor = pick_cols(df, _INDOOR_OUT)
        # kWh もラグにしたければ以下を追加
        # kwh = pick_cols(df, _KWH_OUT)
        # return indoor + kwh
        return indoor

    def fit(self, base_df: pd.DataFrame) -> pd.DataFrame:
        base_df = _ensure_dtindex(_normalize_raw_columns(base_df))
        cols = self._collect_cols(base_df)
        self.dp = _SameTimeDP(
            cols=cols,
            days=self.days,
            fallback=self.fallback,
            lags_hours=self.lags,
            use_freq_shift=self.use_freq_shift,
            lag_prefix_fmt=self.lag_prefix_fmt,
        )
        raw = self.dp.fit_transform(base_df[cols])

        # 互換用に正規化／並び替え（定義されていれば）
        raw.columns = _normalize_metric_names(list(raw.columns))
        raw = raw[_order_columns_like_builder(list(raw.columns), lags_hours=self.lags)]
        return raw

    def transform(
        self, new_df: pd.DataFrame, *, allow_growth: bool = True
    ) -> pd.DataFrame:
        if self.dp is None:
            raise RuntimeError("fit() を先に呼んでください。")
        new_df = _ensure_dtindex(_normalize_raw_columns(new_df))
        cols = self._collect_cols(new_df)
        out = self.dp.transform_new(new_df[cols], allow_growth=allow_growth)

        out.columns = _normalize_metric_names(list(out.columns))
        out = out[_order_columns_like_builder(list(out.columns), lags_hours=self.lags)]
        return out


# =========================================
# 共通アセンブリ（ラグ + 時間 + 天気 + 制御）
# =========================================


def _assemble_features_common(
    *,
    index: pd.DatetimeIndex,
    dp: _SameTimeDP,
    lags_hours: Iterable[int],
    include_weather_raw: bool,
    include_original_controls: bool,
    weather_info: Optional[pd.DataFrame],
    control_values: Optional[pd.DataFrame],
    drop_initial_window: bool = False,
    days_for_baseline: int = 7,
    finalize_numeric: bool = True,
    return_baseline: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    DP 由来の特徴量は「ラグ列」だけ。
    baseline (bl__) は作らないので、Y_baseline は空 DataFrame を返す。
    """
    ti = pd.DatetimeIndex(index)

    # 1) 時間特徴
    tf = make_time_features(ti)

    # 2) DP のラグ群（状態は更新しない）
    empty = pd.DataFrame(index=ti)
    res_feats = dp.transform_new(empty, allow_growth=False, mutate=False)

    res_feats.columns = _normalize_metric_names(list(res_feats.columns))
    res_feats = res_feats[
        _order_columns_like_builder(list(res_feats.columns), lags_hours=lags_hours)
    ]

    # ベースライン: 今回は使わないので空
    Y_baseline = pd.DataFrame(index=ti)

    # 3) 追加の天気/操作量
    parts: list[pd.DataFrame] = [tf, res_feats]
    if include_weather_raw and (weather_info is not None):
        parts.append(_ensure_dtindex(weather_info).reindex(ti))
    if include_original_controls and (control_values is not None):
        parts.append(_ensure_dtindex(control_values).reindex(ti))

    X = pd.concat(parts, axis=1)

    # 4) 列順を時間 → DPラグ → それ以外
    time_cols = [
        c for c in ["hour", "month", "weekday", "is_weekend"] if c in X.columns
    ]
    dp_cols = [c for c in res_feats.columns if c in X.columns]
    tail_cols = [c for c in X.columns if c not in time_cols + dp_cols]
    X = X[time_cols + dp_cols + tail_cols]

    # 5) 初期ウィンドウ（days + max_lag）を落とす（必要なら）
    if drop_initial_window:
        kept_index = _cut_head_for_baseline_and_lag(ti, days_for_baseline, lags_hours)
        X = X.loc[kept_index]
        Y_baseline = Y_baseline.reindex(X.index)

    # 6) 数値化・欠損処理
    if finalize_numeric:
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return (X, Y_baseline) if return_baseline else X


# =========================================
# 最終統合クラス: InputDataBuilderDP（オンライン用）
# =========================================


class InputDataBuilderDP:
    """
    - ラグ DP で室内温度ラグ状態を持つ（オンライン予測用）
    - fit(base_df) が返す特徴量 X は build_features_residualized と同じ構造：
      X = [時間特徴]
          + [室内温度ラグ (indoor_temp__*, lag{h}h__…)]
          + [天気の生値 (任意)]
          + [操作量（原値 + Duration + Diff1h）]
    """

    def __init__(
        self,
        *,
        days: int = 7,
        baseline_fallback: float = 0.0,
        lags_hours: Iterable[int] = (1,),
        use_freq_shift: bool = False,
        include_weather_raw: bool = True,
        include_original_controls: bool = True,
        lag_prefix_fmt: str = "lag{h}h__",
    ):
        # ラグ DP
        self.include_weather_raw = bool(include_weather_raw)
        self.include_original_controls = bool(include_original_controls)
        self.res_dp = ResidualFeatureDP(
            days=days,
            baseline_fallback=baseline_fallback,
            lags_hours=lags_hours,
            use_freq_shift=use_freq_shift,
            lag_prefix_fmt=lag_prefix_fmt,
        )
        self._lags = list(lags_hours)
        self._days = int(days)
        self._use_freq_shift = bool(use_freq_shift)

        # 制御・Duration 用の内部状態
        self._setT_prefix = "A/C Set Temperature__"
        self._mode_prefix = "A/C Mode__"
        self._wind_prefix = "A/C Fan Speed__"
        self._onoff_prefix = "A/C ON/OFF__"
        self._derived_prefixes = [
            "Duration_ON__",
            "Duration_OFF__",
            "Duration_Mode__",
            "Duration_Fan__",
            "Diff1h_SetT__",
        ]

        self._control_cols_original: list[str] = []  # 元の制御列
        self._control_cols_derived: list[str] = []  # Duration / Diff1h
        self._control_cols_all: list[str] = []  # 上2つの和
        self._control_state_last: Optional[pd.Series] = None  # 直近1ステップ分

        # オンライン履歴ポインタ
        self._online_started = False
        self._history_index_last: Optional[pd.Timestamp] = None

    # ---- 内部ヘルパ：履歴から制御状態を初期化 ----
    def _init_control_state_from_history(self, df: pd.DataFrame) -> None:
        """
        過去の実績時系列 df から Duration 等を計算し、
        その最終行を self._control_state_last として保持する。
        """
        if df is None or len(df) == 0:
            self._control_state_last = None
            self._control_cols_original = []
            self._control_cols_derived = []
            self._control_cols_all = []
            return

        df = _ensure_dtindex(df)

        # 元制御 + Duration を一旦フルで計算
        df_with_der = add_control_derived_features(
            df,
            setT_prefix=self._setT_prefix,
            mode_prefix=self._mode_prefix,
            wind_prefix=self._wind_prefix,
            onoff_prefix=self._onoff_prefix,
        )

        # 元の制御列
        control_prefixes = [
            self._setT_prefix,
            self._mode_prefix,
            self._wind_prefix,
            self._onoff_prefix,
        ]
        ctrl_orig: list[str] = []
        for p in control_prefixes:
            ctrl_orig += pick_cols(df_with_der, p)

        # Duration/Diff1h 列
        ctrl_der: list[str] = []
        for p in self._derived_prefixes:
            ctrl_der += pick_cols(df_with_der, p)

        self._control_cols_original = ctrl_orig
        self._control_cols_derived = ctrl_der
        self._control_cols_all = ctrl_orig + ctrl_der

        if not self._control_cols_all:
            self._control_state_last = None
            return

        # 直近1ステップ分の状態
        last_row = df_with_der.iloc[-1]
        self._control_state_last = last_row[self._control_cols_all].copy()

    def _update_control_state_step(self, ctrl_row: pd.Series) -> pd.Series:
        """
        単一時刻の制御値（元制御のみ）から、
        直近状態 self._control_state_last を使って Duration_* / Diff1h_SetT__ を
        手計算で更新し、「元制御＋Duration/ Diff1h」を含む1行の Series を返す。
        """

        # --- 初回: まだ state が無いときは、add_control_derived_features で初期化してOK ---
        if self._control_state_last is None:
            df0 = pd.DataFrame([ctrl_row])
            df_with_der = add_control_derived_features(
                df0,
                setT_prefix=self._setT_prefix,
                mode_prefix=self._mode_prefix,
                wind_prefix=self._wind_prefix,
                onoff_prefix=self._onoff_prefix,
            )
            # 列リストもここで確定
            control_prefixes = [
                self._setT_prefix,
                self._mode_prefix,
                self._wind_prefix,
                self._onoff_prefix,
            ]
            ctrl_orig: list[str] = []
            for p in control_prefixes:
                ctrl_orig += pick_cols(df_with_der, p)

            ctrl_der: list[str] = []
            for p in self._derived_prefixes:
                ctrl_der += pick_cols(df_with_der, p)

            self._control_cols_original = ctrl_orig
            self._control_cols_derived = ctrl_der
            self._control_cols_all = ctrl_orig + ctrl_der

            self._control_state_last = df_with_der.iloc[-1][
                self._control_cols_all
            ].copy()
            return self._control_state_last.copy()

        # --- 2回目以降: 自前で Duration / Diff を更新していく ---
        prev = self._control_state_last
        # 元制御列を揃える（欠けている列は前回値で埋める）
        cur_orig = ctrl_row.reindex(
            self._control_cols_original
        ).copy()  # index が足りないと NaN

        new_state: Dict[str, float] = {}

        # まず元制御列を new_state にセット（NaN は前回値で補完）
        for col in self._control_cols_original:
            v = cur_orig.get(col)
            if pd.isna(v):
                v = prev.get(col, np.nan)
            new_state[col] = v

        # ---- ON/OFF 由来の Duration_ON / Duration_OFF ----
        for col in self._control_cols_original:
            if not col.startswith(self._onoff_prefix):
                continue
            unit = col.replace(self._onoff_prefix, "")

            on_col = col
            dur_on_col = f"Duration_ON__{unit}"
            dur_off_col = f"Duration_OFF__{unit}"

            prev_on_raw = prev.get(on_col, 0)
            prev_on = 1 if pd.notna(prev_on_raw) and prev_on_raw > 0 else 0

            cur_on_raw = new_state[on_col]
            cur_on = 1 if pd.notna(cur_on_raw) and cur_on_raw > 0 else 0

            prev_dur_on = prev.get(dur_on_col, 0) or 0
            prev_dur_off = prev.get(dur_off_col, 0) or 0

            if cur_on == 1:
                # ON 継続 or 新たに ON
                dur_on = prev_dur_on + 1 if prev_on == 1 else 1
                dur_off = 0
            else:
                # OFF 継続 or 新たに OFF
                dur_off = prev_dur_off + 1 if prev_on == 0 else 1
                dur_on = 0

            new_state[dur_on_col] = float(dur_on)
            new_state[dur_off_col] = float(dur_off)

        # ---- Mode 由来 Duration_Mode ----
        for col in self._control_cols_original:
            if not col.startswith(self._mode_prefix):
                continue
            unit = col.replace(self._mode_prefix, "")
            mode_col = col
            dur_mode_col = f"Duration_Mode__{unit}"

            prev_mode = prev.get(mode_col)
            cur_mode = new_state[mode_col]

            prev_dur_mode = prev.get(dur_mode_col, 0) or 0

            if pd.notna(prev_mode) and pd.notna(cur_mode) and prev_mode == cur_mode:
                dur_mode = prev_dur_mode + 1
            else:
                dur_mode = 1  # モードが変わった / 初回

            new_state[dur_mode_col] = float(dur_mode)

        # ---- Fan 由来 Duration_Fan ----
        for col in self._control_cols_original:
            if not col.startswith(self._wind_prefix):
                continue
            unit = col.replace(self._wind_prefix, "")
            fan_col = col
            dur_fan_col = f"Duration_Fan__{unit}"

            prev_fan = prev.get(fan_col)
            cur_fan = new_state[fan_col]

            prev_dur_fan = prev.get(dur_fan_col, 0) or 0

            if pd.notna(prev_fan) and pd.notna(cur_fan) and prev_fan == cur_fan:
                dur_fan = prev_dur_fan + 1
            else:
                dur_fan = 1

            new_state[dur_fan_col] = float(dur_fan)

        # ---- 設定温度の 1時間差分 Diff1h_SetT ----
        for col in self._control_cols_original:
            if not col.startswith(self._setT_prefix):
                continue
            unit = col.replace(self._setT_prefix, "")
            setT_col = col
            diff_col = f"Diff1h_SetT__{unit}"

            prev_T = prev.get(setT_col)
            cur_T = new_state[setT_col]

            if pd.notna(prev_T) and pd.notna(cur_T):
                diff = float(cur_T) - float(prev_T)
            else:
                diff = 0.0

            new_state[diff_col] = diff

        # --- state を Series にまとめて保存 ---
        # 既知の derived 列がまだ無ければここで補完
        if not self._control_cols_derived:
            ctrl_der: list[str] = []
            for p in self._derived_prefixes:
                ctrl_der += [c for c in new_state.keys() if c.startswith(p)]
            self._control_cols_derived = ctrl_der
            self._control_cols_all = (
                self._control_cols_original + self._control_cols_derived
            )

        # 既存の列順で reindex
        s = pd.Series(new_state)
        s = s.reindex(self._control_cols_all).fillna(0.0)

        self._control_state_last = s.copy()
        return s.copy()

    # ---- fit: ラグ DP 初期化 + 制御状態初期化 + オフライン X 構築 ----
    def fit(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """
        base_df から訓練用の特徴量 X を構築して返す。
        返り値 X は build_features_residualized(base_df, ...) と
        同じ構造／値になる。
        """
        # 1) ラグ DP を初期化（オンラインで使う）
        _ = self.res_dp.fit(_normalize_raw_columns(base_df))

        # 2) 制御 + Duration の内部状態を履歴から初期化
        self._init_control_state_from_history(base_df)

        # 3) オフライン用の特徴量は build_features_residualized に任せる
        X = build_features_residualized(
            base_df,
            indoor_prefix="Indoor Temp.__",
            setT_prefix=self._setT_prefix,
            mode_prefix=self._mode_prefix,
            wind_prefix=self._wind_prefix,
            onoff_prefix=self._onoff_prefix,
            weather_cols=None,
            include_weather_raw=self.include_weather_raw,
            days_for_baseline=self._days,
            baseline_fallback=self.res_dp.fallback,
            include_original_controls=self.include_original_controls,
            lags_hours=self._lags,
            use_freq_shift=self._use_freq_shift,
            drop_initial_window=True,
        )

        # オンライン履歴ポインタもリセット
        self._online_started = False
        self._history_index_last = None

        return X

    def begin_online(self, initial_df: pd.DataFrame | None = None) -> None:
        """
        オンライン履歴ポインタをセット。initial_df を渡せば、その末尾を「直近実績」に。
        制御 + Duration の内部状態も initial_df から再初期化する。
        """
        self._online_started = True
        if initial_df is not None and len(initial_df) > 0:
            ini = _ensure_dtindex(initial_df)
            self._history_index_last = ini.index[-1]
            self._init_control_state_from_history(ini)
        else:
            self._history_index_last = None

    def reset_online(self) -> None:
        """オンライン履歴ポインタだけクリア（DP自体のfit状態は維持）。"""
        self._online_started = False
        self._history_index_last = None
        # 制御状態は明示的には消さない（必要なら begin_online で上書き）

    def make_input_next(
        self,
        index: pd.DatetimeIndex | list[pd.Timestamp] | pd.Timestamp,
        weather_info: Optional[pd.DataFrame] = None,
        control_values: Optional[pd.DataFrame] = None,
        *,
        return_baseline: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        次時刻（複数可）の特徴量を生成。ラグ DP の状態は更新しない（look-ahead）。
        ただし制御の Duration / Diff1h については、
        内部状態 self._control_state_last を使ってステップ毎に更新する。
        """
        if self.res_dp.dp is None:
            raise RuntimeError("fit(base_df) の後に呼んでください。")

        ti = (
            pd.DatetimeIndex([index])
            if isinstance(index, pd.Timestamp)
            else pd.DatetimeIndex(index)
        )

        # === 制御の Duration / Diff1h をオンラインで更新 ===
        control_df_for_assemble: Optional[pd.DataFrame] = None
        if control_values is not None and self.include_original_controls:
            cv = _ensure_dtindex(control_values).reindex(ti)

            rows: list[pd.Series] = []
            for ts in ti:
                row_ctrl = cv.loc[ts]
                updated = self._update_control_state_step(row_ctrl)
                rows.append(updated)

            control_df_for_assemble = pd.DataFrame(rows, index=ti)
        else:
            control_df_for_assemble = None

        # === 共通アセンブリ（時間特徴 + DPラグ + 天気 + 制御） ===
        out = _assemble_features_common(
            index=ti,
            dp=self.res_dp.dp,
            lags_hours=self._lags,
            include_weather_raw=self.include_weather_raw,
            include_original_controls=self.include_original_controls,
            weather_info=weather_info,
            control_values=control_df_for_assemble,
            drop_initial_window=False,
            days_for_baseline=self._days,
            finalize_numeric=True,
            return_baseline=return_baseline,
        )
        return out

    def accumulate_actuals(self, y_actual: pd.DataFrame) -> None:
        """
        実績データを取り込み、DP内部のラグ状態だけを前進させる（破壊的更新）。

        - DP（室温ラグなど）は、indoor_temp__* など正規化済みカラムに対して更新。
        - 制御（SetT/Mode/Fan/ONOFF → Duration_* / Diff1h_SetT）はここでは更新しない。
        （Duration / Diff1h は make_input_next(...) 側だけで更新する設計）
        """
        if self.res_dp.dp is None:
            raise RuntimeError("fit(base_df) の後に呼んでください。")

        ya_raw = _ensure_dtindex(y_actual)
        if len(ya_raw) == 0:
            return

        # 時系列順チェック（過去に戻らないように）
        if (
            self._history_index_last is not None
            and ya_raw.index[0] <= self._history_index_last
        ):
            raise ValueError(
                f"accumulate_actuals の index は {self._history_index_last} より後である必要があります。"
            )

        # --- DP（ラグ）用: 室温などの列を正規化して DP に食わせる ---
        ya_norm = _normalize_raw_columns(ya_raw)

        # DP が管理している列だけ抜き出す
        dp_cols = [c for c in (self.res_dp.dp.cols) if c in ya_norm.columns]
        if dp_cols:
            ya_dp = ya_norm[dp_cols]
            # mutate=True なので DP 内部の履歴（= 1h_lag の元データ）が更新される
            _ = self.res_dp.dp.transform_new(ya_dp, allow_growth=True, mutate=True)

        # 最後に index を更新
        self._history_index_last = ya_raw.index[-1]
