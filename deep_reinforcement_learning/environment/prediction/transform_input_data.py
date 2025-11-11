from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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


# ========= レガシー版: 週平均BL + 残差ラグ（互換用） =========


def compute_same_time_baseline(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    days: int = 7,
    fallback: float = 0.0,
) -> pd.DataFrame:
    df = _ensure_dtindex(df)
    cols = list(cols)
    shifted_list = [
        df[cols].shift(freq=pd.Timedelta(days=k)) for k in range(1, days + 1)
    ]
    sum_vals, count_vals = None, None
    for s in shifted_list:
        sum_vals = s if sum_vals is None else sum_vals.add(s, fill_value=0.0)
        c = s.notna().astype(int)
        count_vals = c if count_vals is None else count_vals.add(c, fill_value=0)
    baseline = sum_vals.divide(count_vals).where(count_vals > 0, other=float(fallback))
    return baseline


def add_baseline_and_residual(
    df: pd.DataFrame,
    cols: Iterable[str],
    *,
    days: int = 7,
    fallback: float = 0.0,
    bl_prefix: str = "bl__",
    res_prefix: str = "res__",
    include_original: bool = True,
) -> pd.DataFrame:
    cols = list(cols)
    bl = compute_same_time_baseline(df, cols, days=days, fallback=fallback)
    res = df[cols] - bl
    out = pd.DataFrame(index=df.index)
    if include_original:
        out[cols] = df[cols]
    out[[f"{bl_prefix}{c}" for c in cols]] = bl
    out[[f"{res_prefix}{c}" for c in cols]] = res
    return out


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


def build_features_residualized(
    base_df: pd.DataFrame,
    *,
    indoor_prefix="Indoor Temp.__",
    setT_prefix="A/C Set Temperature__",
    mode_prefix="A/C Mode__",
    wind_prefix="A/C Fan Speed__",
    onoff_prefix="A/C ON/OFF__",
    weather_cols: Optional[Iterable[str]] = None,
    include_weather_raw: bool = False,
    days_for_baseline: int = 7,
    baseline_fallback: float = 0.0,
    include_original_controls: bool = True,
    lags_hours=[1],
    use_freq_shift=False,
    drop_initial_window=True,
) -> pd.DataFrame:
    """
    既存互換のレガシー出力（「同時刻・過去N日平均」+ 残差ラグ）を返す。
    ※ make_input_data のカラム規約と一致
    """
    df = _ensure_dtindex(base_df)
    original_index = df.index
    parts: list[pd.DataFrame] = []

    # 1) 時間特徴
    tf = make_time_features(original_index)
    if not isinstance(tf.index, pd.DatetimeIndex):
        tf = tf.set_index(original_index)
    parts.append(tf.reindex(original_index))

    # 2) 室内温度: BL + 残差ラグ（列名は indoor_temp__* に統一）
    indoor_cols_raw = pick_cols(df, indoor_prefix)
    if indoor_cols_raw:
        tmp_in = df[indoor_cols_raw].copy()
        rename_in = {
            c: c.replace(indoor_prefix, "indoor_temp__") for c in indoor_cols_raw
        }
        tmp_in.rename(columns=rename_in, inplace=True)

        bl_in = compute_same_time_baseline(
            tmp_in, tmp_in.columns, days=days_for_baseline, fallback=baseline_fallback
        ).reindex(original_index)
        parts.append(bl_in.add_prefix("bl__").reindex(original_index))

        res_in = (tmp_in.reindex(original_index) - bl_in).add_prefix("res__")
        lag_res_in = add_lagged_features(
            res_in, res_in.columns, lags_hours=lags_hours, use_freq_shift=use_freq_shift
        ).reindex(original_index)
        parts.append(lag_res_in)

    # 3) 室外機 kWh: BL + 残差ラグ
    odu_cols = pick_cols(df, "total_kwh__")
    if odu_cols:
        bl_odu = compute_same_time_baseline(
            df, odu_cols, days=days_for_baseline, fallback=baseline_fallback
        ).reindex(original_index)
        parts.append(bl_odu.add_prefix("bl__").reindex(original_index))

        raw_odu = df[odu_cols].reindex(original_index)
        res_odu = (raw_odu - bl_odu).add_prefix("res__")
        lag_res_odu = add_lagged_features(
            res_odu,
            res_odu.columns,
            lags_hours=lags_hours,
            use_freq_shift=use_freq_shift,
        ).reindex(original_index)
        parts.append(lag_res_odu)

    # 4) 天気（元値）
    if include_weather_raw:
        if weather_cols is None:
            candidates = ["Outdoor Temp.", "Outdoor Humidity", "Solar Radiation"]
            weather_cols = [c for c in candidates if c in df.columns]
        if weather_cols:
            parts.append(df[list(weather_cols)].copy().reindex(original_index))

    # 5) 操作量（元値）
    if include_original_controls:
        control_cols: list[str] = []
        for p in (setT_prefix, mode_prefix, wind_prefix, onoff_prefix):
            control_cols += pick_cols(df, p)
        if control_cols:
            parts.append(df[control_cols].copy().reindex(original_index))

    X = pd.concat(parts, axis=1).reindex(original_index)
    if drop_initial_window:
        kept_index = _cut_head_for_baseline_and_lag(
            original_index, days_for_baseline, lags_hours
        )
        X = X.loc[kept_index]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X


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


# ========= DP法：同時刻BL + 残差ラグ（オンライン） =========


@dataclass
class _SlotState:
    buf: np.ndarray
    ptr: int
    sums: np.ndarray
    counts: np.ndarray


class _SameTimeDP:
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
        self.cols = list(cols)  # "indoor_temp.__X" / "total_kwh__Y"
        self.col_index: Dict[str, int] = {c: i for i, c in enumerate(self.cols)}
        self.n = len(self.cols)
        self.days = int(days)
        self.fallback = float(fallback)
        self.lags = sorted(set(int(h) for h in (lags_hours or [])))
        self.max_lag = max(self.lags) if self.lags else 0
        self.use_freq_shift = bool(use_freq_shift)
        self.lag_prefix_fmt = lag_prefix_fmt
        self.slots: Dict[Tuple[int, int, int], _SlotState] = {}
        self.res_row_hist: deque[np.ndarray] = deque(
            maxlen=self.max_lag if not self.use_freq_shift else 1
        )
        self.raw_by_ts: Dict[pd.Timestamp, np.ndarray] = {}
        self.res_by_ts: Dict[pd.Timestamp, np.ndarray] = {}

    def _baseline_exact_by_days(self, ts: pd.Timestamp) -> np.ndarray:
        """★ ちょうど k 日前 (k=1..days) の同一時刻だけを平均（NaN 除外, 無ければ fallback）"""
        out = np.full(self.n, self.fallback, float)
        s = np.zeros(self.n, float)
        c = np.zeros(self.n, dtype=np.int64)

        # tz付きなら naive化（壁時計時刻を維持）
        t = ts.tz_convert(None) if ts.tz is not None else ts

        for k in range(1, self.days + 1):
            prev_ts = t - pd.Timedelta(days=k)
            v = self.raw_by_ts.get(prev_ts)
            if v is None:
                continue
            mask = ~np.isnan(v)
            s[mask] += v[mask]
            c[mask] += 1

        valid = c > 0
        out[valid] = s[valid] / c[valid]
        return out

    def _slot_key(self, ts: pd.Timestamp) -> Tuple[int, int, int]:
        t = ts.tz_convert(None) if ts.tz is not None else ts
        return (t.hour, t.minute, t.second)

    def _get_slot(self, ts: pd.Timestamp) -> _SlotState:
        key = self._slot_key(ts)
        st = self.slots.get(key)
        if st is None:
            buf = np.full((self.days, self.n), np.nan, float)
            st = _SlotState(
                buf=buf,
                ptr=0,
                sums=np.zeros(self.n, float),
                counts=np.zeros(self.n, dtype=np.int64),
            )
            self.slots[key] = st
        return st

    def _baseline(self, st: _SlotState) -> np.ndarray:
        out = np.full(self.n, self.fallback, float)
        valid = st.counts > 0
        out[valid] = st.sums[valid] / st.counts[valid]
        return out

    def _evict_insert(self, st: _SlotState, new_vals: np.ndarray):
        i = st.ptr
        old = st.buf[i, :]
        old_valid = ~np.isnan(old)
        st.sums[old_valid] -= old[old_valid]
        st.counts[old_valid] -= 1
        st.buf[i, :] = new_vals
        new_valid = ~np.isnan(new_vals)
        st.sums[new_valid] += new_vals[new_valid]
        st.counts[new_valid] += 1
        st.ptr = (st.ptr + 1) % self.days

    def _make_output_columns(self) -> List[str]:
        cols = []
        cols.extend([f"bl__{c}" for c in self.cols])
        for h in self.lags:
            cols.extend(
                [f"{self.lag_prefix_fmt.format(h=h)}res__{c}" for c in self.cols]
            )
        return cols

    def _compute_lag_concat(self, ts: pd.Timestamp) -> np.ndarray:
        if not self.lags:
            return np.empty(0, float)
        vecs = []
        for h in self.lags:
            if self.use_freq_shift:
                v = self.res_by_ts.get(ts - pd.Timedelta(hours=h))
                vecs.append(v if v is not None else np.full(self.n, np.nan))
            else:
                if len(self.res_row_hist) >= h:
                    vecs.append(self.res_row_hist[-h])
                else:
                    vecs.append(np.full(self.n, np.nan))
        return np.concatenate(vecs)

    def grow_columns(self, new_cols: List[str]):
        add = [c for c in new_cols if c not in self.col_index]
        if not add:
            return
        k = len(add)
        for st in self.slots.values():
            st.buf = np.concatenate([st.buf, np.full((self.days, k), np.nan)], axis=1)
            st.sums = np.concatenate([st.sums, np.zeros(k)], axis=0)
            st.counts = np.concatenate([st.counts, np.zeros(k, dtype=np.int64)], axis=0)
        if self.res_row_hist.maxlen is not None:
            new_hist = deque(maxlen=self.res_row_hist.maxlen)
            for v in self.res_row_hist:
                new_hist.append(np.concatenate([v, np.full(k, np.nan)]))
            self.res_row_hist = new_hist
        for ts, v in list(self.res_by_ts.items()):
            self.res_by_ts[ts] = np.concatenate([v, np.full(k, np.nan)])
        base = self.n
        for i, c in enumerate(add):
            self.col_index[c] = base + i
        self.cols.extend(add)
        self.n += k

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = _ensure_dtindex(df)
        use_cols = [c for c in self.cols if c in df.columns]
        X = df[use_cols].reindex(columns=self.cols, fill_value=np.nan).to_numpy(float)
        out_cols = self._make_output_columns()
        out = np.empty((len(df), len(out_cols)), float)
        for i, (ts, row) in enumerate(zip(df.index, X)):
            st = self._get_slot(ts)
            bl = self._baseline_exact_by_days(ts)
            lag = self._compute_lag_concat(ts)
            pos = 0
            out[i, pos : pos + self.n] = bl
            pos += self.n
            for j in range(len(self.lags)):
                out[i, pos : pos + self.n] = lag[j * self.n : (j + 1) * self.n]
                pos += self.n
            res = row - bl
            if self.use_freq_shift:
                self.res_by_ts[ts] = res
            else:
                self.res_row_hist.append(res)
            self._evict_insert(st, row)
            self.raw_by_ts[ts] = row
        return pd.DataFrame(out, index=df.index, columns=out_cols)

    def transform_new(
        self, df: pd.DataFrame, *, allow_growth: bool = False, mutate: bool = True
    ) -> pd.DataFrame:
        df = _ensure_dtindex(df)
        if allow_growth:
            # ★ 受け入れ条件も正規化済みプレフィクスに変更
            incoming = [
                c
                for c in list(df.columns)
                if c in self.cols or c.startswith(_INDOOR_OUT) or c.startswith(_KWH_OUT)
            ]
            self.grow_columns([c for c in incoming if c not in self.col_index])

        use_cols = [c for c in self.cols if c in df.columns]
        X = df[use_cols].reindex(columns=self.cols, fill_value=np.nan).to_numpy(float)
        out_cols = self._make_output_columns()
        out = np.empty((len(df), len(out_cols)), float)

        for i, (ts, row) in enumerate(zip(df.index, X)):
            st = self._get_slot(ts)
            bl = self._baseline_exact_by_days(ts)
            lag = self._compute_lag_concat(ts)

            pos = 0
            out[i, pos : pos + self.n] = bl
            pos += self.n
            for j in range(len(self.lags)):
                out[i, pos : pos + self.n] = lag[j * self.n : (j + 1) * self.n]
                pos += self.n

            # --- ここが変更点：非破壊モードなら更新しない ---
            if mutate:
                res = row - bl
                if self.use_freq_shift:
                    self.res_by_ts[ts] = res
                else:
                    self.res_row_hist.append(res)
                self._evict_insert(st, row)
                self.raw_by_ts[ts] = row

        return pd.DataFrame(out, index=df.index, columns=out_cols)


class ResidualFeatureDP:
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
        # ★ 入力は正規化済み前提に切り替え
        indoor = pick_cols(df, _INDOOR_OUT)
        kwh = pick_cols(df, _KWH_OUT)
        return indoor + kwh

    def fit(self, base_df: pd.DataFrame) -> pd.DataFrame:
        # ★ 入力を正規化
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
        raw.columns = _normalize_metric_names(
            list(raw.columns)
        )  # 出力側の整形は従来通り
        raw = raw[_order_columns_like_builder(list(raw.columns), lags_hours=self.lags)]
        return raw

    def transform(
        self, new_df: pd.DataFrame, *, allow_growth: bool = True
    ) -> pd.DataFrame:
        if self.dp is None:
            raise RuntimeError("fit() を先に呼んでください。")
        new_df = _ensure_dtindex(_normalize_raw_columns(new_df))  # ★ 入力を正規化
        cols = self._collect_cols(new_df)
        out = self.dp.transform_new(new_df[cols], allow_growth=allow_growth)
        out.columns = _normalize_metric_names(list(out.columns))
        out = out[_order_columns_like_builder(list(out.columns), lags_hours=self.lags)]
        return out


# ========= 最終統合クラス: make_input_data を提供 =========


class InputDataBuilderDP:
    """
    - DP法で BL（同時刻・直近N日平均）と 残差ラグ を内部計算
    - make_input_data(time_info, weather_info, control_values) で
      [時間] → [室温BL/ラグ] → [kWh BL/ラグ] → [天気(元値)] → [操作量(元値)]
      を連結して返す（列名は build_features_residualized と互換）
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

    def fit(self, base_df: pd.DataFrame) -> pd.DataFrame:
        out = self.res_dp.fit(_normalize_raw_columns(base_df))
        self._online_started = False
        self._history_index_last = None
        return out

    # ====== ★ オンライン運用API ======

    def begin_online(self, initial_df: pd.DataFrame | None = None) -> None:
        """
        オンライン運用を開始。内部履歴ポインタをセット。
        initial_df を渡せば、その末尾を「直近実績」として扱う。
        渡さない場合は、fit 済み DP の内部状態に依存。
        """
        self._online_started = True
        if initial_df is not None:
            ini = _ensure_dtindex(initial_df)
            if len(ini) == 0:
                self._history_index_last = None
            else:
                self._history_index_last = ini.index[-1]
        else:
            self._history_index_last = None  # DP内部の最後時刻に委ねる

    def reset_online(self) -> None:
        """オンライン履歴ポインタだけクリア（DP自体のfit状態は維持）。"""
        self._online_started = False
        self._history_index_last = None

    def make_input_next(
        self,
        index: pd.DatetimeIndex | list[pd.Timestamp] | pd.Timestamp,
        weather_info: Optional[pd.DataFrame] = None,
        control_values: Optional[pd.DataFrame] = None,
        *,
        return_baseline: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        次時刻（複数でも可）の特徴量を作る。DPの内部状態は更新しない（予測用）。
        """
        if self.res_dp.dp is None:
            raise RuntimeError("fit(base_df) の後に呼んでください。")

        if isinstance(index, pd.Timestamp):
            ti = pd.DatetimeIndex([index])
        else:
            ti = pd.DatetimeIndex(index)

        # 1) 時間特徴
        tf = make_time_features(ti)

        # 2) DP法の BL/ラグ群（状態は更新しない）
        empty_obs = pd.DataFrame(index=ti)
        res_feats = self.res_dp.dp.transform_new(
            empty_obs, allow_growth=False, mutate=False
        )
        res_feats.columns = _normalize_metric_names(list(res_feats.columns))
        res_feats = res_feats[
            _order_columns_like_builder(list(res_feats.columns), lags_hours=self._lags)
        ]

        # ベースライン（任意返却）
        bl_cols = [c for c in res_feats.columns if c.startswith("bl__")]
        Y_baseline = res_feats[bl_cols].rename(columns=lambda c: c.replace("bl__", ""))

        # 3) 連結
        parts: list[pd.DataFrame] = [tf, res_feats]
        if self.include_weather_raw and (weather_info is not None):
            parts.append(_ensure_dtindex(weather_info).reindex(ti))
        if self.include_original_controls and (control_values is not None):
            parts.append(_ensure_dtindex(control_values).reindex(ti))

        X = pd.concat(parts, axis=1)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        # 列順整形
        time_cols = ["hour", "month", "weekday", "is_weekend"]
        time_cols = [c for c in time_cols if c in X.columns]
        dp_cols = [c for c in res_feats.columns if c in X.columns]
        tail_cols = [c for c in X.columns if c not in time_cols + dp_cols]
        X = X[time_cols + dp_cols + tail_cols]

        return (X, Y_baseline) if return_baseline else X

    def accumulate_actuals(self, y_actual: pd.DataFrame) -> None:
        """
        実績データ（少なくとも室温/電力など DP が必要とする系列）を取り込み、
        DP内部のラグ/BL状態を前進させる。ここで初めて allow_growth=True にする。
        """
        if self.res_dp.dp is None:
            raise RuntimeError("fit(base_df) の後に呼んでください。")

        ya = _ensure_dtindex(y_actual)
        if len(ya) == 0:
            return

        # 単調増加チェック（必要なら厳格化）
        if self._history_index_last is not None:
            if ya.index[0] <= self._history_index_last:
                raise ValueError(
                    f"accumulate_actuals の index は {self._history_index_last} より後である必要があります。"
                )

        # DP の内部状態を前進（成長 & 破壊的に更新）
        #   ここで empty_obs ではなく、実績を渡すのが重要。
        #   ResidualFeatureDP 側の transform_new が新規データを内部に取り込める前提。
        _ = self.res_dp.dp.transform_new(ya, allow_growth=True, mutate=True)

        self._history_index_last = ya.index[-1]


# ========= 目的変数（Y）を残差化 =========
def build_targets_residualized(
    base_df: pd.DataFrame,
    *,
    odu_prefix: str = "total_kwh__",
    indoor_prefix: str = "Indoor Temp.__",
    days_for_baseline: int = 7,
    baseline_fallback: float = 0.0,
    out_indoor_prefix: str = "indoor_temp__",
    fillna_value: float | None = None,  # 欠損を0などで埋めたいときに指定
    drop_initial_window=True,
) -> pd.DataFrame:
    """
    返り値: Y_res = 元値 - 同時刻過去平均（行数は base_df と一致させる）
    列: total_kwh__*, indoor_temp__*
    """
    # --- 1) index整備（既存の _ensure_dtindex を尊重、なければソートのみ） ---
    if "_ensure_dtindex" in globals():
        try:
            # 新しい引数に対応している場合
            df = _ensure_dtindex(base_df, duplicate_strategy="keep")
        except TypeError:
            # 旧版：引数なし
            df = _ensure_dtindex(base_df)
    else:
        # 念のためのフォールバック
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

    # --- 3) 元ターゲット（出力名で統一） ---
    Y_raw = pd.DataFrame(index=original_index)
    if y_odu_cols:
        Y_raw[y_odu_cols] = df[y_odu_cols].reindex(original_index)
    if y_ind_cols_raw:
        Y_raw[list(y_ind_cols_map.values())] = (
            df[y_ind_cols_raw].rename(columns=y_ind_cols_map).reindex(original_index)
        )

    if not Y_raw.columns.tolist():
        Y_res = pd.DataFrame(index=original_index)
        return Y_res if fillna_value is None else Y_res.fillna(fillna_value)

    # --- 4) BL を計算（必ず元 index に揃える） ---
    Y_bl = compute_same_time_baseline(
        Y_raw, Y_raw.columns, days=days_for_baseline, fallback=baseline_fallback
    ).reindex(original_index)

    # --- 5) 残差 + 整形（行は落とさない） ---
    Y_res = (Y_raw - Y_bl).reindex(original_index)
    Y_res = Y_res.apply(pd.to_numeric, errors="coerce")
    if drop_initial_window:
        kept_index = _cut_head_for_baseline_and_lag(
            original_index, days_for_baseline, [0]
        )
        Y_res = Y_res.loc[kept_index]
    if fillna_value is not None:
        Y_res = Y_res.fillna(fillna_value)

    return Y_res
