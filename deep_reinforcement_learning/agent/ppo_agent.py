# === 必要な import ===
from typing import List, Sequence

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Critic  # V(s)
from torch.distributions import Categorical


# ========= 条件付きマスク付き分布 =========
class MultiHeadCategoricalMasked:
    def __init__(
        self,
        logits_cat: torch.Tensor,  # [B, S]
        slice_sizes: List[int],  # [n_temp, n_mode, n_wind, n_onoff] * n_devices
        n_devices: int,
        n_temp: int,
        n_mode: int,
        n_wind: int,
        n_onoff: int,
        off_idx: int,
        fan_mode_idx: int,  # ← 追加: mode の 'fan' のインデックス
        auto_wind_idx: int,  # ← 追加: wind の 'auto' のインデックス
    ):
        self.logits_cat = logits_cat
        self.slice_sizes = slice_sizes
        self.n_devices = n_devices
        self.n_temp, self.n_mode, self.n_wind, self.n_onoff = (
            n_temp,
            n_mode,
            n_wind,
            n_onoff,
        )
        self.off_idx = off_idx
        self.fan_mode_idx = fan_mode_idx
        self.auto_wind_idx = auto_wind_idx
        chunks = torch.split(logits_cat, slice_sizes, dim=-1)
        self.T, self.M, self.W, self.O = [], [], [], []
        for d in range(n_devices):
            self.T.append(chunks[4 * d + 0])
            self.M.append(chunks[4 * d + 1])
            self.W.append(chunks[4 * d + 2])
            self.O.append(chunks[4 * d + 3])
        self.T = torch.stack(self.T, dim=1)
        self.M = torch.stack(self.M, dim=1)
        self.W = torch.stack(self.W, dim=1)
        self.O = torch.stack(self.O, dim=1)

    @staticmethod
    def _force_one_hot_logits(logits_2d: torch.Tensor, forced_idx: int) -> torch.Tensor:
        out = torch.full_like(logits_2d, -1e30)  # ≈ -inf
        out[:, forced_idx] = 0.0
        return out

    def sample(self) -> torch.Tensor:
        outs = []
        for d in range(self.n_devices):
            cat_on = Categorical(logits=self.O[:, d, :])
            a_on = cat_on.sample()  # [B]
            is_off = a_on == self.off_idx

            # mode: OFF なら 'fan' を強制
            m_logits = self.M[:, d, :].clone()
            if is_off.any():
                idx = torch.nonzero(is_off, as_tuple=False).squeeze(-1)
                m_logits[idx] = self._force_one_hot_logits(
                    m_logits[idx], self.fan_mode_idx
                )
            a_m = Categorical(logits=m_logits).sample()

            # wind: OFF なら 'auto' を強制
            w_logits = self.W[:, d, :].clone()
            if is_off.any():
                idx = torch.nonzero(is_off, as_tuple=False).squeeze(-1)
                w_logits[idx] = self._force_one_hot_logits(
                    w_logits[idx], self.auto_wind_idx
                )
            a_w = Categorical(logits=w_logits).sample()

            # temp は自由
            a_t = Categorical(logits=self.T[:, d, :]).sample()

            outs.append(torch.stack([a_t, a_m, a_w, a_on], dim=-1))
        return torch.cat(outs, dim=-1)

    @property
    def mode(self) -> torch.Tensor:
        outs = []
        for d in range(self.n_devices):
            a_on = torch.argmax(self.O[:, d, :], dim=-1)
            is_off = a_on == self.off_idx
            a_t = torch.argmax(self.T[:, d, :], dim=-1)
            a_mf = torch.argmax(self.M[:, d, :], dim=-1)
            a_wf = torch.argmax(self.W[:, d, :], dim=-1)
            a_m = torch.where(
                is_off, torch.tensor(self.fan_mode_idx, device=a_mf.device), a_mf
            )
            a_w = torch.where(
                is_off, torch.tensor(self.auto_wind_idx, device=a_wf.device), a_wf
            )
            outs.append(torch.stack([a_t, a_m, a_w, a_on], dim=-1))
        return torch.cat(outs, dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        B = actions.size(0)
        total = torch.zeros(B, device=actions.device, dtype=self.logits_cat.dtype)
        for d in range(self.n_devices):
            a_t = actions[:, 4 * d + 0]
            a_m = actions[:, 4 * d + 1]
            a_w = actions[:, 4 * d + 2]
            a_o = actions[:, 4 * d + 3]

            total = total + Categorical(logits=self.O[:, d, :]).log_prob(a_o)
            total = total + Categorical(logits=self.T[:, d, :]).log_prob(a_t)

            is_off = a_o == self.off_idx

            # mode: OFF のとき fan 以外は -inf、fan は 0
            lp_m_free = Categorical(logits=self.M[:, d, :]).log_prob(a_m)
            ok_m = a_m == self.fan_mode_idx
            lp_m = torch.where(
                is_off,
                torch.where(
                    ok_m, torch.zeros_like(lp_m_free), torch.full_like(lp_m_free, -1e30)
                ),
                lp_m_free,
            )
            total = total + lp_m

            # wind: OFF のとき auto 以外は -inf、auto は 0
            lp_w_free = Categorical(logits=self.W[:, d, :]).log_prob(a_w)
            ok_w = a_w == self.auto_wind_idx
            lp_w = torch.where(
                is_off,
                torch.where(
                    ok_w, torch.zeros_like(lp_w_free), torch.full_like(lp_w_free, -1e30)
                ),
                lp_w_free,
            )
            total = total + lp_w
        return total

    def entropy(self) -> torch.Tensor:
        # 固定されない on/off と temp のみ足し合わせる簡易形
        ent = torch.zeros(
            self.logits_cat.size(0),
            device=self.logits_cat.device,
            dtype=self.logits_cat.dtype,
        )
        for d in range(self.n_devices):
            ent = ent + Categorical(logits=self.O[:, d, :]).entropy()
            ent = ent + Categorical(logits=self.T[:, d, :]).entropy()
        return ent


# ========= 俳優ネット（出力は [temp, mode, wind, onoff]×台数 の連結） =========
class MultiDeviceQuadHeadDiscreteActor(nn.Module):
    def __init__(
        self,
        preprocess_net: Net,
        n_devices: int,
        n_temp: int,
        n_mode: int,
        n_wind: int,
        n_onoff: int,
        device=None,
    ):
        super().__init__()
        self.device = device or preprocess_net.device
        self.backbone = preprocess_net
        hid = preprocess_net.output_dim
        self.head_temp = nn.Linear(
            hid, n_devices * n_temp
        )  # t[Bのサイズ, デバイス, 温度クラスのロジットの値]
        self.head_mode = nn.Linear(hid, n_devices * n_mode)
        self.head_wind = nn.Linear(hid, n_devices * n_wind)
        self.head_on_off = nn.Linear(hid, n_devices * n_onoff)
        self.n_devices, self.n_temp, self.n_mode, self.n_wind, self.n_onoff = (
            n_devices,
            n_temp,
            n_mode,
            n_wind,
            n_onoff,
        )
        self.to(self.device)

    @staticmethod
    def _standardize_temp_idx(
        temp_idx, B: int, n_devices: int, device, dtype=torch.long
    ):
        """
        temp_idx を [B, n_devices] の LongTensor に正規化。
        受け付ける形:
          - スカラー int
          - 長さ n_devices の 1D
          - 形 [B, n_devices] の 2D
          - torch / numpy / list いずれもOK
        """
        if temp_idx is None:
            return None
        if isinstance(temp_idx, torch.Tensor):
            t = temp_idx.to(device=device, dtype=dtype)
        else:
            t = torch.as_tensor(temp_idx, device=device, dtype=dtype)

        if t.ndim == 0:
            t = t.view(1, 1).expand(B, n_devices)
        elif t.ndim == 1:
            assert (
                t.numel() == n_devices
            ), f"temp_idx len must be {n_devices}, got {t.numel()}"
            t = t.view(1, n_devices).expand(B, n_devices)
        elif t.ndim == 2:
            assert t.shape == (
                B,
                n_devices,
            ), f"temp_idx shape must be {(B, n_devices)}, got {tuple(t.shape)}"
        else:
            raise ValueError("temp_idx must be scalar, [n_devices], or [B, n_devices].")
        return t

    def _mask_temp_logits_window(
        self,
        logits_T: torch.Tensor,  # [B, n_devices, n_temp]
        temp_idx_BD: torch.Tensor,  # [B, n_devices]
        temp_range_idx_list: list,
        K: int,  # 半径（±K を許可）
    ) -> torch.Tensor:
        B, D, Ktemp = logits_T.shape
        # 許可位置のブールマスクを作る
        allow = torch.zeros((B, D, Ktemp), device=logits_T.device, dtype=torch.bool)
        base = torch.clamp(temp_idx_BD, 0, Ktemp - 1)  # 中心
        for delta in range(-K, K + 1):
            idx = torch.clamp(base + delta, 0, Ktemp - 1).unsqueeze(-1)  # [B,D,1]
            oh = torch.zeros_like(allow)
            oh.scatter_(-1, idx, True)
            allow |= oh
        # デバイスごとに設定温度の幅を適応。
        B, D, Ktemp = allow.shape  # バッチ, デバイスの数, 温度リストの次元数
        # --- temp_range_idx_list: List[(low_idx, high_idx)] (len = D) をテンソルに ---

        ranges = torch.as_tensor(temp_range_idx_list, device=allow.device)
        if ranges.ndim == 3:
            # Batch 次元が乗っているので 0 番目だけ使う（今は全 env 同じレンジの前提）
            ranges = ranges[0]  # → shape (8, 2)
        elif ranges.ndim != 2:
            raise ValueError(f"unexpected temp_range_idx_list shape: {ranges.shape}")
        # rangesのshapeは[D,2]の形である
        low_idx = ranges[:, 0].view(1, D, 1)
        hight_idx = ranges[:, 1].view(1, D, 1)
        idx = torch.arange(Ktemp, device=allow.device).view(
            1, 1, Ktemp
        )  # [1,1,Ktemp]のlogitの形状に変形
        oh = (idx >= low_idx) & (idx <= hight_idx)
        allow = allow & oh

        # 許可外を -inf へ
        masked = torch.where(allow, logits_T, torch.full_like(logits_T, -1e30))
        invalid = (~allow).all(dim=-1)  # [B, D]
        if invalid.any():
            print(
                "[WARN] no valid temp class for some (B,D):",
                invalid.nonzero(as_tuple=False),
            )

        return masked

    def _normalize_temp_idx(self, cur, B, device):
        """
        cur: scalar / [B] / [n_devices] / [B, n_devices] / torch/np/list
        → torch.LongTensor [B, n_devices] に整形。-1 は“無効(マスクしない)”扱い。
        """
        if cur is None:
            return None
        if isinstance(cur, torch.Tensor):
            arr = cur.detach().to(device)
        else:
            arr = torch.as_tensor(cur, device=device)
        arr = arr.long()
        if arr.ndim == 0:  # scalar → 全デバイスにブロードキャスト
            arr = arr.view(1, 1).expand(B, self.n_devices)
        elif arr.ndim == 1:
            if arr.numel() == B:  # [B] → 各行にブロードキャスト
                arr = arr.view(B, 1).expand(B, self.n_devices)
            elif (
                arr.numel() == self.n_devices
            ):  # [n_devices] → 全バッチにブロードキャスト
                arr = arr.view(1, self.n_devices).expand(B, self.n_devices)
            else:
                raise ValueError(f"current_temp_index 1D 長さ不正: {arr.shape}")
        elif arr.ndim == 2:
            if arr.shape != (B, self.n_devices):
                raise ValueError(
                    f"current_temp_index 形状不正: {arr.shape} (期待 {(B, self.n_devices)})"
                )
        else:
            raise ValueError(f"current_temp_index 次元不正: {arr.ndim}")
        return arr

    def forward(self, batch, state=None, info=None):
        obs = batch.obs if hasattr(batch, "obs") else batch
        feat, hidden = self.backbone(obs, state)
        B = feat.size(0)

        # 各ヘッドのロジット
        t = self.head_temp(feat).view(B, self.n_devices, self.n_temp)
        m = self.head_mode(feat).view(B, self.n_devices, self.n_mode)
        w = self.head_wind(feat).view(B, self.n_devices, self.n_wind)
        o = self.head_on_off(feat).view(B, self.n_devices, self.n_onoff)
        """ 設定温度について制約を付加する """
        # ---- ここで info.current_temp_index を読み、±1 以外を事前にマスク ----
        cur_idx = None
        temp_range_idx_list = None

        # 1) batch.info から取る（Batch or dict 両対応）
        if hasattr(batch, "info") and batch.info is not None:
            bi = batch.info
            if isinstance(bi, dict):
                if "current_temp_index" in bi:
                    cur_idx = bi["current_temp_index"]
                if "temp_range_idx_list" in bi:
                    temp_range_idx_list = bi["temp_range_idx_list"]
            else:
                # Tianshou の Batch を想定
                if hasattr(bi, "current_temp_index"):
                    cur_idx = bi.current_temp_index
                if hasattr(bi, "temp_range_idx_list"):
                    temp_range_idx_list = bi.temp_range_idx_list

        # 2) まだ足りないものがあれば、info 引数側を補助的に見る
        if info is not None:
            if isinstance(info, dict):
                if cur_idx is None and "current_temp_index" in info:
                    cur_idx = info["current_temp_index"]
                if temp_range_idx_list is None and "temp_range_idx_list" in info:
                    temp_range_idx_list = info["temp_range_idx_list"]
            else:
                if cur_idx is None and hasattr(info, "current_temp_index"):
                    cur_idx = info.current_temp_index
                if temp_range_idx_list is None and hasattr(info, "temp_range_idx_list"):
                    temp_range_idx_list = info.temp_range_idx_list

        cur_idx = self._normalize_temp_idx(cur_idx, B, t.device)
        if cur_idx is not None:
            # 有効(0..n_temp-1)だけをマスク、無効(-1 等)は素通し
            valid_mask = (cur_idx >= 0) & (cur_idx < self.n_temp)  # [B, D]
            if valid_mask.any():
                c = torch.clamp(cur_idx, 0, self.n_temp - 1)  # [B, D]
                t_masked = self._mask_temp_logits_window(
                    t, c, temp_range_idx_list, K=2
                )  # ±2 を許可＆各デバイスごとの設定温度のマスクを適用
                vm = valid_mask.unsqueeze(-1)  # [B, D, 1]
                t = torch.where(vm, t_masked, t)

        # 以降は従来どおり連結
        logits_chunks = []
        for d in range(self.n_devices):
            logits_chunks += [t[:, d, :], m[:, d, :], w[:, d, :], o[:, d, :]]
        logits_cat = torch.cat(logits_chunks, dim=-1)
        return logits_cat, hidden


# ========= Policy ファクトリ（Masked 分布を dist_fn に設定） =========
def create_ppo_for_hvac(
    single_env: gym.Env,
    device: torch.device,
    lr: float,
    set_temp_list: Sequence,
    set_mode_list: Sequence,  # 例: ["fan","cool","heat"]
    set_wind_list: Sequence,  # 例: ["low","mid","high","top","auto"]
    set_on_off_list: Sequence,  # 例: ["OFF","ON"] もしくは [0,1]
    n_devices: int,
    *,
    actor_hidden=(256, 256),
    critic_hidden=(512, 256),
    **ppo_kwargs,
):
    """HVAC用のPPO Policyを作成（MultiDiscrete × 台数、条件付きマスク対応）"""

    # ---- 形状など ----
    obs_shape = single_env.observation_space.shape
    act_space = single_env.action_space
    assert isinstance(
        act_space, gym.spaces.MultiDiscrete
    ), "action_space は MultiDiscrete 想定です。"

    n_temp = int(len(set_temp_list))
    n_mode = int(len(set_mode_list))
    n_wind = int(len(set_wind_list))
    n_onoff = int(len(set_on_off_list))
    slice_sizes = [n_temp, n_mode, n_wind, n_onoff] * n_devices

    expected = np.array(slice_sizes, dtype=np.int64)
    if not (
        act_space.nvec.size == expected.size and np.all(act_space.nvec == expected)
    ):
        print(
            "⚠️ action_space.nvec が [temp, mode, wind, onoff]×台数 と一致していません。"
        )

    # ---- インデックス検出（"OFF"/"fan"/"auto" でも数値でもOKな汎用関数） ----
    def _find_index(seq: Sequence, target, default_idx: int):
        seq_list = list(seq)
        # 数値で指定された場合（例: 0 や 4）
        if isinstance(target, int):
            return target if 0 <= target < len(seq_list) else default_idx
        # 文字列で指定された場合（例: "OFF", "fan", "auto"）
        if isinstance(target, str):
            low = target.strip().lower()
            for i, v in enumerate(seq_list):
                if isinstance(v, str) and v.strip().lower() == low:
                    return i
        return default_idx

    # "OFF" / "fan" / "auto" を優先して探し、なければ fallback
    off_idx = _find_index(set_on_off_list, "OFF", 0)
    fan_mode_idx = _find_index(set_mode_list, "fan", 0)
    auto_wind_idx = _find_index(set_wind_list, "auto", max(0, n_wind - 1))

    # ---- dist_fn: Actor出力(logits_cat) → 条件付きマスク付き分布 ----
    def dist_fn(action_dist_input_BS: torch.Tensor):
        return MultiHeadCategoricalMasked(
            logits_cat=action_dist_input_BS,
            slice_sizes=slice_sizes,
            n_devices=n_devices,
            n_temp=n_temp,
            n_mode=n_mode,
            n_wind=n_wind,
            n_onoff=n_onoff,
            off_idx=off_idx,
            fan_mode_idx=fan_mode_idx,
            auto_wind_idx=auto_wind_idx,
        )

    # ---- ネットワーク ----
    actor_backbone = Net(
        state_shape=obs_shape, hidden_sizes=list(actor_hidden), device=device
    )
    critic_backbone = Net(
        state_shape=obs_shape, hidden_sizes=list(critic_hidden), device=device
    )

    actor = MultiDeviceQuadHeadDiscreteActor(
        actor_backbone, n_devices, n_temp, n_mode, n_wind, n_onoff, device=device
    ).to(device)
    critic = Critic(critic_backbone, device=device).to(device)

    # ---- Optimizer ----
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=lr
    )

    # PPOPolicy に渡す不要キーを除去
    policy_kwargs = dict(ppo_kwargs)
    policy_kwargs.pop("action_scaling", None)  # 本ポリシーでは使わない
    policy_kwargs.pop("action_space", None)  # policy 側で設定するため除去

    # ---- Policy ----
    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,  # ← 最重要：ここで分布を組み立てる
        action_space=act_space,  # MultiDiscrete([n_temp, n_mode, n_wind, n_onoff] * n_devices)
        action_scaling=False,  # 離散なのでスケーリング不要
        **policy_kwargs,
    ).to(device)

    return policy


# ========= 予測アクションを Series にマップするヘルパ =========
def actions_to_frame(
    act_indices: np.ndarray | torch.Tensor,
    *,
    current_time: pd.Timestamp | str,
    set_temp_list: Sequence,
    set_mode_list: Sequence,
    set_wind_list: Sequence,
    set_on_off_list: Sequence,
    n_devices: int,
    set_cols: Sequence[str],
    mode_cols: Sequence[str],
    fan_cols: Sequence[str],
    onoff_cols: Sequence[str],
    index_name: str = "Datetime_hour",
) -> pd.DataFrame:
    """
    policy.forward(...).act（形は [4*n_devices] または [B,4*n_devices]の先頭行）を
    各デバイスの (temp,mode,wind,onoff)→実値にデコードし、
    インデックス= current_time の 1行DataFrameで返す。

    返却: DataFrame(1行)
      index.name = index_name（デフォルト: 'Datetime_hour'）
      columns = set_cols + mode_cols + fan_cols + onoff_cols
    """
    # --- act を 1次元 numpy に揃える ---
    if isinstance(act_indices, torch.Tensor):
        act = act_indices.detach().cpu().numpy()
    else:
        act = np.asarray(act_indices)
    if act.ndim == 2:
        act = act[0]
    assert act.shape[0] == 4 * n_devices, f"期待長 4*n_devices に不一致: {act.shape[0]}"

    # --- 選択肢 ---
    vals_temp = list(set_temp_list)
    vals_mode = list(set_mode_list)
    vals_wind = list(set_wind_list)
    vals_onof = list(set_on_off_list)

    # --- デコード（列名→値 の辞書を作る）---
    out_map: dict[str, object] = {}
    for d in range(n_devices):
        i_t, i_m, i_w, i_o = act[4 * d : 4 * d + 4].astype(int)
        out_map[set_cols[d]] = vals_temp[i_t]
        out_map[mode_cols[d]] = vals_mode[i_m]
        out_map[fan_cols[d]] = vals_wind[i_w]
        out_map[onoff_cols[d]] = vals_onof[i_o]

    # --- 1行DataFrame化（インデックスに current_time を使用）---
    ts = pd.Timestamp(current_time)
    df = pd.DataFrame([out_map], index=[ts])
    df.index.name = index_name
    return df
