# === 必要な import ===
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Critic  # そのままV(s)でOK
from torch.distributions import Categorical


class MultiHeadCategorical:
    def __init__(self, logits_list):

        self.cats = [Categorical(logits=logits) for logits in logits_list]

    def sample(self):
        return torch.stack([c.sample() for c in self.cats], dim=-1)  # [B, H]

    @property
    def mode(self):
        # greedy を「属性」で返す（Tianshouがそのまま読む）
        return torch.stack([torch.argmax(c.logits, dim=-1) for c in self.cats], dim=-1)

    def log_prob(self, actions):
        return torch.stack(
            [c.log_prob(actions[:, i]) for i, c in enumerate(self.cats)], dim=-1
        ).sum(dim=-1)

    def entropy(self):
        return torch.stack([c.entropy() for c in self.cats], dim=-1).sum(dim=-1)


# ============== 分布：複数Categoricalをまとめる薄いラッパ ==============
class MultiDeviceTriHeadDiscreteActor(nn.Module):
    def __init__(
        self,
        preprocess_net: Net,
        n_devices: int,
        n_temp: int,
        n_mode: int,
        n_wind: int,
        device=None,
    ):
        super().__init__()
        self.device = device or preprocess_net.device
        self.backbone = preprocess_net
        hid = preprocess_net.output_dim
        self.head_temp = nn.Linear(hid, n_devices * n_temp)
        self.head_mode = nn.Linear(hid, n_devices * n_mode)
        self.head_wind = nn.Linear(hid, n_devices * n_wind)
        self.n_devices, self.n_temp, self.n_mode, self.n_wind = (
            n_devices,
            n_temp,
            n_mode,
            n_wind,
        )
        self.to(self.device)

    def forward(self, batch, state=None, info=None):
        obs = batch.obs if hasattr(batch, "obs") else batch
        feat, hidden = self.backbone(obs, state)
        B = feat.size(0)
        t = self.head_temp(feat).view(B, self.n_devices, self.n_temp)
        m = self.head_mode(feat).view(B, self.n_devices, self.n_mode)
        w = self.head_wind(feat).view(B, self.n_devices, self.n_wind)

        # (devごとに temp, mode, wind の順で) 最後の次元に連結 → [B, S]
        logits_chunks = []
        for d in range(self.n_devices):
            logits_chunks += [t[:, d, :], m[:, d, :], w[:, d, :]]
        logits_cat = torch.cat(logits_chunks, dim=-1)  # ★ これだけ返す

        return logits_cat, hidden


# ============== Policy を作るファクトリ ==============
def create_ppo_for_hvac(
    single_env,
    device,
    lr,
    set_temp_list,
    set_mode_list,
    set_wind_list,
    n_devices: int,
    **ppo_kwargs,
):
    """
    - single_env: 観測/行動空間を持つ環境（行動空間は確認だけに使う）
    - set_*_list: 各カテゴリの候補値のリスト（長さだけ使う）
    - n_devices: 制御対象台数
    """

    obs_shape = single_env.observation_space.shape
    # 行動空間は MultiDiscrete([n_temp, n_mode, n_wind] * n_devices) を推奨
    act_space = single_env.action_space
    assert isinstance(
        act_space, gym.spaces.MultiDiscrete
    ), "action_space は MultiDiscrete を想定"

    n_temp = len(set_temp_list)
    n_mode = len(set_mode_list)
    n_wind = len(set_wind_list)

    # 期待の nvec パターンと一致するか軽く検査（形だけ）
    slice_sizes = [n_temp, n_mode, n_wind] * n_devices
    expected = np.array([n_temp, n_mode, n_wind] * n_devices, dtype=np.int64)
    if act_space.nvec.size == expected.size and np.all(act_space.nvec == expected):
        pass  # OK
    else:
        # 多少違っていても学習自体はできるが、意図ズレ検出用に警告
        print(
            "⚠️ action_space.nvec が (temp,mode,wind)*n_devices の想定と一致しません。"
        )

    def dist_fn(action_dist_input_BS):
        # actor.forward が返す logits_cat [B, S]
        logits_cat = action_dist_input_BS
        logits_list = list(torch.split(logits_cat, slice_sizes, dim=-1))
        return MultiHeadCategorical(logits_list)

    # ===== Nets =====
    actor_backbone = Net(state_shape=obs_shape, hidden_sizes=[256, 128], device=device)
    critic_backbone = Net(state_shape=obs_shape, hidden_sizes=[512, 256], device=device)

    actor = MultiDeviceTriHeadDiscreteActor(
        actor_backbone,
        n_devices=n_devices,
        n_temp=n_temp,
        n_mode=n_mode,
        n_wind=n_wind,
        device=device,
    ).to(device)
    critic = Critic(critic_backbone, device=device).to(device)

    # ===== Optimizer =====
    params_ = list(actor.parameters()) + list(critic.parameters())
    assert any(p.requires_grad for p in params_), "no trainable parameters!"
    optim = torch.optim.Adam(params_, lr=lr)
    # 衝突しそうなキーを除去
    policy_kwargs = dict(ppo_kwargs)
    policy_kwargs.pop("action_scaling", None)
    policy_kwargs.pop("action_space", None)

    policy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist_fn,  # ★ 各ヘッドのCategoricalを束ねる分布
        action_space=act_space,  # gym.spaces.MultiDiscrete
        action_scaling=False,  # 離散なので False 固定
        **policy_kwargs,
    ).to(device)

    return policy
