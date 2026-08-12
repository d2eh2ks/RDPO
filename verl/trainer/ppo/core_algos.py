# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict
from math import comb as _comb

import verl.utils.torch_functional as verl_F


def passk_response_advantages(n_rollout, n_pos, k):
    """Analytical Pass@k advantage (Chen et al. 2025, arXiv:2508.10751).

    For a prompt group of `n_rollout` rollouts with `n_pos` correct (binary EM),
    returns (A_pos, A_neg): the advantage for a correct and a wrong rollout,
    respectively, under the Pass@k objective with the given k. Derivation:
        R̄    = 1 - C(n_neg, k) / C(n_rollout, k)        # = group's unbiased pass@k
        σ    = sqrt(R̄ · (1 - R̄))
        A_pos = (1 - R̄) / σ
        A_neg = (1 - R̄ - C(n_neg-1, k-1) / C(n_rollout-1, k-1)) / σ
    Returns (0, 0) when there is no variance / no learning signal: all-correct,
    all-wrong, or fewer than k wrong rollouts (then every k-group already passes,
    so Pass@k is saturated and gives no gradient — the "ignore easy prompts"
    behavior that focuses learning on harder prompts).
    """
    if n_rollout <= 1 or k < 1 or n_rollout < k:
        return 0.0, 0.0
    n_neg = n_rollout - n_pos
    Rbar = 1.0 - (_comb(n_neg, k) / _comb(n_rollout, k))
    var = Rbar * (1.0 - Rbar)
    if var <= 0.0:                      # all-correct, all-wrong, or n_neg < k
        return 0.0, 0.0
    sigma = var ** 0.5
    A_pos = (1.0 - Rbar) / sigma
    denom = _comb(n_rollout - 1, k - 1)            # > 0 since n_rollout-1 >= k-1
    # var > 0 ⟹ k <= n_neg <= n_rollout-1, so n_neg-1 >= k-1 >= 0 (comb is safe)
    frac = _comb(n_neg - 1, k - 1) / denom
    A_neg = (1.0 - Rbar - frac) / sigma
    return A_pos, A_neg


def passk_response_rewards(n_rollout, n_pos, k):
    """RAW (pre-σ) Pass@k transformed reward — passk_response_advantages WITHOUT the /σ.

    Returns (r_pos, r_neg) = (1-R̄, 1-R̄-C(n_neg-1,k-1)/C(n_rollout-1,k-1)); these are
    already mean-0 over the group. Use this (NOT passk_response_advantages) when the
    caller will GRPO-normalize the COMBINED reward itself (Option B: r'_passk + R_fmt
    then one (·-mean)/std): feeding the /σ advantage into a second normalization would
    DOUBLE-normalize and corrupt the pass@k scale. (0,0) when pass@k is saturated."""
    if n_rollout <= 1 or k < 1 or n_rollout < k:
        return 0.0, 0.0
    n_neg = n_rollout - n_pos
    Rbar = 1.0 - (_comb(n_neg, k) / _comb(n_rollout, k))
    if Rbar * (1.0 - Rbar) <= 0.0:          # all-correct / all-wrong / n_neg < k
        return 0.0, 0.0
    r_pos = 1.0 - Rbar
    frac = _comb(n_neg - 1, k - 1) / _comb(n_rollout - 1, k - 1)
    r_neg = 1.0 - Rbar - frac
    return r_pos, r_neg


def best_of_k_advantages(rewards, k):
    """Best-of-k (generalized Pass@k) advantage for an ORDINAL / real group reward.

    Pass@k's group objective is the expected MAX reward over a random k-subset
    (for a binary reward this is the usual pass@k = 1 - C(n_neg,k)/C(n,k)). This
    generalizes it to ANY real-valued reward — e.g. R_total = R_acc + R_fmt ∈
    {0,1,2} — by keeping the "best-of-k" semantics:
        R̄   = E_{|S|=k}[ max_{i∈S} r_i ]
        σ   = std of that max-of-k variable  = sqrt(E[max²] - R̄²)
        m_i = E_{|T|=k-1, T⊆others}[ max(r_i, max_{j∈T} r_j) ]   (i's k-marginal value)
        A_i = (m_i - R̄) / σ
    Returns a list of per-rollout advantages (same order as `rewards`). All-zero
    when the max-of-k has no variance (saturated → no gradient), matching the
    binary saturation behavior. Reduces EXACTLY to passk_response_advantages for a
    binary {0,1} reward (so it is a strict generalization).

    rewards: list of per-rollout scalar rewards within ONE prompt group.
    """
    N = len(rewards)
    if N <= 1 or k < 1 or N < k:
        return [0.0] * N
    levels = sorted(set(rewards))                  # distinct reward values, ascending
    Ck_N = _comb(N, k)

    def cnt_le(vals, v):
        return sum(1 for x in vals if x <= v)

    # group objective R̄ = E[max-of-k] and E[max²] (for σ); P(max=v) via order stats
    Rbar = 0.0
    R2 = 0.0
    prev = 0                                        # C(#(reward < v), k)
    for v in levels:
        c_le = _comb(cnt_le(rewards, v), k)         # C(#(reward <= v), k)
        p = (c_le - prev) / Ck_N                    # P(max-of-k == v)
        Rbar += v * p
        R2   += (v * v) * p
        prev = c_le
    var = R2 - Rbar * Rbar
    if var <= 0.0:                                  # all-equal max → no gradient
        return [0.0] * N
    sigma = var ** 0.5

    Ck1_N1 = _comb(N - 1, k - 1)                    # > 0 since N-1 >= k-1
    adv = [0.0] * N
    for i in range(N):
        ri = rewards[i]
        others = rewards[:i] + rewards[i + 1:]
        # m_i = ri·P(M_T <= ri) + Σ_{v>ri} v·P(M_T == v),  T = (k-1)-subset of others
        base_le = _comb(cnt_le(others, ri), k - 1)
        mi = ri * (base_le / Ck1_N1)
        prevo = base_le
        for v in levels:
            if v <= ri:
                continue
            c_le_o = _comb(cnt_le(others, v), k - 1)
            mi += v * ((c_le_o - prevo) / Ck1_N1)   # v · P(M_T == v)
            prevo = c_le_o
        adv[i] = (mi - Rbar) / sigma
    return adv


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config): # seems never used?
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6,
                                   std_floor: float = 0.0,
                                   sigma_stats: dict = None):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        if sigma_stats is not None:
            # 그룹 σ 분포 로깅: floor 발동(또는 floor=0 이면 기준 0.086958 미만) 빈도와
            # 실제 적용되는 최대 증폭 배율. 크기 1 그룹(σ=1 더미)은 제외.
            watch = std_floor if std_floor > 0.0 else 0.086958
            sig = [float(id2std[idx]) for idx in id2score if len(id2score[idx]) > 1]
            nz = [s for s in sig if s > 1e-8]
            sigma_stats.update({
                'watch': watch,
                'n_groups': len(sig),
                'n_below': sum(1 for s in sig if s < watch),
                'n_zero': len(sig) - len(nz),
                'p50': float(np.median(sig)) if sig else 0.0,
                'min_nonzero': min(nz) if nz else 0.0,
                'max_amp': (1.0 / (max(min(nz), std_floor) + epsilon)) if nz else 0.0,
            })
        for i in range(bsz):
            # std_floor > 0 이면 A_i = (r - mu) / max(sigma, std_floor).
            # 그룹 내 보상 차이가 극히 작을 때 advantage 가 과증폭되는 것을 막는다.
            denom = max(float(id2std[index[i]]), std_floor) if std_floor > 0.0 \
                else float(id2std[index[i]])
            scores[i] = (scores[i] - id2mean[index[i]]) / (denom + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_grpo_step_level_advantage(token_level_step_rewards: torch.Tensor,
                                      stratum_step_id: torch.Tensor,
                                      eos_mask: torch.Tensor,
                                      epsilon: float = 1e-6):
    """Compute step-level local GRPO advantage from per-token step rewards.

    Each token at position (i, t) is tagged with a positive group ID in
    `stratum_step_id` if it belongs to a (stratum, step_k) group, else 0.
    Within each group, trajectories are normalized (mean/std) over their
    per-trajectory step reward (mean of group-positions, all equal in
    practice). The resulting advantage is written ONLY at group-tagged
    positions; other positions stay 0. Trajectories not in a given group
    are left untouched for that group.

    Args:
        token_level_step_rewards: (bs, response_length) float
        stratum_step_id: (bs, response_length) long. 0 = no group.
        eos_mask: (bs, response_length)
    Returns:
        advantages: (bs, response_length)
    """
    advantages = torch.zeros_like(token_level_step_rewards, dtype=torch.float32)

    with torch.no_grad():
        unique_gids = torch.unique(stratum_step_id)
        unique_gids = unique_gids[unique_gids > 0]
        if unique_gids.numel() == 0:
            return advantages * eos_mask

        for gid in unique_gids.tolist():
            group_mask = (stratum_step_id == gid).float()  # (bs, response_length)
            n_per_traj = group_mask.sum(dim=-1)  # (bs,)
            traj_in_group = n_per_traj > 0  # (bs,)

            if traj_in_group.sum() == 0:
                continue

            # mean reward at group positions (all positions in a traj have equal value)
            rewards_per_traj = (token_level_step_rewards * group_mask).sum(dim=-1) / n_per_traj.clamp(min=1)
            rewards_in_group = rewards_per_traj[traj_in_group]

            if rewards_in_group.numel() == 1:
                mean = torch.tensor(0.0, device=rewards_per_traj.device, dtype=rewards_per_traj.dtype)
                std = torch.tensor(1.0, device=rewards_per_traj.device, dtype=rewards_per_traj.dtype)
            else:
                mean = rewards_in_group.mean()
                std = rewards_in_group.std() + epsilon

            advs_per_traj = (rewards_per_traj - mean) / std  # (bs,)
            # apply only at group positions (group_mask is 0 outside the group, including
            # non-group trajectories), so non-group rows are untouched.
            advantages = advantages + advs_per_traj.unsqueeze(-1) * group_mask

    advantages = advantages * eos_mask
    return advantages


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(old_log_prob, log_prob, advantages, eos_mask, cliprange, cliprange_high=None,
                        kl_cov_ratio=0.0, kl_cov_coef=1.0):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        cliprange: lower-side clip (ratio floor = 1 - cliprange)
        cliprange_high: upper-side clip (ratio ceiling = 1 + cliprange_high). If
            None, falls back to symmetric clipping (DAPO Clip-Higher: pass a
            value larger than cliprange to relax the upper side only).
        kl_cov_ratio: if > 0, use KL-Cov (Cui et al. 2025, arXiv:2505.22617)
            instead of PPO clipping. The surrogate is the raw -ratio·A (NO clip),
            and on the top `kl_cov_ratio` fraction of tokens by covariance
            Cov = (logπ − mean logπ)·(A − mean A), a KL penalty β·|logπ − logπ_old|
            is added to the loss. This restricts the few high-covariance tokens
            that drive entropy collapse, encouraging exploration.
        kl_cov_coef: the KL penalty weight β (paper default 1.0).
    """
    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

    pg_losses = -advantages * ratio

    if kl_cov_ratio > 0.0:
        # ---- KL-Cov: no clipping; KL-penalize the highest-covariance tokens ----
        mask = eos_mask.bool()
        n_valid = int(mask.sum().item())
        select_num = int(kl_cov_ratio * n_valid)
        if select_num > 0:
            with torch.no_grad():
                lp_mean = verl_F.masked_mean(log_prob, eos_mask)
                adv_mean = verl_F.masked_mean(advantages, eos_mask)
                # token-wise centered cross-product (Eq. 10); invalid tokens -inf
                # so they are never picked by top-k.
                covs = (log_prob - lp_mean) * (advantages - adv_mean)
                covs = covs.masked_fill(~mask, float('-inf'))
                topk_idx = torch.topk(covs.reshape(-1), k=select_num, largest=True).indices
                sel = torch.zeros_like(pg_losses).reshape(-1)
                sel[topk_idx] = 1.0
                sel = sel.reshape(pg_losses.shape)   # 0/1 selection mask, no grad
            kl_penalty = (log_prob - old_log_prob).abs()            # |logπ − logπ_old|
            pg_losses = pg_losses + kl_cov_coef * sel * kl_penalty  # Eq. 14
            pg_clipfrac = verl_F.masked_mean(sel, eos_mask)         # fraction KL-penalized
        else:
            pg_clipfrac = torch.zeros((), device=pg_losses.device)
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
        return pg_loss, pg_clipfrac, ppo_kl

    # ---- Standard PPO clip (+ optional DAPO Clip-Higher) ----
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange_high)

    pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)
    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
