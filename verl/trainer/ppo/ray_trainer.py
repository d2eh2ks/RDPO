# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import re
import json
from collections import defaultdict

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance

import re
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, lambda_div=0.0,
                      per_group_constraint=False, passk_k=2, passk_fmt_weight=1.0,
                      passk_reward_mode='joint_binary', batch_explore_c=0.2,
                      allwrong_adv_scale=False, allcorrect_center_only=False, grpo_std_floor=0.0):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        _sigma_stats = {}
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        std_floor=grpo_std_floor,
                                                                        sigma_stats=_sigma_stats)
        data.meta_info['grpo_sigma_stats'] = _sigma_stats
        # All-wrong-group advantage scaling (governance knob). On every prompt
        # group whose rollouts are ALL wrong (R_acc==0), scale the (diversity-
        # driven) advantage by c = this step's batch accuracy:
        #     Â_i ← c · Â_i    for i in an all-wrong group,   c = mean_j 1[R_acc_j>0]
        # Groups containing >=1 correct rollout are untouched. This is a per-group
        # CONSTANT factor applied AFTER GRPO group-normalization, so — unlike a
        # pre-norm λ — it does not cancel in (r-mean)/std. Early in training c is
        # small (few correct), so the ungated wrong-group diversity push is weak;
        # as accuracy climbs c grows and the push strengthens. Requires per-rollout
        # correctness in meta_info['em_acc'] (set by the em_fmt_doc RewardManager);
        # a no-op unless wrong groups carry diversity variance (budget_ungate=true).
        if allwrong_adv_scale and 'em_acc' in data.meta_info:
            em_acc = data.meta_info['em_acc']
            c = (sum(em_acc) / len(em_acc)) if len(em_acc) > 0 else 0.0
            from collections import defaultdict as _dd
            uid_to_idx = _dd(list)
            for i in range(len(index)):
                uid_to_idx[index[i]].append(i)
            n_scaled = 0
            for _uid, idx_list in uid_to_idx.items():
                if all(em_acc[j] <= 0 for j in idx_list):   # all-wrong group
                    for j in idx_list:
                        advantages[j] = advantages[j] * c
                        returns[j] = returns[j] * c
                    n_scaled += len(idx_list)
            data.meta_info['allwrong_adv_scale_stats'] = {'c': float(c), 'n_scaled': n_scaled}
        # ALL-CORRECT groups: mean-centering only, NO division by std.
        # There R_acc is constant, so plain GRPO divides by a std driven purely by R_div
        # spread — a group whose rollouts differ by 0.01 in R_div gets its advantages
        # blown up to the same +/-1.4 scale as a group that differs by 0.5. Subtracting
        # the mean without dividing keeps the advantage in RAW reward units, so a
        # saturated group only pushes as hard as its actual R_div spread warrants.
        #     rewards {1.9,1.7,1.5,1.3,1.1} -> mean 1.5
        #     A_i = r_i - mean  ->  {+0.4, +0.2, 0.0, -0.2, -0.4}   (sum = 0)
        if allcorrect_center_only and 'em_acc' in data.meta_info:
            em_acc = data.meta_info['em_acc']
            non_zero_mask = (token_level_rewards != 0)
            raw_scores = (token_level_rewards * non_zero_mask).sum(dim=-1)
            from collections import defaultdict as _dd3
            _grp = _dd3(list)
            for i in range(len(index)):
                _grp[index[i]].append(i)
            _n_grp, _spread = 0, []
            for _uid, _idxs in _grp.items():
                if len(_idxs) < 2 or not all(em_acc[j] > 0 for j in _idxs):
                    continue
                _n_grp += 1
                _v = torch.stack([raw_scores[j].float() for j in _idxs])
                _mean = _v.mean()
                _spread.append(float(_v.max() - _v.min()))
                for j in _idxs:
                    a = raw_scores[j].float() - _mean
                    advantages[j] = a * response_mask[j]
                    returns[j] = advantages[j]
            data.meta_info['allcorrect_center_only_stats'] = {
                'n_groups': _n_grp,
                'mean_spread': float(sum(_spread) / len(_spread)) if _spread else 0.0}
        # MIXED-group advantage separation: on groups holding BOTH correct and wrong
        # rollouts, how far above the wrong ones do the correct ones sit? R_div adds
        # within-group variance that is orthogonal to correctness, so if diversity is
        # crowding out the accuracy signal this gap SHRINKS. Watch it against
        # div/penalty_gap — gap collapsing toward 0 is the failure mode.
        if 'em_acc' in data.meta_info:
            _acc = data.meta_info['em_acc']
            _den = response_mask.sum(dim=-1).clamp(min=1)
            _scalar = (advantages * response_mask).sum(dim=-1) / _den   # per-rollout Â
            from collections import defaultdict as _dd2
            _g = _dd2(list)
            for i in range(len(index)):
                _g[index[i]].append(i)
            _gaps, _cs, _ws = [], [], []
            for _uid, _idxs in _g.items():
                _c = [j for j in _idxs if _acc[j] > 0]
                _w = [j for j in _idxs if _acc[j] <= 0]
                if not _c or not _w:
                    continue
                mc = float(sum(_scalar[j] for j in _c) / len(_c))
                mw = float(sum(_scalar[j] for j in _w) / len(_w))
                _gaps.append(mc - mw); _cs.append(mc); _ws.append(mw)
            if _gaps:
                data.meta_info['mixed_adv_gap'] = {
                    'gap': float(sum(_gaps) / len(_gaps)),
                    'correct': float(sum(_cs) / len(_cs)),
                    'wrong': float(sum(_ws) / len(_ws)),
                    'n_groups': len(_gaps),
                }
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo_step_level':
        # Outcome (EM) advantage: standard group-relative on token_level_rewards
        # (which holds the outcome reward at the last token per RewardManager).
        # Step-level (diversity) advantage: grouped by (stratum, step) tags in
        # `stratum_step_id`, normalized over per-trajectory step reward, and
        # written back only at the tagged positions. The two are summed.
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        outcome_adv, _ = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            eos_mask=response_mask,
            index=index,
        )

        if 'step_level_scores' in data.batch.keys() and 'stratum_step_id' in data.batch.keys():
            step_rewards = data.batch['step_level_scores']
            stratum_step_id = data.batch['stratum_step_id']
            step_adv = core_algos.compute_grpo_step_level_advantage(
                token_level_step_rewards=step_rewards,
                stratum_step_id=stratum_step_id,
                eos_mask=response_mask,
            )
        else:
            step_adv = torch.zeros_like(outcome_adv)

        if per_group_constraint:
            # Per-prompt λ_i derived from the two safety constraints:
            #   (1) x_i ≥ λ_i · z_i        (answer dominates max diversity)
            #   (2) x_i + λ_i · y_i ≥ 0    (correct never punished below 0)
            #   → λ_i = min(x_i / z_i, x_i / |y_i|)
            # NOTE: strict same-group ranking (correct.min ≥ wrong.max) is
            # already implied by (2): wrong trajectories receive no diversity
            # (accuracy gating zeros their step_adv) so wrong.max = x_wrong < 0,
            # while (2) ensures correct.min ≥ 0. No third constraint needed.
            # where for prompt-group i,
            #   x_i = post-norm value at outcome=1  = (1 - μ_outcome_i) / σ_outcome_i
            #   y_i = min of step_adv at tagged positions across the group
            #   z_i = max of step_adv at tagged positions across the group
            # Edge cases:
            #   - all-wrong group  → no diversity reward existed (accuracy gating
            #     in RewardManager), step_adv is 0 → λ_i = 0 (irrelevant).
            #   - all-correct group → x undefined; use λ_i = 1.0 (let diversity
            #     drive the update since outcome gives no signal).
            #   - diversity has no variance in the group → skip (λ_i = 0).
            bs = outcome_adv.shape[0]
            device = outcome_adv.device
            # per-trajectory raw outcome reward (0 or 1).
            raw_outcome = (token_level_rewards * (token_level_rewards != 0).float()).sum(dim=-1)
            # per-token diversity-position mask.
            stratum_mask = (stratum_step_id > 0).float() if 'stratum_step_id' in data.batch.keys() else torch.zeros_like(outcome_adv)
            lambda_per_traj = torch.zeros(bs, device=device)
            lambdas_logged = []
            n_all_correct = 0
            n_all_wrong = 0
            n_div_degenerate = 0
            n_constraint = 0
            eps = 1e-6
            from collections import defaultdict as _dd
            uid_to_idx = _dd(list)
            for i in range(bs):
                uid_to_idx[index[i]].append(i)
            for uid_val, idx_list in uid_to_idx.items():
                idx_t = torch.tensor(idx_list, device=device, dtype=torch.long)
                grp_raw = raw_outcome[idx_t]  # (G,)
                G = grp_raw.numel()
                n_correct = int((grp_raw > 0).sum().item())
                # Collect step_adv values at tagged positions across this group.
                grp_step_adv = step_adv[idx_t]
                grp_mask = stratum_mask[idx_t]
                tagged_vals = grp_step_adv[grp_mask > 0]
                if n_correct == 0:
                    # all wrong → no diversity, λ_i = 0
                    n_all_wrong += 1
                    continue
                if n_correct == G:
                    # all correct → outcome adv is all 0, just use λ_i = 1
                    lambda_per_traj[idx_t] = 1.0
                    n_all_correct += 1
                    lambdas_logged.append(1.0)
                    continue
                if tagged_vals.numel() == 0:
                    # no diversity values at all for this prompt (no step group
                    # formed e.g. all singleton steps). λ_i irrelevant.
                    n_div_degenerate += 1
                    continue
                # x_i from raw outcome group stats.
                mu = grp_raw.mean()
                sd = grp_raw.std(unbiased=False) + eps
                x_i = ((1.0 - mu) / sd).item()
                y_i = tagged_vals.min().item()
                z_i = tagged_vals.max().item()
                # If diversity has effectively no variance, skip.
                if (z_i - y_i) < eps:
                    n_div_degenerate += 1
                    continue
                # Each ratio guards against tiny denominators.
                cand = []
                if z_i > eps:
                    cand.append(x_i / z_i)
                if abs(y_i) > eps:
                    cand.append(x_i / abs(y_i))
                if len(cand) == 0:
                    n_div_degenerate += 1
                    continue
                lam_i = min(cand)
                lam_i = max(0.0, lam_i)  # clamp to ≥ 0; should normally already be positive
                lambda_per_traj[idx_t] = lam_i
                lambdas_logged.append(lam_i)
                n_constraint += 1
            advantages = outcome_adv + lambda_per_traj.unsqueeze(-1) * step_adv
            data.meta_info['per_group_lambda_stats'] = {
                'lambdas': lambdas_logged,
                'n_all_correct': n_all_correct,
                'n_all_wrong': n_all_wrong,
                'n_div_degenerate': n_div_degenerate,
                'n_constraint': n_constraint,
            }
        else:
            # lambda_div applied POST-normalization so it actually controls the
            # relative strength of the diversity signal vs the EM signal. (Pre-
            # normalization scaling cancels out in GRPO group normalization.)
            advantages = outcome_adv + lambda_div * step_adv
        data.batch['advantages'] = advantages
        data.batch['returns'] = advantages
    elif adv_estimator == 'grpo_step_combined':
        # Per-block COMBINED reward, normalized per block-position stratum.
        # RewardManager (combine_mode='step_reward') already wrote the full
        # per-block reward R_acc + λ·1[acc]·div_k into `step_level_scores` and
        # tagged each block with a (prompt, position) group id in
        # `stratum_step_id`. We just GRPO-normalize within each such group and use
        # the result DIRECTLY as the advantage — there is NO separate outcome
        # advantage and NO post-hoc λ multiply (both are already baked into the
        # per-block reward). This is the multi-turn literal of the slide formula:
        # the k-th search blocks of all rollouts normalize together, the answer
        # blocks normalize together.
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        assert 'step_level_scores' in data.batch.keys() and \
            'stratum_step_id' in data.batch.keys(), \
            "grpo_step_combined requires RewardManager combine_mode='step_reward'"
        advantages = core_algos.compute_grpo_step_level_advantage(
            token_level_step_rewards=data.batch['step_level_scores'],
            stratum_step_id=data.batch['stratum_step_id'],
            eos_mask=response_mask,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = advantages
    elif adv_estimator == 'grpo_passk':
        # Pass@k Training (Chen et al. 2025). Advantage built from per-rollout
        # labels stashed by RewardManager(combine_mode='passk'). passk_reward_mode:
        #   'joint_binary' : pass@k over binary success 1[R_acc ∧ R_fmt] — treats
        #                    R_acc + R_fmt as ONE criterion (symmetric to Proposed). [default]
        #   'best_of_k'    : best-of-k (generalized pass@k) over R_total = R_acc +
        #                    R_fmt ∈ {0,1,2} → 3-valued advantage A_2>A_1>A_0.
        #   'acc_only'     : A_i = pass@k_adv(1[R_acc]) + passk_fmt_weight·(R_fmt−mean(R_fmt))
        #                    — acc channel = analytic pass@k advantage (÷ analytic σ,
        #                    saturated→0); fmt channel = group-CENTERED only (no /σ), β
        #                    scaled. NO empirical normalization anywhere, so the pass@k
        #                    signal is preserved (does NOT collapse to Pass@1). This is
        #                    the R_div-removed counterpart of the method (R_acc slot →
        #                    pass@k advantage; R_fmt kept at weight β). β=passk_fmt_weight
        #                    default 1.0. Cold-start safe (R_fmt≡0 → centered 0).
        #   'acc_analytic' : success = 1[R_acc] only; paper-faithful analytic advantage
        #                    (÷σ_group) assigned directly, format ignored. The correct
        #                    R_acc-only Pass@k baseline (does NOT collapse to Pass@1).
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        index = data.non_tensor_batch['uid']
        bs = responses.size(0)
        acc = data.meta_info['passk_acc']                 # binary R_acc (EM)
        fmt = data.meta_info['passk_fmt']                 # binary R_fmt
        rtot = data.meta_info['passk_reward']             # R_acc + R_fmt ∈ {0,1,2}

        uid2idx = defaultdict(list)
        for i in range(bs):
            uid2idx[index[i]].append(i)

        adv_per = [0.0] * bs
        for _uid, idxs in uid2idx.items():
            N = len(idxs)
            if passk_reward_mode == 'best_of_k':
                a_group = core_algos.best_of_k_advantages([rtot[i] for i in idxs], passk_k)
                for loc, i in enumerate(idxs):
                    adv_per[i] = a_group[loc]
            elif passk_reward_mode == 'joint_binary':
                # success = R_acc ∧ R_fmt; analytic pass@k advantage applied DIRECTLY.
                # (Warning: dead when R_fmt≡0 — joint≡0 → R̄=σ=0 → zero advantage.)
                succ = {i: (acc[i] > 0 and fmt[i] > 0) for i in idxs}
                n_pos = sum(1 for i in idxs if succ[i])
                A_pos, A_neg = core_algos.passk_response_advantages(N, n_pos, passk_k)
                for i in idxs:
                    adv_per[i] = A_pos if succ[i] else A_neg
            elif passk_reward_mode == 'acc_analytic':
                # success = 1[R_acc] ONLY (format ignored); paper-faithful analytic
                # pass@k advantage (÷σ_group=√(R̄(1−R̄))) assigned DIRECTLY — NO re-norm.
                # This is the R_acc-only counterpart of joint_binary. It must divide by
                # the ANALYTIC σ_group, NOT an empirical group std: the pass@k transform
                # r_pos/r_neg is affine in the binary label, and (·−mean)/std is affine-
                # invariant, so re-normalizing (as 'acc_only' does) collapses it back to
                # plain Pass@1 GRPO. Dividing by σ_group is what keeps the pass@k signal.
                succ = {i: (acc[i] > 0) for i in idxs}
                n_pos = sum(1 for i in idxs if succ[i])
                A_pos, A_neg = core_algos.passk_response_advantages(N, n_pos, passk_k)
                for i in idxs:
                    adv_per[i] = A_pos if succ[i] else A_neg
            else:  # 'acc_only' — advantage-level combine, NO empirical normalization.
                #   A_i = passk_adv(1[R_acc])_i  +  β · (R_fmt_i − mean(R_fmt))
                # acc channel  = paper-faithful analytic pass@k advantage (÷ analytic
                #   σ=√(R̄(1−R̄)); saturated group → 0 = self-annealing, focuses on hard
                #   prompts). fmt channel = GROUP-CENTERED format only, scaled by β
                #   (=passk_fmt_weight). NO empirical std anywhere → the pass@k signal is
                #   NOT flattened toward Pass@1 (that was Option B's affine-collapse). On a
                #   saturated group the acc channel is 0, so only the centered fmt nudge
                #   survives — format keeps being learned. Cold-start safe: R_fmt≡0 →
                #   (fmt−mean)=0, no blow-up. This is the R_div-removed counterpart of the
                #   method (R_acc slot → pass@k analytic advantage; R_fmt kept at weight β).
                succ = {i: (acc[i] > 0) for i in idxs}
                n_pos = sum(1 for i in idxs if succ[i])
                A_pos, A_neg = core_algos.passk_response_advantages(N, n_pos, passk_k)
                fmt_mean = sum(fmt[i] for i in idxs) / N
                for i in idxs:
                    adv_per[i] = (A_pos if succ[i] else A_neg) \
                        + passk_fmt_weight * (fmt[i] - fmt_mean)

        advantages = torch.tensor(adv_per, dtype=torch.float32,
                                  device=responses.device).unsqueeze(-1) * response_mask
        data.batch['advantages'] = advantages
        data.batch['returns'] = advantages
    elif adv_estimator == 'grpo_batch':
        # Outcome-based Batch Exploration (Song et al. 2025, arXiv:2509.06941).
        # Base GRPO advantage on (R_acc + R_fmt at last token) PLUS an UN-NORMALIZED
        # within-batch answer-repetition penalty:
        #   b_batch_i = −(1/N) Σ_{j≠i} 1{a_i = a_j}      (a = final answer outcome)
        #   A_i       = GRPO_adv_i + c · b_batch_i        (faithful Eq.2; b NOT /σ)
        # Same base reward as adv_estimator='grpo' (apples-to-apples vs the proposed
        # run). NOT correct-gated — diverse answers on unsolved questions is the point.
        # An unparseable answer (None) never matches any sibling → contributes 0.
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        base_adv, _ = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index)
        outcomes = data.meta_info['batch_outcome']
        accs = data.meta_info.get('batch_acc', None)   # R_acc per rollout (for all-correct zeroing)
        bs = responses.size(0)
        uid2idx = defaultdict(list)
        for i in range(bs):
            uid2idx[index[i]].append(i)
        bonus = [0.0] * bs
        for _uid, idxs in uid2idx.items():
            N = len(idxs)
            if N < 2:
                continue
            if accs is not None and all(accs[i] > 0 for i in idxs):
                # All-correct group: base GRPO advantage is 0 (constant reward), so the
                # always-≤0 batch penalty would dominate and push DOWN the correct
                # answers just for repeating. Zero the bonus -> recover plain GRPO on
                # this group (Song et al. 2025, arXiv:2509.06941, §3.2).
                continue
            for i in idxs:
                ai = outcomes[i]
                if ai is None:
                    continue
                same = sum(1 for j in idxs if j != i and outcomes[j] == ai)
                bonus[i] = -(1.0 / N) * same
        bonus_t = torch.tensor(bonus, dtype=torch.float32,
                               device=responses.device).unsqueeze(-1)
        advantages = base_adv + batch_explore_c * bonus_t * response_mask
        data.batch['advantages'] = advantages
        data.batch['returns'] = advantages
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())


    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()
    
    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)

        # Lightweight pass@k eval: keep only the first N rows per data_source
        # (default 1000), but take ALL of bamboogle (it has only 125). The full
        # test.parquet (~51k across 7 sources) is too slow to sample-16× every
        # test_freq steps; this trims it to a fixed, comparable per-dataset subset.
        if self.config.get('val_passk', False):
            import pandas as _pd
            cap = int(self.config.get('val_passk_cap', 1000))
            full_sources = set(self.config.get('val_passk_full_sources', ['bamboogle']))
            df = self.val_dataset.dataframe
            parts = []
            for ds, sub in df.groupby('data_source', sort=False):
                parts.append(sub if ds in full_sources else sub.head(cap))
            self.val_dataset.dataframe = _pd.concat(parts).reset_index(drop=True)
            print("[val_passk] per-source eval sizes: " +
                  str(self.val_dataset.dataframe['data_source'].value_counts().to_dict()))
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        # drop_last=False so no eval questions are silently dropped. For pass@k the
        # batch is repeated n_samples× before generation; n_samples is a multiple of
        # the world size here so divisibility still holds for the last partial batch.
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=not self.config.get('val_passk', False),
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        if self.config.get('val_passk', False):
            return self._validate_passk()
        import torch
        reward_tensor_lst = []
        data_source_lst = []

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                # test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_tensor = self.val_reward_fn(test_batch)

                    reward_tensor_lst.append(reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def _validate_passk(self):
        """Lightweight pass@k validation.

        For every eval question, sample n_samples rollouts at temperature 1
        (do_sample=True → uses rollout.temperature) through the multi-turn search
        loop, score each with the EM reward, then report the unbiased pass@k
        estimate per data_source (and a macro 'avg' over questions):

            pass@k = 1 - C(n-c, k) / C(n, k)      (n samples, c correct)

        The eval set is the per-source-capped subset built in _create_dataloader.
        """
        import torch
        from math import comb

        n_samples = int(self.config.get('val_passk_n', 16))
        ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= n_samples]

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url=self.config.retriever.url,
            topk=self.config.retriever.topk,
        )
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=True,
        )

        # data_source -> k -> list of per-question pass@k values
        per_source_passk = defaultdict(lambda: defaultdict(list))

        for batch_dict in self.val_dataloader:
            test_batch: DataProto = DataProto.from_single_dict(batch_dict)
            # n_samples independent rollouts per question, kept in contiguous
            # blocks of n_samples (interleave=True): q0×n, q1×n, ...
            test_batch = test_batch.repeat(repeat_times=n_samples, interleave=True)

            test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': True,   # temperature-1 sampling (rollout.temperature)
                'validate': True,
            }

            first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
            final_gen_batch_output = generation_manager.run_llm_loop(
                gen_batch=test_gen_batch,
                initial_input_ids=first_input_ids,
            )
            test_batch = test_batch.union(final_gen_batch_output)
            for key in test_batch.batch.keys():
                test_batch.batch[key] = test_batch.batch[key].long()

            # EM reward per sample (val_reward_fn has lambda_div=0 → pure accuracy).
            reward_tensor = self.val_reward_fn(test_batch)
            seq_reward = reward_tensor.sum(-1).cpu()  # (num_q * n_samples,)
            sources = test_batch.non_tensor_batch.get(
                'data_source', np.array(['unknown'] * seq_reward.shape[0]))

            num_q = seq_reward.shape[0] // n_samples
            seq_reward = seq_reward[:num_q * n_samples].view(num_q, n_samples)
            for qi in range(num_q):
                src = sources[qi * n_samples]
                c = int((seq_reward[qi] > 0).sum().item())  # correct out of n_samples
                for k in ks:
                    if n_samples - c < k:
                        pk = 1.0
                    else:
                        pk = 1.0 - comb(n_samples - c, k) / comb(n_samples, k)
                    per_source_passk[src][k].append(pk)

        metric_dict = {}
        all_by_k = defaultdict(list)
        for src, kd in per_source_passk.items():
            for k, vals in kd.items():
                metric_dict[f'val/{src}/pass@{k}'] = float(np.mean(vals))
                all_by_k[k].extend(vals)
        for k, vals in all_by_k.items():
            metric_dict[f'val/avg/pass@{k}'] = float(np.mean(vals))
        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator in ('grpo', 'grpo_step_level', 'grpo_step_combined', 'grpo_passk', 'grpo_batch'):
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _compute_lambda_div(self, metrics):
        """Resolve the current lambda_div from the configured schedule.

        Returns (cur_lambda_div, per_group_constraint). Writes diagnostic
        metrics into `metrics`. Called once per step BEFORE the reward fn so
        pre_norm reward combination can use the same lambda the post_norm
        advantage combination would.
        """
        l_type = self.config.get('lambda_div_schedule_type', 'linear')
        per_group_constraint = (l_type == 'per_group_constraint')
        if per_group_constraint:
            # λ_i computed per-prompt inside compute_advantage. Placeholder.
            cur_lambda_div = float('nan')
        elif l_type == 'entropy_binary':
            # Step function: λ_max while current entropy ≤ first-observed
            # baseline, λ_min above it. Step 1: no entropy yet → λ=0.
            lam_max = float(self.config.get('lambda_div_max', 1.0))
            lam_min = float(self.config.get('lambda_div_min', 0.0))
            last_H = self.last_entropy_loss
            if last_H is None:
                cur_lambda_div = 0.0
                metrics['actor/lambda_div_entropy_used'] = float('nan')
                metrics['actor/lambda_div_target_entropy'] = float('nan')
            else:
                H = abs(float(last_H))
                if self.baseline_entropy is None:
                    self.baseline_entropy = H
                target_H = self.baseline_entropy
                cur_lambda_div = lam_max if H <= target_H else lam_min
                metrics['actor/lambda_div_entropy_used'] = H
                metrics['actor/lambda_div_target_entropy'] = target_H
        elif l_type == 'entropy_adaptive':
            lam_max = float(self.config.get('lambda_div_max', 1.0))
            lam_min = float(self.config.get('lambda_div_min', 0.0))
            last_H = self.last_entropy_loss
            if last_H is None:
                cur_lambda_div = 0.0
                metrics['actor/lambda_div_entropy_used'] = float('nan')
                metrics['actor/lambda_div_target_entropy'] = float('nan')
            else:
                H = abs(float(last_H))
                if self.baseline_entropy is None:
                    self.baseline_entropy = H
                target_H = self.baseline_entropy
                if H <= target_H:
                    cur_lambda_div = lam_max
                else:
                    lam = lam_max * (target_H / max(H, 1e-8))
                    cur_lambda_div = max(lam_min, min(lam_max, lam))
                metrics['actor/lambda_div_entropy_used'] = H
                metrics['actor/lambda_div_target_entropy'] = target_H
        else:
            l_start = self.config.get('lambda_div_start', None)
            l_end   = self.config.get('lambda_div_end',   None)
            l_steps = self.config.get('lambda_div_schedule_steps', None)
            if l_start is not None and l_end is not None and l_steps:
                progress = min(1.0, float(self.global_steps) / float(l_steps))
                if l_type == 'cosine':
                    import math as _math
                    cur_lambda_div = float(l_end) + (float(l_start) - float(l_end)) \
                                     * 0.5 * (1.0 + _math.cos(_math.pi * progress))
                else:
                    cur_lambda_div = float(l_start) + (float(l_end) - float(l_start)) * progress
            else:
                cur_lambda_div = self.config.get('lambda_div', 0.0)
        metrics['actor/lambda_div'] = cur_lambda_div
        return cur_lambda_div, per_group_constraint

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        # last observed actor entropy (set after each actor update). Used by the
        # 'entropy_adaptive' lambda_div schedule on the *following* step.
        self.last_entropy_loss = None
        # baseline target entropy for 'entropy_adaptive' schedule. Set to the
        # first observed entropy (from step 1's actor update). Used as the
        # divisor in subsequent steps so the controller adapts to whatever
        # entropy regime this run actually starts in.
        self.baseline_entropy = None
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)

                        # batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        #                                         dtype=object)
                        batch.non_tensor_batch['uid'] = batch.non_tensor_batch['index'].copy()
                                            
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(final_gen_batch_output)

                    ####################
                    ####################

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Resolve lambda_div BEFORE the reward fn so pre_norm
                        # combination (R_total = R_answer + λ·R_diversity) can use
                        # it. post_norm uses the same value later in compute_advantage.
                        cur_lambda_div, per_group_constraint = self._compute_lambda_div(metrics)
                        self.reward_fn.cur_lambda_div = cur_lambda_div

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor
                        # Reward-component breakdown (R_acc/R_fmt/R_div) for wandb,
                        # stashed by the RewardManager (em_fmt_doc combine mode).
                        metrics.update(batch.meta_info.get('reward_components', {}))

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process.
                        # lambda_div (cur_lambda_div) and per_group_constraint were
                        # resolved before the reward fn via _compute_lambda_div.
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  lambda_div=cur_lambda_div,
                                                  per_group_constraint=per_group_constraint,
                                                  passk_k=int(self.config.get('passk_k', 2)),
                                                  passk_fmt_weight=float(self.config.get('passk_fmt_weight', 1.0)),
                                                  passk_reward_mode=str(self.config.get('passk_reward_mode', 'joint_binary')),
                                                  batch_explore_c=float(self.config.get('batch_explore_c', 0.2)),
                                                  allwrong_adv_scale=bool(self.config.get('allwrong_adv_scale', False)),
                                                  allcorrect_center_only=bool(self.config.get('allcorrect_center_only', False)),
                                                  grpo_std_floor=float(self.config.get('grpo_std_floor', 0.0)))
                        _aws = batch.meta_info.get('allwrong_adv_scale_stats')
                        if _aws is not None:
                            metrics['actor/allwrong_adv_scale/c']        = _aws['c']
                            metrics['actor/allwrong_adv_scale/n_scaled'] = _aws['n_scaled']
                        _av0 = batch.meta_info.get('allcorrect_center_only_stats')
                        if _av0 is not None:
                            metrics['div/center_only_groups'] = _av0['n_groups']
                            metrics['div/center_only_spread'] = _av0['mean_spread']
                            print(f"    (8) center-only (allcorrect): {_av0['n_groups']} groups "
                                  f"mean-centered (no /std) | 그룹 내 보상 폭 평균 "
                                  f"{_av0['mean_spread']:.4f}")
                        _sg = batch.meta_info.get('grpo_sigma_stats')
                        if _sg and _sg.get('n_groups'):
                            _n = _sg['n_groups']
                            metrics['div/sigma_below_floor_frac'] = _sg['n_below'] / _n
                            metrics['div/sigma_zero_frac'] = _sg['n_zero'] / _n
                            metrics['div/sigma_p50'] = _sg['p50']
                            metrics['div/sigma_max_amp'] = _sg['max_amp']
                            print(f"    (9) group-σ: p50 {_sg['p50']:.4f} | "
                                  f"σ<{_sg['watch']:.3f} 그룹 {_sg['n_below']}/{_n} "
                                  f"({_sg['n_below']/_n*100:.1f}%) | σ=0 {_sg['n_zero']} | "
                                  f"최대 증폭 x{_sg['max_amp']:.1f}")
                        _mag = batch.meta_info.get('mixed_adv_gap')
                        if _mag is not None:
                            metrics['div/mixed_adv_gap']     = _mag['gap']
                            metrics['div/mixed_adv_correct'] = _mag['correct']
                            metrics['div/mixed_adv_wrong']   = _mag['wrong']
                            print(f"    (5) mixed-group adv  : correct={_mag['correct']:+.4f} "
                                  f"vs wrong={_mag['wrong']:+.4f} | gap={_mag['gap']:+.4f} "
                                  f"(over {_mag['n_groups']} mixed groups)"
                                  f"   ← 0으로 붕괴하면 다양성이 정확도 신호를 압축")
                        if per_group_constraint:
                            stats = batch.meta_info.get('per_group_lambda_stats', {})
                            lams = stats.get('lambdas', [])
                            if lams:
                                import numpy as _np
                                arr = _np.array(lams, dtype=float)
                                metrics['actor/lambda_div_per_group/mean']   = float(arr.mean())
                                metrics['actor/lambda_div_per_group/min']    = float(arr.min())
                                metrics['actor/lambda_div_per_group/max']    = float(arr.max())
                                metrics['actor/lambda_div_per_group/median'] = float(_np.median(arr))
                                metrics['actor/lambda_div_per_group/p25']    = float(_np.percentile(arr, 25))
                                metrics['actor/lambda_div_per_group/p75']    = float(_np.percentile(arr, 75))
                            metrics['actor/lambda_div_per_group/n_all_correct']   = stats.get('n_all_correct', 0)
                            metrics['actor/lambda_div_per_group/n_all_wrong']     = stats.get('n_all_wrong', 0)
                            metrics['actor/lambda_div_per_group/n_div_degenerate'] = stats.get('n_div_degenerate', 0)
                            metrics['actor/lambda_div_per_group/n_constraint']    = stats.get('n_constraint', 0)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                                batch, metrics = self._create_loss_mask(batch, metrics)
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)
                        # Cache last entropy for next step's entropy_adaptive λ schedule.
                        if 'actor/entropy_loss' in actor_output_metrics:
                            self.last_entropy_loss = actor_output_metrics['actor/entropy_loss']

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # No forced final validation: with test_freq dividing the total
                    # step count, the last step is already validated by the periodic
                    # block above. Just stop training here.
                    return
    
    def _answer_content_idx(self, resp_row, rl, ans_id, gt_id):
        """Response-token indices of the <answer>…</answer> CONTENT span(s).
        Anchor on the single 'answer' token (ans_id); classify OPEN vs CLOSE by the
        preceding token decoding to '…<' vs '…</' (robust to the leading-space merge
        that shifts the '</' token in-context). BPE often merges the tag's '>' with
        the following content ('>New'); when it does (ids[p+1] != gt_id '>') that
        merged token is part of the answer, so content starts at it. Tags stay
        unmasked; only a is returned -> loss updates π(y|x) but not π(a|y,x)."""
        ids = resp_row[:rl].tolist()
        out, open_start, L = [], None, len(ids)
        for p in range(1, L):
            if ids[p] != ans_id:
                continue
            nxt = self.tokenizer.decode([ids[p + 1]]) if p + 1 < L else ""
            if not nxt.startswith(">"):          # 'answer' not followed by '>' -> not a tag
                continue
            prev = self.tokenizer.decode([ids[p - 1]]).rstrip()
            if prev.endswith("</"):              # close: </answer>
                if open_start is not None:
                    out.extend(range(open_start, p - 1))   # up to the '</' token (exclusive)
                    open_start = None
            elif prev.endswith("<"):             # open: <answer>
                open_start = p + 2 if ids[p + 1] == gt_id else p + 1
        return out

    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]

        loss_mask = batch.batch['info_mask'][:, -response_length:]

        if self.config.get('mask_answer_span', False):
            # OBE (Song et al. 2025, §3.1): drop policy gradient through the final-answer
            # span so training updates π(y|x) but NOT π(a|y,x) — prevents bonus-hacking by
            # rewriting the answer string to dodge the batch repetition penalty.
            loss_mask = loss_mask.clone()
            responses = batch.batch['responses']
            tag = self.tokenizer.encode("<answer>", add_special_tokens=False)  # [..,'answer','>']
            ans_id, gt_id = tag[-2], tag[-1]
            n_ans = 0
            for i in range(responses.size(0)):
                rl = int(response_mask[i].sum().item())
                for q in self._answer_content_idx(responses[i], rl, ans_id, gt_id):
                    loss_mask[i, q] = 0
                    n_ans += 1
            metrics['state_tokens/answer_masked'] = n_ans

        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })

        return batch, metrics
