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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, qa_em_format
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import math
import random
import hashlib
import unicodedata
import numpy as np
from collections import defaultdict, Counter

def _select_rm_score_fn(data_source):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        return qa_em.compute_score_em
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.

    Returns the OUTCOME (EM) reward tensor as the function output (last-token reward,
    shape (bs, response_length)). Additionally mutates `data.batch` to add:
        - `step_level_scores`: (bs, response_length) float, raw diversity reward
          (div_score in [0,1]) placed at each step's think+search tokens, accuracy-
          gated. The lambda_div multiplier is applied AFTER group normalization in
          compute_advantage (in ray_trainer), so that lambda_div actually controls
          the relative gradient strength.
        - `stratum_step_id`: (bs, response_length) long, group ID > 0 at the same
          positions, 0 elsewhere. Trajectories sharing (prompt, search_count, step_k)
          share a group ID.

    Here, `lambda_div` is just an enable toggle: > 0 means compute diversity, == 0
    means skip entirely.
    """

    def __init__(self, tokenizer, num_examine, format_score=0., lambda_div=0.25, n_agent=8,
                 diversity_type='doc_set_unique',
                 embed_model_name='sentence-transformers/all-MiniLM-L6-v2',
                 embed_device=None,
                 combine_mode='post_norm',
                 diversity_grouping='trajectory',
                 useful_weight=False,
                 useful_probe_path='Qwen/Qwen2.5-3B',
                 useful_device=None,
                 useful_probe_url=None,
                 useful_hybrid=False,
                 stage1_block=False,
                 budget_map=None,
                 budget_default=2,
                 budget_ungate=False,
                 budget_ungate_gold=False,
                 budget_allrollouts=False,
                 budget_alltraj=False,
                 budget_correct_nolimit=False,
                 budget_mean_div=False,
                 budget_lone_pool=False,
                 div_sum_mode=False,
                 div_capped=False,
                 div_topk=3,
                 div_lambda_map=None,
                 div_lambda_default=1.0 / 3.0,
                 div_off_all_correct=False,
                 div_gold_only=False,
                 div_gold_watch=False,
                 use_format_reward=True,
                 bleu_sigma=0.65,
                 guesser_gate=True,
                 gq_path_nq=None,
                 gq_path_hotpotqa=None) -> None:
        # Budget-gated soft-novelty (diversity_type='budget'): per-question search
        # budget = #supporting paragraphs, derived from data_source. HotpotQA is 2-hop
        # (2 supporting paras), NQ single-hop (1). A rollout that searches MORE than its
        # budget is excluded from the diversity game (over-search earns no R_div).
        self.budget_map = dict(budget_map) if budget_map else {
            'nq': 1, 'hotpotqa': 2, 'triviaqa': 1, 'popqa': 1,
            '2wikimultihopqa': 2, 'musique': 3, 'bamboogle': 2}
        self.budget_default = int(budget_default)
        # budget_ungate: asymmetric conditional-ungate. False = original budget gate
        # (R_div only on correct & within-budget, <2-eligible→0). True = on ALL-WRONG
        # groups drop the R_acc condition (pool = within-budget wrong rollouts) and drop
        # the <2 guard everywhere (lone-pool keeps raw R_div). See _budget_soft_diversity.
        self.budget_ungate = bool(budget_ungate)
        # budget_ungate_gold: gold-anchored ungate. On ALL-WRONG groups, the pool is
        # further restricted to rollouts whose RETRIEVED DOCS actually contain the
        # gold answer (EM-normalized substring match). Rewards decorrelated retrieval
        # ONLY among rollouts that reached the evidence but failed to answer — i.e.
        # anchored diversity (coverage), not blind dispersion. Implies the ungate
        # pool structure regardless of budget_ungate. See _budget_soft_diversity.
        self.budget_ungate_gold = bool(budget_ungate_gold)
        # budget_allrollouts=True: ABLATION — drop the correctness gate entirely. The
        # budget-soft diversity is computed AND paid over ALL within-budget rollouts
        # (correct AND wrong) in EVERY group. Isolates the conditional-shaping variable
        # vs train_ours.sh (DIVER Fig.7 "all responses" variant). Takes priority over
        # budget_ungate/_gold. Over-budget rollouts still excluded (over-search axis).
        self.budget_allrollouts = bool(budget_allrollouts)
        # budget_alltraj=True: ABLATION — no correctness gate AND no budget filter. The
        # pool is EVERY rollout in the group (correct/wrong, in/over budget); m_d counts
        # over all of them. The most permissive variant: diversity measured & paid on
        # literally all trajectories. Takes priority over budget_allrollouts/ungate/gold.
        self.budget_alltraj = bool(budget_alltraj)
        # budget_correct_nolimit=True: ABLATION — KEEP the correctness gate but DROP the
        # budget filter. pool = {correct rollouts}, regardless of over-search; m_d counts
        # over all correct rollouts. Isolates the budget-filter axis vs train_ours.sh.
        self.budget_correct_nolimit = bool(budget_correct_nolimit)
        # budget_mean_div=True: CONTROL — keep the same pool selection, but replace each
        # pool member's per-rollout R_div with the POOL MEAN R_div (constant within the
        # pool). Kills the within-pool diversity DIFFERENTIATION, leaving only a flat
        # bonus for pool membership. Isolates "does per-rollout diversity do work, or is
        # it just extra reward for correct+budget?". NB: under GRPO a flat bonus on an
        # all-correct(binary) group is affine-invariant -> collapses to plain GRPO;
        # differs only via budget-compliance on groups with over-budget correct rollouts.
        self.budget_mean_div = bool(budget_mean_div)
        # budget_lone_pool=True relaxes the ORIGINAL budget gate's <2 guard to min_pool=1:
        # a lone correct-&-within-budget rollout (size-1 pool) keeps its raw R_div=1.0
        # (all m_d=1) instead of being zeroed. Only affects the original gate branch
        # (not budget_ungate/_gold, which set their own min_pool). Default False = current.
        self.budget_lone_pool = bool(budget_lone_pool)
        # guesser_gate=False 는 evidence 계열(evidence_exempt / evidence_doc_exempt /
        # gq_doc_exempt) 의 E' 자격 게이트를 끈다. 기본 True = 기존 동작 (E' 만 지급).
        #   True  — pool = E' = {i in E : D_i ∩ Evd(group) != {}}; count(d) 도 E' 기준.
        #           evidence 를 하나도 못 건진 정답 궤적은 R_div=0. GRPO 는 그룹 내
        #           상대값이므로 이 0 은 "보너스 미지급" 이 아니라 같은 그룹의 다른
        #           정답 궤적 대비 음수 advantage 로 들어간다.
        #   False — pool = E = {정답 AND 예산 내}. 자격 판정 없이 전원 지급하고
        #           count(d) 도 E 전체에서 센다 (guesser 문서도 분모에 반영되므로
        #           novelty 가 "남들이 안 찾은 문서" 라는 뜻을 되찾는다).
        #           그룹 전체가 evidence 미검출인 경우도 전원 0 대신 기존 공식
        #           (면제 없는 순수 1/count) 으로 환원한다 — 같은 병리의 그룹 단위
        #           버전이라 게이트를 끄면 함께 꺼지는 것이 일관적이다.
        # 게이트를 꺼도 n_guesser 통계는 계속 집계한다 (모니터링용).
        self.guesser_gate = bool(guesser_gate)
        # div_sum_mode=True replaces the per-rollout MEAN novelty with a per-dataset
        # SCALED SUM:  R_div_i = lambda_ds * sum_{d in D_i} 1/m_d   (was: .../|D_i|).
        # Dropping the 1/|D_i| stops a rollout from being penalised for holding more
        # docs; lambda_ds restores a comparable scale per dataset (roughly 1/expected
        # |D_i|, i.e. 1/(topk*budget_ds)): NQ 1 hop -> 1/3, HotpotQA 2 hops -> 1/6.
        # Unmapped data_sources fall back to div_lambda_default. Default False keeps
        # the original mean form so every other budget run is untouched.
        self.div_sum_mode = bool(div_sum_mode)
        # div_capped=True: denominator max(|D_i|, topk) instead of |D_i| or 1/lambda.
        # NOTE the monotonicity property differs from div_sum_mode. Adding a doc of
        # weight w=1/m_d to a rollout holding |D| docs:
        #   |D| <  topk -> denominator pinned at topk, R always rises by w/topk.
        #   |D| >= topk -> max() picks |D|, so this IS the mean form and R rises only
        #                  when w > S/|D|; a below-average doc still DILUTES.
        # So capped removes the "extra doc" tax only below topk. div_sum_mode (constant
        # denominator 1/lambda) is the variant where monotonicity is unconditional.
        # Precedence: div_capped > div_sum_mode > mean.
        self.div_capped = bool(div_capped)
        self.div_topk = int(div_topk)
        self.div_lambda_map = dict(div_lambda_map) if div_lambda_map else {
            'nq': 1.0 / 3.0, 'hotpotqa': 1.0 / 6.0}
        self.div_lambda_default = float(div_lambda_default)
        # div_off_all_correct=True zeroes R_div on ALL-CORRECT groups. There R_acc is
        # constant, so the GRPO advantage carries no accuracy signal and R_div becomes
        # the ONLY thing differentiating rollouts — it then pushes probability mass
        # toward whichever already-correct rollout happened to be most novel, which is
        # a plausible pass@1 leak. Off by default.
        self.div_off_all_correct = bool(div_off_all_correct)
        # GOLD-ONLY NUMERATOR (diversity_type='budget'):
        #     R_div_i = (1/|D_i|) Σ_{d ∈ D_i ∩ GoldBearing_i} 1/m_d
        # The sum runs ONLY over docs that CONTAIN the gold answer; the denominator
        # stays |D_i| = ALL retrieved docs. Rationale: under the all-docs numerator,
        # answer-bearing docs are retrieved by most pool members (high m_d → low
        # 1/m_d) while junk docs are rare (low m_d → high 1/m_d), so maximizing R_div
        # pushed the policy AWAY from the evidence it needs. Restricting the numerator
        # to gold-bearing docs makes the only way to raise R_div "reach the answer via
        # a doc few siblings found". Keeping |D_i| in the denominator still charges for
        # padding the trajectory with junk. m_d is unchanged (counted over all pool
        # docs) — gold-bearing is a property of the doc, so m_d for a gold-bearing doc
        # is identical either way.
        self.div_gold_only = bool(div_gold_only)
        # div_gold_watch: parse the per-rollout gold-bearing doc sets PURELY for the
        # div/gold_* diagnostics, on arms whose reward does not need them (GRPO,
        # DIVER, batch). Costs one extra decode+regex pass over the observation
        # chunks per rollout, so it is opt-in; gold-aware arms get it for free.
        self.div_gold_watch = bool(div_gold_watch)
        # evidence containment 판정 캐시: (prompt_hash, doc identity) -> bool.
        # 같은 질문의 G개 rollout 이 같은 문서를 자주 회수하므로 적중률이 높다.
        self._evd_cache = {}
        # use_format_reward=False drops R_fmt from R_total in the em_fmt_doc combine
        # (R_total = R_acc + 1[·]·λ·R_div only). Default True keeps existing behavior.
        # fmt_scores is still COMPUTED and logged (diagnostics) — only the placement drops it.
        self.use_format_reward = bool(use_format_reward)
        # DIVER Balanced Shaping (diversity_type='bleu'): clip TD to [0, σ] to cap the
        # diversity bonus (DIVER §3.5, σ=0.65 in the paper). Applied inside _bleu_diversity.
        self.bleu_sigma = float(bleu_sigma)
        # Stage 1 (v2/v1 filter) granularity: False=per-doc, True=per-search-step
        # (top-k block). Block mode gates whole retrieval blocks (matches the
        # model's query->top-k action; resists over-search-driven doc inflation).
        self._v2_block = bool(stage1_block)
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score
        self.lambda_div = lambda_div
        self.n_agent = n_agent
        self._call_count = 0

        # Reward combination mode:
        #   'post_norm' — outcome and diversity are normalized SEPARATELY by GRPO
        #                 (each in its own group statistics), then summed at the
        #                 advantage level as outcome_adv + lambda_div * div_adv.
        #                 Requires algorithm.adv_estimator=grpo_step_level.
        #   'pre_norm'  — outcome and diversity are summed at the REWARD level
        #                 (r_i = base_r_i + traj_div_score_i), then a single
        #                 GRPO normalization is applied. No lambda. The combined
        #                 reward is placed at the last valid token (outcome-style),
        #                 and the step_level_tensor stays zero. Use with
        #                 algorithm.adv_estimator=grpo. Structurally avoids the
        #                 exp_007 answer-refusal mode because correct trajectories
        #                 always have r >= 1 (above mean if any wrong exist) so
        #                 their advantage is always positive — div only modulates
        #                 the magnitude, never flips the sign.
        #   'step_reward' — per-block COMBINED reward + per-block-position
        #                 normalization (multi-turn literal of the slide formula).
        #                 Each block's reward = R_acc + λ·1[acc]·div_k (search
        #                 blocks) or R_acc (final answer block); the reward is
        #                 written into step_level_scores and tagged with a stratum
        #                 id = (prompt, position) so that downstream
        #                 adv_estimator=grpo_step_combined normalizes the k-th
        #                 search blocks together, and the answer blocks together —
        #                 a SINGLE normalization per position, NOT a separate
        #                 outcome advantage. λ is baked into the reward here.
        #   'em_fmt_doc' — trajectory-level scalar reward (use adv_estimator=grpo):
        #                 R_total = R_acc + R_fmt + 1[R_acc=1]·λ·R_div, where
        #                   R_acc = EM (0/1),
        #                   R_fmt = is_valid_sequence (0/1, the v0.3 structure check),
        #                   R_div = |docs unique to this rollout| / |all unique docs
        #                           across the whole prompt group (ALL rollouts,
        #                           correct AND wrong)|.
        #                 One scalar per trajectory at the last token; GRPO then
        #                 normalizes the 5-rollout group. No per-step machinery.
        #   'passk' — Pass@k Training (Chen et al. 2025, arXiv:2508.10751). The
        #             advantage is designed ANALYTICALLY downstream (adv_estimator=
        #             grpo_passk) from the per-rollout binary EM success label and
        #             k. R_fmt is NOT mixed into the success label — it is stashed
        #             and added downstream as a centered shaping term:
        #               A_i = A_passk_i + (R_fmt_i − group_mean(R_fmt)).
        #             RewardManager here only computes/stashes R_acc and R_fmt.
        assert combine_mode in ('post_norm', 'pre_norm', 'step_reward', 'em_fmt_doc', 'passk',
                                'batch_outcome'), \
            f"Unknown combine_mode: {combine_mode}"
        self.combine_mode = combine_mode

        # Diversity grouping unit:
        #   'trajectory' — one diversity score per trajectory (D_i = union of all docs),
        #                  normalized within (prompt). Current default (exp_007/008/010).
        #   'step_prompt' — per-step diversity scoring, stratum = (prompt, step_k).
        #                  All trajectories of same prompt that reached step_k are
        #                  compared together (regardless of total n_search). Each
        #                  scored step's chunk gets its own advantage. Forces post_norm
        #                  style (advantage-level combination via grpo_step_level).
        #                  Variant of exp_002 with n_search dimension dropped.
        assert diversity_grouping in ('trajectory', 'step_prompt', 'step_prompt_nsearch'), \
            f"Unknown diversity_grouping: {diversity_grouping}"
        self.diversity_grouping = diversity_grouping

        # Diversity arm selection:
        #   'doc_set_unique'      — original info-access reward
        #   'reasoning_embedding' — embed first <think> per step (representation)
        #   'search_embedding'    — embed first <search> query per step (representation)
        #   'none'                — outcome reward only
        # Kept under a single flag so only one arm is ever active per run.
        # The two embedding arms share infrastructure (encoder, cache, scoring);
        # only the *text being embedded* differs.
        assert diversity_type in ('doc_set_unique', 'reasoning_embedding',
                                  'search_embedding', 'v2_shapley', 'v1_filtered',
                                  'meaningful', 'rgsn', 'budget', 'evidence_exempt',
                                  'evidence_doc_exempt', 'gq_doc_exempt',
                                  'bleu', 'gold_cov', 'gold_cov_all', 'none'), \
            f"Unknown diversity_type: {diversity_type}"
        self.diversity_type = diversity_type
        # --- 오프라인 G_q (diversity_type='gq_doc_exempt') ---
        # 질문(정규화 키) -> G_q 청크 md5 집합. 온라인 검출(title/containment)을
        # 대체하며, evd_exempt = evd_eligible = D_i ∩ G_q. G_q 미보유 질문은
        # 그룹 evidence 미검출 -> 기존 budget 공식 fallback (필터드 데이터에선 희귀).
        self._gq_md5 = None
        if gq_path_nq or gq_path_hotpotqa:
            import pickle as _pkl
            self._gq_md5 = {}
            for _ds, _p in (('nq', gq_path_nq), ('hotpotqa', gq_path_hotpotqa)):
                if not _p:
                    continue
                _d = _pkl.load(open(_p, 'rb'))
                for _v in _d.values():
                    self._gq_md5[self._gq_question_key(_v['question'])] = \
                        frozenset(c[2] for c in _v['chunks'])
                print(f"[RewardManager] G_q 로드: {_ds} {len(_d):,}질문 ({_p})")
        if diversity_type == 'gq_doc_exempt':
            assert self._gq_md5, "gq_doc_exempt 에는 gq_path_nq/gq_path_hotpotqa 필요"
        self.embed_model_name = embed_model_name
        self.embed_device = embed_device  # None => auto
        self._embed_model = None  # lazy
        self._embed_cache = {}    # md5(text) -> tensor (d,) on CPU

        # --- ΔP useful-weight (over-search suppression) ---
        # When on, each retrieved doc's soft-novelty credit is multiplied by
        # w_k = max(0, ΔP_k) where prefix ΔP_k = logP_ref(gold | …through the k-th
        #   <information>) − logP_ref(gold | …just before it) — the marginal answer
        #   info gained when that retrieval ARRIVES (keeps the turn's think+search,
        #   isolates the info content). Scored by a FROZEN base probe (Qwen2.5-3B),
        #   teacher-forced over gold, stop-grad. Chosen over leave-one-out: LOO
        #   under-credits multi-hop bridges a downstream block already covers
        #   (offline: 24% of useful new-gold steps LOO≤0; prefix halves that, with
        #   only 6% false-rescue of over-search). So over-search docs earn ~0 novelty.
        # The probe runs as a SEPARATE GPU http server (the Ray driver running this
        # RewardManager has NO GPU): set useful_probe_url to its /score endpoint.
        # Falls back to a local (possibly CPU) probe model when the url is unset.
        # useful_hybrid: peak-anchored weighting. peak = argmax over the rollout's
        # per-step prefix ΔP (= where the answer is "found"). Steps UP TO the peak
        # (the build-up chain incl. multi-hop bridges that score ΔP≈0) get full
        # credit (w=1); steps AFTER the peak (the over-search tail) get w=max(0,ΔP).
        # Fixes per-search ΔP zeroing whole correct multi-hop rollouts (bridges have
        # ΔP≈0 because a later answer-doc covers the gold / base can't infer the hop).
        # When False, plain per-search w=max(0,ΔP) (the prefix arm).
        self.useful_weight = bool(useful_weight)
        self.useful_probe_path = useful_probe_path
        self.useful_device = useful_device   # None => auto
        self.useful_probe_url = useful_probe_url
        self.useful_hybrid = bool(useful_hybrid)
        self._probe_model = None  # lazy
        self._probe_tok = None
        self._probe_device_resolved = None

    def _extract_docs_per_step(self, response_token_ids, response_info_mask):
        """Extract doc identity sets per genuine env-injected <information> block.

        Uses info_mask = 0 chunks to identify environment-injected observations.
        This is robust to the model fabricating <information> tags in its own
        think/search text (which info_mask correctly assigns to mask=1).

        Wikipedia is chunked, so the same title can appear across different chunks
        with different content. Doc identity therefore uses (title, content_hash)
        rather than title alone.

        Returns: list of sets of (title, content_md5) tuples, length = number of
        env-injected info chunks.
        """
        # info_mask=0 chunks = environment-injected observations
        obs_chunks = []
        in_obs = False
        start = None
        for i, m in enumerate(response_info_mask):
            mv = int(m)
            if mv == 0 and not in_obs:
                in_obs = True
                start = i
            elif mv == 1 and in_obs:
                in_obs = False
                obs_chunks.append((start, i))
        if in_obs:
            obs_chunks.append((start, len(response_info_mask)))

        per_step = []
        for s, e in obs_chunks:
            chunk_ids = response_token_ids[s:e]
            chunk_text = self.tokenizer.decode(chunk_ids)
            matches = re.findall(
                r'Doc \d+\(Title: ("?)([^"\n]*?)\1\)(.*?)(?=Doc \d+\(Title:|\Z)',
                chunk_text,
                re.DOTALL,
            )
            docs = set()
            for _q, title, content in matches:
                content_norm = ' '.join(content.split())
                content_hash = hashlib.md5(content_norm.encode('utf-8')).hexdigest()
                docs.add((title, content_hash))
            per_step.append(docs)
        return per_step

    def _support_titles(self, data_item):
        """HotpotQA: metadata.supporting_facts.title -> 정규화 title 집합.
        supporting_facts 가 없으면(NQ 등) None 을 돌려 containment 경로로 보낸다."""
        try:
            md = data_item.non_tensor_batch.get('metadata', None)
        except AttributeError:
            md = None
        if md is None or not isinstance(md, dict):
            return None
        sf = md.get('supporting_facts', None)
        if sf is None:
            return None
        titles = sf.get('title', None) if isinstance(sf, dict) else None
        if titles is None:
            return None
        out = set()
        for t in titles:
            if t is None:
                continue
            t = str(t)
            out.add(self._norm_title(t))
            out.add(re.sub(r'\s*\([^)]*\)\s*$', '', self._norm_title(t)).strip())
        return out or None

    @staticmethod
    def _norm_title(t):
        """lowercase + unicode(NFKC) + whitespace 정규화. HotpotQA supporting_facts
        title 을 corpus 문서 title 과 맞추기 위한 것."""
        return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', t).lower().strip())

    @staticmethod
    def _gq_question_key(q):
        """G_q 조회 키 — 구축 스크립트(qnorm)와 동일 정규화."""
        return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', '', q.lower())).strip()

    def _gq_key_from_prompt(self, prompt_str):
        # 템플릿의 'Question:' 은 항상 첫 번째 등장 — rfind 를 쓰면 질문 텍스트
        # 자체가 'Question:' 으로 시작하는 특이 케이스에서 안쪽을 잡아 키가 어긋남.
        idx = prompt_str.find('Question:')
        if idx == -1:
            return ''
        q = prompt_str[idx + len('Question:'):].split('\n', 1)[0]
        return self._gq_question_key(q)

    def _div_normalize(self, novelty, n_docs, ds):
        """R_div 정규화 — budget 분기와 동일 규약 (capped > sum(λ_ds) > mean)."""
        if self.div_capped:
            return novelty / max(n_docs, self.div_topk)
        if self.div_sum_mode:
            return novelty * self.div_lambda_map.get(ds, self.div_lambda_default)
        return novelty / n_docs

    def _docs_and_evidence_per_step(self, response_token_ids, response_info_mask,
                                    golds_norm, sup_titles, prompt_hash):
        """obs chunk 를 한 번만 디코드해서
            (per-call 문서집합, Evd_exempt, Evd_eligible) 를 반환한다.

        두 evidence 집합의 용도가 다르다:
          Evd_exempt   — exempt call 판정 + w(d)=1 면제 대상
          Evd_eligible — E' 자격(=guesser 여부) 판정에만 사용

        HotpotQA (sup_titles 주어짐):
            Evd_exempt   = supporting_facts title 매칭 문서
            Evd_eligible = title 매칭 ∪ answer containment
            단 정답이 yes/no 로 정규화되면 containment 생략 (모든 문서가 걸려
            전면 면제가 되는 것을 막는다; 학습셋 HotpotQA 의 6.1%).
        NQ 등 (sup_titles 없음):
            Evd_exempt = Evd_eligible = answer containment  (동작 변화 없음)

        containment/title 판정은 (prompt_hash, doc identity) 로 캐시한다 — 같은
        질문의 G개 rollout 이 같은 문서를 자주 회수하므로 적중률이 높다.
        """
        # yes/no 예외는 containment 가 '추가' 자격 소스인 HotpotQA 경로에만 적용한다.
        skip_contain = (sup_titles is not None and bool(golds_norm)
                        and any(g in ('yes', 'no') for g in golds_norm))
        obs_chunks, in_obs, start = [], False, None
        for i, m in enumerate(response_info_mask):
            mv = int(m)
            if mv == 0 and not in_obs:
                in_obs, start = True, i
            elif mv == 1 and in_obs:
                in_obs = False
                obs_chunks.append((start, i))
        if in_obs:
            obs_chunks.append((start, len(response_info_mask)))

        per_step, evd_exempt, evd_eligible = [], set(), set()
        for s, e in obs_chunks:
            chunk_text = self.tokenizer.decode(response_token_ids[s:e])
            matches = re.findall(
                r'Doc \d+\(Title: ("?)([^"\n]*?)\1\)(.*?)(?=Doc \d+\(Title:|\Z)',
                chunk_text, re.DOTALL)
            docs = set()
            for _q, title, content in matches:
                content_norm = ' '.join(content.split())
                ident = (title, hashlib.md5(content_norm.encode('utf-8')).hexdigest())
                docs.add(ident)
                key = (prompt_hash, ident)
                flags = self._evd_cache.get(key)
                if flags is None:
                    # 제목 매칭은 회수 문서 쪽도 disambiguator 를 벗겨 비교한다
                    # (sup_titles 에는 양쪽 형태가 다 들어 있음). cf_replay 와 대칭.
                    tn = self._norm_title(title)
                    tn_nd = re.sub(r'\s*\([^)]*\)\s*$', '', tn).strip()
                    t_hit = (sup_titles is not None
                             and (tn in sup_titles or tn_nd in sup_titles))
                    c_hit = False
                    if golds_norm and not skip_contain:
                        # containment 은 title+content — 검증 replay(verify_b_neg,
                        # cf_replay)의 is_evd 와 동일 범위.
                        cn = qa_em_format.normalize_answer(title + ' ' + content_norm)
                        c_hit = any(gn in cn for gn in golds_norm)
                    flags = (t_hit, c_hit)
                    self._evd_cache[key] = flags
                t_hit, c_hit = flags
                if sup_titles is not None:
                    if t_hit:
                        evd_exempt.add(ident)
                    if t_hit or c_hit:
                        evd_eligible.add(ident)
                elif c_hit:                      # NQ: 두 집합이 동일
                    evd_exempt.add(ident)
                    evd_eligible.add(ident)
            per_step.append(docs)
        return per_step, evd_exempt, evd_eligible

    def _gold_bearing_docs(self, response_token_ids, response_info_mask, golds):
        """Set of (title, content_md5) doc identities whose TEXT (Title + content)
        contains a gold answer (EM-normalized, word-boundary match). Doc identity
        (title, content_md5) matches _extract_docs_per_step, so same-title but
        different-chunk docs are DISTINCT (content_md5 differs). Match is per-DOC
        over the whole doc text (title AND content) so a gold appearing only in the
        title still counts; word-boundary so short golds don't match inside longer
        words (gold "cat" ∌ "category"). Uses info_mask (not text-tag matching),
        robust to the model fabricating <information> tags. Used for
        (a) budget_ungate_gold pool membership (rollout eligible iff it holds >=1
        gold-bearing doc) and (b) the gold-share junk-padding canary."""
        if golds is None:
            return set()
        if isinstance(golds, str):
            golds = [golds]     # single answer; a numpy array / list iterates as-is
        golds_norm = []
        for g in golds:         # NB: avoid `golds or []` — ambiguous for numpy arrays
            if g is None:
                continue
            gn = qa_em_format.normalize_answer(g)
            if gn:                      # drop empty-normalizing golds (articles/punct)
                golds_norm.append(gn)
        if not golds_norm:
            return set()
        # info_mask=0 chunks = environment-injected observations
        obs_chunks = []
        in_obs = False
        start = None
        for i, m in enumerate(response_info_mask):
            mv = int(m)
            if mv == 0 and not in_obs:
                in_obs = True
                start = i
            elif mv == 1 and in_obs:
                in_obs = False
                obs_chunks.append((start, i))
        if in_obs:
            obs_chunks.append((start, len(response_info_mask)))
        bearing = set()
        for s, e in obs_chunks:
            chunk_text = self.tokenizer.decode(response_token_ids[s:e])
            matches = re.findall(
                r'Doc \d+\(Title: ("?)([^"\n]*?)\1\)(.*?)(?=Doc \d+\(Title:|\Z)',
                chunk_text,
                re.DOTALL,
            )
            for _q, title, content in matches:
                content_norm = ' '.join(content.split())
                # Match the gold against the WHOLE doc text (Title + content), so a
                # gold that appears only in the title still counts. Identity/dedup
                # stays content-based (content_md5), independent of the match text.
                doc_text = (title + ' ' + content_norm) if title else content_norm
                cn = qa_em_format.normalize_answer(doc_text)
                # Word-boundary match (not raw substring): a normalized gold must
                # appear as a whole token span, so short answers no longer spuriously
                # match inside longer words (gold "cat" must NOT match "category").
                # normalize_answer already strips punctuation/lowercases, so \b sits
                # on space bounds.
                if any(re.search(r'\b' + re.escape(gn) + r'\b', cn) for gn in golds_norm):
                    content_hash = hashlib.md5(content_norm.encode('utf-8')).hexdigest()
                    bearing.add((title, content_hash))
        return bearing

    def _find_step_token_ranges(self, response_info_mask, n_info_blocks):
        """Identify model-generated token chunks via info_mask (1 = model-gen,
        0 = info-block / padding). Each consecutive run of 1s is one chunk:
            - chunks[k] for k < n_info_blocks  →  step k's think+search range
            - chunks[-1] (if it exists beyond n_info_blocks)  →  final think+answer
              (no diversity reward attached to it)

        info_mask is the exact mask used by the trainer, so this avoids BPE
        re-tokenization drift entirely.

        Args:
            response_info_mask: 1-D iterable of {0,1}, length = valid_response_length
            n_info_blocks: number of <information> blocks observed in the response

        Returns: list of (start, end) tuples, length = n_info_blocks. Entries
        beyond the available chunks are None.
        """
        chunks = []
        in_gen = False
        start = None
        for i, m in enumerate(response_info_mask):
            mv = int(m)
            if mv == 1 and not in_gen:
                in_gen = True
                start = i
            elif mv == 0 and in_gen:
                in_gen = False
                chunks.append((start, i))
        if in_gen:
            chunks.append((start, len(response_info_mask)))

        # The first n_info_blocks model-gen chunks correspond to step 0..N-1's
        # think+search. Any extra chunk after that is the final answer.
        ranges = []
        for k in range(n_info_blocks):
            if k < len(chunks):
                ranges.append(chunks[k])
            else:
                ranges.append(None)
        return ranges

    @staticmethod
    def _find_all_gen_chunks(response_info_mask):
        """Return every model-generated token chunk as (start, end), i.e. each
        consecutive run of info_mask==1. For a complete trajectory the chunks are
        [think+search_0, think+search_1, ..., think+search_{n-1}, think+answer].
        So chunks[:n_info] are the search steps and chunks[n_info] (if present) is
        the final answer block."""
        chunks = []
        in_gen = False
        start = None
        for i, m in enumerate(response_info_mask):
            mv = int(m)
            if mv == 1 and not in_gen:
                in_gen = True
                start = i
            elif mv == 0 and in_gen:
                in_gen = False
                chunks.append((start, i))
        if in_gen:
            chunks.append((start, len(response_info_mask)))
        return chunks

    # Format-reward weight for _acc_fmt_reward (Jin et al. 2025, arXiv:2505.15117
    # Eq.3 lambda_f). The paper grid-searched {0.2, 0.4, 0.6, 0.8} and picked 0.2
    # for 3B, which is what this uses.
    LAMBDA_FMT = 0.2

    def _acc_fmt_reward(self, r_acc, r_fmt):
        """Four-quadrant EM x format reward (Jin et al. 2025 Eq.3), lambda_f = 0.1:

            correct + format ok   -> 1.0
            correct + format bad  -> 1 - lambda_f = 0.9
            wrong   + format ok   -> lambda_f     = 0.1
            wrong   + format bad  -> 0.0

        Replaces the earlier additive R_acc + R_fmt, whose {0,1,1,2} spacing is
        equivalent to lambda_f = 0.5 under GRPO's std normalization — i.e. it rated
        a well-formed wrong answer exactly as high as a badly-formed correct one.
        use_format_reward=False collapses this to plain EM.
        """
        if not self.use_format_reward:
            return r_acc
        if r_acc > 0:
            return 1.0 if r_fmt > 0 else 1.0 - self.LAMBDA_FMT
        return self.LAMBDA_FMT if r_fmt > 0 else 0.0

    def _policy_only_text(self, response_token_ids, response_info_mask):
        """Decode ONLY the model-generated spans (info_mask==1), joined by a space.

        Same identity rule as _extract_docs_per_step: what the environment injected
        is info_mask==0, everything the policy emitted is info_mask==1. A regex over
        the decoded string cannot make that distinction — it also strips
        <information> blocks the MODEL fabricated (which are policy tokens and must
        stay), and an unclosed model-side <information> pairs with the environment's
        next </information>, swallowing the model's own think/search in between.
        """
        parts = []
        for s, e in self._find_all_gen_chunks(response_info_mask):
            parts.append(self.tokenizer.decode(response_token_ids[s:e]))
        return ' '.join(parts)

    # ------------------------------------------------------------------
    # Reasoning-embedding diversity helpers (new ablation arm)
    # ------------------------------------------------------------------
    def _get_embed_model(self):
        """Lazy-load + freeze the sentence-transformers encoder once per process."""
        if self._embed_model is not None:
            return self._embed_model
        from sentence_transformers import SentenceTransformer
        if self.embed_device is not None:
            device = self.embed_device
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            model = SentenceTransformer(self.embed_model_name, device=device)
        except Exception as e:
            print(f"[RewardManager] Failed loading embed model on {device} ({e}); falling back to cpu")
            device = 'cpu'
            model = SentenceTransformer(self.embed_model_name, device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        self._embed_model = model
        self._embed_device_resolved = device
        print(f"[RewardManager] Loaded embedding model '{self.embed_model_name}' on {device} (frozen)")
        return self._embed_model

    def _embed_query_prefix(self):
        """e5 models are trained with 'query:' / 'passage:' instruction prefixes
        and noticeably underperform without them. The texts we embed here (search
        queries, or per-step think) are treated as queries, so prepend 'query: '.
        Auto-enabled whenever the embedding model name contains 'e5'; a no-op for
        MiniLM and other sentence-transformers (their training used no prefix)."""
        name = (self.embed_model_name or '').lower()
        return 'query: ' if 'e5' in name else ''

    @staticmethod
    def _extract_think_per_step(response_str: str, n_steps: int) -> list:
        """Return n_steps strings, one per step's <think> block (in order).

        Trajectory structure:
            <think>_0</think><search>_0</search><information>_0</information>
            <think>_1</think><search>_1</search><information>_1</information>
            ...
            <think>_final</think><answer>...</answer>

        The first n_steps think blocks correspond to steps 0..n-1 (each preceding
        a <search>). Any trailing think (the pre-answer one) is dropped, mirroring
        doc-set's `_find_step_token_ranges` which also assigns no diversity to the
        final answer chunk. Missing/malformed step thinks become ''.
        """
        blocks = re.findall(r'<think>(.*?)</think>', response_str, re.DOTALL)
        out = []
        for k in range(n_steps):
            out.append(blocks[k].strip() if k < len(blocks) else '')
        return out

    @staticmethod
    def _extract_search_per_step(response_str: str, n_steps: int) -> list:
        """Return n_steps strings, one per step's <search>...</search> query.

        For multi-turn agent: each step has exactly one <search> query (n_steps =
        number of <information> blocks = number of completed search calls).
        Missing/malformed step queries become ''.

        Used by 'search_embedding' diversity arm: embed the search query of each
        step and reward diverse query phrasings within (prompt, n_search) strata.
        """
        blocks = re.findall(r'<search>(.*?)</search>', response_str, re.DOTALL)
        out = []
        for k in range(n_steps):
            out.append(blocks[k].strip() if k < len(blocks) else '')
        return out

    @staticmethod
    def _extract_reasoning_turns(response_str: str) -> list:
        """List of ALL <think>...</think> blocks for one rollout, in order.

        Every turn's reasoning, INCLUDING the final answer-turn <think> (the
        reasoning right before <answer>). Unlike `_extract_think_per_step`, NOT
        capped at n_search, so the last reasoning block is kept. Empty/whitespace
        blocks are dropped.

        Used by the 'reasoning_embedding' arm: each turn's reasoning is embedded
        SEPARATELY then mean-pooled into one rollout vector (no 512-token
        truncation, equal weight per turn). By construction excludes <search>,
        <answer>, and injected <information> (they live outside <think>).
        """
        blocks = re.findall(r'<think>(.*?)</think>', response_str, re.DOTALL)
        return [b.strip() for b in blocks if b.strip()]

    @staticmethod
    def _extract_search_turns(response_str: str) -> list:
        """List of ALL <search>...</search> queries for one rollout, in order.

        Symmetric to `_extract_reasoning_turns` (per-turn embed → mean-pool), but
        on the query layer.
        """
        blocks = re.findall(r'<search>(.*?)</search>', response_str, re.DOTALL)
        return [b.strip() for b in blocks if b.strip()]

    def _embed_texts(self, texts):
        """Batch-encode a list of strings with a hash-based cache.

        Returns a list of length len(texts), each entry either a CPU tensor (d,)
        of L2-normalized embedding, or None if the input string is empty.
        """
        embeddings = [None] * len(texts)
        cache_keys = [None] * len(texts)
        to_encode_idx = []
        to_encode_text = []
        for i, t in enumerate(texts):
            if not t:
                continue
            k = hashlib.md5(t.encode('utf-8')).hexdigest()
            cache_keys[i] = k
            cached = self._embed_cache.get(k)
            if cached is not None:
                embeddings[i] = cached
            else:
                to_encode_idx.append(i)
                to_encode_text.append(t)

        if to_encode_text:
            model = self._get_embed_model()
            prefix = self._embed_query_prefix()
            enc_text = [prefix + t for t in to_encode_text] if prefix else to_encode_text
            with torch.no_grad():
                new_embs = model.encode(
                    enc_text,
                    batch_size=64,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            new_embs = new_embs.detach().cpu()
            for j, idx in enumerate(to_encode_idx):
                emb = new_embs[j]
                embeddings[idx] = emb
                self._embed_cache[cache_keys[idx]] = emb
        return embeddings

    @staticmethod
    def _compute_emb_diversity_per_traj(embs_in_stratum):
        """Per-trajectory pairwise cosine distance, averaged over sibling partners.

        For each trajectory i in the stratum:
            score_i = (1/(N-1)) * sum_{j!=i} (1 - cos(e_i, e_j))

        Score lives in [0, 2] (typically [0, 1] for natural text). HIGHER = more
        diverse from siblings on average.

        Why this shape (vs soft-min) is robust to reward hacking:
        Trajectories with valid (non-empty) think text receive POSITIVE scores.
        Trajectories with missing think (None embedding) get score 0 by default.
        Within a stratum, missing-think trajectories therefore fall BELOW the
        mean of valid scores, so GRPO normalization assigns them a NEGATIVE
        advantage. The policy is implicitly penalized for omitting <think> blocks
        — without any explicit format penalty term and without breaking symmetry
        with the doc-set arm's reward shape.

        (Earlier soft-min variant had scores in negative domain, which placed
        the missing-data score=0 ABOVE the stratum mean, accidentally rewarding
        format collapse. Reverted intentionally.)

        Args:
            embs_in_stratum: list of tensor (d,) or None, one per trajectory in
                the stratum (same prompt, same search count).

        Returns:
            (scores, stats):
              scores: list of floats, len(embs_in_stratum). 0 for None
                  embeddings or strata with < 2 valid embeddings.
              stats: dict with debug stats per call:
                  'cos_sims': flat list of off-diagonal cos sims (for distribution log)
                  'nn_sims':  per-(valid)-traj NN cos sim (max excluding self)
        """
        n = len(embs_in_stratum)
        scores = [0.0] * n
        stats = {'cos_sims': [], 'nn_sims': []}
        valid = [(i, e) for i, e in enumerate(embs_in_stratum) if e is not None]
        if len(valid) < 2:
            return scores, stats
        idxs = [v[0] for v in valid]
        mat = torch.stack([v[1] for v in valid]).float()  # (m, d), L2-normalized
        m = mat.shape[0]
        sim = mat @ mat.t()                                # (m, m), cos similarity
        diag_mask = torch.eye(m, dtype=torch.bool)
        # off-diagonal sims for debug stats (each pair counted twice; fine for distribution)
        stats['cos_sims'] = sim[~diag_mask].tolist()
        sim_off = sim.masked_fill(diag_mask, float('-inf'))
        stats['nn_sims'] = sim_off.max(dim=-1).values.tolist()
        # Pairwise mean distance: (1 - cos), averaged over j != i
        dist = (1.0 - sim).clamp(min=0.0)                  # numeric-noise floor
        dist.fill_diagonal_(0.0)
        scores_valid = (dist.sum(dim=-1) / (m - 1)).tolist()
        for k, idx in enumerate(idxs):
            scores[idx] = float(scores_valid[k])
        return scores, stats

    def _stepwise_query_diversity(self, bs, base_scores, prompt_hashes,
                                  think_per_step_all, use_nsearch):
        """Per-step query-embedding diversity among CORRECT rollouts.

        For each search step k, the k-th query of every correct rollout that
        reached step k is compared (e5 embedding, accuracy-gated):
            div_{i,k} = (1/(N-1)) Σ_{j≠i} (1 - cos(e_i^k, e_j^k))
        where N = number of correct rollouts that reached step k (same prompt,
        plus same n_search if use_nsearch). Groups with N < 2 are skipped (the
        lone rollout's step-k query gets NO diversity — no entry in the result).

        Returns:
            step_scores: dict  traj_i -> {step_k: div_float}
            stats: dict with 'cos','nn','sizes','by_step','gsize_by_step' for logs
        """
        # Embed each per-step query of every correct trajectory (hash-cached).
        flat_texts, flat_ref = [], []
        for i in range(bs):
            if base_scores[i] <= 0:
                continue
            for k, t in enumerate(think_per_step_all[i]):
                if t and t.strip():
                    flat_texts.append(t)
                    flat_ref.append((i, k))
        flat_embs = self._embed_texts(flat_texts) if flat_texts else []
        emb_by_traj_step = {}
        for (i, k), e in zip(flat_ref, flat_embs):
            if e is not None:
                emb_by_traj_step[(i, k)] = e

        # Group correct rollouts' steps by (prompt[, n_search], step_k).
        groups = defaultdict(list)
        for i in range(bs):
            if base_scores[i] <= 0:
                continue
            ph = prompt_hashes[i]
            n_search_i = len(think_per_step_all[i])
            for k in range(n_search_i):
                key = (ph, n_search_i, k) if use_nsearch else (ph, k)
                groups[key].append((i, k))

        step_scores = defaultdict(dict)
        stats = {'cos': [], 'nn': [], 'sizes': [],
                 'by_step': defaultdict(list), 'gsize_by_step': defaultdict(list)}
        for key, entries in groups.items():
            k = key[-1]
            stats['sizes'].append(len(entries))
            stats['gsize_by_step'][k].append(len(entries))
            if len(entries) < 2:
                continue
            embs = [emb_by_traj_step.get((i, kk)) for (i, kk) in entries]
            scores, st = self._compute_emb_diversity_per_traj(embs)
            stats['cos'].extend(st['cos_sims'])
            stats['nn'].extend(st['nn_sims'])
            for local_i, (i, kk) in enumerate(entries):
                step_scores[i][kk] = scores[local_i]
                stats['by_step'][kk].append(scores[local_i])
        return step_scores, stats

    def _stepwise_doc_diversity(self, bs, base_scores, prompt_hashes,
                                per_step_docs_all, use_nsearch):
        """Per-step document-set diversity among CORRECT rollouts.

        Same shape as `_stepwise_query_diversity`, but the per-step signal is the
        fraction of the rollout's step-k retrieved docs that NO sibling retrieved
        at the same step (slide-12/13 doc-set globally-unique reward):
            div_{i,k} = |D_{i,k} \\ ∪_{j≠i} D_{j,k}| / |D_{i,k}|
        over correct rollouts that reached step k (same prompt, plus same n_search
        if use_nsearch). Groups with < 2 members are skipped (div = 0). Empty
        doc-set → 0. Returned dict structure matches the query helper so the
        step_reward branch can consume either interchangeably.
        """
        groups = defaultdict(list)  # key -> [(traj_i, step_k, docs_k), ...]
        for i in range(bs):
            if base_scores[i] <= 0:
                continue
            ph = prompt_hashes[i]
            n_search_i = len(per_step_docs_all[i])
            for k, docs_k in enumerate(per_step_docs_all[i]):
                key = (ph, n_search_i, k) if use_nsearch else (ph, k)
                groups[key].append((i, k, docs_k))

        step_scores = defaultdict(dict)
        stats = {'cos': [], 'nn': [], 'sizes': [],
                 'by_step': defaultdict(list), 'gsize_by_step': defaultdict(list)}
        for key, entries in groups.items():
            k = key[-1]
            stats['sizes'].append(len(entries))
            stats['gsize_by_step'][k].append(len(entries))
            if len(entries) < 2:
                continue
            for local_i, (i, kk, docs_i) in enumerate(entries):
                if len(docs_i) == 0:
                    score = 0.0
                else:
                    other = set()
                    for local_j, (_j, _kj, docs_j) in enumerate(entries):
                        if local_i != local_j:
                            other |= docs_j
                    score = len(docs_i - other) / len(docs_i)
                step_scores[i][kk] = score
                stats['by_step'][kk].append(score)
        return step_scores, stats

    def _get_probe_model(self):
        """Lazy-load + freeze the base reference probe (causal LM) for ΔP scoring."""
        if self._probe_model is not None:
            return self._probe_model, self._probe_tok
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = self.useful_device or ('cuda' if torch.cuda.is_available() else 'cpu')
        tok = AutoTokenizer.from_pretrained(self.useful_probe_path, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.useful_probe_path, torch_dtype=torch.float16,
                trust_remote_code=True).to(dev).eval()
        except Exception as e:
            print(f"[RewardManager] probe load on {dev} failed ({e}); cpu fallback")
            dev = 'cpu'
            model = AutoModelForCausalLM.from_pretrained(
                self.useful_probe_path, torch_dtype=torch.float32,
                trust_remote_code=True).to(dev).eval()
        for p in model.parameters():
            p.requires_grad = False
        self._probe_model, self._probe_tok, self._probe_device_resolved = model, tok, dev
        print(f"[RewardManager] Loaded ΔP probe '{self.useful_probe_path}' on {dev} (frozen)")
        return model, tok

    @staticmethod
    def _parse_info_spans(seq_str):
        """(context, info_char_spans). context = decoded text up to end of last
        </information> (answer turn dropped); spans = char (start,end) of each
        <information>..</information> within context. None if no info block."""
        idx = seq_str.find("<|im_start|>assistant")
        asst = seq_str[idx:] if idx >= 0 else seq_str
        infos = list(re.finditer(r'<information>.*?</information>', asst, re.DOTALL))
        if not infos:
            return None
        ctx_end = infos[-1].end()
        return asst[:ctx_end], [(m.start(), m.end()) for m in infos]

    @torch.no_grad()
    def _score_gold(self, context, gold, max_len=7000):
        """sum logP(gold tokens | context + ' <answer>') under the frozen probe.
        Gold is tokenized SEPARATELY (add_special_tokens=False, leading space) and
        appended explicitly, so n_gold is the true gold token count — avoids the
        boundary-merge bug where a trailing-space scaffold absorbs short golds
        (yes/no/single word) into 0 net tokens. Same scaffold for full and
        leave-one-out so format cancels in ΔP."""
        model, tok = self._get_probe_model()
        dev = self._probe_device_resolved
        ids_p = tok(context + " <answer>", return_tensors="pt").input_ids
        gold_ids = tok(" " + str(gold), add_special_tokens=False, return_tensors="pt").input_ids
        n_gold = gold_ids.shape[1]
        if n_gold < 1:
            return None
        ids_f = torch.cat([ids_p, gold_ids], dim=1)
        if ids_f.shape[1] > max_len:
            ids_f = ids_f[:, -max_len:]
        ids_f = ids_f.to(dev)
        logits = model(ids_f).logits[0, :-1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        tgt = ids_f[0, 1:]
        return lp[torch.arange(tgt.shape[0], device=dev), tgt][-n_gold:].sum().item()

    def _compute_useful_weights(self, indices, em_seqs, golds_all, per_step_docs_all):
        """For each CORRECT rollout i, per-search-step useful weight w_k=max(0,ΔP_k),
        prefix ΔP_k = logP(gold | …through k-th <information>) − logP(gold | …before it):
        truncate at the block and score only the marginal at ARRIVAL (keeps this turn's
        think+search, removes only the info content). Chosen over leave-one-out: LOO
        under-credits multi-hop bridges / early-gold that a downstream block already
        covers (offline: 24% of useful new-gold steps got LOO≤0; prefix halves that to
        ~12% with only 6% false-rescue of over-search). ALL (context,gold) score
        requests across the batch are scored in ONE shot — via the remote GPU probe
        server when useful_probe_url is set (the driver has no GPU), else the local
        model. Returns {i:[w_0,...]} aligned with per_step_docs_all[i]. Unparseable/
        unscoreable steps get weight 1.0 (neutral)."""
        # Phase 1 — parse each rollout and enumerate the flat list of score requests.
        # plan[i] is one of: ('empty',) | ('neutral', n_steps)
        #                  | ('ok', n_steps, [(k, thru_j, before_j) | (k, None, None)])
        plans = {}
        ctxs, glds = [], []   # flat (context, gold) requests; j indexes into these
        for i in indices:
            n_steps = len(per_step_docs_all[i])
            if n_steps == 0:
                plans[i] = ('empty',)
                continue
            gold = golds_all[i]
            if hasattr(gold, 'tolist'):          # numpy array of golden answers
                gold = gold.tolist()
            if isinstance(gold, (list, tuple)):  # take first acceptable answer
                gold = gold[0] if len(gold) > 0 else ""
            gold = "" if gold is None else str(gold)
            parsed = self._parse_info_spans(em_seqs[i] or "")
            if parsed is None or not gold:
                plans[i] = ('neutral', n_steps)
                continue
            context, spans = parsed
            rlist = []
            for k in range(n_steps):
                if k >= len(spans):
                    rlist.append((k, None, None)); continue
                s, e = spans[k]
                tj = len(ctxs); ctxs.append(context[:e]); glds.append(gold)  # through info_k
                bj = len(ctxs); ctxs.append(context[:s]); glds.append(gold)  # before it
                rlist.append((k, tj, bj))
            plans[i] = ('ok', n_steps, rlist)

        # Phase 2 — score every request at once.
        scores = self._score_many(ctxs, glds)   # list[float|None] aligned with ctxs

        # Phase 3 — assemble prefix-ΔP weights.
        out = {}
        for i in indices:
            p = plans[i]
            if p[0] == 'empty':
                out[i] = []
            elif p[0] == 'neutral':
                out[i] = [1.0] * p[1]
            else:
                _, n_steps, rlist = p
                # per-step prefix ΔP (None if unscoreable)
                dps = []
                for (k, tj, bj) in rlist:
                    if tj is None or scores[tj] is None or scores[bj] is None:
                        dps.append(None)
                    else:
                        dps.append(scores[tj] - scores[bj])
                if self.useful_hybrid:
                    # peak = step with max ΔP (= where the answer is found). Build-up
                    # chain (≤ peak) gets full credit so bridges survive; over-search
                    # tail (> peak) is ΔP-gated.
                    valid = [(k, d) for k, d in enumerate(dps) if d is not None]
                    peak = max(valid, key=lambda kd: kd[1])[0] if valid else len(dps) - 1
                    w = [1.0 if (d is None or k <= peak) else max(0.0, d)
                         for k, d in enumerate(dps)]
                else:
                    # plain per-search prefix arm: w = max(0, ΔP)
                    w = [1.0 if d is None else max(0.0, d) for d in dps]
                out[i] = w
        return out

    def _score_many(self, contexts, golds):
        """sum logP(gold | context + ' <answer>') for each pair. Uses the remote GPU
        probe server when useful_probe_url is set (a single batched POST), else the
        local probe model per item. Returns list[float|None]."""
        if not contexts:
            return []
        if self.useful_probe_url:
            return self._score_remote(contexts, golds)
        return [self._score_gold(c, g) for c, g in zip(contexts, golds)]

    def _score_remote(self, contexts, golds, chunk_size=32, reduce="sum", tok_budget=None):
        """POST the batch to the frozen-probe http server in bounded CHUNKS. v2's
        Shapley enumeration can send thousands of long contexts at once; a single
        giant POST makes the probe's PyTorch cache balloon (~30GB) and starve the
        retriever co-located on the same GPU. Chunking caps the probe's per-request
        peak (the probe file is untouched). Returns list[float|None]."""
        import requests
        golds = [str(g) for g in golds]
        out = []
        for off in range(0, len(contexts), chunk_size):
            c = contexts[off:off + chunk_size]
            g = golds[off:off + chunk_size]
            try:
                payload = {"contexts": c, "golds": g, "reduce": reduce}
                if tok_budget is not None:
                    payload["tok_budget"] = tok_budget   # bigger micro-batches on a dedicated GPU
                resp = requests.post(self.useful_probe_url, json=payload, timeout=1200)
                resp.raise_for_status()
                logps = resp.json()["logps"]
                if len(logps) != len(c):
                    raise ValueError(f"probe returned {len(logps)} != {len(c)} scores")
                out.extend(logps)
            except Exception as ex:
                print(f"[RewardManager] probe server {self.useful_probe_url} failed "
                      f"on chunk {off}:{off+len(c)} ({ex}); neutral weights for it")
                out.extend([None] * len(c))
        return out

    # ===================== v2 Shapley diversity (PDF final spec) =====================
    # Single axiom "pay only conditional marginal info", three granularities, one
    # scorer s(D)=logP(gt|template,D) via the SAME frozen probe used by useful-weight.
    #   Stage 1  per-rollout 2-round greedy filter (fwd then rescue) -> survivor E_i
    #   Stage 2  v(S)=s(union E_i for i in S)-s(empty); Shapley phi_i (exact k<=8)
    #   Stage 3  R_div_i = phi_i / (sum_j phi_j + eps)     (gating done by caller)
    # All s(.) scored in 3 batched probe rounds across the whole batch.
    _V2_INFO_RE = re.compile(r"<information>(.*?)</information>", re.DOTALL)
    # NOTE: the live training rollout emits docs as `Doc N(Title: "X") <text>`
    # (generation.py:479) with NO "Content:" label, while the eval harness
    # (eval_passk.py) emits `Doc N(Title: "X")\nContent: <text>`. The old regex
    # REQUIRED "Content:", so it parsed ZERO docs from real training rollouts ->
    # empty survivor sets -> diversity reward identically 0. Fix: quoted title
    # (handles internal parens like "Beauty and the Beast (1991 film)") and make
    # the "Content:" label OPTIONAL so BOTH formats parse.
    _V2_DOC_RE = re.compile(
        r'Doc\s*\d+\s*\(Title:\s*("?)([^"\n]*?)\1\)\s*(?:\n?\s*Content:\s*)?(.*?)(?=Doc\s*\d+\s*\(Title:|\Z)',
        re.DOTALL)

    @classmethod
    def _v2_parse_docs(cls, em_seq):
        out, seen = [], set()
        for blk in cls._V2_INFO_RE.findall(em_seq or ""):
            for _q, title, content in cls._V2_DOC_RE.findall(blk):
                title, content = title.strip(), " ".join(content.split())
                key = content[:200]
                if content and key not in seen:
                    seen.add(key); out.append((title, content))
        return out

    @classmethod
    def _v2_parse_blocks(cls, em_seq):
        """Docs grouped by SEARCH STEP — one inner list per <information> block.
        A doc is assigned to the FIRST block that retrieved it (global dedup), so
        each block = the NEW docs that step introduced; a step that re-fetched only
        already-seen docs yields no unit (and so cannot inflate over-search)."""
        out, seen = [], set()
        for blk in cls._V2_INFO_RE.findall(em_seq or ""):
            unit = []
            for _q, title, content in cls._V2_DOC_RE.findall(blk):
                title, content = title.strip(), " ".join(content.split())
                key = content[:200]
                if content and key not in seen:
                    seen.add(key); unit.append((title, content))
            if unit:
                out.append(unit)
        return out

    @staticmethod
    def _v2_question(em_seq):
        m = re.search(r"Question:\s*(.+?)\s*<\|im_end\|>", em_seq or "", re.DOTALL)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _v2_gold(g):
        if hasattr(g, 'tolist'):
            g = g.tolist()
        if isinstance(g, (list, tuple)):
            g = g[0] if len(g) > 0 else ""
        return "" if g is None else str(g)

    @staticmethod
    def _v2_ctx(docs, question):
        passage = "\n\n".join(f"{t}. {c}" for t, c in docs)
        return ("Passage: " + passage + "\nQuestion: " + question + "\n"
                "Please answer the question using the facts of the passage.\n"
                "Keep your answer grounded to the facts of the passage.\n"
                "Directly provide the answer without detailed illustrations.")

    @staticmethod
    def _v2_dedup_union(doc_lists):
        seen, out = set(), []
        for docs in doc_lists:
            for d in docs:
                k = d[1][:200]
                if k not in seen:
                    seen.add(k); out.append(d)
        return out

    @staticmethod
    def _pct(xs, p):
        if not xs:
            return 0.0
        xs = sorted(xs); k = (len(xs) - 1) * p / 100.0
        lo, hi = int(k // 1), int(-(-k // 1))
        return xs[lo] if lo == hi else xs[lo] * (hi - k) + xs[hi] * (k - lo)

    # ===================== Meaningful-Diversity RL (revised spec) =====================
    #   R_div(i) = max( p̂ · Σ_{d∈players_i} φ_d · (1/m_d),  0 )   on CORRECT rollouts only.
    #   players_i = distinct retrieved docs of rollout i (content-hash, arrival order).
    #   s(S)  = length-normalized logP (per-token mean logP) of gt under the evidence-
    #           only template (docs S in arrival order); s(∅)=closed-book. frozen base LM.
    #   v(S)  = (s(S) − s(∅)) / max(−s(∅), ε): fraction of the closed-book→confidence gap
    #           coalition S fills. v(∅)=0, v(full)≤1. ε=0.5 floors the denom (no-headroom
    #           safety). Shapley efficiency ⇒ Σ_d φ_d = v(full) ≤ 1 → R_div STRUCTURALLY
    #           bounded (no λ/tanh); also puts obscure & easy answers on one unit so the
    #           across-rollout 1/m aggregation is coherent.
    #   φ_d   = EXACT 2^n within-rollout Shapley on v (negatives kept, NO clip).
    #   1/m_d = across-rollout novelty, m_d = #correct rollouts in the GROUP holding d.
    #   p̂    = (#correct in group)/G: correctness-first gate — auto-suppresses 1-correct
    #           groups, ramps diversity in only as accuracy rises. parameter-free.
    # No Stage-1 filter, no MC. (Scorer swaps probe->KL-ref later.)
    @classmethod
    def _md_parse(cls, em_seq):
        """Distinct docs of one rollout in ARRIVAL order (Shapley players). Identity =
        md5 of the FULL normalized content (content-hash proxy for corpus doc-ID; NOT
        title — same article's different passages share a title and must stay separate).
        Same doc re-fetched later = one player (dedup by key, keep first occurrence)."""
        out, seen = [], set()
        for blk in cls._V2_INFO_RE.findall(em_seq or ""):
            for _q, title, content in cls._V2_DOC_RE.findall(blk):
                title = title.strip()
                content = " ".join(content.split())
                if not content:
                    continue
                key = hashlib.md5(content.encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    out.append((key, title, content))
        return out

    def _md_diversity(self, strata, em_seqs, golds_all, base_scores):
        """Returns {i: R_div_i}; incorrect rollouts stay 0 (not scored). See block above."""
        import itertools, math
        out = {i: 0.0 for i in range(len(em_seqs))}
        # 1) distinct docs per CORRECT rollout
        docs, ph_of = {}, {}
        for ph, idxs in strata.items():
            for i in idxs:
                ph_of[i] = ph
                if base_scores[i] > 0:
                    d = self._md_parse(em_seqs[i])
                    if d:
                        docs[i] = d
        correct = list(docs.keys())
        if not correct:
            self._md_last = {'maxn': 0, 'n_ctx': 0, 'n_correct': 0, 'nhist': {}}
            return out
        # 2) across-rollout m_d (content-hash), correct rollouts only, per stratum
        m_by_ph = {}
        for ph, idxs in strata.items():
            cnt = Counter()
            for i in idxs:
                if i in docs:
                    for (k, t, c) in docs[i]:
                        cnt[k] += 1
            m_by_ph[ph] = cnt
        # 3) collect coalitions to score. EXACT 2^n for n ≤ K_EXACT; for n > K_EXACT a
        #    single large-n rollout would blow up (2^14 = 16k forwards), so fall back to
        #    MC permutation-Shapley: P antithetic perms, score only their n+1 prefix
        #    coalitions. Each perm's marginals telescope to v(full)−v(∅), so Σφ = v(full)
        #    ≤ 1 holds EXACTLY even under MC (bound & efficiency preserved; only per-φ_d
        #    gets sampling noise, on the rare large-n rollouts).
        # AUTO cost-crossover (no hand-picked K): exact costs 2^n coalitions, MC costs
        # ~P·n. Use whichever is cheaper per rollout — exact while 2^n ≤ P·n (≈ n≤8 for
        # P=32; exact is also noiseless so it wins on ties), MC above. P is the ONLY knob
        # (= MC sample count = accuracy/cost of the rare large-n fallback), and the
        # exact/MC boundary falls out of it automatically.
        P_PERM = 32         # antithetic permutations for MC (16 random + 16 reverses)
        ctxs, golds, tags, meta, perm_plan = [], [], [], {}, {}
        for i in correct:
            q = self._v2_question(em_seqs[i]); g = self._v2_gold(golds_all[i])
            dl = docs[i]; n = len(dl)
            meta[i] = (q, g, dl, n)
            if 2 ** n <= P_PERM * n:          # exact when cheaper-or-equal (also noiseless); else MC
                coals = [S for r in range(n + 1) for S in itertools.combinations(range(n), r)]
            else:
                rng = random.Random((hash(em_seqs[i]) ^ (n * 2654435761)) & 0xffffffff)
                perms = []
                for _ in range(P_PERM // 2):
                    p = list(range(n)); rng.shuffle(p)
                    perms.append(tuple(p)); perms.append(tuple(reversed(p)))
                perm_plan[i] = perms
                cs = {()}
                for p in perms:
                    for kk in range(1, n + 1):
                        cs.add(tuple(sorted(p[:kk])))
                coals = list(cs)
            for S in coals:
                tags.append((i, S))
                ctxs.append(self._v2_ctx([(dl[k][1], dl[k][2]) for k in S], q))
                golds.append(g)
        scores = self._score_remote(ctxs, golds, reduce='mean', chunk_size=256)   # fewer POSTs (compute-bound)
        v = {tag: s for tag, s in zip(tags, scores)}
        # group-level p̂ gate = (#correct in group)/G — correctness-first curriculum,
        # auto-suppresses 1-correct groups (parameter-free, hack-proof).
        phat = {}
        for ph, idxs in strata.items():
            G = len(idxs)
            phat[ph] = (sum(1 for j in idxs if base_scores[j] > 0) / G) if G else 0.0
        EPS = 0.5   # logP-unit headroom floor for the v-normalizer (denominator safety)
        # 4) exact within-rollout Shapley on HEADROOM-normalized v, then ·(1/m_d)·p̂.
        #    v(S) = (s(S)−s(∅))/(−s(∅)): fraction of the closed-book→confidence gap S
        #    fills; v(∅)=0, v(full)≤1. Affine in s, so φ on normalized v = φ_raw/denom
        #    (the −s(∅) shift cancels in marginals). Shapley efficiency ⇒ Σ_d φ_d ≤ 1,
        #    so R_div is structurally bounded (no λ/tanh) AND obscure vs easy answers
        #    share one unit, making the across-rollout 1/m aggregation coherent.
        maxn = 0; nhist = Counter(); phi_vals = []
        for i in correct:
            q, g, dl, n = meta[i]; maxn = max(maxn, n); nhist[n] += 1
            s_empty = v.get((i, ()))
            if s_empty is None:                 # closed-book unscoreable -> no headroom basis
                out[i] = 0.0; continue
            denom = max(-s_empty, EPS)           # closed-book headroom; floor avoids blow-up
            phi = [0.0] * n
            if i not in perm_plan:               # EXACT 2^n Shapley
                for d in range(n):
                    others = [x for x in range(n) if x != d]
                    for r in range(len(others) + 1):
                        w = math.factorial(r) * math.factorial(n - r - 1) / math.factorial(n)
                        for S in itertools.combinations(others, r):
                            vS = v.get((i, S)); vSd = v.get((i, tuple(sorted(S + (d,)))))
                            if vS is None or vSd is None:
                                continue
                            phi[d] += w * (vSd - vS)
            else:                                # MC permutation-Shapley (large n)
                perms = perm_plan[i]
                for p in perms:
                    for kk in range(n):
                        vb = v.get((i, tuple(sorted(p[:kk]))))
                        va = v.get((i, tuple(sorted(p[:kk + 1]))))
                        if vb is not None and va is not None:
                            phi[p[kk]] += (va - vb)
                phi = [x / len(perms) for x in phi]
            phi = [p / denom for p in phi]       # headroom-normalize -> Σφ ≤ 1
            m = m_by_ph[ph_of[i]]
            # R_div floored at 0: net-misleading rollouts earn no bonus (not a penalty).
            # φ_d stays signed internally (Stage-3 attribution); only the per-rollout total
            # is clamped so R_div is a pure non-negative bonus on top of R_acc.
            out[i] = max(phat[ph_of[i]] * sum(phi[d] * (1.0 / m[dl[d][0]]) for d in range(n)), 0.0)
            phi_vals.extend(phi)
        # novelty histogram: m_d over DISTINCT (group, doc) among correct rollouts
        m_hist = Counter()
        for ph, cnt in m_by_ph.items():
            for k, mc in cnt.items():
                m_hist[mc] += 1
        pv = [phat[ph] for ph in strata if any(base_scores[j] > 0 for j in strata[ph])]
        self._md_last = {'maxn': maxn, 'n_ctx': len(ctxs), 'n_correct': len(correct),
                         'nhist': dict(sorted(nhist.items())), 'phi': phi_vals,
                         'm_hist': dict(sorted(m_hist.items())),
                         'phat_mean': (sum(pv) / len(pv)) if pv else 0.0,
                         'n_mc': len(perm_plan)}
        return out

    # ============== RGSN: Relevance-Gated Soft-Novelty (fix for Meaningful-Div) ========
    #   R_div(i) = max( p̂ · (1/n_i) Σ_{d} w_d·(1/m_d), 0 )   on CORRECT rollouts only.
    #     w_d   = max(0, v({d})),  v({d}) = (s({d})−s(∅)) / max(−s(∅), 0.5)
    #             — ORDER-INDEPENDENT isolated marginal, NOT Shapley. Two redundant-but-
    #             distinct gold docs BOTH have high v({d}) → both counted in full (no
    #             submodular credit-split that makes Meaningful's φ_d decay with #gold).
    #     s(S)  = per-token mean logP(gt | _v2_ctx(docs S)); SAME frozen probe as _md.
    #     m_d   = #correct rollouts in the GROUP holding d (content-hash novelty).
    #     p̂    = (#correct in group)/G  — same parameter-free correctness gate as _md.
    #   Normalised by n_i (#distinct docs), NOT Σw: off-topic docs (w≈0) DILUTE the mean
    #   (over-search self-penalises), distinct gold docs ACCUMULATE (gold-coverage
    #   diversity rewarded additively — the property that lifts pass@k union-recall).
    #   Offline-validated on Meaningful-Div rollouts (diag_out/): gating 4.6–7.3×, w-per-
    #   gold FLAT vs #gold, R rises with #gold, corr(R,coverage)=+0.52–0.79. See DIAGNOSIS.md.
    #   Cost = n+1 probe scores/rollout (vs 2^n for _md's exact Shapley).
    def _rgsn_diversity(self, strata, em_seqs, golds_all, base_scores):
        out = {i: 0.0 for i in range(len(em_seqs))}
        # 1) distinct docs (content-hash, arrival order) per CORRECT rollout
        docs, ph_of = {}, {}
        for ph, idxs in strata.items():
            for i in idxs:
                ph_of[i] = ph
                if base_scores[i] > 0:
                    d = self._md_parse(em_seqs[i])
                    if d:
                        docs[i] = d
        correct = list(docs.keys())
        if not correct:
            self._rgsn_last = {'n_correct': 0, 'w_gold_mean': 0.0, 'w_off_mean': 0.0}
            return out
        # 2) across-rollout m_d (content-hash), correct rollouts only, per stratum
        m_by_ph = {}
        for ph, idxs in strata.items():
            cnt = Counter()
            for i in idxs:
                if i in docs:
                    for (k, t, c) in docs[i]:
                        cnt[k] += 1
            m_by_ph[ph] = cnt
        # 3) score requests: empty coalition + each singleton, batched in ONE shot
        ctxs, golds, tags = [], [], []
        for i in correct:
            q = self._v2_question(em_seqs[i]); g = self._v2_gold(golds_all[i])
            dl = docs[i]
            tags.append((i, -1)); ctxs.append(self._v2_ctx([], q)); golds.append(g)
            for d in range(len(dl)):
                tags.append((i, d))
                ctxs.append(self._v2_ctx([(dl[d][1], dl[d][2])], q)); golds.append(g)
        scores = self._score_remote(ctxs, golds, reduce='mean', chunk_size=256)
        sval = {tag: s for tag, s in zip(tags, scores)}
        # 4) group-level p̂ gate (same as _md)
        phat = {}
        for ph, idxs in strata.items():
            G = len(idxs)
            phat[ph] = (sum(1 for j in idxs if base_scores[j] > 0) / G) if G else 0.0
        EPS = 0.5
        w_gold_log, w_off_log = [], []
        for i in correct:
            dl = docs[i]; n = len(dl)
            s0 = sval.get((i, -1))
            if s0 is None:
                out[i] = 0.0; continue
            denom = max(-s0, EPS)
            m = m_by_ph[ph_of[i]]
            g_low = self._v2_gold(golds_all[i]).lower()
            tot = 0.0
            for d in range(n):
                sd = sval.get((i, d))
                vd = ((sd - s0) / denom) if sd is not None else 0.0
                w = vd if vd > 0.0 else 0.0
                tot += w * (1.0 / m[dl[d][0]])
                (w_gold_log if (g_low and g_low in dl[d][2].lower()) else w_off_log).append(w)
            out[i] = max(phat[ph_of[i]] * tot / n, 0.0) if n > 0 else 0.0
        self._rgsn_last = {
            'n_correct': len(correct),
            'w_gold_mean': (sum(w_gold_log) / len(w_gold_log)) if w_gold_log else 0.0,
            'w_off_mean': (sum(w_off_log) / len(w_off_log)) if w_off_log else 0.0}
        return out

    @staticmethod
    def _bleu(cand, ref, max_n=4):
        """Sentence BLEU of candidate token list `cand` against reference `ref`
        (Papineni et al. 2002; DIVER Eq.15-16). Up to 4-grams, uniform weights
        w_n=1/N, brevity penalty. Chen&Cherry method-1 smoothing: a zero-overlap
        n-gram order gets a tiny floor instead of collapsing BLEU to 0 (needed for
        short sentence-level texts). Returns BLEU in [0,1]; 1−BLEU is the
        dissimilarity used for Textual Diversity."""
        c = len(cand)
        if c == 0:
            return 0.0
        log_p = 0.0
        for n in range(1, max_n + 1):
            if c - n + 1 <= 0:
                # candidate shorter than n: no n-grams of this order → floor
                log_p += (1.0 / max_n) * math.log(1e-9)
                continue
            cand_ng = Counter(tuple(cand[i:i + n]) for i in range(c - n + 1))
            ref_ng = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)) \
                if len(ref) - n + 1 > 0 else Counter()
            overlap = sum(min(cnt, ref_ng[g]) for g, cnt in cand_ng.items())
            total = sum(cand_ng.values())
            p_n = (overlap / total) if overlap > 0 else 1e-9
            log_p += (1.0 / max_n) * math.log(p_n)
        r = len(ref)
        bp = 1.0 if c > r else math.exp(1.0 - r / c)   # c>0 guaranteed above
        return bp * math.exp(log_p)

    def _bleu_diversity(self, strata, policy_strs, base_scores, bs):
        """DIVER Textual Diversity (Hu et al. 2026, arXiv:2509.26209) as a baseline
        R_div, replacing doc-novelty with text-novelty:
            TD(o_i) = (1/(|G|-1)) Σ_{j∈G, j≠i} (1 − BLEU(o_i, o_j))
        computed over ALL rollouts of the prompt group G (correct AND wrong) — the
        FULL-group d(o_i) of DIVER §3.4; application stays accuracy-gated downstream
        (DIVER's Conditional Shaping I(r_i), done in the placement loop). TD is then
        clipped to [0, σ] (Balanced Shaping, DIVER §3.5; σ=self.bleu_sigma, paper 0.65).
        `policy_strs` already holds POLICY-ONLY text — the caller built it from
        info_mask==1 spans (_policy_only_text), so env-injected observations are gone
        and model-fabricated <information> is kept. The final <answer> is KEPT — same
        as DIVER, where every correct response ends in the same \\boxed{} answer, so a
        constant-answer floor is part of the baseline."""
        toks = []
        for i in range(bs):
            toks.append((policy_strs[i] or '').lower().split())
        out = {i: 0.0 for i in range(bs)}
        for ph, idxs in strata.items():
            if len(idxs) < 2:
                continue
            for i in idxs:
                diss = sum(1.0 - self._bleu(toks[i], toks[j])
                           for j in idxs if j != i)
                out[i] = min(diss / (len(idxs) - 1), self.bleu_sigma)  # DIVER Balanced Shaping clip(TD;0,σ)
        return out

    def _soft_novelty_measure(self, strata, per_step_docs_all, base_scores, bs):
        """MEASUREMENT-ONLY soft-novelty R_div (the Proposed / soft-novelty value),
        computed for EVERY arm regardless of the active reward (diversity_type /
        combine_mode) so wandb can report it head-to-head (incl. passk and rgsn runs).
            R_div_i = (1/|D_i|) Σ_{d∈D_i} 1/m_d   over CORRECT rollouts of ≥2-correct
            groups (m_d = #correct rollouts holding d); 0 otherwise.
        Doc identity = union of per_step_docs_all[i] — the SAME identity the live
        soft-novelty branch uses, so for the diversity_type=none soft run this is
        identical to reward/r_div_*. Pure measurement: does NOT touch the reward."""
        out = [0.0] * bs
        traj_docs = [set().union(*per_step_docs_all[i]) if per_step_docs_all[i] else set()
                     for i in range(bs)]
        for ph, idxs in strata.items():
            correct_idx = [i for i in idxs if base_scores[i] > 0]
            if len(correct_idx) < 2:
                continue
            doc_count = defaultdict(int)
            for j in correct_idx:
                for d in traj_docs[j]:
                    doc_count[d] += 1
            for i in correct_idx:
                if traj_docs[i]:
                    out[i] = sum(1.0 / doc_count[d] for d in traj_docs[i]) / len(traj_docs[i])
        return out

    def _budget_soft_diversity(self, strata, base_scores, per_step_docs_all,
                               data_sources, bs, gold_in_docs=None,
                               gold_bearing_docs=None):
        """Budget-gated soft-novelty. Pool selection depends on the active flag:
            within_budget_i = (n_search_i <= budget_i);  n_search_i = len(per_step_docs_all[i])
            budget_i        = #supporting paragraphs of the question (from data_source)
            Per GROUP (stratum = the n_agent rollouts of one prompt):
          budget_ungate_gold=True — GOLD-ANCHORED, CORRECTNESS-AGNOSTIC:
              pool = {i : within_budget_i AND gold_in_docs_i}, for EVERY group.
              R_acc already rewards correctness, so diversity is measured over every
              rollout that RETRIEVED THE GOLD evidence (correct or wrong). min_pool=1:
              a lone gold-reacher still gets R_div=1.0.
          budget_ungate=True — asymmetric conditional-ungate (min_pool=2):
              • ≥1 CORRECT → pool = {correct AND within_budget}
              • ALL-WRONG  → pool = {within_budget}   (R_acc ungated)
          neither — original gate (min_pool=2): pool = {correct AND within_budget}.
            R_div_i = (1/|D_i|) Σ_{d∈D_i} 1/m_d,  m_d = #POOL rollouts holding d.
        Over-budget rollouts are excluded from the pool AND from m_d in BOTH cases —
        over-search earns no diversity and can't dampen in-budget novelty. ≤budget
        (not ==): searching FEWER is efficient and keeps full reward. NO <2-eligible
        guard: a lone-pool rollout keeps its raw R_div (=1.0 when its docs are all
        unique in a size-1 pool). Doc identity = union of per_step_docs_all[i]. The
        asymmetric R_acc gate is applied HERE (pool selection), so the downstream
        placement must NOT re-gate budget on 1[correct] (see combine em_fmt_doc)."""
        out = {i: 0.0 for i in range(bs)}
        traj_docs = [set().union(*per_step_docs_all[i]) if per_step_docs_all[i] else set()
                     for i in range(bs)]
        within_budget = [len(per_step_docs_all[i])
                         <= self.budget_map.get(str(data_sources[i]), self.budget_default)
                         for i in range(bs)]
        # |Pool| diagnostics: a size-1 pool has no comparison partner, so its lone
        # member auto-scores R_div=1.0 (a budget-compliance CONSTANT, not a measured
        # novelty). Track how often that happens, split by group type, so we know
        # whether ungate gains come from diversity (|Pool|≥2) or the constant.
        pool_hist = {'correct_grp': Counter(), 'allwrong_grp': Counter()}
        _gid = gold_in_docs if gold_in_docs is not None else [False] * bs
        _gbd = gold_bearing_docs if gold_bearing_docs is not None else [set()] * bs
        # gold-share canary (budget_ungate_gold only): over the docs held by the
        # GOLD-ANCHORED pools — the docs R_div is computed on — what fraction actually
        # bear gold? If R_div rises while this FALLS, the model is farming novelty by
        # padding trajectories with rare non-gold (junk) docs. Instance-level:
        # Σ|D_i ∩ gold_bearing_i| / Σ|D_i| over those pools.
        gs_num = 0
        gs_den = 0
        gs_ntraj = 0
        # --- live diagnostics (printed per step) ---
        # gate: how often the all-correct group gate fires. Should RISE as accuracy does.
        n_groups = 0
        n_allcorrect = 0
        # penalty gap: R_div of pool members that hold a gold-bearing doc vs those that
        # do not. The mean form penalised gold-holders (they share docs -> high m_d);
        # under div_sum_mode this gap should shrink or flip.
        gap_gold, gap_nogold = [], []
        nd_gold, nd_nogold = [], []      # |D_i| split by gold-holding
        ns_gold, ns_nogold = [], []      # n_search split by gold-holding
        # R_div distribution over the rollouts that actually get paid. The λ-sum form
        # is bounded by 1 (Σ1/m_d ≤ |D_i| ≤ 1/λ within budget), so max→1 means some
        # rollout held the full budget of docs that nobody else in the pool held.
        paid_vals = []
        for ph, idxs in strata.items():
            has_correct = any(base_scores[i] > 0 for i in idxs)
            n_groups += 1
            if all(base_scores[i] > 0 for i in idxs):
                n_allcorrect += 1
            is_gold_pool = (self.budget_ungate_gold and not self.budget_allrollouts
                            and not self.budget_alltraj and not self.budget_correct_nolimit)
            if self.budget_alltraj:
                # ABLATION: no correctness gate, no budget filter — pool = EVERY rollout
                # in the group. m_d counts over all. The most permissive variant.
                pool = list(idxs)
                min_pool = 1 if self.budget_lone_pool else 2
            elif self.budget_correct_nolimit:
                # ABLATION: correctness gate KEPT, budget filter DROPPED — pool = correct
                # rollouts regardless of over-search. Isolates the budget-filter axis.
                pool = [i for i in idxs if base_scores[i] > 0]
                min_pool = 1 if self.budget_lone_pool else 2
            elif self.budget_allrollouts:
                # ABLATION: correctness gate OFF — pool = every within-budget rollout
                # (correct AND wrong), for EVERY group. Same budget filter + lone-pool
                # guard as the original gate; only the R_acc condition is removed.
                pool = [i for i in idxs if within_budget[i]]
                min_pool = 1 if self.budget_lone_pool else 2
            elif self.budget_ungate_gold:
                # GOLD-ANCHORED, CORRECTNESS-AGNOSTIC: R_acc already rewards correctness,
                # so diversity is measured over EVERY rollout that retrieved the gold
                # evidence (within budget) — correct OR wrong, no has_correct split.
                # min_pool=1: a lone gold-reacher (size-1 pool) still gets R_div=1.0.
                pool = [i for i in idxs if within_budget[i] and _gid[i]]
                min_pool = 1
            elif self.budget_ungate:
                # asymmetric POOL (correct-pool if the group has any correct, else the
                # ALL-WRONG within-budget pool — R_acc ungated), but SYMMETRIC <2 guard:
                # a lone pool member (correct OR wrong) has no comparison partner, so its
                # R_div would be a constant 1.0 (budget-compliance), not measured novelty
                # → force 0 in BOTH tiers so every paid R_div is a genuine ≥2-way compare.
                if has_correct:
                    pool = [i for i in idxs if base_scores[i] > 0 and within_budget[i]]
                else:
                    pool = [i for i in idxs if within_budget[i]]
                min_pool = 2
            else:
                # original budget gate: correct & within-budget only. <2-eligible → 0,
                # unless budget_lone_pool (min_pool=1): a lone correct+budget rollout
                # keeps its raw R_div=1.0 (all m_d=1) instead of being zeroed.
                pool = [i for i in idxs if base_scores[i] > 0 and within_budget[i]]
                min_pool = 1 if self.budget_lone_pool else 2
            pool_hist['correct_grp' if has_correct else 'allwrong_grp'][len(pool)] += 1
            if self.div_off_all_correct and all(base_scores[i] > 0 for i in idxs):
                # ALL-CORRECT group: R_acc is constant so the group's GRPO advantage is
                # pure R_div. Zero it -> the saturated group contributes no gradient.
                continue
            if len(pool) < min_pool:
                continue
            doc_count = defaultdict(int)
            for j in pool:
                for d in traj_docs[j]:
                    doc_count[d] += 1
            for i in pool:
                if traj_docs[i]:
                    # div_gold_only: numerator restricted to answer-bearing docs;
                    # denominator below stays |D_i| (ALL retrieved docs), so junk
                    # padding still dilutes. Empty intersection -> novelty 0.
                    num_docs = (traj_docs[i] & _gbd[i]) if self.div_gold_only else traj_docs[i]
                    novelty = sum(1.0 / doc_count[d] for d in num_docs)
                    if self.div_capped:
                        out[i] = novelty / max(len(traj_docs[i]), self.div_topk)
                    elif self.div_sum_mode:
                        out[i] = novelty * self.div_lambda_map.get(
                            str(data_sources[i]), self.div_lambda_default)
                    else:
                        out[i] = novelty / len(traj_docs[i])
                if traj_docs[i]:
                    # Split |D_i| and n_search by gold-holding: without these the
                    # penalty gap cannot be decomposed into "rarity differs" vs
                    # "doc count differs", and they directly test the
                    # "found gold early -> stopped searching early" hypothesis.
                    if _gid[i]:
                        gap_gold.append(out[i])
                        nd_gold.append(len(traj_docs[i]))
                        ns_gold.append(len(per_step_docs_all[i]))
                    else:
                        gap_nogold.append(out[i])
                        nd_nogold.append(len(traj_docs[i]))
                        ns_nogold.append(len(per_step_docs_all[i]))
                    paid_vals.append(out[i])
                if is_gold_pool and traj_docs[i]:
                    gs_num += len(traj_docs[i] & _gbd[i])
                    gs_den += len(traj_docs[i])
                    gs_ntraj += 1
            if self.budget_mean_div:
                # CONTROL: replace each pool member's R_div with the POOL MEAN, killing
                # within-pool diversity differentiation (flat bonus only). Mean is over
                # the whole pool (members with no docs count as 0), so it matches the
                # signal a per-rollout run pays on average to this pool.
                mean_div = sum(out[i] for i in pool) / len(pool)
                for i in pool:
                    out[i] = mean_div
        self._budget_pool_hist = pool_hist
        self._div_gate = {'n_groups': n_groups, 'n_allcorrect': n_allcorrect,
                          'rate': (n_allcorrect / n_groups) if n_groups else float('nan'),
                          'active': bool(self.div_off_all_correct)}
        self._div_penalty_gap = {
            'gold_mean': (sum(gap_gold) / len(gap_gold)) if gap_gold else float('nan'),
            'nogold_mean': (sum(gap_nogold) / len(gap_nogold)) if gap_nogold else float('nan'),
            'n_gold': len(gap_gold), 'n_nogold': len(gap_nogold),
            'docs_gold': (sum(nd_gold) / len(nd_gold)) if nd_gold else float('nan'),
            'docs_nogold': (sum(nd_nogold) / len(nd_nogold)) if nd_nogold else float('nan'),
            'search_gold': (sum(ns_gold) / len(ns_gold)) if ns_gold else float('nan'),
            'search_nogold': (sum(ns_nogold) / len(ns_nogold)) if ns_nogold else float('nan'),
        }
        self._div_penalty_gap['gap'] = (self._div_penalty_gap['gold_mean']
                                        - self._div_penalty_gap['nogold_mean'])
        if paid_vals:
            _pv = np.array(paid_vals, dtype=float)
            self._div_rdiv_dist = {
                'n': int(_pv.size), 'mean': float(_pv.mean()), 'std': float(_pv.std()),
                'p50': float(np.percentile(_pv, 50)), 'p90': float(np.percentile(_pv, 90)),
                'max': float(_pv.max()), 'near1': float((_pv >= 0.9).mean()),
            }
        else:
            self._div_rdiv_dist = None
        self._budget_gold_share = {
            'num': gs_num, 'den': gs_den, 'n_traj': gs_ntraj,
            'share': (gs_num / gs_den) if gs_den else float('nan'),
        }
        return out

    def _evidence_exempt_diversity(self, strata, base_scores, per_step_docs_all,
                                   data_sources, bs, evd_exempt_all,
                                   evd_eligible_all=None, doc_level=False):
        """Evidence-Exempt R_div + guesser 제외. 면제 단위가 두 가지:

            E   = {i : EM=1 AND n_search_i <= budget(ds)}
            E'  = {i in E : |D_i ∩ Evd_i| >= 1}          # guesser 제외
            count(d) = |{j in E' : d in D_j}|

        call 면제 (doc_level=False, diversity_type='evidence_exempt'):
            call c is exempt  <=>  docs(c) ∩ Evd != {}
            w_i(d) = 1           if d in Evd  OR  d in (i 의 exempt call 회수 문서)
                   = 1/count(d)  otherwise
            (step-120 검증에서 top-3 call 단위 면제가 R_div≈1.0 포화를 낳아 폐기)

        doc 면제 (doc_level=True, diversity_type='evidence_doc_exempt', B'):
            w_i(d) = 1           if d in Evd_exempt(group)   # evidence 문서 자신만
                   = 1/count(d)  otherwise                   # 같은 call 이웃도 경쟁
            cf_replay B변형과 동일식. 음수는 쿼리 중복(이웃 count) 주도, evidence
            보유량과는 양(+)의 상관임을 replay 로 확인함 (verify_b_neg [D]).

        guesser_gate=False 이면 위 E' 자격 판정을 건너뛰고 pool = E, count(d) 도 E
        전체에서 센다 (지급 정책과 측정 분모를 함께 되돌림). 아래 '전원 R_div=0'
        규칙도 같이 꺼져 evidence 미검출 그룹은 면제 없는 1/count 로 환원된다.

        공통: R_div(i) = (1/|D_i|) Σ_{d in D_i} w_i(d)  for i in E'; D_i 는 set.
        (div_sum_mode 면 평균 대신 λ_ds·Σ — _div_normalize 참조)
        그룹 전체에서 evidence 미검출이면 기존(budget) 공식으로 자동 환원.
        단 gq_doc_exempt 는 환원 없이 전원 R_div=0 (자격 게이트 원칙).
        all-wrong 그룹은 E 가 비어 R_div 없음(기존과 동일).
        """
        if evd_eligible_all is None:
            evd_eligible_all = evd_exempt_all
        out = {i: 0.0 for i in range(bs)}
        traj_docs = [set().union(*per_step_docs_all[i]) if per_step_docs_all[i] else set()
                     for i in range(bs)]
        within_budget = [len(per_step_docs_all[i])
                         <= self.budget_map.get(str(data_sources[i]), self.budget_default)
                         for i in range(bs)]
        # exempt call = Evd_exempt 를 하나라도 반환한 call. 그 call 의 문서 전부 면제.
        exempt_docs, n_exempt_calls = [], []
        for i in range(bs):
            ex, nc = set(), 0
            for docs in per_step_docs_all[i]:
                if docs & evd_exempt_all[i]:
                    ex |= docs
                    nc += 1
            exempt_docs.append(ex)
            n_exempt_calls.append(nc)

        stats = {'n_groups': 0, 'n_fallback': 0, 'n_guesser': 0, 'n_Ep': 0,
                 'n_E': 0, 'n_zero_contrast': 0}
        paid_vals = []          # 실제 지급된 R_div — 포화(≈1.0) 감시용, div-watch (6)
        # containment 로만 자격을 얻은(= exempt call 0개) E' 멤버 비율, 데이터셋별.
        # detection 이상 신호 감시용 — replay 예상은 guesser 감소분(~2.6%p) 근처.
        stats['elig_only'] = defaultdict(int)
        stats['Ep_by_ds'] = defaultdict(int)
        for ph, idxs in strata.items():
            stats['n_groups'] += 1
            E = [i for i in idxs if base_scores[i] > 0 and within_budget[i]]
            stats['n_E'] += len(E)
            if not E:
                continue
            evd_group = set().union(*[evd_eligible_all[i] for i in idxs])
            evd_exempt_group = set().union(*[evd_exempt_all[i] for i in idxs])
            if not evd_group:
                stats['n_fallback'] += 1
                if getattr(self, 'diversity_type', '') == 'gq_doc_exempt' \
                        and getattr(self, 'guesser_gate', True):
                    # G_q 모드: 그룹 전원 자격 미달 -> 전원 R_div=0 (vanilla GRPO).
                    # 옛 공식 환원은 온라인 검출 실패 대비 장치였고 G_q 에선 모순.
                    # guesser_gate=False 면 이 규칙도 끄고 아래 환원 경로를 탄다.
                    continue
                # evidence 미검출 질문 -> 기존 공식으로 환원 (pool = E)
                cnt = defaultdict(int)
                for j in E:
                    for d in traj_docs[j]:
                        cnt[d] += 1
                for i in E:
                    if traj_docs[i]:
                        out[i] = self._div_normalize(
                            sum(1.0 / cnt[d] for d in traj_docs[i]),
                            len(traj_docs[i]), str(data_sources[i]))
                        paid_vals.append(out[i])
                continue
            # 자격 판정은 게이트 on/off 와 무관하게 항상 계산한다 — 게이트를 꺼도
            # "evidence 를 못 건진 정답 궤적" 비율은 계속 봐야 하기 때문.
            _qualified = [i for i in E if traj_docs[i] & evd_group]
            stats['n_guesser'] += len(E) - len(_qualified)
            Ep = _qualified if getattr(self, 'guesser_gate', True) else list(E)
            stats['n_Ep'] += len(Ep)
            if not Ep:
                continue                      # 전원 guesser -> 전원 0 (vanilla GRPO)
            cnt = defaultdict(int)
            for j in Ep:
                for d in traj_docs[j]:
                    cnt[d] += 1
            for i in Ep:
                ds = str(data_sources[i])
                stats['Ep_by_ds'][ds] += 1
                if n_exempt_calls[i] == 0:
                    stats['elig_only'][ds] += 1       # containment 로만 자격 획득
                if not traj_docs[i]:
                    continue
                # 면제 범위: doc_level 이면 Evd_exempt 문서 자신만, 아니면 exempt
                # call 의 이웃 문서까지. containment 로만 잡힌 문서는 어느 쪽에서도
                # 면제 대상이 아니라 1/count 을 그대로 받는다.
                if doc_level:
                    novelty = sum(
                        1.0 if d in evd_exempt_group else 1.0 / cnt[d]
                        for d in traj_docs[i])
                else:
                    novelty = sum(
                        1.0 if (d in evd_exempt_group or d in exempt_docs[i]) else 1.0 / cnt[d]
                        for d in traj_docs[i])
                out[i] = self._div_normalize(novelty, len(traj_docs[i]),
                                             str(data_sources[i]))
                paid_vals.append(out[i])
            if len(Ep) >= 2 and max(out[i] for i in Ep) - min(out[i] for i in Ep) < 1e-12:
                stats['n_zero_contrast'] += 1
        self._evd_exempt_stats = stats
        if paid_vals:
            _pv = np.array(paid_vals, dtype=float)
            self._div_rdiv_dist = {
                'n': int(_pv.size), 'mean': float(_pv.mean()), 'std': float(_pv.std()),
                'p50': float(np.percentile(_pv, 50)), 'p90': float(np.percentile(_pv, 90)),
                'max': float(_pv.max()), 'near1': float((_pv >= 0.9).mean()),
            }
        else:
            self._div_rdiv_dist = None
        return out

    def _div_watch(self, strata, per_step_docs_all, data_sources, bs):
        """Per-step diversity watch: search count split by BUDGET, and within-group
        document overlap. Budget-split matters because a macro mean over an NQ(B=1) +
        HotpotQA(B=2) mix moves for two unrelated reasons; per-B it is readable.
        Overlap is over ALL rollouts of a group (not just the R_div pool), so it
        tracks behaviour even when no diversity is paid (e.g. the lambda_div=0
        baseline)."""
        by_b = defaultdict(list)
        for i in range(bs):
            b = self.budget_map.get(str(data_sources[i]), self.budget_default)
            by_b[b].append(len(per_step_docs_all[i]))
        jac, uniq = [], []
        for idxs in strata.values():
            sets = [set().union(*per_step_docs_all[i]) if per_step_docs_all[i] else set()
                    for i in idxs]
            nz = [s for s in sets if s]
            uniq.append(len(set().union(*nz)) if nz else 0)
            if len(nz) >= 2:
                tot = cnt = 0
                for a in range(len(nz)):
                    for b2 in range(a + 1, len(nz)):
                        u = len(nz[a] | nz[b2])
                        if u:
                            tot += len(nz[a] & nz[b2]) / u
                            cnt += 1
                if cnt:
                    jac.append(tot / cnt)
        return {
            'b_mean': {b: (sum(v) / len(v)) for b, v in by_b.items()},
            'b_n': {b: len(v) for b, v in by_b.items()},
            # budget utilisation: share of rollouts spending the FULL budget (n_search
            # == B). Higher resolution than the mean — on B=2 this is the earliest
            # visible sign that the model started paying for a second search.
            'b_full': {b: (sum(1 for x in v if x >= b) / len(v)) for b, v in by_b.items()},
            'b_hist': {b: dict(sorted(Counter(v).items())) for b, v in by_b.items()},
            'jaccard': (sum(jac) / len(jac)) if jac else float('nan'),
            'n_jac_groups': len(jac),
            'uniq': (sum(uniq) / len(uniq)) if uniq else float('nan'),
        }

    def _gold_watch(self, strata, gold_bearing_docs, per_step_docs_all):
        """Per-group GOLD spread: how many DISTINCT gold-bearing docs a group's rollouts
        collectively retrieve (union), against how many a single rollout carries.
        Answers "does the group reach the answer through DIFFERENT evidence, or does
        every rollout land on the SAME gold doc?" — which _div_watch's uniq/jaccard
        cannot, since those count junk docs too: a group can diversify its padding
        while its gold doc stays fixed, and the two metrics would not tell you.
            uniq_g   = |union of (D_i & Gold_i) over i in g|   (0 if the group found none)
            per_traj = |D_i & Gold_i|, over gold-HOLDING rollouts only
            jaccard  = mean pairwise J over gold-holding rollouts (groups with >=2)
        uniq_hit > per_traj  =>  siblings hold different gold docs.
        uniq_hit == per_traj =>  every gold-holder holds the identical gold doc set.
        Gold detection + doc identity are _gold_bearing_docs' (per-doc, content-level
        EM-normalized containment), so these sets are subsets of the doc sets
        _div_watch counts. Rollouts that never retrieved cannot hold gold and are
        dropped from per_traj/jaccard, but their group still counts in grp_frac's
        denominator. Over ALL rollouts of a group (not just the R_div pool), matching
        _div_watch, so it stays comparable on arms that pay no diversity. Purely
        diagnostic: nothing computed here enters any reward."""
        def _mean(v):
            return (sum(v) / len(v)) if v else float('nan')
        uniq, per_traj, jac = [], [], []
        n_groups = n_groups_gold = n_traj = n_traj_gold = 0
        for idxs in strata.values():
            n_groups += 1
            searched = [i for i in idxs if per_step_docs_all[i]]
            n_traj += len(searched)
            sets = [gold_bearing_docs[i] for i in searched if gold_bearing_docs[i]]
            n_traj_gold += len(sets)
            u = set()
            for s in sets:
                u |= s
            uniq.append(len(u))
            if sets:
                n_groups_gold += 1
                per_traj.extend(len(s) for s in sets)
            if len(sets) >= 2:
                tot = cnt = 0
                for a in range(len(sets)):
                    for b in range(a + 1, len(sets)):
                        un = len(sets[a] | sets[b])
                        if un:
                            tot += len(sets[a] & sets[b]) / un
                            cnt += 1
                if cnt:
                    jac.append(tot / cnt)
        return {
            'uniq': _mean(uniq),                          # gold-less groups counted as 0
            'uniq_hit': _mean([u for u in uniq if u]),    # gold-holding groups only
            'per_traj': _mean(per_traj),
            'jaccard': _mean(jac),
            'n_jac_groups': len(jac),
            'traj_frac': (n_traj_gold / n_traj) if n_traj else float('nan'),
            'grp_frac': (n_groups_gold / n_groups) if n_groups else float('nan'),
            'n_groups': n_groups, 'n_traj': n_traj,
        }

    def _v2_shapley_div(self, strata, em_seqs, golds_all, base_scores,
                        P_max_exact=8, n_placebo=12, alpha=95, credit='shapley'):
        """Returns {i: R_div_i} (share over the group). Stage 0 (scorer) + Stage 1
        (2-round greedy filter -> survivor E_i) are SHARED; only Stage 2 credit
        differs by `credit`:
            'shapley' = MC-Shapley (antithetic perms) over v(S)=s(union E_i)-s(empty) (v2)
            'count'   = article-level 1/m_d over E_i (R_div_i = Σ 1/m / |∪titles|) (v1)
        PDF-faithful: Stage 2 is "전원 계산" — phi is computed over ALL rollouts of
        EVERY group with NO correctness reference whatsoever (base_scores is not read
        here). Correctness enters only at Stage 3 (the caller's 1[correct] multiplier;
        denominator sum_j phi_j keeps the forfeited incorrect j). v1 is the
        'doc=atomic, article=content' special case of the Shapley identity."""
        out = {i: 0.0 for i in range(len(em_seqs))}
        # Stage 2 is correctness-agnostic: compute for every group, no gate.
        live = dict(strata)
        if not live:
            return out

        # parse rollouts in live strata into UNITS — the granularity Stage 1 gates.
        #   per-doc  (self._v2_block False): each unit = 1 doc  (legacy behaviour)
        #   per-step (self._v2_block True):  each unit = that <information> block's
        #                                    top-k docs (matches the model's action
        #                                    query->top-k, keeps the filter from
        #                                    collapsing to the convergent answer doc).
        units, ques, gold = {}, {}, {}
        for ph, idxs in live.items():
            for i in idxs:
                if self._v2_block:
                    units[i] = self._v2_parse_blocks(em_seqs[i])
                else:
                    units[i] = [[d] for d in self._v2_parse_docs(em_seqs[i])]
                ques[i] = self._v2_question(em_seqs[i])
                gold[i] = self._v2_gold(golds_all[i])
        # placebo pool: UNITS (blocks/docs at the gate granularity) from all live
        # rollouts, keyed by stratum so a stratum draws placebos from OTHERS.
        pool_by_ph = {ph: [u for i in idxs for u in units[i]] for ph, idxs in live.items()}

        # -------- ROUND A: s(empty), forward prefixes (over units), placebo --------
        A_ctx, A_gold, A_idx = [], [], {}   # A_idx[tag] -> position
        def add_A(tag, ctx, g):
            A_idx[tag] = len(A_ctx); A_ctx.append(ctx); A_gold.append(g)
        for ph, idxs in live.items():
            # placebo units: from OTHER strata (same granularity as the gate)
            other = []
            for ph2, ul in pool_by_ph.items():
                if ph2 != ph:
                    other += ul
            random.Random(hash(ph) & 0xffffffff).shuffle(other)
            placebo = other[:n_placebo]          # list of placebo UNITS (each a doc-list)
            live[ph] = (idxs, placebo)
            qrep = ques[idxs[0]]; grep = gold[idxs[0]]
            for pj, p in enumerate(placebo):
                add_A(('plac', ph, pj), self._v2_ctx(p, qrep), grep)   # p is a doc-list
            for i in idxs:
                add_A(('empty', i), self._v2_ctx([], ques[i]), gold[i])
                for n in range(1, len(units[i]) + 1):
                    add_A(('pref', i, n),
                          self._v2_ctx(self._v2_dedup_union(units[i][:n]), ques[i]), gold[i])
        A = self._score_many(A_ctx, A_gold)
        def gA(tag):
            v = A[A_idx[tag]] if tag in A_idx else None
            return v

        # per-rollout fwd over units + per-stratum tau, R1 accepted
        s_empty, fwd, tau, ph_of = {}, {}, {}, {}
        for ph, (idxs, placebo) in live.items():
            se0 = gA(('empty', idxs[0])) or 0.0
            marg = []
            for pj, p in enumerate(placebo):
                sp = gA(('plac', ph, pj))
                if sp is not None:
                    marg.append(sp - se0)
            tau[ph] = self._pct(marg, alpha) if marg else 0.0
            for i in idxs:
                ph_of[i] = ph
                se = gA(('empty', i)); se = se if se is not None else 0.0
                s_empty[i] = se
                m = len(units[i]); prev = se; f = []
                for n in range(1, m + 1):
                    sn = gA(('pref', i, n)); sn = sn if sn is not None else prev
                    f.append(sn - prev); prev = sn
                fwd[i] = f
        accepted = {i: [n for n, fv in enumerate(fwd[i]) if fv > tau[ph_of[i]]]
                    for i in fwd}

        return self._v2_round_bc(out, live, units, ques, gold, s_empty, tau, accepted,
                                 fwd, P_max_exact, credit)

    def _v2_credit_count(self, out, live, E):
        """v1 Stage 2: article-level fractional credit on the SAME survivors E_i.
        m_t = #rollouts (article-title level) holding title t; R_div_i =
        (Σ_{t∈titles(E_i)} 1/m_t) / |∪ titles|. Sums to 1 over the group (the
        m=1-atomic special case of the Shapley efficiency identity)."""
        for ph, (idxs, _pl) in live.items():
            titles = {i: set(t for (t, c) in E.get(i, [])) for i in idxs}
            cnt = Counter()
            union = set()
            for i in idxs:
                union |= titles[i]
                for t in titles[i]:
                    cnt[t] += 1
            Z = len(union)
            for i in idxs:
                out[i] = (sum(1.0 / cnt[t] for t in titles[i]) / Z) if Z > 0 else 0.0
        return out

    def _v2_round_bc(self, out, live, units, ques, gold, s_empty, tau, accepted, fwd,
                     P_max_exact, credit='shapley'):
        # -------- ROUND B: rescue rejected UNITS vs the R1-accepted set --------
        ph_of = {}
        for ph, (idxs, _pl) in live.items():
            for i in idxs:
                ph_of[i] = ph
        B_ctx, B_gold, B_idx = [], [], {}
        def add_B(tag, ctx, g):
            B_idx[tag] = len(B_ctx); B_ctx.append(ctx); B_gold.append(g)
        for i in accepted:
            acc_units = [units[i][n] for n in accepted[i]]
            acc_docs = self._v2_dedup_union(acc_units)
            rejected = [n for n in range(len(units[i])) if n not in set(accepted[i])]
            if rejected:
                add_B(('base', i), self._v2_ctx(acc_docs, ques[i]), gold[i])
                for n in rejected:
                    cand = self._v2_dedup_union(acc_units + [units[i][n]])
                    add_B(('with', i, n), self._v2_ctx(cand, ques[i]), gold[i])
        B = self._score_many(B_ctx, B_gold) if B_ctx else []
        def gB(tag):
            return B[B_idx[tag]] if tag in B_idx else None
        E = {}
        for i in accepted:
            acc = list(accepted[i])
            base = gB(('base', i))
            if base is not None:
                for n in range(len(units[i])):
                    if n in set(accepted[i]):
                        continue
                    w = gB(('with', i, n))
                    if w is not None and (w - base) > tau[ph_of[i]]:
                        acc.append(n)
            acc.sort()
            E[i] = self._v2_dedup_union([units[i][n] for n in acc])

        # stash survivor TITLE-sets per rollout so the caller can classify each
        # >=2-correct group as collapse (identical survivors) vs forfeiture
        # (differing survivors) deterministically over ALL groups, not log samples.
        self._v2_last_E_titles = {
            i: tuple(sorted(set(t for (t, c) in E[i]))) for i in E}

        # v1 path: article 1/m credit on the survivors (no coalition scoring)
        if credit == 'count':
            return self._v2_credit_count(out, live, E)

        # -------- ROUND C: coalition values v(S), MC-Shapley (antithetic) --------
        # PDF spec: MC-Shapley with P≈8 antithetic permutations for EVERY group
        # (no exact-enumeration special case). 4 random perms + their reverses = 8.
        C_ctx, C_gold, C_idx = [], [], {}
        plans = {}        # ph -> (idxs, subsets list[frozenset], k)
        perms_by_ph = {}  # ph -> antithetic permutation list (kept separate from plans)
        for ph, (idxs, _pl) in live.items():
            k = len(idxs)
            qrep = ques[idxs[0]]; grep = gold[idxs[0]]
            rng = random.Random(hash(ph) & 0xffff)
            perms = []
            for _ in range(4):
                pi = list(range(k)); rng.shuffle(pi); perms.append(pi); perms.append(pi[::-1])
            ss = set()
            for pi in perms:
                for n in range(1, k + 1):
                    ss.add(frozenset(pi[:n]))
            subsets = list(ss)
            perms_by_ph[ph] = perms
            for S in subsets:
                tag = ('coal', ph, S)
                if tag not in C_idx:
                    union = self._v2_dedup_union([E[idxs[j]] for j in S])
                    C_idx[tag] = len(C_ctx); C_ctx.append(self._v2_ctx(union, qrep)); C_gold.append(grep)
            plans[ph] = (idxs, subsets, k)
        C = self._score_many(C_ctx, C_gold) if C_ctx else []
        def vS(ph, S, se):
            if not S:
                return 0.0
            j = C_idx.get(('coal', ph, frozenset(S)))
            if j is None or C[j] is None:
                return 0.0
            return C[j] - se

        for ph, (idxs, subsets, k) in plans.items():
            se = s_empty[idxs[0]]
            grand = vS(ph, frozenset(range(k)), se)
            if grand <= tau[ph]:           # no-learning guard
                for i in idxs:
                    out[i] = 0.0
                continue
            phi = [0.0] * k
            perms = perms_by_ph[ph]        # MC Shapley (antithetic perms)
            for pi in perms:
                pref = frozenset()
                for i in pi:
                    nxt = pref | {i}
                    phi[i] += max(0.0, vS(ph, nxt, se) - vS(ph, pref, se))
                    pref = nxt
            phi = [p / len(perms) for p in phi]
            Z = sum(phi) + 1e-6
            for loc, i in enumerate(idxs):
                out[i] = phi[loc] / Z
        return out

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        self._call_count += 1
        # evidence 캐시는 배치 내부에서만 유효하다 (같은 프롬프트의 G개 rollout 이 같은
        # 문서를 회수하는 경우를 노림). step 마다 프롬프트가 바뀌므로 비워서 누적을 막는다.
        self._evd_cache.clear()

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        bs = len(data)
        response_length = data.batch['responses'].shape[-1]
        outcome_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        step_level_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        stratum_step_id_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.long)

        # Per-trajectory collected info
        base_scores = []
        # R_fmt (0/1 structure-format reward), only populated for combine_mode=
        # 'em_fmt_doc'; 0.0 elsewhere.
        fmt_scores = []
        # Decoded prompt+response per trajectory, kept only for em_fmt_doc so the
        # placement branch can occasionally print a full sample (1/64), mirroring
        # qa_em's debug print which this path bypasses.
        em_seqs = []
        golds_all = []   # ground_truth target per rollout (for ΔP useful-weight)
        valid_response_lengths = []
        prompt_hashes = []
        per_step_docs_all = []
        evd_exempt_all = []        # per-rollout Evd_exempt (면제 대상)
        evd_eligible_all = []      # per-rollout Evd_eligible (E' 자격 판정 전용)
        gold_in_docs_all = []      # per-rollout bool: holds >=1 gold-bearing doc (budget_ungate_gold)
        gold_bearing_docs_all = [] # per-rollout set of gold-bearing (title,hash) — for gold-share canary
        data_sources_all = []   # per-rollout data_source (for budget-gated diversity)
        response_strs = []      # per-rollout decoded model response (for BLEU TD baseline)
        policy_strs = []        # same, but POLICY-ONLY (info_mask==1 spans) — BLEU TD input
        outcome_answers = []    # per-rollout normalized final answer (for batch_outcome exploration)
        step_ranges_all = []
        # Final <think>...<answer> chunk span per trajectory (the model-gen chunk
        # after the last <information>). Used by combine_mode='step_reward' to give
        # the answer block its own R_acc reward + its own normalization stratum.
        answer_ranges_all = []
        # For embedding arms (reasoning_embedding | search_embedding):
        #   per-traj list of per-step text strings (think OR search query).
        # Length matches per_step_docs_all[i] (= n_search of that traj).
        # Variable name kept as `think_per_step_all` for git-diff minimality —
        # contents are think OR search query depending on diversity_type.
        think_per_step_all = []
        # Per-turn embedding texts for the em_fmt_doc representation arms:
        # per-rollout LIST of per-turn strings (all <think> blocks for
        # reasoning_embedding, all <search> queries for search_embedding); []
        # for other arms. Each turn is embedded SEPARATELY then mean-pooled into
        # one e5 vector per rollout (correct-pool mean-pairwise), symmetric to the
        # doc soft-novelty arm and to the original search mean-pool shape.
        traj_emb_texts = []

        already_print_data_sources = {}
        # Keep first decoded sequence of each data_source for printing
        first_decoded = {}

        has_info_mask = 'info_mask' in data.batch.keys()

        for i in range(bs):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            response_str = self.tokenizer.decode(valid_response_ids)

            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            prompt_hash = hashlib.md5(prompt_str.encode()).hexdigest()

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            if self.combine_mode in ('em_fmt_doc', 'passk', 'batch_outcome'):
                # Decompose the reward into independent EM and FORMAT components
                # (the v0.3 graded reward couples them; here we need them separate).
                # extract_solution returns None unless there are >= 2 <answer> tags
                # (prompt example + model answer) — see qa_em_format. R_acc/R_fmt
                # are clean 0/1.
                ans = qa_em_format.extract_solution(sequences_str)
                r_acc = 1.0 if (ans is not None and
                                qa_em_format.em_check(ans, ground_truth['target'])) else 0.0
                is_fmt, _ = qa_em_format.is_valid_sequence(sequences_str)
                score = r_acc
                fmt_scores.append(1.0 if is_fmt else 0.0)
                em_seqs.append(sequences_str)
                # Normalized final answer = the "outcome" for batch_outcome
                # exploration. None (unparseable) → never matches any sibling.
                outcome_answers.append(
                    qa_em_format.normalize_answer(ans) if ans is not None else None)
            else:
                compute_score_fn = _select_rm_score_fn(data_source)
                score = compute_score_fn(
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    format_score=self.format_score,
                )
                fmt_scores.append(0.0)
                em_seqs.append(None)

            resp_info_mask = None
            if has_info_mask and self.diversity_type in ('evidence_exempt',
                                                         'evidence_doc_exempt',
                                                         'gq_doc_exempt'):
                # 단일 디코드로 per-call 문서 + evidence 를 함께 뽑는다.
                resp_info_mask = data_item.batch['info_mask'][prompt_length:][:int(valid_response_length)].tolist()
                _g = ground_truth['target'] if isinstance(ground_truth, dict) else ground_truth
                if isinstance(_g, str):
                    _g = [_g]
                # NB: `_g or []` 금지 — numpy array 는 truth value 가 모호하다
                # (_gold_bearing_docs 의 같은 주석 참고).
                if _g is None:
                    _g = []
                _gn = []
                for g in _g:
                    if g is None:
                        continue
                    gn = qa_em_format.normalize_answer(g)
                    if gn:
                        _gn.append(gn)
                if self.diversity_type == 'gq_doc_exempt':
                    # 오프라인 G_q 조회: 온라인 검출 생략(_gn=[], _sup=None) 후
                    # evd = D_i ∩ G_q (md5 기준). 면제·자격 동일 집합.
                    docs_per_step, _, _ = self._docs_and_evidence_per_step(
                        valid_response_ids, resp_info_mask, [], None, prompt_hash)
                    _gq = self._gq_md5.get(
                        self._gq_key_from_prompt(prompt_str), frozenset())
                    _evx = _eve = {ident for step in docs_per_step
                                   for ident in step if ident[1] in _gq}
                else:
                    _sup = self._support_titles(data_item)
                    docs_per_step, _evx, _eve = self._docs_and_evidence_per_step(
                        valid_response_ids, resp_info_mask, _gn, _sup, prompt_hash)
                evd_exempt_all.append(_evx)
                evd_eligible_all.append(_eve)
                n_info = len(docs_per_step)
                gen_chunks = self._find_all_gen_chunks(resp_info_mask)
                ranges = [gen_chunks[k] if k < len(gen_chunks) else None
                          for k in range(n_info)] if n_info > 0 else []
                answer_range = gen_chunks[n_info] if len(gen_chunks) > n_info else None
            elif has_info_mask:
                evd_exempt_all.append(set()); evd_eligible_all.append(set())
                resp_info_mask = data_item.batch['info_mask'][prompt_length:][:int(valid_response_length)].tolist()
                docs_per_step = self._extract_docs_per_step(valid_response_ids, resp_info_mask)
                n_info = len(docs_per_step)
                # All model-generated chunks (info_mask=1 runs): the first n_info
                # are the per-step think+search spans, and the chunk right after
                # them (if any) is the final think+answer span.
                gen_chunks = self._find_all_gen_chunks(resp_info_mask)
                if n_info > 0:
                    ranges = [gen_chunks[k] if k < len(gen_chunks) else None
                              for k in range(n_info)]
                else:
                    ranges = []
                answer_range = gen_chunks[n_info] if len(gen_chunks) > n_info else None
            else:
                evd_exempt_all.append(set()); evd_eligible_all.append(set())
                docs_per_step = []
                n_info = 0
                ranges = []
                answer_range = None

            # gold-bearing docs: which of the rollout's retrieved docs CONTAIN the gold
            # answer (EM-norm, per-doc). gold_in = holds >=1. Only for budget_ungate_gold.
            if has_info_mask and (self.budget_ungate_gold or self.div_gold_only
                                  or self.div_gold_watch
                                  or self.diversity_type in ('gold_cov', 'gold_cov_all')):
                _golds = ground_truth['target'] if isinstance(ground_truth, dict) else ground_truth
                if isinstance(_golds, str):
                    _golds = [_golds]
                gbset = self._gold_bearing_docs(valid_response_ids, resp_info_mask, _golds)
                gold_in = len(gbset) > 0
            else:
                gbset = set()
                gold_in = False

            base_scores.append(score)
            golds_all.append(ground_truth.get('target') if isinstance(ground_truth, dict) else ground_truth)
            valid_response_lengths.append(int(valid_response_length))
            prompt_hashes.append(prompt_hash)
            per_step_docs_all.append(docs_per_step)
            gold_in_docs_all.append(gold_in)
            gold_bearing_docs_all.append(gbset)
            data_sources_all.append(data_source)
            response_strs.append(response_str)
            # BLEU TD input: policy-only text. info_mask is the authority; the regex
            # is only a fallback for runs whose batch carries no info_mask at all.
            if resp_info_mask is not None:
                policy_strs.append(
                    self._policy_only_text(valid_response_ids, resp_info_mask))
            else:
                policy_strs.append(
                    re.sub(r'<information>.*?</information>', ' ',
                           response_str or '', flags=re.DOTALL))
            step_ranges_all.append(ranges)
            answer_ranges_all.append(answer_range)

            if self.lambda_div > 0 and self.diversity_type == 'reasoning_embedding':
                think_per_step_all.append(
                    self._extract_think_per_step(response_str, n_info)
                )
                traj_emb_texts.append(self._extract_reasoning_turns(response_str))
            elif self.lambda_div > 0 and self.diversity_type == 'search_embedding':
                think_per_step_all.append(
                    self._extract_search_per_step(response_str, n_info)
                )
                traj_emb_texts.append(self._extract_search_turns(response_str))
            else:
                think_per_step_all.append([])
                traj_emb_texts.append([])

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        # ---- Stratify: stratum = (prompt_hash) only ----
        # Step-level sub-grouping was dropped — moved to trajectory-level diversity:
        # one score per trajectory (D_i = union of docs across all info chunks),
        # placed on the last token (outcome-style), normalized within (prompt_hash)
        # group of n_agent trajectories. This removes singleton strata entirely
        # and removes the per-step credit assignment that wasn't verified to help.
        strata = defaultdict(list)
        for idx, ph in enumerate(prompt_hashes):
            strata[ph].append(idx)

        # ---- Compute trajectory-level diversity scores ----
        # traj_div_scores[i] in [0, 1] = uniqueness of trajectory i's full doc set
        # (or full text embedding) vs other trajectories in same prompt group.
        traj_div_scores = [0.0] * bs
        traj_div_log = []           # all trajectory-level scores (for logging)
        stratum_size_log = []
        cos_sim_log = []
        nn_sim_log = []

        if self.diversity_type == 'doc_set_unique':
            # D_i = union of all docs across all info chunks in trajectory i.
            # Comparison pool: ONLY other correct trajectories within same prompt.
            # Wrong trajectories' docs are NOT included (they represent failed paths,
            # not alternative correct paths; including them would unfairly penalize
            # correct trajectories that happen to retrieve docs also retrieved by
            # a failed peer).
            traj_docs = [
                set().union(*per_step_docs_all[i]) if per_step_docs_all[i] else set()
                for i in range(bs)
            ]
            for ph, indices in strata.items():
                stratum_size_log.append(len(indices))
                correct_indices = [i for i in indices if base_scores[i] > 0]
                # < 2 correct: no comparison possible (singleton-correct also
                # handled at placement, but skipping here avoids wasted work).
                if len(correct_indices) < 2:
                    continue
                correct_docs = [traj_docs[i] for i in correct_indices]
                for local_i, global_i in enumerate(correct_indices):
                    docs_i = correct_docs[local_i]
                    if len(docs_i) == 0:
                        score = 0.0
                    else:
                        other_docs = set()
                        for local_j in range(len(correct_indices)):
                            if local_i != local_j:
                                other_docs |= correct_docs[local_j]
                        unique = docs_i - other_docs
                        score = len(unique) / len(docs_i)
                    traj_div_scores[global_i] = score
                    traj_div_log.append(score)
        elif self.diversity_type in ('reasoning_embedding', 'search_embedding') \
                and self.lambda_div > 0 \
                and self.diversity_grouping == 'trajectory':
            # Trajectory representation = MEAN of per-step text embeddings.
            # NOTE: only the 'trajectory' grouping arm mean-pools steps into one
            # vector. Under step_prompt[/_nsearch] grouping, each step's query is
            # scored against the *same-step* queries of sibling rollouts instead
            # (slide-18 "search query diversity at step k") — see the step-level
            # embedding branch in the placement section below. Skipping this block
            # there avoids redundant encode + double-logging into traj_div_log.
            # Why not concat-then-embed: long concatenations risk Sentence-Transformer
            # truncation and introduce length-bias (trajectory with more steps gets
            # a different-scale embedding). Mean-of-per-step embeddings avoids both:
            # each step embedded individually at full resolution, then averaged.
            flat_texts = []
            flat_ref = []  # (traj_i, step_k) parallel to flat_texts
            for i in range(bs):
                for step_k, t in enumerate(think_per_step_all[i]):
                    if t and t.strip():
                        flat_texts.append(t)
                        flat_ref.append((i, step_k))
            flat_embs = self._embed_texts(flat_texts) if flat_texts else []

            # Group per-step embeddings by trajectory
            per_traj_step_embs = defaultdict(list)
            for (traj_i, _step_k), e in zip(flat_ref, flat_embs):
                if e is not None:
                    per_traj_step_embs[traj_i].append(e)

            # Mean over steps → one embedding per trajectory
            import torch as _torch
            embed_by_traj = {}
            for traj_i, embs in per_traj_step_embs.items():
                stacked = _torch.stack(embs, dim=0)
                mean_emb = stacked.mean(dim=0)
                # Re-normalize so downstream cosine distance behaves consistently
                norm = mean_emb.norm(p=2)
                if norm > 0:
                    mean_emb = mean_emb / norm
                embed_by_traj[traj_i] = mean_emb

            # Same restriction as doc-set: compare correct trajectories only.
            for ph, indices in strata.items():
                stratum_size_log.append(len(indices))
                correct_indices = [i for i in indices if base_scores[i] > 0]
                if len(correct_indices) < 2:
                    continue
                embs = [embed_by_traj.get(i) for i in correct_indices]
                scores, stats = self._compute_emb_diversity_per_traj(embs)
                cos_sim_log.extend(stats['cos_sims'])
                nn_sim_log.extend(stats['nn_sims'])
                for local_i, global_i in enumerate(correct_indices):
                    traj_div_scores[global_i] = scores[local_i]
                    traj_div_log.append(scores[local_i])
        elif self.diversity_type in ('reasoning_embedding', 'search_embedding') \
                and self.lambda_div > 0 \
                and self.diversity_grouping in ('step_prompt', 'step_prompt_nsearch') \
                and self.combine_mode == 'pre_norm':
            # Step-wise (position-wise) query diversity, aggregated to ONE scalar
            # per trajectory, for REWARD-LEVEL combination (combine_mode=pre_norm):
            #     R_total_i = R_acc_i + λ·1[R_acc_i=1]·traj_div_scores[i]
            # combined downstream in the pre_norm placement branch, then a SINGLE
            # GRPO normalization (adv_estimator=grpo). This realizes the slide-18
            # formula literally (reward weighted-sum → one advantage), unlike the
            # post_norm/grpo_step_level path which normalizes the two signals
            # separately and sums at the advantage level.
            #
            # Per step k, the k-th query of correct rollout i is compared to the
            # k-th queries of its siblings in the same (prompt[, n_search], step_k)
            # stratum (accuracy-gated, group needs >=2 members):
            #     score_{i,k} = (1/(N-1)) Σ_{j!=i} (1 - cos(e_i^k, e_j^k))
            # Trajectory scalar = MEAN over i's scored steps. Mean (not sum) keeps
            # R_div ∈ [0,1] so R_total ∈ [1, 1+λ] for correct trajs, and removes the
            # incentive to add searches purely to accumulate more diversity bonus.
            use_nsearch = (self.diversity_grouping == 'step_prompt_nsearch')

            # Embed each per-step query of every correct trajectory (cached).
            flat_texts, flat_ref = [], []
            for i in range(bs):
                if base_scores[i] <= 0:
                    continue
                for k, t in enumerate(think_per_step_all[i]):
                    if t and t.strip():
                        flat_texts.append(t)
                        flat_ref.append((i, k))
            flat_embs = self._embed_texts(flat_texts) if flat_texts else []
            emb_by_traj_step = {}  # (traj_i, step_k) -> tensor (d,)
            for (i, k), e in zip(flat_ref, flat_embs):
                if e is not None:
                    emb_by_traj_step[(i, k)] = e

            # Group correct trajectories' steps into strata (k-th query together).
            step_groups = defaultdict(list)
            for i in range(bs):
                if base_scores[i] <= 0:
                    continue
                ph = prompt_hashes[i]
                n_search_i = len(think_per_step_all[i])
                for k in range(n_search_i):
                    key = (ph, n_search_i, k) if use_nsearch else (ph, k)
                    step_groups[key].append((i, k))

            # Score each stratum; collect per-trajectory step scores.
            per_traj_step_scores = defaultdict(list)
            for key, entries in step_groups.items():
                stratum_size_log.append(len(entries))
                if len(entries) < 2:
                    continue
                embs = [emb_by_traj_step.get((i, kk)) for (i, kk) in entries]
                scores, stats = self._compute_emb_diversity_per_traj(embs)
                cos_sim_log.extend(stats['cos_sims'])
                nn_sim_log.extend(stats['nn_sims'])
                for local_i, (i, kk) in enumerate(entries):
                    per_traj_step_scores[i].append(scores[local_i])

            # Aggregate to one scalar per trajectory (mean over scored steps).
            for i, sc in per_traj_step_scores.items():
                if sc:
                    traj_div_scores[i] = float(sum(sc) / len(sc))
                    traj_div_log.append(traj_div_scores[i])
        else:
            # 'none', or embedding arm with lambda_div == 0
            stratum_size_log = [len(indices) for indices in strata.values()]

        # ---- Build outcome_tensor, step_level_tensor, stratum_step_id_tensor ----
        # Trajectory-level diversity reward: single score per trajectory broadcast
        # across all valid response tokens. Group = prompt_hash. Accuracy-gated:
        # only correct trajectories receive the bonus. Additionally: skip prompts
        # with < 2 correct trajectories — single-correct case has no group to
        # normalize against, so the diversity advantage degenerates to raw score
        # (≥ 0, no negative direction) which is not meaningful learning signal.
        correct_count_per_prompt = defaultdict(int)
        for i in range(bs):
            if base_scores[i] > 0:
                correct_count_per_prompt[prompt_hashes[i]] += 1

        group_id_map = {}      # prompt_hash -> gid
        next_gid = 1
        applied_count = 0
        skipped_singleton_count = 0

        # Per-step breakdown collector (step_prompt mode only)
        step_score_by_step = defaultdict(list)  # step_k -> list of scores
        step_group_size_by_step = defaultdict(list)  # step_k -> list of group sizes (incl singleton)

        if self.combine_mode == 'passk':
            # ===== Pass@k Training =====
            # Stash the raw per-rollout labels; the advantage (adv_estimator=
            # grpo_passk) is built downstream per passk_reward_mode:
            #   'joint_binary' : pass@k over binary success 1[R_acc ∧ R_fmt] — treats
            #                    R_acc + R_fmt as ONE criterion (symmetric to the
            #                    Proposed run's summed base). [default]
            #   'best_of_k'    : best-of-k (generalized pass@k) over R_total =
            #                    R_acc + R_fmt ∈ {0,1,2} — 3-valued adv A_2>A_1>A_0.
            #   'acc_only'     : pass@k over binary R_acc + centered R_fmt term.
            # outcome_tensor holds R_acc only, for accuracy logging.
            for i in range(bs):
                valid_len = valid_response_lengths[i]
                if valid_len > 0:
                    outcome_tensor[i, valid_len - 1] = base_scores[i]   # R_acc (EM), logging
            data.meta_info['passk_acc']    = [float(base_scores[i]) for i in range(bs)]
            data.meta_info['passk_fmt']    = [float(fmt_scores[i]) for i in range(bs)]
            data.meta_info['passk_reward'] = [float(base_scores[i] + fmt_scores[i])
                                              for i in range(bs)]     # R_total ∈ {0,1,2}
            for i in range(bs):
                if self.num_examine == 0 and random.randint(1, 64) == 1:
                    print("-------------------------------- [passk sample]")
                    print(f"R_acc={base_scores[i]:.0f} R_fmt={fmt_scores[i]:.0f} "
                          f"R_total={base_scores[i]+fmt_scores[i]:.0f} "
                          f"(advantage analytic downstream per passk_reward_mode)")
                    print(f"Solution string: {em_seqs[i]}")
        elif self.combine_mode == 'batch_outcome':
            # ===== Outcome-based Batch Exploration (Song et al. 2025, arXiv:2509.06941) =====
            # Base reward = R_acc + R_fmt at the last token — SAME base as
            # train_grpo.sh, so this baseline differs from the proposed run only
            # in the diversity mechanism. The batch-exploration bonus
            #   b_batch_i = −(1/N) Σ_{j≠i} 1{a_i = a_j}        (a = final answer)
            # is added at the ADVANTAGE level downstream (adv_estimator=grpo_batch),
            # un-normalized — faithful to Eq.2 (Â_i + c·b_i). NOT correct-gated: the
            # bonus rewards answer diversity on ALL rollouts incl. unsolved ones
            # (the method's whole point — combats transfer of diversity degradation).
            # outcome_tensor → token_level_rewards → base GRPO advantage.
            for i in range(bs):
                valid_len = valid_response_lengths[i]
                if valid_len > 0:
                    outcome_tensor[i, valid_len - 1] = self._acc_fmt_reward(
                        base_scores[i], fmt_scores[i])
            data.meta_info['batch_outcome'] = list(outcome_answers)
            data.meta_info['batch_acc'] = [float(base_scores[i]) for i in range(bs)]  # R_acc, for all-correct zeroing
            for i in range(bs):
                if self.num_examine == 0 and random.randint(1, 64) == 1:
                    print("-------------------------------- [batch_outcome sample]")
                    print(f"R_acc={base_scores[i]:.0f} R_fmt={fmt_scores[i]:.0f} "
                          f"answer={outcome_answers[i]!r} (batch bonus c·b at advantage)")
                    print(f"Solution string: {em_seqs[i]}")
        elif self.combine_mode == 'em_fmt_doc':
            # ===== Trajectory-level: R_total = R_acc + R_fmt + 1[acc]·λ·R_div =====
            # R_div is measured over ALL rollouts in the prompt group (correct AND
            # wrong); only the *application* is accuracy-gated (wrong rollouts add
            # no diversity). One scalar at the last token; adv_estimator=grpo
            # normalizes the 5-rollout group. The R_div definition switches on
            # diversity_type (all three are trajectory-level, CORRECT-pool only):
            #   'reasoning_embedding' / 'search_embedding' → embed EACH turn's text
            #       (all <think> blocks incl. final reasoning, OR all <search>
            #       queries) separately, MEAN-POOL into one vector e_i, then
            #       R_div_i = (1/(|C|-1))·Σ_{j∈C,j≠i}(1 − cos(e_i, e_j)), C=correct.
            #   otherwise (doc) → soft-novelty:
            #       R_div_i = (1/|D_i|)·Σ_{d∈D_i} 1/(#correct rollouts holding d).
            lam = getattr(self, 'cur_lambda_div', self.lambda_div)
            if lam != lam:  # NaN guard
                lam = self.lambda_div

            div_dbg = [""] * bs   # per-rollout label for the 1/64 debug print

            if self.diversity_type in ('reasoning_embedding', 'search_embedding') \
                    and self.lambda_div > 0:
                # --- Representation (reasoning OR search-query) embedding diversity ---
                # SYMMETRIC to the doc soft-novelty arm: trajectory-level, CORRECT-
                # pool only. Per rollout, embed EACH turn's text separately (no
                # 512-token truncation, equal weight per turn), MEAN-POOL +
                # renormalize into one vector e_i. Then within each prompt group,
                # mean pairwise (1-cos) is computed over CORRECT rollouts ONLY
                # (|C|-1 denominator), matching the doc arm's correct-pool gating.
                # Wrong rollouts get 0 (gated downstream too). reasoning / query /
                # doc then differ ONLY in what is embedded — a clean layer ablation.
                # (This is the original search mean-pool shape, now (a) applied to
                # reasoning too and (b) switched from ungated to correct-pool.)
                flat_texts, flat_ref = [], []
                for i in range(bs):
                    for k, t in enumerate(traj_emb_texts[i]):
                        if t and t.strip():
                            flat_texts.append(t)
                            flat_ref.append((i, k))
                flat_embs = self._embed_texts(flat_texts) if flat_texts else []
                per_traj_embs = defaultdict(list)
                for (i, k), e in zip(flat_ref, flat_embs):
                    if e is not None:
                        per_traj_embs[i].append(e)
                # mean over a rollout's per-turn embeddings, then renormalize
                embed_by_traj = {}
                for i, embs in per_traj_embs.items():
                    mvec = torch.stack(embs, dim=0).mean(dim=0)
                    nrm = mvec.norm(p=2)
                    if nrm > 0:
                        mvec = mvec / nrm
                    embed_by_traj[i] = mvec
                for ph, indices in strata.items():
                    correct_idx = [i for i in indices if base_scores[i] > 0]
                    if len(correct_idx) < 2:
                        for i in indices:
                            traj_div_scores[i] = 0.0
                            div_dbg[i] = "(<2 correct)"
                        continue
                    # only correct rollouts in the comparison pool (None → score 0,
                    # excluded from the others' (|C|-1) denominator by the helper)
                    embs = [embed_by_traj.get(i) for i in correct_idx]
                    scores, stats = self._compute_emb_diversity_per_traj(embs)
                    cos_sim_log.extend(stats['cos_sims'])
                    nn_sim_log.extend(stats['nn_sims'])
                    for local_i, i in enumerate(correct_idx):
                        traj_div_scores[i] = scores[local_i]
                        traj_div_log.append(scores[local_i])
                        div_dbg[i] = (f"(emb-div {scores[local_i]:.2f} corr-pool, "
                                      f"{len(per_traj_embs.get(i, []))}t)")
                    for i in indices:
                        if base_scores[i] <= 0:
                            traj_div_scores[i] = 0.0
                            div_dbg[i] = "(wrong)"
            elif self.diversity_type == 'meaningful' and self.lambda_div > 0:
                # --- Meaningful-Diversity (revised spec): CORRECT rollouts only.
                #   R_div(i) = Σ_d φ_d·(1/m_d); φ_d = EXACT 2^n within-rollout Shapley
                #   over distinct docs (content-hash); v(S)=length-norm logP(gt|docs S);
                #   m_d = #correct rollouts in the GROUP holding d. raw, negatives kept.
                md = self._md_diversity(strata, em_seqs, golds_all, base_scores)
                for i in range(bs):
                    traj_div_scores[i] = md.get(i, 0.0)
                    if base_scores[i] <= 0:
                        div_dbg[i] = "(wrong)"
                    elif traj_div_scores[i] != 0.0:
                        traj_div_log.append(traj_div_scores[i])
                        div_dbg[i] = f"(md {traj_div_scores[i]:+.3f})"
                    else:
                        div_dbg[i] = "(md 0)"
            elif self.diversity_type == 'rgsn' and self.lambda_div > 0:
                # --- RGSN: Relevance-Gated Soft-Novelty (fix for Meaningful-Div's
                #   Shapley submodularity). R_div(i) = max(p̂·(1/n)Σ_d w_d·(1/m_d), 0);
                #   w_d = max(0, isolated marginal v({d})). See _rgsn_diversity. ---
                rg = self._rgsn_diversity(strata, em_seqs, golds_all, base_scores)
                for i in range(bs):
                    traj_div_scores[i] = rg.get(i, 0.0)
                    if base_scores[i] <= 0:
                        div_dbg[i] = "(wrong)"
                    elif traj_div_scores[i] != 0.0:
                        traj_div_log.append(traj_div_scores[i])
                        div_dbg[i] = f"(rgsn {traj_div_scores[i]:.3f})"
                    else:
                        div_dbg[i] = "(rgsn 0)"
            elif self.diversity_type in ('evidence_exempt', 'evidence_doc_exempt',
                                         'gq_doc_exempt') \
                    and self.lambda_div > 0:
                # --- Evidence-Exempt R_div + guesser 제외 (see method docstring).
                #     doc_exempt(B') = doc 단위 면제: evidence 문서만 w=1 ---
                bd = self._evidence_exempt_diversity(
                    strata, base_scores, per_step_docs_all, data_sources_all, bs,
                    evd_exempt_all, evd_eligible_all,
                    doc_level=(self.diversity_type in ('evidence_doc_exempt',
                                                       'gq_doc_exempt')))
                for i in range(bs):
                    traj_div_scores[i] = bd.get(i, 0.0)
                    if base_scores[i] <= 0:
                        div_dbg[i] = "(wrong)"
                    elif traj_div_scores[i] != 0.0:
                        traj_div_log.append(traj_div_scores[i])
                        div_dbg[i] = f"(evd {traj_div_scores[i]:.3f})"
                    else:
                        div_dbg[i] = "(guesser/over-budget)"
            elif self.diversity_type == 'budget' and self.lambda_div > 0:
                # --- Budget-gated soft-novelty (Proposed-Revised + search budget):
                #   eligible = correct AND n_search ≤ budget(#supporting paras);
                #   R_div = soft-novelty over the eligible pool, 0 if over-budget.
                #   See _budget_soft_diversity. Over-search earns no diversity. ---
                bd = self._budget_soft_diversity(
                    strata, base_scores, per_step_docs_all, data_sources_all, bs,
                    gold_in_docs=gold_in_docs_all,
                    gold_bearing_docs=gold_bearing_docs_all)
                for i in range(bs):
                    traj_div_scores[i] = bd.get(i, 0.0)
                    if traj_div_scores[i] != 0.0:
                        traj_div_log.append(traj_div_scores[i])
                        nsd = len(per_step_docs_all[i])
                        pool = 'wrong-pool' if base_scores[i] <= 0 else 'correct-pool'
                        div_dbg[i] = f"(budget-soft {traj_div_scores[i]:.3f}, {nsd}srch, {pool})"
                    elif base_scores[i] <= 0:
                        div_dbg[i] = "(wrong, no div: over-budget or mixed group)"
                    else:
                        div_dbg[i] = "(R_div 0: over-budget/no-docs/no-gold)"
            elif self.diversity_type == 'bleu' and self.lambda_div > 0:
                # --- DIVER Textual Diversity baseline (BLEU dissimilarity over the
                #   prompt group): R_div_i = (1/(|G|-1)) Σ_{j≠i}(1 − BLEU(o_i,o_j)),
                #   measured over ALL rollouts, applied to correct only (gate below).
                #   Replaces doc-novelty with text-novelty. See _bleu_diversity. ---
                bl = self._bleu_diversity(strata, policy_strs, base_scores, bs)
                for i in range(bs):
                    traj_div_scores[i] = bl.get(i, 0.0)
                    if base_scores[i] <= 0:
                        div_dbg[i] = "(wrong)"
                    elif traj_div_scores[i] != 0.0:
                        traj_div_log.append(traj_div_scores[i])
                        div_dbg[i] = f"(bleu-td {traj_div_scores[i]:.3f})"
                    else:
                        div_dbg[i] = "(bleu-td 0)"
            elif self.diversity_type in ('gold_cov', 'gold_cov_all') and self.lambda_div > 0:
                # --- OURS: per-rollout gold(answer)-doc coverage SHARE (Σ_i R_div_i = 1) ---
                #   Per prompt group, over the CORRECT rollouts:
                #     D        = ∪_{j∈correct} gold_bearing_docs_all[j]   (distinct gold docs)
                #     count(d) = #{j∈correct : d ∈ gold_bearing_docs_all[j]}
                #   Each correct rollout i is scored on its OWN gold docs D_i, but
                #   NORMALIZED by the GROUP total |D| (NOT |D_i|) so the shares sum to 1:
                #     R_div_i = (1/|D|) · Σ_{d∈D_i} 1/count(d)
                #   Σ_{i∈correct} R_div_i = 1 exactly: each doc d contributes
                #   count(d)·(1/count(d))/|D| = 1/|D|, summed over |D| docs → 1.
                #   R_div_i DIFFERS per rollout (D_i differs). Wrong rollouts → 0
                #   (also gated at placement; λ=1 here). A rollout's share is HIGH
                #   when it covers many gold docs that FEW other correct rollouts hold.
                #   gold_bearing_docs_all[i] = set of (title, content_hash) docs held
                #   by rollout i whose CONTENT contains the gold answer (_gold_bearing_docs).
                #   'gold_cov_all' widens the pool from the CORRECT rollouts to ALL
                #   rollouts of the group (correct AND wrong): D, count(d) and the shares
                #   are all computed over every rollout, and the 1[correct] gate at
                #   placement is lifted so a wrong rollout that reached a gold doc few
                #   siblings hold still gets paid. Formula, |D| normalization and Σ_i
                #   R_div_i = 1 are unchanged — only the pool differs.
                _all_pool = (self.diversity_type == 'gold_cov_all')
                for ph, idxs in strata.items():
                    correct_idx = list(idxs) if _all_pool \
                        else [i for i in idxs if base_scores[i] > 0]
                    doc_count = defaultdict(int)   # count(d) over the pool
                    D = set()                      # distinct gold docs of the group
                    for j in correct_idx:
                        gdocs = gold_bearing_docs_all[j]
                        D |= gdocs
                        for d in gdocs:
                            doc_count[d] += 1
                    nD = len(D)
                    for i in idxs:
                        if not _all_pool and base_scores[i] <= 0:
                            traj_div_scores[i] = 0.0
                            div_dbg[i] = "(wrong)"
                            continue
                        D_i = gold_bearing_docs_all[i]
                        if not D_i or nD == 0:
                            traj_div_scores[i] = 0.0
                            div_dbg[i] = "(no gold doc)"
                            traj_div_log.append(0.0)
                            continue
                        r_div = sum(1.0 / doc_count[d] for d in D_i) / nD
                        traj_div_scores[i] = r_div
                        div_dbg[i] = f"(gold-cov {r_div:.3f} |D_i|={len(D_i)}/|D|={nD})"
                        traj_div_log.append(r_div)
            elif self.diversity_type in ('v2_shapley', 'v1_filtered') and self.lambda_div > 0:
                # --- v2 (Shapley) / v1 (article 1/m): SAME Stage-0 scorer + Stage-1
                # filter; only the Stage-2 credit rule differs. Clean head-to-head of
                # "coverage Shapley" vs "doc-count 1/m" on identical survivor sets.
                # R_div_i is the share of group coverage held by rollout i; computed
                # over ALL rollouts of each >=2-correct group; gating at placement.
                _credit = 'shapley' if self.diversity_type == 'v2_shapley' else 'count'
                _tag = 'v2-shapley' if _credit == 'shapley' else 'v1-count'
                self._v2_last_E_titles = {}   # reset; populated only if live strata exist
                v2 = self._v2_shapley_div(strata, em_seqs, golds_all, base_scores,
                                          credit=_credit)
                for i in range(bs):
                    traj_div_scores[i] = v2.get(i, 0.0)
                    if base_scores[i] > 0 and traj_div_scores[i] > 0:
                        traj_div_log.append(traj_div_scores[i])
                        div_dbg[i] = f"({_tag} {traj_div_scores[i]:.3f})"
                    elif base_scores[i] <= 0:
                        div_dbg[i] = "(wrong)"
                    else:
                        div_dbg[i] = f"({_tag} 0)"
                # DIAGNOSTIC (tests the "info set is unique" hypothesis): WITHIN each
                # >=2-correct group, how much does phi-share spread across the CORRECT
                # rollouts?  ~0 => correct rollouts converge on the same (unique) info
                # set => no info-diversity to reward => v2 ≈ vanilla GRPO. This is the
                # decisive number, NOT the pooled std (which conflates across-group).
                _disp = []   # (within-group std, range, CV) over correct rollouts
                for ph, idxs in strata.items():
                    cs = [v2.get(i, 0.0) for i in idxs if base_scores[i] > 0]
                    if len(cs) >= 2:
                        a = np.array(cs)
                        cv = (a.std() / a.mean()) if a.mean() > 0 else 0.0
                        _disp.append((a.std(), a.max() - a.min(), cv))
                if _disp:
                    d = np.array(_disp)
                    self._v2_within_disp = (float(d[:, 0].mean()), float(d[:, 1].mean()),
                                            float(d[:, 2].mean()), len(_disp))
                    print(f"  [{_tag}] WITHIN-group div-share over CORRECT rollouts: "
                          f"mean_std={d[:,0].mean():.4f} mean_range={d[:,1].mean():.4f} "
                          f"mean_CV={d[:,2].mean():.4f}  over {len(_disp)} groups  "
                          f"(≈0 => correct rollouts converge on the unique info set)")
            else:
                # --- Doc soft-novelty diversity over CORRECT rollouts only ---
                # Replaces the old hard globally-unique ratio
                #     R_div_i = |docs only i has| / |all distinct docs in group|
                # which was near-zero for ~half of correct rollouts (a doc shared
                # with ANY sibling — even a wrong one — gave zero credit), so the
                # diversity signal was inert. Two fixes here:
                #   (1) soft-novelty: each doc gets partial credit by inverse
                #       frequency instead of all-or-nothing uniqueness.
                #   (2) the frequency is counted over CORRECT rollouts only — since
                #       reward is accuracy-gated, the diversity reference must also
                #       be the correct rollouts (a correct rollout must not be
                #       penalized for sharing a doc with a *wrong* sibling).
                #     R_div_i = (1/|D_i|) · Σ_{d∈D_i} 1 / (#correct rollouts holding d)
                # → dense, bounded in [1/N_correct, 1]; rare-among-solutions docs
                #   score ~1, ubiquitous ones ~1/N_correct. A group with <2 correct
                #   rollouts has no solution-diversity to measure → R_div = 0.
                traj_docs = [
                    set().union(*per_step_docs_all[i]) if per_step_docs_all[i] else set()
                    for i in range(bs)
                ]
                # ΔP useful-weights (over-search suppression): each distinct doc d
                # in rollout i is weighted by the max useful-weight w_k=max(0,ΔP_k)
                # over the search steps that fetched it. R_div becomes a WEIGHTED
                # mean of 1/count(d) — docs from over-search steps (w≈0) earn ~0
                # novelty, so re-fetching known info can't farm diversity. Scale of
                # w cancels (num/den), so R_div stays in the same range (λ unchanged).
                doc_w_by_traj = {}
                if self.useful_weight:
                    correct_all = [i for i in range(bs) if base_scores[i] > 0]
                    useful_w = self._compute_useful_weights(
                        correct_all, em_seqs, golds_all, per_step_docs_all)
                    for i in correct_all:
                        wlist = useful_w.get(i, [])
                        dwt = {}
                        for k, docs_k in enumerate(per_step_docs_all[i]):
                            wk = wlist[k] if k < len(wlist) else 1.0
                            for d in docs_k:
                                if wk > dwt.get(d, -1.0):
                                    dwt[d] = wk
                        doc_w_by_traj[i] = dwt
                for ph, indices in strata.items():
                    correct_idx = [i for i in indices if base_scores[i] > 0]
                    if len(correct_idx) < 2:
                        # nothing to diversify against; rely on R_acc alone
                        for i in indices:
                            traj_div_scores[i] = 0.0
                            div_dbg[i] = "(<2 correct)"
                        continue
                    doc_count = defaultdict(int)
                    for j in correct_idx:
                        for d in traj_docs[j]:
                            doc_count[d] += 1
                    for i in indices:
                        docs_i = traj_docs[i]
                        if base_scores[i] <= 0:
                            traj_div_scores[i] = 0.0      # gated out downstream
                            div_dbg[i] = "(wrong)"
                        elif not docs_i:
                            traj_div_scores[i] = 0.0
                            div_dbg[i] = "(no docs)"
                            traj_div_log.append(0.0)
                        elif self.useful_weight:
                            dwt = doc_w_by_traj.get(i, {})
                            num = sum(dwt.get(d, 1.0) * (1.0 / doc_count[d]) for d in docs_i)
                            den = sum(dwt.get(d, 1.0) for d in docs_i)
                            r_div = (num / den) if den > 0 else 0.0
                            traj_div_scores[i] = r_div
                            wbar = den / len(docs_i)
                            div_dbg[i] = f"(useful-nov {r_div:.2f}/{len(docs_i)}d w̄={wbar:.2f})"
                            traj_div_log.append(r_div)
                        else:
                            r_div = sum(1.0 / doc_count[d] for d in docs_i) / len(docs_i)
                            traj_div_scores[i] = r_div
                            div_dbg[i] = f"(soft-nov {r_div:.2f}/{len(docs_i)}d corr-pool)"
                            traj_div_log.append(r_div)

            # --- Place R_total = _acc_fmt_reward(R_acc, R_fmt) + 1[acc]·λ·R_div ---
            for i in range(bs):
                r_acc = base_scores[i]              # EM 0/1, also gates the diversity term
                r_fmt = fmt_scores[i]               # format 0/1
                r_base = self._acc_fmt_reward(r_acc, r_fmt)
                r_div = traj_div_scores[i]
                if self.diversity_type == 'budget':
                    # asymmetric gate already applied in _budget_soft_diversity (pool
                    # selection): all-wrong groups ungate R_acc so wrong rollouts may
                    # carry R_div. Do NOT re-gate on 1[correct] here — that would zero
                    # exactly the ungated wrong-pool payments. r_div is already 0 for
                    # any rollout outside its group's pool.
                    div_add = lam * r_div
                elif self.diversity_type == 'gold_cov_all':
                    # gold_cov_all: R_div is a share over ALL rollouts of the group, so
                    # the 1[correct] gate is lifted here too — otherwise the wrong
                    # rollouts' shares would be computed and then thrown away, and Σ_i
                    # (paid R_div) would no longer be 1. r_div is already 0 for any
                    # rollout holding no gold-bearing doc.
                    div_add = lam * r_div
                else:
                    div_add = lam * r_div if r_acc > 0 else 0.0   # accuracy-gated
                r_total = r_base + div_add
                valid_len = valid_response_lengths[i]
                if valid_len > 0:
                    outcome_tensor[i, valid_len - 1] = r_total
                if div_add > 0:
                    applied_count += 1
                # Occasional full-trajectory dump (1/64), replacing the qa_em debug
                # print this path bypasses. Shows the reward breakdown.
                if self.num_examine == 0 and random.randint(1, 64) == 1:
                    print("-------------------------------- [em_fmt_doc sample]")
                    print(f"R_acc={r_acc:.0f} R_fmt={r_fmt:.0f} R_base={r_base:.2f} "
                          f"R_div={r_div:.3f} {div_dbg[i]} "
                          f"| gate={'on' if div_add > 0 else 'off'} lambda={lam:.3g} "
                          f"=> R_total={r_total:.3f}")
                    print(f"Solution string: {em_seqs[i]}")
            # Per-rollout correctness (R_acc>0) for the all-wrong-group advantage
            # scaling (adv_estimator=grpo + allwrong_adv_scale): downstream
            # compute_advantage reads this to find all-wrong groups and the batch
            # accuracy c, then scales Â_i ← c·Â_i on those groups only.
            data.meta_info['em_acc'] = [1.0 if base_scores[i] > 0 else 0.0
                                        for i in range(bs)]
        elif self.combine_mode == 'step_reward':
            # ===== Per-block COMBINED reward + per-block-position normalization =====
            # search block k:  R_total = R_acc + λ·1[R_acc=1]·div_k
            # answer block:     R_total = R_acc            (no search → no diversity)
            # Each block's reward is written into step_level_scores and tagged with
            # a stratum id = (prompt,'search',k) or (prompt,'answer'). Downstream
            # adv_estimator=grpo_step_combined then GRPO-normalizes the k-th search
            # blocks of all rollouts together, and the answer blocks together — ONE
            # normalization per position. NO separate outcome advantage; λ is baked
            # in here (cur_lambda_div). div_k compares the k-th query of correct
            # rollouts reaching step k (N = that count; N<2 → div_k=0).
            lam = getattr(self, 'cur_lambda_div', self.lambda_div)
            if lam != lam:  # NaN guard
                lam = self.lambda_div
            use_nsearch = (self.diversity_grouping == 'step_prompt_nsearch')

            # outcome_tensor: raw EM at last token (metric logging only; the
            # grpo_step_combined estimator ignores token_level_rewards entirely).
            for i in range(bs):
                valid_len = valid_response_lengths[i]
                if valid_len > 0:
                    outcome_tensor[i, valid_len - 1] = base_scores[i]

            # Per-step diversity among correct rollouts. The ONLY thing that
            # differs between the two arms is HOW div_k is measured — everything
            # downstream (per-block reward, per-position normalization, λ, gating)
            # is identical, so the two are a fair head-to-head:
            #   search_embedding / reasoning_embedding → e5 cosine distance of the
            #       k-th query among correct rollouts.
            #   doc_set_unique → fraction of step-k docs unique vs siblings.
            # Enable toggle = self.lambda_div > 0 (val_reward_fn has lambda_div=0 →
            # div all 0, no encoder load).
            step_scores = {}
            dstats = None
            if self.lambda_div > 0:
                if self.diversity_type in ('reasoning_embedding', 'search_embedding'):
                    step_scores, dstats = self._stepwise_query_diversity(
                        bs, base_scores, prompt_hashes, think_per_step_all, use_nsearch)
                elif self.diversity_type == 'doc_set_unique':
                    step_scores, dstats = self._stepwise_doc_diversity(
                        bs, base_scores, prompt_hashes, per_step_docs_all, use_nsearch)
            if dstats is not None:
                cos_sim_log.extend(dstats['cos'])
                nn_sim_log.extend(dstats['nn'])
                stratum_size_log.extend(dstats['sizes'])
                for k, lst in dstats['by_step'].items():
                    step_score_by_step[k].extend(lst)
                for k, lst in dstats['gsize_by_step'].items():
                    step_group_size_by_step[k].extend(lst)

            # Place per-block reward + per-position stratum id.
            for i in range(bs):
                base_r = base_scores[i]
                ph = prompt_hashes[i]
                placed_any = False
                for k, rng in enumerate(step_ranges_all[i]):
                    if rng is None:
                        continue
                    s, e = rng
                    if e <= s:
                        continue
                    div_k = step_scores.get(i, {}).get(k, 0.0)
                    block_r = base_r + (lam * div_k if base_r > 0 else 0.0)
                    key = (ph, 'search', k)
                    if key not in group_id_map:
                        group_id_map[key] = next_gid
                        next_gid += 1
                    step_level_tensor[i, s:e] = block_r
                    stratum_step_id_tensor[i, s:e] = group_id_map[key]
                    if div_k > 0:
                        traj_div_log.append(div_k)
                        placed_any = True
                ar = answer_ranges_all[i]
                if ar is not None:
                    s, e = ar
                    if e > s:
                        key = (ph, 'answer')
                        if key not in group_id_map:
                            group_id_map[key] = next_gid
                            next_gid += 1
                        step_level_tensor[i, s:e] = base_r
                        stratum_step_id_tensor[i, s:e] = group_id_map[key]
                if base_r > 0 and not placed_any:
                    skipped_singleton_count += 1
                if placed_any:
                    applied_count += 1
        elif self.diversity_grouping in ('step_prompt', 'step_prompt_nsearch') \
                and self.combine_mode != 'pre_norm':
            # Step-level diversity (advantage-level / post_norm path).
            # NOTE: when combine_mode=='pre_norm', step-wise diversity is instead
            # aggregated to a per-trajectory scalar in the computation phase above
            # and combined at the REWARD level by the pre_norm branch below — so we
            # deliberately fall through to it rather than placing step_level_tensor.
            #   step_prompt:         stratum = (prompt, step_k).
            #   step_prompt_nsearch: stratum = (prompt, n_search, step_k) — exp_002
            #     style. Trajectories are additionally split by their TOTAL search
            #     count, so over-searching moves a trajectory to a different cohort
            #     rather than simply accumulating more diversity reward. This removes
            #     the over-search / junk-query incentive that pure (prompt, step_k)
            #     creates. n_agent should be large (e.g. 8) since the extra
            #     dimension fragments groups (many singletons get skipped).
            # Accuracy-gated (only correct trajectories enter comparison).
            # Min group size = 2. combine_mode ignored — uses post_norm-style
            # step_level_tensor (requires adv_estimator=grpo_step_level downstream).
            use_nsearch = (self.diversity_grouping == 'step_prompt_nsearch')

            for i in range(bs):
                valid_len = valid_response_lengths[i]
                if valid_len > 0:
                    outcome_tensor[i, valid_len - 1] = base_scores[i]

            if self.diversity_type == 'doc_set_unique' and self.lambda_div > 0:
                # group key -> list of (traj_idx, docs_at_step_k)
                # accuracy gating: only correct trajectories enter group.
                # key = (ph, n_search, step_k) if nsearch else (ph, step_k)
                step_groups = defaultdict(list)
                for i in range(bs):
                    if base_scores[i] <= 0:
                        continue
                    ph = prompt_hashes[i]
                    n_search_i = len(per_step_docs_all[i])
                    for k, docs_k in enumerate(per_step_docs_all[i]):
                        key = (ph, n_search_i, k) if use_nsearch else (ph, k)
                        step_groups[key].append((i, docs_k))

                # Score each group.
                # score_i_k = |D_{i,k} \ ∪_{j≠i in same group} D_{j,k}| / |D_{i,k}|
                step_scores = defaultdict(dict)  # step_scores[traj_i][step_k] = float
                for key, entries in step_groups.items():
                    k = key[-1]  # step_k is always the last element
                    stratum_size_log.append(len(entries))
                    step_group_size_by_step[k].append(len(entries))
                    if len(entries) < 2:
                        continue
                    for local_i, (i, docs_i) in enumerate(entries):
                        if len(docs_i) == 0:
                            score = 0.0
                        else:
                            other_docs = set()
                            for local_j, (j, docs_j) in enumerate(entries):
                                if local_i != local_j:
                                    other_docs |= docs_j
                            unique = docs_i - other_docs
                            score = len(unique) / len(docs_i)
                        step_scores[i][k] = score
                        traj_div_log.append(score)
                        step_score_by_step[k].append(score)

                # Placement: each scored step's chunk gets the score and its
                # group id (matching the stratum key incl. n_search if used).
                for i in range(bs):
                    if i not in step_scores:
                        if base_scores[i] > 0 and \
                                correct_count_per_prompt[prompt_hashes[i]] < 2:
                            skipped_singleton_count += 1
                        continue
                    ph = prompt_hashes[i]
                    n_search_i = len(per_step_docs_all[i])
                    placed_any = False
                    for k, score in step_scores[i].items():
                        if k >= len(step_ranges_all[i]):
                            continue
                        rng = step_ranges_all[i][k]
                        if rng is None:
                            continue
                        s, e = rng
                        if e <= s:
                            continue
                        key = (ph, n_search_i, k) if use_nsearch else (ph, k)
                        if key not in group_id_map:
                            group_id_map[key] = next_gid
                            next_gid += 1
                        gid = group_id_map[key]
                        step_level_tensor[i, s:e] = score
                        stratum_step_id_tensor[i, s:e] = gid
                        placed_any = True
                    if placed_any:
                        applied_count += 1

            elif self.diversity_type in ('reasoning_embedding', 'search_embedding') \
                    and self.lambda_div > 0:
                # Step-level embedding diversity (search query / reasoning).
                #   stratum = (prompt, step_k)                for step_prompt
                #           = (prompt, n_search, step_k)      for step_prompt_nsearch
                #   Accuracy-gated: only correct rollouts enter a group.
                #   For each stratum with >= 2 valid (non-empty) embeddings, the
                #   k-th query of rollout i is compared against the k-th query of
                #   its siblings (slide-18 search-query diversity at step k):
                #       score_i = (1/(N-1)) * sum_{j!=i} (1 - cos(e_i^(k), e_j^(k)))
                #   This is the position-wise comparison ("first queries vs first
                #   queries, second vs second"), NOT the trajectory mean-pool.
                # think_per_step_all[i] holds per-step text: the <search> query for
                # 'search_embedding', the <think> block for 'reasoning_embedding'.

                # 1) Embed every per-step text of every correct trajectory (cached).
                flat_texts = []
                flat_ref = []  # (traj_i, step_k) parallel to flat_texts
                for i in range(bs):
                    if base_scores[i] <= 0:
                        continue
                    for k, t in enumerate(think_per_step_all[i]):
                        if t and t.strip():
                            flat_texts.append(t)
                            flat_ref.append((i, k))
                flat_embs = self._embed_texts(flat_texts) if flat_texts else []
                emb_by_traj_step = {}  # (traj_i, step_k) -> tensor (d,)
                for (i, k), e in zip(flat_ref, flat_embs):
                    if e is not None:
                        emb_by_traj_step[(i, k)] = e

                # 2) Group correct trajectories' steps into strata.
                step_groups = defaultdict(list)  # key -> [(traj_i, step_k), ...]
                for i in range(bs):
                    if base_scores[i] <= 0:
                        continue
                    ph = prompt_hashes[i]
                    n_search_i = len(think_per_step_all[i])
                    for k in range(n_search_i):
                        key = (ph, n_search_i, k) if use_nsearch else (ph, k)
                        step_groups[key].append((i, k))

                # 3) Score each stratum (pairwise-mean cosine distance). Reuses the
                #    exact scorer as the trajectory arm — missing-embedding steps
                #    score 0 and fall below the stratum mean (implicit format
                #    penalty), identical semantics to doc-set / reasoning arms.
                step_scores = defaultdict(dict)  # step_scores[traj_i][step_k] = float
                for key, entries in step_groups.items():
                    k = key[-1]
                    stratum_size_log.append(len(entries))
                    step_group_size_by_step[k].append(len(entries))
                    if len(entries) < 2:
                        continue
                    embs = [emb_by_traj_step.get((i, kk)) for (i, kk) in entries]
                    scores, stats = self._compute_emb_diversity_per_traj(embs)
                    cos_sim_log.extend(stats['cos_sims'])
                    nn_sim_log.extend(stats['nn_sims'])
                    for local_i, (i, kk) in enumerate(entries):
                        step_scores[i][kk] = scores[local_i]
                        traj_div_log.append(scores[local_i])
                        step_score_by_step[kk].append(scores[local_i])

                # 4) Placement: each scored step's think+search chunk gets the score
                #    and its stratum's group id (matching the key incl. n_search if
                #    used). lambda_div multiplier applied after group norm downstream.
                for i in range(bs):
                    if i not in step_scores:
                        if base_scores[i] > 0 and \
                                correct_count_per_prompt[prompt_hashes[i]] < 2:
                            skipped_singleton_count += 1
                        continue
                    ph = prompt_hashes[i]
                    n_search_i = len(think_per_step_all[i])
                    placed_any = False
                    for k, score in step_scores[i].items():
                        if k >= len(step_ranges_all[i]):
                            continue
                        rng = step_ranges_all[i][k]
                        if rng is None:
                            continue
                        s, e = rng
                        if e <= s:
                            continue
                        key = (ph, n_search_i, k) if use_nsearch else (ph, k)
                        if key not in group_id_map:
                            group_id_map[key] = next_gid
                            next_gid += 1
                        gid = group_id_map[key]
                        step_level_tensor[i, s:e] = score
                        stratum_step_id_tensor[i, s:e] = gid
                        placed_any = True
                    if placed_any:
                        applied_count += 1
        elif self.combine_mode == 'pre_norm':
            # (2a) Pre-normalization combination: r_i = base_r_i + traj_div_score_i.
            # - Wrong trajectory: base_r=0, traj_div=0 → r=0
            # - Singleton-correct: base_r=1, traj_div=0 (skipped in scoring) → r=1
            # - Multi-correct: base_r=1, traj_div=unique_frac ∈ [0,1] → r ∈ [1,2]
            # Single GRPO normalization happens downstream in compute_advantage
            # (use algorithm.adv_estimator=grpo). step_level_tensor and
            # stratum_step_id_tensor are left as zeros (unused in this mode).
            # λ for pre_norm: set by the trainer each step (entropy_binary etc.).
            # Default 1.0 if unset (constant 1:1 combination).
            lam = getattr(self, 'cur_lambda_div', 1.0)
            if lam != lam:  # NaN guard (e.g., per_group_constraint placeholder)
                lam = 1.0
            for i in range(bs):
                valid_len = valid_response_lengths[i]
                base_r = base_scores[i]
                # traj_div_scores[i] is 0 unless i is correct AND in a non-
                # singleton prompt (handled inside trajectory-diversity scoring).
                # R_total = R_answer + λ·R_diversity (single GRPO norm downstream).
                div_add = lam * traj_div_scores[i] if base_r > 0 else 0.0
                combined_r = base_r + div_add
                if valid_len > 0:
                    outcome_tensor[i, valid_len - 1] = combined_r
                if base_r > 0 and div_add > 0:
                    applied_count += 1
                elif base_r > 0 and correct_count_per_prompt[prompt_hashes[i]] < 2:
                    skipped_singleton_count += 1
        else:
            # 'post_norm' (default): outcome and diversity normalized separately,
            # then combined as outcome_adv + lambda_div * div_adv downstream.
            for i in range(bs):
                valid_len = valid_response_lengths[i]
                base_r = base_scores[i]
                if valid_len > 0:
                    outcome_tensor[i, valid_len - 1] = base_r

                if base_r <= 0 or self.lambda_div <= 0 or self.diversity_type == 'none':
                    continue
                if valid_len <= 0:
                    continue
                # Skip prompts where this is the only correct trajectory.
                if correct_count_per_prompt[prompt_hashes[i]] < 2:
                    skipped_singleton_count += 1
                    continue

                ph = prompt_hashes[i]
                if ph not in group_id_map:
                    group_id_map[ph] = next_gid
                    next_gid += 1
                gid = group_id_map[ph]

                div_score = traj_div_scores[i]
                # Place diversity score on the model-generated think+search chunks
                # ONLY (the first n_info chunks of model-gen tokens). The final
                # think+answer chunk is intentionally EXCLUDED so the answer-region
                # tokens (and the </think>→<answer>/<search> transition tokens)
                # receive ONLY the clean outcome advantage. Without this exclusion,
                # the diversity advantage bleeds into the answer-decision tokens,
                # noising their gradient — empirically caused the argmax after
                # </think> to flip to EOS (model-refuses-to-answer failure).
                # Storing raw div_score; lambda multiplier is applied AFTER group
                # normalization in compute_advantage so it actually controls signal
                # strength (otherwise GRPO normalization cancels it out).
                placed_any = False
                for rng in step_ranges_all[i]:
                    if rng is None:
                        continue
                    s, e = rng
                    if e > s:
                        step_level_tensor[i, s:e] = div_score
                        stratum_step_id_tensor[i, s:e] = gid
                        placed_any = True
                if placed_any:
                    applied_count += 1

        # Store extra tensors in data.batch (will be picked up in compute_advantage)
        data.batch['step_level_scores'] = step_level_tensor
        data.batch['stratum_step_id'] = stratum_step_id_tensor

        # ---- Stats ----
        base_arr = np.array(base_scores)
        outcome_sum = outcome_tensor.sum().item()
        step_sum = step_level_tensor.sum().item()
        group_sizes = list(Counter(prompt_hashes).values())

        # Expose the reward decomposition to the trainer → wandb (em_fmt_doc only).
        # The trainer merges data.meta_info['reward_components'] into its metrics.
        if self.combine_mode in ('em_fmt_doc', 'passk', 'batch_outcome'):
            fmt_arr = np.array(fmt_scores)
            corr = base_arr > 0
            rc = {
                'reward/r_acc': float(base_arr.mean()),               # = accuracy (EM)
                'reward/r_fmt': float(fmt_arr.mean()),                # format pass rate
            }
            # Soft-novelty (the Proposed value) measured for EVERY arm — independent of
            # the active reward — so it is comparable head-to-head in wandb (incl. the
            # passk and rgsn runs, where traj_div_scores is NOT soft-novelty / absent).
            sn_arr = np.array(self._soft_novelty_measure(
                strata, per_step_docs_all, base_scores, bs))
            rc['reward/soft_nov_all'] = float(sn_arr.mean())
            rc['reward/soft_nov_correct'] = float(sn_arr[corr].mean()) if corr.any() else 0.0
            # div-watch: search count / budget utilisation / within-group doc overlap.
            # Reward-agnostic (needs only strata + retrieved docs), so logged for EVERY
            # arm — same rationale as soft-novelty above. Keeps search behaviour
            # comparable head-to-head across em_fmt_doc / batch_outcome / passk runs.
            _dw = self._div_watch(strata, per_step_docs_all, data_sources_all, bs)
            rc['div/searches_B1'] = _dw['b_mean'].get(1, float('nan'))
            rc['div/searches_B2'] = _dw['b_mean'].get(2, float('nan'))
            rc['div/group_jaccard'] = _dw['jaccard']
            rc['div/group_unique_docs'] = _dw['uniq']
            rc['div/budget_full_B1'] = _dw['b_full'].get(1, float('nan'))
            rc['div/budget_full_B2'] = _dw['b_full'].get(2, float('nan'))
            # gold-spread: DISTINCT gold-bearing docs a group collectively retrieves.
            # Needs the per-rollout gold sets, which are only parsed when a gold-aware
            # arm asked for them (or +div_gold_watch=true forced the extra pass).
            if (self.budget_ungate_gold or self.div_gold_only or self.div_gold_watch
                    or self.diversity_type in ('gold_cov', 'gold_cov_all')):
                _gw = self._gold_watch(strata, gold_bearing_docs_all, per_step_docs_all)
                rc['div/gold_uniq'] = _gw['uniq']
                rc['div/gold_uniq_hit'] = _gw['uniq_hit']
                rc['div/gold_per_traj'] = _gw['per_traj']
                rc['div/gold_jaccard'] = _gw['jaccard']
                rc['div/gold_traj_frac'] = _gw['traj_frac']
                rc['div/gold_grp_frac'] = _gw['grp_frac']
            # diff@N: mean number of DISTINCT answers per prompt group (the quantity
            # Batch exploration optimizes). Also reward-agnostic -> every arm.
            _distinct = []
            for ph, idxs in strata.items():
                answs = [outcome_answers[i] for i in idxs if outcome_answers[i] is not None]
                _distinct.append(len(set(answs)))
            rc['reward/distinct_answers_per_group'] = float(np.mean(_distinct)) if _distinct else 0.0
            if self.combine_mode == 'em_fmt_doc':
                div_arr = np.array(traj_div_scores)
                rc['reward/r_div_all'] = float(div_arr.mean())
                rc['reward/r_div_correct'] = float(div_arr[corr].mean()) if corr.any() else 0.0
                # asymmetric-ungate diagnostic: R_div paid to WRONG rollouts (all-wrong
                # groups). 0 for the plain budget gate; >0 tracks ungate activity.
                rc['reward/r_div_wrong'] = float(div_arr[~corr].mean()) if (~corr).any() else 0.0
                rc['reward/r_total'] = float(outcome_sum / bs) if bs else 0.0
                # gold-share canary (budget_ungate_gold): fraction of gold-anchored
                # all-wrong POOL docs that bear gold. ↓ while r_div_wrong ↑ = junk-padding.
                _gs = getattr(self, '_budget_gold_share', None)
                if self.budget_ungate_gold and _gs is not None and _gs['den'] > 0:
                    rc['reward/gold_share'] = float(_gs['share'])
                _rd = getattr(self, '_div_rdiv_dist', None)
                if _rd is not None:
                    rc['div/rdiv_mean'] = _rd['mean']
                    rc['div/rdiv_p90'] = _rd['p90']
                    rc['div/rdiv_max'] = _rd['max']
                    rc['div/rdiv_near1'] = _rd['near1']
                _gt = getattr(self, '_div_gate', None)
                if _gt is not None:
                    rc['div/allcorrect_gate_rate'] = _gt['rate']
                _pg = getattr(self, '_div_penalty_gap', None)
                if _pg is not None and _pg['n_gold'] and _pg['n_nogold']:
                    rc['div/rdiv_gold'] = _pg['gold_mean']
                    rc['div/rdiv_nogold'] = _pg['nogold_mean']
                    rc['div/penalty_gap'] = _pg['gap']
                    rc['div/docs_gold'] = _pg['docs_gold']
                    rc['div/docs_nogold'] = _pg['docs_nogold']
                    rc['div/search_gold'] = _pg['search_gold']
                    rc['div/search_nogold'] = _pg['search_nogold']
            # passk merges reward_components AND passk_acc/passk_fmt — keep both.
            data.meta_info['reward_components'] = rc

        ss_min = min(stratum_size_log) if stratum_size_log else 0
        ss_max = max(stratum_size_log) if stratum_size_log else 0
        ss_mean = float(np.mean(stratum_size_log)) if stratum_size_log else 0.0

        # Header — show all active mode flags
        _is_v2 = self.diversity_type in ('v2_shapley', 'v1_filtered')
        _is_md = self.diversity_type == 'meaningful'
        if _is_v2 or _is_md:
            # v2/v1/md: prompt-level strata + constant lambda. No grouping / schedule knobs.
            print(f"=== Step {self._call_count} reward stats "
                  f"[type={self.diversity_type}, combine={self.combine_mode}, "
                  f"lambda={self.lambda_div:g}] ===")
        else:
            print(f"=== Step {self._call_count} reward stats ===")

        print(f"  Trajectories: {bs}, Unique prompts: {len(group_sizes)}")
        if self.diversity_grouping == 'step_prompt':
            print(f"  Stratum (prompt,step_k) — count:{len(stratum_size_log)} sizes min:{ss_min} max:{ss_max} mean:{ss_mean:.2f}")
        elif self.diversity_grouping == 'step_prompt_nsearch':
            print(f"  Stratum (prompt,n_search,step_k) — count:{len(stratum_size_log)} sizes min:{ss_min} max:{ss_max} mean:{ss_mean:.2f}")

        # Search-count distribution + mean
        sc_dist = Counter(len(d) for d in per_step_docs_all)
        sc_mean = sum(k * v for k, v in sc_dist.items()) / bs if bs else 0
        print(f"  Search-count distribution: {dict(sorted(sc_dist.items()))} | mean: {sc_mean:.2f}")

        print(f"  Accuracy: {(base_arr > 0).sum()}/{bs} = {(base_arr > 0).mean():.4f}")

        # ---------------- evidence-exempt 설계 검증 로깅 (데이터셋 태그) -------------
        if self.diversity_type in ('evidence_exempt', 'evidence_doc_exempt',
                                   'gq_doc_exempt') \
                and self.combine_mode == 'em_fmt_doc':
            _st = getattr(self, '_evd_exempt_stats', None)
            by_ds = defaultdict(lambda: {'n': 0, 'evd_corr': 0, 'n_corr': 0,
                                         'evd_wrong': 0, 'n_wrong': 0, 'srch': []})
            for i in range(bs):
                ds = str(data_sources_all[i])
                b = by_ds[ds]
                b['n'] += 1
                b['srch'].append(len(per_step_docs_all[i]))
                _has = bool(set().union(*per_step_docs_all[i]) & evd_eligible_all[i]) \
                    if per_step_docs_all[i] else False
                if base_scores[i] > 0:
                    b['n_corr'] += 1; b['evd_corr'] += int(_has)
                else:
                    b['n_wrong'] += 1; b['evd_wrong'] += int(_has)
            # 그룹 단위: all-correct 비율 / Jaccard / evidence 보유자 advantage 중앙값
            g_by_ds = defaultdict(lambda: {'n': 0, 'ac': 0, 'jac': [], 'rdiv_evd': []})
            for ph, idxs in strata.items():
                ds = str(data_sources_all[idxs[0]])
                gb = g_by_ds[ds]
                gb['n'] += 1
                if all(base_scores[i] > 0 for i in idxs):
                    gb['ac'] += 1
                    for i in idxs:
                        docs = set().union(*per_step_docs_all[i]) if per_step_docs_all[i] else set()
                        if docs & evd_eligible_all[i]:
                            gb['rdiv_evd'].append(traj_div_scores[i])
                sets = [set().union(*per_step_docs_all[i]) if per_step_docs_all[i] else set()
                        for i in idxs]
                nz = [s for s in sets if s]
                if len(nz) >= 2:
                    tot = cnt2 = 0
                    for a_ in range(len(nz)):
                        for b_ in range(a_ + 1, len(nz)):
                            u = len(nz[a_] | nz[b_])
                            if u:
                                tot += len(nz[a_] & nz[b_]) / u; cnt2 += 1
                    if cnt2:
                        gb['jac'].append(tot / cnt2)
            _mode = ('G_q' if self.diversity_type == 'gq_doc_exempt'
                     else 'doc(B\')' if self.diversity_type == 'evidence_doc_exempt'
                     else 'call')
            print(f"  [evidence-exempt/{_mode}]")
            for ds in sorted(by_ds):
                b, gb = by_ds[ds], g_by_ds[ds]
                sh = Counter(b['srch'])
                jac = float(np.mean(gb['jac'])) if gb['jac'] else float('nan')
                med = float(np.median(gb['rdiv_evd'])) if gb['rdiv_evd'] else float('nan')
                print(f"    [{ds}] evidence 보유율 정답 "
                      f"{b['evd_corr']}/{b['n_corr']}="
                      f"{b['evd_corr']/max(b['n_corr'],1)*100:.1f}% | 오답 "
                      f"{b['evd_wrong']}/{b['n_wrong']}="
                      f"{b['evd_wrong']/max(b['n_wrong'],1)*100:.1f}%")
                print(f"    [{ds}] all-correct 그룹 {gb['ac']}/{gb['n']}="
                      f"{gb['ac']/max(gb['n'],1)*100:.1f}% | 그룹 Jaccard {jac:.4f} | "
                      f"검색횟수 분포 {dict(sorted(sh.items()))}")
                print(f"    [{ds}] ★ all-correct·evidence 보유 rollout 의 R_div 중앙값 "
                      f"{med:.4f}  (≥0 이어야 정상, n={len(gb['rdiv_evd'])})")
            if _st:
                print(f"    guesser {_st['n_guesser']}/{_st['n_E']} "
                      f"({_st['n_guesser']/max(_st['n_E'],1)*100:.1f}%) | "
                      f"E'={_st['n_Ep']} | evidence 미검출 환원 그룹 "
                      f"{_st['n_fallback']}/{_st['n_groups']} | zero-contrast "
                      f"{_st['n_zero_contrast']}/{_st['n_groups']}")
                # containment 로만 자격을 얻은(exempt call 0개) E' 멤버 비율.
                # replay 예상 ~2.6%p 근처에서 크게 벗어나면 detection 이상 신호.
                for ds in sorted(_st['Ep_by_ds']):
                    _n, _e = _st['Ep_by_ds'][ds], _st['elig_only'][ds]
                    print(f"    [{ds}] containment-only 자격(E' 중 exempt call 0개) "
                          f"{_e}/{_n} = {_e/max(_n,1)*100:.1f}%")

        # ---------------- div-watch: the five live vitals ----------------
        if self.combine_mode == 'em_fmt_doc':
            _dw = self._div_watch(strata, per_step_docs_all, data_sources_all, bs)
            _gt = getattr(self, '_div_gate', None)
            _pg = getattr(self, '_div_penalty_gap', None)
            print("  [div-watch]")
            if _gt is not None:
                print(f"    (1) all-correct gate : {_gt['n_allcorrect']}/{_gt['n_groups']} groups "
                      f"= {_gt['rate']:.3f}  (R_div OFF gate {'ON' if _gt['active'] else 'OFF'})"
                      f"   ← virtual-0 표준화 대상 그룹 비율. 정확도 오르면 같이 올라야 정상")
            if _pg is not None and _pg['n_gold'] and _pg['n_nogold']:
                print(f"    (2) penalty gap      : R_div gold={_pg['gold_mean']:.4f} "
                      f"(n={_pg['n_gold']}) vs no-gold={_pg['nogold_mean']:.4f} "
                      f"(n={_pg['n_nogold']}) | gap={_pg['gap']:+.4f}"
                      f"   ← 즉시 0이 아니어도 정상, 평균형 대비 축소 + 추세를 봄")
                print(f"    (2b) |D_i| / n_search : gold |D|={_pg['docs_gold']:.2f} "
                      f"search={_pg['search_gold']:.2f}  vs  no-gold |D|={_pg['docs_nogold']:.2f} "
                      f"search={_pg['search_nogold']:.2f}"
                      f"   ← gold 쪽 search가 낮으면 '일찍 찾고 멈춤' 가설 참")
            if (self.budget_ungate_gold or self.div_gold_only or self.div_gold_watch
                    or self.diversity_type in ('gold_cov', 'gold_cov_all')):
                _gw = self._gold_watch(strata, gold_bearing_docs_all, per_step_docs_all)
                print(f"    (2c) gold spread     : uniq/group={_gw['uniq']:.2f} "
                      f"(gold 보유 그룹만 {_gw['uniq_hit']:.2f}) vs per-traj={_gw['per_traj']:.2f} "
                      f"| gold jaccard={_gw['jaccard']:.3f} (n={_gw['n_jac_groups']})"
                      f"   ← uniq_hit > per-traj 면 서로 다른 gold 문서를 물어오는 중")
                print(f"    (2d) gold 보유율    : traj {_gw['traj_frac']*100:.1f}% "
                      f"| group {_gw['grp_frac']*100:.1f}% "
                      f"(groups={_gw['n_groups']}, searched traj={_gw['n_traj']})")
            _rd = getattr(self, '_div_rdiv_dist', None)
            if _rd is not None:
                print(f"    (6) R_div dist (paid) : mean={_rd['mean']:.4f} std={_rd['std']:.4f} "
                      f"p50={_rd['p50']:.4f} p90={_rd['p90']:.4f} max={_rd['max']:.4f} "
                      f"| ≥0.9: {_rd['near1']*100:.1f}% (n={_rd['n']})")
            _bs_ = " | ".join(f"B={b}: {_dw['b_mean'][b]:.3f} (n={_dw['b_n'][b]})"
                              for b in sorted(_dw['b_mean']))
            print(f"    (3) searches by budget: {_bs_}")
            _bf_ = " | ".join(f"B={b}: {_dw['b_full'][b]*100:.1f}% {_dw['b_hist'][b]}"
                              for b in sorted(_dw['b_full']))
            print(f"    (7) budget utilisation: {_bf_}"
                  f"   ← B=2의 2회 비율 상승 = 예산을 쓰기 시작")
            print(f"    (4) group overlap    : Jaccard={_dw['jaccard']:.4f} "
                  f"(over {_dw['n_jac_groups']} groups) | unique docs/group={_dw['uniq']:.2f}"
                  f"   ← GRPO 수준(≈0.55)으로 회귀하면 다양성 소멸")

        # Outcome / diversity reward sums — label depends on combine_mode
        if self.combine_mode == 'pre_norm':
            # outcome_tensor contains combined_r (outcome + diversity)
            base_sum = sum(base_scores)
            div_sum = outcome_sum - base_sum
            print(f"  Combined reward sum (pre_norm): {outcome_sum:.2f} | "
                  f"= outcome ({base_sum:.2f}) + diversity ({div_sum:.2f})")
        elif _is_md:
            # Meaningful-Diversity: R_div(i) = max(p̂·Σ_d φ_d·(1/m_d), 0), CORRECT only.
            #   v headroom-normalized ⇒ Σφ_d ≤ 1 ⇒ R_div ∈ [0, ~p̂] (pure non-neg bonus).
            ml = getattr(self, '_md_last', None) or {}
            print(f"  R_total sum: {outcome_sum:.2f} (mean: {outcome_sum / bs:.4f})")
            # --- reward signal: R_div over correct rollouts (bounded ≤ p̂) ---
            corr = np.array([traj_div_scores[i] for i in range(bs) if base_scores[i] > 0])
            if len(corr):
                pos = int((corr > 0).sum()); neg = int((corr < 0).sum()); zer = int((corr == 0).sum())
                print(f"  R_div (correct n={len(corr)}): mean={corr.mean():+.4f} std={corr.std():.4f} "
                      f"min={corr.min():+.3f} max={corr.max():+.3f} | rollouts +{pos}/−{neg}/0:{zer}")
            paid = self.lambda_div * float(corr.sum()) if len(corr) else 0.0
            print(f"  Paid diversity (λ·Σ R_div → R_total): {paid:+.2f}  (λ={self.lambda_div:g}) | "
                  f"p̂-gate mean={ml.get('phat_mean', 0.0):.3f} (correctness-first)")
            # --- MEANINGFULNESS: per-doc Shapley φ_d on headroom-normalized v (Σ≤1/rollout) ---
            phi = np.array(ml.get('phi', []))
            if len(phi):
                up = int((phi > 0).sum()); dn = int((phi < 0).sum())
                print(f"  φ_d meaningfulness ({len(phi)} docs, headroom-norm): mean={phi.mean():+.3f} "
                      f"min={phi.min():+.3f} max={phi.max():+.3f} | "
                      f"useful φ>0: {up} ({up/len(phi):.0%}) / misleading φ<0: {dn} ({dn/len(phi):.0%})")
            # --- DIVERSITY: novelty m_d over distinct group-docs (the pass@k driver) ---
            mh = ml.get('m_hist', {})
            if mh:
                tot = sum(mh.values()); uniq = mh.get(1, 0)
                print(f"  doc novelty m_d (distinct docs/group, over correct): {mh} | "
                      f"unique m=1: {uniq}/{tot} = {uniq/tot:.0%}  (↑ = solutions use different evidence)")
            # --- compute monitor: exact 2^n, NO cap ---
            print(f"  Shapley cost: correct_scored={ml.get('n_correct',0)} max_n={ml.get('maxn',0)} "
                  f"coalitions={ml.get('n_ctx',0)} MC-fallback={ml.get('n_mc',0)} | "
                  f"per-rollout n-dist={ml.get('nhist',{})}  (auto: exact while 2^n≤P·n, else MC; P=32)")
        elif _is_v2:
            # v2/v1: Stage 2 (phi) computed for EVERY group over ALL rollouts (전원
            # 계산, no correctness gate); Stage 3 pays only correct via 1[correct].
            print(f"  Outcome reward sum: {outcome_sum:.2f} (mean: {outcome_sum / bs:.4f})")
            # per-group correct-count distribution: how many groups are 0/1/.../k correct.
            # 1-correct groups compute phi but pay only that one; >=2 is where the
            # diversity credit actually differentiates among correct rollouts.
            cc = Counter(sum(1 for i in idxs if base_scores[i] > 0) for idxs in strata.values())
            print(f"  Group correct-count dist: {dict(sorted(cc.items()))} | over {len(strata)} groups")
            raw = np.array([traj_div_scores[i] for i in range(bs)])
            corr = np.array([traj_div_scores[i] for i in range(bs) if base_scores[i] > 0])
            nz = int((corr > 0).sum())
            print(f"  phi-share RAW (Stage 2, all rollouts): "
                  f"mean={raw.mean():.4f} std={raw.std():.4f} n={bs}")
            if len(corr):
                print(f"  phi-share PAID (Stage 3, correct only): mean={corr.mean():.4f} "
                      f"std={corr.std():.4f} n={len(corr)}  (nonzero {nz}, zero {len(corr) - nz})")
            # em_fmt_doc folds R_div into outcome_tensor (r_total), NOT step_level_tensor,
            # so read the paid sum straight from the correct-only shares.
            paid = self.lambda_div * float(corr.sum())
            print(f"  Paid diversity sum (= λ·Σ 1[correct]·R_div, folded into outcome_tensor): {paid:.2f}")
            wd = getattr(self, '_v2_within_disp', None)
            if wd:
                print(f"  WITHIN-group share CV over CORRECT (≥2-correct groups): "
                      f"mean_CV={wd[2]:.4f} mean_std={wd[0]:.4f} over {wd[3]} groups  ← decisive")
        else:
            # per-group correct-count distribution: of the n_agent rollouts per prompt,
            # how many groups got 0/1/.../n_agent correct. {0: ...} = all-wrong groups
            # (where budget_ungate fires); {n_agent: ...} = fully-solved groups.
            cc = Counter(sum(1 for i in idxs if base_scores[i] > 0) for idxs in strata.values())
            n_ag = max(group_sizes) if group_sizes else 0
            print(f"  Correct-per-group (out of {n_ag}): {dict(sorted(cc.items()))} | over {len(strata)} groups")
            # budget-diversity |Pool| distribution: how many rollouts actually compete for
            # R_div per group, split by group type. |Pool|=1 (or 0) is skipped by the <2
            # guard → no R_div paid. correct_grp = groups with ≥1 correct; allwrong_grp =
            # 0-correct groups (only ungate touches these).
            _pool_h = getattr(self, '_budget_pool_hist', None)
            if _pool_h is not None:
                cg = dict(sorted(_pool_h['correct_grp'].items()))
                ag = dict(sorted(_pool_h['allwrong_grp'].items()))
                def _paid(h, thr):  # groups with |Pool|>=thr (actually paid R_div)
                    return sum(v for k, v in h.items() if k >= thr)
                # gold-anchoring uses min_pool=1 for EVERY group (correct or all-wrong:
                # correctness-agnostic gold pool, lone gold-reacher paid); else min_pool=2.
                thr = 1 if self.budget_ungate_gold else 2
                print(f"  |Pool| dist — correct_grp: {cg} (paid≥{thr}: {_paid(_pool_h['correct_grp'], thr)})"
                      f" | allwrong_grp: {ag} (paid≥{thr}: {_paid(_pool_h['allwrong_grp'], thr)})")
            # gold-share canary (budget_ungate_gold): fraction of gold-anchored POOL
            # docs that bear gold. ↓share while R_div ↑ = junk-padding alarm.
            _gs = getattr(self, '_budget_gold_share', None)
            if self.budget_ungate_gold and _gs is not None and _gs['den'] > 0:
                print(f"  gold-share (gold-pool docs bearing gold): {_gs['share']:.3f} "
                      f"({_gs['num']}/{_gs['den']} doc-instances over {_gs['n_traj']} pool traj)"
                      f"  ← ↓ while R_div↑ = junk-padding")

        # Diversity score distribution (raw scores, before GRPO normalization)
        if traj_div_log and not _is_v2 and not _is_md:
            tdl = np.array(traj_div_log)
            label = "Step-level scores" if self.diversity_grouping in ('step_prompt', 'step_prompt_nsearch') else "Trajectory diversity"
            print(f"  {label} (all scored, raw): mean={tdl.mean():.4f} "
                  f"std={tdl.std():.4f} min={tdl.min():.4f} max={tdl.max():.4f} n={len(tdl)}")
            if self.diversity_grouping not in ('step_prompt', 'step_prompt_nsearch'):
                corr_scores = [traj_div_scores[i] for i in range(bs) if base_scores[i] > 0]
                if corr_scores:
                    cs_arr = np.array(corr_scores)
                    print(f"  Trajectory diversity (correct only): mean={cs_arr.mean():.4f} "
                          f"std={cs_arr.std():.4f} n={len(cs_arr)}")

        # Per-step breakdown (step_prompt mode only) — see how multi-turn is being learned
        if self.diversity_grouping in ('step_prompt', 'step_prompt_nsearch') and step_score_by_step:
            print(f"  Per-step breakdown:")
            for k in sorted(set(list(step_score_by_step.keys()) + list(step_group_size_by_step.keys()))):
                scores = step_score_by_step.get(k, [])
                sizes = step_group_size_by_step.get(k, [])
                n_groups_total = len(sizes)
                n_groups_scored = sum(1 for s in sizes if s >= 2)
                n_groups_singleton = n_groups_total - n_groups_scored
                if scores:
                    sa = np.array(scores)
                    print(f"    step {k}: scored_groups={n_groups_scored}/{n_groups_total} "
                          f"(singleton skipped:{n_groups_singleton}) | "
                          f"n_scored_trajs={len(scores)} | mean={sa.mean():.3f} std={sa.std():.3f}")
                else:
                    print(f"    step {k}: scored_groups=0/{n_groups_total} "
                          f"(all singletons) | no scores")

        if self.diversity_type in ('reasoning_embedding', 'search_embedding'):
            print(f"  Pairwise-mean diversity ({self.diversity_type}), "
                  f"embedding cache size={len(self._embed_cache)}")
            if cos_sim_log:
                cs = np.array(cos_sim_log)
                print(f"  Pairwise cos sim (all stratum/step pairs): mean={cs.mean():.4f}, "
                      f"std={cs.std():.4f}, min={cs.min():.4f}, max={cs.max():.4f}, n={len(cs)}")
            if nn_sim_log:
                ns = np.array(nn_sim_log)
                print(f"  NN cos sim per traj-step (max excl. self): mean={ns.mean():.4f}, "
                      f"std={ns.std():.4f}, min={ns.min():.4f}, max={ns.max():.4f}, n={len(ns)}")

        return outcome_tensor


import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    n_agent = config.actor_rollout_ref.rollout.get('n_agent', 8)
    lambda_div = config.get('lambda_div', 0.25)
    diversity_type = config.get('diversity_type', 'doc_set_unique')
    embed_model_name = config.get('embed_model_name', 'sentence-transformers/all-MiniLM-L6-v2')
    embed_device = config.get('embed_device', None)
    combine_mode = config.get('combine_mode', 'post_norm')
    diversity_grouping = config.get('diversity_grouping', 'trajectory')
    useful_weight = bool(config.get('useful_weight', False))
    useful_probe_path = config.get('useful_probe_path', 'Qwen/Qwen2.5-3B')
    useful_device = config.get('useful_device', None)
    useful_probe_url = config.get('useful_probe_url', None)
    useful_hybrid = bool(config.get('useful_hybrid', False))
    stage1_block = bool(config.get('stage1_block', False))
    budget_default = int(config.get('budget_default', 2))   # diversity_type='budget': fallback #supporting paras
    budget_ungate = bool(config.get('budget_ungate', False))  # asymmetric conditional-ungate on all-wrong groups
    budget_ungate_gold = bool(config.get('budget_ungate_gold', False))  # gold-anchored all-wrong pool
    budget_allrollouts = bool(config.get('budget_allrollouts', False))  # ablation: div over ALL within-budget rollouts (correct+wrong)
    budget_alltraj = bool(config.get('budget_alltraj', False))          # ablation: div over EVERY rollout (no budget, no correctness)
    budget_correct_nolimit = bool(config.get('budget_correct_nolimit', False))  # ablation: correct rollouts, no budget filter
    budget_mean_div = bool(config.get('budget_mean_div', False))        # control: pay pool-mean R_div (kills per-rollout differentiation)
    budget_lone_pool = bool(config.get('budget_lone_pool', False))  # lone correct+budget rollout → R_div=1.0
    div_sum_mode = bool(config.get('div_sum_mode', False))          # R_div = λ_ds·Σ1/m_d instead of the 1/|D_i| mean
    div_capped = bool(config.get('div_capped', False))              # R_div = Σ1/m_d / max(|D_i|, topk); overrides div_sum_mode
    div_topk = int(config.get('retriever', {}).get('topk', 3)) if hasattr(config, 'retriever') else 3
    div_lambda_default = float(config.get('div_lambda_default', 1.0 / 3.0))  # λ for unmapped data_sources
    _dlm = config.get('div_lambda_map', None)                       # e.g. +div_lambda_map.nq=0.3333
    div_lambda_map = dict(_dlm) if _dlm else None                   # None → {nq: 1/3, hotpotqa: 1/6}
    div_off_all_correct = bool(config.get('div_off_all_correct', False))  # zero R_div on all-correct groups
    div_gold_only = bool(config.get('div_gold_only', False))  # R_div numerator over gold-BEARING docs only (denominator stays |D_i|)
    div_gold_watch = bool(config.get('div_gold_watch', False))  # log div/gold_* on gold-agnostic arms (costs one extra decode pass/rollout)
    use_format_reward = bool(config.get('use_format_reward', True))  # em_fmt_doc: include R_fmt in R_total
    bleu_sigma = float(config.get('bleu_sigma', 0.65))  # diversity_type='bleu': DIVER Balanced-Shaping clip(TD;0,σ)
    guesser_gate = bool(config.get('guesser_gate', True))    # False: evidence 계열 E' 자격 게이트 해제 (pool=E, count 도 E 기준)
    gq_path_nq = config.get('gq_path_nq', None)              # 오프라인 G_q pkl (gq_doc_exempt)
    gq_path_hotpotqa = config.get('gq_path_hotpotqa', None)
    print(f"[main_ppo] diversity_type={diversity_type}, lambda_div={lambda_div}, "
          f"combine_mode={combine_mode}, diversity_grouping={diversity_grouping}, "
          f"embed_model_name={embed_model_name}, embed_device={embed_device}, "
          f"useful_weight={useful_weight}, useful_probe={useful_probe_path}, "
          f"useful_device={useful_device}, useful_probe_url={useful_probe_url}, "
          f"useful_hybrid={useful_hybrid}, stage1_block={stage1_block}, "
          f"budget_ungate={budget_ungate}, budget_ungate_gold={budget_ungate_gold}, "
          f"budget_lone_pool={budget_lone_pool}, budget_allrollouts={budget_allrollouts}, "
          f"budget_alltraj={budget_alltraj}, budget_correct_nolimit={budget_correct_nolimit}, "
          f"budget_mean_div={budget_mean_div}, use_format_reward={use_format_reward}, "
          f"div_sum_mode={div_sum_mode}, div_capped={div_capped} (topk={div_topk}), "
          f"div_lambda_map={div_lambda_map or 'default(nq=1/3,hotpotqa=1/6)'}, "
          f"div_lambda_default={div_lambda_default}, div_off_all_correct={div_off_all_correct}, "
          f"guesser_gate={guesser_gate}")

    reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=0,
        lambda_div=lambda_div,
        n_agent=n_agent,
        diversity_type=diversity_type,
        embed_model_name=embed_model_name,
        embed_device=embed_device,
        combine_mode=combine_mode,
        diversity_grouping=diversity_grouping,
        useful_weight=useful_weight,
        useful_probe_path=useful_probe_path,
        useful_device=useful_device,
        useful_probe_url=useful_probe_url,
        useful_hybrid=useful_hybrid,
        stage1_block=stage1_block,
        budget_default=budget_default,
        budget_ungate=budget_ungate,
        budget_ungate_gold=budget_ungate_gold,
        budget_allrollouts=budget_allrollouts,
        budget_alltraj=budget_alltraj,
        budget_correct_nolimit=budget_correct_nolimit,
        budget_mean_div=budget_mean_div,
        budget_lone_pool=budget_lone_pool,
        div_sum_mode=div_sum_mode,
        div_capped=div_capped,
        div_topk=div_topk,
        div_lambda_map=div_lambda_map,
        div_lambda_default=div_lambda_default,
        div_off_all_correct=div_off_all_correct,
        div_gold_only=div_gold_only,
        div_gold_watch=div_gold_watch,
        use_format_reward=use_format_reward,
        bleu_sigma=bleu_sigma,
        guesser_gate=guesser_gate,
        gq_path_nq=gq_path_nq,
        gq_path_hotpotqa=gq_path_hotpotqa,
    )

    # Validation에서는 diversity reward 끄기 (순수 accuracy만 측정)
    val_reward_fn = RewardManager(
        tokenizer=tokenizer,
        num_examine=1,
        lambda_div=0.0,
        n_agent=n_agent,
        diversity_type=diversity_type,
        embed_model_name=embed_model_name,
        embed_device=embed_device,
        combine_mode=combine_mode,
        diversity_grouping=diversity_grouping,
        gq_path_nq=gq_path_nq,
        gq_path_hotpotqa=gq_path_hotpotqa,
    )

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
