#!/usr/bin/env python3
"""
Frozen-base-probe HTTP server for ΔP useful-weight scoring.

Mirrors the retrieval server pattern (separate process, own GPU) so the training
driver — which runs the RewardManager as a CPU-only Ray driver and has NO GPU —
can get GPU-speed gold scoring over HTTP instead of a ~30 min/step CPU probe.

POST /score  {"contexts": [str, ...], "golds": [str, ...]}   (parallel lists)
  -> {"logps": [float | null, ...]}
where logps[j] = sum_t logP(gold_j token_t | context_j + " <answer>") under the
frozen base LM, teacher-forced. null when the gold tokenizes to 0 tokens.

Scoring matches RewardManager._score_gold exactly (same " <answer>" scaffold,
same separate-gold tokenization with a leading space and add_special_tokens=False,
same left-truncation to max_len), so the caller's prefix ΔP = thru - before equals
the offline-validated signal. Batched with LEFT padding (+ attention_mask and
position_ids) so each row's gold is its final n_gold tokens; micro-batches are
filled under a token budget to bound the (B·L·V) softmax memory.
"""
import argparse
from typing import List, Optional

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

# globals set in __main__
MODEL = None
TOK = None
DEV = None


class ScoreRequest(BaseModel):
    contexts: List[str]
    golds: List[str]
    max_len: Optional[int] = 7000
    tok_budget: Optional[int] = 3000   # cap on B*maxL per micro-batch (softmax memory);
                                       # lowered so the probe's peak GPU footprint stays
                                       # small enough to coexist with faiss on one card
    reduce: Optional[str] = "sum"      # "sum"     = Σ_t logP(gold_t)  (legacy ΔP useful-weight)
                                       # "mean"    = length-normalized logP (per-token mean)
                                       # "logodds" = mean per-token log-odds, log(p/(1-p)),
                                       #             p=exp(logP). The Meaningful-Diversity
                                       #             scorer s(S): evidence adds in log-odds
                                       #             (residual-stream-additive -> Shapley-
                                       #             additive) and does NOT saturate at high
                                       #             confidence, so φ magnitude survives.


@torch.no_grad()
def score_batch(contexts, golds, max_len=7000, tok_budget=6000, reduce="sum"):
    results = [None] * len(contexts)
    # tokenize each request: context+scaffold then gold (separately, like _score_gold)
    items = []   # (orig_idx, ids_1d, n_gold)
    for idx, (c, g) in enumerate(zip(contexts, golds)):
        g = "" if g is None else str(g)
        ids_p = TOK(c + " <answer>", return_tensors="pt").input_ids[0]
        gold_ids = TOK(" " + g, add_special_tokens=False, return_tensors="pt").input_ids[0]
        n_gold = gold_ids.shape[0]
        if n_gold < 1:
            continue
        ids = torch.cat([ids_p, gold_ids])
        if ids.shape[0] > max_len:
            ids = ids[-max_len:]
        items.append((idx, ids, n_gold))
    if not items:
        return results
    # length-sorted, greedily packed under the token budget (less left-pad waste)
    items.sort(key=lambda x: x[1].shape[0])
    pad_id = TOK.pad_token_id
    i = 0
    while i < len(items):
        j, maxL = i, 0
        while j < len(items):
            cand = max(maxL, items[j][1].shape[0])
            if j > i and cand * (j - i + 1) > tok_budget:
                break
            maxL = cand
            j += 1
        chunk = items[i:j]
        i = j
        B = len(chunk)
        maxL = max(it[1].shape[0] for it in chunk)
        input_ids = torch.full((B, maxL), pad_id, dtype=torch.long)
        attn = torch.zeros((B, maxL), dtype=torch.long)
        for r, (idx, ids, ng) in enumerate(chunk):
            L = ids.shape[0]
            input_ids[r, maxL - L:] = ids        # LEFT pad -> gold = final ng tokens
            attn[r, maxL - L:] = 1
        input_ids = input_ids.to(DEV)
        attn = attn.to(DEV)
        pos = attn.long().cumsum(-1) - 1         # correct positions under left padding
        pos.clamp_(min=0)
        logits = MODEL(input_ids, attention_mask=attn, position_ids=pos).logits[:, :-1]
        lp = torch.log_softmax(logits.float(), dim=-1)
        tgt = input_ids[:, 1:].to(lp.device)                   # device_map: logits on last shard
        tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (B, maxL-1)
        for r, (idx, ids, ng) in enumerate(chunk):
            g_lp = tok_lp[r, -ng:]                       # per-token logP of the gold
            if reduce == "logodds":
                # s = mean_t log_odds(p_t), p_t=exp(logP_t); log_odds = logP - log(1-p).
                # clamp p to [eps, 1-eps] (avoid ±inf), stable log1mexp for log(1-p).
                import math
                lp = g_lp.clamp(min=math.log(1e-6), max=math.log(1.0 - 1e-6))
                log1m = torch.where(lp > math.log(0.5),         # stable log(1-exp(lp))
                                    torch.log(-torch.expm1(lp)),
                                    torch.log1p(-torch.exp(lp)))
                results[idx] = (lp - log1m).mean().item()
            elif reduce == "mean":
                results[idx] = g_lp.mean().item()
            else:
                results[idx] = g_lp.sum().item()
        del logits, lp, tok_lp, input_ids, attn, pos
    return results


@app.post("/score")
def score_endpoint(req: ScoreRequest):
    logps = score_batch(req.contexts, req.golds,
                        req.max_len or 7000, req.tok_budget or 6000,
                        reduce=req.reduce or "sum")
    return {"logps": logps}


@app.get("/health")
def health():
    return {"ok": True, "device": str(DEV)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Frozen base-probe server for ΔP useful-weight.")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--dtype", type=str, default="float16")
    ap.add_argument("--attn", type=str, default="flash_attention_2",
                    help="attention impl: flash_attention_2 (fastest, default) | sdpa | eager")
    ap.add_argument("--device_map", type=str, default=None,
                    help="HF device_map (e.g. 'balanced') to SHARD the model across all "
                         "visible GPUs; pipelines the forward so activation peak is split. "
                         "None = single GPU (legacy).")
    ap.add_argument("--max_mem_gib", type=float, default=None,
                    help="per-GPU memory cap (GiB) for device_map; set low (e.g. 4) to FORCE "
                         "an even split when the model would otherwise fit on one card.")
    args = ap.parse_args()

    DEV = "cuda" if torch.cuda.is_available() else "cpu"
    TOK = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if TOK.pad_token is None:
        TOK.pad_token = TOK.eos_token
    # flash-attention-2: the transformer body (over the full doc context) is ~84% of the
    # probe forward; FA2 speeds it ~1.5-2x for long sequences. Falls back to sdpa if the
    # build is incompatible.
    def _load(**kw):
        try:
            return AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=getattr(torch, args.dtype), trust_remote_code=True,
                attn_implementation=args.attn, **kw).eval()
        except Exception as ex:
            print(f"[probe] attn_implementation={args.attn} failed ({ex}); falling back to sdpa")
            return AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=getattr(torch, args.dtype), trust_remote_code=True,
                attn_implementation="sdpa", **kw).eval()
    if args.device_map:
        mm = None
        if args.max_mem_gib:
            mm = {i: f"{args.max_mem_gib}GiB" for i in range(torch.cuda.device_count())}
        print(f"[probe] loading {args.model} device_map={args.device_map} "
              f"max_memory={mm} attn={args.attn} ({args.dtype})")
        MODEL = _load(device_map=args.device_map, max_memory=mm)
        DEV = MODEL.get_input_embeddings().weight.device
        print(f"[probe] sharded; input device = {DEV}")
    else:
        print(f"[probe] loading {args.model} on {DEV} attn={args.attn} ({args.dtype})")
        MODEL = _load().to(DEV)
    for p in MODEL.parameters():
        p.requires_grad = False
    print(f"[probe] ready on :{args.port}  (pad_id={TOK.pad_token_id}, frozen)")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
