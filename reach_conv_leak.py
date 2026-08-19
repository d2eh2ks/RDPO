import os
import re
import sys
import json
import glob
from math import comb
from collections import defaultdict

import importlib.util

# 학습과 완전히 같은 정규화를 쓰되, verl 패키지 __init__ (tensordict 의존) 은 피한다.
_spec = importlib.util.spec_from_file_location(
    "qa_em_format", "verl/utils/reward_score/qa_em_format.py")
_qaf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qaf)
normalize_answer = _qaf.normalize_answer

RESULTS_DIR = "test_results"
KS = [1, 2, 4, 8, 16, 32, 64]

INFO_RE = re.compile(r"<information>(.*?)</information>", re.DOTALL)
DOC_RE = re.compile(r'Doc \d+\s*\(Title:\s*"([^"]*)"\)\s*(?:\nContent:)?(.*?)(?=Doc \d+\s*\(Title:|\Z)',
                    re.DOTALL)


def reached(trajectory, golds_norm):
    for block in INFO_RE.findall(trajectory):
        for _title, content in DOC_RE.findall(block):
            cn = normalize_answer(" ".join(content.split()))
            if any(g in cn for g in golds_norm):
                return True
    return False


def load(model):
    out = {}
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, model, "*.jsonl"))):
        ds = os.path.splitext(os.path.basename(path))[0]
        agg = defaultdict(lambda: [0, 0, 0, 0])   # n, correct, reach, correct&!reach
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                golds = [normalize_answer(g) for g in r["golden_answers"] if g]
                golds = [g for g in golds if g]
                rc = reached(r["trajectory"], golds)
                c = int(r["correct"])
                a = agg[r["id"]]
                a[0] += 1
                a[1] += c
                a[2] += int(rc)
                a[3] += int(c and not rc)
        if agg:
            out[ds] = dict(agg)
    return out


def C(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def per_question(n, c, rch, leak, k):
    """Exact expectation over size-k subsets of the n rollouts."""
    tot = C(n, k)
    p_pass = 1.0 - C(n - c, k) / tot
    p_reach = 1.0 - C(n - rch, k) / tot
    # no rollout reached (subset inside the n-rch non-reachers) AND at least one correct
    p_noreach_pass = (C(n - rch, k) - C(n - rch - leak, k)) / tot
    return p_pass, p_reach, p_pass - p_noreach_pass, p_noreach_pass


def main():
    models = sys.argv[1:]
    if not models:
        raise SystemExit("usage: reach_conv_leak.py <model> [<model> ...]")
    data = {m: load(m) for m in models}
    datasets = sorted(set.intersection(*(set(d) for d in data.values())))

    for ds in datasets + (["macro-avg"] if len(datasets) > 1 else []):
        print(f"\n=== {ds} ===")
        hdr = f"{'model':<34}{'metric':<10}" + "".join(f"{'k='+str(k):>9}" for k in KS)
        print(hdr)
        print("-" * len(hdr))
        for m in models:
            acc = {key: [] for key in ("pass", "reach", "joint", "leak")}
            for k in KS:
                dss = datasets if ds == "macro-avg" else [ds]
                vals = []
                for d in dss:
                    q = data[m][d]
                    rows = [per_question(*q[qid], k) for qid in q]
                    vals.append([sum(x[j] for x in rows) / len(rows) for j in range(4)])
                mean = [sum(v[j] for v in vals) / len(vals) for j in range(4)]
                acc["pass"].append(mean[0])
                acc["reach"].append(mean[1])
                acc["joint"].append(mean[2])
                acc["leak"].append(mean[3])
            conv = [j / r if r else float("nan") for j, r in zip(acc["joint"], acc["reach"])]
            for name, series in (("reach@k", acc["reach"]), ("conv@k", conv),
                                 ("leak@k", acc["leak"]), ("pass@k", acc["pass"])):
                print(f"{m:<34}{name:<10}" + "".join(f"{v * 100:>8.1f}" for v in series))
            chk = [r * c + l for r, c, l in zip(acc["reach"], conv, acc["leak"])]
            print(f"{'':<34}{'(check)':<10}" + "".join(f"{v * 100:>8.1f}" for v in chk))
            print("-" * len(hdr))


if __name__ == "__main__":
    main()
