import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# HF 캐시를 쓰기 가능한 곳으로. 홈의 ~/.cache/huggingface 는 root 소유라
# trust_remote_code=True 경로의 init_hf_modules() 가 modules/ 를 못 만들고 죽는다.
# transformers 가 HF_MODULES_CACHE 를 import 시점에 확정하므로 반드시
# transformers/vllm import 보다 먼저 설정해야 한다. 학습 스크립트와 같은 경로.
os.environ.setdefault("HF_HOME", "/shared/models/hf-dial")

import re
import json
import time
import glob
import logging
import argparse
import multiprocessing as mp
from math import comb
from collections import defaultdict

import requests
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_PATH  = "/home/dongwan/models/verl_checkpoints/DIVER-3B/actor/global_step_200"
DATA_DIR    = "test_data"
OUTPUT_DIR  = "test_results"

N_SAMPLES   = 64
PASS_KS     = [1, 2, 4, 8, 16, 32, 64]
TEMPERATURE = 1.0

DATASET_ORDER = ["nq", "triviaqa", "popqa", "hotpotqa",
                 "2wikimultihopqa", "musique", "bamboogle"]

EVAL_GPU_IDS   = [0, 1, 2, 3, 4, 5]
BATCH_SIZE     = 128
MAX_TURNS      = 4
MAX_NEW_TOKENS = 1024
TOP_K          = 3
RETRIEVAL_URL  = "http://127.0.0.1:8000/retrieve"
INVALID_OBS    = ("\nMy previous action is invalid. "
                  "If I want to search, I should put the query between <search> and </search>. "
                  "If I want to give the final answer, I should put the answer between <answer> "
                  "and </answer>. Let me try again.\n")

def parse_contents(contents):
    lines = contents.strip().split("\n", 1)
    title = lines[0].strip('"') if lines else ""
    text  = lines[1].strip() if len(lines) > 1 else ""
    return title, text

class WikiSearchEngine:
    def __init__(self, url=RETRIEVAL_URL, timeout=300, max_retries=8):
        self.url, self.timeout, self.max_retries = url, timeout, max_retries
        self.session = requests.Session()

    def search_batch(self, queries, k=TOP_K):
        if not queries:
            return []
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    self.url,
                    json={"queries": queries, "topk": k, "return_scores": True},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                result = resp.json()["result"]
                break
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    requests.exceptions.HTTPError) as e:
                last_err = e
                wait = min(30, 2 ** attempt)
                print(f"[search retry {attempt+1}/{self.max_retries}] {type(e).__name__}, wait {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError(f"search_batch failed after {self.max_retries} retries: {last_err}")

        out = []
        for items in result:
            buf = ""
            for i, item in enumerate(items):
                doc = item["document"] if isinstance(item, dict) and "document" in item else item
                title, text = parse_contents(doc.get("contents", ""))
                buf += f'Doc {i+1} (Title: "{title}")\nContent: {text}\n\n'
            out.append(buf or "No relevant documents found.")
        return out

def make_prompt(question, tokenizer):
    base = (
        f"Answer the given question. "
        f"You must conduct reasoning inside <think> and </think> first every time you get new information. "
        f"After reasoning, if you find you lack some knowledge, you can call a search engine by "
        f"<search> query </search> and it will return the top searched results between "
        f"<information> and </information>. "
        f"You can search as many times as your want. "
        f"If you find no further external knowledge needed, you can directly provide the answer inside "
        f"<answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. "
        f"Question: {question}\n"
    )
    if tokenizer.chat_template:
        base = tokenizer.apply_chat_template(
            [{"role": "user", "content": base}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return base

def extract(text, open_tag, close_tag):
    if open_tag in text and close_tag in text:
        s = text.index(open_tag) + len(open_tag)
        e = text.index(close_tag, s)
        return text[s:e].strip()
    return ""

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def em_match(pred, golds):
    return bool(pred) and any(normalize(g) == normalize(pred) for g in golds if g)

def unbiased_pass_at_k(n, c, k):
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def worker(gpu_id, tasks, result_queue, cfg):
    logging.getLogger("vllm").setLevel(logging.WARNING)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    model_path = cfg["model_path"]
    batch_size = cfg["batch_size"]
    max_turns  = cfg["max_turns"]

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(model=model_path, dtype="float16", trust_remote_code=True,
              tensor_parallel_size=1, gpu_memory_utilization=cfg.get("gpu_mem_util", 0.7),
              enforce_eager=False)
    wiki = WikiSearchEngine()
    print(f"[GPU {gpu_id}] ready — {len(tasks)} rollouts")

    sampling_params = SamplingParams(
        temperature=cfg["temperature"], top_p=1.0, max_tokens=MAX_NEW_TOKENS,
        stop=["</search>", "</answer>"], include_stop_str_in_output=True,
    )

    states = []
    for t in tasks:
        p = make_prompt(t["question"], tokenizer)
        states.append({"task": t, "context": p, "prompt_len": len(p),
                       "answer": "", "step": 0, "invalid": 0, "finished": False})

    t0, done, total = time.time(), 0, len(states)
    while True:
        active = [i for i, st in enumerate(states) if not st["finished"]]
        if not active:
            break
        batch_idx = active[:batch_size]
        outputs = llm.generate([states[i]["context"] for i in batch_idx],
                               sampling_params, use_tqdm=False)

        search_targets, search_queries, finished_now = [], [], []
        for i, out in zip(batch_idx, outputs):
            text = out.outputs[0].text
            states[i]["context"] += text
            states[i]["step"]    += 1

            ans = extract(text, "<answer>", "</answer>")
            if ans:
                states[i]["answer"] = ans
                states[i]["finished"] = True
                finished_now.append(i)
                continue
            sq = extract(text, "<search>", "</search>")
            if sq and states[i]["step"] <= max_turns:
                search_queries.append(sq)
                search_targets.append(i)
            elif states[i]["step"] > max_turns:
                states[i]["finished"] = True
                finished_now.append(i)
            else:
                # Invalid action: emitted neither a well-formed <search> nor <answer>.
                # Training injects this correction and keeps the rollout alive
                # (generation.py execute_predictions, dones=0), so eval must too —
                # otherwise the model is judged without the retry it was trained with.
                states[i]["context"] += INVALID_OBS
                states[i]["invalid"] += 1

        if search_queries:
            for i, res in zip(search_targets, wiki.search_batch(search_queries, k=TOP_K)):
                states[i]["context"] += f"\n\n<information>\n{res}</information>\n\n"

        for i in finished_now:
            t = states[i]["task"]
            correct = int(em_match(states[i]["answer"], t["golden_answers"]))
            traj = states[i]["context"][states[i]["prompt_len"]:]
            result_queue.put({
                "dataset":        t["dataset"],
                "id":             t["id"],
                "sample_idx":     t["sample_idx"],
                "answer":         states[i]["answer"],
                "correct":        correct,
                "golden_answers": t["golden_answers"],
                "retrieved":      traj.count("<information>"),
                "invalid":        states[i]["invalid"],
                "trajectory":     traj,
            })
            states[i]["context"] = ""
            done += 1
        if finished_now and done % 500 < len(finished_now):
            el = time.time() - t0
            rate = done / el if el > 0 else 0
            eta = (total - done) / rate if rate > 0 else float("inf")
            print(f"[GPU {gpu_id}] {done}/{total} ({done/total*100:.0f}%) | "
                  f"{rate:.1f}/s | ETA {eta/60:.1f}m")

    print(f"[GPU {gpu_id}] done {done}/{total} ({time.time()-t0:.0f}s)")
    result_queue.put({"_worker_done": gpu_id})

def load_dataset(ds, data_dir=None):
    path = os.path.join(data_dir or DATA_DIR, f"{ds}.jsonl")
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            samples.append({
                "dataset":        ds,
                "id":             item.get("id", f"{ds}_{n}"),
                "question":       item.get("question", item.get("query", "")),
                "golden_answers": item.get("golden_answers", item.get("answers", [])),
            })
    return samples

def generate(samples_per_ds, model_path, gpu_ids, batch_size, n_samples,
             temperature, max_turns, out_dir, gpu_mem_util):
    os.makedirs(out_dir, exist_ok=True)
    ds_paths = {ds: os.path.join(out_dir, f"{ds}.jsonl") for ds in samples_per_ds}
    expected = sum(len(s) for s in samples_per_ds.values()) * n_samples

    raw = []
    for ds, p in ds_paths.items():
        if os.path.exists(p) and os.path.getsize(p) > 0:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    raw.append((r["dataset"], r["id"], int(r.get("correct", 0))))
    if len(raw) >= expected:
        print(f"  resume: {len(raw)} rollouts on disk -> recompute pass@k only")
        return raw

    tasks = []
    for ds, samples in samples_per_ds.items():
        for s in samples:
            for j in range(n_samples):
                tasks.append({**s, "sample_idx": j})
    shards = [[] for _ in gpu_ids]
    for i, t in enumerate(tasks):
        shards[i % len(gpu_ids)].append(t)

    cfg = {"model_path": model_path, "batch_size": batch_size,
           "temperature": temperature, "max_turns": max_turns,
           "gpu_mem_util": gpu_mem_util}
    manager = mp.Manager()
    q = manager.Queue()
    procs = []
    for i, g in enumerate(gpu_ids):
        if not shards[i]:
            continue
        p = mp.Process(target=worker, args=(g, shards[i], q, cfg))
        p.start()
        procs.append(p)

    handles = {ds: open(p, "w", encoding="utf-8") for ds, p in ds_paths.items()}
    raw = []
    done_workers = 0
    while done_workers < len(procs):
        r = q.get()
        if "_worker_done" in r:
            done_workers += 1
            continue
        handles[r["dataset"]].write(json.dumps(r, ensure_ascii=False) + "\n")
        handles[r["dataset"]].flush()
        raw.append((r["dataset"], r["id"], int(r["correct"])))
    for h in handles.values():
        h.close()
    for p in procs:
        p.join()
    return raw

def compute_passk(raw, datasets, pass_ks):
    by_q = defaultdict(lambda: [0, 0])
    for ds, qid, correct in raw:
        by_q[(ds, qid)][0] += 1
        by_q[(ds, qid)][1] += correct

    per_ds = defaultdict(lambda: defaultdict(list))
    for (ds, _qid), (n_seen, c_cor) in by_q.items():
        for k in pass_ks:
            per_ds[ds][k].append(unbiased_pass_at_k(n_seen, c_cor, k))

    result = {}
    for ds in datasets:
        kp = per_ds.get(ds, {})
        n_q = len(next(iter(kp.values()))) if kp else 0
        result[ds] = {"n": n_q}
        for k in pass_ks:
            v = kp.get(k, [])
            result[ds][f"pass@{k}"] = (sum(v) / len(v)) if v else 0.0
    return result

def write_outputs(result, datasets, pass_ks, out_dir, label, n_samples, temperature):
    os.makedirs(out_dir, exist_ok=True)
    for ds in datasets:
        info = result.get(ds)
        if not info:
            continue
        lines = [
            f"dataset    : {ds}",
            f"model      : {label}",
            f"n_questions: {info['n']}",
            f"n_samples  : {n_samples}   (T={temperature}, unbiased: 1 - C(n-c,k)/C(n,k))",
            "",
        ]
        for k in pass_ks:
            lines.append(f"pass@{k:<3}= {info[f'pass@{k}']*100:6.2f}%")
        with open(os.path.join(out_dir, f"{ds}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    L = []
    L.append("=" * 90)
    L.append(f"TEST unbiased pass@k (EM)  |  model={label}  |  n_samples={n_samples}, T={temperature}")
    L.append("pass@k = 1 - C(n-c,k)/C(n,k)   (test_data/)")
    L.append("=" * 90)
    head = f"{'dataset':<18} | " + " | ".join(f"pass@{k}".rjust(8) for k in pass_ks) + " |     n"
    L.append(head)
    L.append("-" * len(head))
    avg = {k: [] for k in pass_ks}
    for ds in datasets:
        info = result.get(ds)
        if not info:
            continue
        L.append(f"{ds:<18} | " + " | ".join(f"{info[f'pass@{k}']*100:7.2f}%" for k in pass_ks)
                 + f" | {info['n']:>6d}")
        for k in pass_ks:
            avg[k].append(info[f"pass@{k}"])
    L.append("-" * len(head))
    L.append(f"{'macro-avg':<18} | " +
             " | ".join(f"{(sum(avg[k])/len(avg[k]) if avg[k] else 0)*100:7.2f}%" for k in pass_ks)
             + " |")
    L.append("=" * 90)
    txt = "\n".join(L)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return txt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--n_samples", type=int, default=N_SAMPLES)
    parser.add_argument("--pass_ks", nargs="+", type=int, default=PASS_KS)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max_turns", type=int, default=MAX_TURNS)
    parser.add_argument("--gpu_ids", default=",".join(map(str, EVAL_GPU_IDS)))
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.65)
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    model_path = args.model_path.rstrip("/")
    if not os.path.isdir(model_path):
        raise SystemExit(f"[error] no checkpoint dir: {model_path}")

    step_name = os.path.basename(model_path)
    parent = os.path.dirname(model_path)
    if os.path.basename(parent) == "actor":
        parent = os.path.dirname(parent)
    exp = os.path.basename(parent)

    label = f"{exp}/{step_name}"
    pass_ks = [k for k in args.pass_ks if k <= args.n_samples]

    mp.set_start_method("spawn", force=True)
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]

    data_dir = args.data_dir.rstrip("/")
    present = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(data_dir, "*.jsonl"))}
    order = DATASET_ORDER if args.datasets is None else list(args.datasets)
    datasets = [d for d in order if d in present]
    if not datasets:
        raise SystemExit(f"[error] no test datasets found in {data_dir}")

    samples_per_ds = {ds: load_dataset(ds, data_dir) for ds in datasets}
    total_q = sum(len(s) for s in samples_per_ds.values())

    out_dir = os.path.join(args.output_dir, exp, step_name)
    print("=" * 72)
    print(f"TEST  model={label}\n      {model_path}")
    print(f"datasets={datasets} (q={total_q})  n_samples={args.n_samples}  pass@{pass_ks}")
    print(f"data={data_dir}  out={out_dir}  GPUs={gpu_ids}")
    print("=" * 72)

    t0 = time.time()
    raw = generate(samples_per_ds, model_path, gpu_ids, args.batch_size,
                   args.n_samples, args.temperature, args.max_turns, out_dir,
                   args.gpu_memory_utilization)
    result = compute_passk(raw, datasets, pass_ks)
    txt = write_outputs(result, datasets, pass_ks, out_dir, label,
                        args.n_samples, args.temperature)
    print("\n" + txt)
    print(f"\nDone in {(time.time()-t0)/60:.1f}m -> {out_dir}/summary.txt")

if __name__ == "__main__":
    main()