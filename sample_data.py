import argparse
import random
from pathlib import Path

SPLIT = {
    "nq": "test",
    "triviaqa": "test",
    "popqa": "test",
    "hotpotqa": "dev",
    "2wikimultihopqa": "dev",
    "musique": "dev",
    "bamboogle": "test",
}

N_SAMPLES = 500
SEED = 42


def subsample(src, dst, n, seed):
    lines = [ln for ln in src.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if n < len(lines):
        idx = sorted(random.Random(seed).sample(range(len(lines)), n))
        lines = [lines[i] for i in idx]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def build(data_dir, out_dir, n=N_SAMPLES, seed=SEED, split=SPLIT):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {ds: subsample(data_dir / ds / f"{sp}.jsonl",
                          out_dir / f"{ds}.jsonl", n, seed)
            for ds, sp in split.items()}


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(here / "data"))
    ap.add_argument("--out", default=str(here / "test_data"))
    ap.add_argument("--n", type=int, default=N_SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    build(args.data, args.out, args.n, args.seed)


if __name__ == "__main__":
    main()