import argparse
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

DATASETS = ["nq", "triviaqa", "popqa", "hotpotqa",
            "2wikimultihopqa", "musique", "bamboogle"]
SPLITS = ["train", "dev", "test"]

REPO_ID = "RUC-NLPIR/FlashRAG_datasets"

def discover(datasets, splits):
    files = set(HfApi().list_repo_files(REPO_ID, repo_type="dataset"))
    wanted = [f"{ds}/{sp}.jsonl" for ds in datasets for sp in splits]
    return [rel for rel in wanted if rel in files]

def download(out, datasets=DATASETS, splits=SPLITS, force=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for rel in discover(datasets, splits):
        dest = out / rel
        if force or not dest.exists():
            hf_hub_download(REPO_ID, rel, repo_type="dataset", local_dir=str(out))
        paths.append(dest)
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    download(args.out, args.datasets, args.splits, args.force)

if __name__ == "__main__":
    main()