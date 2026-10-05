import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from src.retrieve import Retriever

QUESTIONS = Path("eval/questions.jsonl")
RESULTS_DIR = Path("eval/results")


def load_questions():
    with QUESTIONS.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def is_hit(retrieved, gold):
    gold_pages = {(g["source"], g["page"]) for g in gold}
    return any((c["source"], c["page"]) in gold_pages for c in retrieved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3, 4, 6])
    args = parser.parse_args()
    tag = os.getenv("RUN_TAG") or "base"

    questions = [q for q in load_questions() if q["answerable"]]
    retriever = Retriever()
    max_k = max(args.ks)

    rows = []
    for q in questions:
        retrieved = retriever.search(q["question"], k=max_k)
        rows.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "hits": {k: is_hit(retrieved[:k], q["gold"]) for k in args.ks},
            "top1": f'{retrieved[0]["source"]} p.{retrieved[0]["page"]}' if retrieved else None,
        })

    print(f"\nRun: {tag} | answerable questions: {len(rows)}")
    for k in args.ks:
        n = sum(r["hits"][k] for r in rows)
        print(f"  hit@{k}: {n}/{len(rows)} = {n / len(rows):.0%}")

    focus_k = 4 if 4 in args.ks else args.ks[0]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["hits"][focus_k])
    print(f"\nBy category at k={focus_k}:")
    for cat, vals in by_cat.items():
        print(f"  {cat}: {sum(vals)}/{len(vals)}")

    print(f"\nMisses at k={focus_k}:")
    for r in rows:
        if not r["hits"][focus_k]:
            print(f"  #{r['id']} [{r['category']}] {r['question']}  -> top1: {r['top1']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"retrieval_{tag}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
