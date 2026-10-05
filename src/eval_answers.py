import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

from src.eval_retrieval import load_questions
from src.generate import answer
from src.retrieve import Retriever

RESULTS_DIR = Path("eval/results")
FIELDS = ["id", "category", "answerable", "question", "gold_answer",
          "answer", "sources", "refused", "grade"]

def csv_path(k):
    tag = os.getenv("RUN_TAG") or "base"
    return RESULTS_DIR / f"answers_{tag}_k{k}.csv"

def run(k):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    retriever = Retriever()
    with csv_path(k).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for q in load_questions():
            result = answer(q["question"], retriever, k=k)
            text = result["answer"].replace("\u2019", "'")
            writer.writerow({
                "id": q["id"], "category": q["category"], "answerable": q["answerable"],
                "question": q["question"], "gold_answer": q["gold_answer"],
                "answer": text,
                "sources": "; ".join(f'{c["source"]} p.{c["page"]}' for c in result["sources"]),
                "refused": "i don't know" in text.lower(),
                "grade": "",
            })
            print(f"done #{q['id']}")
    print(f"Saved {csv_path(k)}. Open it, fill the 'grade' column for answerable rows, then run summarize.")

def summarize(k):
    with csv_path(k).open() as f:
        rows = list(csv.DictReader(f))
    answerable = [r for r in rows if r["answerable"] == "True"]
    graded = [r for r in answerable if r["grade"].strip() != ""]
    unans = [r for r in rows if r["answerable"] == "False"]

    if graded:
        score = sum(float(r["grade"]) for r in graded) / len(graded)
        print(f"\nAnswer score: {score:.2f} over {len(graded)} graded answerable questions")
        by_cat = defaultdict(list)
        for r in graded:
            by_cat[r["category"]].append(float(r["grade"]))
        for cat, vals in by_cat.items():
            print(f"  {cat}: {sum(vals) / len(vals):.2f} ({len(vals)} qs)")
    false_refusals = sum(r["refused"] == "True" for r in answerable)
    print(f"\nFalse refusals on answerable questions: {false_refusals}/{len(answerable)}")
    if unans:
        good = sum(r["refused"] == "True" for r in unans)
        print(f"Correct refusals on unanswerable: {good}/{len(unans)}")

if __name__ == "__main__":
    mode, k = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4
    run(k) if mode == "run" else summarize(k)
