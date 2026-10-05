from src.eval_retrieval import load_questions, is_hit
from src.retrieve import Retriever

retriever = Retriever()

rows = []
for q in load_questions():
    results = retriever.search(q["question"], k=4)
    top = results[0]["score"] if results else 0.0
    if not q["answerable"]:
        group = "UNANSWERABLE"
    elif is_hit(results, q["gold"]):
        group = "answerable-HIT"
    else:
        group = "answerable-MISS"
    rows.append((group, top, q["id"], q["category"], q["question"][:55]))

print("=== Per question (sorted by group, then score) ===")
for group, top, qid, cat, text in sorted(rows, key=lambda r: (r[0], r[1])):
    print(f"{group:<16} {top:.3f}  #{qid:<3} [{cat}] {text}")

print("\n=== Summary ===")
for name in ["answerable-HIT", "answerable-MISS", "UNANSWERABLE"]:
    vals = sorted(r[1] for r in rows if r[0] == name)
    if vals:
        print(f"{name:<16} n={len(vals):<3} min={vals[0]:.3f}  max={vals[-1]:.3f}  all={[round(v, 2) for v in vals]}")

print("\n=== Junk and off-topic probes ===")
probes = [
    "how are you", "hello", "hi there", "thanks", "asdf",
    "give me a pasta recipe", "who won the 2022 world cup",
    "what is the weather today", "tell me a joke",
    "what is artificial intelligence",
    "tell me about risk",
]
for text in probes:
    top = retriever.search(text, k=1)[0]["score"]
    print(f"{top:.3f}  {text!r}")