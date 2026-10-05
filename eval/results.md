# RAG Evaluation & Experiment Results

## Experiment Matrix

| Run | hit@4 | Answer Score | False Refusals | Correct Refusals | Primary Variable Tested |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **base** (1000/150, k=4) | 32% (8/25) | 0.52 | 10/25 | 5/5 | Baseline parameters |
| **k=3** | 32% (8/25) | — | — | — | Retrieval depth reduction |
| **k=6** | 36% (9/25) | — | — | — | Retrieval depth expansion |
| **norefs** | 28% (7/25) | — | — | — | Drop bibliography chunks (Failed: removed helpful semantic signals) |
| **c500** | 32% (8/25) | — | — | — | Chunk granularity (Failed: stripped surrounding context) |
| **c1500** | 24% (6/25) | — | — | — | Truncation limit test (Failed: hit 256-token limit of embedding model) |

---

## Detailed Failure Mode Diagnostics

### 1. The Diagram Trap (Ingestion Extraction Limit)
* **Question:** #23 - "How is the AI lifecycle structured according to Figure 2?"
* **Top-1 Retrieved:** `nist_ai_rmf.pdf p.5`
* **Diagnosis:** The ground-truth lifecycle phases reside inside a graphical flowchart on page 37. Because `pypdf` extracts only raw text streams, the structural data inside the graphic was never indexed. 

### 2. The Vocabulary Gap (Embedding Representational Limit)
* **Question:** #11 - "How do we make sure an algorithm isn't racist or sexist?"
* **Top-1 Retrieved:** `nist_adversarial_ml.pdf p.84`
* **Diagnosis:** NIST uses the formal terminology "harmful bias." The semantic distance between casual vernacular ("racist") and administrative taxonomy ("harmful bias") was too large for `all-MiniLM-L6-v2` to traverse, resulting in keyword trapping.

### 3. The Needle-in-the-Haystack Degradation
* **Question:** #12 - "What happens if a company ignores a deployed model for too long?"
* **Top-1 Retrieved:** `nist_adversarial_ml.pdf p.39`
* **Diagnosis:** The NIST framework categorizes post-deployment neglect under "model drift". The query latched onto generic tokens ("deployed model"), pulling adversarial ML chunks discussing backdoor durability instead.
