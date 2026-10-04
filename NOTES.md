# Retrieval Evaluation Checkpoint

**Model used:** `all-MiniLM-L6-v2`
**Index type:** FAISS `IndexFlatIP`
**Documents:** NIST AI RMF, GenAI Profile, and Adversarial ML

## Results Summary
* **Accuracy:** 6/10 questions retrieved the correct passage in the top 4 results.

## Question Log & Analysis

1. "What are the four core functions of the AI RMF?"
* Result: Success
* Notes: Perfect match (Score 0.792). The chunking preserved the exact introductory paragraph defining GOVERN, MAP, MEASURE, and MANAGE.

2. "How does Executive Order 14110 relate to Generative AI?"
* Result: Success
* Notes: Found the exact definition (Score 0.473). The confidence score dropped because it relied on the highly specific keyword "EO 14110" rather than broad semantic overlap, but it still surfaced the correct chunk.

3. "What is the definition of Explainability and Interpretability?"
* Result: Success
* Notes: Clean semantic and keyword match (Score 0.655). The chunk clearly contrasts both terms exactly as requested.

4. "How is the AI lifecycle structured?"
* Result: Fail
* Notes: The pipeline retrieved references to "Figure 2," but missed the actual stages because `pypdf` cannot extract visual text from diagrams.

5. "What are the specific risks of data poisoning?"
* Result: Success
* Notes: The correct answer was found in the 4th position (Score 0.512) from the Adversarial ML document. Generic words like "risks" artificially inflated the scores of irrelevant chunks above it.

6. "Who is in charge of overseeing model safety in a corporation?"
* Result: Success
* Notes: Excellent semantic search win. The model understood that "in charge of" means "roles and responsibilities" and successfully routed to the GOVERN function without literal string matches.

7. "How do we make sure an algorithm isn't racist or sexist?"
* Result: Fail
* Notes: Caught in the "Bibliography Trap." The academic text uses "harmful bias," but the model fell back to literal keyword matching for "racist" and pulled citations from the bibliography instead of the actual mitigation guidelines.

8. "What happens if a company ignores a deployed model for too long?"
* Result: Fail
* Notes: Failed to bridge the semantic gap between "ignores a deployed model" and the formal terminology "drift" or "degradation" used in the MANAGE function.

9. "Does this framework apply to chatbots differently than traditional machine learning?"
* Result: Fail
* Notes: Caught in the "Keyword Attractor Trap." The model searched for "chatbots" and found footnotes instead of mapping the concept to "Generative AI" or "Foundation Models" which are the focus of the GenAI Profile.

10. "What is the environmental footprint of training these systems?"
* Result: Success
* Notes: Strong semantic mapping (Score 0.570). Successfully linked "environmental footprint" to "energy and water consumption" and "environmental impacts."

## Misses & Guesses (Why did the retrieval fail?)

* **Guess 1 (Diagram Trap):** Basic text extractors like `pypdf` ignore visual data. Critical structural information embedded in flowcharts (like the AI lifecycle) is lost entirely during ingestion.
* **Guess 2 (Vocabulary Gap / Semantic Limits):** `all-MiniLM-L6-v2` is a lightweight model (384 dimensions). While fast, it struggles to bridge wide semantic gaps between casual phrasing (e.g., "ignores a deployed model", "chatbots") and formal government taxonomies (e.g., "model drift", "Generative AI"). 
* **Guess 3 (The Bibliography Trap):** When semantic similarity fails, the model falls back to literal keyword matching. This often pulls in URLs or citations from the references section that contain the keyword but provide zero contextual value for the LLM. 
