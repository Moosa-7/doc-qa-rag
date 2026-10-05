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

## Generation Evaluation Checkpoint (llama3.2:3b)

### Answerable Questions (Synthesis & Citation Test)
1. "What are the four core functions of the AI RMF?"
* Result: Success - Perfectly formatted the four functions into a clean numbered list and cited correctly.
2. "What is the definition of Explainability and Interpretability?"
* Result: Success - Synthesized both concepts cleanly into a single sentence using "respectively" and cited correctly.
3. "What are the specific risks of data poisoning?"
* Result: Success - Found the correct information buried in chunk #4 and synthesized it into clear bullet points.
4. "Who is in charge of overseeing model safety in a corporation?"
* Result: Fail (Overly Cautious) - The model responded "I don't know." The context mentioned "roles and responsibilities" being defined, but since no specific job title was explicitly named in the text, the 3B model conservatively refused to guess. 
5. "What is the environmental footprint of training these systems?"
* Result: Success - Accurately extracted concrete examples like energy and water consumption.

### Unanswerable Questions (Prompt Adherence Test)
6. "Who won the 2022 FIFA World Cup?"
* Result: Refused - Successfully replied "I don't know" despite its pre-trained knowledge. No knowledge bleed.

7. "What is the recipe for a traditional Italian carbonara?"
* Result: Refused - Successfully replied "I don't know". 

## Generation Analysis
* **Synthesis Quality:** Excellent. `llama3.2:3b` defaults to clean lists and concise sentences rather than walls of text. It successfully extracts data even when it is located at the very bottom of the context window.
* **Citation Accuracy:** Flawless. It reliably appends the source metadata provided by the retriever.
* **Boundary Adherence:** Highly resilient to hallucinations. The strict system prompt successfully prevents pre-trained knowledge bleed, making the model safely default to "I don't know" for both out-of-domain questions and ambiguous in-domain text.

## Experiment Predictions

1. **k3 / k6 (Varying Context Window Depth):**
   * *Prediction:* No re-indexing needed. Higher k (k=6) will increase hit@k slightly (we saw hit@6 was 36% vs hit@4 at 32%), but risks diluting context for llama3.2:3b with lower-scoring irrelevant chunks. Lower k (k=3) saves context space but increases false refusals on distributed answers.
2. **norefs (FILTER_REFS=1):**
   * *Prediction:* Dropping citation-dense chunks removes false lexical attractors (like URLs containing keywords). Expect paraphrase retrieval to improve or stay equal, with fewer irrelevant chunks crowding out real content in questions like #11.
3. **c500 (CHUNK_SIZE=500, CHUNK_OVERLAP=75):**
   * *Prediction:* Smaller chunks yield tighter semantic density per vector. direct factoid retrieval (e.g., definitions) should improve because the embedding isn't averaged over extraneous surrounding sentences. However, multi_page questions may degrade if context is split across chunk boundaries.
4. **c1500 (CHUNK_SIZE=1500, CHUNK_OVERLAP=200):**
   * *Prediction:* **The Truncation Trap.** all-MiniLM-L6-v2 has a hard sequence limit of 256 wordpiece tokens (~1,000 characters). In a 1,500-character chunk, characters ~1,001 through 1,500 are silently truncated during embedding. The vector only indexes the first two-thirds of the chunk, causing retrieval to miss information positioned toward the end of large text blocks.
