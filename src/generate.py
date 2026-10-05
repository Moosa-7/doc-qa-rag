import json
import os
import sys
import urllib.request

MIN_SCORE = float(os.getenv("MIN_SCORE", "0.32"))   # 0 = disabled; set after calibration
BACKEND = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b") # verify in the Groq console


def build_prompt(question, chunks):
    blocks = [
        f"[{i}] ({c['source']}, p.{c['page']})\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    ]
    context = "\n\n".join(blocks)
    return f"""You are a strict technical assistant. Answer using ONLY the context below.
Cite the sources you use with their bracket numbers, like [1] or [2].
If the context does not contain the answer, reply exactly: I don't know.
Do not invent information.

Context:
{context}

Question: {question}
Answer:"""


def generate(prompt, backend=None):
    backend = backend or BACKEND

    if backend == "ollama":
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=json.dumps({
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0},
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))["response"]

    if backend == "groq":
        from groq import Groq   # imported here so local use doesn't need it
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        extra = {}
        if GROQ_MODEL.startswith("openai/gpt-oss"):
            extra["reasoning_effort"] = "low"   # keep the hidden thinking short
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1500,
            **extra,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("The model returned an empty answer (possibly hit the token limit).")
        return text


def answer(question, retriever, k=4, backend=None):
    chunks = retriever.search(question, k=k)
    if not chunks or chunks[0]["score"] < MIN_SCORE:
        return {
            "answer": "I don't know. I couldn't find anything relevant to that in the indexed documents. "
                      "Try asking about the content of the NIST AI publications.",
            "sources": [],
        }
    text = generate(build_prompt(question, chunks), backend=backend)
    return {"answer": text, "sources": chunks}


if __name__ == "__main__":
    from src.retrieve import Retriever

    question = " ".join(sys.argv[1:])
    result = answer(question, Retriever())
    print(result["answer"])
    print("\nSources:")
    for i, c in enumerate(result["sources"], start=1):
        print(f"[{i}] {c['source']} p.{c['page']}")