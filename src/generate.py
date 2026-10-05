import sys
import ollama
from src.retrieve import Retriever

LLM_MODEL = "llama3.2:3b"

PROMPT_TEMPLATE = """You answer questions using ONLY the context below.
Cite the sources you use with their bracket numbers, like [1] or [2].
If the answer is not in the context, reply exactly: I don't know.

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question, chunks):
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"[{i}] ({c['source']}, p.{c['page']})\n{c['text']}")
    return PROMPT_TEMPLATE.format(context="\n\n".join(blocks), question=question)


def generate(prompt, backend="ollama"):
    if backend == "ollama":
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        return response["message"]["content"]
    raise NotImplementedError(f"Backend {backend} not added yet")


def answer(question, retriever, k=4, backend="ollama"):
    chunks = retriever.search(question, k=k)
    text = generate(build_prompt(question, chunks), backend=backend)
    return {"answer": text, "sources": chunks}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    result = answer(question, Retriever())
    print(result["answer"])
    print("\nSources:")
    for i, c in enumerate(result["sources"], start=1):
        print(f"[{i}] {c['source']} p.{c['page']}")