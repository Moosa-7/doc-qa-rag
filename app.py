try:
    import spaces   # must be imported before torch or gradio on ZeroGPU Spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        """Never called. ZeroGPU Spaces require at least one @spaces.GPU function."""
        return None
except ImportError:
    pass   # running locally, where the spaces package isn't installed

import gradio as gr

from src.generate import answer
from src.retrieve import Retriever

MAX_QUESTION_CHARS = 500

retriever = Retriever()   # load the embedding model and index once at startup


def format_sources(chunks):
    if not chunks:
        return "_No sources retrieved._"
    parts = []
    for i, c in enumerate(chunks, start=1):
        snippet = c["text"][:400].replace("\n", " ")
        parts.append(
            f"**[{i}] {c['source']}, p.{c['page']}** (similarity {c['score']:.2f})\n\n> {snippet}..."
        )
    return "\n\n".join(parts)


def ask(question, k):
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""
    if len(question) > MAX_QUESTION_CHARS:
        return f"Please keep your question under {MAX_QUESTION_CHARS} characters.", ""
    try:
        result = answer(question, retriever, k=int(k))
    except Exception as e:
        return (
            f"The language model is unavailable ({e}). "
            "Check that Ollama is running (or your API key and rate limit if using Groq).",
            "",
        )
    return result["answer"], format_sources(result["sources"])


with gr.Blocks(title="Document Q&A (RAG)") as demo:
    gr.Markdown(
        "# Document Q&A\n"
        "Ask questions about the indexed NIST AI documents. "
        "Answers use only the retrieved passages and cite them as [1], [2]..."
    )
    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(label="Your question", lines=2)
            k = gr.Slider(1, 8, value=4, step=1, label="Passages to retrieve (k)")
            button = gr.Button("Ask", variant="primary")
            gr.Markdown("### Answer")
            answer_box = gr.Markdown()
        with gr.Column(scale=3):
            gr.Markdown("### Retrieved sources")
            sources_box = gr.Markdown()

    gr.Examples(
        examples=[
            ["What are the core functions of the AI Risk Management Framework?"],
            ["Who won the 2022 World Cup?"],
        ],
        inputs=question,
    )

    button.click(ask, inputs=[question, k], outputs=[answer_box, sources_box])
    question.submit(ask, inputs=[question, k], outputs=[answer_box, sources_box])

if __name__ == "__main__":
    demo.launch()
