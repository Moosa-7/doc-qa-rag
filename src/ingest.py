import json
from pypdf import PdfReader
from src.config import PDF_DIR, CHUNKS_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def extract_pages(pdf_path):
    """Yield (page_number, cleaned_text) for each non-empty page."""
    reader = PdfReader(str(pdf_path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())  # collapse newlines/extra spaces
        if len(text) > 50:             # skip blank or near-empty pages
            yield page_number, text


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character windows."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def main():
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    chunk_id = 0
    with CHUNKS_PATH.open("w") as out:
        for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
            for page_number, text in extract_pages(pdf_path):
                for piece in chunk_text(text):
                    record = {
                        "id": chunk_id,
                        "source": pdf_path.name,
                        "page": page_number,
                        "text": piece,
                    }
                    out.write(json.dumps(record) + "\n")
                    chunk_id += 1
            print(f"Processed {pdf_path.name}")
    print(f"Total chunks: {chunk_id}")


if __name__ == "__main__":
    main()