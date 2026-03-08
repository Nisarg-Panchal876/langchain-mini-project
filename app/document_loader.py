from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# ── Configuration ────────────────────────────────────────────────────────────
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "company_docs.pdf")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 130

# ── Text Splitter ─────────────────────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],  # Semantic boundary priority
    length_function=len,
)


def load_and_split_pdf(pdf_path: str = PDF_PATH) -> list:
    """
    Load a PDF using PyMuPDFLoader and split it into overlapping text chunks.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A list of LangChain Document objects (chunks) with page metadata.

    Raises:
        FileNotFoundError: If the PDF does not exist at the given path.
    """
    abs_path = os.path.abspath(pdf_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"PDF not found at: {abs_path}\n"
            "Please place your company documentation PDF at:\n"
            "  D:\\langchain-mini-project\\data\\company_docs.pdf"
        )

    print(f"[Loader] Loading PDF from: {abs_path}")
    loader = PyMuPDFLoader(abs_path)
    raw_documents = loader.load()
    print(f"[Loader] Pages loaded: {len(raw_documents)}")

    chunks = text_splitter.split_documents(raw_documents)
    print(f"[Splitter] Total chunks created: {len(chunks)}")
    print(f"[Splitter] Chunk size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP}")

    return chunks


if __name__ == "__main__":
    # Quick smoke-test — run:  python -m app.document_loader
    chunks = load_and_split_pdf()
    print(f"\nSample chunk [0]:\n{'-'*60}")
    print(chunks[0].page_content)
    print(f"\nMetadata: {chunks[0].metadata}")
