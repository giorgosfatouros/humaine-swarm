# Data injection

## Purpose

The **data injection** script populates the **Pinecone** vector index used by the RAG tool **get_docs**. Source content comes from:

- **`.docs/`** — PDFs (project deliverables, Kubeflow documentation)
- **`.md_files/`** — Markdown files (API references, AL/XAI guides, HAIC docs, etc.)
- **`.files-for-rag/`** — Staging drop zone for new files before they are moved to `.docs` / `.md_files` after a successful ingest

Run it once or whenever the corpus changes so the assistant can retrieve up-to-date documentation.

## Staging workflow (`.files-for-rag/`)

1. Add new PDFs and/or `.md` files to **`.files-for-rag/`**
2. Ingest from staging (see commands below)
3. On success, move files to the permanent archive:
   - `mv .files-for-rag/*.pdf .docs/`
   - `mv .files-for-rag/*.md .md_files/`

See also [`.files-for-rag/README.md`](../../.files-for-rag/README.md).

## Pipeline

```mermaid
flowchart LR
    PDFs[".docs PDFs"] --> LoadPdf[PyPDFLoader]
  MD[".md_files markdown"] --> LoadMd[UTF-8 load]
    LoadPdf --> ChunkPdf[SemanticChunker]
    LoadMd --> ChunkMd[MarkdownHeaderTextSplitter]
    ChunkPdf --> Embed[OpenAI Embeddings]
    ChunkMd --> Embed
    Embed --> Upsert[Pinecone upsert]
```

1. **Load PDFs**: **`load_pdf_documents()`** walks the PDF directory. Each PDF is loaded with **LangChain `PyPDFLoader`**; pages are concatenated per file.
2. **Load markdown**: **`load_markdown_documents()`** walks the markdown directory. Each `.md` file is read as UTF-8; title is taken from the first `#` heading when present.
3. **IDs**: **`doc_id`** is derived from the filename stem (ASCII-safe). Markdown vector ids use prefix `md_{doc_id}_chunk_{i}`; PDF ids use `{doc_id}_chunk_{i}`.
4. **Chunk**:
   - **PDF**: **SemanticChunker** (gradient breakpoints, `min_chunk_size=3000` characters, `breakpoint_threshold_amount=0.8`).
   - **Markdown**: **MarkdownHeaderTextSplitter** on `#` / `##` / `###`, with **RecursiveCharacterTextSplitter** fallback (1200 / 150) for long sections. Chunk text is prefixed with `[MD | {title} | {section}]`.
5. **Re-ingest hygiene**: Before upserting a markdown document, existing vectors with the same `doc_id` and `source=md` are deleted. Legacy `source=haic` vectors are removed once when ingesting markdown.
6. **Embed**: **OpenAIEmbeddings** (`text-embedding-3-small`).
7. **Upsert**: Batch upsert (size 32) with metadata: `title`, `text`, `source` (`pdf` | `md`), `doc_id`, `topic` (semantic category from `DOC_TOPIC_BY_DOC_ID`).

## When to run

- After adding or updating PDFs in **`.docs`**.
- After adding or updating markdown in **`.md_files`**.
- After dropping a new batch in **`.files-for-rag/`** (ingest from staging, then move).
- One-off setup for a new environment using index name **humaine**.

From project root:

```bash
# Ingest both sources (default, from .docs and .md_files)
python data_injection.py

# Ingest a new batch from staging
mkdir -p .docs .md_files
python data_injection.py --sources pdf,md --docs-dir .files-for-rag --md-dir .files-for-rag

# Markdown only (faster while editing .md_files)
python data_injection.py --sources md

# PDFs only
python data_injection.py --sources pdf
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--sources` | `pdf,md` | Comma-separated: `pdf`, `md` |
| `--docs-dir` | `.docs` | PDF source directory |
| `--md-dir` | `.md_files` | Markdown source directory |

## Environment

- **PINECONE_API_KEY**: Required for Pinecone client and index operations.
- **OpenAI API key**: Required for embeddings (e.g. `OPENAI_API_KEY`). Uses **langchain_openai.embeddings.OpenAIEmbeddings**.

Index name is **humaine** (same as **PINECONE_INDEX** in [utils/config.py](../../utils/config.py) used by the chat app at runtime).
