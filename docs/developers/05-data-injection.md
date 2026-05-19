# Data injection

## Purpose

The **data injection** script populates the **Pinecone** vector index used by the RAG tool **get_docs**. Source content comes from:

- **`.docs/`** — PDFs (project / Kubeflow documentation)
- **`.md_files/`** — Markdown files (eg HAIC benchmarking documentation)

Run it once or whenever either directory changes so the assistant can retrieve up-to-date documentation.

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

1. **Load PDFs**: **`load_pdf_documents()`** walks **`.docs`**. Each PDF is loaded with **LangChain `PyPDFLoader`**; pages are concatenated per file.
2. **Load HAIC markdown**: **`load_markdown_documents()`** walks **`.md_files`**. Each `.md` file is read as UTF-8; title is taken from the first `#` heading when present.
3. **IDs**: **`doc_id`** is derived from the filename stem (ASCII-safe). HAIC vector ids use prefix `haic_{doc_id}_chunk_{i}`; PDF ids use `{doc_id}_chunk_{i}`.
4. **Chunk**:
   - **PDF**: **SemanticChunker** (gradient breakpoints, `min_chunk_size=100`, `breakpoint_threshold_amount=0.8`).
   - **HAIC**: **MarkdownHeaderTextSplitter** on `#` / `##` / `###`, with **RecursiveCharacterTextSplitter** fallback (1200 / 150) for long sections. Chunk text is prefixed with `[HAIC | {title} | {section}]`.
5. **Re-ingest hygiene**: Before upserting a HAIC document, existing vectors with the same `doc_id` and `source=haic` are deleted from Pinecone.
6. **Embed**: **OpenAIEmbeddings** (`text-embedding-3-small`).
7. **Upsert**: Batch upsert (size 32) with metadata: `title`, `text`, `source` (`pdf` | `haic`), `doc_id`, `topic` (HAIC only: overview, metrics, interpretation, schema, platform).

## When to run

- After adding or updating PDFs in **`.docs`**.
- After adding or updating HAIC markdown in **`.md_files`**.
- One-off setup for a new environment using index name **humaine**.

From project root:

```bash
# Ingest both sources (default)
python data_injection.py

# HAIC markdown only (faster while editing .md_files)
python data_injection.py --sources haic

# PDFs only
python data_injection.py --sources pdf
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--sources` | `pdf,haic` | Comma-separated: `pdf`, `haic` |
| `--docs-dir` | `.docs` | PDF source directory |
| `--md-dir` | `.md_files` | HAIC markdown source directory |

## Environment

- **PINECONE_API_KEY**: Required for Pinecone client and index operations.
- **OpenAI API key**: Required for embeddings (e.g. `OPENAI_API_KEY`). Uses **langchain_openai.embeddings.OpenAIEmbeddings**.

Index name is **humaine** (same as **PINECONE_INDEX** in [utils/config.py](../../utils/config.py) used by the chat app at runtime).
