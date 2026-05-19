import argparse
import logging
import os
import re
import time
from typing import List, Optional

import pandas as pd
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

from utils.helper_functions import setup_logging

logger = setup_logging("Data Injection", level=logging.INFO)

EMBED_MODEL = "text-embedding-3-small"
INDEX_NAME = "humaine"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
embedding_model = OpenAIEmbeddings(model=EMBED_MODEL)

HAIC_TOPIC_BY_DOC_ID = {
    "01_haic_overview": "overview",
    "02_haic_metrics_reference": "metrics",
    "03_metric_interpretation_guide": "interpretation",
    "04_logging_schema": "schema",
    "05_benchmark_suite_platform": "platform",
}

MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def sanitize_doc_id(title: str) -> str:
    doc_id = "".join(char for char in title if ord(char) < 128)
    return re.sub(r"[^a-zA-Z0-9-]", "_", doc_id)


def title_from_markdown(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return fallback.replace("_", " ").title()


def process_data(
    sources: Optional[List[str]] = None,
    docs_dir: str = ".docs",
    md_dir: str = ".md_files",
):
    sources = sources or ["pdf", "haic"]
    logger.info("Starting data processing (sources=%s)", sources)

    df = get_deliverables(sources=sources, docs_dir=docs_dir, md_dir=md_dir)
    if df.empty:
        logger.warning("No data to process")
        return

    save_to_pinecone(df)
    logger.info("Data processing completed successfully")


def get_embeddings(text_list) -> List[List[float]]:
    try:
        return embedding_model.embed_documents(text_list)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return []


def load_pdf_documents(docs_dir: str = ".docs") -> pd.DataFrame:
    if not os.path.exists(docs_dir):
        logger.warning("Directory %s does not exist", docs_dir)
        return pd.DataFrame()

    from langchain_community.document_loaders import PyPDFLoader

    data = []
    for root, _, files in os.walk(docs_dir):
        for file in files:
            if not file.lower().endswith(".pdf"):
                continue
            file_path = os.path.join(root, file)
            try:
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                content = " ".join(page.page_content for page in pages)
                title = os.path.splitext(file)[0]
                doc_id = sanitize_doc_id(title)
                data.append(
                    {
                        "title": title,
                        "text": content,
                        "source": "pdf",
                        "doc_id": doc_id,
                    }
                )
            except Exception as e:
                logger.error("Error processing file %s: %s", file_path, e)

    if not data:
        logger.warning("No PDF documents found in %s", docs_dir)
    return pd.DataFrame(data)


def load_markdown_documents(md_dir: str = ".md_files") -> pd.DataFrame:
    if not os.path.exists(md_dir):
        logger.warning("Directory %s does not exist", md_dir)
        return pd.DataFrame()

    data = []
    for root, _, files in os.walk(md_dir):
        for file in sorted(files):
            if not file.lower().endswith(".md"):
                continue
            file_path = os.path.join(root, file)
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                stem = os.path.splitext(file)[0]
                doc_id = sanitize_doc_id(stem)
                title = title_from_markdown(content, stem)
                data.append(
                    {
                        "title": title,
                        "text": content,
                        "source": "haic",
                        "doc_id": doc_id,
                    }
                )
            except Exception as e:
                logger.error("Error processing file %s: %s", file_path, e)

    if not data:
        logger.warning("No markdown documents found in %s", md_dir)
    return pd.DataFrame(data)


def get_deliverables(
    sources: Optional[List[str]] = None,
    docs_dir: str = ".docs",
    md_dir: str = ".md_files",
) -> pd.DataFrame:
    sources = sources or ["pdf", "haic"]
    frames = []

    try:
        if "pdf" in sources:
            pdf_df = load_pdf_documents(docs_dir)
            if not pdf_df.empty:
                frames.append(pdf_df)

        if "haic" in sources:
            md_df = load_markdown_documents(md_dir)
            if not md_df.empty:
                frames.append(md_df)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df["id"] = df["doc_id"]

        chunked_data = []
        for _, row in df.iterrows():
            if row["source"] == "haic":
                delete_haic_vectors_for_doc(row["doc_id"])
                chunked_data.extend(chunk_haic_markdown(row))
            else:
                chunked_data.extend(chunk_pdf_documents(row))

        if not chunked_data:
            return pd.DataFrame()

        chunked_df = pd.DataFrame(chunked_data)
        logger.info("Generating embeddings for %d chunks...", len(chunked_df))
        embeddings = get_embeddings(chunked_df["chunk_text"].tolist())
        if not embeddings or len(embeddings) != len(chunked_df):
            logger.error("Embedding count mismatch")
            return pd.DataFrame()
        chunked_df["embeddings"] = embeddings
        return chunked_df

    except Exception as e:
        logger.error("Error in get_deliverables: %s", e)
        return pd.DataFrame()


def chunk_pdf_documents(row) -> list:
    chunked_data = []
    text_splitter = SemanticChunker(
        embedding_model,
        breakpoint_threshold_type="gradient",
        min_chunk_size=100,
        breakpoint_threshold_amount=0.8,
    )
    text = row["text"]
    if not text:
        return chunked_data

    try:
        docs = text_splitter.create_documents([text])
        for i, doc in enumerate(docs):
            chunk_id = f"{row['doc_id']}_chunk_{i}".replace(" ", "_")
            chunked_data.append(
                {
                    "id": chunk_id,
                    "chunk_text": doc.page_content,
                    "title": row["title"],
                    "source": row["source"],
                    "doc_id": row["doc_id"],
                    "topic": "",
                }
            )
    except Exception as e:
        logger.error("Error chunking PDF document %s: %s", row["doc_id"], e)
    return chunked_data


def chunk_haic_markdown(row) -> list:
    chunked_data = []
    text = row["text"]
    if not text:
        return chunked_data

    topic = HAIC_TOPIC_BY_DOC_ID.get(row["doc_id"], "")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS, strip_headers=False
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150
    )

    try:
        header_docs = header_splitter.split_text(text)
        for i, doc in enumerate(header_docs):
            section = " > ".join(
                v for k, v in sorted(doc.metadata.items()) if v
            ) or "document"
            body = doc.page_content.strip()
            if len(body) > 1200:
                sub_docs = fallback_splitter.create_documents([body])
                for j, sub in enumerate(sub_docs):
                    chunk_text = format_haic_chunk(row["title"], section, sub.page_content)
                    chunk_id = f"haic_{row['doc_id']}_chunk_{i}_{j}"
                    chunked_data.append(
                        {
                            "id": chunk_id,
                            "chunk_text": chunk_text,
                            "title": row["title"],
                            "source": "haic",
                            "doc_id": row["doc_id"],
                            "topic": topic,
                        }
                    )
            else:
                chunk_text = format_haic_chunk(row["title"], section, body)
                chunk_id = f"haic_{row['doc_id']}_chunk_{i}"
                chunked_data.append(
                    {
                        "id": chunk_id,
                        "chunk_text": chunk_text,
                        "title": row["title"],
                        "source": "haic",
                        "doc_id": row["doc_id"],
                        "topic": topic,
                    }
                )
    except Exception as e:
        logger.error("Error chunking HAIC document %s: %s", row["doc_id"], e)
    return chunked_data


def format_haic_chunk(title: str, section: str, body: str) -> str:
    return f"[HAIC | {title} | {section}]\n{body}"


def delete_haic_vectors_for_doc(doc_id: str) -> None:
    try:
        index = pc.Index(INDEX_NAME)
        index.delete(filter={"doc_id": doc_id, "source": "haic"})
        logger.info("Deleted existing HAIC vectors for doc_id=%s", doc_id)
    except Exception as e:
        logger.warning(
            "Could not delete HAIC vectors for %s (may not exist yet): %s",
            doc_id,
            e,
        )


def save_to_pinecone(df):
    try:
        spec = ServerlessSpec(cloud="aws", region="us-east-1")
        check_and_create_index(INDEX_NAME, len(df["embeddings"].iloc[0]), spec)

        index = pc.Index(INDEX_NAME)
        time.sleep(1)
        logger.info(index.describe_index_stats())

        batch_size = 32
        for i in tqdm(range(0, len(df), batch_size)):
            i_end = min(i + batch_size, len(df))
            ids_batch = df["id"].iloc[i:i_end].tolist()

            metadata = [
                {
                    "title": df.loc[j, "title"],
                    "text": df.loc[j, "chunk_text"],
                    "source": df.loc[j, "source"],
                    "doc_id": df.loc[j, "doc_id"],
                    "topic": df.loc[j, "topic"] or "",
                }
                for j in range(i, i_end)
            ]

            to_upsert = zip(ids_batch, df["embeddings"].iloc[i:i_end].tolist(), metadata)
            index.upsert(vectors=list(to_upsert))

    except Exception as e:
        logger.error(f"Error saving to Pinecone: {e}")


def check_and_create_index(index_name, dimension, spec):
    try:
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                index_name,
                dimension=dimension,
                metric="dotproduct",
                spec=spec,
            )
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise


def parse_sources(value: str) -> List[str]:
    sources = [s.strip().lower() for s in value.split(",") if s.strip()]
    valid = {"pdf", "haic"}
    invalid = set(sources) - valid
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid sources: {invalid}. Use pdf and/or haic."
        )
    if not sources:
        raise argparse.ArgumentTypeError("At least one source is required.")
    return sources


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Injection Script")
    parser.add_argument(
        "--sources",
        type=parse_sources,
        default="pdf,haic",
        help="Comma-separated sources to ingest: pdf, haic (default: pdf,haic)",
    )
    parser.add_argument(
        "--docs-dir",
        default=".docs",
        help="Directory containing PDF documents (default: .docs)",
    )
    parser.add_argument(
        "--md-dir",
        default=".md_files",
        help="Directory containing HAIC markdown documents (default: .md_files)",
    )
    args = parser.parse_args()

    process_data(sources=args.sources, docs_dir=args.docs_dir, md_dir=args.md_dir)
