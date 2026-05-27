# RAG staging directory

Drop new PDFs and markdown files here before indexing them into Pinecone.

`README.md` in this directory is workflow documentation only and is **not** indexed.

## Workflow

1. Add `.pdf` and/or `.md` files to this directory.
2. From the project root, with `PINECONE_API_KEY` and `OPENAI_API_KEY` set:

```bash
mkdir -p .docs .md_files

poetry run python data_injection.py \
  --sources pdf,md \
  --docs-dir .files-for-rag \
  --md-dir .files-for-rag
```

3. After a successful run, move files to the local archive (gitignored):

```bash
mv .files-for-rag/*.pdf .docs/
mv .files-for-rag/*.md .md_files/
```

4. This directory should be empty and ready for the next batch.

## Re-ingest after edits

If you change files already in `.docs` or `.md_files`:

```bash
poetry run python data_injection.py --sources pdf,md
```

See [docs/developers/05-data-injection.md](../docs/developers/05-data-injection.md) for pipeline details.
