# Architecture

## Ingestion
The ingestion pipeline reads Markdown, source code, and SQL DDL.
Each document becomes a File node; headings become Section nodes.

## Extraction
Terms are extracted from:
- bold text (**Term**)
- inline code (`Term`)
- definitions like "X is ..." or "X — это ..."

Relations are extracted from:
- definition patterns (is_a)
- part_of patterns (part_of)
- code structure (defines, calls)
- SQL foreign keys (fk)
