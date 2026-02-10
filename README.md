# KAG POC (text + code + SQL)

This is a minimal, offline proof-of-concept that builds a Knowledge-Augmented Graph (KAG)
from Markdown, Python/Java code, and PostgreSQL DDL.

## What it does
- Ingests files under `data/`
- Extracts terms, entities, and basic relations
- Produces a small property-graph-like JSON output

## Inputs
- `data/text/*.md` (Markdown docs)
- `data/code/*.py` (Python)
- `data/code/*.java` (Java)
- `data/sql/*.sql` (PostgreSQL DDL, inspired by common sample DBs like Northwind)

## Run
```bash
python3 /Users/family/Downloads/kag_poc/scripts/kag_poc.py
```

Enable NER + LLM extraction:
```bash
python3 /Users/family/Downloads/kag_poc/scripts/kag_poc.py --ner --llm
```

Outputs:
- `output/nodes.json`
- `output/edges.json`
- `output/graph.json`
- `output/report.md`

## Query a term
```bash
python3 /Users/family/Downloads/kag_poc/scripts/query_graph.py --term Customer
```

## Notes
- This is a pure-stdlib implementation (no external dependencies).
- The extraction is heuristic and intended only for MVP/POC exploration.
- For production: plug in a proper NLP pipeline, entity linking, and a graph DB.

## LLM config (optional)
Set these env vars to enable LLM relation extraction:
- `KAG_LLM_ENDPOINT`
- `KAG_LLM_API_KEY`
- `KAG_LLM_MODEL` (default: `gpt-4o-mini`)

## spaCy (optional)
If you enable `--ner`, install spaCy + a model:
```bash
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm
```
