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

## Open-source libraries to consider

### Term extraction and NLP
- **spaCy**: production-grade NLP/NER pipeline (already optional in this POC).
- **Stanza**: strong multilingual tokenization, POS, and dependency parsing.
- **KeyBERT**: keyword/term extraction based on transformer embeddings.
- **YAKE**: lightweight unsupervised keyword extraction.

### Relation extraction
- **REBEL (Babelscape/rebel-large)**: seq2seq relation extraction model that returns triples.
- **OpenNRE**: toolkit for supervised relation extraction.
- **textacy**: rule/pattern-based relation extraction helpers on top of spaCy.
- **[oie-resources](https://github.com/gkiril/oie-resources)**: curated Open Information Extraction tools, datasets, and papers for bootstrapping OIE experiments.

### Entity linking and ontology
- **BLINK / REL**: entity linking pipelines for grounding mentions to canonical entities.
- **Owlready2**: work with OWL ontologies directly from Python.
- **RDFLib**: RDF/OWL graph creation and querying.
- **pySHACL**: SHACL validation for ontology/graph constraints.

### Knowledge-graph platform / ontology workflow
- **[OpenSPG](https://github.com/OpenSPG/openspg)**: full-stack open-source semantic graph platform (ontology, extraction, graph build, applications).
- Useful if you want to move from a script-level POC to a managed KG pipeline with schema-first governance.

### Knowledge graph storage and query
- **Neo4j (community) + py2neo/neo4j driver**: property graph storage and Cypher queries.
- **ArangoDB**: multi-model document+graph database.
- **Apache Jena / Fuseki**: RDF triplestore and SPARQL endpoint.

### Hybrid orchestration pattern (recommended)
1. Rules + parser-based extraction for high-precision edges.
2. ML/LLM extraction for long-tail relations.
3. Entity linking + ontology validation (SHACL/OWL constraints).
4. Persist into graph DB and run consistency checks.

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
