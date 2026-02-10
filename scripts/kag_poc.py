#!/usr/bin/env python3
"""Build a small Knowledge-Augmented Graph (KAG) from mixed sources."""

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import spacy
except Exception:
    spacy = None

from llm_client import extract_relations as llm_extract_relations


@dataclass
class Node:
    id: int
    type: str
    label: str
    attrs: Dict[str, str] = field(default_factory=dict)


@dataclass
class Edge:
    source: int
    target: int
    type: str
    attrs: Dict[str, str] = field(default_factory=dict)


class Graph:
    def __init__(self) -> None:
        self._next_id = 1
        self._nodes_by_key: Dict[str, Node] = {}
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self._edge_keys: Set[Tuple[int, str, int]] = set()

    def _key(self, node_type: str, label: str, scope: Optional[str] = None) -> str:
        base = re.sub(r"\s+", "_", label.strip().lower())
        if scope:
            return f"{node_type}:{base}:{scope}"
        return f"{node_type}:{base}"

    def add_node(self, node_type: str, label: str, scope: Optional[str] = None, **attrs: str) -> int:
        key = self._key(node_type, label, scope)
        if key in self._nodes_by_key:
            return self._nodes_by_key[key].id
        node = Node(id=self._next_id, type=node_type, label=label, attrs=dict(attrs))
        self._next_id += 1
        self._nodes_by_key[key] = node
        self.nodes.append(node)
        return node.id

    def add_edge(self, source: int, target: int, edge_type: str, **attrs: str) -> None:
        edge_key = (source, edge_type, target)
        if edge_key in self._edge_keys:
            return
        self._edge_keys.add(edge_key)
        self.edges.append(Edge(source=source, target=target, type=edge_type, attrs=dict(attrs)))


@dataclass
class ExtractionContext:
    nlp: Optional[object]
    ner_enabled: bool
    llm_enabled: bool
    llm_calls_left: int
    spacy_model: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def normalize_term(term: str) -> str:
    return term.strip().strip(".:")


def extract_definition_relations(text: str) -> List[Tuple[str, str]]:
    relations: List[Tuple[str, str]] = []
    # English: "X is a Y"
    for match in re.finditer(r"\b([A-Z][\w\- ]{1,80})\s+is\s+(?:an?|the)\s+([A-Za-z][\w\- ]{1,80})", text):
        relations.append((normalize_term(match.group(1)), normalize_term(match.group(2))))
    # Russian: "X — это Y"
    for match in re.finditer(r"\b([^\n]{1,80})\s+—\s+это\s+([^\n\.]{1,80})", text):
        relations.append((normalize_term(match.group(1)), normalize_term(match.group(2))))
    # Explicit "is_a"
    for match in re.finditer(r"\b([A-Za-z_][\w\- ]{1,80})\s+is_a\s+([A-Za-z_][\w\- ]{1,80})", text):
        relations.append((normalize_term(match.group(1)), normalize_term(match.group(2))))
    return relations


def extract_part_of_relations(text: str) -> List[Tuple[str, str]]:
    relations: List[Tuple[str, str]] = []
    for match in re.finditer(r"\b([A-Za-z_][\w\- ]{1,80})\s+part_of\s+([A-Za-z_][\w\- ]{1,80})", text):
        relations.append((normalize_term(match.group(1)), normalize_term(match.group(2))))
    return relations


def extract_alias_relations(text: str) -> List[Tuple[str, str]]:
    relations: List[Tuple[str, str]] = []
    for match in re.finditer(r"\b([A-Za-zА-Яа-я0-9_ ]{1,80})\s+corresponds\s+to\s+([A-Za-zА-Яа-я0-9_ ]{1,80})", text, re.IGNORECASE):
        relations.append((normalize_term(match.group(1)), normalize_term(match.group(2))))
    for match in re.finditer(r"\b([A-Za-zА-Яа-я0-9_ ]{1,80})\s+соответствует\s+([A-Za-zА-Яа-я0-9_ ]{1,80})", text, re.IGNORECASE):
        relations.append((normalize_term(match.group(1)), normalize_term(match.group(2))))
    return relations


def extract_markdown_terms(text: str) -> Dict[str, List[str]]:
    terms: Dict[str, List[str]] = {"heading": [], "bold": [], "code": []}
    for line in text.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*)$", line)
        if heading:
            terms["heading"].append(normalize_term(heading.group(1)))
        for bold in re.findall(r"\*\*(.+?)\*\*", line):
            terms["bold"].append(normalize_term(bold))
        for code in re.findall(r"`([^`]+)`", line):
            terms["code"].append(normalize_term(code))
    return terms


def sanitize_relation(label: str) -> str:
    cleaned = label.strip().lower()
    cleaned = re.sub(r"[^a-z0-9_\\- ]+", "", cleaned)
    cleaned = cleaned.replace("-", "_")
    cleaned = re.sub(r"\\s+", "_", cleaned)
    return cleaned or "related_to"


def iter_files(root: str, exts: Iterable[str]) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in exts):
                yield os.path.join(dirpath, filename)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_spacy_model(models: List[str]) -> Tuple[Optional[object], Optional[str], Optional[str]]:
    if spacy is None:
        return None, None, "spaCy not installed"
    for model in models:
        model = model.strip()
        if not model:
            continue
        try:
            return spacy.load(model), model, None
        except Exception:
            continue
    return None, None, f"spaCy models not available: {', '.join(models)}"


def extract_ner_entities(graph: Graph, text: str, source_id: int, context: ExtractionContext) -> List[str]:
    if not context.ner_enabled or context.nlp is None:
        return []
    try:
        doc = context.nlp(text)
    except Exception:
        return []
    terms: List[str] = []
    for ent in doc.ents:
        label = normalize_term(ent.text)
        if not label:
            continue
        ent_id = graph.add_node("Entity", label, ner_label=ent.label_)
        graph.add_edge(source_id, ent_id, "mentions")
        terms.append(label)
    return terms


def extract_llm_relations(graph: Graph, text: str, terms: List[str], source_id: int, context: ExtractionContext) -> None:
    if not context.llm_enabled or context.llm_calls_left <= 0:
        return
    triples = llm_extract_relations(text, terms)
    if not triples:
        context.llm_calls_left -= 1
        return
    for head, rel, tail in triples:
        head_id = graph.add_node("Term", head)
        tail_id = graph.add_node("Term", tail)
        graph.add_edge(head_id, tail_id, sanitize_relation(rel), source="llm")
        graph.add_edge(source_id, head_id, "mentions")
        graph.add_edge(source_id, tail_id, "mentions")
    context.llm_calls_left -= 1


def extract_from_markdown(graph: Graph, path: str, context: ExtractionContext) -> None:
    text = read_text(path)
    file_id = graph.add_node("File", path, scope=path)

    terms = extract_markdown_terms(text)
    term_list: List[str] = []
    for heading in terms["heading"]:
        section_id = graph.add_node("Section", heading)
        graph.add_edge(section_id, file_id, "defined_in")

    for term in terms["bold"] + terms["code"]:
        term_list.append(term)
        term_id = graph.add_node("Term", term)
        graph.add_edge(file_id, term_id, "mentions")

    for left, right in extract_definition_relations(text):
        term_list.extend([left, right])
        left_id = graph.add_node("Term", left)
        right_id = graph.add_node("Term", right)
        graph.add_edge(left_id, right_id, "is_a")
        graph.add_edge(file_id, left_id, "defines")

    for left, right in extract_part_of_relations(text):
        term_list.extend([left, right])
        left_id = graph.add_node("Term", left)
        right_id = graph.add_node("Term", right)
        graph.add_edge(left_id, right_id, "part_of")

    for left, right in extract_alias_relations(text):
        term_list.extend([left, right])
        left_id = graph.add_node("Term", left)
        right_id = graph.add_node("Term", right)
        graph.add_edge(left_id, right_id, "same_as")

    ner_terms = extract_ner_entities(graph, text, file_id, context)
    term_list.extend(ner_terms)
    extract_llm_relations(graph, text, term_list, file_id, context)


def extract_from_python(graph: Graph, path: str, context: ExtractionContext) -> None:
    source = read_text(path)
    file_id = graph.add_node("File", path, scope=path)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    module_doc = ast.get_docstring(tree)
    if module_doc:
        extract_ner_entities(graph, module_doc, file_id, context)
        extract_llm_relations(graph, module_doc, [], file_id, context)

    class PythonExtractor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: List[int] = []
            self.func_stack: List[int] = []

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                mod_id = graph.add_node("Module", alias.name)
                graph.add_edge(file_id, mod_id, "imports")

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            mod = node.module or ""
            if mod:
                mod_id = graph.add_node("Module", mod)
                graph.add_edge(file_id, mod_id, "imports")

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            class_id = graph.add_node("Class", node.name)
            graph.add_edge(class_id, file_id, "defined_in")
            doc = ast.get_docstring(node)
            if doc:
                extract_ner_entities(graph, doc, class_id, context)
            self.class_stack.append(class_id)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            func_id = graph.add_node("Function", node.name)
            graph.add_edge(func_id, file_id, "defined_in")
            if self.class_stack:
                graph.add_edge(self.class_stack[-1], func_id, "defines_method")

            doc = ast.get_docstring(node)
            if doc:
                for left, right in extract_definition_relations(doc):
                    left_id = graph.add_node("Term", left)
                    right_id = graph.add_node("Term", right)
                    graph.add_edge(left_id, right_id, "is_a")
                    graph.add_edge(func_id, left_id, "defines")
                extract_ner_entities(graph, doc, func_id, context)

            self.func_stack.append(func_id)
            self.generic_visit(node)
            self.func_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            if func_name:
                callee_id = graph.add_node("Symbol", func_name)
                if self.func_stack:
                    graph.add_edge(self.func_stack[-1], callee_id, "calls")
                else:
                    graph.add_edge(file_id, callee_id, "calls")
            self.generic_visit(node)

    PythonExtractor().visit(tree)


def extract_from_java(graph: Graph, path: str, context: ExtractionContext) -> None:
    text = read_text(path)
    file_id = graph.add_node("File", path, scope=path)

    class_matches = re.finditer(r"\b(class|interface|enum)\s+(\w+)", text)
    classes: List[str] = [m.group(2) for m in class_matches]
    last_class: Optional[str] = classes[0] if classes else None

    for cls in classes:
        class_id = graph.add_node("Class", cls)
        graph.add_edge(class_id, file_id, "defined_in")

    for line in text.splitlines():
        class_match = re.search(r"\bclass\s+(\w+)", line)
        if class_match:
            last_class = class_match.group(1)

        method_match = re.search(r"\b(?:public|protected|private)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(", line)
        if method_match:
            method = method_match.group(1)
            method_id = graph.add_node("Method", method)
            graph.add_edge(method_id, file_id, "defined_in")
            if last_class:
                class_id = graph.add_node("Class", last_class)
                graph.add_edge(class_id, method_id, "defines_method")

    for match in re.finditer(r"\bnew\s+(\w+)", text):
        cls = match.group(1)
        cls_id = graph.add_node("Class", cls)
        graph.add_edge(file_id, cls_id, "instantiates")


def _split_columns(def_block: str) -> List[str]:
    lines = [line.strip() for line in def_block.split("\n") if line.strip()]
    cleaned = []
    for line in lines:
        if line.endswith(","):
            line = line[:-1]
        cleaned.append(line)
    return cleaned


def extract_from_sql(graph: Graph, path: str, context: ExtractionContext) -> None:
    text = read_text(path)
    file_id = graph.add_node("File", path, scope=path)

    for match in re.finditer(r"CREATE\s+TABLE\s+(\w+)\s*\((.*?)\);", text, re.IGNORECASE | re.DOTALL):
        table = match.group(1)
        body = match.group(2)
        table_id = graph.add_node("Table", table)
        graph.add_edge(table_id, file_id, "defined_in")

        for line in _split_columns(body):
            if line.upper().startswith("CONSTRAINT") or line.upper().startswith("PRIMARY KEY"):
                continue
            fk_match = re.search(r"FOREIGN\s+KEY\s*\((\w+)\)\s+REFERENCES\s+(\w+)\s*\((\w+)\)", line, re.IGNORECASE)
            if fk_match:
                col = fk_match.group(1)
                ref_table = fk_match.group(2)
                ref_col = fk_match.group(3)
                col_id = graph.add_node("Column", f"{table}.{col}")
                ref_id = graph.add_node("Column", f"{ref_table}.{ref_col}")
                graph.add_edge(col_id, ref_id, "fk")
                continue

            col_match = re.match(r"(\w+)\s+([A-Za-z0-9_()]+)", line)
            if col_match:
                col = col_match.group(1)
                col_type = col_match.group(2)
                col_id = graph.add_node("Column", f"{table}.{col}", data_type=col_type)
                graph.add_edge(table_id, col_id, "has_column")


def build_graph(input_root: str, context: ExtractionContext) -> Graph:
    graph = Graph()

    for path in iter_files(input_root, [".md"]):
        extract_from_markdown(graph, path, context)

    for path in iter_files(input_root, [".py"]):
        extract_from_python(graph, path, context)

    for path in iter_files(input_root, [".java"]):
        extract_from_java(graph, path, context)

    for path in iter_files(input_root, [".sql"]):
        extract_from_sql(graph, path, context)

    return graph


def write_outputs(graph: Graph, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    nodes_path = os.path.join(out_dir, "nodes.json")
    edges_path = os.path.join(out_dir, "edges.json")
    graph_path = os.path.join(out_dir, "graph.json")

    with open(nodes_path, "w", encoding="utf-8") as f:
        json.dump([node.__dict__ for node in graph.nodes], f, indent=2)

    with open(edges_path, "w", encoding="utf-8") as f:
        json.dump([edge.__dict__ for edge in graph.edges], f, indent=2)

    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump({"nodes": [node.__dict__ for node in graph.nodes], "edges": [edge.__dict__ for edge in graph.edges]}, f, indent=2)

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# KAG POC Report\n\n")
        f.write(f"Nodes: {len(graph.nodes)}\n\n")
        f.write(f"Edges: {len(graph.edges)}\n\n")
        f.write("## Sample nodes\n")
        for node in graph.nodes[:15]:
            f.write(f"- {node.type}: {node.label}\n")
        f.write("\n## Sample edges\n")
        for edge in graph.edges[:15]:
            f.write(f"- {edge.type}: {edge.source} -> {edge.target}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Knowledge-Augmented Graph (KAG) POC")
    parser.add_argument("--input", default=os.path.join(os.path.dirname(__file__), "..", "data"), help="Input root")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "output"), help="Output directory")
    parser.add_argument("--ner", action="store_true", help="Enable spaCy NER extraction")
    parser.add_argument("--llm", action="store_true", help="Enable LLM relation extraction")
    parser.add_argument("--llm-max-calls", type=int, default=5, help="Max LLM calls per run")
    parser.add_argument(
        "--spacy-models",
        default="en_core_web_sm,ru_core_news_sm,xx_ent_wiki_sm",
        help="Comma-separated spaCy model names to try",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.spacy_models.split(",") if m.strip()]
    nlp, model_name, model_err = load_spacy_model(models) if args.ner else (None, None, None)
    context = ExtractionContext(
        nlp=nlp,
        ner_enabled=args.ner and nlp is not None,
        llm_enabled=args.llm,
        llm_calls_left=max(args.llm_max_calls, 0),
        spacy_model=model_name,
    )
    if args.ner and nlp is None and model_err:
        context.notes.append(model_err)

    graph = build_graph(os.path.abspath(args.input), context)
    write_outputs(graph, os.path.abspath(args.out))
    print(f"Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"Output: {os.path.abspath(args.out)}")
    if context.notes:
        for note in context.notes:
            print(f"Note: {note}")


if __name__ == "__main__":
    main()
