#!/usr/bin/env python3
"""Query a small KAG graph by node label."""

import argparse
import json
import os
from typing import Dict, List


def load_graph(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["nodes"], data["edges"]


def build_index(nodes) -> Dict[str, List[dict]]:
    index: Dict[str, List[dict]] = {}
    for node in nodes:
        key = node["label"].lower()
        index.setdefault(key, []).append(node)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a KAG graph by label")
    parser.add_argument("--graph", default=os.path.join(os.path.dirname(__file__), "..", "output", "graph.json"))
    parser.add_argument("--term", required=True)
    args = parser.parse_args()

    nodes, edges = load_graph(os.path.abspath(args.graph))
    index = build_index(nodes)

    term_key = args.term.lower()
    if term_key not in index:
        print("No node found for term:", args.term)
        return

    node_ids = {n["id"] for n in index[term_key]}
    id_to_node = {n["id"]: n for n in nodes}

    print(f"Matches for '{args.term}':")
    for n in index[term_key]:
        print(f"- {n['id']} {n['type']}: {n['label']}")

    print("\nEdges:")
    for edge in edges:
        if edge["source"] in node_ids or edge["target"] in node_ids:
            src = id_to_node[edge["source"]]
            tgt = id_to_node[edge["target"]]
            print(f"- {src['type']}:{src['label']} --{edge['type']}--> {tgt['type']}:{tgt['label']}")


if __name__ == "__main__":
    main()
