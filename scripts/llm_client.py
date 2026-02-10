#!/usr/bin/env python3
"""Minimal OpenAI-compatible LLM client for relation extraction."""

import json
import os
import re
import urllib.request
from typing import List, Optional, Tuple


def _extract_json_block(text: str) -> Optional[str]:
    text = text.strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        return text
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if match:
        return match.group(1)
    return None


def _parse_triples(payload: str) -> List[Tuple[str, str, str]]:
    data = json.loads(payload)
    triples: List[Tuple[str, str, str]] = []

    if isinstance(data, dict):
        items = data.get("triples") or data.get("relations") or []
    else:
        items = data

    for item in items:
        if not isinstance(item, dict):
            continue
        head = str(item.get("head", "")).strip()
        rel = str(item.get("relation", "")).strip()
        tail = str(item.get("tail", "")).strip()
        if head and rel and tail:
            triples.append((head, rel, tail))
    return triples


def extract_relations(text: str, terms: List[str], max_chars: int = 4000) -> List[Tuple[str, str, str]]:
    endpoint = os.getenv("KAG_LLM_ENDPOINT")
    if not endpoint:
        return []

    api_key = os.getenv("KAG_LLM_API_KEY", "")
    model = os.getenv("KAG_LLM_MODEL", "gpt-4o-mini")

    clipped = text.strip()
    if len(clipped) > max_chars:
        clipped = clipped[:max_chars] + "\n[TRUNCATED]"

    term_list = ", ".join(sorted(set(t for t in terms if t)))

    system = (
        "You extract knowledge graph relations. "
        "Return JSON only: a list of objects with keys head, relation, tail."
    )
    user = (
        "Text:\n" + clipped + "\n\n" +
        "Known terms (if helpful):\n" + term_list + "\n\n" +
        "Output JSON only. Example: "
        "[{\"head\":\"Customer\",\"relation\":\"is_a\",\"tail\":\"Entity\"}]"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return []

    try:
        response = json.loads(raw)
        content = response["choices"][0]["message"]["content"]
    except Exception:
        return []

    block = _extract_json_block(content)
    if not block:
        return []

    try:
        return _parse_triples(block)
    except Exception:
        return []
