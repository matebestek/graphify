"""Local semantic extraction helpers for Ollama and torch-backed models.

This module keeps graphify usable outside Claude/Codex by providing a small,
local-first semantic layer for docs, papers, images, and optionally code.

Backend strategy:
- ``ollama``: call a local Ollama server and request JSON graph fragments.
- ``torch``: try a local Hugging Face/transformers model; if unavailable,
  fall back to sentence-transformer embeddings for semantic similarity.
- ``auto``: prefer Ollama, then torch, then heuristic extraction.
- ``none``: disable semantic extraction entirely.

Even without optional model packages, the heuristic fallback still emits a
schema-valid extraction with rationale and semantic-similarity edges.
"""
from __future__ import annotations

import base64
import json
import math
import re
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from .detect import (
    FileType,
    classify_file,
    docx_to_markdown,
    extract_pdf_text,
    xlsx_to_markdown,
)
from .extract import _make_id

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "by", "for", "from",
    "how", "if", "in", "into", "is", "it", "its", "of", "on", "or", "that", "the",
    "their", "this", "to", "uses", "using", "via", "we", "when", "why", "with", "you",
}

_REASON_CUES = (
    "because", "so that", "in order to", "trade-off", "tradeoff", "why", "reason",
    "motivation", "therefore", "to avoid", "to reduce", "to improve", "due to",
    "so we", "so the", "keeps", "allows", "helps",
)


def empty_extraction(*, backend: str = "none", warnings: list[str] | None = None) -> dict:
    return {
        "nodes": [],
        "edges": [],
        "hyperedges": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "backend": backend,
        "warnings": warnings or [],
    }


def _resolve_backend(backend: str, ollama_host: str) -> str:
    choice = (backend or "auto").lower()
    if choice in {"none", "off", "disabled"}:
        return "none"
    if choice == "heuristic":
        return "heuristic"
    if choice == "ollama":
        return "ollama"
    if choice == "torch":
        return "torch"
    if _ollama_available(ollama_host):
        return "ollama"
    try:
        import sentence_transformers  # noqa: F401
        return "torch"
    except Exception:
        pass
    try:
        import transformers  # noqa: F401
        return "torch"
    except Exception:
        return "heuristic"


def _ollama_available(host: str) -> bool:
    url = host.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=1.5) as response:
            return 200 <= response.status < 300
    except Exception:
        return False


def _read_semantic_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf_text(path)
    if ext == ".docx":
        return docx_to_markdown(path)
    if ext == ".xlsx":
        return xlsx_to_markdown(path)
    if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _file_type_value(path: Path) -> str:
    ftype = classify_file(path)
    return ftype.value if ftype else FileType.DOCUMENT.value


def _file_node(path: Path, file_type: str) -> dict:
    return {
        "id": _make_id(path.stem),
        "label": path.name,
        "file_type": file_type,
        "source_file": str(path),
        "source_location": None,
    }


def _clean_label(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip("`*_#-:,.()[]{} \n\t"))
    return text[:100].strip()


def _candidate_terms(text: str, path: Path) -> list[str]:
    candidates: list[str] = []

    for heading in re.findall(r"^\s{0,3}#{1,6}\s+(.+)$", text, flags=re.MULTILINE):
        label = _clean_label(heading)
        if len(label) >= 3:
            candidates.append(label)

    for code_term in re.findall(r"`([^`]{2,60})`", text):
        label = _clean_label(code_term)
        if len(label) >= 3:
            candidates.append(label)

    for titled in re.findall(r"\b[A-Z][A-Za-z0-9_./:-]{2,}\b", text):
        label = _clean_label(titled)
        if len(label) >= 3 and label.lower() not in _STOPWORDS:
            candidates.append(label)

    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{3,}\b", text.lower())
    counts = Counter(w for w in words if w not in _STOPWORDS)
    for word, _ in counts.most_common(12):
        candidates.append(word.replace("_", " "))

    # Keep unique terms while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for candidate in candidates:
        label = _clean_label(candidate)
        key = label.lower()
        if len(label) < 3 or key in seen:
            continue
        seen.add(key)
        deduped.append(label)
        if len(deduped) >= 8:
            break

    if not deduped:
        deduped.append(path.stem.replace("_", " "))
    return deduped


def _rationale_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text)
    pieces = re.split(r"(?<=[.!?])\s+", text)
    out: list[str] = []
    for sentence in pieces:
        lower = sentence.lower()
        if any(cue in lower for cue in _REASON_CUES):
            cleaned = sentence.strip()
            if len(cleaned) >= 20:
                out.append(cleaned[:220])
        if len(out) >= 3:
            break
    return out


def _heuristic_extract_one(path: Path, text: str, file_type: str) -> tuple[dict, dict[str, Any]]:
    file_node = _file_node(path, file_type)
    nodes = [file_node]
    edges: list[dict] = []

    concepts = _candidate_terms(text, path)
    concept_ids: list[str] = []
    for idx, concept in enumerate(concepts[:6], start=1):
        nid = _make_id(path.stem, concept)
        concept_ids.append(nid)
        nodes.append(
            {
                "id": nid,
                "label": concept,
                "file_type": file_type,
                "source_file": str(path),
                "source_location": None,
            }
        )
        relation = "mentions" if idx == 1 else "conceptually_related_to"
        edges.append(
            {
                "source": file_node["id"],
                "target": nid,
                "relation": relation,
                "confidence": "EXTRACTED",
                "confidence_score": 1.0,
                "source_file": str(path),
                "source_location": None,
                "weight": 1.0,
            }
        )

    for idx, sentence in enumerate(_rationale_sentences(text), start=1):
        rid = _make_id(path.stem, "rationale", str(idx))
        nodes.append(
            {
                "id": rid,
                "label": sentence,
                "file_type": "rationale",
                "source_file": str(path),
                "source_location": None,
            }
        )
        target = concept_ids[0] if concept_ids else file_node["id"]
        edges.append(
            {
                "source": rid,
                "target": target,
                "relation": "rationale_for",
                "confidence": "EXTRACTED",
                "confidence_score": 1.0,
                "source_file": str(path),
                "source_location": None,
                "weight": 1.0,
            }
        )

    keywords = {re.sub(r"\s+", " ", c.lower()) for c in concepts}
    return {
        "nodes": nodes,
        "edges": edges,
        "hyperedges": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }, {"file_node": file_node, "keywords": keywords, "text": text[:4000]}


def _json_from_text(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _extraction_prompt(path: Path, file_type: str, text: str) -> str:
    excerpt = text[:12000]
    return (
        "You are a local graph extraction model for graphify. "
        "Return ONLY valid JSON with this schema: "
        "{\"nodes\":[{\"id\":str,\"label\":str,\"file_type\":str,\"source_file\":str}],"
        "\"edges\":[{\"source\":str,\"target\":str,\"relation\":str,\"confidence\":str,\"source_file\":str,\"confidence_score\":float,\"weight\":float}],"
        "\"hyperedges\":[],\"input_tokens\":0,\"output_tokens\":0}. "
        "Extract concepts, rationale, citations, and non-obvious semantic relationships. "
        "Use confidence values EXTRACTED, INFERRED, or AMBIGUOUS. "
        f"File: {path}\n"
        f"Type: {file_type}\n\n"
        f"Content:\n{excerpt}"
    )


def _call_ollama(path: Path, file_type: str, text: str, model: str, host: str) -> dict:
    url = host.rstrip("/") + "/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "user",
                "content": _extraction_prompt(path, file_type, text),
            }
        ],
    }
    if file_type == FileType.IMAGE.value:
        payload["messages"][0]["images"] = [base64.b64encode(path.read_bytes()).decode("ascii")]

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    message = body.get("message", {}).get("content", "{}")
    return _json_from_text(message)


class _TorchGenerator:
    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self._generator = None

    def available(self) -> bool:
        if self._generator is not None:
            return True
        try:
            from transformers import pipeline
        except Exception:
            return False
        kwargs: dict[str, Any] = {"model": self.model_name}
        if self.device:
            kwargs["device"] = self.device
        try:
            self._generator = pipeline("text-generation", **kwargs)
            return True
        except Exception:
            return False

    def extract(self, path: Path, file_type: str, text: str) -> dict:
        if not self.available():
            raise RuntimeError("transformers pipeline is not available")
        prompt = _extraction_prompt(path, file_type, text)
        outputs = self._generator(
            prompt,
            max_new_tokens=600,
            do_sample=False,
            return_full_text=False,
        )
        generated = outputs[0]["generated_text"] if outputs else "{}"
        return _json_from_text(generated)


def _normalize_llm_result(path: Path, file_type: str, raw: dict) -> tuple[dict, dict[str, Any]]:
    file_node = _file_node(path, file_type)
    nodes = []
    seen_ids: set[str] = set()
    for node in raw.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node.setdefault("source_file", str(path))
        node.setdefault("file_type", file_type)
        node.setdefault("source_location", None)
        if node.get("id") and node["id"] not in seen_ids:
            seen_ids.add(node["id"])
            nodes.append(node)
    if file_node["id"] not in seen_ids:
        nodes.insert(0, file_node)
        seen_ids.add(file_node["id"])

    edges = []
    for edge in raw.get("edges", []):
        if not isinstance(edge, dict):
            continue
        edge.setdefault("source_file", str(path))
        edge.setdefault("source_location", None)
        edge.setdefault("confidence", "INFERRED")
        edge.setdefault("confidence_score", 0.7 if edge.get("confidence") == "INFERRED" else 1.0)
        edge.setdefault("weight", 1.0)
        if edge.get("source") in seen_ids and edge.get("target") in seen_ids:
            edges.append(edge)

    # If the model produced nodes but no connecting edge, link the file node to the first few concepts.
    concept_nodes = [n for n in nodes if n["id"] != file_node["id"]][:4]
    if concept_nodes and not any(e.get("source") == file_node["id"] for e in edges):
        for node in concept_nodes:
            edges.append(
                {
                    "source": file_node["id"],
                    "target": node["id"],
                    "relation": "conceptually_related_to",
                    "confidence": "INFERRED",
                    "confidence_score": 0.72,
                    "source_file": str(path),
                    "source_location": None,
                    "weight": 1.0,
                }
            )

    keywords = {n["label"].lower() for n in concept_nodes if n.get("label")}
    return {
        "nodes": nodes,
        "edges": edges,
        "hyperedges": raw.get("hyperedges", []),
        "input_tokens": raw.get("input_tokens", 0),
        "output_tokens": raw.get("output_tokens", 0),
    }, {"file_node": file_node, "keywords": keywords, "text": _read_semantic_text(path)[:4000]}


def _similarity_score(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_keys = set(left.get("keywords", set()))
    right_keys = set(right.get("keywords", set()))
    if left_keys and right_keys:
        overlap = len(left_keys & right_keys) / max(len(left_keys | right_keys), 1)
        if overlap > 0:
            return overlap
    left_words = {w for w in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{3,}\b", left.get("text", "").lower()) if w not in _STOPWORDS}
    right_words = {w for w in re.findall(r"\b[a-zA-Z][a-zA-Z0-9_-]{3,}\b", right.get("text", "").lower()) if w not in _STOPWORDS}
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / max(len(left_words | right_words), 1)


def _embedding_similarity(contexts: list[dict[str, Any]], model_name: str | None = None) -> list[float] | None:
    texts = [ctx.get("text", "") for ctx in contexts]
    if len(texts) < 2 or not any(texts):
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    model = SentenceTransformer(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.encode(texts, normalize_embeddings=True)
    sims: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(float(vectors[i] @ vectors[j]))
    return sims


def _add_similarity_edges(contexts: list[dict[str, Any]], output: dict, *, embedding_model: str | None = None) -> None:
    if len(contexts) < 2:
        return

    pair_scores: dict[tuple[int, int], float] = {}
    emb_scores = _embedding_similarity(contexts, embedding_model)
    score_iter = iter(emb_scores or [])

    for i in range(len(contexts)):
        for j in range(i + 1, len(contexts)):
            heuristic = _similarity_score(contexts[i], contexts[j])
            score = next(score_iter, heuristic)
            score = max(score, heuristic)
            if score < 0.18:
                continue
            confidence = max(0.6, min(0.95, 0.55 + score))
            left_node = contexts[i]["file_node"]
            right_node = contexts[j]["file_node"]
            output["edges"].append(
                {
                    "source": left_node["id"],
                    "target": right_node["id"],
                    "relation": "semantically_similar_to",
                    "confidence": "INFERRED",
                    "confidence_score": round(confidence, 2),
                    "source_file": left_node["source_file"],
                    "source_location": None,
                    "weight": round(max(1.0, confidence * 1.2), 2),
                }
            )
            pair_scores[(i, j)] = score


def _dedupe(output: dict) -> dict:
    seen_nodes: set[str] = set()
    deduped_nodes: list[dict] = []
    for node in output.get("nodes", []):
        nid = node.get("id")
        if nid and nid not in seen_nodes:
            seen_nodes.add(nid)
            deduped_nodes.append(node)

    seen_edges: set[tuple[str, str, str, str]] = set()
    deduped_edges: list[dict] = []
    for edge in output.get("edges", []):
        key = (
            str(edge.get("source")),
            str(edge.get("target")),
            str(edge.get("relation")),
            str(edge.get("source_file")),
        )
        if key not in seen_edges and edge.get("source") in seen_nodes and edge.get("target") in seen_nodes:
            seen_edges.add(key)
            deduped_edges.append(edge)

    output["nodes"] = deduped_nodes
    output["edges"] = deduped_edges
    output.setdefault("hyperedges", [])
    return output


def extract_semantic(
    paths: list[Path],
    *,
    backend: str = "auto",
    model: str | None = None,
    ollama_model: str | None = None,
    ollama_host: str = "http://127.0.0.1:11434",
    device: str | None = None,
    progress_callback: Any | None = None,
) -> dict:
    """Extract semantic graph fragments using a local backend or heuristics.

    The returned dict always conforms to the standard graphify extraction
    schema and never requires a network call unless ``backend='ollama'`` or
    ``backend='auto'`` successfully selects a local Ollama server.
    """
    file_list = [Path(p) for p in paths if Path(p).exists()]
    resolved = _resolve_backend(backend, ollama_host)
    output = empty_extraction(backend=resolved)

    if resolved == "none" or not file_list:
        return output

    contexts: list[dict[str, Any]] = []
    warnings: list[str] = []
    torch_generator = _TorchGenerator(model or "distilgpt2", device=device) if resolved == "torch" else None

    total = len(file_list)
    for index, path in enumerate(file_list, start=1):
        file_type = _file_type_value(path)
        if progress_callback is not None:
            progress_callback(path, index, total, resolved, file_type)
        text = _read_semantic_text(path)

        # Images need a vision-capable Ollama model; heuristic fallback still adds a node.
        raw: dict | None = None
        if resolved == "ollama":
            chosen_model = ollama_model or ("llava" if file_type == FileType.IMAGE.value else "llama3.2")
            try:
                raw = _call_ollama(path, file_type, text, chosen_model, ollama_host)
            except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
                warnings.append(f"Ollama fallback for {path.name}: {exc}")
        elif resolved == "torch" and torch_generator is not None:
            try:
                raw = torch_generator.extract(path, file_type, text)
            except Exception as exc:
                warnings.append(f"Torch fallback for {path.name}: {exc}")

        if raw is not None:
            normalized, ctx = _normalize_llm_result(path, file_type, raw)
        else:
            normalized, ctx = _heuristic_extract_one(path, text, file_type)

        output["nodes"].extend(normalized.get("nodes", []))
        output["edges"].extend(normalized.get("edges", []))
        output["hyperedges"].extend(normalized.get("hyperedges", []))
        output["input_tokens"] += int(normalized.get("input_tokens", 0) or 0)
        output["output_tokens"] += int(normalized.get("output_tokens", 0) or 0)
        contexts.append(ctx)

    _add_similarity_edges(contexts, output, embedding_model=model)
    output["warnings"] = warnings
    return _dedupe(output)


__all__ = ["extract_semantic", "empty_extraction"]
