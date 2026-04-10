"""Standalone graphify pipeline for local terminal use.

This module wires together detect -> extract -> semantic -> build -> cluster ->
analyze -> report/export so `graphify .` works without requiring a hosted agent
platform. Local semantic enrichment can use Ollama or a torch-backed model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .analyze import god_nodes, suggest_questions, surprising_connections
from .build import build
from .cache import check_semantic_cache, save_semantic_cache
from .cluster import cluster, score_all
from .detect import count_words, detect, save_manifest
from .export import to_graphml, to_html, to_json, to_obsidian, to_svg
from .extract import extract
from .report import generate
from .semantic_local import empty_extraction, extract_semantic
from .wiki import to_wiki


def _log(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def _filter_detection_to_target(detection: dict, target: Path) -> dict:
    if not target.is_file():
        return detection
    filtered = {kind: [] for kind in detection.get("files", {})}
    target_resolved = target.resolve()
    for kind, file_list in detection.get("files", {}).items():
        filtered[kind] = [f for f in file_list if Path(f).resolve() == target_resolved]
    return {
        **detection,
        "files": filtered,
        "total_files": sum(len(v) for v in filtered.values()),
        "total_words": count_words(target),
        "warning": None,
    }


def _combine_extractions(*parts: dict) -> dict:
    combined = {"nodes": [], "edges": [], "hyperedges": [], "input_tokens": 0, "output_tokens": 0}
    for part in parts:
        combined["nodes"].extend(part.get("nodes", []))
        combined["edges"].extend(part.get("edges", []))
        combined["hyperedges"].extend(part.get("hyperedges", []))
        combined["input_tokens"] += int(part.get("input_tokens", 0) or 0)
        combined["output_tokens"] += int(part.get("output_tokens", 0) or 0)
    return combined


def run_pipeline(
    target: str | Path = ".",
    *,
    semantic_backend: str = "auto",
    ollama_model: str | None = None,
    ollama_host: str = "http://127.0.0.1:11434",
    local_model: str | None = None,
    device: str | None = None,
    no_viz: bool = False,
    graphml: bool = False,
    svg: bool = False,
    wiki: bool = False,
    obsidian: bool = False,
    obsidian_dir: str | None = None,
    follow_symlinks: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full graphify pipeline locally and write outputs to graphify-out/."""
    path = Path(target).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"graphify target does not exist: {path}")

    root = path if path.is_dir() else path.parent
    out_dir = root / "graphify-out"
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(verbose, f"[graphify] Target: {path}")
    _log(verbose, "[graphify] Step 1/5 · Detecting supported files...")
    detection = _filter_detection_to_target(detect(path if path.is_dir() else root, follow_symlinks=follow_symlinks), path)
    (out_dir / ".graphify_detect.json").write_text(json.dumps(detection, indent=2), encoding="utf-8")
    _log(
        verbose,
        "[graphify]   found "
        f"{detection.get('total_files', 0)} files "
        f"({len(detection.get('files', {}).get('code', []))} code, "
        f"{len(detection.get('files', {}).get('document', []))} docs, "
        f"{len(detection.get('files', {}).get('paper', []))} papers, "
        f"{len(detection.get('files', {}).get('image', []))} images)",
    )

    code_files = [Path(f) for f in detection.get("files", {}).get("code", [])]
    semantic_files: list[Path] = []
    for kind in ("document", "paper", "image"):
        semantic_files.extend(Path(f) for f in detection.get("files", {}).get(kind, []))
    if not semantic_files:
        # Code-only repos can still benefit from local semantic similarity/rationale extraction.
        semantic_files = list(code_files)

    _log(verbose, f"[graphify] Step 2/5 · Running AST extraction on {len(code_files)} code file(s)...")

    def _ast_progress(path_obj: Path, index: int, total: int, status: str) -> None:
        try:
            shown = path_obj.relative_to(root)
        except ValueError:
            shown = path_obj
        cache_suffix = " [cached]" if status == "cached" else ""
        _log(verbose, f"[graphify]   AST [{index}/{total}] {shown}{cache_suffix}")

    ast_result = extract(code_files, progress_callback=_ast_progress) if code_files else empty_extraction(backend="ast")
    _log(
        verbose,
        f"[graphify]   AST produced {len(ast_result.get('nodes', []))} nodes and {len(ast_result.get('edges', []))} edges",
    )

    _log(verbose, f"[graphify] Step 3/5 · Running semantic extraction via backend '{semantic_backend}'...")
    if semantic_backend.lower() not in {"none", "off", "disabled"}:
        cacheable_files = [p for p in semantic_files if p not in code_files]
        uncached = [str(p) for p in semantic_files]
        cached_nodes: list[dict] = []
        cached_edges: list[dict] = []
        cached_hyperedges: list[dict] = []
        if cacheable_files:
            cached_nodes, cached_edges, cached_hyperedges, cached_uncached = check_semantic_cache(
                [str(p) for p in cacheable_files],
                root=root,
            )
            uncached = cached_uncached + [str(p) for p in semantic_files if p in code_files]
        _log(
            verbose,
            f"[graphify]   semantic cache: {len(semantic_files) - len(uncached)} hit(s), {len(uncached)} file(s) to analyze",
        )
        cached_semantic = {
            "nodes": cached_nodes,
            "edges": cached_edges,
            "hyperedges": cached_hyperedges,
            "input_tokens": 0,
            "output_tokens": 0,
            "backend": "cache",
            "warnings": [],
        }
        def _semantic_progress(path_obj: Path, index: int, total: int, resolved_backend: str, file_type: str) -> None:
            try:
                shown = path_obj.relative_to(root)
            except ValueError:
                shown = path_obj
            _log(verbose, f"[graphify]   Semantic [{index}/{total}] Processing file: {shown} ({file_type}, backend={resolved_backend})")

        new_semantic = extract_semantic(
            [Path(p) for p in uncached],
            backend=semantic_backend,
            model=local_model,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            device=device,
            progress_callback=_semantic_progress,
        )
        if cacheable_files and (new_semantic.get("nodes") or new_semantic.get("edges") or new_semantic.get("hyperedges")):
            cacheable_paths = {str(p) for p in cacheable_files}
            save_semantic_cache(
                [n for n in new_semantic.get("nodes", []) if n.get("source_file") in cacheable_paths],
                [e for e in new_semantic.get("edges", []) if e.get("source_file") in cacheable_paths],
                [h for h in new_semantic.get("hyperedges", []) if h.get("source_file") in cacheable_paths],
                root=root,
            )
        semantic_result = _combine_extractions(cached_semantic, new_semantic)
        semantic_result["backend"] = new_semantic.get("backend", semantic_backend)
        semantic_result["warnings"] = cached_semantic.get("warnings", []) + new_semantic.get("warnings", [])
    else:
        semantic_result = empty_extraction(backend="none")

    _log(
        verbose,
        f"[graphify]   semantic backend resolved to '{semantic_result.get('backend', semantic_backend)}' "
        f"with {len(semantic_result.get('nodes', []))} nodes and {len(semantic_result.get('edges', []))} edges",
    )

    (out_dir / ".graphify_ast.json").write_text(json.dumps(ast_result, indent=2), encoding="utf-8")
    (out_dir / ".graphify_semantic.json").write_text(json.dumps(semantic_result, indent=2), encoding="utf-8")

    merged = _combine_extractions(ast_result, semantic_result)
    (out_dir / ".graphify_extract.json").write_text(json.dumps(merged, indent=2), encoding="utf-8")

    _log(verbose, "[graphify] Step 4/5 · Building graph, clustering communities, and generating the report...")
    G = build([ast_result, semantic_result])
    communities = cluster(G)
    cohesion = score_all(G, communities)
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    labels = {cid: f"Community {cid}" for cid in communities}
    questions = suggest_questions(G, communities, labels)

    _log(
        verbose,
        f"[graphify]   graph contains {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, and {len(communities)} communities",
    )

    report = generate(
        G,
        communities,
        cohesion,
        labels,
        gods,
        surprises,
        detection,
        {"input": merged.get("input_tokens", 0), "output": merged.get("output_tokens", 0)},
        str(path),
        suggested_questions=questions,
    )
    (out_dir / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")

    _log(verbose, "[graphify] Step 5/5 · Writing outputs to graphify-out/...")
    to_json(G, communities, str(out_dir / "graph.json"))

    generated: list[str] = [str(out_dir / "graph.json"), str(out_dir / "GRAPH_REPORT.md")]
    _log(verbose, f"[graphify]   wrote {out_dir / 'graph.json'}")
    _log(verbose, f"[graphify]   wrote {out_dir / 'GRAPH_REPORT.md'}")
    warnings = list(semantic_result.get("warnings", []))

    if not no_viz:
        try:
            to_html(G, communities, str(out_dir / "graph.html"), community_labels=labels)
            generated.append(str(out_dir / "graph.html"))
            _log(verbose, f"[graphify]   wrote {out_dir / 'graph.html'}")
        except Exception as exc:
            warnings.append(f"HTML export skipped: {exc}")

    if graphml:
        try:
            to_graphml(G, str(out_dir / "graph.graphml"))
            generated.append(str(out_dir / "graph.graphml"))
            _log(verbose, f"[graphify]   wrote {out_dir / 'graph.graphml'}")
        except Exception as exc:
            warnings.append(f"GraphML export skipped: {exc}")

    if svg:
        try:
            to_svg(G, communities, str(out_dir / "graph.svg"), community_labels=labels)
            generated.append(str(out_dir / "graph.svg"))
            _log(verbose, f"[graphify]   wrote {out_dir / 'graph.svg'}")
        except Exception as exc:
            warnings.append(f"SVG export skipped: {exc}")

    if wiki:
        try:
            wiki_dir = out_dir / "wiki"
            to_wiki(G, communities, wiki_dir, community_labels=labels, cohesion=cohesion, god_nodes_data=gods)
            generated.append(str(wiki_dir / "index.md"))
            _log(verbose, f"[graphify]   wrote {wiki_dir / 'index.md'}")
        except Exception as exc:
            warnings.append(f"Wiki export skipped: {exc}")

    if obsidian:
        try:
            vault_dir = Path(obsidian_dir).expanduser() if obsidian_dir else (out_dir / "obsidian")
            to_obsidian(G, communities, str(vault_dir), community_labels=labels, cohesion=cohesion)
            generated.append(str(vault_dir))
            _log(verbose, f"[graphify]   wrote {vault_dir}")
        except Exception as exc:
            warnings.append(f"Obsidian export skipped: {exc}")

    try:
        save_manifest(detection.get("files", {}), manifest_path=str(out_dir / "manifest.json"))
    except Exception as exc:
        warnings.append(f"Manifest save skipped: {exc}")

    return {
        "graph": G,
        "communities": communities,
        "cohesion": cohesion,
        "labels": labels,
        "gods": gods,
        "surprises": surprises,
        "questions": questions,
        "detection": detection,
        "ast_result": ast_result,
        "semantic_result": semantic_result,
        "merged_extraction": merged,
        "out_dir": out_dir,
        "generated": generated,
        "warnings": warnings,
        "semantic_backend": semantic_result.get("backend", semantic_backend),
    }


__all__ = ["run_pipeline"]
