import json
import sys
from pathlib import Path

from graphify.validate import validate_extraction


FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_semantic_local_auto_returns_valid_graph(tmp_path):
    from graphify.semantic_local import extract_semantic

    doc1 = tmp_path / "design.md"
    doc1.write_text(
        "# Cache design\n"
        "We use a shared cache because latency matters and repeated fetches are expensive.\n"
        "This trade-off reduces network cost.\n",
        encoding="utf-8",
    )
    doc2 = tmp_path / "notes.md"
    doc2.write_text(
        "# Architecture notes\n"
        "The caching layer reduces latency and avoids repeated requests to the API.\n"
        "This is why the service keeps a local store.\n",
        encoding="utf-8",
    )

    result = extract_semantic([doc1, doc2], backend="auto")

    assert validate_extraction(result) == []
    assert result["nodes"], "expected at least one semantic node"
    assert any(e["relation"] in {"semantically_similar_to", "conceptually_related_to", "rationale_for"} for e in result["edges"])


def test_main_accepts_path_as_implicit_scan_command(tmp_path, monkeypatch):
    from graphify.__main__ import main

    sample_py = tmp_path / "sample.py"
    sample_py.write_text(
        "def fetch_data():\n"
        "    return 42\n\n"
        "def main():\n"
        "    return fetch_data()\n",
        encoding="utf-8",
    )
    sample_md = tmp_path / "README.md"
    sample_md.write_text(
        "# Demo project\n\n"
        "The fetch layer exists because startup latency matters.\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["graphify", ".", "--semantic-backend", "none", "--no-viz"])

    main()

    out_dir = tmp_path / "graphify-out"
    assert (out_dir / "graph.json").exists()
    assert (out_dir / "GRAPH_REPORT.md").exists()

    graph = json.loads((out_dir / "graph.json").read_text(encoding="utf-8"))
    assert graph["nodes"], "expected graph.json to contain nodes"


def test_main_supports_ollama_flag_in_ast_only_mode(tmp_path, monkeypatch, capsys):
    from graphify.__main__ import main

    (tmp_path / "module.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["graphify", ".", "--semantic-backend", "ollama", "--ollama-model", "llama3.2", "--no-viz"],
    )

    main()

    captured = capsys.readouterr()
    assert "ollama" in captured.out.lower() or "ollama" in captured.err.lower()
    assert (tmp_path / "graphify-out" / "graph.json").exists()


def test_main_prints_progress_to_console(tmp_path, monkeypatch, capsys):
    from graphify.__main__ import main

    (tmp_path / "module.py").write_text("def hello():\n    return 'hi'\n", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["graphify", ".", "--semantic-backend", "none", "--no-viz"])

    main()

    captured = capsys.readouterr()
    assert "[graphify] Step 1/5" in captured.out
    assert "[graphify] Step 4/5" in captured.out
    assert "module.py" in captured.out
    assert "Processing file:" in captured.out or "AST [1/1]" in captured.out
