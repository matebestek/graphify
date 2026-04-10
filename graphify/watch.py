# monitor a folder and auto-trigger --update when files change
from __future__ import annotations
import json
import time
from pathlib import Path


from graphify.detect import CODE_EXTENSIONS, DOC_EXTENSIONS, PAPER_EXTENSIONS, IMAGE_EXTENSIONS

_WATCHED_EXTENSIONS = CODE_EXTENSIONS | DOC_EXTENSIONS | PAPER_EXTENSIONS | IMAGE_EXTENSIONS
_CODE_EXTENSIONS = CODE_EXTENSIONS


def _rebuild_code(
    watch_path: Path,
    *,
    follow_symlinks: bool = False,
    semantic_backend: str = "none",
    ollama_model: str | None = None,
    ollama_host: str = "http://127.0.0.1:11434",
    local_model: str | None = None,
    device: str | None = None,
) -> bool:
    """Re-run the local graphify pipeline for the watched folder."""
    try:
        from graphify.pipeline import run_pipeline

        result = run_pipeline(
            watch_path,
            semantic_backend=semantic_backend,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            local_model=local_model,
            device=device,
            no_viz=True,
            follow_symlinks=follow_symlinks,
        )

        G = result["graph"]
        communities = result["communities"]
        out = result["out_dir"]

        flag = out / "needs_update"
        if flag.exists():
            flag.unlink()

        print(f"[graphify watch] Rebuilt: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges, {len(communities)} communities")
        print(f"[graphify watch] graph.json and GRAPH_REPORT.md updated in {out}")
        for warning in result.get("warnings", []):
            print(f"[graphify watch] warning: {warning}")
        return True

    except Exception as exc:
        print(f"[graphify watch] Rebuild failed: {exc}")
        return False


def _notify_only(watch_path: Path) -> None:
    """Write a flag file and print a notification when semantic refresh is disabled."""
    flag = watch_path / "graphify-out" / "needs_update"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("1")
    print(f"\n[graphify watch] New or changed files detected in {watch_path}")
    print("[graphify watch] Non-code files changed, but semantic refresh is disabled.")
    print("[graphify watch] Re-run with `graphify . --semantic-backend ollama` or `--semantic-backend torch` to refresh locally.")
    print(f"[graphify watch] Flag written to {flag}")


def _has_non_code(changed_paths: list[Path]) -> bool:
    return any(p.suffix.lower() not in _CODE_EXTENSIONS for p in changed_paths)


def watch(
    watch_path: Path,
    debounce: float = 3.0,
    *,
    semantic_backend: str = "none",
    ollama_model: str | None = None,
    ollama_host: str = "http://127.0.0.1:11434",
    local_model: str | None = None,
    device: str | None = None,
    follow_symlinks: bool = False,
) -> None:
    """
    Watch watch_path for new or modified files and auto-update the graph.

    For code-only changes: re-runs the local pipeline immediately.
    For doc/paper/image changes: either refreshes semantic edges via the chosen
    local backend or writes a needs_update flag if semantic refresh is disabled.

    debounce: seconds to wait after the last change before triggering (avoids
    running on every keystroke when many files are saved at once).
    """
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError as e:
        raise ImportError("watchdog not installed. Run: pip install watchdog") from e

    last_trigger: float = 0.0
    pending: bool = False
    changed: set[Path] = set()

    class Handler(FileSystemEventHandler):
        def on_any_event(self, event):
            nonlocal last_trigger, pending
            if event.is_directory:
                return
            path = Path(event.src_path)
            if path.suffix.lower() not in _WATCHED_EXTENSIONS:
                return
            if any(part.startswith(".") for part in path.parts):
                return
            if "graphify-out" in path.parts:
                return
            last_trigger = time.monotonic()
            pending = True
            changed.add(path)

    handler = Handler()
    observer = Observer()
    observer.schedule(handler, str(watch_path), recursive=True)
    observer.start()

    print(f"[graphify watch] Watching {watch_path.resolve()} - press Ctrl+C to stop")
    print(f"[graphify watch] Semantic backend: {semantic_backend}")
    print(f"[graphify watch] Code changes rebuild graph automatically. "
          f"Doc/image changes trigger a local semantic refresh when enabled.")
    print(f"[graphify watch] Debounce: {debounce}s")

    try:
        while True:
            time.sleep(0.5)
            if pending and (time.monotonic() - last_trigger) >= debounce:
                pending = False
                batch = list(changed)
                changed.clear()
                print(f"\n[graphify watch] {len(batch)} file(s) changed")
                if _has_non_code(batch):
                    if semantic_backend.lower() in {"none", "off", "disabled"}:
                        _notify_only(watch_path)
                    else:
                        print(f"[graphify watch] Non-code changes detected - refreshing semantic graph via {semantic_backend}.")
                        _rebuild_code(
                            watch_path,
                            follow_symlinks=follow_symlinks,
                            semantic_backend=semantic_backend,
                            ollama_model=ollama_model,
                            ollama_host=ollama_host,
                            local_model=local_model,
                            device=device,
                        )
                else:
                    _rebuild_code(
                        watch_path,
                        follow_symlinks=follow_symlinks,
                        semantic_backend=semantic_backend,
                        ollama_model=ollama_model,
                        ollama_host=ollama_host,
                        local_model=local_model,
                        device=device,
                    )
    except KeyboardInterrupt:
        print("\n[graphify watch] Stopped.")
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watch a folder and auto-update the graphify graph")
    parser.add_argument("path", nargs="?", default=".", help="Folder to watch (default: .)")
    parser.add_argument("--debounce", type=float, default=3.0,
                        help="Seconds to wait after last change before updating (default: 3)")
    parser.add_argument("--semantic-backend", default="none", choices=["auto", "none", "ollama", "torch", "heuristic"],
                        help="Local semantic backend to use while watching")
    parser.add_argument("--ollama-model", help="Ollama model name for semantic refreshes")
    parser.add_argument("--ollama-host", default="http://127.0.0.1:11434", help="Ollama server URL")
    parser.add_argument("--local-model", help="Local Hugging Face / torch model name or path")
    parser.add_argument("--device", help="Torch device override (cpu, cuda, mps)")
    parser.add_argument("--follow-symlinks", action="store_true", help="Follow symlinked directories while watching")
    args = parser.parse_args()
    watch(
        Path(args.path),
        debounce=args.debounce,
        semantic_backend=args.semantic_backend,
        ollama_model=args.ollama_model,
        ollama_host=args.ollama_host,
        local_model=args.local_model,
        device=args.device,
        follow_symlinks=args.follow_symlinks,
    )
