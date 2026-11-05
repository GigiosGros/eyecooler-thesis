"""Shared provenance helpers for the figure scripts.

Every figure in figures/aim2 that is produced by code is written by a script in
this folder from inputs kept under figures/aim2/sources/. Each run records, in
figures/aim2/MANIFEST.json, the script, its git commit, the SHA-256 of every
input and output, and where each input snapshot was copied from.
"""
import datetime
import hashlib
import json
import pathlib
import subprocess

THESIS = pathlib.Path(__file__).resolve().parents[2]
FIG = THESIS / "figures" / "aim2"
SOURCES = FIG / "sources"
MANIFEST = FIG / "MANIFEST.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(path):
    try:
        out = subprocess.run(["git", "-C", str(path), "log", "-1", "--format=%H"],
                             capture_output=True, text=True, check=True)
        # "+uncommitted" = the figure CODE differs from the commit. Untracked files
        # (backups, review folders) and the figures being regenerated used to mark
        # every entry, which made the flag meaningless (fixed 2026-10-19).
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain",
                                "--untracked-files=no", "--", "code"],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("+uncommitted" if dirty else "")
    except Exception:
        return None


def record(script, outputs, inputs, origins=None, notes=None, manifest=MANIFEST):
    """Add or replace the manifest entries for `outputs` (default figures/aim2/MANIFEST.json)."""
    manifest = pathlib.Path(manifest)
    man = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    script = pathlib.Path(script).resolve()
    for out in outputs:
        out = pathlib.Path(out)
        man[out.name] = {
            "generated": stamp,
            "script": str(script.relative_to(THESIS)).replace("\\", "/"),
            "script_sha256": sha256(script),
            "thesis_repo_commit_at_run": git_commit(THESIS),
            "output_sha256": sha256(out),
            "inputs": {str(pathlib.Path(i).relative_to(THESIS)).replace("\\", "/"): sha256(i)
                       for i in inputs},
            "input_origins": origins or {},
            "notes": notes or "",
        }
    manifest.write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
