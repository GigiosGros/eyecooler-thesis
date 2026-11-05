"""Shared provenance helper: code stamp, git state, input hashes and one MANIFEST.json schema for every output folder.

    import manifest
    stamp = manifest.code_stamp()                        # "sha256-12 (run_all.py + replica/*.py + dev/*.py, LF)"
    manifest.write(out_dir, script=__file__, inputs=[...], outputs=[...], notes={...})

    python -X utf8 dev/manifest.py                       # prints the stamps and the git state

Code stamp.  run_all.code_version() hashes run_all.py + replica/*.py only, so runs made
before and after a change in dev/ (for example the cad_base fix of 2026-10-19) carry the same stamp.  This stamp
covers the driver and its helpers as well:

    stamp = SHA-256( for each file f in sorted(run_all.py, replica/*.py, dev/*.py): name(f) + LF-normalised bytes(f) )

where name(f) is the path relative to the repository root with "/" separators and LF-normalised means every CR LF
pair replaced by LF, so a Windows checkout and a Linux checkout of the same commit give the same stamp.  The old
stamp (run_all.code_version()) is kept in run logs for comparison with earlier results; it is NOT normalised.

MANIFEST.json schema (version "scr-manifest-1"):
    schema, made (local time), script (relative path), command (argv), code_stamp, code_stamp_files (count),
    git: {commit, dirty (bool), dirty_files (list, at most 50)}, environment: {python, numpy, scipy, platform},
    inputs: {path: sha256}, outputs: {name: sha256}, notes: {...}
Hashes are SHA-256 of the file bytes as stored (no normalisation).
"""
import datetime
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCHEMA = "scr-manifest-1"


def code_files():
    files = [os.path.join(REPO, "run_all.py")]
    files += sorted(glob.glob(os.path.join(REPO, "replica", "*.py")))
    files += sorted(glob.glob(os.path.join(HERE, "*.py")))
    return [f for f in files if os.path.isfile(f)]


def code_stamp(full=False):
    """Short (12 hex) or full SHA-256 over run_all.py, replica/*.py and dev/*.py, CR LF normalised to LF."""
    h = hashlib.sha256()
    for path in code_files():
        rel = os.path.relpath(path, REPO).replace(os.sep, "/")
        with open(path, "rb") as f:
            h.update(rel.encode() + b"\0" + f.read().replace(b"\r\n", b"\n") + b"\0")
    d = h.hexdigest()
    return d if full else d[:12]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def git_state():
    """{commit, dirty, dirty_files}; None values if git is not available (for example on the cluster copy)."""
    def run(*a):
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, timeout=60).stdout
    try:
        commit = run("rev-parse", "HEAD").strip() or None
        st = [ln[3:] for ln in run("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip()]
        return dict(commit=commit, dirty=bool(st), dirty_files=st[:50])
    except Exception:
        return dict(commit=None, dirty=None, dirty_files=[])


def environment():
    env = dict(python=sys.version.split()[0], platform=platform.platform())
    for m in ("numpy", "scipy", "matplotlib"):
        try:
            env[m] = __import__(m).__version__
        except Exception:
            env[m] = None
    return env


def write(out_dir, script, inputs=(), outputs=None, notes=None, command=None, name="MANIFEST.json"):
    """Write out_dir/MANIFEST.json in the shared schema.  inputs: paths read; outputs: file names in out_dir
    (default: every file in out_dir except the manifest itself)."""
    if outputs is None:
        outputs = sorted(f for f in os.listdir(out_dir) if f != name and os.path.isfile(os.path.join(out_dir, f)))
    m = dict(schema=SCHEMA, made=datetime.datetime.now().isoformat(timespec="seconds"),
             script=os.path.relpath(os.path.abspath(script), REPO).replace(os.sep, "/"),
             command=list(command if command is not None else sys.argv),
             code_stamp=code_stamp(), code_stamp_files=len(code_files()), git=git_state(), environment=environment(),
             inputs={str(p): (sha256(p) if os.path.isfile(p) else "missing") for p in inputs},
             outputs={o: sha256(os.path.join(out_dir, o)) for o in outputs if os.path.isfile(os.path.join(out_dir, o))},
             notes=notes or {})
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)
    return m


if __name__ == "__main__":
    print("code stamp (run_all.py + replica/*.py + dev/*.py, LF):", code_stamp(), f"({len(code_files())} files)")
    print("git:", json.dumps(git_state()))
    print("environment:", json.dumps(environment()))
