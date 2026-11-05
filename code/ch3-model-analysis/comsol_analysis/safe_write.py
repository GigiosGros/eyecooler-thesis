"""Write output files atomically, tolerating cloud-sync locks (e.g. Box Drive).

Each file is written to a temporary name and then swapped in; if the sync client
or a viewer briefly holds the target, the swap is retried for a few seconds.
"""
from __future__ import annotations

import os
import time


def _swap(tmp: str, path: str, tries: int = 10, wait_s: float = 1.0):
    for k in range(tries):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            if k == tries - 1:
                os.remove(tmp)
                raise
            time.sleep(wait_s)


def _tmp(path: str) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.tmp{ext}"            # keep the extension so writers pick the right format


def save_figure(fig, path: str, dpi: int = 200, **kw):
    tmp = _tmp(path)
    fig.savefig(tmp, dpi=dpi, **kw)
    _swap(tmp, path)


def save_csv(df, path: str):
    tmp = _tmp(path)
    df.to_csv(tmp, index=False)
    _swap(tmp, path)


def save_text(text: str, path: str):
    tmp = _tmp(path)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    _swap(tmp, path)
