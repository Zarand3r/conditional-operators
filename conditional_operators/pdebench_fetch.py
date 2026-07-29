"""Fetch the PDEBench 1D Reaction-Diffusion grid and reduce it to something we can train on.

PDEBench ships this equation as one file per parameter pair, `ReacDiff_Nu{nu}_Rho{rho}.hdf5`, over
nu in {0.5, 1, 2, 5} and rho in {1, 2, 5, 10}. That is a 4x4 grid of physical settings, which is
exactly the structure our compositional protocol needs: train on some cells, hold out others.

Each file is 4.1 GB and holds 10,000 trajectories, of which we need a few hundred. The `tensor`
dataset is stored **contiguously**, so the first N trajectories are one byte range: reading them over
HTTP Range requests pulls ~212 MB instead of 4.1 GB, a 19x saving, and the whole 4x4 grid costs about
3 GB of transfer rather than 66. Only the timesteps the task uses are written to disk.

    .venv/bin/python -m conditional_operators.pdebench_fetch
"""

from __future__ import annotations

import os
import sys
import time
import urllib.request
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets" / "pdebench_raw"
OUT = ROOT / "datasets" / "pdebench_reacdiff"
BASE = "https://darus.uni-stuttgart.de/api/access/datafile/"

NUS = (0.5, 1.0, 2.0, 5.0)
RHOS = (1.0, 2.0, 5.0, 10.0)
N_KEEP = 512            # trajectories kept per parameter cell
KEEP_T = (0, 25, 50, 75, 100)    # timesteps written to disk; the task uses the first and last
N_WORKERS = 4

# DaRUS file ids, read from the dataset listing (doi:10.18419/darus-2986).
FILE_IDS = {
    (0.5, 1.0): 133177, (0.5, 2.0): 133179, (0.5, 5.0): 133180, (0.5, 10.0): 133178,
    (1.0, 1.0): 133181, (1.0, 2.0): 133183, (1.0, 5.0): 133184, (1.0, 10.0): 133182,
}


def _discover_ids() -> dict:
    """Ask DaRUS for the remaining ids rather than hard-coding guesses."""
    import json
    url = ("https://darus.uni-stuttgart.de/api/datasets/:persistentId/"
           "?persistentId=doi:10.18419/darus-2986")
    d = json.load(urllib.request.urlopen(url, timeout=120))
    ids = {}
    for f in d["data"]["latestVersion"]["files"]:
        name = f["dataFile"]["filename"]
        if not name.startswith("ReacDiff_Nu"):
            continue
        nu = float(name.split("_Nu")[1].split("_")[0])
        rho = float(name.split("_Rho")[1].replace(".hdf5", ""))
        ids[(nu, rho)] = f["dataFile"]["id"]
    return ids


def cell_path(nu, rho) -> Path:
    return OUT / f"reacdiff_nu{nu}_rho{rho}.pt"


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests, so h5py can open a remote HDF5."""

    def __init__(self, url: str):
        self.url, self._pos, self.bytes_read = url, 0, 0
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, off, whence=0):
        self._pos = off if whence == 0 else (self._pos + off if whence == 1 else self.size + off)
        return self._pos

    def readinto(self, b):
        n = min(len(b), self.size - self._pos)
        if n <= 0:
            return 0
        req = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self._pos}-{self._pos + n - 1}"})
        for attempt in range(4):
            try:
                data = urllib.request.urlopen(req, timeout=300).read()
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        b[:len(data)] = data
        self._pos += len(data)
        self.bytes_read += len(data)
        return len(data)


def fetch_one(nu: float, rho: float, fid: int) -> str:
    """Range-read the first N_KEEP trajectories straight out of the remote file."""
    import h5py
    out = cell_path(nu, rho)
    if out.exists():
        return f"  nu={nu} rho={rho}: already present"
    t0 = time.time()
    hf = HttpFile(BASE + str(fid))
    with h5py.File(hf, "r", driver="fileobj") as f:
        name = "tensor" if "tensor" in f else next(k for k in f if f[k].ndim == 3)
        arr = np.asarray(f[name][:N_KEEP], dtype=np.float32)          # one contiguous range
        xs = np.asarray(f["x-coordinate"][:]) if "x-coordinate" in f else None
        ts = np.asarray(f["t-coordinate"][:]) if "t-coordinate" in f else None
    keep = [t for t in KEEP_T if t < arr.shape[1]]
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"u": torch.from_numpy(arr[:, keep]), "nu": nu, "rho": rho, "kept_t": keep,
                "x": None if xs is None else torch.from_numpy(np.asarray(xs, dtype=np.float32)),
                "t": None if ts is None else torch.from_numpy(np.asarray(ts, dtype=np.float32))},
               cell_path(nu, rho))
    return (f"  nu={nu} rho={rho}: {hf.bytes_read/1e6:.0f} MB transferred in "
            f"{(time.time()-t0)/60:.1f} min -> {out.stat().st_size/1e6:.0f} MB on disk")


def main():
    ids = dict(FILE_IDS)
    try:
        ids.update(_discover_ids())
    except Exception as e:      # the hard-coded half still works offline
        print(f"  (id discovery failed: {type(e).__name__}; using the known ids)", flush=True)
    want = [(nu, rho) for nu in NUS for rho in RHOS if (nu, rho) in ids]
    todo = [(nu, rho) for nu, rho in want if not cell_path(nu, rho).exists()]
    print(f"{len(want)} cells known, {len(todo)} to fetch, {N_KEEP} trajectories kept each",
          flush=True)
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(fetch_one, nu, rho, ids[(nu, rho)]) for nu, rho in todo]
        for f in futs:
            try:
                print(f.result(), flush=True)
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}", flush=True)
    have = sorted(p.name for p in OUT.glob("*.pt")) if OUT.exists() else []
    print(f"\n{len(have)} cells available")


if __name__ == "__main__":
    sys.exit(main())
