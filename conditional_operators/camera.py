"""Synthetic camera-control benchmark (docs/specs/CAMERA1_SPEC.md).

A deterministic renderer draws a scene of coloured 3D points through a pinhole camera. A camera
pose is four sliders, the way a user manipulates one:

    yaw, pitch, log-distance, roll

A *move* is a vector of increments to those sliders, so the space of moves is (R^4, +): abelian,
which is the regime the composition law applies to. Full SE(3) trajectories are out of scope.

The task mirrors the image-transformation experiments: encode the frame at pose p, apply the
conditioning arm with the move as the condition, decode, and score pixel error against the true
frame at p + move. Because the renderer is analytic, that target is exact for every move,
including ones no model was trained on.

Run:  .venv/bin/python -m conditional_operators.camera [N_SEEDS] [STEPS]
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time

import torch
from torch import nn

from . import stage8  # noqa: F401  (registers the Complex FiLM arms in ARM_CLASSES)
from .stage4 import ARM_CLASSES, CondArm, DEVICE, H, RESULTS_DIR, _fl, _mean, _std, _zero_
from .verdict import Arm, ArmResult, decide

# ------------------------------------------------------------------ scene and renderer

RES = 64                 # rendered image is RES x RES RGB
N_POINTS = 48            # coloured points making up a scene
FOCAL = 1.6              # pinhole focal length in normalised image units
AXES = ("yaw", "pitch", "dist", "roll")
DC = 4                   # the slider vector
# per-axis scale of one unit of move, chosen so a unit move is visible but not extreme
MOVE_SCALE = torch.tensor([0.45, 0.35, 0.22, 0.50])
BASE_DIST = 4.0


def make_scene(seed: int, n: int = N_POINTS):
    """A scene is a fixed cloud of coloured points. Deterministic in the seed."""
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn(n, 3, generator=g) * torch.tensor([1.1, 1.1, 1.1])
    cols = torch.rand(n, 3, generator=g) * 0.75 + 0.25
    return pts.to(DEVICE), cols.to(DEVICE)


def _rot(yaw, pitch, roll):
    """Camera rotation from three angles. [B,3,3]."""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    z = torch.zeros_like(cy)
    o = torch.ones_like(cy)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).view(-1, 3, 3)
    Rx = torch.stack([o, z, z, z, cp, -sp, z, sp, cp], -1).view(-1, 3, 3)
    Rz = torch.stack([cr, -sr, z, sr, cr, z, z, z, o], -1).view(-1, 3, 3)
    return Rz @ Rx @ Ry


def render(pts, cols, pose):
    """Render a batch of poses. pose is [B,4] = (yaw, pitch, log-dist, roll). Returns [B,3,RES,RES].

    Points are projected through a pinhole camera and splatted as Gaussians whose size falls off
    with depth, which gives a smooth, differentiable image with an unambiguous ground truth.
    """
    B = pose.shape[0]
    R = _rot(pose[:, 0], pose[:, 1], pose[:, 3])
    dist = BASE_DIST * torch.exp(pose[:, 2])                       # [B]
    cam = pts[None] @ R.transpose(1, 2)                            # [B,N,3]
    cam = cam + torch.stack([torch.zeros_like(dist), torch.zeros_like(dist), dist], -1)[:, None]
    zc = cam[..., 2].clamp(min=0.35)
    uv = FOCAL * cam[..., :2] / zc[..., None]                      # [B,N,2] in [-1,1]-ish

    grid = torch.linspace(-1, 1, RES, device=pose.device)
    gy, gx = torch.meshgrid(grid, grid, indexing="ij")
    d2 = ((gx[None, None] - uv[..., 0][..., None, None]) ** 2
          + (gy[None, None] - uv[..., 1][..., None, None]) ** 2)   # [B,N,RES,RES]
    sigma = (0.085 * BASE_DIST / zc).clamp(0.02, 0.35)[..., None, None]
    w = torch.exp(-d2 / (2 * sigma ** 2))                          # [B,N,RES,RES]
    w = w * (1.0 / zc.clamp(min=0.5))[..., None, None]             # nearer points dominate
    img = torch.einsum("bnhw,nc->bchw", w, cols)
    return (img / (w.sum(1)[:, None] + 1e-3)).clamp(0, 1) * (w.sum(1)[:, None] > 0.02).float()


# ------------------------------------------------------------------ move types and splits

SINGLES = tuple((i,) for i in range(DC))
TRAIN_PAIRS = ((0, 1), (2, 3))
VAL_PAIRS = ((0, 2), (1, 3))
TEST_PAIRS = ((0, 3), (1, 2))
TRIPLES = tuple(itertools.combinations(range(DC), 3))
TRAIN_TYPES = SINGLES + TRAIN_PAIRS
TRAIN_SCENES, TEST_SCENES = 4096, 512      # disjoint scene-seed ranges


def sample(types, n, gen, train=True):
    """n triplets (x1, move, x2). Scene seeds come from disjoint pools for train and test."""
    lo, hi = (0, TRAIN_SCENES) if train else (10**6, 10**6 + TEST_SCENES)
    seeds = torch.randint(lo, hi, (n,), generator=gen).tolist()
    pose = torch.rand(n, DC, generator=gen) * 2 - 1
    pose[:, 2] *= 0.25                                          # keep distance sane
    move = torch.zeros(n, DC)
    which = torch.randint(len(types), (n,), generator=gen)
    for ti, t in enumerate(types):
        m = which == ti
        k = int(m.sum())
        if k == 0:
            continue
        for ax in t:
            mag = torch.rand(k, generator=gen) * 0.7 + 0.3
            sign = torch.randint(0, 2, (k,), generator=gen).float() * 2 - 1
            move[m, ax] = mag * sign
    pose, move = pose.to(DEVICE), move.to(DEVICE)
    scaled = move * MOVE_SCALE.to(DEVICE)
    x1 = torch.cat([render(*make_scene(s), pose[i:i + 1]) for i, s in enumerate(seeds)])
    x2 = torch.cat([render(*make_scene(s), (pose + scaled)[i:i + 1]) for i, s in enumerate(seeds)])
    return x1, move, x2


# ------------------------------------------------------------------ model

DZ = 128


class Backbone(nn.Module):
    """Shared encoder/decoder. Identical for every arm; only the conditioning module differs."""

    def __init__(self, ch: int = 48):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.SiLU(),            # 32
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.SiLU(),       # 16
            nn.Conv2d(ch * 2, ch * 2, 4, 2, 1), nn.SiLU(),   # 8
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.SiLU(),   # 4
            nn.Flatten(), nn.Linear(ch * 4 * 16, DZ))
        self.fc = nn.Linear(DZ, ch * 4 * 16)
        self.ch = ch
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch * 2, ch * 2, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1))

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(self.fc(z).view(-1, self.ch * 4, 4, 4))


Additive = ARM_CLASSES["additive"]   # the mechanism CameraCtrl deploys; now shared, see stage4

GATE_ARMS = ("additive", "film", "concat_mlp", "cond_layernorm", "hypernet",
             "dynamic_linear", "proposed")
REPORTED = ("cfilm_hyb", "proposed_mlp_gs")
ALL_ARMS = GATE_ARMS + REPORTED


class Model(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        self.backbone = Backbone()
        self.cond = ARM_CLASSES[arm](dc=DC)

    def forward(self, x1, move):
        return self.backbone.decode(self.cond(move, self.backbone.encode(x1)))


# ------------------------------------------------------------------ train and score

def train_one(arm, seed, steps, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    m = Model(arm).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    throttle = int(os.environ.get("CAMERA_THROTTLE_MS", "0"))
    diverged = False
    for step in range(steps):
        if throttle and step % 5 == 0:
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            time.sleep(throttle / 1000.0)
        x1, move, x2 = sample(TRAIN_TYPES, batch, gen, train=True)
        loss = torch.mean((torch.sigmoid(m(x1, move)) - x2) ** 2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()

    row = dict(arm=arm, seed=seed, diverged=diverged,
               params=m.cond.n_params(), flops=m.cond.flops())
    if diverged:
        row |= {k: math.nan for k in ("indist", "val", "test", "triples")}
        return row

    @torch.no_grad()
    def mse(types, train):
        eg = torch.Generator().manual_seed(555_000)      # identical eval data for every arm
        tot = 0.0
        for _ in range(4):
            x1, move, x2 = sample(types, 512, eg, train=train)
            tot += torch.mean((torch.sigmoid(m(x1, move)) - x2) ** 2).item()
        return tot / 4

    row |= dict(indist=mse(TRAIN_TYPES, True), val=mse(VAL_PAIRS, False),
                test=mse(TEST_PAIRS, False), triples=mse(TRIPLES, False))
    return row


def run(n_seeds, steps):
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = {a: [] for a in ALL_ARMS}
    log_path = RESULTS_DIR / "camera1_log.jsonl"
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            runs[r["arm"]].append(r); done.add((r["arm"], r["seed"]))
        if done:
            print(f"resuming: {len(done)} runs already done", flush=True)
    with log_path.open("a") as log:
        for arm in ALL_ARMS:
            for seed in range(n_seeds):
                if (arm, seed) in done:
                    continue
                t = time.time()
                r = train_one(arm, seed, steps)
                runs[arm].append(r)
                log.write(json.dumps(r) + "\n"); log.flush(); os.fsync(log.fileno())
                print(f"{arm:16} seed={seed} test={r['test']:.5f} indist={r['indist']:.5f} "
                      f"triples={r['triples']:.5f} [{time.time()-t:.0f}s]", flush=True)

    return summarise(runs, n_seeds, steps)


def summarise(runs, n_seeds, steps):
    def arr(a, k):
        return tuple(r[k] for r in runs[a] if not r["diverged"])

    # C2/C3 reuse the shared gate, which needs the six standard arms
    std = ("film", "concat_mlp", "cond_layernorm", "hypernet", "dynamic_linear", "proposed")
    results = {Arm(a): ArmResult(Arm(a), arr(a, "test"), arr(a, "indist"),
                                 sum(1 for r in runs[a] if r["diverged"]),
                                 runs[a][0]["params"], runs[a][0]["flops"], 1)
               for a in std}
    gate = decide(results, n_required=n_seeds)

    from .verdict import cliffs_delta, mann_whitney_u
    prop, add = arr("proposed", "test"), arr("additive", "test")
    m1 = (_mean(add) - _mean(prop)) / _mean(add)
    _, p1 = mann_whitney_u(prop, add); d1 = cliffs_delta(prop, add)
    crit = {
        "C1": m1 >= 0.25 and p1 <= 0.01 and d1 <= -0.474,
        "C2": bool(gate.criteria.get("AC-2")),
        "C3": bool(gate.criteria.get("AC-5")),
        "C4": runs["proposed"][0]["flops"] <= 1.20 * runs["film"][0]["flops"],
    }
    verdict = "confirmed" if all(crit.values()) else "kill"
    summary = {
        "experiment": "camera-sliders", "spec": "docs/specs/CAMERA1_SPEC.md",
        "config": {"n_seeds": n_seeds, "steps": steps, "res": RES},
        "final_verdict": verdict, "criteria": crit,
        "margin_vs_additive": m1, "p_vs_additive": p1, "delta_vs_additive": d1,
        "margin_vs_unstructured": gate.margin_observed,
        "best_unstructured": gate.best_unstructured.value if gate.best_unstructured else None,
        "per_arm": {a: {k: _mean(arr(a, k)) for k in ("indist", "val", "test", "triples")}
                       | {"test_std": _std(arr(a, "test")),
                          "params": runs[a][0]["params"], "flops": runs[a][0]["flops"],
                          "flops_vs_film": runs[a][0]["flops"] / runs["film"][0]["flops"]}
                    for a in ALL_ARMS},
    }
    (RESULTS_DIR / "camera1_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    t = time.time()
    s = run(n, steps)
    print("\n" + "=" * 60)
    print(f"CAMERA-SLIDERS VERDICT: {s['final_verdict'].upper()}  {s['criteria']}")
    for a in ALL_ARMS:
        p = s["per_arm"][a]
        print(f"  {a:16} test={p['test']:.5f} indist={p['indist']:.5f} "
              f"triples={p['triples']:.5f} flops={p['flops_vs_film']:.2f}x")
    print(f"total wall: {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
