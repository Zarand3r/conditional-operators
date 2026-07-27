"""Guidance with a conditional distribution that actually has spread.

Stages 6 and 9 condition on every factor, so each condition names exactly one image and the
conditional distribution is a point mass. That measures distortion, which is a real thing, but it
cannot measure diversity: there is none to collapse. See docs/specs/GUIDANCE_PROGRAM.md.

Here the condition fixes shape, scale and orientation and leaves **position free**, so

    p(x | c) = uniform over the 64 (posX, posY) cells

exactly, by construction. That makes fidelity and coverage separately measurable against a target
known in closed form, with no proxy metric and no FID.

    .venv/bin/python -m conditional_operators.guidance2 --screen
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time

import torch
from torch import nn

from . import stage6, stage8
from .stage4 import Data, DEVICE, RESULTS_DIR, _mean, _std

# Condition: shape one-hot(3) + scale + orientation. Position is deliberately absent.
PCDIM = 5
SHAPES, SCALES, ORIENTS = 3, 6, 8
POS = 8                                   # posX and posY each take 8 of the 32 dSprites values
N_FREE = POS * POS                        # 64 images consistent with any one condition
CONDITIONS = [(sh, sc, o) for sh in range(SHAPES) for sc in range(SCALES) for o in range(ORIENTS)]

# Screen gate: CFG must visibly trade diversity for adherence here, or the harness cannot show a
# contrast and nothing downstream is worth running.
MAX_COVERAGE_RATIO = 0.7                  # coverage(w=8) / coverage(w=1) for CFG
STRENGTHS = (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)
DROPOUT = 0.1


def cond_vec(conds) -> torch.Tensor:
    """Only the conditioned factors. Position is not represented at all."""
    c = torch.zeros(len(conds), PCDIM)
    for i, (sh, sc, o) in enumerate(conds):
        c[i, sh] = 1.0
        c[i, 3] = sc / (SCALES - 1)
        c[i, 4] = o / (ORIENTS - 1)
    return c.to(DEVICE)


def images_for(data: Data, conds, xs, ys) -> torch.Tensor:
    lat = torch.zeros(len(conds), 6, dtype=torch.int64)
    for i, (sh, sc, o) in enumerate(conds):
        lat[i, 1], lat[i, 2], lat[i, 3] = sh, sc, o * 5
        lat[i, 4], lat[i, 5] = xs[i] * 4, ys[i] * 4
    idx = (lat * data.bases).sum(1).to(DEVICE)
    return data.imgs[idx].float().unsqueeze(1) * 2.0 - 1.0


def free_bank(data: Data, cond) -> torch.Tensor:
    """All 64 images consistent with one condition: the exact support of p(x|c)."""
    conds = [cond] * N_FREE
    xs = [i // POS for i in range(N_FREE)]
    ys = [i % POS for i in range(N_FREE)]
    return images_for(data, conds, xs, ys)


def sample_batch(data: Data, n, gen):
    """A training pair: a condition, and an image drawn uniformly from its free positions."""
    ci = torch.randint(len(CONDITIONS), (n,), generator=gen).tolist()
    conds = [CONDITIONS[i] for i in ci]
    xs = torch.randint(POS, (n,), generator=gen).tolist()
    ys = torch.randint(POS, (n,), generator=gen).tolist()
    return cond_vec(conds), images_for(data, conds, xs, ys)


# ---------------------------------------------------------------- the exactly-powerable arm

class PsiLinearDiT(stage6.MiniDiT):
    """Both channels linear in a nonlinear embedding, so powering is exact on both.

    stage-8's `hyb` takes its magnitude through the adaLN path, which is nonlinear in c, so
    powering it was only approximate and stage 9 disclosed that. Here an MLP produces psi(c) --
    all the expressiveness lives there -- and magnitude and phase are *linear* in psi. Scaling psi
    by alpha therefore scales the algebra coordinates exactly:

        (m e^{i theta})^alpha = m^alpha e^{i alpha theta}

    What is given up is additivity across different conditions, psi(c1+c2) != psi(c1)+psi(c2),
    which is provable: a continuous additive map is linear (Cauchy), so a nonlinear head can never
    have it. Powering one condition and adding two conditions are different properties, and
    guidance only needs the first.
    """

    def __init__(self):
        super().__init__("film")
        self.c_mlp = nn.Sequential(nn.Linear(PCDIM, stage6.DIM), nn.SiLU(),
                                   nn.Linear(stage6.DIM, stage6.DIM))
        self.Smag = nn.Linear(stage6.DIM, stage6.DIM // 2, bias=False)
        self.TH = nn.Linear(stage6.DIM, stage6.DIM // 2, bias=False)
        nn.init.zeros_(self.Smag.weight)
        nn.init.zeros_(self.TH.weight)

    def forward(self, x, t, c, alpha: float = 1.0):
        xp = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        psi = self.c_mlp(c) * alpha                       # the group coordinate; alpha powers it
        m = self.t_mlp(self.t_embed(t))                   # adaLN carries t only
        n, tok, dch = xp.shape
        flat = xp.reshape(n * tok, dch)
        flat = stage8._cmul(self.Smag(psi).repeat_interleave(tok, 0),
                            self.TH(psi).repeat_interleave(tok, 0), flat)
        xp = flat.view(n, tok, dch)
        for blk in self.blocks:
            xp = blk(xp, m)
        s2, sc = self.final_mod(m).chunk(2, dim=1)
        xp = self.final(stage6.modulate(self.final_n(xp), s2, sc))
        side = 64 // stage6.PATCH
        return xp.view(n, side, side, stage6.PATCH, stage6.PATCH).permute(
            0, 1, 3, 2, 4).reshape(n, 1, 64, 64)


class FilmDiT(stage6.MiniDiT):
    """The CFG baseline: standard adaLN, condition width reduced to the partial condition."""

    def __init__(self):
        super().__init__("film")
        self.c_mlp = nn.Sequential(nn.Linear(PCDIM, stage6.DIM), nn.SiLU(),
                                   nn.Linear(stage6.DIM, stage6.DIM))


ARMS = ("film_cfg", "psi_gp")


def build(arm, seed):
    torch.manual_seed(seed)
    return (FilmDiT() if arm == "film_cfg" else PsiLinearDiT()).to(DEVICE)


# ---------------------------------------------------------------- sampling

@torch.no_grad()
def ddim(model, c, noise, *, w=1.0, alpha=1.0, steps=None):
    """DDIM. `w` is CFG (2 NFE/step when w!=1); `alpha` is group power (always 1 NFE/step)."""
    steps = steps or stage6.DDIM_STEPS
    x = noise
    null = torch.zeros_like(c)
    ts = torch.linspace(stage6.T_STEPS, 0, steps + 1, device=DEVICE).long()
    nfe = 0
    for i in range(steps):
        t, t2 = ts[i], ts[i + 1]
        ab, ab2 = stage6.ABAR[t], stage6.ABAR[t2]
        tt = t.expand(x.shape[0])
        if alpha != 1.0:
            eps = model(x, tt, c, alpha=alpha); nfe += 1
        else:
            eps_c = model(x, tt, c); nfe += 1
            if w != 1.0:
                eps = model(x, tt, null) * (1 - w) + eps_c * w; nfe += 1
            else:
                eps = eps_c
        x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-1, 1)
        x = ab2.sqrt() * x0 + (1 - ab2).sqrt() * eps
    return x0, nfe


# ---------------------------------------------------------------- metrics against exact truth

@torch.no_grad()
def score(samples: torch.Tensor, bank: torch.Tensor) -> dict:
    """Fidelity and diversity, both against the exact support of p(x|c).

    Every generated image is assigned to its nearest valid image. How far it sits from that image
    is fidelity; how the assignments spread over the 64 cells is diversity, whose truth is uniform.
    """
    d = torch.cdist(samples.flatten(1), bank.flatten(1))          # [K, 64]
    dist, which = d.min(dim=1)
    k = samples.shape[0]
    counts = torch.bincount(which, minlength=bank.shape[0]).float()
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy = float(-(nz * nz.log()).sum() / math.log(bank.shape[0]))   # 1.0 == uniform
    tv = float(0.5 * (p - 1.0 / bank.shape[0]).abs().sum())
    return {
        "fidelity": float((dist ** 2).mean() / samples[0].numel()),     # per-pixel MSE to nearest
        "coverage": float((counts > 0).sum()) / min(k, bank.shape[0]),
        "entropy": entropy,
        "tv_from_uniform": tv,
    }


@torch.no_grad()
def evaluate(model, data, arm, strength, *, n_cond=16, k=64, steps=None, seed=777):
    """Average the metrics over a fixed set of conditions, identical across arms and strengths."""
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(CONDITIONS), generator=g)[:n_cond].tolist()
    acc, nfe_total = [], 0
    for ci in pick:
        cond = CONDITIONS[ci]
        bank = free_bank(data, cond)
        c = cond_vec([cond] * k)
        noise = torch.randn(k, 1, 64, 64, generator=g).to(DEVICE)
        if arm == "film_cfg":
            s, nfe = ddim(model, c, noise, w=strength, steps=steps)
        else:
            s, nfe = ddim(model, c, noise, alpha=strength, steps=steps)
        nfe_total += nfe
        acc.append(score(s, bank))
    out = {k2: _mean([a[k2] for a in acc]) for k2 in acc[0]}
    out["nfe_per_sample"] = nfe_total / len(pick)
    return out


# ---------------------------------------------------------------- training

def _throttle(step):
    ms = int(os.environ.get("GUIDANCE_THROTTLE_MS", "0"))
    if ms and step % 5 == 0:
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        time.sleep(ms / 1000.0)


def ckpt_path(arm, seed):
    return RESULTS_DIR / f"guidance2_{arm}_s{seed}.pt"


def train_one(arm, seed, data, steps, batch=256, lr=1e-3):
    """Trains and saves an EMA checkpoint. E1, E2 and E3 all sample from these."""
    path = ckpt_path(arm, seed)
    if path.exists():
        m = build(arm, seed)
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        return m.eval()
    m = build(arm, seed)
    ema = copy.deepcopy(m)
    opt = torch.optim.Adam(m.parameters(), lr=lr, betas=(0.9, 0.99))
    gen = torch.Generator().manual_seed(seed)
    for step in range(steps):
        _throttle(step)
        c, x0 = sample_batch(data, batch, gen)
        drop = (torch.rand(batch, generator=gen) < DROPOUT).to(DEVICE)
        c = c.clone(); c[drop] = 0.0                        # null condition, for CFG
        t = torch.randint(1, stage6.T_STEPS, (batch,), generator=gen).to(DEVICE)
        ab = stage6.ABAR[t].view(-1, 1, 1, 1)
        eps = torch.randn(x0.shape, generator=gen).to(DEVICE)
        loss = torch.mean((m(ab.sqrt() * x0 + (1 - ab).sqrt() * eps, t, c) - eps) ** 2)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"{arm}/seed{seed} diverged")
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), m.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)
        if (step + 1) % 5000 == 0:
            print(f"  {arm} s{seed} step {step+1}: loss {loss.item():.4f}", flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    torch.save(ema.state_dict(), path)
    return ema.eval()


# ---------------------------------------------------------------- E0: the instrument check

def screen(steps=15000, seed=0):
    """Does CFG trade diversity for adherence here? If not, nothing downstream is worth running."""
    data = Data()
    rows = {}
    for arm in ARMS:
        t0 = time.time()
        m = train_one(arm, seed, data, steps)
        print(f"  trained {arm} [{(time.time()-t0)/60:.0f} min]", flush=True)
        rows[arm] = {s: evaluate(m, data, arm, s) for s in STRENGTHS}
        for s in STRENGTHS:
            r = rows[arm][s]
            print(f"  {arm:9} strength={s:<4} fidelity={r['fidelity']:.5f} "
                  f"coverage={r['coverage']:.3f} entropy={r['entropy']:.3f} "
                  f"nfe={r['nfe_per_sample']:.0f}", flush=True)

    cfg = rows["film_cfg"]
    ratio = cfg[8.0]["coverage"] / cfg[1.0]["coverage"] if cfg[1.0]["coverage"] else float("nan")
    ok = ratio <= MAX_COVERAGE_RATIO
    print(f"\nE0 instrument check: CFG coverage {cfg[1.0]['coverage']:.3f} -> "
          f"{cfg[8.0]['coverage']:.3f} (ratio {ratio:.2f}, need <= {MAX_COVERAGE_RATIO})")
    print("VERDICT:", "USABLE — CFG collapses diversity here, so a contrast exists"
          if ok else "NOT USABLE — CFG does not trade diversity here; H is untestable on it")
    out = {"experiment": "E0-instrument-check", "steps": steps, "seed": seed,
           "coverage_ratio": ratio, "threshold": MAX_COVERAGE_RATIO, "usable": ok,
           "rows": {a: {str(s): v for s, v in r.items()} for a, r in rows.items()}}
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "guidance2_e0.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    if "--screen" in sys.argv:
        screen(steps=int(sys.argv[sys.argv.index("--screen") + 1])
               if len(sys.argv) > sys.argv.index("--screen") + 1 else 15000)
        return
    print(__doc__)


if __name__ == "__main__":
    main()
