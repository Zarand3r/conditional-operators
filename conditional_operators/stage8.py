"""Stage-8: Complex FiLM — magnitude (content) x phase (composition) per feature pair.

Two sub-suites (STAGE8_SPEC.md):
  8a (transform role): dSprites delta task, Stage-4 harness, arms film/hypernet/proposed/
      cfilm_lin/cfilm_hyb. Gate: cfilm_hyb beats film >=20% OOD at fit parity and budget.
  8b (content role): Stage-6 mini-DiT conditional diffusion, arms film/cfilm_hyb/cfilm_lin.
      Gate: NON-INFERIORITY (cfilm_hyb within 1.10x film on OOD and in-dist).

Run:  .venv/bin/python -m conditional_operators.stage8 a [N] [STEPS]
      .venv/bin/python -m conditional_operators.stage8 b [N] [STEPS]
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
from torch import nn

from . import stage4, stage6
from .stage4 import (ARM_CLASSES, CondArm, Data, DEVICE, DZ, H, RESULTS_DIR, _fl, _mean, _std,
                     _zero_)
from .verdict import cliffs_delta, mann_whitney_u

SCLAMP = 4.0


def _cmul(s, th, z):
    """Multiply feature pairs of z by m*e^{i theta}, m = exp(clamped s)."""
    m = torch.exp(s.clamp(-SCLAMP, SCLAMP))
    c, sn = torch.cos(th), torch.sin(th)
    z0, z1 = z[:, 0::2], z[:, 1::2]
    y = torch.empty_like(z)
    y[:, 0::2] = m * (c * z0 - sn * z1)
    y[:, 1::2] = m * (sn * z0 + c * z1)
    return y


class CFiLMLin(CondArm):
    """Fully compositional: magnitude and phase both linear bias-free in the raw condition."""

    def __init__(self, dc=stage4.DC):
        super().__init__(dc)
        self.S = nn.Linear(dc, DZ // 2, bias=False)
        self.TH = nn.Linear(dc, DZ // 2, bias=False)
        nn.init.zeros_(self.S.weight); nn.init.zeros_(self.TH.weight)

    def op(self, h, d, z):
        return _cmul(self.S(d), self.TH(d), z)

    def op_flops(self):
        return 2 * _fl(self.dc, DZ // 2) + 4 * DZ


class CFiLMHyb(CondArm):
    """The candidate FiLM successor: MLP magnitude head (content), linear phase (composition)."""

    def __init__(self, dc=stage4.DC):
        super().__init__(dc)
        self.S = _zero_(nn.Linear(H, DZ // 2))          # content channel: FiLM-style head
        self.TH = nn.Linear(dc, DZ // 2, bias=False)    # composition channel: exact
        nn.init.zeros_(self.TH.weight)

    def op(self, h, d, z):
        return _cmul(self.S(h), self.TH(d), z)

    def op_flops(self):
        return _fl(H, DZ // 2) + _fl(self.dc, DZ // 2) + 4 * DZ


ARM_CLASSES["cfilm_lin"] = CFiLMLin
ARM_CLASSES["cfilm_hyb"] = CFiLMHyb

ARMS_8A = ("film", "hypernet", "proposed", "cfilm_lin", "cfilm_hyb")


def run_8a(n_seeds, steps):
    data = Data()
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = {a: [] for a in ARMS_8A}
    log_path = RESULTS_DIR / "stage8a_log.jsonl"
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            runs[r["arm"]].append(r); done.add((r["arm"], r["seed"]))
        if done:
            print(f"resuming: {len(done)} runs", flush=True)
    throttle = int(os.environ.get("STAGE8_THROTTLE_MS", "0"))
    with log_path.open("a") as log:
        for arm in ARMS_8A:
            for seed in range(n_seeds):
                if (arm, seed) in done:
                    continue
                t = time.time()
                if throttle:
                    os.environ["STAGE4_THROTTLE_MS"] = str(throttle)  # consumed below via loop
                r = _train48(arm, seed, data, steps, throttle)
                runs[arm].append(r)
                log.write(json.dumps(r) + "\n"); log.flush(); os.fsync(log.fileno())
                print(f"{arm:12} seed={seed} ood={r['ood_test']:.6f} indist={r['indist']:.6f} "
                      f"[{time.time()-t:.0f}s]", flush=True)

    def arr(a, k):
        return tuple(r[k] for r in runs[a] if not r["diverged"])
    hyb, film = arr("cfilm_hyb", "ood_test"), arr("film", "ood_test")
    m = (_mean(film) - _mean(hyb)) / _mean(film)
    _, p = mann_whitney_u(hyb, film); d = cliffs_delta(hyb, film)
    crit = {
        "AC-8a1": m >= 0.20 and p <= 0.01 and d <= -0.474,
        "AC-8a2": _mean(arr("cfilm_hyb", "indist")) <= 1.10 * _mean(arr("film", "indist")),
        "AC-8a3": runs["cfilm_hyb"][0]["flops"] <= 1.20 * runs["film"][0]["flops"],
    }
    verdict = "confirmed" if all(crit.values()) else "kill"
    summary = {
        "stage": "8a", "spec": "docs/specs/STAGE8_SPEC.md",
        "config": {"n_seeds": n_seeds, "steps": steps},
        "final_verdict": verdict, "criteria": crit,
        "margin_vs_film": m, "p_vs_film": p, "delta_vs_film": d,
        "per_arm": {a: {"ood_test_mean": _mean(arr(a, "ood_test")),
                        "ood_test_std": _std(arr(a, "ood_test")),
                        "indist_mean": _mean(arr(a, "indist")),
                        "params": runs[a][0]["params"], "flops": runs[a][0]["flops"],
                        "flops_vs_film": runs[a][0]["flops"] / runs["film"][0]["flops"]}
                    for a in ARMS_8A},
    }
    (RESULTS_DIR / "stage8a_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _train48(arm, seed, data, steps, throttle_ms):
    """Stage-4 training loop with throttle (kept local to avoid touching stage4's registered code)."""
    torch.manual_seed(seed)
    m = stage4.Model(arm).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    bce = nn.BCEWithLogitsLoss()
    diverged = False
    for step in range(steps):
        if throttle_ms and step % 5 == 0:
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            time.sleep(throttle_ms / 1000.0)
        x1, d, x2 = data.sample(stage4.TRAIN_TYPES, 256, gen)
        loss = bce(m(x1, d), x2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()
    row = dict(arm=arm, seed=seed, diverged=diverged,
               params=m.cond.n_params(), flops=m.cond.flops())
    eg = torch.Generator().manual_seed(70_000 + seed)
    if diverged:
        row |= dict(indist=math.nan, ood_val=math.nan, ood_test=math.nan, triples=math.nan)
        return row

    @torch.no_grad()
    def mse(types):
        tot = 0.0
        for _ in range(4):
            x1, d, x2 = data.sample(types, 512, eg)
            tot += torch.mean((torch.sigmoid(m(x1, d)) - x2) ** 2).item()
        return tot / 4
    row |= dict(indist=mse(stage4.TRAIN_TYPES), ood_val=mse(stage4.VAL_PAIRS),
                ood_test=mse(stage4.TEST_PAIRS), triples=mse(stage4.TRIPLES))
    return row


# ---------------------------------------------------------------- 8b: content role (mini-DiT)

class CFiLMDiT(stage6.MiniDiT):
    """film adaLN(t [+c for hyb via mod]) + complex-FiLM at transformer entry.

    hyb: magnitude via the standard mod path (c joins t in adaLN, FiLM's content mechanism)
         PLUS linear phase rotation at entry.
    lin: t-only mod; magnitude AND phase linear from raw c at entry.
    """

    def __init__(self, variant: str):
        super().__init__("film")            # build the standard film backbone (mod from t+c)
        self.variant = variant
        self.TH = nn.Linear(stage6.CDIM, stage6.DIM // 2, bias=False)
        nn.init.zeros_(self.TH.weight)
        if variant == "lin":
            self.Smag = nn.Linear(stage6.CDIM, stage6.DIM // 2, bias=False)
            nn.init.zeros_(self.Smag.weight)

    def forward(self, x, t, c):
        xp = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        te = self.t_mlp(self.t_embed(t))
        ce = self.c_mlp(c)
        m = te + ce if self.variant == "hyb" else te
        n, tok, dch = xp.shape
        flat = xp.reshape(n * tok, dch)
        th = self.TH(c).repeat_interleave(tok, 0)
        if self.variant == "lin":
            s = self.Smag(c).repeat_interleave(tok, 0)
            flat = _cmul(s, th, flat)
        else:
            from .stage3 import _rotate
            flat = _rotate(th, flat)
        xp = flat.view(n, tok, dch)
        for blk in self.blocks:
            xp = blk(xp, m)
        s2, sc = self.final_mod(m).chunk(2, dim=1)
        xp = self.final(stage6.modulate(self.final_n(xp), s2, sc))
        side = 64 // stage6.PATCH
        return xp.view(n, side, side, stage6.PATCH, stage6.PATCH).permute(0, 1, 3, 2, 4).reshape(
            n, 1, 64, 64)

    def class_path_flops(self):
        base = 2 * stage6.CDIM * stage6.DIM + 2 * stage6.DIM * stage6.DIM  # c_mlp (hyb only)
        phase = 2 * stage6.CDIM * (stage6.DIM // 2) + stage6.TOKENS * 3 * stage6.DIM
        if self.variant == "hyb":
            return base + phase
        return 2 * 2 * stage6.CDIM * (stage6.DIM // 2) + stage6.TOKENS * 4 * stage6.DIM


ARMS_8B = ("film", "cfilm_hyb", "cfilm_lin")


def _build_8b(arm, seed):
    torch.manual_seed(seed)
    if arm == "film":
        return stage6.MiniDiT("film").to(DEVICE)
    return CFiLMDiT("hyb" if arm == "cfilm_hyb" else "lin").to(DEVICE)


def run_8b(n_seeds, steps):
    import copy
    data = Data()
    splits = stage6.make_combos()
    RESULTS_DIR.mkdir(exist_ok=True)
    runs = {a: [] for a in ARMS_8B}
    log_path = RESULTS_DIR / "stage8b_log.jsonl"
    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            runs[r["arm"]].append(r); done.add((r["arm"], r["seed"]))
        if done:
            print(f"resuming: {len(done)} runs", flush=True)
    throttle = int(os.environ.get("STAGE8_THROTTLE_MS", "0"))
    os.environ["STAGE6_THROTTLE_MS"] = str(throttle)
    train_c, val_c, test_c = splits
    with log_path.open("a") as log:
        for arm in ARMS_8B:
            for seed in range(n_seeds):
                if (arm, seed) in done:
                    continue
                t = time.time()
                r = _train68(arm, seed, data, splits, steps, throttle)
                runs[arm].append(r)
                log.write(json.dumps(r) + "\n"); log.flush(); os.fsync(log.fileno())
                print(f"{arm:10} seed={seed} ood={r['ood_test']:.5f} indist={r['indist']:.5f} "
                      f"[{time.time()-t:.0f}s]", flush=True)

    def arr(a, k):
        return tuple(r[k] for r in runs[a] if not r["diverged"])
    film_ood, hyb_ood = _mean(arr("film", "ood_test")), _mean(arr("cfilm_hyb", "ood_test"))
    film_ind, hyb_ind = _mean(arr("film", "indist")), _mean(arr("cfilm_hyb", "indist"))
    bb_fwd = 100_000_000  # order-of-magnitude backbone forward; ratio check is the honest bound
    crit = {
        "AC-8b1": hyb_ood <= 1.10 * film_ood,
        "AC-8b2": hyb_ind <= 1.10 * film_ind,
        "AC-8b3": (runs["cfilm_hyb"][0]["class_path_flops"] <=
                   2.0 * runs["film"][0]["class_path_flops"]),
    }
    verdict = "confirmed" if all(crit.values()) else "kill"
    summary = {
        "stage": "8b", "spec": "docs/specs/STAGE8_SPEC.md",
        "config": {"n_seeds": n_seeds, "steps": steps},
        "final_verdict": verdict, "criteria": crit,
        "per_arm": {a: {"ood_test_mean": _mean(arr(a, "ood_test")),
                        "ood_test_std": _std(arr(a, "ood_test")),
                        "indist_mean": _mean(arr(a, "indist")),
                        "class_path_flops": runs[a][0]["class_path_flops"]}
                    for a in ARMS_8B},
    }
    (RESULTS_DIR / "stage8b_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _train68(arm, seed, data, splits, steps, throttle_ms):
    import copy
    train_c, val_c, test_c = splits
    m = _build_8b(arm, seed)
    ema = copy.deepcopy(m)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.99))
    gen = torch.Generator().manual_seed(seed)
    tc_all = stage6.cond_vec(train_c)
    diverged = False
    for step in range(steps):
        if throttle_ms and step % 5 == 0:
            torch.cuda.synchronize() if DEVICE == "cuda" else None
            time.sleep(throttle_ms / 1000.0)
        idx = torch.randint(len(train_c), (256,), generator=gen)
        x0 = stage6.combo_images(data, [train_c[i] for i in idx.tolist()])
        c = tc_all[idx.to(DEVICE)]
        t = torch.randint(1, stage6.T_STEPS, (256,), generator=gen).to(DEVICE)
        ab = stage6.ABAR[t].view(-1, 1, 1, 1)
        eps = torch.randn(x0.shape, generator=gen).to(DEVICE)
        loss = torch.mean((m(ab.sqrt() * x0 + (1 - ab).sqrt() * eps, t, c) - eps) ** 2)
        if not math.isfinite(loss.item()):
            diverged = True
            break
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), m.parameters()):
                pe.mul_(0.999).add_(pm, alpha=0.001)
    row = dict(arm=arm, seed=seed, diverged=diverged,
               class_path_flops=m.class_path_flops(),
               params=sum(p.numel() for p in m.parameters()))
    if diverged:
        row |= dict(indist=math.nan, ood_val=math.nan, ood_test=math.nan)
        return row

    @torch.no_grad()
    def gen_mse(combos):
        eg = torch.Generator().manual_seed(123_000)
        pick = torch.randperm(len(combos), generator=eg)[:stage6.EVAL_COMBOS].tolist()
        sel = [combos[i] for i in pick]
        gt = stage6.combo_images(data, sel)
        cv = stage6.cond_vec(sel)
        noise = torch.randn(gt.shape, generator=eg).to(DEVICE)
        outs = []
        for i in range(0, len(sel), 128):
            outs.append(stage6.ddim_sample(ema, cv[i:i + 128], noise[i:i + 128]))
        return torch.mean((torch.cat(outs) - gt) ** 2).item()

    row |= dict(indist=gen_mse(train_c), ood_val=gen_mse(val_c), ood_test=gen_mse(test_c))
    return row


def main():
    mode = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    if mode == "a":
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
        s = run_8a(n, steps)
    else:
        steps = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
        s = run_8b(n, steps)
    print("\n" + "=" * 60)
    print(f"STAGE-8{mode.upper()} VERDICT: {s['final_verdict'].upper()}  criteria={s['criteria']}")
    for a, p in s["per_arm"].items():
        print(f"  {a:10} ood={p['ood_test_mean']:.6f} indist={p['indist_mean']:.6f}")


if __name__ == "__main__":
    main()
