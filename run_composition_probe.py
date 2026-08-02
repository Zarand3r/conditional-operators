"""Do trained conditioners compose additively or multiplicatively?

The attention hypothesis rests on a claim that can be checked without any attention: additive
aggregation of condition effects is the first-order expansion of multiplicative (group)
composition, `I + ΣA` against `exp(ΣA)`, so they agree only while the conditioning is weak.

This probes what a *trained* model actually does. For each arm we extract the effective linear
operator `J(c) = ∂/∂z [arm(c, z)]` by autograd — valid even for arms that are nonlinear in `z` —
and compare the two-factor operator against the two candidate compositions of its one-factor
operators:

    additive        J(c1) + J(c2) - I
    multiplicative  J(c1) @ J(c2)

`proposed` is the control: it is multiplicative by construction, so it must come out that way or
the probe is broken.
"""

import json

import torch

from conditional_operators import stage4
from conditional_operators.stage4 import ARM_CLASSES, DEVICE, DC, DZ

ARMS = ("film", "concat_mlp", "hypernet", "proposed")
STEPS = 3000


def jacobian(arm, c, z0):
    """Effective linear operator of the conditioning at z0, for condition c."""
    z = z0.clone().requires_grad_(True)
    rows = []
    out = arm(c.unsqueeze(0), z.unsqueeze(0)).squeeze(0)
    for i in range(DZ):
        g = torch.autograd.grad(out[i], z, retain_graph=(i < DZ - 1))[0]
        rows.append(g)
    return torch.stack(rows)


def probe(arm, seed=0, n_pairs=12):
    g = torch.Generator().manual_seed(seed)
    z0 = torch.randn(DZ, generator=g).to(DEVICE)
    add_err, mul_err, scale = [], [], []
    for _ in range(n_pairs):
        c1 = torch.zeros(DC, device=DEVICE); c2 = torch.zeros(DC, device=DEVICE)
        i, j = torch.randperm(DC, generator=g)[:2].tolist()
        c1[i] = float(torch.randn((), generator=g) * 0.5)
        c2[j] = float(torch.randn((), generator=g) * 0.5)
        J1, J2, J12 = (jacobian(arm, c, z0) for c in (c1, c2, c1 + c2))
        eye = torch.eye(DZ, device=DEVICE)
        add_err.append((J12 - (J1 + J2 - eye)).norm().item())
        mul_err.append((J12 - J1 @ J2).norm().item())
        scale.append((J12 - eye).norm().item())
    m = lambda v: sum(v) / len(v)
    return m(add_err), m(mul_err), m(scale)


def main():
    data = stage4.Data()
    rows = {}
    for name in ARMS:
        torch.manual_seed(0)
        model = stage4.Model(name).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        gen = torch.Generator().manual_seed(0)
        bce = torch.nn.BCEWithLogitsLoss()
        for _ in range(STEPS):
            x1, d, x2 = data.sample(stage4.TRAIN_TYPES, 256, gen)
            loss = bce(model(x1, d), x2)
            opt.zero_grad(); loss.backward(); opt.step()
        a, mu, sc = probe(model.cond)
        rows[name] = {"additive_err": a, "multiplicative_err": mu, "effect_size": sc}
        verdict = "MULTIPLICATIVE" if mu < a else "additive"
        print(f"  {name:12} additive {a:8.4f} | multiplicative {mu:8.4f} | "
              f"effect {sc:7.4f} | closer to {verdict}", flush=True)
    json.dump(rows, open("results/composition_probe.json", "w"), indent=2)


if __name__ == "__main__":
    main()
