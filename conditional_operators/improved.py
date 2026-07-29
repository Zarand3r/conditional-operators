"""Two relaxations of the operator, aimed at the one failure that recurs everywhere.

The fit penalty is not a tuning problem. `T(c) = P R(Wc) P^T` with `P` orthogonal and `R` a
rotation is a *strict isometry*: it can turn features but never rescale them. That single fact
explains both persistent failures --- the in-distribution fit penalty (1.46x on 3D Shapes, 2.43x
on frozen latents, 33% in diffusion) and the 8x content failure, since injecting information means
changing how strongly features are present.

Orthogonality was chosen for numerical stability. It was never needed for the composition
guarantee, and both relaxations below keep that guarantee exactly.

**Scaled rotation.** Use the full complex algebra per 2x2 block, `A(v) = s*I + theta*J`. Since `I`
and `J` commute, `exp(A) = e^s R(theta)` and the algebra stays abelian, so
`T(c1+c2) = T(c1)T(c2)` still holds for any weights. The group is `(C*)^{d/2}` rather than
`SO(2)^{d/2}`. The operator can now change magnitudes.

**Conjugated rotation.** Replace `P (.) P^T` with `P D (.) D^{-1} P^T` for a learned positive
diagonal `D`. Conjugation is a group automorphism, so composition, `T(0)=I` and `T(c)^a = T(ac)`
all survive untouched; the result is a rotation in a stretched metric, which is no longer an
isometry in the original one. `D` is diagonal on purpose: a dense conjugator would cost `O(d^2)`
and repeat the stage-2 erratum, where a dense basis blew the FLOP ceiling and turned a favourable
verdict into UNFAIR. Diagonal costs `O(d)`.

**The two are not equally safe, and the difference is worth stating.** A rotation is bounded by
nature, so the pure-rotation arms are exact for any weights whatever. A magnitude is not: `e^s`
overflows, so `s` is clamped, and a clamp is nonlinear --- `clamp(s1+s2) != clamp(s1)+clamp(s2)`.
Measured, composition error is ~1e-6 while `|s| <= 4` and jumps to order 1 the moment the clamp
engages. So:

* `proposed_conj` relaxes the isometry and keeps composition **unconditionally** exact.
* `proposed_scaled` and `proposed_scaled_conj` buy more freedom (norm changes of 9-15% against
  conjugation's 3%) but are exact **only inside the clamp's range**.

Complex FiLM carries the same caveat and stage 8 disclosed it. It is a real cost of touching the
magnitude channel, not an implementation detail, and it is the reason the conjugated arm may be
the better bet despite being the weaker relaxation.

    .venv/bin/python -m conditional_operators.improved --check
"""

from __future__ import annotations

import sys

import torch
from torch import nn

from .stage3 import GSOrthogonal, _rotate
from .stage4 import ARM_CLASSES, DC, DZ, CondArm, _fl
from .stage8 import _cmul

SCLAMP = 4.0            # |log magnitude| ceiling, as in Complex FiLM; active only far off-support


class ScaledLie4(CondArm):
    """T(c) = P exp(s(c) I + theta(c) J) P^T. Abelian, so composition is still exact."""

    def __init__(self, dc=DC):
        super().__init__(dc)
        self.W = nn.Linear(dc, DZ // 2, bias=False)      # phase: composition channel
        self.S = nn.Linear(dc, DZ // 2, bias=False)      # log-magnitude: the new freedom
        nn.init.zeros_(self.W.weight)
        nn.init.zeros_(self.S.weight)
        self.P = GSOrthogonal()

    def op(self, h, d, z):
        return self.P.apply_t(_cmul(self.S(d), self.W(d), self.P.apply(z)))

    def op_flops(self):
        return (_fl(self.dc, DZ // 2) * 2 + 5 * DZ + 2 * GSOrthogonal.apply_flops())


class ConjLie4(CondArm):
    """T(c) = P D R(Wc) D^{-1} P^T. A rotation in a learned stretched metric, not an isometry."""

    def __init__(self, dc=DC):
        super().__init__(dc)
        self.W = nn.Linear(dc, DZ // 2, bias=False)
        nn.init.zeros_(self.W.weight)
        self.P = GSOrthogonal()
        self.log_d = nn.Parameter(torch.zeros(DZ))       # D = exp(log_d), starts at the identity

    def op(self, h, d, z):
        scale = torch.exp(self.log_d.clamp(-SCLAMP, SCLAMP))
        zp = self.P.apply(z) / scale                     # D^{-1}
        return self.P.apply_t(_rotate(self.W(d), zp) * scale)

    def op_flops(self):
        return _fl(self.dc, DZ // 2) + 5 * DZ + 2 * GSOrthogonal.apply_flops()


class ScaledConjLie4(ConjLie4):
    """Both relaxations at once: magnitudes free, and the metric stretched."""

    def __init__(self, dc=DC):
        super().__init__(dc)
        self.S = nn.Linear(dc, DZ // 2, bias=False)
        nn.init.zeros_(self.S.weight)

    def op(self, h, d, z):
        scale = torch.exp(self.log_d.clamp(-SCLAMP, SCLAMP))
        zp = self.P.apply(z) / scale
        return self.P.apply_t(_cmul(self.S(d), self.W(d), zp) * scale)

    def op_flops(self):
        return (_fl(self.dc, DZ // 2) * 2 + 7 * DZ + 2 * GSOrthogonal.apply_flops())


ARM_CLASSES["proposed_scaled"] = ScaledLie4
ARM_CLASSES["proposed_conj"] = ConjLie4
ARM_CLASSES["proposed_scaled_conj"] = ScaledConjLie4

NEW_ARMS = ("proposed_scaled", "proposed_conj", "proposed_scaled_conj")


def check() -> bool:
    """The relaxations must buy expressiveness without costing the guarantee."""
    ok = True
    torch.manual_seed(0)
    for name in NEW_ARMS:
        arm = ARM_CLASSES[name](dc=DC)
        with torch.no_grad():                            # give it real, non-trivial weights
            for p in arm.parameters():
                p.normal_(0, 0.25)
            arm.beta.weight.zero_(); arm.beta.bias.zero_()   # beta is a shift, not part of T

        z = torch.randn(64, DZ)
        c1, c2 = torch.randn(64, DC) * 0.3, torch.randn(64, DC) * 0.3

        # composition: T(c1+c2) == T(c1) T(c2)
        both = arm.op(arm.enc(c1 + c2), c1 + c2, z)
        seq = arm.op(arm.enc(c1), c1, arm.op(arm.enc(c2), c2, z))
        comp = (both - seq).abs().max().item() / z.abs().max().item()

        # identity at c = 0
        ident = (arm.op(arm.enc(torch.zeros_like(c1)), torch.zeros_like(c1), z) - z
                 ).abs().max().item()

        # powering: T(a c) == T(c)^a, checked for integer a by repeated application
        z3 = arm.op(arm.enc(3 * c1), 3 * c1, z)
        step = z
        for _ in range(3):
            step = arm.op(arm.enc(c1), c1, step)
        power = (z3 - step).abs().max().item() / z.abs().max().item()

        # the point of the change: the operator must no longer preserve norms
        norm_ratio = (arm.op(arm.enc(c1), c1, z).norm(dim=1) / z.norm(dim=1))
        breaks_isometry = (norm_ratio - 1.0).abs().max().item()

        good = comp < 1e-4 and ident < 1e-5 and power < 1e-4 and breaks_isometry > 0.01
        ok &= good
        print(f"  {name:22} composition {comp:.2e}  identity {ident:.2e}  "
              f"power {power:.2e}  norm change {breaks_isometry:.3f}  "
              f"{'OK' if good else 'FAILED'}")

    base = ARM_CLASSES["proposed"](dc=DC)
    nr = (base.op(base.enc(torch.zeros(8, DC)), torch.randn(8, DC), torch.randn(8, DZ)).norm(dim=1)
          / torch.randn(8, DZ).norm(dim=1))
    film = ARM_CLASSES["film"](dc=DC)
    print(f"\n  budget vs film (ceiling 1.20x):")
    for name in NEW_ARMS:
        arm = ARM_CLASSES[name](dc=DC)
        r = arm.flops() / film.flops()
        ok &= r <= 1.20
        print(f"    {name:22} {r:.3f}x  params {arm.n_params()}")
    print("\nRELAXATIONS:", "VALID" if ok else "BROKEN")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
