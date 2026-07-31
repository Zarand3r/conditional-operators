"""Complex FiLM against FiLM on real analog-compressor data, in this field's own setup.

Every application attempt in this project invented a task and then asked whether the method won.
This one does the opposite: black-box neural audio effect modelling already uses FiLM to condition
a TCN on a device's control knobs, and "which conditioning mechanism is best" is already a
published question in that field. So the comparison runs on their data (the SignalTrain LA-2A
recordings), their architecture (a causal dilated TCN), and their incumbent (FiLM). Our claim is
narrow and pre-existing: Complex FiLM beats FiLM in the transformation role at lower cost.

**Why the arms are rewritten here.** The conditioners used elsewhere in this project are hardwired
to a 128-wide latent, and the orthogonal basis is built for that width. A TCN in this field runs
32 channels, where a 2.1M-parameter hypernetwork would be sixteen times the model it conditions --
so porting the old arms unchanged would make the budget rule meaningless and the comparison silly.
Each mechanism is therefore reimplemented as a function of the backbone's channel count, which is
how the field states them anyway: FiLM emits a scale and shift per channel, Complex FiLM emits a
magnitude and phase per channel *pair*. Both are O(C), and the parameter counts stay comparable to
the block they modulate.

Conditioning is applied inside every residual block, which is where this field puts it.
"""

from __future__ import annotations

import math

import torch
from torch import nn

SR = 44_100


# ---------------------------------------------------------------- conditioners, width-parameterised

class _Cond(nn.Module):
    """Shared control-parameter encoder. Subclasses modulate a [B, C, T] activation."""

    def __init__(self, dc: int, ch: int, hidden: int = 32):
        super().__init__()
        self.dc, self.ch = dc, ch
        self.enc = nn.Sequential(nn.Linear(dc, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class FiLMCond(_Cond):
    """The incumbent: y = (1 + g(c)) * x + b(c), one scale and shift per channel."""

    def __init__(self, dc, ch, hidden=32):
        super().__init__(dc, ch, hidden)
        self.head = nn.Linear(hidden, 2 * ch)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, c):
        g, b = self.head(self.enc(c)).chunk(2, dim=-1)
        return (1 + g).unsqueeze(-1) * x + b.unsqueeze(-1)


class ComplexFiLMCond(_Cond):
    """The candidate: multiply channel pairs by m*e^{i theta}.

    Magnitude carries content through an expressive head; phase carries composition through a head
    that is linear in the control parameters, so phases add exactly when settings combine. Same
    O(C) cost as FiLM -- in fact slightly fewer parameters, since a pair needs one magnitude and one
    phase where FiLM needs two scales and two shifts.
    """

    def __init__(self, dc, ch, hidden=32, clamp=4.0):
        super().__init__(dc, ch, hidden)
        assert ch % 2 == 0, "Complex FiLM modulates channel pairs"
        self.clamp = clamp
        self.mag = nn.Linear(hidden, ch // 2)              # content channel: expressive
        self.phase = nn.Linear(dc, ch // 2, bias=False)    # composition channel: exact
        self.shift = nn.Linear(hidden, ch)
        for lin in (self.mag, self.phase, self.shift):
            nn.init.zeros_(lin.weight)
            if lin.bias is not None:
                nn.init.zeros_(lin.bias)

    def forward(self, x, c):
        h = self.enc(c)
        m = torch.exp(self.mag(h).clamp(-self.clamp, self.clamp)).unsqueeze(-1)
        th = self.phase(c).unsqueeze(-1)
        cos, sin = torch.cos(th), torch.sin(th)
        a, b = x[:, 0::2], x[:, 1::2]
        y = torch.empty_like(x)
        y[:, 0::2] = m * (cos * a - sin * b)
        y[:, 1::2] = m * (sin * a + cos * b)
        return y + self.shift(h).unsqueeze(-1)


class ConcatCond(_Cond):
    """What most of this field does: fold the parameters in through an MLP on the channels."""

    def __init__(self, dc, ch, hidden=32):
        super().__init__(dc, ch, hidden)
        self.w1 = nn.Linear(ch + hidden, ch)
        self.w2 = nn.Linear(ch, ch)
        nn.init.zeros_(self.w2.weight); nn.init.zeros_(self.w2.bias)

    def forward(self, x, c):
        h = self.enc(c).unsqueeze(-1).expand(-1, -1, x.shape[-1])
        z = torch.cat([x, h], dim=1).transpose(1, 2)
        return x + self.w2(torch.relu(self.w1(z))).transpose(1, 2)


class HyperCond(_Cond):
    """Reported for contrast: a per-setting 1x1 mixing matrix. O(C^2) and the reason the budget
    rule matters here -- at this width it is already comparable to the block it modulates."""

    def __init__(self, dc, ch, hidden=32):
        super().__init__(dc, ch, hidden)
        self.head = nn.Linear(hidden, ch * ch)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, c):
        w = self.head(self.enc(c)).view(-1, self.ch, self.ch)
        return x + torch.bmm(w, x)


CONDS = {"film": FiLMCond, "cfilm": ComplexFiLMCond, "concat": ConcatCond, "hyper": HyperCond}


# ---------------------------------------------------------------- the field's backbone

class TCNBlock(nn.Module):
    def __init__(self, ch, kernel, dilation, cond: nn.Module):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.res = nn.Conv1d(ch, ch, 1)
        self.cond = cond

    def forward(self, x, c):
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))     # causal
        y = self.cond(y, c)
        return torch.nn.functional.prelu(y, torch.tensor(0.2, device=x.device)) + self.res(x)


class TCN(nn.Module):
    """Causal dilated TCN, conditioning inside every block. Identical for every arm."""

    def __init__(self, arm: str, dc: int = 2, ch: int = 32, blocks: int = 10,
                 kernel: int = 13, growth: int = 2):
        super().__init__()
        self.inp = nn.Conv1d(1, ch, 1)
        self.blocks = nn.ModuleList(
            TCNBlock(ch, kernel, growth ** i, CONDS[arm](dc, ch)) for i in range(blocks))
        self.out = nn.Conv1d(ch, 1, 1)
        self.receptive = sum((kernel - 1) * growth ** i for i in range(blocks)) + 1

    def forward(self, x, c):
        z = self.inp(x)
        for b in self.blocks:
            z = b(z, c)
        return self.out(z)

    def cond_params(self):
        return sum(b.cond.n_params() for b in self.blocks)

    def backbone_params(self):
        return sum(p.numel() for p in self.parameters()) - self.cond_params()


def budget_table(dc=2, ch=32):
    """The conditioning budget in this domain, where the conditioner can outweigh the backbone."""
    rows = {}
    for name in CONDS:
        m = TCN(name, dc=dc, ch=ch)
        rows[name] = (m.cond_params(), m.backbone_params())
    film_c = rows["film"][0]
    print(f"  {'arm':8} {'cond params':>12} {'vs film':>9} {'backbone':>10} {'cond/backbone':>14}")
    for n, (c, b) in rows.items():
        print(f"  {n:8} {c:12,} {c/film_c:8.2f}x {b:10,} {c/b:13.2f}x")
    return rows


if __name__ == "__main__":
    print("conditioning budget at the width this field actually uses (32 channels):")
    rows = budget_table()
    m = TCN("cfilm")
    print(f"\n  receptive field: {m.receptive} samples "
          f"({m.receptive / SR * 1000:.0f} ms at {SR} Hz)")
