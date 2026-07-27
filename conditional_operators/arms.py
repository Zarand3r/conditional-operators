"""Six conditioning arms with a shared encoder + auditable param/FLOP counter (R5, STAGE1_SPEC).

Every arm maps a condition c (multi-hot, K) through the SAME encoder to a hidden vector h, then an
arm-specific head produces operator parameters applied to x as y = T(c)x + beta(c). All heads are
zero-initialized so T(c)=I and beta(c)=0 at step 0 (INV-1: identity init).

FLOPs are counted analytically (FLOPs = 2*in*out per matmul) with one shared helper so cross-arm
ratios are apples-to-apples. FiLM is the cheap floor; AC-4 requires proposed FLOPs <= 1.20x FiLM.

Design note on the proposed operator: Q(c) is block-diagonal orthogonal via
closed-form 2x2 rotations (exact orthogonality, no per-sample matrix_exp). The low-rank deformation
uses a SHARED learned basis U,V with input-conditioned gains s(c) (r outputs) and a bounded scale, so
it stays input-conditioned yet within budget; a fully per-sample U(c)V(c)^T head emits 2*D*r numbers
and provably exceeds the <=1.20x-FiLM budget (documented finding, see RESULTS.md).
"""

from __future__ import annotations

import torch
from torch import nn

from .data import D, K

H = 128          # encoder hidden width (shared across arms)
RANK = 4         # low-rank rank for proposed + dynamic-linear
LOWRANK_SCALE = 0.008  # bounds proposed low-rank spectral norm for INV-3 (sigma_max <= 1.01)


def _flops_linear(in_f: int, out_f: int) -> int:
    return 2 * in_f * out_f


def _zero_(layer: nn.Linear) -> nn.Linear:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class Encoder(nn.Module):
    """Shared condition encoder: multi-hot(K) -> h (H)."""

    def __init__(self) -> None:
        super().__init__()
        self.l1 = nn.Linear(K, H)
        self.l2 = nn.Linear(H, H)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.l2(torch.relu(self.l1(c))))

    @staticmethod
    def flops() -> int:
        return _flops_linear(K, H) + _flops_linear(H, H)


class Arm(nn.Module):
    """Base arm: shared encoder + zero-init bias head. Subclasses add the operator head."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = Encoder()
        self.beta = _zero_(nn.Linear(H, D))

    def operator(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, c: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(c)
        return self.operator(h, x) + self.beta(h)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def flops(self) -> int:
        """Per-sample forward FLOPs: shared encoder + beta head + operator (subclass adds op_flops)."""
        return Encoder.flops() + _flops_linear(H, D) + self.op_flops()

    def op_flops(self) -> int:
        raise NotImplementedError


class FiLM(Arm):
    """Diagonal floor: y = gamma(c) (.) x + beta(c)."""

    def __init__(self) -> None:
        super().__init__()
        self.gamma = _zero_(nn.Linear(H, D))  # gamma = 1 + head -> identity at init

    def operator(self, h, x):
        return (1.0 + self.gamma(h)) * x

    def op_flops(self):
        return _flops_linear(H, D) + D  # gamma head + elementwise scale


class ConcatMLP(Arm):
    """Concatenation baseline: y = x + W2 relu(W1 [x; h])."""

    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(D + H, H)
        self.w2 = _zero_(nn.Linear(H, D))  # zero -> y = x at init

    def operator(self, h, x):
        return x + self.w2(torch.relu(self.w1(torch.cat([x, h], dim=1))))

    def op_flops(self):
        return _flops_linear(D + H, H) + _flops_linear(H, D) + D


class CondLayerNorm(Arm):
    """Conditional LayerNorm: y = (1 + gamma(c)) (.) LN(x) + beta(c)."""

    def __init__(self) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(D, elementwise_affine=False)
        self.gamma = _zero_(nn.Linear(H, D))

    def operator(self, h, x):
        return (1.0 + self.gamma(h)) * self.ln(x)

    def op_flops(self):
        return _flops_linear(H, D) + 5 * D  # gamma head + layernorm (~5D) + scale


class Hypernet(Arm):
    """Unstructured full W(c): y = (I + dW(c)) x. Large head, O(D^2) apply."""

    def __init__(self) -> None:
        super().__init__()
        self.w = _zero_(nn.Linear(H, D * D))  # dW = 0 -> W = I at init

    def operator(self, h, x):
        dw = self.w(h).view(-1, D, D)
        return x + torch.einsum("nij,nj->ni", dw, x)

    def op_flops(self):
        return _flops_linear(H, D * D) + 2 * D * D  # W head + matvec apply


class DynamicLinear(Arm):
    """Unstructured low-rank W(c) = I + A(c) B(c)^T (LoRA-style, input-conditioned, per-sample)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = _zero_(nn.Linear(H, D * RANK))  # A = 0 -> W = I at init
        self.b = nn.Linear(H, D * RANK)

    def operator(self, h, x):
        a = self.a(h).view(-1, D, RANK)
        b = self.b(h).view(-1, D, RANK)
        btx = torch.einsum("ndr,nd->nr", b, x)
        return x + torch.einsum("ndr,nr->nd", a, btx)

    def op_flops(self):
        return (_flops_linear(H, D * RANK) * 2      # A and B heads
                + 2 * D * RANK + 2 * D * RANK)      # B^T x and A (.) apply


class Proposed(Arm):
    """Block-orthogonal + input-conditioned low-rank: T(c) = Q(c) + U diag(s(c)) V^T.

    Q(c): D/2 closed-form 2x2 rotations (exact orthogonality, INV-2). Low-rank: shared learned basis
    U,V with per-sample gains s(c) and bounded scale (INV-3). Identity at init (INV-1).
    """

    def __init__(self) -> None:
        super().__init__()
        self.angles = _zero_(nn.Linear(H, D // 2))  # angles = 0 -> Q = I at init
        self.gains = _zero_(nn.Linear(H, RANK))     # s = 0 -> low-rank inert at init
        self.U = nn.Parameter(torch.randn(D, RANK) * 0.02)
        self.V = nn.Parameter(torch.randn(D, RANK) * 0.02)

    def _rotate(self, h, x):
        ang = self.angles(h)                 # [N, D/2]
        c, s = torch.cos(ang), torch.sin(ang)
        x0, x1 = x[:, 0::2], x[:, 1::2]
        y0 = c * x0 - s * x1
        y1 = s * x0 + c * x1
        y = torch.empty_like(x)
        y[:, 0::2], y[:, 1::2] = y0, y1
        return y

    def _lowrank(self, h, x):
        # Column-normalized basis + bounded gains keep spectral norm <= LOWRANK_SCALE*RANK (INV-3).
        u = self.U / (self.U.norm(dim=0, keepdim=True) + 1e-8)
        v = self.V / (self.V.norm(dim=0, keepdim=True) + 1e-8)
        s = LOWRANK_SCALE * torch.tanh(self.gains(h))   # [N, r]
        vtx = x @ v                                       # [N, r]
        return (s * vtx) @ u.T                            # [N, D]

    def operator(self, h, x):
        return self._rotate(h, x) + self._lowrank(h, x)

    def op_flops(self):
        return (_flops_linear(H, D // 2)          # angle head
                + 3 * D                            # 2x2 rotation apply (~3 flops/coord)
                + _flops_linear(H, RANK)           # gain head
                + 2 * D * RANK + 2 * D * RANK)     # V^T x and U apply

    def dense_operator(self, c: torch.Tensor) -> torch.Tensor:
        """Materialize T(c) as a dense D x D matrix for a single condition (for invariant tests)."""
        h = self.enc(c)
        eye = torch.eye(D)
        # Apply operator to identity columns: T = operator(I).
        q = self._rotate(h.expand(D, H), eye) + self._lowrank(h.expand(D, H), eye)
        return q.T  # columns are T(e_i)

    def rotation_only(self, c: torch.Tensor) -> torch.Tensor:
        h = self.enc(c)
        return self._rotate(h.expand(D, H), torch.eye(D)).T


ARM_CLASSES = {
    "film": FiLM,
    "concat_mlp": ConcatMLP,
    "cond_layernorm": CondLayerNorm,
    "hypernet": Hypernet,
    "dynamic_linear": DynamicLinear,
    "proposed": Proposed,
}


def build(arm_name: str, seed: int) -> Arm:
    torch.manual_seed(seed)
    return ARM_CLASSES[arm_name]()
