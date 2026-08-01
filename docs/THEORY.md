# When does structured conditioning help?

A predictive account of conditioning-mechanism selection, assembled from this project's results —
most of them failures.

The question is treated as empirical in the literature. Papers compare FiLM against concatenation
against hypernetworks and report a table; the audio-effects field has at least two such papers in
the last two years. Geometric deep learning gives a rigorous theory of when symmetries of the
*data domain* help, but conditioning is a different object: whatever structure exists lives in the
**condition space**, not the input space. We are not aware of a framework that predicts which
conditioning mechanism suits which problem.

This is a candidate. It makes a falsifiable prediction from two quantities that cost two short
training runs to measure, and it is validated against nine tasks, six of which the method lost.

---

## 1. Setup

A conditioned network computes `y = dec(T_c(enc(x)))`. A **conditioning mechanism** is a
parameterised family of feature maps

```
𝓕_θ = { T_c^θ : c ∈ C }
```

Mechanisms differ in the *reachable set* `R(𝓕) = { T_c^θ : c ∈ C, θ ∈ Θ }`:

| mechanism | `T_c(z)` | reachable set |
|---|---|---|
| FiLM | `γ(c) ⊙ z + β(c)` | diagonal affine maps |
| concat-MLP | `z + g([z, h(c)])` | (approximately) arbitrary maps |
| hypernetwork | `W(c) z` | all linear maps |
| CGA | `P exp(A(Wc)) Pᵀ z` | a `dim(c)`-dimensional abelian subgroup |

Write `C_train` for the conditions seen in training and `C_test` for those evaluated.

## 2. The mechanism: closure determines the unvisited condition space

**Definition.** A mechanism is **closed** if `C` carries a group operation `∘` such that
`T_{c ∘ c'} = T_c T_{c'}` holds *identically in θ* — for every parameter setting, not merely at
convergence.

**Proposition (closure).** If `𝓕` is closed and `C_train` generates `C` under `∘`, then
`{T_c : c ∈ C_train}` determines `T_c` for every `c ∈ C`, for any `θ`.

*Proof.* Immediate: any `c ∈ C` is a finite product of generators, and `T` is a homomorphism. ∎

Trivial as a statement, and it is the entire content of the method. A closed mechanism does not
*learn* its behaviour at unseen conditions — that behaviour is **implied**. An unstructured
mechanism's `T_c` off `C_train` is whatever its head interpolates, constrained only by smoothness.

For CGA the group is `(ℝᵏ, +)`, closure holds because `c ↦ A(Wc)` is linear into an abelian
algebra, and we verify it numerically to `~10⁻⁷` with arbitrary random weights.

**Corollary (the guarantee is conditional on coverage).** Closure gives nothing unless
`C_test ⊆ ⟨C_train⟩`. Training on conditions that do not generate the tested ones buys no
extrapolation, however exact the homomorphism.

## 3. The decomposition

Error at test conditions splits three ways:

```
E_test  =  E_approx  +  E_est  +  E_extrap
```

- **`E_approx`** — distance from the true conditional map to `R(𝓕)`. **Not a function of
  capacity**; see §3.1.
- **`E_est`** — finite-sample estimation error. Grows with capacity, classically.
- **`E_extrap`** — error at conditions outside `C_train`. **Closure sets this to zero by
  construction** (given §2's corollary). Without closure it grows with distance from `C_train`.

Structured conditioning is a trade: **`E_approx` up, `E_extrap` down.** It never adds capability.
It removes freedom, and removing freedom pays only when freedom was being misused.

### 3.1 `E_approx` is class membership, not headroom

The natural reading of "more structure ⟹ larger `E_approx`" is that structure shrinks a nested
sequence of reachable sets. That reading is wrong, and it misled us for most of this project.

The reachable sets of the standard mechanisms are **nearly disjoint, not nested**. Let `D` be the
diagonal maps (FiLM) and `O` the orthogonal maps (CGA). Then

```
D ∩ O = { diagonal matrices with entries ±1 }
```

a finite set of `2^d` points. Two `d`-dimensional families meeting in measure zero. Switching from
FiLM to CGA does not shrink the reachable set — **it moves it**.

Parameter counts obscure this completely. Complex FiLM uses `0.85×` FiLM's parameters, which reads
as a 15% trim and is nothing of the kind: it is a substitution of one operation class for a mostly
non-overlapping one, with a similar-sized head to emit it.

**The right coordinates are the polar decomposition.** Every invertible `T` factors uniquely as

```
T = U S        U orthogonal (rotation),  S symmetric positive definite (stretch)
```

and each mechanism reaches essentially one factor:

| mechanism | reaches | singular values |
|---|---|---|
| FiLM | `S` only, and only diagonal `S` | any positive reals |
| CGA | `U` only | all exactly 1 |
| Complex FiLM | `U S`, with `S` constant on channel pairs | `d/2` stretch DOF, not `d` |
| hypernetwork | both, unrestricted | any |

So `E_approx` is governed by **which polar factor the task's transformation actually needs**, not by
how many parameters the head has:

```
E_approx(CGA)  =  min_{Q orthogonal} ‖T* − Q‖_F  =  ‖S* − I‖_F
```

which is exact, not a heuristic: the minimiser is the polar factor `U*` and the residual is
`‖S* − I‖_F` (polar / orthogonal-Procrustes). Verified numerically. Analogously
`E_approx(FiLM)` is governed by how far `T*` sits from the diagonal maps, and Complex FiLM's by
the part of `S*` that varies *within* a channel pair, since it holds one scale per pair.

**Two caveats on this identity.** It is stated for the linear part only — every mechanism here is
affine (`T_c z + β(c)`), and the shift is unaccounted for. And FiLM's `γ` may be negative, so FiLM
reaches diagonal *sign flips*: a discrete subset of the rotations, not none. Neither changes the
qualitative picture, and both should be tightened before the claim is leaned on.

This is computable in advance for any task whose operator can be written down, and it is why an
isometry fails at content by 8× at *any* parameter count: "suppress this channel" has `‖log S*‖`
large and `U* = I`, so it lies at maximal distance from `O`. Not a shortfall — an exclusion.

## 4. Measurable proxies, and the prediction

Both sides of the trade are observable from short training runs:

| quantity | proxy | how |
|---|---|---|
| `E_approx` | **fit ratio** | in-distribution error of the structured arm ÷ that of the best arm. Near 1 ⟹ the truth lies near `R`. |
| `E_extrap` | **compositional gap** | held-out error ÷ in-distribution error, for the best-fitting arm. Near 1 ⟹ nothing is being extrapolated. |

**Prediction (two independent axes).**

1. **Match the class.** Choose the mechanism whose reachable factor matches the polar type the task
   requires — stretch-dominated tasks want a diagonal mechanism, rotation-dominated tasks want an
   orthogonal one, mixed tasks want scaled rotations and pay resolution in each.
2. **Then ask whether closure is worth it.** Only if the evaluation visits unvisited conditions.

The fit ratio is the empirical proxy for axis 1 and the compositional gap for axis 2. Calibrated:

> **fit ratio ≤ ~1.05  AND  gap ≥ ~1.5**, with both conditions independently necessary.

The gap also has an implicit upper bound. A very large gap means even the best model fails badly
off-distribution, which by §2's corollary indicates `C_test ⊄ ⟨C_train⟩` — extrapolation rather
than composition. Closure does not help there either.

## 5. Validation

Nine tasks, evaluated under one protocol with pre-registered criteria and a shared budget counter.

| task | fit ratio | gap | predicted | actual |
|---|---|---|---|---|
| dSprites | 0.986 | 2.1× | win | won, 53.8% |
| dSprites + shape | 1.005 | ~2× | win | won, 42.9% |
| synthetic PDE (Complex FiLM) | 0.938 | 4.3× | win | won, 64.6% |
| synthetic PDE (rotation) | 1.118 | 4.3× | loss | lost (fit criterion) |
| 3D Shapes | 1.457 | 2.1× | loss | lost (fit criterion) |
| latent editing (frozen) | 2.433 | 2.1× | loss | lost |
| PDEBench | 2.381 | ~140× | loss | lost |
| camera sliders | — | 1.26× | no contrast | withdrawn |
| audio effects / LA-2A | 1.000 | 1.13× | tie | tied |

Nine for nine. What makes this more than a curve fit is that **each factor is independently
observed to fail while the other passes**: camera had an acceptable fit and no gap; audio effects
had a fit ratio of exactly 1.000 and no gap; PDEBench had an enormous gap and an unusable fit. A
one-factor account cannot produce that pattern.

## 5.1 The polar account, checked against the tasks

The synthetic Navier-Stokes operator is diagonal in Fourier, so its polar factors are exact and
computable — magnitude and phase of `exp(T·L(c))`:

| axis | ‖log magnitude‖ | ‖phase‖ | polar type |
|---|---|---|---|
| viscosity | ∞ | 0.000 | **pure stretch** |
| drag | 0.600 | 0.000 | **pure stretch** |
| advection | 0.000 | 1.594 | **pure rotation** |
| forcing | — | — | inhomogeneous, enters neither |

The task needs **both factors**. And that is exactly what happened: the pure-rotation arm could not
represent the viscous and drag decay and failed on fit at `1.118×`, while Complex FiLM — scaled
rotations, reaching both factors — fit at `0.938×` and won by `64.6%`. The polar type predicted
which of our own arms would fail, and why.

The same account covers the rest:

| task | polar type required | predicted best class | observed |
|---|---|---|---|
| dSprites | rotation (geometric rearrangement) | orthogonal | CGA won 53.8% |
| content suite | stretch (specify what is present) | diagonal | CGA lost 8× |
| synthetic PDE | both | scaled rotation | Complex FiLM won 64.6% |
| LA-2A compressor | **pure gain = pure diagonal stretch** | diagonal | FiLM won; Complex FiLM tied at half the stretch resolution |
| PDEBench reaction-diffusion | **saturating nonlinearity — not a linear operator at all** | none of them | `concat_mlp`, unrestricted, won |

The last row matters: when the required map is not a linear operator in any basis, every
operator-family mechanism is equally out of class, and the unrestricted one wins. That is not a
failure of structure so much as a task outside the framework's scope.

## 5.2 A structural derivation, and a hard ceiling

Working the algebra rather than the empirics gives a chain in which every step is forced. It
subsumes several results we had been treating as separate design choices.

**(i) Exact composition forces linearity.** `T(c₁+c₂) = T(c₁)T(c₂)` requires
`A(c₁+c₂) = A(c₁) + A(c₂)` in algebra coordinates. By Cauchy's functional equation, a continuous
additive map is linear. There is no nonlinear exactly-compositional conditioner.

**(ii) Content forces nonlinearity.** Specifying *what* to produce is not an additive operation —
composing "cat" with "dog" is not "cat + dog" — so the condition-to-modulation map must be
expressive. Measured: the fully linear Complex FiLM scores 0.19 in the content role against the
expressive variant's 0.97.

**(iii) Therefore content and composition cannot share a channel.** Not a design tension — a
theorem, by (i) and (ii). Any conditioner that must do both has to *split its representation*.

**(iv) The minimal split is `ℂ*`.** The smallest algebra carrying both an expressive
(non-compact) and an exactly-compositional (compact) direction is `ℂ = ℝ ⊕ iℝ`, whose
multiplicative group factors as `ℂ* ≅ ℝ_{>0} × U(1)`. That is precisely magnitude × phase.
**Complex FiLM is not a heuristic; it is the forced resolution of (iii).**

**(v) Compactness is why the assignment goes that way.** Both factors are groups, so either could
in principle carry composition. The asymmetry is boundedness: rotations are bounded by nature, so
exact composition survives; `e^s` is unbounded, needs a clamp, and a clamp is nonlinear. Measured:
composition error `~10⁻⁶` while `|s| ≤ 4`, order 1 the moment the clamp engages. **The compact
factor is the only one on which exactness is practically stable.**

**(vi) The complex algebra is maximal.** Its dimension is 2 per `2×2` block × `d/2` blocks = `d`,
which is exactly the dimension of a Cartan subalgebra of `gl(d)`. Verified: the commutator
vanishes identically and the count matches. **No exactly-compositional linear conditioner can
reach a larger family.** Going further requires a non-abelian algebra, which forfeits exactness
for arbitrary weights.

### What exactness actually costs (a correction)

An earlier version of this section claimed that exact compositionality "caps expressiveness at
`dim(c)`, independent of network width", and used that to explain the PDEBench failure. **That was
wrong.** Every mechanism's transformation family is parameterised by `c ∈ ℝᵏ`, so every one has
image dimension at most `k` — FiLM's `{diag(γ(c))}`, a hypernetwork's `{W(c)}`, and CGA's
`{exp(A(Wc))}` alike. A universal bound distinguishes nothing.

Exactness does not restrict the family's *dimension*. It restricts its *shape*:

- FiLM's family is an arbitrary `k`-dimensional manifold; it may curve freely as `c` varies.
- CGA's is a `k`-dimensional **flat** subspace of the algebra, exponentiated into a subgroup.

Flatness forces `T(3c) = T(c)³`. That is closure viewed from the cost side, so it is the *same*
trade as `E_extrap` rather than an additional one.

The genuine sources of `E_approx`, then, are two, and neither is dimensional:

1. **Which individual maps are reachable** — orthogonal vs diagonal vs unrestricted (§3.1). This is
   the large effect, and it is what the fit ratio measures.
2. **Whether the conditional transformation is linear in `z` at all.** FiLM, CGA and hypernetworks
   apply a linear map to the latent; `concat_mlp` applies a *nonlinear function of `z`*.

(2) repairs the PDEBench account. The saturating reaction term is not a linear operator on the
latent in any basis, so every operator-family mechanism is out of class and the unrestricted
`concat_mlp` wins. That also explains why `proposed_scaled` and `proposed_conj` changed nothing:
they explored different linear classes when no linear class was going to work.

## 6. Three corollaries, each independently measured

**6.1 An isometric conditioner cannot inject content.** If `T_c` is orthogonal then
`‖T_c z‖ = ‖z‖`, so conditioning can redistribute representational energy but not create it. Tasks
requiring the condition to *specify what to generate* lie outside `R`. Measured: 8× regression on
content conditioning, and the ablation fails identically. The remedy is a magnitude channel
(Complex FiLM), which recovers content at the cost of exactness on that channel.

**6.2 Capacity has opposite signs in the two regimes** — and only once the class is right. In the interpolation regime
(`gap ≈ 1`), `E_extrap ≈ 0` for every mechanism, so only `E_approx + E_est` matter and capacity
helps. In the composition regime (`gap > 1`), `E_extrap` dominates and capacity increases it.
Measured: hypernetworks lose by 57–90% on held-out combinations and memorise on an FNO backbone
(2815× train/val gap) — yet **win** on the LA-2A, where there is no gap at all. Both halves are
now observed. The earlier claim ("capacity hurts conditioning") was the first half without its
boundary.

**6.3 Dense coverage dissolves the benefit.** The samples needed to cover `C` grow exponentially
in `dim(C)`. For `dim(C) ≲ 3` the condition space is enumerable, so `gap ≈ 1` and structure is
useless by §4. Measured: the LA-2A has two controls and 42 reachable settings, of which the dataset
contains 38 — nothing to compose. dSprites has five factors, whose combinations cannot be covered.
**Structured conditioning requires roughly four or more genuinely interacting condition axes to
have anything to do.**

## 7. Status of each claim

| claim | status |
|---|---|
| Closure proposition (§2) | **proven**; verified numerically to 10⁻⁷ at arbitrary weights |
| Reachable family is `dim(c)`-dimensional | **proven** |
| Isometry cannot inject content (6.1) | **proven**, and measured |
| Error decomposition (§3) | standard, applied to this setting |
| Fit ratio and gap as proxies (§4) | **empirical**: 9 observations, thresholds calibrated post hoc |
| Capacity sign flip (6.2) | **empirical**, both regimes observed |
| Coverage criterion (6.3) | argued from counting, consistent with all 9 tasks |

The thresholds are guidance, not bounds. Nine observations, and both were fitted after the fact;
they are reported so that a future task can falsify them.

## 8. What is missing

The theory's weakest joint is that `E_extrap` is measured by a proxy rather than bounded. What is
wanted is a statement of the form

```
E_test  ≤  E_approx  +  ε(coverage of a generating set)
```

with `ε → 0` when `C_train` generates `C_test`. This is formalisable: "does the training set span a
generating set" is a precise question about the group, and the compositional gap is a crude
empirical stand-in for it. Deriving the bound, and identifying what replaces it under *partial*
generation, is the natural next step and would turn a predictive rule into a theorem.

A second gap: the account treats the condition encoder as given. The choice between a linear map
into the algebra (exact composition, `dim(c)`-dimensional family) and a nonlinear one (expressive,
composition lost) is a real design axis this framework does not yet resolve. Measured evidence
exists on both sides — the linear head is the reason the PDEBench family was dimensionally starved,
while the nonlinear head is why the magnitude channel breaks exactness.

## 9. Why this is the contribution rather than the operator

The project set out to show that a group-structured conditioner wins. It wins on three tasks and
loses on six, and no application survived contact with a benchmark we did not construct.

But the failures are not noise: **every one of them is predicted**, and the prediction costs two
short training runs. That is more useful to a practitioner than the operator ever was, because the
common case is deciding whether to reach for structure at all — and the answer, most of the time,
is no, for reasons that can now be checked in an afternoon rather than discovered in a month.

The honest one-line summary: *structured conditioning is a regulariser against combinatorial
memorisation, and combinatorial memorisation is rarely the binding failure in practice.*
