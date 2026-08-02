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
| latent editing (frozen)† | 2.433 | 2.1× | loss | lost |
| PDEBench | 2.381 | ~140× | loss | lost |
| camera sliders | — | 1.26× | no contrast | withdrawn |
| audio effects / LA-2A | 1.000 | 1.13× | tie | tied |

† **The frozen/co-trained pair is not a single-variable A/B**, though it is quoted that way
elsewhere in this repository (8× better co-trained, 1.6× worse frozen). The co-trained run
optimises BCE on images and is scored in pixel space; the frozen run optimises and is scored in
latent space. Representation, objective and metric all differ. The within-run ratios are the
defensible part; the pair is suggestive, not controlled.

**A second caveat on the whole table:** the *splits* were pre-registered, the *tasks* were not. All
three wins are on tasks whose dominant factors are geometric — exactly the rotation-dominated
regime §3.1 says suits an orthogonal operator. Task selection is not controlled for anywhere in
this programme, and that weakens all three positives.

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

## 5.2 A structural derivation (and a retracted ceiling)

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

**6.2 Capacity has opposite signs in the two regimes** — but see the audit below: the ordering is
*not* monotone in parameter count. In the interpolation regime
(`gap ≈ 1`), `E_extrap ≈ 0` for every mechanism, so only `E_approx + E_est` matter and capacity
helps. In the composition regime (`gap > 1`), `E_extrap` dominates and capacity increases it.
Measured: hypernetworks lose by 57–90% on held-out combinations and memorise on an FNO backbone
(2815× train/val gap) — yet **win** on the LA-2A, where there is no gap at all. Both halves are
now observed. The earlier claim ("capacity hurts conditioning") was the first half without its
boundary.

**Audit (2026-08-01): parameter count does not order the arms.** On dSprites the 2.1M
hypernetwork (0.00383) *beats* the 298K low-rank arm (0.00508); on 3D Shapes likewise (0.00928 vs
0.01978); and the 83K concat-MLP often beats the 50K FiLM. Sorting by capacity does not sort by
error. What survives is the weaker statement that *the unstructured arms are usually worst on
held-out combinations, without being ordered among themselves* — which points at the **operation
class** (low-rank linear vs full linear vs diagonal vs nonlinear-in-`z`) rather than at capacity,
and is therefore consistent with §3.1 rather than with a capacity story.

**6.3 Dense coverage dissolves the benefit.** The samples needed to cover `C` grow exponentially
in `dim(C)`. For `dim(C) ≲ 3` the condition space is enumerable, so `gap ≈ 1` and structure is
useless by §4. Measured: the LA-2A has two controls and 42 reachable settings, of which the dataset
contains 38 — nothing to compose. dSprites has five factors, whose combinations cannot be covered.
**Structured conditioning requires roughly four or more genuinely interacting condition axes to
have anything to do.**

## 6.5 Three results, with proofs

The framework above is stated with two empirical proxies. This section replaces both with derived
quantities. Throughout, `‖·‖` is Frobenius, `T*` is the task's true conditional map (assumed
linear in the features — this excludes concatenation-style mechanisms, §3.1(2)), and `R` is a
mechanism's reachable set.

### Theorem 1 (exact approximation error, per mechanism class)

`E_approx = min_{R ∈ 𝓡} ‖T* − R‖` admits a closed form for each standard mechanism:

| mechanism | reachable set | `E_approx` |
|---|---|---|
| hypernetwork | all linear maps | `0` |
| FiLM | diagonal | `‖T* − diag(T*)‖` |
| CGA | orthogonal | `‖S* − I‖`, from the polar `T* = U*S*` |
| Complex FiLM | block-diag. scaled rotations | `√( ‖offblk(T*)‖² + Σ_b ‖B_b − αI − βJ‖² )` |

*Proof.* **FiLM.** The Frobenius norm is entrywise, so the objective separates; diagonal entries are
matched exactly by `D_ii = T*_ii`, off-diagonal entries are unreachable. **CGA.** Orthogonal
Procrustes: `‖T*−Q‖² = ‖T*‖² − 2⟨T*,Q⟩ + d`, so minimising is maximising `tr(QᵀT*)`. With
`T* = UΣVᵀ`, write `Z = VᵀQᵀU`, orthogonal, giving `tr(QᵀT*) = Σᵢ σᵢ Zᵢᵢ ≤ Σᵢ σᵢ`, attained at
`Z = I`, i.e. `Q = UVᵀ = U*`. Then `T* − Q = U(Σ−I)Vᵀ`, so the residual is `‖Σ−I‖ = ‖S*−I‖`.
**Complex FiLM.** The reachable set is block-diagonal, so off-diagonal blocks are unreachable and
contribute in full. Within a `2×2` block the reachable set `{αI + βJ}` is a *linear subspace*, so
the minimiser is the orthogonal projection and the residual is the component along the complement
`span{[[1,0],[0,−1]], [[0,1],[1,0]]}`. ∎

These are identities, not bounds, each attained by an explicit minimiser. All four verified
numerically against brute-force minimisation.

### Lemma (the exponential is 1-Lipschitz on the skew algebra)

For skew-symmetric `X, Y`: `‖e^X − e^Y‖ ≤ ‖X − Y‖`.

*Proof.* Skew matrices form a linear space, so `Z(t) = Y + t(X−Y)` is skew for `t ∈ [0,1]`. The
derivative of `exp` at `Z` in direction `H` is `∫₀¹ e^{sZ} H e^{(1−s)Z} ds`. For skew `Z` each
`e^{sZ}` is orthogonal, and Frobenius norm is invariant under orthogonal multiplication on either
side, so the integrand has norm `‖H‖` and `‖d/dt e^{Z(t)}‖ ≤ ‖X−Y‖`. Integrating over `[0,1]`
gives the claim. ∎

Measured: max ratio `0.9976` over 200 random pairs. With a symmetric part present the bound fails,
as it must — the observed amplification reaches `115×`.

### Theorem 2 (extrapolation error of a closed mechanism)

Let `T(c) = exp(A(Wc))` with `A` linear into the skew algebra, and let `T̂(c) = exp(A(Ŵc))` be the
learned model. Then for every `c`,

```
‖T̂(c) − T(c)‖  ≤  ‖A‖_op · ‖Ŵ − W‖_op · ‖c‖
```

*Proof.* `A(Ŵc)` and `A(Wc)` are both skew, so the Lemma gives
`‖T̂(c) − T(c)‖ ≤ ‖A(Ŵc) − A(Wc)‖ = ‖A((Ŵ−W)c)‖ ≤ ‖A‖_op‖Ŵ−W‖_op‖c‖`. ∎

**The bound depends on `‖c‖` alone — not on the position of `c` relative to `C_train`.** That is
the closure benefit made quantitative: reaching an unvisited *combination* of conditions costs
nothing beyond reaching the same magnitude, and no comparable bound holds for a mechanism whose
head is an MLP.

**Corollary 2.1 (identifiability).** If `C_train` spans `ℝᵏ`, then `W` is identifiable and
`‖Ŵ − W‖` is an ordinary linear-regression error, vanishing with data. `E_extrap → 0`.

**Corollary 2.2 (why the compact factor is the stable one).** If `A` carries a symmetric part of
magnitude `s`, the Lemma no longer applies and the Lipschitz constant along the path is bounded
only by `e^{max(‖X‖,‖Y‖)}`. Error therefore grows like `e^s`. Under strength scaling `c → αc` this
is **linear in `α` on the rotation channel and exponential on the magnitude channel** — which
derives §5.2(v), previously asserted, and predicts both the guidance behaviour and the necessity of
the magnitude clamp.

### Theorem 3 (selection criterion)

Combining, for a closed mechanism,

```
E_test  ≤  E_approx[Thm 1]  +  ‖A‖_op‖Ŵ−W‖_op‖c‖ [Thm 2]  +  E_est
```

and the mechanism minimising this bound is the one to choose. The two empirical proxies of §4 are
then *estimators of terms in this expression* rather than heuristics with post-hoc thresholds: the
fit ratio estimates the first term, the compositional gap the second.

### Theorem 4 (the spectral obstruction: an impossibility)

Let `T*` be the transformation a task requires. If `T*` has any eigenvalue with `|λ| ≠ 1`, then no
orthogonal conditioner can represent it **in any representation**.

*Proof.* Every orthogonal matrix has all eigenvalues of modulus 1. A change of representation acts
on the operator by conjugation, `T ↦ M T M⁻¹`, and conjugate matrices have the same characteristic
polynomial, hence the same spectrum. So if `|λ| ≠ 1` for some eigenvalue of `T*`, that remains true
in every basis, and `T*` is similar to no orthogonal matrix. ∎

Verified numerically: conjugating `diag(0.1, 0.3, 1, 1, 2, 3)` by 500 random invertible matrices
leaves the eigenvalue moduli unchanged and `min ‖S − I‖ = 3.004`.

**This is the sharpest negative result in the framework, and it upgrades our largest measured
failure.** Content conditioning — "produce this rather than that" — requires suppressing features,
i.e. `|λ| → 0`. The 8× regression we measured was therefore not a shortfall of capacity, not a
poorly chosen basis, and not something co-training could have repaired. It was structurally
unreachable, and would have been in any representation.

**The line it draws.** Compare with §6.6 below: a learned representation *can* repair a rotation
wearing the wrong basis, and *cannot* repair a change of scale. Every result in §5 falls on one
side or the other of that line.

### 6.6 `E_approx` is a property of the task *and the representation*

Theorem 1 computes `E_approx` from `T*`, but `T*` is the transformation required **in whatever
coordinates the encoder produced**. It is not an invariant of the task. Learn different coordinates
and `S*` moves.

This weakens the framework as stated in §4: the fit-ratio screen measures whether a
*task-plus-representation pair* suits a mechanism, not whether a task does. It should be read that
way, and a poor reading would attribute to a task what belongs to its encoder.

It also reframes what the encoder is for. Under
`z' = h⁻¹(T(c) h(z))` with `h` invertible, composition survives because `h` cancels, so the network
is free to search for coordinates in which the condition acts as a group. Our encoder/decoder is a
crude version of this, which is why the frozen-representation result was not a quirk: `latent-edit`
supplied an `h` optimised for reconstruction with no reason to linearise conditioning, and the arm
lost by 60% to a hypernetwork carrying 48× the parameters. Theorem 4 bounds how much this can ever
buy.

**Corollary (the no-bypass rule).** The pressure to find such coordinates exists only if the
operator is the *sole* conditioning path. Given a parallel unstructured route, the model can use it
and the encoder learns nothing group-friendly. This predicts our own hybrid result: `hybrid_concat`
contains a concat-MLP and scored `+1.7%` against concat — a tie, which is what a model that
reverted to the bypass would give. The natural engineering instinct, to hedge the structured
operator with a residual, destroys the mechanism that makes it work.

**What this does not yet cover.** `E_est` has no treatment here; standard capacity bounds apply but
have not been imported. And Theorems 1 and 3 require `T*`, whose recovery from a fitted
hypernetwork is an empirical question that §9 flags as untested.

**Novelty, stated plainly.** Theorem 1 is orthogonal Procrustes plus two elementary projections;
the Lemma is a known property of the exponential on compact groups, used in the unitary-RNN
literature for temporal recurrence. Neither is deep. Their value here is that they replace the
fitted thresholds with derived quantities and resolve two claims — §5.2(v) and the clamp — that the
document previously stated as observations.

## 7. A prediction for a setting we have not run: world models

The framework's value is only testable outside the tasks it was built on. This is the sharpest
prediction it makes for a setting we have no data on, recorded before we test it so that it can
fail publicly.

**Setting.** A world model conditioned on two very different things: past **proprioception** (joint
angles, velocities, end-effector pose) and past **actions**. In practice both are "a vector fed to
the dynamics model", and both are usually conditioned the same way.

**The theory says they should be conditioned differently, and says which is which.**

**Proprioception is state, and wants FiLM.** A proprio reading tells the model what configuration
the body is in — which features should be active. That is *content*: specifying what, not applying
an operation. Content lives in the stretch factor (§3.1, §6.1), and the diagonal mechanism is the
one that reaches the stretch factor. Two further reasons point the same way:

- **No compositional gap.** Proprio states do not compose — "arm here" ∘ "arm there" is not an
  operation — so closure has nothing to determine (§4). World models also train on trajectories
  covering the reachable state space fairly densely and evaluate in-distribution, so `gap ≈ 1`.
- **`E_est` binds.** Proprio is 20–50 dimensional. A hypernetwork head emitting a `d×d` matrix from
  that is on the order of `10⁵`–`10⁶` parameters per conditioning site, estimated from limited
  trajectory data. The "capacity helps in the interpolation regime" half of §6.2 only holds where
  the capacity can actually be estimated, which is why the LA-2A result (hypernetwork wins, two
  knobs, densely sampled, tiny head) should **not** transfer here.

**Actions are transformations, and want an operator that composes.** Actions compose by
construction — apply `a₁` then `a₂` — so the group structure is real and closure has something to
determine. Measured on our own rollout suite: a composing operator plus a latent-consistency loss
beat FiLM by **78%** at horizon 20 and went flat where FiLM's error grew `4.5×`.

**The prediction, stated so it can be falsified.**

> In a world model conditioned on both, the two paths should use *different* mechanisms: a
> diagonal/FiLM path for proprioception and a composing operator path for actions. Making both
> FiLM should cost long-horizon accuracy; making both an operator should cost fit.

Note that both are *operator-style* in the sense of §1 — the condition produces a linear map on the
features. They differ in **which polar factor that map reaches**, not in whether a map exists. This
is not a claim about concatenation, which lies outside the framework entirely because no matrix is
formed.

**Evidential status: weak, and external.** The proprio half matches practitioner experience
reported to us for DreamZero-style models, where FiLM is said to work best for conditioning on past
proprioception. That is an anecdote, retrodicted, and it has mundane alternative explanations —
FiLM is cheap for fast rollouts, it is the default and therefore the best-tuned, and "best" may
mean best per unit of engineering effort. It is nonetheless the **first data point this framework
has touched that we did not construct**, which matters given that §5's audit found task selection
uncontrolled throughout our own programme.

**The cheap test.** In any system that already conditions on both, swap only the *action* path to a
composing operator and leave proprioception on FiLM. The framework predicts a gain concentrated at
long horizons and roughly no change to single-step fit. That is a smaller intervention than
anything we built, and it tests the framework rather than the method.

## 8. Status of each claim

| claim | status |
|---|---|
| Closure proposition (§2) | **proven**; verified numerically to 10⁻⁷ at arbitrary weights |
| Reachable family is `dim(c)`-dimensional | **proven** |
| Isometry cannot inject content (6.1) | **proven**, and measured |
| Error decomposition (§3) | standard, applied to this setting |
| Fit ratio and gap as proxies (§4) | **empirical**: 9 observations, thresholds calibrated post hoc |
| Capacity sign flip (6.2) | **empirical**, both regimes observed |
| Coverage criterion (6.3) | argued from counting, consistent with all 9 tasks |
| Theorem 1, closed-form `E_approx` (6.5) | **proven**; all four verified numerically |
| Lemma, `exp` 1-Lipschitz on skew (6.5) | **proven**; measured max ratio 0.9976 |
| Theorem 2, extrapolation bound (6.5) | **proven** |
| Theorem 4, spectral obstruction (6.5) | **proven**; verified over 500 conjugations |
| `E_approx` is representation-dependent (6.6) | **proven**; weakens the §4 screen as stated |
| World-model asymmetry (§7) | **prediction, untested by us**; proprio half matches one external anecdote |

The thresholds are guidance, not bounds. Nine observations, and both were fitted after the fact;
they are reported so that a future task can falsify them.

## 9. What is missing

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

## 10. Why this is the contribution rather than the operator

The project set out to show that a group-structured conditioner wins. It wins on three tasks and
loses on six, and no application survived contact with a benchmark we did not construct.

But the failures are not noise: **every one of them is predicted**, and the prediction costs two
short training runs. That is more useful to a practitioner than the operator ever was, because the
common case is deciding whether to reach for structure at all — and the answer, most of the time,
is no, for reasons that can now be checked in an afternoon rather than discovered in a month.

The honest one-line summary: *structured conditioning is a regulariser against combinatorial
memorisation, and combinatorial memorisation is rarely the binding failure in practice.*
