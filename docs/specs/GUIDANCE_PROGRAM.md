# Review: what has to be run for the guidance hypothesis

**Written 2026-07-27, before any code for it exists.** This is a scoping document, not a
pre-registration. Each experiment that survives review gets its own spec with margins fixed before
its decision run, as always.

## The hypothesis

CGA's conditioning strength knob is `T(αc) = T(c)^α` — an exact group power. Classifier-free
guidance instead samples from `p(x) · [p(c|x)]^w`, which reweights the whole distribution.

> **H:** These differ in kind. CFG's reweighting provably shrinks the conditional distribution's
> variance, trading diversity for adherence. Group power only moves the conditional mean and has no
> sharpening mechanism at all, so as strength rises CFG should collapse the conditional
> distribution while group power holds it.

If true, CGA offers something guidance distillation cannot: adherence that does not cost diversity.
That matters because distilled CFG already reaches 1 NFE *with* sharpening, so a pure speed claim
loses to it. Diversity preservation is a property, not a saving, and distillation cannot copy a
property its teacher does not have.

## The blocker this review found

**Our existing guidance harness cannot test this hypothesis, and neither could the CIFAR experiment
as I first scoped it.**

In stage 6 and stage 9, `combo_images()` maps each condition to exactly one dSprites image. The
conditional distribution is a point mass. There is no variance for CFG to collapse and none for us
to preserve. Stage 9's own spec said as much and registered it as a caveat:

> "CFG sharpens the sampling distribution; group power amplifies the conditioning signal. These are
> different semantics; the deterministic-MSE metric measures distortion and favors neither a
> priori."

So stage 9 measured *distortion*, which is a real result, but it is not evidence about diversity,
and I was wrong to describe it as though the diversity claim followed from it. Any experiment
testing H needs a genuinely stochastic conditional.

## The fix: condition partially

Condition on shape, scale and orientation. Leave **position free**. Then

```
p(x | c)  =  uniform over the 64 held-free (posX, posY) cells
```

and that is not an approximation of the ground truth — it *is* the ground truth, exactly, by
construction. Every quantity we need is then computable without a proxy:

- **Fidelity:** distance from a generated image to the nearest of the 64 valid images for its
  condition. Low means the sample is a real member of `p(x|c)`.
- **Coverage:** how many of the 64 cells the samples actually hit.
- **Uniformity:** distance between the empirical distribution over cells and the true uniform.

This is a strictly better instrument than the CIFAR/FID plan it replaces. FID is a proxy that
conflates fidelity and diversity into one number; here they separate cleanly and the target
distribution is known in closed form. It is also about a thousand times cheaper.

## Experiments

### E0 — instrument check (screen). Necessary.

Does CFG actually trade diversity for adherence *in this harness*? If cranking `w` does not
collapse coverage here, there is no contrast to measure and the line dies for one hour of GPU
rather than a week.

This is the camera lesson. The camera experiment was fully built, pre-registered and calibrated
before we noticed the instrument could not separate the arms. No sweep starts before its screen
passes. **Gate:** CFG coverage at `w=8` must fall to ≤0.7× its coverage at `w=1`; both arms must
reach fidelity good enough that samples are recognisably valid.

### E1 — the fidelity/diversity frontier. Necessary; this is H.

Both arms, strength swept, measuring fidelity and coverage at each setting. The claim is a frontier
comparison, not a single number: at matched fidelity, does group power retain more coverage?

Registered separately once E0 passes.

### E2 — quality at equal NFE. Necessary if any speed claim is made.

CFG costs 2 NFE per step, group power 1. Nobody compares at equal *steps*; they compare at equal
*NFE*. So the honest comparison is group power at 2N steps against CFG at N steps. Sampling only,
reuses E1's checkpoints, costs minutes. Without it the 2× is a number a reviewer discounts on
sight.

### E3 — multi-condition scaling. **Not necessary for H.** Separate hypothesis.

Guiding N concepts at independent strengths costs N+1 forward passes for composable-diffusion-style
methods and 1 for us, because the algebra is abelian:
`T(Σαᵢcᵢ) = Π T(cᵢ)^αᵢ`. This is probably the strongest claim available, and it connects the
guidance work to the compositional results we have already confirmed — but it tests a different
proposition and must not be folded into H's evidence. It gets its own spec and its own gate.

### E4 — CIFAR-10. Deferred, and possibly unnecessary.

Its purpose was to be "real data." But E1's ground truth is exact where FID is a proxy, so E4 is
not more rigorous, only more realistic. It answers generality, not mechanism. Its cost is a week at
N=3 seeds and three weeks at the registered N=10, and the cheap latent-diffusion shortcut is
unavailable: precomputed VAE latents are exactly the frozen-representation regime the `latent-edit`
run killed today. Revisit only if E1 and E3 both pass.

## Implementation gaps found

None of these existed when the plan was written. All are prerequisites, not experiments.

1. **Partial-conditioning harness.** `cond_vec` and `combo_images` assume the condition determines
   the image. Needs a variant that conditions on a factor subset and samples the rest.
2. **Fidelity / coverage / uniformity metrics.** Do not exist in any form.
3. **The ψ-linear arm.** Stage 9 powered `cfilm_hyb`, whose magnitude head is nonlinear in `c`, so
   its powering was approximate and the spec disclosed it. A claim that the knob is *exact* cannot
   rest on it. The fix: embed `ψ = MLP(c)` for expressiveness, take both magnitude and phase
   **linearly from ψ**, and scale ψ. Powering is then exact on both channels. Cross-condition
   additivity is lost, which H does not use.
4. **Checkpoint saving.** E1, E2 and E3 all sample from the same trained models. Without saved
   checkpoints we would train three times.
5. **A composable-diffusion baseline** for E3 only.

## Order of execution

Prerequisites (1-4), then E0. E0 decides whether E1 happens. E1 decides whether E3 and E4 happen.
E2 rides along with E1 at negligible cost. Nothing is pre-registered until its screen has passed,
and no test split is read before its decision run.
