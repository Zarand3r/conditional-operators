# Research log

What was tried, in the order it was tried, and how each one turned out. The pre-registrations
live in [`specs/`](specs/), the raw logs in [`../results/`](../results/), and the write-up in
[`paper/`](paper/).

Three words appear in the table. **Confirmed** means every criterion registered before the run
passed. **Kill** means at least one failed, so the hypothesis is rejected for that setting.
**Unfair** means the compute-budget rule was broken, so the comparison proves nothing either way.

| Experiment | Question | Verdict | The numbers |
|---|---|---|---|
| Aligned | Does structure beat unstructured conditioning at equal budget? | confirmed | 61.5% below the best hypernetwork, at 0.87× FiLM's cost |
| — | Does it survive a hidden basis, using a dense learned basis? | unfair | The basis really cost 1.52× FiLM, over the registered ceiling |
| Hidden basis | Same question, with a structured basis inside the budget | confirmed | 3.3e-8 error; flat on never-trained triples; angles recovered to 2e-4 rad |
| dSprites | Does it survive a learned latent, on real images? | confirmed | 53.8% below the best baseline; error grows 1.25× on unseen combinations against 2.1–3.3× |
| dSprites+shape | Does it survive a categorical, non-group factor? | confirmed | 42.9%; the advantage shrinks but holds |
| 3D Shapes | Does it hold on a second dataset? | kill | Best on unseen combinations by 87.5%, but 1.46× worse on the training fit |
| Content | Can it specify image content, in diffusion? | kill | 7× worse than FiLM, and the ablation fails identically |
| Rollouts | Does exact composition stop rollout drift? | kill | Every arm lands near 0.043 at horizon 20; the contractive one does best |
| Rollouts+loss | Does a latent-consistency loss unlock it? | kill | Horizon-20 error goes flat at 0.0039, 90% below the hypernetwork, but a separate single-step criterion fails |
| — | Is a fixed contraction rate the knob instead? | kill | No rate helps; larger ones only hurt |
| Complex FiLM (changes) | Complex FiLM, in the transformation role | confirmed | 36.1% over FiLM, fit parity, 0.84× its cost |
| Complex FiLM (content) | Complex FiLM, in the content role | confirmed | 0.97× FiLM's error; the linear-magnitude variant fails at 0.19, so content needs the expressive head |
| camera-sliders | Does it beat CameraCtrl's additive mechanism on camera moves? | withdrawn | Abandoned at calibration: gap 1.2x against dSprites' 2.1x, arms within 3%. Test split never read, no verdict claimed |
| latent-edit | Does it still win when the representation is frozen instead of co-trained? | kill | No. Loses to a hypernet by 60% with 48x fewer parameters, and fits 2.43x worse. Same data and splits as 3D Shapes, where co-training made it 8x better — so the advantage needs the representation, not just the task |
| guidance-diversity (E0) | Does group power hold sample diversity where CFG collapses it? | screen failed | No, and the pilot points the other way: at matched fidelity CFG keeps more diversity, the knob is non-monotone (14x worse at a=1.5 than at a=1, then recovers), and the base model fits 33% worse. One seed, no test split read, no verdict claimed |
| pde-params | Does it hold on a real physics task, conditioned on physical parameters? | kill | AC-5 only, by 1.8 points (fit 1.118x vs a 1.10x ceiling). Composition passed hugely: 57.5% below the best unstructured arm, delta=-1.00 (every seed beat every seed), at 1.11x FiLM cost with 7x fewer parameters. Smallest fit penalty in the program |
| pde-conj | Does relaxing the isometry fix the fit criterion? | kill | AC-5 again, by 0.065 points (1.1007x vs 1.10x). Validation over 5 seeds said 0.962x; 10 seeds on held-out data said 1.1007x. Conjugation buys ~1.5% of fit, not the 14% validation advertised, so the fit penalty is a real cost of the operator rather than an artifact of the orthogonal basis |
| Guidance | Does powering the condition beat classifier-free guidance? | confirmed | Parity at strength 1; grows 23.7× to strength 8 against CFG's 50.7× (p=1.6e-4), with no second pass |

Sixteen gates in total: eight passed, seven failed, one was inconclusive on budget.

## Things worth remembering

**The erratum.** An audit of the FLOP counters found the dense-basis arm had been undercounted
by 2×: applying the basis costs `4d²` per sample, not `2d²`, because it is applied twice. The
true cost was 1.52× FiLM against a registered ceiling of 1.20×, so a result that had read as a
win became inconclusive. The corrected counter is now pinned by a regression test, and the
structured basis used from then on is what brought the cost back inside the budget honestly.

**The rollout experiment refuted a prediction I had already made in writing.** Exact composition was supposed to
stop error compounding over rollouts, and it did nothing; every conditioner drifted at the same
rate. Chasing the reason produced a better result than the prediction would have: rollout error
is dominated by the latent drifting off the decodable manifold, not by the transition composing
badly. Adding a latent-consistency loss makes the isometric operator's error flat far past the
training horizon, while a fixed contraction — the obvious alternative — does nothing at any rate.

**A file-descriptor accident, July 23.** Switching git branches while a sweep was running
replaced its log file on disk; the writer kept appending to the now-unlinked inode, so the file
on disk stopped growing. 43 of 50 rows were recovered through `/proc/<pid>/fd`, and the last
seven survive in `results/stage8a_stdout.log` and the summary aggregates. The rule since: no
branch switching while a sweep is running.

## Prior-art sweep, July 24

Three independent searches, each told to try to prove the work had already been done, across
roughly 35 papers. No fatal collision. Complex FiLM and guidance-as-an-operator-power appear to
be unclaimed.

Two areas are crowded enough to be careful about. Rotation conditioning has Worrall et al. (2017)
rotating hidden features with exact composition for geometric transformations, the commutative
Lie group VAE, attributes-as-operators, and the 2025–26 wave of RoPE generalizations. World
models have more: the homomorphism autoencoder (Keurti et al., 2023) already pairs rotation
transitions with a latent-prediction loss and shows long stable rollouts, Quessard et al. (2020)
learn per-action latent rotations, latent-consistency losses are standard in model-based RL, and
generator-affine dependence on the control input is folklore in bilinear Koopman theory.

The claims were narrowed to match: the combination, the exactness-by-construction contract, the
consistency-versus-composition decomposition, the contraction null result, and the budget-fair
pre-registered protocol. Not the operators, and not the loss. CFG's distortion at high guidance
is established (Bradley and Nakkiran; APG), so the guidance experiment isolates it against an exact
alternative rather than claiming to have found it. About eleven citations were added to the paper as a result.

## Standing rules

- Register the criteria before the run. Margins never move once data exists; deviations become
  dated amendments inside the spec.
- Never hand-edit the paper's tables or figures. They regenerate from `results/*.json`.
- Don't switch branches while a sweep is running.
- Throttle long GPU runs (`STAGE*_THROTTLE_MS=60`) or cap the card. This machine's power supply
  trips at a sustained 600 W, which cost one overnight run.
