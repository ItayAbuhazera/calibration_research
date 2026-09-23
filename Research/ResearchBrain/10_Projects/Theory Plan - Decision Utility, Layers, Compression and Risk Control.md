---
type: project_note
status: draft_for_discussion
created: 2026-09-21
project: Full-Vector Geometric Calibration
audience: researcher + collaborator (theory sessions to follow)
tags: [theory, decision-utility, jl, conformal-risk-control, distribution-shift, layers]
---

# Theory plan — decision utility, layers, compression, risk control

Purpose: fix definitions, record **verified** literature, list **candidate**
lemmas with proof sketches, assumptions, counterexamples and the empirical
quantities that test the assumptions. Nothing here is a claimed contribution.
Parent experiments: [[2026-09-21 Layer-Selection Pilot]],
[[2026-09-20 Full-Vector DAC POC]]; hypothesis [[H-LAYER-01 Selected internal layers add decision value beyond logits]];
paper link [[From Similarity to Decisions - PCE (AAAI submission)]].

**Label legend** (every statement carries one): `[established]` a theorem in the
cited source, statement checked in this session · `[standard]` textbook-level,
not source-checked · `[derived]` my derivation here, proof sketched, not peer
reviewed · `[conjecture]` · `[unverified]` cited from memory, source not opened ·
`[measured]` a number from a repo artifact.

**How sources were checked.** Text was extracted from the PDFs of: Conformal
Risk Control (vault PDF), Learn-then-Test (arXiv 2110.01052), Tibshirani et al.
2019 (arXiv 1904.06019), Tong–Feng–Li NP umbrella (arXiv 1608.03109), Geifman &
El-Yaniv SGR (arXiv 1705.08500), Dasgupta–Gupta (UCSD PDF), Larsen–Nelson
(arXiv 1609.02094), Barber et al. 2023 (arXiv 2202.13415), Ben-David et al. 2010
(Mach. Learn. 79). Everything else is `[unverified]`.

---

## 0. Definitions

* Base classifier: logits `z(x) ∈ R^C`, base decision `i(x)=argmax z`. Candidate
  rule `j(·)`: an alternative class (e.g. the runner-up of `z`, or the argmin of a
  corrected score). For a fixed pair `(i,j)` and label `y`:

  `D(x,y;i,j) = 1{j=y} − 1{i=y} ∈ {−1,0,+1}` (+1 = wrong→correct **W**, −1 =
  correct→wrong **H**, 0 = either unchanged or wrong→different-wrong **U**).
* Gate `g(F) ∈ {0,1}` acts on evidence `F` (a σ-field / random vector). The system
  outputs `j` where `g=1`, else `i`. Then `ΔAccuracy = E[g·D]` exactly
  (`[standard]`; the repo asserts `ΔAcc=(W−H)/N` per cell).
* Evidence sets: `Z` = full logit vector; `G` = geometric evidence (class-conditional
  distances `r_{l,k}(x)` for a layer set A, or probe outputs); `F=(Z,G)`.
* Family A correction (pilot): `q = softmax((z−βR_A)/S)`, `R_A=(1/|A|)Σ_{l∈A} r_l`.
  Pairwise: `j` beats `i` ⇔ `β(R_i−R_j) > z_i−z_j`. `S` is a positive scalar and
  cannot change the argmax (see [[Sample-dependent scalar temperature cannot change the predicted class]]).
* Utility `u(F)=E[D|F]`. Harm rate `P(H)=E[g 1{i=y,j≠y}]`, benefit `P(W)`.

---

## 1. Verified literature (what each source actually says)

| source | theorem / proposition (checked) | assumptions | conclusion | what we would change |
|---|---|---|---|---|
| Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control* (ICLR 2024; arXiv 2208.02814) — **Thm 1** | `[established]` losses `L_i(λ)` **exchangeable**, **non-increasing** in λ, right-continuous, `L_i(λ_max) ≤ α`, `sup_λ L_i ≤ B <∞` a.s. Set `λ̂ = inf{λ : (n/(n+1)) R̂_n(λ) + B/(n+1) ≤ α}` | as stated | `E[L_{n+1}(λ̂)] ≤ α` (expectation over calibration **and** test draw) | apply to `L(λ)=1{H-event under gate g_λ}` with nested gates (larger λ ⇒ fewer interventions ⇒ non-increasing). Controls **harm rate only**; net utility `E[gD]` is **not** monotone in λ, so Thm 1 does not apply to it |
| same — **Thm 2 (lower bound)** | `[established]` if additionally i.i.d., `L_i ≥ 0`, no atoms of a jump function | as stated | `E[L_{n+1}(λ̂)] ≥ α − 2B/(n+1)`-type bound (tightness) | shows conservativeness is not free: the gate cannot be made much tighter at small n |
| same — **Prop 3** (covariate shift) | `[established]` labelled train `~P_train`, test `~P_test` with known `w(x)=dP_test/dP_train`, weighted `λ̂` | shift is **covariate shift** (`P(Y|X)` unchanged), absolute continuity | `E[L_{n+1}(λ̂(X_{n+1}))] ≤ α` on the test distribution | requires `P(Y|X)` unchanged and known/estimable `w`: fails under corruption (§T5) |
| same — **TV bound** | `[established]` for arbitrary shift, unweighted CRC degrades with total variation between train/test | bound stated in paper | risk ≤ α + B·TV-type term | vacuous when clean/corrupted images have TV≈1 (separable) — `[derived]` remark, not from the paper |
| Angelopoulos, Bates, Candès, Zrnic, Jordan, *Learn then Test* (arXiv 2110.01052) — **Thm 1** | `[established]` p-values `p_j` stochastically dominating U(0,1) under `H_j: R(λ_j) > α`; FWER-controlling algorithm `A` at level δ | as stated | `P( sup_{λ∈Λ̂} R(λ) ≤ α ) ≥ 1−δ` (**high-probability over calibration data**, any risk `R`, not necessarily monotone) | the right tool if we want to certify **net utility** or harm for a *grid of (layer set, β, threshold)*; multiplicity cost enters through the grid size |
| Tibshirani, Barber, Candès, Ramdas 2019 — **Cor. 1 / Thm 2** | `[established]` `P̃_X ≪ P_X`, `w=dP̃_X/dP_X`, weights `p_i^w(x)=w(X_i)/(Σ w(X_j)+w(x))` | `Y|X` shared across domains | `P(Y_{n+1}∈Ĉ_n(X_{n+1})) ≥ 1−α` (**marginal coverage**) | marginal coverage of a prediction *set*, not conditional error among accepted interventions and not utility |
| Barber, Candès, Ramdas, Tibshirani 2023 — **Thm 2** | `[established]` non-exchangeable full/split conformal with weights `w̃_i` | as stated | `P(Y∈Ĉ) ≥ 1−α−Σ_i w̃_i·d_TV(R(Z),R(Z_i))` | quantifies degradation by *swapped-residual* TV, not raw feature TV |
| Tong, Feng, Li 2018, *NP umbrella* (arXiv 1608.03109) — **Prop 1** | `[established]` `k`-th order statistic of `q` held-out class-0 scores, independent of the base scorer | continuous scores for tightness | `P(type-I error(φ_k) > α) ≤ Σ_{j=k}^{q} C(q,j)(1−α)^j α^{q−j}`; a valid threshold exists only if `q ≥ log δ / log(1−α)` | PCE's FPR-budget threshold is this construction; our "negatives" are events we must not create (harm) |
| Geifman & El-Yaniv 2017, *Selective classification for DNNs* — **Thm 3.2 (SGR)** | `[established]` i.i.d. sample; binary-search over thresholds with binomial-tail bound `B*` at level δ/k | i.i.d. | `P(∃ i: R(f|g_i) > B*(r̂_i, δ/k, g_i(S_m))) < δ` — high-probability **conditional** risk among accepted examples | the "conditional error among accepted interventions" notion |
| Dasgupta & Gupta 2003 — **Thm 2.1** | `[established]` `0<ε<1`, `k ≥ 4(ε²/2−ε³/3)^{-1} ln n` | any n-point `V⊂R^d` | ∃ `f:R^d→R^k` with `(1−ε)‖u−v‖² ≤ ‖f(u)−f(v)‖² ≤ (1+ε)‖u−v‖²` ∀ u,v∈V (random projection succeeds with prob ≥ 1/n per the proof's pair bound `2/n²`) | note: bound is on **squared** distances, for a **fixed finite set**; future queries need a separate per-query union bound over the bank |
| Larsen & Nelson (arXiv 1609.02094) — **Thm 2** | `[established]` for `ε ∈ (lg^{-0.5001}n/√min{n,d}, 1)` there exist n-point sets requiring `m = Ω(ε^{-2} lg(ε²n))` | as stated | JL dimension is essentially optimal in the worst case | no worst-case improvement of the `ε^{-2}log n` barrier is possible; any useful statement about compression must use data structure (intrinsic/doubling dimension, margins) |
| Ben-David et al. 2010, *A theory of learning from different domains* — **Thm 1, Thm 2** | `[established]` Thm 1: `ε_T(h) ≤ ε_S(h)+d_1(D_S,D_T)+min{E_S|f_S−f_T|, E_T|f_S−f_T|}`; Thm 2: `ε_T(h) ≤ ε_S(h)+½ d_{HΔH}(U_S,U_T)+λ+O(√(d log m/m))` w.p. ≥ 1−δ | `λ` = best joint error of the class on both domains | target error is controlled only through a divergence **and** the (unobservable) joint-error/labeling-difference term | a certificate for intervention utility needs an analogous term; it cannot be estimated from unlabeled target data |

`[unverified]` (cited from memory, not opened here): Blackwell 1953 comparison of
experiments; Lehmann–Romano generalized Neyman–Pearson lemma; Vovk 2012
calibration-conditional validity; Achlioptas 2003; Indyk–Naor 2007 (nearest-
neighbour-preserving embeddings depending on doubling dimension); Alain & Bengio
2016 (linear probes); Ueda–Nakano 1996 and Brown–Wyatt–Tiño 2005 (ensemble
bias/variance/covariance); Krogh–Vedelsby 1995; Beyer et al. 1999 (distance
concentration); Ben-David, Lu, Luu, Pál 2010 (impossibility theorems for domain
adaptation); Zhao et al. 2019; Mammen–Tsybakov / Audibert–Tsybakov margin rates.

---

## T1. Decision utility beyond logits

**What would be proved.** For a fixed candidate rule `j` and evidence `F`:

* (`[standard]`) `sup_g E[gD] = E[(u(F))^+]`, attained by `g*(F)=1{u(F)>0}`.
  Proof: `E[gD]=E[g u(F)]` by the tower property; pointwise maximization.
* (`[derived]`, two-line) **Value of evidence as Bayes-accuracy headroom.** Let the
  candidate be the best alternative for the evidence, `j*(F)=argmax_{k≠i} P(Y=k|F)`.
  Then `Δ*(F)=E[(max_{k≠i} P(Y=k|F) − P(Y=i|F))^+] = Acc*(F) − Acc(base)`, with
  `Acc*(F)=E[max_k P(Y=k|F)]`. So the question "does `G` add decision value beyond
  `Z`?" is exactly `Acc*(Z,G) − Acc*(Z) > 0`, i.e. does `P(Y|Z,G)` have a different
  argmax from `P(Y|Z)` with positive probability. `Acc*(Z) − Acc(base)` is the ceiling
  for *any* logit-only recalibration (Vector Scaling, matrix scaling, a full-logit
  probe are lower bounds of it).
* (`[derived]`, Jensen) **Strictness criterion.** `Δ*(Z,G) ≥ Δ*(Z)` always; equality
  iff for a.e. `Z` the conditional law of `u(Z,G)` given `Z` puts no mass on *both*
  `(0,∞)` and `(−∞,0)`. Strictly more decision value ⇔ on a set of `Z` of positive
  probability, `G` can flip the sign of the utility. *Counterexample to "G lowers NLL
  ⇒ G adds decision value":* if `sign u(Z,G)=sign u(Z)` a.s., NLL can drop while
  `Δ*` is unchanged (the tower property gives equality). Decision value is a
  sign-straddling property, not an information-gain property.
* (`[derived]`) **Under perfect target calibration nothing can be gained.** If
  `P_T(Y=k|x)=softmax_k(z(x))` then `u(x)=q_j−q_i ≤ 0` for every candidate `j`, so
  `Δ*=0`. Any positive utility on the target needs the base posterior *ordering* to
  be wrong there. Temperature/ECE improvements are orthogonal to this (they never
  reorder).

**Would tell us to do differently.** Evaluate evidence by cross-fitted
`Acc*(Z,G)−Acc*(Z)` (probe with and without `G`, same family, same data) instead of NLL;
select layers on a decision-aligned criterion or accept that NLL selection is only a
proxy.

**Closest existing result.** Bayes decision theory (Devroye–Györfi–Lugosi ch. 2
`[standard]`); Blackwell's comparison of experiments `[unverified]`.

**Standard vs new.** All standard. The identity `Δ*=Acc*(F)−Acc(base)` is a
restatement; novelty: none. Its use is to name what the pilot's full-logit probe
and layer probes are lower-bounding.

**How assumptions fail here.** (i) Finite-sample learnability: the plug-in gate
`1{û>0}` needs `û` accurate where `|u|` is small; regret depends on a margin
condition `[unverified]`. (ii) Population value ≠ clean-to-target transfer (T5).
(iii) Fixed-candidate rule: with `j=runner-up(z)` only (`top-2`), `Δ*` is smaller than
the full Bayes gap; the pilot's Family A uses the whole `R` vector, so it can pick
any `j`.

**Empirical quantities.** Cross-fitted probe accuracy with/without `G` on target-
labelled held-out data (headroom, not deployable); `P(sign flip)` of the fitted
`û(Z,G)` vs `û(Z)`; W/H/U by base-margin bin (the pilot reports base-error
enrichment vs a margin-matched null — see [[2026-09-21 Layer-Selection Pilot]]).

---

## T2. Compression, distances and decision margins

**Chain.** representation compression → distance approximation → class-evidence
(order-statistic) approximation → corrected-margin perturbation → argmax stability.
The five links are **not interchangeable**: preserving pairwise distances ⇏
preserving the K_c-th neighbour distance ⇏ preserving neighbour *identities* ⇏
preserving class-evidence *ordering* ⇏ preserving the final decision.

* **L2.1 (order-statistic stability)** `[derived]` (3-line proof). If `a_i ∈ [(1−ε')b_i,(1+ε')b_i]` for all `i`, then the `K`-th smallest satisfies
  `a_(K) ∈ [(1−ε')b_(K),(1+ε')b_(K)]`. Proof: order statistics are monotone in every
  coordinate. Hence multiplicative distance distortion on *all* pairs transfers to
  `r_{l,k}` for every class and every `K_c`. It does **not** preserve neighbour
  *identity* (ties within the distortion band can swap members).
* **L2.2 (JL for the operator)** `[derived]` from `[established]` Dasgupta–Gupta. Apply a JL
  map to the point set `V = bank ∪ queries` (size `n`): all squared distances within
  `1±ε` ⇒ `r̃ ∈ [√(1−ε), √(1+ε)] · r ⊂ [(1−ε), (1+ε/2)]·r`. Two caveats that the
  standard statement does not cover: (a) the guarantee is for a **fixed finite set**; for
  **future queries** one needs a per-query union bound over the bank
  (`k = O(ε^{-2} log(n_bank/δ))` per query, over `N` future queries `log(N n_bank/δ)`),
  and it holds over the randomness of the map, not uniformly for one fixed map;
  (b) the DAC operator **L2-normalizes before** the distance. Compress-then-normalize
  is not covered: polarization applied to `u±v` (set size `O(n²)`, so `ln n → 2 ln n`)
  gives additive `ε` error on inner products of unit vectors, hence `O(ε)` error on
  the normalized distances — `[derived]`, sketch only.
* **P2.3 (margin certificate)** `[derived]`. Let `m_β(x) = (z_i−βR_i) − max_{j≠i}(z_j−βR_j)`.
  If every `R̃_k ∈ [(1−ε)R_k,(1+ε)R_k]`, then the argmax of `z − βR̃` equals that of
  `z − βR` whenever `m_β(x) > 2 β ε · max_k R_k(x)` (tighter: `> βε(R_i+R_{j*})`).
  Consequences: (1) compression can only **preserve or destroy** correction
  decisions — it cannot create decision value, so it addresses tag **B/C** only as a
  *loss* mechanism; (2) the fraction of samples with `m_β/(β R_max)` below `2ε` is a
  measurable upper bound on decision churn due to compression.
* **Vacuity check (`[measured]` arithmetic, n=45 000)** with Dasgupta–Gupta:
  required `k` = 35 469 (ε=0.05), 9 184 (0.1), 2 473 (0.2), 1 191 (0.3), 515 (0.5).
  Our pooled dimensions are 256–2 048. **The worst-case JL bound gives no guarantee at
  any of the dimensions we use** except at ε≳0.3–0.5 for the largest layers, and
  Larsen–Nelson show the worst case cannot be improved. So JL *cannot* explain why a
  compressed bank works empirically; an explanation must use intrinsic/doubling
  dimension `[unverified: Indyk–Naor]` or the margin structure of P2.3.
* **L2.4 (pooling is a contraction, not a JL map)** `[derived]`, exact algebra. For a
  feature tensor `a ∈ R^{C×S}` with spatial mean `ā`:
  `‖a−b‖_F² = S‖ā−b̄‖² + ‖(a−ā)−(b−b̄)‖_F²`. So `S·(GAP distance)²` is a **lower
  bound** on the full-tensor distance², with equality iff the difference is spatially
  constant; distinct tensors can collapse under GAP, and a k×k grid pooling refines the
  bound monotonically (nested partitions). No two-sided guarantee exists; a GAP
  null says nothing about the un-pooled tensor (this is why the pilot pre-declares one
  bounded `grid2` sensitivity). Exact only before the L2 normalization.

**Would tell us to do differently.** Report, per layer, the empirical margin
distribution `m_β/(βR_max)` and the fraction below `2ε` for the compression actually
used; do not cite JL as justification for 1 024-d projections without checking `ε`.

**Closest existing.** JL (Dasgupta–Gupta `[established]`); Indyk–Naor `[unverified]`;
the repo's SPP+JL compression (`utils/compression_utils.py`).

**Standard vs new.** L2.1, P2.3 elementary; L2.2 is the standard JL with two
specialization caveats; L2.4 exact algebra. **Novelty: low.** Value: the vacuity
observation and the margin certificate connect approximation error to a decision
property, which distance preservation alone does not.

**Failure in our experiments.** The bank labels have 14/45 000 duplicate images
with conflicting labels (`docs/full_vector_dac_experiment.md` §19.2) — a set-level
assumption violation; TF32 noise (~1e-3) is comparable to `ε`-scale effects; the
`r` used is the *class-conditional* K_c-th neighbour (K_c=5), far more variable than
a K=200 global statistic.

---

## T3. When additional layers help

**Claim to test, not assume:** more layers do not monotonically help.

* **P3.1 (equal-weight aggregation of dependent statistics)** `[derived]` (algebra) +
  `[unverified]` (ensemble literature). Model the per-layer statistic for the pair
  `(i,j)` as `δ_l = ±μ_l + ξ + η_l` (sign by which class is correct), `ξ` common noise
  (variance `ρσ²`), `η_l` layer-specific (variance `(1−ρ)σ²`). The equal-weight average
  over a set of size `L` has `SNR_L = μ̄_L / (σ √(ρ+(1−ρ)/L))`; in the binary Gaussian model
  accuracy `=Φ(SNR/2)` is monotone in SNR. Adding a layer with signal `μ'` raises the
  SNR iff `μ'/μ̄_L > (L+1)√((ρ+(1−ρ)/(L+1))/(ρ+(1−ρ)/L)) − L`. `[measured]` numerics of that
  threshold: independent layers (ρ=0) ≈ 0.41–0.48 (any layer with more than ~half the
  current mean signal helps); ρ=0.6 ≈ 0.79–0.96; ρ=0.9 ≈ 0.95–0.99; ρ→1 ⇒ 1.
  **With strongly correlated layers, adding any layer weaker than the current average
  strictly hurts.**
* **Counterexample (explicit):** `μ=(1, 0.2)`, `ρ=0.8`, `σ=1`: `SNR_1=1.00`,
  `SNR_2=0.63`, accuracy `Φ(0.5)=0.691 → Φ(0.316)=0.624`. The second layer is informative
  yet lowers the decision quality of the equal-weight average.
* **Scale dominance** `[derived]`: the pilot uses raw (unscaled) distances, so a layer
  with larger typical `r` dominates the mean irrespective of information — a second
  mechanism by which equal weights can harm.
* **Selection error / winner's curse** `[standard]`: greedy selection on a noisy
  criterion over `m=12` candidates has expected optimism of order `σ_sel·E[max of m
  Gaussians] ≈ 1.6σ_sel`; the selection role has 2 500 rows. Layer-count comparisons
  (1/4/6/8) are therefore contaminated by selection noise that grows with the number of
  greedy steps.
* **Dimension / sample size:** distance statistics in 2 048-d with ~450 class
  exemplars are concentrated (relative contrast shrinks, Beyer et al. 1999
  `[unverified]`); the class-conditional K_c-th neighbour noise scales with the bank
  size per class, not with the total.

**Would tell us to do differently.** Estimate `ρ` (correlation of `δ_l` across layers
on clean data) and per-layer `μ_l` from the pilot's stored `r` arrays before claiming
anything about L; if `ρ` is large, prefer selection to aggregation, and test the
depth-spaced control (already in the pilot) against greedy.

**Closest existing.** Ensemble bias–variance–covariance (Ueda–Nakano, Brown et al.)
`[unverified]`; Condorcet/majority-vote limits with dependence `[unverified]`.

**Novelty.** None; the threshold table is a convenient closed form, not a result.

---

## T4. Safe intervention and finite-sample control

**Notions that must not be conflated:**

1. marginal coverage of a prediction set (Tibshirani/CRC-style);
2. **conditional** error among accepted interventions `P(H | g=1)` (SGR-style);
3. **global** harm probability `P(H)=P(g=1, i correct, j wrong)`;
4. **excess classification risk vs base** `−E[gD]=P(H)−P(W)` (net utility);
5. guarantees **in expectation** over calibration data (CRC Thm 1) vs
   **with probability ≥1−δ** over calibration data (LTT Thm 1, NP umbrella Prop 1, SGR).

* **P4.1 (generalized Neyman–Pearson form of the harm-budgeted gate)** `[standard]` /
  `[unverified]` source (Lehmann–Romano gen. NP lemma). Maximize `P(W)` s.t. `P(H) ≤ α`
  over gates on `F`: the optimum thresholds the ratio `dP(W,F)/dP(H,F)` (randomized at the
  boundary). This is PCE's "maximize TPR s.t. FPR ≤ α" with the harm event as the
  negative class and the repair event as the positive class; it is also the decision
  target that predicting "is `i` wrong?" misses (that merges W and U).
* **P4.2 (which theorem certifies what)** `[derived]` from the verified statements:
  * harm rate with nested gates: CRC Thm 1 (`E[harm] ≤ α`, expectation over
    calibration), monotone;
  * harm, high probability: NP umbrella / LTT with a p-value from a binomial or
    Hoeffding–Bentkus bound;
  * **net utility** `E[gD] ≥ 0`: **not monotone in the gate threshold** (loosening
    adds both W and H), so CRC Thm 1 does not apply; LTT (arbitrary risk, FWER over a
    finite grid Λ) does; SGR-like for conditional risk `P(H|g=1)`.
  * **Selection is part of the hypothesis set.** Layer set × β × threshold must be a
    *pre-declared finite grid* Λ inside the LTT family (multiplicity cost `∝ ln|Λ|`),
    or the selection must use a role disjoint from the certificate role. The pilot's
    inner-SELECT role is used for layer choice, leaving **no** independent role for a
    certificate: any later formal claim needs new clean data (spec §3).
* **Sample-size arithmetic (`[measured]` arithmetic, not experiment results).**
  * NP umbrella existence condition, δ=0.05: `q ≥ ln δ/ln(1−α)` = 149 (α=0.02), 299
    (0.01), 598 (0.005), **2 995 (α=0.001)** held-out negatives. With ~3 850 base-correct
    rows in a 5 000-row validation split, the smallest certifiable α is ≈ 7.8e-4 with
    **zero** empirical violations; with 10 violations the Clopper–Pearson 95 % upper
    bound is 4.4e-3.
  * Utility resolution. Using the closed pilot's aggregate rates as a proxy
    (`P(W)=5.4e-3`, `P(H)=4.0e-3`, `[measured]` 653/483 over 120 000 sample-cells),
    `Var(D)≈9.5e-3` (sd ≈ 0.097). The standard error of an estimated `ΔAcc` is 0.195 pp
    at n=2 500, 0.137 pp at 5 000, 0.097 pp at 10 000, 0.049 pp at 40 000; resolving
    ±0.05 pp needs ≈ 38 000 examples. **The FV-DAC effect (+0.14 pp) is below the
    standard error of a 2 500-row selection role.** Decision-utility selection or
    certification on a 2 500-row role is statistically impossible at these effect
    sizes; NLL-based selection is a smoother proxy but is not decision-aligned (T1).
* **Rare events.** Harm events are ~0.4 % of samples; certificates that control them
  at 0.1 % need thousands of *negatives* and tens of thousands of *positives* to have
  power. State this as a limitation, not a footnote.

**Would tell us to do differently.** (i) Define the harm budget on `H` (not on "i
wrong"); (ii) reserve a certificate role of ≥ ~4×10⁴ clean labelled rows *or* accept
non-guaranteed nominal thresholds and say so (the AAAI appendix says exactly that);
(iii) certify only over a declared finite grid.

**Closest existing.** CRC, LTT, NP umbrella, SGR (all `[established]` above); PCE's
NP threshold.

**Standard vs possibly new.** Each piece is standard. A possibly-new *combination*:
harm-budgeted repair gating with the W/H/U trichotomy and the explicit non-monotonicity
of net utility; **open**: sample complexity of certifying net utility for a
gate learned from `F` under selection over Λ. Novelty unknown until the literature on
"decision calibration"/"selective classification with cost" is checked
(`main_text_only/` holds several candidate PDFs, not read here).

**Failure in our experiments.** Clean-fit certificates do not transfer (T5); calibration
data are also used to select layers.

---

## T5. Distribution shift and impossibility

* **P5.1 (non-identifiability of target utility from unlabeled data)** `[derived]`
  (elementary), same spirit as Ben-David et al. `[established]` (the joint-error term
  `λ` is unobservable). For any fixed gate and candidate rule, `E_T[gD|x] =
  g(x)(P_T(Y=j|x) − P_T(Y=i|x))`. Unlabeled target data identify `P_T(X)` only; the
  conditional ranges freely over `[−g(x), g(x)]`. Two target laws with the *same*
  `P_T(X)` (hence the same feature-density statistics) can have utility `+c` and `−c`.
  **Marginal feature-density similarity does not certify utility.**
* **Explicit counterexample.** One binary evidence coordinate `f∈{a,b}`, equal mass
  under source and target. Source: `P(Y=j|a)=0.8`, `P(Y=j|b)=0.5`; target: labels for
  `a` swapped. A gate `g=1{f=a}` has source utility `+0.6·½` and target utility
  `−0.6·½`, with identical `P(X)` and identical density-ratio diagnostics.
* **P5.2 (what standard shift theory gives, and why it does not fit)** `[established]`
  weighted CRC (Prop 3) and weighted conformal (Cor. 1) need covariate shift
  (`P(Y|X)` unchanged) and `P_T ≪ P_S` with known `w`. A CIFAR-100-C image is not in
  the support of clean CIFAR-100 in pixel space (`w=∞` or undefined), and the label given a
  degraded image is *less* determined than given the clean image — a concept shift in
  `P(Y|x)`, not a covariate shift. The TV bounds (CRC TV bound; Barber et al. Thm 2)
  degrade with `d_TV`, which is ≈1 for e.g. Gaussian noise. `[derived]` remark.
* **Structural conditions under which transfer could be justified** (each needs a
  checkable status):
  1. *stable conditional utility*: `E[D|F]` invariant across domains — **needs target
     labels** (cannot be certified from unlabeled data);
  2. *bounded utility discrepancy*: `sup_F |u_T(F) − u_S(F)| ≤ κ` — needs target labels
     (estimate by a labelled held-out target slice);
  3. *density-ratio + covariate-shift in a learned representation where `P(Y|F)`
     is shared* — only the density ratio is observable; `P(Y|F)` sharing is an assumption;
  4. *target-side calibration of the base model* — observable with labels only, and
     if it holds then **no gain is possible** (T1), so it is not a route to a
     guarantee of gain.
* **Observable / needs labels / uncertifiable:** observable — feature and logit marginals,
  density ratios (if support overlaps), classifier-based two-sample statistics;
  needs labels — utility, harm, calibration under shift, `d_TV(R(Z),R(Z_i))`;
  **cannot be certified from the available data** — the sign of the utility on an
  unlabeled target.

**Would tell us to do differently.** Never present clean-fitted W/H trade-offs as
transferable guarantees; treat the 12 corruption cells as *development* conditions
and hold out corruption families/severities for any claim; design *labelled* target
slices when a guarantee is wanted.

**Closest existing.** Ben-David et al. (Thm 1/2); Ben-David–Lu–Luu–Pál impossibility
theorems `[unverified]`; Barber et al. TV bound.

**Novelty.** None; the counterexample is a re-instantiation for gated utility.

---

## Strongest candidate propositions (for the next discussion)

1. **Value-of-evidence identity and sign-straddling criterion (T1).** `Δ*(F)=Acc*(F)−Acc(base)`;
   `G` adds decision value iff it can flip the sign of `u` on a positive-probability set
   of `Z`. Novelty none, but it *changes what we measure*: a cross-fitted logit-vs-logit+geometry
   headroom, not NLL. Directly testable with the pilot's probes.
2. **Certification cost (T4).** Net utility is non-monotone in the gate, so CRC Thm 1 does not
   apply; LTT does over a declared grid; at n=2 500 the utility SE (≈0.2 pp) exceeds the
   observed effect (0.14 pp). A precise sample-complexity statement for certifying net
   utility under selection over Λ is the one place a *new* result might exist.
3. **Layer-addition threshold under correlation (T3).** Closed-form condition for an added
   layer to raise equal-weight SNR; explains when 4→6→8 layers must hurt. Low novelty, high
   diagnostic value: requires estimating `ρ` and `μ_l` from stored `r` arrays.

Also record as a cautionary result: **JL is vacuous at our dimensions** (T2) and the
target-utility non-identifiability (T5) — both limit what any theorem can promise here.

## Open items / not done

* Read the sources marked `[unverified]`, especially Indyk–Naor, Lehmann–Romano, Blackwell.
* Check the `main_text_only/` decision-calibration PDFs (Dimension-free decision calibration;
  Calibrating predictions to decisions; Efficient calibration for decision making) for
  overlap with T1/T4 before claiming any novelty.
* Estimate `ρ`, `μ_l` and the margin distributions from the pilot's `per_sample.npz` `raw__r` arrays.
* Proof details for compress-then-normalize (L2.2b) and the Gaussian SNR→accuracy step.

---

## Corrections appended 2026-09-21 (second pass; nothing above edited)

Labels as in the legend above. These narrow or replace specific statements; the originals stay for provenance.

1. **T1 value-of-evidence identity — scope.** `Δ*(F)=E[(E[D|F])_+]` is the optimal value of an *evidence-measurable binary gate for a fixed alternative j*. The identity `Δ*(F)=Acc*(F)−Acc(base)` `[derived]` holds only when the alternative may be chosen as the best `F`-measurable class (`j*(F)=argmax_{k≠i}P(Y=k|F)`, a multiclass action set). It does **not** generally equal unrestricted multiclass Bayes accuracy for a fixed `j`; the two coincide only under extra action-set conditions. "Family A can pick any j" applies to the argmax of `z−βR`, which is a restricted function class, not the Bayes selector.
2. **T2 JL vacuity.** A conservative sufficient dimension above our embedding dimension means the Dasgupta–Gupta bound does not *certify* the compression used. It is not a necessary dimension and does not show practical distance preservation is impossible or that the compression fails on this dataset. "Cannot explain why a compressed bank works" should read: the worst-case bound does not supply an explanation; data structure (intrinsic dimension, margins) would have to. (Larsen–Nelson show the worst case cannot be improved for *some* point sets; this says nothing about ours.)
3. **T3 layer-addition threshold.** The 0.8–0.95 figure is model-dependent: equicorrelated homoscedastic Gaussian layer statistics, equal weights, within-class ρ≈0.6 (measured on the pilot's evaluation cells, exploratory). Downgraded to an **illustration**; it is not a universal condition.
4. **Attribution of probe averaging.** "Four-layer probe averaging improved probability metrics" is a measured change; the claim "not evidence for distance geometry" was an untested attribution (probes are linear readouts of pooled features; whether geometry-related information mattered was not isolated). The residual-evidence study (`docs/residual_evidence_study_spec.md`) adds controls that can address it.
5. **AUCs.** Marginal AUCs of a layer distance and the base logit margin for W-vs-H (≈0.81–0.82 vs 0.81) do not show zero incremental information *conditional on the full logits*; nor does a higher AUC show usefulness. Matched comparisons (same images, same readout family) are used in the follow-up.
6. **Wording.** "No material evidence to continue" is a practical verdict under a frozen rule; the corrected-protocol arms were small **positive** on average (+0.03 to +0.10 pp) and Vector Scaling +0.19 pp.
7. **T5 identifiability.** P5.1 is a statement about **unrestricted** target shift and **unlabeled** evidence only. Restricted shift families (label shift with known/estimable priors, covariate shift with `P(Y|X)` shared and overlapping support, bounded utility discrepancy with a labelled target slice) change the conclusion; the CIFAR-C support argument is a `[derived]` remark, not a theorem.
8. **Astra theoretical report** (referred to in the task): not found on the local filesystem; its distinctions (information in a representation vs retained by a statistic vs learnable with finite data vs transferable; label information can improve proper scores without changing the optimal class; a prototype-distance counterexample is not a theorem about class-conditional kNN; a Gaussian-model estimation-cost threshold; JL preserves an existing correction under margin conditions but cannot show it improves accuracy; conformal coverage does not certify top-1 improvement; LTT needs independent calibration and tiny gains are hard to certify) are adopted **as stated in the task summary, not as read**.

## Measurements supplied by the residual-evidence study (2026-09-21; measured, not theorems)
From [[2026-09-21 Residual Evidence Study]] (two development checkpoints, `layer3.22`): retained rank of standardized 100-d evidence (participation ratio: G ≈ 38, S ≈ 60, class-radius vectors DG/DS ≈ 2–3, logit-space radii DL ≈ 10, random-ReLU logit features O ≈ 30);
Gaussian-projection distortion at d=100 (mean ratio ≈ 1.0, sd ≈ 0.07, p95 relative squared-distance error 0.26–0.29; JL sufficient dimension ≫ 100, so no JL guarantee applies); selected ridge strength at the top of the grid with ‖W‖_F ≈ 0.1–0.2;
fit-vs-selection NLL gaps 0.15–0.5 nats for λ=1 and none at λ=100; interventions confined to the lowest base-margin quintile; clean→corruption signed-utility changes small and heterogeneous across families. **Reading for the theory (unresolved interpretation):** at n_fit=2 500 the estimation cost of a 10 000-parameter residual exceeds any detectable clean-learnable signal — consistent with the estimation-cost-versus-signal statement of the restrictive Gaussian model, which is *motivation*, not proof that the neural setting satisfies it.
A finite probe comparison does not estimate a population Bayes gap. Standard facts (JL, conformal/LTT, Hadamard-prototype≈linear classifier) are not claimed as contributions.

## Measurements supplied by the atlas program (2026-09-21; measured, not theorems)
From [[2026-09-21 Representation Atlas Program]]: candidate quality and harm are separable (kNN accuracy on base errors 0.10–0.14 at deep layer3 vs harm 0.38–0.52 on base-correct; logit kNN 0.025/0.02) — a concrete instance of T1's sign-straddling requirement: the utility sign varies within the evidence the clean rule could see, and 65–151 clean disagreements per candidate bound learnability (T4 sample-size arithmetic). Frozen low-rank+isotropic Mahalanobis estimator: top-256 subspace explains 99–99.8 % of hidden-feature variance yet whitening up-weights the omitted subspace and destroys neighbours (T2: regularization choice, not covariance per se). Clean-fitted scalar temperature increased corruption NLL (T5: clean-calibrated quantities do not transfer). Oracle-union increments are reported separately from learned-policy gains (correction 1 of the earlier pass).

## Update 2026-09-21 (fixed deep-candidate gate study) — [[2026-09-21 Fixed Deep Candidate Gate Study]], [[H-GATE-01 Candidate selection versus gate utility mismatch]]
Status: completed (development, checkpoints 2 and 4). Fixed `layer3.22` 2×2 kNN candidate with C0/C1/Z0/Z1 ridge gates and n∈{625,1250,2500} learning curves: no consistent held-out benefit (Z1−Z0 clean +0.10/+0.13 pp, corruption −0.04/−0.04); deep gates ≈ output-evidence and layer4/output controls; practical targets not met; probability quality not improved. Deep candidate has 873/908 fit disagreements (the 52–151-event limit was layer4/output-specific) but H:W ≈ 3.8:1. Audit corrections to the atlas report appended in its card; TF32 mismatch cause verified (benchmark = TF32 convs, batch 128); seed-4 reconciliation deferred jobs 21537819–21. Measurement note for theory: standalone E[D] and gated E[gD] differ, but here neither exposed a large positive-utility region (top score bin ≈ 0 utility). Not established: absence of information; more-label benefit.

## Pointer 2026-09-22
New research lead for the readout/admissible-family question this theory plan tracks: [[Counting and Covering in Nearest-Neighbour Representations of Boolean Functions]] (abstract verified, theorem-level reading pending — do not cite a theorem number from it yet). Standing cross-study summary: [[Current Evidence - Representation-Based Correction Program]].

## Memo propositions appended 2026-09-23 (numbering is the memo's; it does not renumber this plan's T1–T5)

Source: the design memo (Claude Doc, 2026-09-22) <https://claude.ai/code/artifact/ebf21486-1669-4afb-9748-f624a976e2dd>, §5 "Theorem targets". Both are labelled **[derived]**: derivations stated in the memo, each checked there only by simulation; not independently re-derived here, not from a published source.

**Memo-T1 [derived] — clean identifiability corollary.** Setting: source and target laws of (H, Y), H the intermediate state, Z = ζ(H) the logits, base i = argmax Z, candidate j = κ(H), D = 1{j=Y} − 1{i=Y}, gate g any H-measurable function in [0,1], ḡ = E_S[g | Z], ε_S = E_S|u_S(H) − u_S(Z)| the source insufficiency of Z for D. As stated in the memo: (b) if ε_S = 0 every gate has the same source utility as its Z-projection, so clean data carry no population information about how a gate uses H, while the target value of that part can have either sign; (c) 0 ≤ V_S(H) − V_S(Z) ≤ ε_S/2; (d) under a shift-stable increment and a density ratio bounded by W, V_T(H) − V_T(Z) ≤ W·ε_S/2, vacuous under corruption because support expands. The memo says it bounds what clean data can reveal, not target utility. **Stage 0 is compatible with Memo-T1 but does not establish it as the cause:** a target-fit gain alongside a clean-fit null fits it equally with clean-fit regularization, a shift in the distribution of the evidence, and a readout mismatch; Stage 0 shows a recoverability gap and cannot attribute it. See [[H-STAGE0-01 Layer3.22 probe evidence is redundant with the logits on clean data but complementary under corruption, and clean supervision cannot identify the useful combination]].

**Memo-T2 [derived] — class-agnostic evidence can only correct toward the runner-up.** If F₁ ⊇ F₀ is class-agnostic given F₀ (informs whether the base is right, not which class), then argmax η¹ ∈ {i, r} almost surely (r the runner-up of the F₀-Bayes posterior), and correction is detection at a sample-specific threshold q(Z) = s/(1+s) ≤ 1/2, where s is the runner-up's share of the non-base mass. It concerns optimal posterior decisions of the target-fitted Z readout, not raw logits or kNN repairs. Stage 0a stayed descriptive: the candidate j's base-logit rank distribution among disagreements resembles the true-class rank distribution among base errors (rank 2 ≈ 20–21 % vs 21–22 %); this neither proves nor disproves class-agnostic evidence and is not a test of Memo-T2. The Memo-T2 chain (Z ⊆ (Z, π̂_C) ⊆ (Z, P)) belongs to Stage 0d, which is not started.
