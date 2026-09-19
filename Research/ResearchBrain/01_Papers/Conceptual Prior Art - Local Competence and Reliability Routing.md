---
type: paper_map
status: active
date: 2026-09-15
tags: [local-competence, routing, shift, prior-art-audit]
---

# Conceptual Prior Art — Local Competence and Reliability Routing

This map records adjacent literatures. They are prior-art threats to broad claims
about predicting when an auxiliary signal should affect an action; they do not by
themselves establish a solution to [[Geometry contains complementary accuracy information under corruption]].

## Dynamic classifier / ensemble selection

### Dynamic classifier selection: Recent advances and perspectives — Cruz et al. (2018)

- **Signal:** a validation-set local region of competence (often kNN) plus local accuracy, ranking, probabilistic, behaviour, or meta features for each base classifier.
- **Target / decision:** local competence; select one classifier or a sub-ensemble for a query.
- **Argmax:** yes, because the selected classifier/ensemble can output a different class.
- **Assumptions:** validation neighbourhood labels represent each classifier's local future competence.
- **Benchmark / shift:** broad conventional classification benchmark comparison; not a controlled deep corruption-transfer study.
- **Asks whether geometry is reliable?** It directly asks whether a *local competence estimate* is reliable enough to route, but usually assumes a stationary dynamic-selection set.
- **Vault overlap / gap:** strongest historical threat to the broad routing framing; it does not target a frozen head versus a representation-derived geometric expert under unseen shift.
- **Source:** https://doi.org/10.1016/j.inffus.2017.08.010

### Adapting dynamic classifier selection for concept drift — Souza et al. (2019)

- **Signal:** neighbourhood-defined competence of a classifier pool, updated as labelled batches arrive.
- **Target / decision:** which classifier/ensemble predicts each query under concept drift.
- **Argmax:** yes.
- **Assumptions:** new labelled data arrive sufficiently promptly to refresh the dynamic-selection set.
- **Benchmark / shift:** synthetic/stream drift and PKLot real-world concept drift; not label-free deployment shift.
- **Asks whether geometry is reliable?** Only indirectly: competence is re-estimated after drift rather than inferred without target labels.
- **Vault overlap / gap:** shows that local competence under shift is an established problem; it avoids the vault's central label-free transfer difficulty by obtaining post-shift labels.
- **Source:** https://doi.org/10.1016/j.eswa.2018.10.013

## Adaptive retrieval / model interpolation

### Generalization through Memorization: Nearest Neighbor Language Models — Khandelwal et al. (2020)

- **Signal:** nearest-neighbour distances and values in a pretrained LM representation datastore.
- **Target / decision:** a kNN next-token distribution interpolated with the parametric LM distribution.
- **Argmax:** yes, interpolation can change the selected token.
- **Assumptions:** embedding proximity retrieves label-relevant contexts; datastore coverage matches useful deployment patterns.
- **Benchmark / shift:** Wikitext-103 language modelling and domain adaptation by replacing the datastore.
- **Asks whether geometry is reliable?** No explicit per-token reliability estimator; interpolation weight is global in the base formulation.
- **Vault overlap / gap:** direct analogue of global probability blending with a nonparametric expert; it reports domain-adaptation utility but not label-free oracle recovery or risk-controlled routing.
- **Source:** https://arxiv.org/abs/1911.00172

### Adaptive Nearest Neighbor Machine Translation — Zheng et al. (2021)

- **Signal:** token-specific pattern of retrieved-neighbour distances / retrieval noise, summarized by a Meta-k network.
- **Target / decision:** importance of candidate k values and hence how much retrieved evidence enters token prediction.
- **Argmax:** yes, the retrieval-augmented token distribution can change the token.
- **Assumptions:** retrieval-result statistics predict noise; a meta-k learner transfers across the tested domains.
- **Benchmark / shift:** four multi-domain machine-translation datasets; cross-domain application is reported.
- **Asks whether geometry is reliable?** Partly: it asks whether a retrieval neighbourhood is useful for the current token, but not whether a geometry expert beats a base classifier at matched risk.
- **Vault overlap / gap:** closest adaptive-interpolation precedent; its deployment setting has a task-specific datastore and supervised meta-training, unlike the current clean-to-corruption reliability problem.
- **Source:** https://aclanthology.org/2021.acl-short.47/

## Learning to defer / reliability of the router

### Learning to Defer with an Uncertain Rejector via Conformal Prediction — Fang & Nalisnick (2024; TMLR version 2026)

- **Signal:** a learned model-versus-expert rejector and conformal uncertainty set over that rejector's routing decision.
- **Target / decision:** whether to predict, defer, abstain, or seek model--expert consensus.
- **Argmax:** not the point; it changes the system action rather than relabelling the model prediction.
- **Assumptions:** exchangeability for conformal validity on the relevant calibration/deployment distribution; expert labels and rejector supervision are available.
- **Benchmark / shift:** object detection, HAM10000, and hate-speech tasks, including covariate-shift stress tests.
- **Asks whether geometry is reliable?** It directly treats the router as fallible, but its router is not a geometric local-competence signal.
- **Vault overlap / gap:** strongest operational threat to a general claim that the action-selector's uncertainty is neglected; it does not study complementary head/geometry predictions.
- **Source:** https://openreview.net/forum?id=TWb9y4PNSW

## OOD / failure scoring with representation neighbours

### Out-of-Distribution Detection with Deep Nearest Neighbors — Sun et al. (2022)

- **Signal:** nonparametric kNN distance from a test embedding to an ID feature bank.
- **Target / decision:** ID versus OOD status; flag/reject abnormal inputs.
- **Argmax:** no; it is an OOD score, not an ID class correction.
- **Assumptions:** embedding distance separates ID from OOD; it intentionally avoids a parametric feature-distribution assumption.
- **Benchmark / shift:** ImageNet-1k models and standard OOD benchmarks, compared with parametric Mahalanobis/SSD+ detection.
- **Asks whether geometry is reliable?** No. It evaluates geometry as an OOD detector, not geometry's conditional value relative to the head on shifted ID-label tasks.
- **Vault overlap / gap:** strong baseline family for any claim that kNN geometry carries novel uncertainty information; OOD AUROC/FPR is not oracle headroom or action utility.
- **Source:** https://proceedings.mlr.press/v162/sun22d.html

## Predicting performance under shift

### Leveraging Unlabeled Data to Predict Out-of-Distribution Performance — Garg et al. (2022)

- **Signal:** source-calibrated confidence threshold applied to unlabeled target examples (ATC), plus target score distributions.
- **Target / decision:** dataset-level target-domain accuracy; model monitoring or model selection.
- **Argmax:** no.
- **Assumptions:** a source-fitted confidence threshold remains related to target correctness; the paper proves performance estimation is impossible without restrictions on shift.
- **Benchmark / shift:** FMoW-WILDS, ImageNet, CIFAR, and MNIST under corruptions, dataset reproduction, and novel-subpopulation shifts.
- **Asks whether geometry is reliable?** No; it evaluates confidence for aggregate accuracy prediction.
- **Vault overlap / gap:** directly addresses clean-to-unlabelled-target transfer, but at population rather than per-example routing granularity.
- **Source:** https://arxiv.org/abs/2201.04234

### Agreement-on-the-Line: Predicting the Performance of Neural Networks under Distribution Shift — Baek et al. (2022)

- **Signal:** agreement between pairs of neural-network classifiers on unlabeled target data.
- **Target / decision:** dataset-level OOD accuracy; model selection / monitoring.
- **Argmax:** no.
- **Assumptions:** agreement-on-the-line relation between ID and OOD agreement/accuracy holds for the shift and model family.
- **Benchmark / shift:** neural-network OOD benchmarks studied through labelled source and unlabelled target data; both settings where accuracy-on-the-line holds and fails.
- **Asks whether geometry is reliable?** No; it is a global performance estimator.
- **Vault overlap / gap:** establishes that unlabelled shift can reveal aggregate reliability, but leaves pointwise ``geometry-better-than-head`` unresolved.
- **Source:** https://arxiv.org/abs/2206.13089

### Predicting With Confidence on Unseen Distributions — Guillory et al. (2021)

- **Signal:** source-versus-target feature-distribution statistics from unlabelled target data.
- **Target / decision:** change in a fixed model's target accuracy (automatic model evaluation).
- **Argmax:** no.
- **Assumptions:** features and source/target distributional relations contain enough information to identify performance degradation.
- **Benchmark / shift:** multiple model architectures under natural and synthetic distribution shifts.
- **Asks whether geometry is reliable?** No; it predicts aggregate model performance rather than local competence or an auxiliary predictor's conditional value.
- **Vault overlap / gap:** a direct failure-monitoring comparator, but cannot decide which individual predictions to replace.
- **Source:** https://openaccess.thecvf.com/content/ICCV2021/html/Guillory_Predicting_With_Confidence_on_Unseen_Distributions_ICCV_2021_paper.html

## Known gaps in this audit (added 2026-09-15)

Not yet carded. Listed as named threats only - no summary is given here because
none of them has been read in-session, and a guessed summary would be worse than
an empty slot.

- **T3A - Test-time classifier adjustment (Iwasawa & Matsuo, NeurIPS 2021).** A
  prototype/centroid classifier replaces the head at test time under shift. This
  is the closest thing to "let geometry act instead of the head" and is a direct
  threat to the framing of [[Predicting geometric reliability under distribution shift]].
- **Learning to defer, original line (Madras et al. 2018; Mozannar & Sontag 2020).**
  The formal framing for "which predictor acts". Only the conformal-rejector
  descendant is carded above.
- **Failure detection under shift (Jaeger et al. 2023).** Benchmarks the claim
  that a score can flag its own errors under shift.
- **Dynamic ensemble selection library / META-DES line**, if the routing framing
  is kept.

Also absent, and needed to *interpret* results rather than to establish novelty:

- **Ensemble diversity / oracle-accuracy literature (Kuncheva & Whitaker line).**
  The oracle union is a standard diversity measure there. Required before
  [[Geometry contains complementary accuracy information under corruption]] can
  be read as being about geometry rather than about disagreement.
- **Top-label vs class-wise calibration theory (Gupta & Ramdas).** The Pareto
  note currently treats the ECE/NLL split as an empirical discovery; part of it
  is a known definitional difference.
- **Uncertainty under dataset shift (Ovadia et al. 2019).** The standard
  reference for how calibration methods degrade under corruption.
- **Weighted / covariate-shift conformal prediction.** Relevant if any of this
  ever moves to a matched-risk operating point (README rule 6).

Until the novelty threats in the first list are read, no note in this vault may claim novelty for
"deciding where geometry should override the head" (README rule 3).
