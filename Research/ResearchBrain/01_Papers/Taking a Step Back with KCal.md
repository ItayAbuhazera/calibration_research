---
type: paper
status: audited
year: 2023
venue: ICLR
short_name: KCal
tags: [kde, metric-learning, latent-space, calibration]
---

# Taking a Step Back with KCal

## Why this is mandatory

KCal is a direct conceptual neighbour, not merely another baseline. It learns a
low-dimensional representation / metric and performs class-conditional
kernel-density calibration in that space.

## Why it matters specifically to current work

Your repository now contains full KCal plus a factorial decomposition that
separates:
- learned KCal projection versus RGCL representation;
- full KDE versus top-k KDE;
- validation versus training reference bank;
- squared-L2 RBF versus exponential-L2 kernel;
- replace versus blending strategies.

That makes KCal useful both as literature and as an experimental lens for asking:
**which part of representation-space calibration actually matters?**

## Links

- [[Top-label calibration and full-vector proper scoring form a Pareto frontier]]
- [[Geometry contains complementary accuracy information under corruption]]

## Literature extraction (audit 2026-09-15)

- **Signal:** a learned low-dimensional projection of penultimate embeddings plus class-conditional KDE over labeled calibration examples.
- **Target:** the full class-probability vector / full calibration, not a correctness or routing score.
- **Decision:** supplies a replacement predictive distribution for downstream decisions.
- **Can change argmax?** Yes. KDE predicts the distribution directly; preservation of the base class is not imposed.
- **Assumptions:** the learned embedding admits sufficiently regular class-conditional densities; KDE bandwidth and calibration reference set are adequate. Its full-calibration guarantee is asymptotic under those density and bandwidth assumptions.
- **Benchmark:** CIFAR-10/100, SVHN, ImageNet, and IIIC/ISRUC/PN2017 healthcare tasks.
- **Distribution shift?** No controlled corruption/OOD evaluation in the paper.
- **Does it ask when geometry itself is reliable?** No. Geometry is used as the predictive mechanism, rather than as a potentially unreliable complementary expert.
- **Strongest vault overlap:** direct ancestor of the full-vector geometry line and of the observed top-label versus full-distribution trade-off.
- **Remaining gap:** does not test whether a geometry-derived correction is preferable to a frozen head on a particular shifted example.
- **Primary source:** https://arxiv.org/abs/2202.07679
