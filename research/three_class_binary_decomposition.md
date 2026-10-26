# Three-Class App Design: Binary Decomposition Research

*Date: 2026-10-25. Companion to `research_results.md`. Question: the app needs 3 outputs (healthy / symptomatic / COVID), but training binary is stronger — how to bridge?*

## Conclusion

**Use one-vs-rest: train two binary models and combine their outputs at prediction time.**

1. **COVID vs everything else** (COVID vs healthy + symptomatic)
2. **Symptomatic vs everything else** (symptomatic vs healthy + COVID)

Decision logic at inference:

- Neither fires → **healthy**
- Only COVID fires → **COVID**
- Only symptomatic fires → **symptomatic**
- Both fire → higher probability wins (or return "possible COVID — get tested"; a screening app should say that when classes genuinely overlap)

## Why one-vs-rest over the hierarchical (two-stage) alternative

Hierarchical design (stage 1: abnormal vs healthy; stage 2: COVID vs symptomatic on stage-1 positives) mirrors clinical logic and gives stage 1 a large training set, **but stage 2's sole job is the exact boundary the literature says is unreliable**:

- **Orlandic et al. 2023** (COUGHVID label-quality study): physician re-review of 581 "symptomatic" recordings found **only 32 actually COVID-positive**. The "symptomatic" class is defined by *self-reported symptoms, not acoustics* — a symptomatic non-COVID cough has no reason to sound different from a COVID cough, and the labels themselves are substantially noise.
- Leakage/label-noise literature consistently shows the hard confusion is **COVID vs symptomatic**, not COVID vs healthy. A hierarchical stage 2 trains a small model (~658 COVID vs ~1,283 symptomatic, both noisy) on that boundary alone.
- The best-performing papers avoid this: binary framing, or using the noisy intermediate class only as training-side context, never as the sole decision boundary.

In one-vs-rest, "COVID vs symptomatic" is never a model's sole job: the COVID model learns it implicitly, with the large clean healthy majority stabilizing training, and the final decision combines two probabilities instead of trusting one hard boundary on the noisiest class pair.

## Practical requirements (inherited from prior research)

- **Combine probabilities, not hard labels.** Keep both models' outputs; tune thresholds on validation (they need not both be 0.5).
- **Each binary model needs the recording-wise split** — decomposition does not remove leakage.
- **COVID-positive training data is tiny** (~658 in COUGHVID): keep the class-weighted loss, expect modest absolute performance, report per-class recall. AUC / PR-AUC per model is the honest headline metric.
- **Calibration > accuracy for the app**: the experience is threshold-driven ("flag for test" vs "all clear"). Tune thresholds for the operating point (high COVID recall, accept more symptomatic false alarms).
- If only one extra model is affordable, the better single-model framing is **COVID vs rest** with the app treating "not COVID" as the healthy/symptomatic fallback — but two one-vs-rest models is the recommended full design.

## Sources

- [Orlandic et al. 2023, COUGHVID SSL relabeling](https://www.sciencedirect.com/science/article/pii/S0169260723004091) — the 581/32 symptomatic re-review finding
- Leakage study and honest-metrics context: see `research_results.md` (npj Digital Medicine 2021; Lancet Digital Health comment)
