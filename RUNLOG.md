# RUNLOG — End-of-Turn detector

All scores are **mean response delay (ms) at ≤5% interrupted turns** from the
official `score.py`. Two numbers matter and they are very different:

* **Honest score = grouped-CV out-of-fold** (each turn scored by models that
  never saw it; GroupKFold by `turn_id`). This estimates unseen-set skill.
  **These are the delivered `predictions_*.csv`.**
* **In-sample** = the all-data model scored on its own training folders.
  Wildly optimistic; listed only to flag the trap.

Reference: silence baseline **EN 1600 / HI 850**.

| # | Change | EN (ms) | HI (ms) | Note |
|---|--------|--------:|--------:|------|
| 0 | Silence-only baseline | 1600 | 850 | reference |
| 1 | Starter template features (in-sample) | 1190 | 850 | starter's own optimistic number |
| 2 | 23 causal feats (energy+F0 yin+rhythm), HistGBM, **grouped-CV** | 1215 | 872 | first honest number; F0 noisy |
| 3 | + median-filtered F0, multi-scale fall / energy-release, turn-relative pitch (26 feats) | 1192 | 865 | EN AUC .60→.65 |
| 4 | + 5 clean causal **pyin-tail** features (31 feats) | 1229 | 865 | HI AUC .65→.70 |
| 5 | + creak / offset-abruptness / long-decay (35 feats), HistGBM d3 | 1138 | 872 | EN best single-model |
| 6 | Ensemble 2×GBM + scaled LogReg (35 feats) | 1145 | 850 | AUC EN .68 / HI .71 |
| — | sample-weighting long holds | 1186 | 850 | hurt AUC, discarded |
| — | per-language models | 1315 | 872 | less data, worse, discarded |
| — | isotonic calibration | 1248 | 1124 | overfit; discarded |
| — | drop `pause_start` (robustness check) | 1187 | 850 | AUC unchanged → position is not a crutch |
| ⚠ | **run 6 model, IN-SAMPLE on train folders** | 145 | 145 | AUC .997/.999 — the illusion the brief warns about |

### Run 7 — packaging only, no model change

The submission takes one file per deliverable, so `predict.py` now carries the
trained ensemble inline instead of loading a companion `.pkl`. I exported both
forests to flat arrays (split feature, threshold, child indices, leaf value,
missing-direction) plus the scaler stats and LR coefficients, and wrote ~40
lines of numpy to walk them. Checked against the original estimators on 400
random feature vectors: max difference 1.1e-16, and an end-to-end run gives
byte-identical `predictions.csv`. Side benefit — inference no longer imports
scikit-learn, so a version mismatch on the grader's machine can't break it.

### Run 8 — read the scorer properly, then build features for what it rewards

Went back to `score.py` and noticed something I'd missed: firing on a hold is
only a false cutoff if `delay < hold_duration`. Holds shorter than the chosen
delay are **free**. So the discrimination that earns points is EOT vs a *long*
hold, not EOT vs any hold. Re-measured every feature on that subset only, and
the picture changed — `offset_abruptness` jumped from AUC .40 to .29 (i.e. .71
inverted) against long holds. Built 20 features around that finding: offset
shape at three scales, tail-energy curvature (a phrase-final release decays
smoothly, a hesitation snaps off), F0 movement *inside* the final syllable, and
silence history detected from the audio itself.

| step | EN (ms) | EN AUC | HI (ms) | HI AUC |
|---|--:|--:|--:|--:|
| 35 feats (run 6), averaged over 8 fold-shuffles | 1192 ± 34 | .688 ± .009 | 850 | .711 ± .008 |
| **55 feats ← shipped** | 1223 ± 38 | **.707 ± .006** | 850 | **.759 ± .009** |
| 55 feats + cost-aware hold weighting | 1216 | .688 | 850 | .747 |

Two things I only found by running the same config eight times with different
fold shuffles:

1. **The English delay number has sd ≈ 36 ms and a range of ~120 ms.** My
   earlier 1145 was a favourable draw, not a real improvement — the same model
   averages 1192. Tuning on that number is fitting fold noise, so I selected on
   AUC (sd ≈ .007) instead and report the delay with its spread.
2. **Cost-aware hold weighting stopped helping once the features got better.**
   It gained ~100 ms on the 35-feature set and lost ground on the 55-feature
   set — it was compensating for a weak feature set, not adding signal. Dropped.

Also tried and rejected: rank-uniform transform of `p_eot`. The scorer sweeps
thresholds on a coarse 0.05 grid, so spreading the scores uniformly gives the
sweep finer resolution — it's strictly monotonic so AUC is untouched. Worth
10 ms on English, nothing on Hindi. Inside the noise band, and it makes `p_eot`
no longer readable as a probability, so I left it out.

Hyperparameters (depth 2/3/4, leaf size, L2, LR blend weight, adding
ExtraTrees) all landed within ±.004 mean AUC of each other. The features were
the win; the model was already saturated.

**Shipped (run 8, honest out-of-fold, 10 seeds averaged): English 1200 ms
(−25% vs 1600), AUC 0.709 · Hindi 850 ms, AUC 0.769.**

Hindi latency sits on the baseline floor because at a 850 ms delay only 5 of
its 148 holds are long enough to be dangerous — exactly the 5% budget — so
"fire on everything at 850 ms" is unbeatable *on this dev set*. The AUC moving
.50 → .77 is the part that should transfer to the hidden, mostly-Hindi set,
whose hold-duration mix will differ.
