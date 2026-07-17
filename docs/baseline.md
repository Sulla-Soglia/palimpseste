# Phase 2 baseline — multi-label period classifier

First model on the Phase 1 corpus. ResNet-18 (ImageNet), final layer →
6 sigmoid outputs, BCE with pos_weight. Two phases: 3 epochs head-only,
8 epochs full fine-tune at lr/10. Best epoch selected on val macro-AUC.

## Dataset (2026-07-17, `make_dataset.py --seed 67`)

500 downloaded photos → 460 usable (40 had no period label).
Split **by monument** (merimee_ref), greedy stratified grouping so every
period appears in every split — a plain random group split dropped
german_occupation from val/test entirely.

| | monuments | photos |
|---|---|---|
| train | 72 | 337 |
| val | 39 | 65 |
| test | 40 | 58 |

## Results

Best val epoch: full-FT epoch 5 (val macro-AUC 0.631).

**Test macro-AUC 0.701**

| period | test AUC | test positives |
|---|---|---|
| german_reich | 0.846 | 32 |
| french_interwar | 0.760 | 18 |
| germanic_hre | 0.753 | 19 |
| german_occupation | 0.745 | 3 |
| french_ancien | 0.738 | 33 |
| french_modern | 0.361 | 4 |

Checkpoint: `results/baseline/resnet18_best.pt` (local only);
metrics: `results/baseline/metrics.json`.

## Reading

- **Overfitting is visible and expected**: train loss 1.08 → 0.11 while val
  macro-AUC plateaus ~0.62. With 337 training photos the model memorizes;
  early selection on val is what keeps the test number honest.
- **german_reich (0.846) is the strongest class** — the separate Neustadt
  collection did its job. Wilhelmine sandstone facades are also visually
  distinctive, which is RQ1 working as hoped.
- **french_modern (0.361) is below chance** on 4 positives: part noise
  (AUC on 4 positives swings wildly), part real — "modern" is a stylistic
  grab-bag with no coherent visual signature, and post-1945 buildings in a
  heritage-protected corpus are mostly *restorations of old buildings*,
  so the weak label points at a facade that looks ancient.
- **Small val positives (occ 3 / mod 5) make val macro-AUC jittery**;
  val 0.631 vs test 0.701 is sampling noise, not a miracle.

## Next

- [ ] Grad-CAM on the trained checkpoint — *where* does it look for
      german_reich? (RQ2 begins here)
- [ ] More 20th-century data: Commons categories for interwar/reconstruction
      architecture in Alsace; consider dropping/merging french_modern until
      it has real support.
- [ ] Rescue the 429 GPS-less photos via category→Wikidata P625 lookup.
- [ ] Linear-probe-only run as a second baseline (is fine-tuning actually
      buying anything at n=337?).
