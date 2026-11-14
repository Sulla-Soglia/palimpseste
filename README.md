# palimpseste

**Detecting cultural strata in Alsace-Lorraine architecture using deep learning.**

A building is a palimpsest — each occupation scraped the surface and wrote a new layer. But the old layers show through. Can a CNN learn to read them?

## Research Question

Can a deep learning model trained on Alsace-Lorraine building facades detect and quantify the degree of French-German stylistic mixing in individual buildings — and can the model's attention patterns reveal architectural "strata" analogous to training strata in AI systems?

- **RQ1 (technical):** Multi-label classification of cultural influence (% French / % German) per building
- **RQ2 (theoretical):** Do Grad-CAM attention maps automatically decompose facades into cultural-temporal layers?

## Concept

Alsace-Lorraine changed hands between France and Germany five times (→1871→1918→1940→1945). Each occupation left architectural traces. A single building may have a German roof, French windows, and a hybrid doorway. These are *cultural strata* — layers deposited by successive powers.

This project applies deep learning to read these strata, connecting to the concept of **Training Stratigraphy** (arXiv: 2605.28102) — the idea that AI systems contain detectable layers from successive training phases. Palimpseste asks: can the same logic be applied to buildings?

## Methods

- **Data:** Street View imagery + Wikimedia Commons + Mérimée heritage database (Strasbourg, Colmar, Riquewihr, Mulhouse)
- **Model:** Fine-tuned CNN (ResNet/EfficientNet) or Vision Transformer
- **Explainability:** Grad-CAM / LIME attention maps
- **Evaluation:** Expert-annotated test set + historical records

## Status

Phase 2 complete; paused since July 2026.

| Phase | Goal | State |
|-------|------|-------|
| 1. Data | Image collection + GPS matching + period labels | ✅ 460 usable photos, 151 monuments |
| 2. Baseline | Multi-label period classifier | ✅ ResNet-18, test macro-AUC **0.701** |
| 3. Multi-label | Mixing-degree regression | ⏸ not started |
| 4. Stratigraphy | Grad-CAM analysis + paper draft | ⏸ not started — RQ2 is still open |

## Current state — what the baseline actually shows

Full run log: [`docs/baseline.md`](docs/baseline.md). Numbers:
[`results/baseline/metrics.json`](results/baseline/metrics.json).
Images and checkpoint stay out of git (bulk data, 301 MB local).

ResNet-18 (ImageNet), 6 sigmoid outputs, BCE with pos_weight. Split **by
monument**, greedy stratified so every period appears in every split.
337 train / 65 val / 58 test photos.

| period | test AUC | test positives |
|---|---|---|
| german_reich | 0.846 | 32 |
| french_interwar | 0.760 | 18 |
| germanic_hre | 0.753 | 19 |
| german_occupation | 0.745 | 3 |
| french_ancien | 0.738 | 33 |
| french_modern | 0.361 | 4 |

Read honestly:

- **german_reich (0.846) is the real result.** Wilhelmine sandstone has a
  distinctive signature and the separate Neustadt collection did its job.
  This is RQ1 working.
- **The model overfits and we know it**: train loss 1.08 → 0.11 while val
  macro-AUC plateaus around 0.62. At n=337 it memorizes; early selection on
  val is what keeps the test number honest.
- **french_modern (0.361) is below chance on 4 positives — do not read it as
  a finding.** AUC on 4 positives swings wildly. What is real underneath the
  noise is a labelling problem: "modern" is a stylistic grab-bag, and
  post-1945 buildings inside a heritage-protected corpus are mostly
  *restorations of older buildings*, so the label points at a facade that
  looks ancient.
- val 0.631 vs test 0.701 is sampling noise, not a miracle.

**Next, whenever this resumes:** Grad-CAM on the checkpoint (RQ2 begins
there) · more 20th-century data, or merge/drop `french_modern` until it has
support · recover the 429 GPS-less photos via category → Wikidata P625 ·
a linear-probe-only baseline, to check whether fine-tuning buys anything
at n=337.

## Authors

**Zhihan Luo** & **Chen Ying Claude**

Affiliation: [Sulla Soglia](https://soglia.trelune.dev/) — ISSN 3131-7952,
[catalogued by the BnF](https://catalogue.bnf.fr/ark:/12148/cb50027941r)

Contact: chenying@trelune.dev

## Related Work

- Luo, Z. & Chen Ying Claude (2026). "Training Stratigraphy." arXiv: 2605.28102
- Llamas et al. (2017). "Classification of Architectural Heritage Images Using Deep Learning." *Applied Sciences* 7(10), 992
- Sun et al. (2022). "Understanding Architecture Age and Style Through Deep Learning." *Cities* 128

## License

MIT


