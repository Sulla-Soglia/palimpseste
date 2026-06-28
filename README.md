# palimpseste


# Palimpseste

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

## Timeline

| Phase | Period | Goal |
|-------|--------|------|
| 1. Data | Jul 2026 | Image collection + annotation |
| 2. Baseline | Jul-Aug 2026 | Binary French/German classifier |
| 3. Multi-label | Aug 2026 | Mixing-degree regression |
| 4. Stratigraphy | Aug-Sep 2026 | Grad-CAM analysis + paper draft |

## Authors

**Zhihan Luo** & **Chen Ying Claude**

Affiliation: [Sulla Soglia](https://sullasoglia.substack.com) (ISSN pending)

Contact: chenying@trelune.dev

## Related Work

- Luo, Z. & Chen Ying Claude (2026). "Training Stratigraphy." arXiv: 2605.28102
- Llamas et al. (2017). "Classification of Architectural Heritage Images Using Deep Learning." *Applied Sciences* 7(10), 992
- Sun et al. (2022). "Understanding Architecture Age and Style Through Deep Learning." *Cities* 128

## License

MIT


