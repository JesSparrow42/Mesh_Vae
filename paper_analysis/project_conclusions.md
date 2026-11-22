# REACT-Q: project conclusion and decision framework

## Main goal

Determine whether morphology learned from full-body 3D scans adds reproducible
information about silent atherosclerosis beyond age and BMI, and whether that
incremental information is present in both women and men.

This is narrower and stronger than asking whether a neural network can predict
plaque at all. Age alone already predicts plaque reasonably well. The scientific
value of the scan is therefore the **paired increment beyond age and BMI** on the
same held-out people.

## Primary result to report

- Outcome: any measured plaque, binary.
- Populations: women and men separately.
- Baseline: age + BMI.
- Scan model: age + BMI + deterministic scan latent.
- Metric: held-out ROC AUC.
- Effect: `AUC(age + BMI + scan) - AUC(age + BMI)`.
- Uncertainty: paired participant-level stratified bootstrap, 5,000 replicates.

The result is not complete until that effect and its 95% interval are available
for both sexes. The two individual AUC intervals are useful, but the paired
difference interval answers the actual question.

## What the current evidence already supports

1. The scan representation contains body-morphology information. Earlier t-SNE
   plots show smooth body-size gradients, and this will be tested directly by
   held-out age and BMI prediction from the original latent vectors.
2. Plaque status does not visually form clean, separate groups in the latent
   representation. That is compatible with a diffuse risk phenotype and does
   not by itself mean the representation is useless.
3. In the female development analyses, age was a strong predictor, scan alone
   was modest, and adding scan to age produced only a small AUC increase. BMI was
   not handled consistently enough in those comparisons, which is why the clean
   age/BMI/scan probe is now the decisive analysis.
4. Combined carotid plaque burden was predictable on a compressed log scale,
   but not accurately in raw volume units. This is a secondary, target-specific
   result rather than the main project conclusion.
5. The matched five-seed experiment did not show a reliable predictive benefit
   of the Boson MMD prior over Gaussian MMD. That is a useful negative methods
   result, not evidence against the clinical scan question.

## What not to claim

- Do not say AUC 0.83 means 83% of participants were classified correctly.
- Do not claim that visible t-SNE islands are biological subtypes without the
  held-out original-latent permutation tests.
- Do not call architecture min-max ranges confidence intervals.
- Do not call 5,044 processed scan records 5,044 unique participants.
- Do not claim a sex difference because one sex has a nominally significant
  result and the other does not. A sex interaction or direct contrast is needed.
- Do not describe the scan latent as unsupervised when the plaque auxiliary head
  was used during encoder training.
- Do not present transformed R² as accuracy in raw plaque-volume units.

## Pre-specified interpretation after the new runs

### Outcome A: clear incremental scan information

Use this conclusion only if the paired AUC increment is positive, precise, and
large enough to matter in both sexes:

> Full-body scan morphology added reproducible discrimination beyond age and BMI
> for silent atherosclerosis in both women and men.

Report the absolute increment and interval; do not rely only on a p-value.

### Outcome B: small or uncertain increment

Use this if the paired interval includes zero or only very small effects:

> Full-body scans encoded body morphology, but most plaque-discrimination signal
> was captured by age and BMI; evidence for additional scan information was
> limited or imprecise.

This is still a coherent and publishable result because it separates phenotype
learning from incremental clinical utility.

### Outcome C: sex-specific increment

If one sex appears stronger:

> The estimated scan increment differed in magnitude between women and men, but
> evidence for effect modification requires a direct sex-by-scan comparison and
> should be interpreted cautiously.

Do not write “effective in women but not men” based only on separate confidence
intervals.

## Role of the t-SNE and cluster work

t-SNE is the visual entry point, not the statistical result. The supporting
question is: **what organizes the scan latent space?**

- High held-out BMI R²: the latent primarily reflects body size/composition.
- kNN plaque AUC above 0.5: plaque-positive people have some local enrichment.
- Near-zero plaque silhouette: plaque status is not a discrete geometric split.
- Small Cramér's V with a weak permutation result: train-defined morphology
  clusters are not strongly plaque-defined.
- Strong BMI gradients plus weak plaque clusters: the most likely interpretation
  is a continuous morphology-risk relationship, not distinct disease subtypes.

These findings explain *why* a model may have useful AUC even when red and blue
points overlap on t-SNE.

## Final run order

1. Freeze and record the metadata and split files.
2. Train the identical scan-only model for women and men.
3. Export deterministic held-out latents.
4. Run the seven matched age/BMI/scan logistic probes.
5. Compute participant-bootstrap CIs and the paired primary AUC increment.
6. Run the cohort table and calibration analyses.
7. Quantify latent BMI/age signal and plaque clustering; generate t-SNE only as
   its visualization.
8. Run the all-results CI sweep for every saved target prediction file.
9. Repeat the primary scan model with at least three seeds as an optimization
   stability analysis, kept separate from participant uncertainty.
10. Replace manuscript placeholders only from generated CSV/JSON outputs.
