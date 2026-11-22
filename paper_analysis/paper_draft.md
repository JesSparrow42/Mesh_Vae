# Three-dimensional body-shape representations for detecting subclinical atherosclerosis in the REACT study

**Working manuscript, analysis plan, and results scaffold**  
**Status:** the female analyses are interim; the sex-stratified age/BMI/scan comparison and participant-level confidence intervals are the remaining confirmatory analyses.

## Main scientific question

**Does morphology learned from a full-body 3D scan provide reproducible information about silent atherosclerosis beyond age and BMI, and is that incremental information similar in women and men?**

This should be the spine of the paper. The MeshVAE, t-SNE, plaque-volume targets, Gefion scaling, and Boson prior are supporting analyses that explain what the representation learned or test extensions of the method. They should not compete with the primary question.

## Pre-specified hierarchy

| Level | Definition |
|---|---|
| Primary target | Any measured plaque, binary |
| Primary metric | Held-out ROC AUC |
| Primary contrast | Age + BMI + scan latent versus age + BMI |
| Primary populations | Women and men, analyzed separately |
| Primary uncertainty | Paired participant-level stratified bootstrap, 5,000 replicates |
| Secondary classification metrics | Average precision, Brier score, sensitivity, specificity, balanced accuracy, F1 |
| Secondary outcomes | Any coronary artery disease and the continuous plaque-burden targets |
| Representation analyses | Held-out BMI/age prediction, local plaque enrichment, train-defined cluster association |
| Exploratory method analysis | Gaussian versus Boson latent prior |

Threshold-dependent metrics must use a threshold selected from training data only. The held-out test data must never be used to select a threshold, architecture, epoch, or target transformation.

## Candidate abstract

### Background

Obesity and body-fat distribution are associated with cardiovascular risk, but it is unclear whether external three-dimensional body morphology contains information about subclinical atherosclerosis beyond conventional covariates.

### Objective

To determine whether a learned representation of full-body 3D point-cloud scans improves held-out detection of silent atherosclerosis beyond age and BMI, and to assess whether performance generalizes across women and men.

### Methods

We analyzed participants from the REACT study with full-body point-cloud scans and vascular imaging-derived plaque labels. A Mesh Variational Autoencoder (MeshVAE) compressed each scan into a latent representation. On fixed participant-level train and held-out test splits, the same regularized logistic model was fitted using age, BMI, the scan latent, and their combinations. The pre-specified primary comparison was age + BMI + scan versus age + BMI, evaluated separately in women and men using paired participant-level bootstrap confidence intervals. Secondary analyses assessed continuous plaque burden, latent-space morphology, unsupervised cluster enrichment, and Gaussian versus Boson-sampling priors.

### Results

The source poster reports 4,300 participants overall. The final sex-specific complete-case counts are **[[insert female and male train/test counts from cohort_flow.csv]]**. In the previous female-only analysis, mean AUC was 0.810 for age alone, 0.822 for age + scan, and 0.624 for scan alone; a longer Gefion run reached AUC 0.833. In the confirmatory analysis, age + BMI achieved AUC **[[value, 95% CI]]** in women and **[[value, 95% CI]]** in men. Adding the scan latent changed AUC by **[[paired delta, 95% CI]]** and **[[paired delta, 95% CI]]**, respectively. The latent representation predicted BMI with held-out R² **[[value, 95% CI]]**, whereas plaque-status cluster separation was **[[weak/modest; statistic and permutation p]]**. For continuous plaque burden, the strongest interim result was combined carotid plaque volume on the log1p scale (transformed R² 0.348; Spearman 0.595), while raw-scale R² was 0.004. Boson and Gaussian MMD priors had comparable predictive performance (paired transformed-R² difference 0.002, 95% CI -0.015 to 0.031).

### Conclusions

Full-body 3D scans encode a strong body-morphology phenotype. Whether that phenotype adds clinically meaningful information beyond age and BMI should be stated from the paired sex-stratified analysis: **[[insert final conclusion after the new run]]**. Current evidence suggests that most classification signal is captured by ordinary covariates and that any incremental scan contribution is modest.

## Introduction

Subclinical atherosclerosis develops before overt cardiovascular disease and may be detectable through vascular imaging. Age and adiposity are major risk factors, but BMI compresses body composition and fat distribution into a single number. Full-body 3D scanning offers a non-invasive description of external morphology, including regional shape and size, that may carry additional information about cardiometabolic risk.

Deep generative models can learn compact representations of irregular 3D point clouds. A useful representation should satisfy two distinct criteria: it should reconstruct or generate plausible body geometry, and it should contain information relevant to the clinical outcome. Strong reconstruction alone does not imply disease discrimination. Conversely, a clinically useful latent representation may encode gradual morphology rather than visually discrete disease clusters.

The primary aim of this study was therefore to quantify the incremental held-out discrimination obtained from scan-derived morphology beyond age and BMI. Secondary aims were to characterize the biological structure of the latent space, test prediction of individual plaque-burden outcomes, and explore whether a Boson-sampling latent prior improved model performance.

## Methods

### Study population

The poster reports 4,300 REACT participants, approximately 50% male and 18–70 years of age. Processing logs contain 5,044 scan records; these must not be described as 5,044 unique participants. The final manuscript will report a participant-level flow diagram containing:

1. unique participants in the metadata export;
2. participants with a usable body scan;
3. participants retained after scan-quality and metadata matching;
4. participants with a non-missing primary target, age, and BMI;
5. fixed train and held-out test counts, by sex;
6. positives and negatives in each split.

All analyses use participant IDs as the resampling unit. Repeated scans from one participant must be resolved before train/test splitting or kept entirely within one split with participant-clustered resampling.

### Outcomes

The primary outcome is plaque at any measured vascular site. Secondary binary outcomes include any coronary artery disease. Continuous secondary outcomes include right, left, and combined carotid and femoral plaque volume, coronary plaque volume, CCTA segment involvement score, and coronary calcium score. Continuous outcomes trained after a log1p transform are evaluated both on the transformed scale and in original units.

### Scan preprocessing

Full-body point clouds were simplified, padded, and accompanied by scan-geometry features. Anatomical point labels divided the scan into six regions. The processing workflow and all exclusions should be described from `gefion_preprocess.sh`, `build_scan_geometry_features.py`, and the generated audit CSV rather than reconstructed from memory.

### MeshVAE representation

The MeshVAE encodes a point cloud into a stochastic latent representation and decodes it back to body geometry. The confirmatory modality experiment uses the deterministic posterior mean (`mu`) from a scan-only encoder that does not receive age, BMI, sex, or plaque status as an input feature. Plaque labels supervise the auxiliary classifier during training, using only the training cohort. This is therefore a supervised scan representation, not a fully unsupervised phenotype.

The primary analysis uses one frozen architecture and optimization protocol for both sexes. Architecture search and the earlier 15-model comparison are treated as development analyses. A fixed seed defines the confirmatory model; additional seeds assess training stability but do not replace participant-level uncertainty.

### Age/BMI/scan comparison

For each sex, the same complete-case train and test participants are used for all seven input sets:

- age;
- BMI;
- scan latent;
- age + BMI;
- age + scan;
- BMI + scan;
- age + BMI + scan.

Each input set is evaluated with the same standardized L2-regularized logistic regression. Regularization is chosen by stratified cross-validation within the training split. A classification threshold is selected from out-of-fold training predictions by balanced accuracy, after which the fitted model is evaluated once on the held-out test split.

The key effect is not whether the final AUC exceeds 0.5. It is the paired change in AUC when the scan latent is added to age and BMI on the same held-out participants.

### Confidence intervals

All held-out metrics receive 95% participant-level percentile-bootstrap intervals using 5,000 replicates. Binary resampling preserves case/control counts. Continuous zero-inflated plaque outcomes preserve zero/non-zero counts when both groups are present. Model differences use the same participant resample for both models, yielding paired intervals.

Repeated-seed intervals and participant-bootstrap intervals answer different questions and are reported separately:

- **participant bootstrap:** uncertainty from having sampled this cohort;
- **repeated seeds:** variability from model initialization and optimization.

Architecture min–max ranges from the earlier presentation are descriptive and must not be labelled confidence intervals.

### Latent-space analysis

t-SNE is used only for visualization. Its axes, global distances, island boundaries, and apparent cluster counts are not interpreted as statistical evidence. Quantitative analyses use standardized original latent vectors after a PCA transformation fitted on training participants only.

The following tests are pre-specified:

- held-out k-nearest-neighbour AUC for local plaque enrichment;
- plaque-label silhouette on held-out original latent vectors with a permutation test;
- k-means cluster number selected by training silhouette only;
- held-out association between train-defined clusters and plaque status using Cramér's V and a permutation test;
- held-out ridge-regression R² for age and BMI from the latent representation.

This design distinguishes a morphology gradient from a true disease-defined cluster. A visually smooth BMI gradient with weak plaque clustering would support the interpretation that the encoder learns body shape well but that plaque is not the dominant organizer of that shape.

### Statistical analysis

ROC AUC is the primary classification metric because it quantifies ranking across all possible thresholds. In this context, AUC 0.83 means that a randomly selected plaque-positive participant receives a higher model score than a randomly selected plaque-free participant approximately 83% of the time. It does not mean that 83% of participants are classified correctly.

Average precision is reported because it reflects positive-class ranking and depends on prevalence. Brier score assesses probabilistic error. Sensitivity, specificity, balanced accuracy, and F1 are secondary threshold-dependent metrics. Calibration plots and calibration slope/intercept should be added if predicted probabilities are presented as risks rather than scores.

For continuous outcomes, transformed R² is the primary metric only when the model was trained on that transformed target. Raw-scale R², MAE, RMSE, and Spearman correlation are reported alongside it so that transformed-scale ranking is not mistaken for accurate prediction in original units.

The eleven-target benchmark is exploratory. Effect sizes, sample sizes, and confidence intervals are reported for every target. Formal target-specific claims require false-discovery-rate control or independent confirmation.

## Results

### Cohort

**[[Insert Table 1 and cohort-flow figure from generated `cohort_flow.csv` files.]]**

Important reconciliation before submission:

- 4,300 is the poster's participant count;
- 5,044 is the number of processed scan records, not necessarily unique participants;
- 1,648 was the usable female training cohort in one Gefion run before the internal validation split;
- 363 was the held-out sample size for the interim female any-plaque diagnostic;
- every final result must show its own target-specific `N`.

### Primary modality comparison

**[[Insert Figure 2: `modality_auc_by_sex.pdf`.]]**

In women, age + BMI produced held-out AUC **[[estimate, CI]]**. Adding the scan latent produced AUC **[[estimate, CI]]**, a paired change of **[[delta, CI]]**. In men, the corresponding values were **[[...]]**. **[[State whether each interval excludes zero, but emphasize magnitude and precision rather than a binary significance label.]]**

Age alone achieved **[[...]]**, BMI alone **[[...]]**, and the scan latent alone **[[...]]**. These secondary comparisons indicate whether the scan mostly recovers age/adiposity information or supplies additional morphology.

### Interim model-development results

The earlier female development comparison averaged AUC 0.810 for age only, 0.822 for age + scan, and 0.624 for scan only. The strongest individual model reached AUC 0.828. A longer 1,000-epoch Gefion run reached AUC 0.8332, indicating little gain from substantially longer optimization. These results motivated the cleaner fixed-classifier comparison with BMI and should not be pooled with the confirmatory estimates.

### Cardiovascular outcomes

In the interim female target benchmark, any measured plaque had AUC 0.838 (held-out N=363), whereas any coronary artery disease had AUC 0.777 (N=355). Participant-level bootstrap intervals from the saved prediction files will replace the previous approximate analytic intervals.

### Continuous plaque burden

Combined carotid plaque volume was the strongest interim continuous outcome: transformed R² 0.348 and Spearman 0.595. Raw-volume R² was 0.004, indicating useful ordering on a compressed scale but poor absolute-volume prediction. Other outcomes were generally modest, near zero, or negative; coronary plaque volume had only 70 held-out observations and should be described as imprecise.

### Latent morphology and cluster signal

**[[Insert Figure 3: plaque status, BMI, and train-defined clusters on one t-SNE map.]]**

The scan-only latent representation showed **[[BMI held-out R², CI]]** and **[[age held-out R², CI]]**, demonstrating that it captures body morphology and age-related shape. Plaque-label silhouette was **[[estimate, permutation p]]**, and the association between train-defined clusters and plaque status was **[[Cramér's V, permutation p]]**. These quantitative tests **[[did/did not]]** support a discrete plaque-related cluster beyond the visible morphology gradient.

### Held-out reconstruction examples

**[[Insert Figure 4: `figures/reconstruction_examples.tiff`.]]**

Figure 4 compares genuine held-out point-cloud inputs with deterministic posterior-mean reconstructions from the best checkpoint. Within each target group, the displayed participant is selected by median reconstruction error rather than visual appearance. These examples document reconstruction fidelity and are not evidence of plaque separation or classification performance.

### Boson prior

Across five matched seeds, mean transformed R² was 0.407 for Gaussian KL, 0.400 for Gaussian MMD, and 0.402 for Boson MMD. The paired Boson-MMD minus Gaussian-MMD effect was 0.002 (95% CI -0.015 to 0.031). The experiment therefore did not demonstrate a reliable predictive advantage for the Boson prior. This is an exploratory methods result, not the paper's primary clinical conclusion.

## Discussion

### Principal interpretation

The current evidence supports a two-part conclusion. First, the scan encoder clearly learns meaningful body morphology: body-size variables form smooth latent gradients and can be predicted from the original latent vectors. Second, plaque status is not the dominant organizer of that morphology: scan-only discrimination is modest and plaque-positive and plaque-free participants overlap strongly in the visual embedding.

The remaining decisive question is whether scan morphology improves prediction after BMI as well as age is included. A small positive paired delta with a narrow interval would support measurable but limited incremental information. An interval spanning clinically negligible effects would support the stronger conclusion that the scan largely recovers conventional adiposity information. A sex difference should only be claimed from a formal interaction or clearly separated estimates with appropriate uncertainty, not from one sex being nominally significant and the other not.

### Why AUC can exceed accuracy

AUC evaluates every possible threshold and only asks whether cases tend to receive higher scores than controls. Accuracy evaluates one chosen threshold and can be reduced by class imbalance, poor probability calibration, or an inappropriate 0.5 cutoff. The observed combination of respectable AUC and lower sensitivity at 0.5 therefore indicates a threshold/calibration issue, not a contradiction.

### Strengths

- full-body 3D point-cloud data rather than manually selected anthropometric measurements;
- fixed participant-level held-out evaluation;
- explicit age, BMI, and scan ablations with one classifier family;
- sex-stratified analysis;
- participant-level paired confidence intervals;
- quantitative latent-space tests rather than relying on t-SNE appearance;
- preservation of predictions and manifests for reproducibility.

### Limitations

- the existing headline results are primarily female and need male replication;
- the cohort flow currently mixes participant and scan-record counts;
- model development involved multiple architectures and outcomes, increasing selection optimism;
- missing vascular measurements produce different sample sizes across outcomes;
- the scan latent is supervised for plaque and is not a purely unsupervised phenotype;
- one held-out split cannot establish external generalizability;
- t-SNE is sensitive to hyperparameters and is non-metric;
- continuous plaque results are much stronger on the transformed than original scale;
- calibration and clinical decision utility have not yet been established.

## Proposed final figures and tables

1. **Figure 1:** cohort flow and analysis design.
2. **Figure 2:** sex-stratified AUC dot-and-whisker plot for age, BMI, scan, and combinations.
3. **Figure 3:** t-SNE colored by plaque, BMI, and train-defined clusters, accompanied by original-latent statistics.
4. **Figure 4:** held-out point-cloud inputs and deterministic checkpoint reconstructions selected by median within-group reconstruction error.
5. **Figure 5:** continuous target transformed R² with participant-bootstrap intervals and raw-scale result annotations.
6. **Table 1:** cohort characteristics by split and sex.
7. **Table 2:** primary model metrics and paired scan increment.
8. **Supplementary Table S1:** all eleven cardiovascular targets with N and 95% intervals.
9. **Supplementary Table S2:** Gaussian/Boson matched-seed ablation.
10. **Supplementary Figure S1:** calibration curves and threshold metrics.

The analysis scripts additionally generate `cohort_characteristics.md`,
`all_result_intervals.csv`, and `calibration_metric_intervals.csv`; these are the
source-of-truth files for manuscript numbers, not values copied from terminal
screenshots or earlier presentation slides.

## Minimum completion checklist

- [ ] Freeze one metadata export and document its date/version.
- [ ] Reconcile unique participants versus scan records.
- [ ] Confirm zero train/test participant overlap.
- [ ] Run the identical scan-only model in women and men.
- [ ] Run age/BMI/scan linear probes on the same complete-case participants.
- [ ] Generate participant-bootstrap intervals for every prediction file.
- [ ] Report paired delta AUC for the primary comparison.
- [ ] Quantify latent age/BMI signal and plaque clustering in original latent space.
- [ ] Add calibration plots for primary binary models.
- [ ] Report every target-specific N.
- [ ] Repeat the primary scan model across at least three seeds as a stability analysis.
- [ ] Keep the held-out test set untouched by model and threshold selection.
- [ ] Replace every manuscript placeholder only from generated CSV/JSON outputs.
