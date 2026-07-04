# Repository Map

A structural overview of the project.

Gaps are due to deleted one-off diagnosis scripts, which were not
important for the main story.

---

## `scripts/` - executable experiments

Numbered roughly in execution order. Outputs (figures) go to 
`results/figures/`. Tables go to `results/tables/`.

### Initial pipeline and exploratory comparison
```
03_methods_comparison.py             Side-by-side run of multiple explanation methods 
                                     on the same model and image. One row per method.
04_check_models.py                   Load all three models and verify target layers 
                                     exist. No heatmaps, just model-bundle integrity.
05_gradcam_all_models.py             GradCAM on all three architectures (CNN, ViT, 
                                     hybrid). Validates if the reshape_transform 
                                     mechanism works for the transformer model.
```

### Matrix construction
```
06_method_model_matrix.py            Consolidation: all available methods × all 
                                     models on one image. Methods × models matrix.
```

### LRP via zennit
```
07_lrp_resnet50.py                    First LRP test: EpsilonGammaBox composite on 
                                      ResNet50, single image. Isolated check before 
                                      matrix integration.
07.5_method_model_matrix_with_lrp.py  Extended matrix: adds LRP as a row, 
                                      architecture-specific variant per cell (zennit 
                                      EpsilonGammaBox for ResNet, Chefer-LRP for ViT, 
                                      n/a for MobileViT).
08_lrp_mobilevit.py                   LRP (same composite) on MobileViT-S. The composite 
                                      is designed for conv-style architectures. This 
                                      script tests whether the default behavior on 
                                      attention is plausible or pathological.
```

### ViT inspection and Chefer-LRP construction
```
09_inspect_vit.py                    Inspect timm ViT-B/16: transformer-block 
                                     structure, submodule order, residual 
                                     connections. Preparation for the Chefer-LRP 
                                     wrapper design. Console output only.
10_test_attention_capture.py         Verify that AttentionWithCapture is a faithful 
                                     replacement: (a) wrapped model is numerically 
                                     identical to the original; (b) captured 
                                     matrices have the expected shape and are valid 
                                     softmax outputs.
11_test_attention_gradients.py       Verify that AttentionWithCapture correctly 
                                     retains gradients of the attention matrices 
                                     after a backward pass from the target-class logit.
12_chefer_lrp_first_heatmap.py       First end-to-end run of manual Chefer-LRP on 
                                     ViT-B/16. Heatmap visualized alongside the input.
```

### Register-token diagnostics
```
13_check_token_norms.py              Are the bright spots in Chefer-LRP on background
                                     locations register-token artifacts? Compute L2 
                                     norms of patch tokens after the final transformer
                                     block, visualize as 14×14 grid.
14_register_token_diagnostic.py      Multi-method comparison on the register-token 
                                     positions: which methods detect them, which miss 
                                     them. IG, GradCAM, Chefer-LRP side by side with 
                                     raw token-norm ground truth.
```

### Chefer-LRP self-consistency tests 
```
15_chefer_selfconsistency.py         Three property-based tests on Chefer-LRP for 
                                     ViT: per-block conservation, determinism, 
                                     target sensitivity. Each tests a claim that 
                                     follows from the algorithm's specification. 
16_chefer_param_randomization.py     Parameter sensitivity across 30 random seeds.
                                     The diagnostic value lies in the distribution 
                                     of Spearman correlations, not in any single 
                                     heatmap. Prototype of the Cascading Parameter 
                                     Randomization sanity check (Adebayo et al., 2018).
```

### MobileViT hybrid LRP
```
17_inspect_mobilevit.py              Inspect timm MobileViT-S: top-level structure, 
                                     transformer blocks inside MobileVitBlocks, 
                                     conv↔token boundary shape transitions. Console 
                                     output only.
18_chefer_lrp_mobilevit.py           First end-to-end run of Chefer-LRP variant on
                                     MobileViT-S. Transformer stages only. conv 
                                     stages and unfold boundary deliberately not 
                                     included.
19_chefer_lrp_mobilevit_per_block.py Diagnostic: per-MobileVitBlock heatmaps + final 
                                     aggregated. Reveals whether individual blocks 
                                     produce meaningful signals (and aggregation 
                                     destroys them), or whether each block is already 
                                     chaotic.
```

### Dataset axis: subset, batch generation, validation
```
20_timing_benchmark.py               Measures per-heatmap compute cost for each 
                                     (method, model) combination on a tiny sample, 
                                     then projects the total compute budget for 
                                     the dataset axis at 500 / 1000 / 2000 images 
                                     and one or two targets per image. Console 
                                     output only.
21_build_subset.py                   Builds a stratified subset of the ImageNet-1k 
                                     validation set and writes the manifest CSV 
                                     that drives all downstream dataset-axis scripts.
                                     Configuration (N_CLASSES, SAMPLES_PER_CLASS) at 
                                     the top. Includes a Top-5 sanity check that 
                                     class_idx values are consistent with what 
                                     timm-pretrained ResNet50 actually predicts, 
                                     catches sorting bugs and off-by-one errors 
                                     before they corrupt downstream comparisons.
22_generate_heatmaps.py              Batch runner: generates heatmaps for every
                                     (sample, model, method, target) combination in
                                     the manifest and saves them as compressed .npz
                                     per (sample, model). Models-outer/samples-inner
                                     loop. Atomic writes, resume-safe. Random baseline
                                     re-seeded per (sample, model, target) so the null
                                     floor is independent on every comparison axis.
25_validate_heatmaps.py              Plausibility check for the heatmaps produced by 
                                     script 22. Loads .npz files for a few sample images, 
                                     visualizes each as a methods × models matrix overlaid
                                     on the input image, and saves one figure per sample. 
                                     Uses per-model resize+crop so the display image 
                                     matches what each model saw, avoiding shifted 
                                     overlays. GT-target heatmaps ('_gt' keys).

```

### Comparison metrics
```
27_compute_comparison_metrics.py     Pairwise similarity metrics (Spearman, SSIM,
                                     HOG-Pearson) across three axes: cross_method
                                     (same model, different methods, GT target),
                                     cross_model (same method family, different
                                     models, GT target), and cross_target (same
                                     model and method, GT vs Pred). Reads the .npz
                                     files from script 22. Output: results/tables/
                                     comparison_metrics__<manifest_stem>.parquet.
                                     Cross-model comparisons involving MobileViT
                                     (256×256) are downsampled to 224×224
                                     (cv2.INTER_AREA) for shape parity. ResNet50's
                                     EpsilonGammaBox-LRP and ViT-B/16's Chefer-LRP
                                     are paired cross-model under a shared family
                                     label.

```


### Visualization and supplementary analysis
```
28_visualize_comparison_metrics.py   Loads the comparison-metrics Parquet from
                                     script 27, aggregates to median + IQR per
                                     (axis, item_a, item_b, metric), and produces
                                     boxplots and violinplots for all three axes.
                                     Non-Random pairs sorted by median Spearman
                                     descending, Random pairs at the right end.
                                     Adds two filtered cross_target variants to
                                     surface method target-sensitivity that the
                                     unfiltered distribution masks: per-model
                                     misclassified samples, and samples
                                     misclassified by all three models. Outputs
                                     the aggregated Parquet and figures for all
                                     three axes.
29_compute_classification_accuracy.py   Reads class_idx_gt and class_idx_pred from
                                        the .npz files generated by script 22 and
                                        produces a small table summarizing per-model
                                        classification accuracy and the
                                        misclassification-overlap structure
                                        (any-model-wrong, all-models-wrong). Output:
                                        results/tables/classification_accuracy__<manifest_stem>.parquet.
```

### Cascading parameter randomization (Experiment 1)
```
30a_inspect_layer_names.py           Diagnostic: prints named_children of each
                                     model at top level and one level deeper
                                     inside the containers split by the cascading
                                     schedule (resnet50.layer*, vit_base.blocks,
                                     mobilevit_s.stages). Console output only.
                                     Used as a reference for the randomization
                                     schedule paths in src/randomization_schedule.py.
30b_inspect_cascading_heatmaps.py    Diagnostic: loads the ViT cascading heatmaps,
                                     scans for NaN/Inf/constant (degenerate)
                                     values, computes per-(method, stage)
                                     min/max/mean/std statistics, and renders
                                     an inspection figure (input image on top,
                                     then method rows × cascading-stage columns;
                                     "CONSTANT" marker for degenerate heatmaps).
                                     Sanity check on the cascading pipeline
                                     output.
30c_test_reinit.py                   Diagnostic: side-by-side comparison of
                                     per-layer parameter standard deviations
                                     for the trained model, naive Gaussian
                                     randomization (randn × 0.02), and
                                     layer-wise re-initialization. Verifies
                                     that layer-wise re-init produces per-layer
                                     std values matching each layer's own
                                     initialization distribution. Console
                                     output only.
30d_saturation_test_reinit.py        Diagnostic: runs cascading randomization
                                     under layer-wise re-init for all three
                                     models on one sample, captures the
                                     GradCAM target-layer activations at each
                                     stage, and verifies that consecutive
                                     stages produce measurably different
                                     activations and that activations remain
                                     numerically well-behaved (no NaN/Inf,
                                     no explosion). Console output only.
30_cascading_param_randomization.py  Batch runner for cascading parameter
                                     randomization: for each (model, stage, seed),
                                     reload the model fresh, layer-wise
                                     re-initialize all parameters in the cumulative
                                     path set of this stage (head-most group first),
                                     then compute heatmaps for every applicable
                                     method and sample in the manifest. Output:
                                     one .npz per (sample, model, stage, seed)
                                     under results/heatmaps_cascading/<manifest_stem>/.
                                     GT target only, since the cascading question
                                     is about model parameters, not class
                                     sensitivity. Random baseline re-seeded per
                                     (sample, model, stage, seed) so the null floor
                                     is independent across cascading depth.
                                     Resume-safe (atomic writes, skip on output
                                     existence).
```

### Analysis of Cascading parameter randomization (Experiment 1)
```
31a_multiseed_validation.py          Multi-seed validation on the 10-sample
                                     subset (seeds 42/43/44). Two analyses:
                                     (1) inter-seed vs. inter-sample variance
                                     of cascading-vs-baseline Spearman, as
                                     empirical backing for the single-seed
                                     design choice; (2) cell-median drift
                                     across seeds, usable as a threshold
                                     heuristic for claim strength. Output:
                                     console tables and figure.
31a2_inspect_mobilevit_gradcam.py    Diagnostic follow-up to 31a: pixel-identity
                                     check of MobileViT-S GradCAM cascading
                                     heatmaps across stages 1-4. Verifies whether
                                     stages that produce identical summary
                                     statistics also produce bit-identical
                                     heatmaps. Console output only.          
31_compute_cascading_metrics.py      Main aggregation script for Experiment 1:
                                     for each (sample, model, stage, method),
                                     loads the cascading heatmap (seed 42) and
                                     the trained baseline (scripts/22), computes
                                     Spearman, SSIM, and HOG-correlation, and
                                     writes a wide-format Parquet with one row
                                     per cascading heatmap (55,000 rows total).
                                     Output: results/tables/cascading_metrics__<manifest_stem>.parquet.
                                     Mirrors scripts/27's metric implementations
                                     exactly for cross-experiment comparability.
                                     Single-threaded. Multi-seed heatmaps (seeds
                                     43, 44) are NOT processed here, they are
                                     exclusively for scripts/31a.
```

### Visualization and findings synthesis (Experiment 1)
```
32_plot_cascading_curves.py          Reads the cascading metrics Parquet from 
                                     script 31, aggregates to q25/q50/q75 per 
                                     (model, family, stage), and injects an 
                                     'original' anchor at stage_idx = -1 with 
                                     value 1.0 (baseline vs. baseline). Output:
                                     figure. 3×3 grid (rows = Spearman ρ, SSIM, 
                                     HOG-Pearson; columns = ResNet50, ViT-B/16, 
                                     MobileViT-S) showing median + IQR 
                                     cascading-vs-baseline decay per method family.
33_plot_collapse_and_tables.py       Companion analyses for the cascading curves
                                     figure. Reads the cascading metrics Parquet
                                     from script 31 and produces (1) median and
                                     IQR-width tables per (model, method_family,
                                     stage) for Spearman, SSIM, and HOG; (2) a
                                     collapse-fraction plot (1×3 grid, one panel
                                     per model) showing the share of samples
                                     with Spearman exactly 0.0 per (model,
                                     family, stage) as a lower-bound proxy for
                                     constant-heatmap collapse. Output: console
                                     tables and figure.
```

### Pixel flipping / deletion (Experiment 2)
```
34_pixel_flipping.py                 Batch runner for the deletion faithfulness
                                     experiment. Reuses the stored GT-target
                                     heatmaps from script 22 (no recomputation),
                                     aggregates each heatmap into 8×8 blocks by
                                     mean, ranks the blocks, and masks them in
                                     both MoRF (most relevant first) and LeRF
                                     (least relevant first) order at fractions
                                     0%, 5%, ..., 50%. Records the GT-class
                                     softmax at each masking fraction. Masking
                                     baseline is the per-channel image mean of
                                     the preprocessed input. Output: two
                                     Parquets in results/tables/, one with the
                                     confidence-vs-fraction curves and one with
                                     per-sample AUCs (trapezoidal over
                                     [0, 0.5]).
35_pixel_flipping_analyze.py         Analysis of Experiment 2 (pixel flipping)
                                     results. Reads the curves and AUC Parquets
                                     from script 34 and produces three tables
                                     (median AUC per (model, method, order);
                                     per-sample LeRF - MoRF diff distribution
                                     with sign-flip counts; raw per-sample
                                     diffs) and three figures
                                     (confidence-vs-fraction curves as a 2×3
                                     grid, per-sample diff boxplot, median-AUC
                                     heatmaps for MoRF and LeRF side by side).
                                     Output: CSVs in results/tables/ and
                                     figures.
```

---

## `src/` - structural scripts
```
models.py                           Model loading and architecture-specific
                                    metadata. Each model is loaded via timm and
                                    wrapped in a ModelBundle carrying the model
                                    itself plus the metadata needed to explain it.
paths.py                            Project path utilities. Resolves the project
                                    root regardless of where a script is executed
                                    from and exposes the standard data, results,
                                    figures, and tables directories.
randomization_schedule.py           Cascading parameter randomization schedules
                                    per architecture. Defines which layer groups
                                    are randomized at each cascading stage: stage
                                    k replaces the parameters of stages[0..k]
                                    (inclusive) with random values, head-most
                                    group first. Provides helpers to resolve
                                    dotted layer paths (e.g. "blocks.9" →
                                    model.blocks[9]) and to validate a schedule
                                    against an actual model.
```

---

## `src/methods` - structural scripts
```
base.py                             Abstract base class for all explanation
                                    methods. Defines the common interface
                                    (explain(x, target) → 2D heatmap) and a
                                    shared [0, 1] normalization helper that all
                                    concrete methods use before returning.
chefer_lrp.py                       Manual Chefer-style LRP for Vision
                                    Transformers. Provides two attribution
                                    methods: CheferLRPMethod for standard ViT
                                    models (timm's vit_base_patch16_224) and
                                    CheferLRPMobileViTMethod for the transformer
                                    stages of MobileViT-S. Both install
                                    AttentionWithCapture wrappers in place of
                                    the original attention modules so post-softmax
                                    attention matrices and their gradients can 
                                    be captured during forward and backward passes 
                                    for relevance computation.
gradcam.py                          GradCAM explanation method. Thin wrapper
                                    around the grad-cam library
                                    (pytorch_grad_cam), which supports both
                                    CNN-style feature maps and transformer
                                    token sequences via the reshape_transform
                                    hook.
integrated_gradients.py             Integrated Gradients explanation method.
                                    Thin wrapper around captum's
                                    IntegratedGradients.
lrp.py                              LRP via zennit's EpsilonGammaBox composite.
                                    Suitable for CNN-style models (ResNet).
                                    ViT is not supported here, transformers
                                    need transformer-specific propagation rules
                                    (see chefer_lrp.py).
random_baseline.py                  Random-heatmap control. Returns a randomly
                                    generated 2D map, independent of the model.
                                    Used as the noise-floor reference in every
                                    comparison. A real explanation method must
                                    beat this to be worth anything. Must be
                                    re-seeded per (sample, model, target) in
                                    batch runs, which the batch runners handle
                                    inline.
```
