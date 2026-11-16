# Maize-leaf classification baseline, 2026-11-15

Training/evaluation code and this report were assisted by OpenAI Codex (`GPT-6`). Metrics are computed from the source images, not generated examples. Human review of the classifier and guidance: pending.

Model: `maize-leaf-lightgbm-0.1.1`, feature version `rgb-hsv-texture-2`. CPU LightGBM baseline over spatial RGB, HSV histograms and texture differences. No pretrained backbone or generative image diagnosis.

## Source audit

Md Shajedur Rahman, *Corn Leaf Disease Classification Dataset*, version 1, [DOI 10.17632/hmkd6nbngr.1](https://data.mendeley.com/datasets/hmkd6nbngr/1), declared CC BY 4.0. Archive: `Corn Leaf Disease.zip`, 1,697,753,372 bytes, SHA-256 `21fa8dcfdac97e093d37b8cd5c94f94b2a226c4dd48e4cc7142d6f708a9d05a9`, verified after download. The landing page's approximate 17,000-image claim differs from the 12,395 image files actually present.

| Class | Images in archive | Test recall |
|---|---:|---:|
| Common rust | 3,024 | 97.11% |
| Corn leaf blight | 3,338 | 92.32% |
| Gray leaf spot | 2,395 | 91.32% |
| Healthy | 1,859 | 98.85% |
| Insect damage | 1,779 | 99.59% |

Augmentation/enhancement suffixes were removed to recover source names. Unions of those names and perceptual image hashes were formed before splitting. Two groups with conflicting class labels (10 images) were excluded. The remaining 12,385 images form 5,064 groups: train 8,702 images/3,577 groups, validation 1,795/743 and test 1,888/744. An audit of the split manifest confirmed no group crosses splits. This does not prove that every physical leaf or acquisition session has been recovered.

## Evaluation

The rule was recorded in training code before fitting real data: macro F1 at least 0.75, balanced accuracy at least 0.75 and recall at least 0.5 in every class. No test-result-driven hyperparameter search was performed. Validation loss chooses the stopping iteration; the test split is not used for fitting.

| Metric on held-out test split | Value |
|---|---:|
| Accuracy | 95.23% |
| Balanced accuracy | 95.84% |
| Macro F1 | 95.43% |
| Minimum class recall | 91.32% |
| Internal gate | Passed |

The full confusion matrix is in `model_card.json`. Most errors are between blight and gray leaf spot: 45 blight images were classified as gray leaf spot and 24 in the reverse direction.

Version 0.1.0 was an initial diagnostic run. A regression check found that JPEG recompression in training did not match runtime preprocessing. Version 0.1.1 uses the same bounded EXIF/downsampling path with lossless staging; the regression test requires identical extracted features. Its grouped split differs, so the two runs are not a controlled performance comparison. The internal acceptance thresholds were unchanged.

## Limits and next steps

- These are same-dataset metrics, not performance on new Guatemalan farms. The source mixes field and repository images; original sources and all near-duplicates have not been independently recovered. Filename/perceptual grouping is an approximation to physical leaf/session grouping.
- Scores are not calibrated disease probabilities. The fixed 0.8 score / 0.2 margin abstention rule is a heuristic; these metrics describe the five-way classifier, not the reliability of accepted diagnoses under that rule.
- Insect damage does not label the insect species, and a healthy candidate does not exclude other disorders. Non-maize or unsupported conditions may receive a high score incorrectly.
- Reference guidance remains a draft for local review. No pesticide product, dose or instruction to spray is inferred from image labels.
- Local artifacts are outside git. The model card records the model checksum. The fitted model is about 1.2 MiB; its split manifest is about 2.5 MiB. Public artifact registration, configured deployment and the frontend flow are not included in this result.
- Reproduce with the command in `services/ml/docs/leaf-screening.md` using the verified source archive. Source images and fitted weights are never downloaded or trained in CI.
