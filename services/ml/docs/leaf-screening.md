# Maize-leaf image screening

This is a separate model from NDVI and rotation scoring. It classifies a photo among common rust, corn leaf blight, gray leaf spot, healthy and generic insect damage. It does not identify the insect species, predict growth or select a pesticide.

## Source and evaluation

Selected source: Md Shajedur Rahman, *Corn Leaf Disease Classification Dataset*, version 1 (2026), DOI [10.17632/hmkd6nbngr.1](https://data.mendeley.com/datasets/hmkd6nbngr/1). The repository declares CC BY 4.0 and describes a mixture of field and public-repository images. Original-source provenance and the coverage of Guatemalan fields remain unverified.

The first CPU baseline uses LightGBM with spatial RGB features, HSV histograms and texture differences. Pillow is added for bounded JPEG/PNG decoding and EXIF orientation. No pretrained network or external image API is called.

Training groups images by original filename (removing augmentation suffixes) and perceptual image hash before a deterministic 70/15/15 train/validation/test split. Conflicting labels within a group are excluded and counted. Near-duplicates can remain because there are no verified physical leaf/session IDs. This image-specific grouped split is separate from the whole-year split required for the seasonal climate/NDVI models; those rules are unchanged. No augmentation is generated before splitting.

The internal acceptance rule, fixed before fitting, is macro F1 at least 0.75, balanced accuracy at least 0.75 and recall at least 0.5 in every class. Passing this rule is not external field validation or score calibration. The model card reports all class recalls and the confusion matrix. The runtime refuses an artifact that failed this gate.

## Reproduce

Keep downloaded images, archives and trained models outside git:

```bash
uv run python -m agrofarm_ml.leaves.training \
  --archive /path/to/Corn-Leaf-Disease-v1.zip \
  --out /path/to/models/maize-leaf/0.1.1
```

The output directory contains `model.lgb.txt`, `model_card.json` and `split_manifest.json`. The card records archive and model checksums, dataset version, exclusions, grouping, metrics and limitations. The manifest records the split of every included file. Test-only synthetic images verify training and loading mechanics; their metrics are never reported as real-data performance.

Source images are bounded at 25 MiB / 64 million pixels and normalized to at most 1024 pixels per side using lossless PNG staging. Runtime uploads keep the stricter public limits. Both paths use the same orientation/downsampling and feature extraction; a regression test checks feature equality. Real-data results and the source audit are recorded in `experiments/2026-11-15-maize-leaf/` at the repository root.

## Internal inference contract

Configure `LEAF_MODEL_DIR` with the artifact directory before starting the ML service. A supplied artifact with incompatible classes, failed gate or checksum mismatch prevents startup. With no directory configured, the existing climate and rotation routes remain usable and the new route returns 503.

`POST /v1/leaves/assess` accepts JPEG or PNG **bytes**, not JSON or a remote image URL. Maximum: 5 MiB and 16 million decoded pixels. Images are processed in memory; the service does not save the photo or use it for retraining. The public core API must authenticate the caller, enforce farm access and forward the bytes with a service identity token before the frontend can use this internal endpoint. Cloud Run invocation remains private.

The response includes all class scores, a candidate or `uncertain`, required local review, reference links and model/data provenance. Scores are not calibrated probabilities. A top score below 0.8 or a margin below 0.2 returns `uncertain`; these are initial abstention heuristics, not a validated safety guarantee. Non-maize photos can still be misclassified even with a high score. A healthy candidate does not rule out another disorder.

## Guidance scope

Reference links come from [University of Minnesota Extension](https://extension.umn.edu/corn/corn-pest-management), [Cornell](https://www.vegetables.cornell.edu/pest-management/disease-factsheets/sweet-corn-diseases-and-control-measures/) and [FAO scouting guidance](https://www.fao.org/fileadmin/templates/fcc/Fall_Armyworm/web_guidance-note-2.pdf). These are references for local review, not Guatemala-specific prescriptions. The classifier returns inspection and confirmation steps, with no product, dosage or instruction to spray. The crop-traits review does not automatically approve this separate guidance catalogue.

Frontend integration, core API forwarding, Cloud Storage artifact registration and staging deployment are separate work. Do not describe the live app as supporting photo assessment until those steps are verified.
