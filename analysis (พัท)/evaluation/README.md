# Bangkok CCTV evaluation

This directory extends the Data Science section. Source images and weights stay outside Git. The notebook includes aggregate results. Reproduce the per-crop predictions locally with the commands below.

## Results

| Evaluation | Result |
| --- | --- |
| Baseline, five camera-grouped folds | 40.4% pooled four-class accuracy, 41.8% heavy precision and 41.8% heavy recall across 450 hand labels from 93 cameras |
| Frozen CLIP ViT-B/16 + linear classifier, five camera-grouped folds | 61.6% four-class accuracy (277/450), 58.4% heavy precision (94/161) and 66.7% heavy recall (94/141). |
| CLIP linear classifier, day and night | 60.9% (254/417) by day and 69.7% (23/33) by night. Heavy recall is 92/137 by day and 2/4 by night. |
| Baseline, day and night | 39.6% accuracy on 417 daytime crops and 51.5% on 33 nighttime crops. Heavy recall was 42.3% by day (58/137) and 25.0% by night (1/4). The class mix and small nighttime sample limit comparison. |
| Supplied CLIP checkpoint | Ran inference on all 450 crops. All 450 classifications agreed with the repository's saved pipeline output. Accuracy on these previously seen labels was 100% in both periods. This measures fit to the training sample. |
| Historical precipitation, 25–31 August 2026 | 22.31 heavy sightings per 100 frames in wet hours (29,344 frames) and 21.09 in dry hours (31,438 frames). Across 2,193 matched camera/hour/weekday strata, the equal-weight wet-minus-dry difference was +0.087 sightings per 100 frames. |
| Baseline classification by rain | Heavy recall was 44.6% in dry hours (33/74) and 38.8% in wet hours (26/67). This exploratory result describes labeled candidate crops. |

Day runs from 06:00 to 18:29. ERA5 precipitation ≥0.1 mm in the enclosing hour defines wet conditions. The adjusted comparison is descriptive for one week. Hourly precipitation comes from one regional grid point, and sighting rates depend on both passing traffic and the detection pipeline. The crop-label results cover the classifier stage and cannot quantify full-frame detector misses.

## Reproduce the evaluation

Run from the repository root with Python 3.9+ and install the project dependencies plus `pillow`, `scikit-learn`, `matplotlib`, `jupyter` and `timm`.

```sh
eval_dir='analysis (พัท)/evaluation'
python "$eval_dir/evaluate.py" \
  --images /path/to/cctv_images \
  --checkpoint /path/to/truck_clipvitb.pt \
  --out "$eval_dir/crop_predictions.csv"
python "$eval_dir/clip_linear_cv.py" \
  --images /path/to/cctv_images \
  --weights /path/to/datacomp.safetensors \
  --predictions "$eval_dir/crop_predictions.csv" \
  --out "$eval_dir/crop_predictions.csv"
cd "$eval_dir"
jupyter nbconvert --to notebook --execute --inplace evaluation.ipynb
```

The image root contains one directory per camera ID. The matching checkpoint has SHA-256 `2cfd3e3f8a0d565ca3fed1d4b65189782fe5931d7b5e4a43026d995d3dc096b0`. The notebook uses repository aggregates and an archived [Open-Meteo ERA5 response](https://archive-api.open-meteo.com/v1/archive?latitude=13.75&longitude=100.5&start_date=2026-08-25&end_date=2026-08-31&hourly=precipitation&models=era5&timezone=Asia%2FBangkok). Timestamps are local to Bangkok and hourly precipitation denotes the preceding hour. The archived response contains the actual ERA5 grid coordinate.

## CLIP cross-validation

The evaluated CLIP configuration freezes the [public DataComp XL ViT-B/16 image encoder](https://huggingface.co/timm/vit_base_patch16_clip_224.datacompxl), at revision `71b64a4a74500385eaf8bc227889735cdeeb2bd1`. The SHA-256 of its `model.safetensors` is `1efb2905ccc4fec2735139693928574f7173e2419530e1aaf66735c6cbeffbc1`. The script normalizes the 512-dimensional image embeddings and fits four-class logistic regression with C=1 independently in each camera-held-out fold. A fresh classifier produces each fold's predictions. The original public weights are used for all crops and remain fixed.

## Cross-validation of the production training recipe

`train_clip_folds.py` runs the existing CUDA training recipe five times with the recorded camera-held-out splits. It requires the original dataset version of the [Thai vehicle supplement](https://universe.roboflow.com/mtltransportation/thailand-vehicles-2024-01-16) processed by the project's training recipe into `thai_x.npy` and `thai_y.npy`, the confirmed learning rate and batch size, and access to the original public pretrained model. Start each fold from the public pretrained model. The executed five-fold results cover the gradient/color baseline and frozen CLIP linear classifier. The full end-to-end CUDA recipe remains unrun. CLIP checkpoint results on the 450 labels are descriptive and potentially optimistic because the project recipe fits the published checkpoint on those labels.

```sh
python 'analysis (พัท)/evaluation/train_clip_folds.py' \
  --images /path/to/cctv_images \
  --thai-tensors /path/to/prepared_thai_tensors \
  --splits 'analysis (พัท)/evaluation/crop_predictions.csv' \
  --out /path/to/clip_oof.csv \
  --lr "$CONFIRMED_LR" --batch-size "$CONFIRMED_BATCH_SIZE"
```

## Road hotspots

Run `python 'analysis (พัท)/evaluation/rank_roads.py'` from the repository root. The rank pools sightings and frames during the weekday six-wheel ban intervals for in-zone cameras. The camera-to-road table records the source description from the CCTV metadata, with explicit Thai camera-side descriptions where available. Conflicting or ambiguous descriptions are excluded. The resulting 26 roads cover 54 of the 94 in-zone cameras with qualifying frames. [Thai slide summary](สรุปสำหรับสไลด์.md)
