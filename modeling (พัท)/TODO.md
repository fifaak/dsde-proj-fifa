# TODO — modeling (พัท)

**Criterion:** Component 2, Data Science (the models behind the analysis); also "technical quality".
**Target date:** Tue 1 Dec 2026.

## Status

- [x] 450 hand labels (`labels/labels.csv`)
- [x] 7 classifiers compared on the same 90-label holdout; fine-tuned CLIP ViT-B/16 chosen
      (heavy-truck precision 0.65 / recall 0.69)
- [x] GPU training script for the RTX 3090 (`train_gpu.py`, data prep `prep_gpu_data.py`)

## Remaining

- [ ] **Publish the weights** (`truck_clipvitb.pt`, 343 MB) as a GitHub Release or on Drive, and note the link in
      the root README
- [ ] **Model card** (`MODEL.md`): training data, labels, holdout metrics, known confusions (vans, box pickups,
      buses), limits (6 vs 10 wheels, night)
- [ ] **More reliable metrics**: 5-fold CV for the final model; the current holdout has only 29 heavy trucks
- [ ] Accuracy split by day vs night (after 18:30) for the limitations slide
- [ ] Optional: label 1,000+ more crops (VLM-assisted, then human-checked) and retrain to raise precision
