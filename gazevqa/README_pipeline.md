# Gazevqa Feature Pipeline

Minimal recipe for turning raw videos/SAM crops into train/test instances. All commands run from repo root in an environment with `torch`, `clip`, `decord`, `pillow`, `pandas`, `numpy`.

## 1. Video CLIP features

```
python -m gazevqa.pipelines.build_video_features \
  --raw-root SAM/SAMv2 \
  --output-dir video_features_pt \
  --views firstview firstviewraw secondview thirdview \
  --device cuda:0
```
Outputs per-step tensors `video_features_pt/Sxxx/*.pt`, each storing four view vectors (`first/second/third/firstraw`).

## 2. Image ROI features

```
python -m gazevqa.pipelines.build_image_features \
  --sam-root SAM/SAMv2 \
  --output-root image_features_pt \
  --device cuda:0
```
Produces `image_features_pt/Sxxx/<mode>/<step>.pt` with Faster R-CNN ROI embeddings and `image_index` tags (derived from `image_N.jpg`).

## 3. Merge with QALIST metadata

```
python -m gazevqa.pipelines.feature_combiner \
  --root-dir QALIST_Data \
  --clip-feature-dir video_features_pt \
  --sam-feature-dir image_features_pt \
  --clip-model ViT-B/32 \
  --device cuda:0
```
Writes `QALIST_Data/Sxxx/Sxxx.pt` where each row contains tokenized questions/answers plus the aligned video & image tensors and their label columns.

## 4. Expand & split instances

```
python -m gazevqa.pipelines.build_instance_splits \
  --input-root QALIST_Data \
  --output-dir qa_instances \
  --train-ratio 0.9 \
  --seed 42
```
Splits questions into train/test, then enumerates every `(video, text, image)` combination. Labels are `1` only when all required modalities pick the correct option; optional modalities (all-zero labels) default to `1`.

Output: `qa_instances/train.pt`, `qa_instances/test.pt`, each a list of ready-to-train instances with metadata (`object_id`, `global_step_id`, etc.).
