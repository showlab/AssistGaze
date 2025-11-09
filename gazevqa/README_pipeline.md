# Gazevqa Feature Pipeline

Minimal recipe for turning ELAN-labelled raw videos + SAM crops into train/test instances. Pipeline order:
1. Use `gazevqa/video_crop.py` to cut single-step clips (e.g., `step01.mp4`) for every view.
2. Run the preprocessing commands below to build features and QA instances.

All commands run from repo root in an environment with `torch`, `clip`, `decord`, `pillow`, `pandas`, `numpy`.

## 0. Cut Raw Videos into Step Clips

Raw data layout per object:

```
raw_videos/
  S031/
    firstview/assembly.mp4
    firstview/disassembly.mp4
    firstviewraw/assembly.mp4
    ...
ELAN/
  S031/seg3/*.eaf
```

Run the cutter (requires ffmpeg with h264 support):

```
python gazevqa/video_crop.py \
  --elan-root /path/to/ELAN \
  --dataset-root /path/to/raw_videos \
  --output-root /path/to/step_clips
```

This writes `step01.mp4`, `step02.mp4`, … into `<output>/<label>/<view>/`. Assembly steps count up first; disassembly continues numbering (so Excel’s `Video_Answer` columns can reference step IDs directly).

## 1. Video CLIP features

```
python -m gazevqa.pipelines.build_video_features \
  --raw-root /path/to/step_clips \
  --output-dir video_features_pt \
  --views firstview firstviewraw secondview thirdview \
  --device cuda:0
```
Outputs per-step tensors `video_features_pt/Sxxx/stepXX.pt`, each storing the available view tensors (missing views stay `None`).

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
  --root-dir QALIST2106 \
  --clip-feature-dir video_features_pt \
  --sam-feature-dir image_features_pt \
  --clip-model ViT-B/32 \
  --device cuda:0
```
Writes `QALIST2106/Sxxx/Sxxx.pt` where each row contains tokenized questions/answers plus the aligned video & image tensors and their label columns.

## 4. Expand & split instances

```
python -m gazevqa.pipelines.build_instance_splits \
  --input-root QALIST2106 \
  --output-dir qa_instances \
  --train-ratio 0.9 \
  --seed 42
```
Splits questions into train/test, then enumerates every `(video, text, image)` combination. Labels are `1` only when all required modalities pick the correct option; optional modalities (all-zero labels) default to `1`.

Output: `qa_instances/train.pt`, `qa_instances/test.pt`, each a list of ready-to-train instances with metadata (`object_id`, `global_step_id`, etc.).
