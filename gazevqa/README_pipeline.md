# Gazevqa Feature Pipeline

Minimal recipe for turning raw ELAN videos/SAM crops into train/test instances. All commands run from repo root in an environment with `torch`, `clip`, `decord`, `pillow`, `pandas`, `numpy`.

> **Data layout tip**  
> Keep ELAN files, raw MP4s, segmented clips, and all derived features outside the repo. Store your absolute paths in an `env.sh` so every script can reference the same locations:
>
> ```bash
> export ELAN_ROOT=/path/to/ELAN_eaf
> export RAW_VIDEO_ROOT=/path/to/raw_videos
> export VIDEO_CLIP_ROOT=/path/to/derived/video_clips
> export VIDEO_FEATURE_ROOT=/path/to/derived/video_features
> export IMAGE_FEATURE_ROOT=/path/to/derived/image_features
> export SAM_ROOT=/path/to/SAM/SAMv2
> export QALIST_ROOT=/path/to/QALIST_Data
> export QA_SPLIT_ROOT=/path/to/derived/qa_instances
> ```
>
> After `source env.sh`, the commands below pick up those placeholders automatically, so you never hardcode machine-specific paths.

## 0. Segment ELAN videos into per-step clips

```
python gazevqa/build_step_clips.py \
  --elan-root "${ELAN_ROOT}" \
  --dataset-root "${RAW_VIDEO_ROOT}" \
  --output-root "${VIDEO_CLIP_ROOT}" \
  --views firstview firstviewraw secondview thirdview \
  --assembly-filename assembly.mp4 \
  --disassembly-filename disassembly.mp4 \
  --codec h264
```

- The script walks every `ELAN/<object>/seg3/*.eaf`, reads the `Subject_Action` tier (plus `Instruction` as a fallback for `start disassembly` markers), and segments the timeline into *one clip per step* for each view that has the requested phase video.
- Output tree: `video_clips/<object>/<view>/stepXX.mp4` plus a manifest `video_clips/annotation.json`.
- Each manifest entry stores both absolute timestamps (`start_time`, `end_time`) and phase-relative offsets (`relative_start`, `relative_end`) together with `phase`, `step_index` (global order across assembly+disassembly), `phase_step_index`, and the list of view-specific paths that actually exist. Missing views are skipped automatically.

`phase` flips to `disassembly` once the ELAN tier reaches the "start disassembly" marker; the script subtracts that timestamp so disassembly clips always start near `0.0s`, which keeps step IDs aligned with the QA spreadsheet (step01...step40).

## 1. Video CLIP features

```
python -m gazevqa.pipelines.build_video_features \
  --raw-root "${VIDEO_CLIP_ROOT}" \
  --output-dir "${VIDEO_FEATURE_ROOT}" \
  --views firstview firstviewraw secondview thirdview \
  --device cuda:0
```

This command consumes the `annotation.json` from Step 0, embeds each `stepXX.mp4`, and saves `${VIDEO_FEATURE_ROOT}/Sxxx/stepXX.pt`. Every tensor concatenates the requested views (missing views become all-zeros) and keeps metadata like `frames_per_view` for downstream reshaping.

## 2. Image ROI features

```
python -m gazevqa.pipelines.build_image_features \
  --sam-root "${SAM_ROOT}" \
  --output-root "${IMAGE_FEATURE_ROOT}" \
  --device cuda:0
```
Produces `${IMAGE_FEATURE_ROOT}/Sxxx/<mode>/<step>.pt` with Faster R-CNN ROI embeddings and `image_index` tags (derived from `image_N.jpg`).

## 3. Merge with QALIST metadata

```
python -m gazevqa.pipelines.feature_combiner \
  --root-dir "${QALIST_ROOT}" \
  --clip-feature-dir "${VIDEO_FEATURE_ROOT}" \
  --sam-feature-dir "${IMAGE_FEATURE_ROOT}" \
  --clip-model ViT-B/32 \
  --device cuda:0
```
Writes `${QALIST_ROOT}/Sxxx/Sxxx.pt` where each row contains tokenized questions/answers plus the aligned video & image tensors and their label columns.
Video answers are written as plain numbers (`7`, `22`) the combiner normalizes these aliases and looks up `${VIDEO_FEATURE_ROOT}/Sxxx/stepXX.pt` generated in Step 1.

## 4. Expand & split instances

```
python -m gazevqa.pipelines.build_instance_splits \
  --input-root "${QALIST_ROOT}" \
  --output-dir "${QA_SPLIT_ROOT}" \
  --train-ratio 0.9 \
  --seed 42
```
Splits questions into train/test, then enumerates every `(video, text, image)` combination. Labels are `1` only when all required modalities pick the correct option; optional modalities (all-zero labels) default to `1`.

Output: `${QA_SPLIT_ROOT}/train.pt`, `${QA_SPLIT_ROOT}/test.pt`, each a list of ready-to-train instances with metadata (`object_id`, `global_step_id`, etc.).
