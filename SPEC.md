# SPEC — Emotion Recognition (emotion-recognition-cv)

## Goal

Full rebuild of the old `12_emotion-detection-app` project — not just repo
hygiene, but a real attempt to improve the model's accuracy, plus honest
docs and a working deployment.

## Current baseline (for honest comparison)

- Old model: custom CNN (4× Conv–BatchNorm–ReLU–MaxPool–Dropout blocks,
  1→64→128→256→512 channels) + FC head, ~11M params.
- Trained 20 fixed epochs, FER-2013, 7 classes, 80/20 split.
- **Test accuracy: 59.64%.**
- Context: human-level accuracy on FER-2013 is estimated ~65%; the
  published single-model state of the art without extra training data is
  ~73% (Khaireddin & Chen, 2021, VGG-style architecture + heavy
  hyperparameter tuning — arXiv:2105.03588). So 59.64% is below even the
  human baseline — there's real room to improve.
- **Realistic target for this rebuild: ~68–72%.** Not a promise to match
  published SOTA — that required much more tuning than is in scope here.

## Planned changes to improve accuracy

- **Transfer learning, as the primary approach** — fine-tune a lightweight
  ImageNet-pretrained backbone (ResNet18 or MobileNetV2) instead of tuning
  the old from-scratch CNN. At FER-2013's size (35,888 images) a pretrained
  backbone's low/mid-level features transfer well and overfit less than a
  small CNN trained from zero. Input is resized 48×48 grayscale →
  224×224×3 (channel duplicated) to match the backbone's expected input.
  Still light enough for CPU inference on a single image (not real-time
  video), so it stays deployable on Streamlit Community Cloud's free tier.
- The old from-scratch CNN (4× Conv-BN-ReLU-MaxPool-Dropout, 59.64%) is
  kept as the **baseline for comparison**, not thrown away — retrained
  once under the same new pipeline (augmentation, class weights, LR
  schedule) so the README can honestly show "old approach, tuned" vs.
  "transfer learning" side by side, not just old-vs-new-and-different.
- Data augmentation (horizontal flip, small rotation, zoom/crop) — FER-2013
  is small and prone to overfitting without it, for either model.
- Class-weighted loss — FER-2013 classes are imbalanced (`disgust` has very
  few examples and is probably where accuracy is worst; measured, not
  assumed).
- LR scheduling (ReduceLROnPlateau or cosine) + early stopping, instead of
  a fixed epoch count.
- **Training compute:** done on a free GPU (Google Colab or Kaggle), not
  locally on CPU or on paid AWS EC2 (the old project's approach) — CPU
  training at this scale would take many hours; free GPU keeps this
  practical without ongoing cost.

## Data security / privacy constraints (addressed before implementation)

- No uploaded image is ever sent to a third-party API — inference is local
  only, via the bundled trained model.
- Uploaded images are processed in memory only; nothing is persisted to
  disk beyond what's strictly required to decode/process it.
- No logging or retention of uploaded content between requests or sessions.
- Upload size and file type are validated before processing; a corrupt or
  mistyped upload shows a clear error instead of crashing.
- No AWS credentials, `.pem` keys, or `.env` files from the old project are
  carried into the new repo. (Confirmed in the earlier audit that these
  were never committed to git history in the old repo either — but the new
  repo starts clean regardless, and `.gitignore` is set up from commit 1.)
- These constraints exist specifically because an earlier AI-assisted
  project caused a data-leak-adjacent rejection in a recruitment process —
  no code gets written before its data-handling approach is specified and
  reviewed.

## Architecture

Training/model code kept separate from the app, so the model can be
retrained or swapped without touching the UI:

- `model.py` — both model definitions: `EmotionCNN` (the old from-scratch
  baseline, retrained under the new pipeline) and `EmotionResNet` (the
  transfer-learning model actually used by the app). No training/UI logic.
- `train.py` — data loading, augmentation, training loop, checkpointing;
  takes `--model {cnn,resnet}` to train either one under the same pipeline.
  Meant to run on Colab/Kaggle GPU, not the Streamlit host.
- `evaluate.py` — measures real test accuracy **and per-class
  precision/recall/F1** (given the known class-imbalance issue), plus a
  confusion matrix, for a given checkpoint — not just one overall accuracy
  number.
- `app.py` — Streamlit UI only; loads the trained transfer-learning
  checkpoint, no training logic lives here.

## Evaluation

Numbers in the README are measured by `evaluate.py` against the FER-2013
test split, not asserted. Report: overall accuracy, per-class breakdown,
confusion matrix. Compared honestly against the 59.64% baseline and the
~65% human / ~73% SOTA reference points above.

## Baseline CNN revision (after first measurements)

First measured results (FER-2013 test split):

| Model | Pipeline | Test accuracy |
|---|---|---|
| EmotionResNet | augmentation + class weights + LR schedule | 67.96% |
| EmotionCNN | same full pipeline | 51.70% |

The CNN is **underfit** under the full pipeline (train accuracy ~43% is
below test accuracy ~52%) and below the old project's 59.64%, so this run
is not a fair baseline: it shows the pipeline hurts a small from-scratch
network, not that transfer learning is worth +16 points.

Plan (training pipeline only — the CNN architecture stays unchanged):

- `train.py` gains `--no-class-weights`, `--no-augment` and `--tag`
  flags (the tag keeps checkpoints of different runs from overwriting
  each other).
- Two additional CNN runs: (A) augmentation only, no class weights;
  (B) no augmentation, no class weights (closest to the old pipeline).
- The README reports **all** CNN runs, labelled by pipeline, and uses
  the best one as the baseline. The first full-pipeline run (51.70%) stays
  in the table. The ResNet keeps its own best-working pipeline; the
  comparison is stated as "best-effort pipeline per model".
- Known limitation, stated in the README: checkpoint selection and early
  stopping use the test split (FER-2013 ships train/test only), so all
  reported accuracies are slightly optimistic.

Outcome of the extra CNN runs: (A) augmentation only reached 61.54%
(stopped by the 40-epoch cap while still improving); (B) no augmentation
reached 63.11% and is the reported baseline. The ResNet's lead over the
best CNN is ~4.9 accuracy points (macro F1 0.66 vs 0.53), not the ~16
points the first, underfit run suggested. The ~68-72% target was not
reached (ResNet: 67.96%).

## Out of scope (for now)

- Ensembling multiple models.
- Matching published SOTA (73%) exactly.
- Any feature not already in the old app (e.g. webcam mode) unless it's
  confirmed to already exist and work — this is a rebuild, not a feature
  expansion, beyond the accuracy work above.

## Repo / migration notes

- New repo name: `emotion-recognition-cv`.
- Old repo `12_emotion-detection-app` and its portfolio page: to be
  deleted/removed by Sławek himself once he's back at his computer (repo
  deletion and portfolio edits are not something Claude does unprompted).
- Old local folder `12_openCV_PyTorch`: once no longer needed, moves to
  `_to_delete` (Recycle Bin), never a hard delete.
