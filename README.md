# 😊 Emotion Recognition

Face detection + facial emotion classification, running entirely locally —
no image you upload is ever sent to a third-party API or stored on disk.

**Live demo:** _to be added after deployment_
**Repository:** https://github.com/slastrzelec/emotion-recognition-cv

![tests](https://github.com/slastrzelec/emotion-recognition-cv/actions/workflows/tests.yml/badge.svg)

## What it does

Upload a photo (or use the bundled sample). The app detects faces with an
OpenCV Haar Cascade, then classifies each detected face into one of 7
emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) using a
fine-tuned CNN, and shows the per-emotion probability breakdown.

## How it works

- **Face detection:** OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`).
- **Emotion classification:** ResNet18, pretrained on ImageNet and
  fine-tuned on [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
  (35,888 grayscale 48×48 face crops, 7 classes).
- Everything runs in memory; nothing uploaded is written to disk or sent
  anywhere outside this app.

## Why transfer learning instead of a from-scratch CNN

This project replaces an earlier version that trained a small custom CNN
from scratch and reached 59.64% test accuracy — below the ~65% human-level
baseline commonly cited for FER-2013. At FER-2013's size (~36k images), a
pretrained backbone's features transfer well and overfit less than a small
CNN trained from zero, so this rebuild uses that as the primary approach.
The old from-scratch CNN was retrained under the same new pipeline
(augmentation, class-weighted loss, LR scheduling) and kept as an explicit
baseline for comparison below — not deleted, since the honest comparison
is part of the point.

## Measured accuracy

_Training runs on a free GPU (Colab/Kaggle) — see "Training" below. These
numbers come directly from `evaluate.py` against the FER-2013 test split
and will be filled in here once that run is complete; they are not
asserted ahead of time._

| Model | Overall accuracy | Notes |
|---|---|---|
| From-scratch CNN (old baseline, retrained under new pipeline) | _pending_ | |
| ResNet18 (transfer learning, this app) | _pending_ | |

For context: human-level accuracy on FER-2013 is estimated at ~65%; the
published single-model state of the art without extra training data is
~73% (Khaireddin & Chen, 2021 — arXiv:2105.03588). This project's target
is ~68–72%, not a claim to match published SOTA.

Per-class precision/recall/F1 and the confusion matrix (FER-2013's
classes are imbalanced — `disgust` has very few examples) are also
produced by `evaluate.py` and will be included here.

## Training

```bash
pip install -r requirements.txt -r requirements-train.txt
python train.py --model resnet --epochs 40   # or --model cnn, or --model both
```

Meant to run on a free GPU notebook (Google Colab or Kaggle), not on the
Streamlit deployment host — CPU training at this scale takes many hours.
On Colab:

```python
!git clone https://github.com/slastrzelec/emotion-recognition-cv.git
%cd emotion-recognition-cv
!pip install -q -r requirements.txt -r requirements-train.txt
!python train.py --model resnet --epochs 40
```

`train.py` downloads FER-2013 via `kagglehub` on first run (needs a
Kaggle API token — see [kagglehub's docs](https://github.com/Kagglehub/kagglehub)
for setup; never hardcode credentials in this repo).

## Evaluating a checkpoint

```bash
python evaluate.py --checkpoint checkpoints/resnet_best.pt --model resnet --data-dir <path-to-fer2013>/test
```

## Data security & privacy

- All inference is local — no uploaded image is ever sent to a
  third-party API.
- Uploaded images are processed in memory only; nothing is persisted to
  disk beyond what's strictly required to decode it.
- No logging or retention of uploaded content between sessions.
- Upload size and file type are validated before processing.

See `SPEC.md` for the full specification this project was built against,
written and reviewed before any code was.

## Limitations, stated plainly

- Face detection is a Haar Cascade — it's fast and dependency-free, but
  misses faces at extreme angles, poor lighting, or partial occlusion
  more often than a modern deep-learning face detector would.
- Trained and evaluated on FER-2013, a posed/curated dataset — accuracy
  on casual, in-the-wild photos will likely be lower than the numbers
  reported above.
- `disgust` has very few examples in FER-2013; even with class-weighted
  loss, it's expected to be the weakest-performing class (see the
  per-class breakdown once filled in above).

## Tech stack

PyTorch · torchvision (ResNet18) · OpenCV · Streamlit · scikit-learn (evaluation metrics)

## License

MIT
