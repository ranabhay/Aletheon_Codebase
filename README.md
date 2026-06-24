# Aletheon_Codebase
Aletheon_Safe
# PPE Safety-Violation Detector

A computer-vision model that detects Personal Protective Equipment (PPE) on
workers and flags **safety violations** — e.g. a worker with no helmet, no vest,
no goggles. Built with [Ultralytics YOLO](https://docs.ultralytics.com/) (YOLO26).

> Status: working proof-of-concept. Detects PPE reliably on clear images;
> violation detection on hard/cluttered scenes is still limited (see
> [Limitations](#limitations)).

---

## What's in this repo

| File | What it is |
|------|------------|
| `detect_violations.py` | Run the model on an image / folder / video / webcam and flag violations |
| `best.pt` | The trained model weights (see [Get the model](#1-get-the-model)) |
| `requirements.txt` | Python dependencies |
| `merge_datasets.py` | Merges multiple PPE datasets into one unified class scheme |
| `train_ppe_colab.ipynb` | Colab notebook — train the base model |
| `retrain_merged_colab.ipynb` | Colab notebook — train on merged datasets (the better model) |
| `examples/` | Sample input images and their detection results |

---

## Quick start (test it in 5 minutes)

You only need Python 3.9+ . A GPU is **not** required to run detection.

### 1. Get the model
Make sure `best.pt` is in the project folder. If it isn't in the repo (model
files can exceed GitHub's size limit), download it from the release/link noted
in the repo and place it next to `detect_violations.py`.

### 2. Install
```bash
git clone https://github.com/arpitraj18/ppe-detector.git
cd ppe-detector

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```
The first install pulls in PyTorch, so it may take a few minutes.

### 3. Run it
```bash
# Single image
python detect_violations.py --weights best.pt --source examples/test.jpg --save

# A folder of images
python detect_violations.py --weights best.pt --source ./my_images/ --save

# A video file
python detect_violations.py --weights best.pt --source video.mp4 --save

# Live webcam
python detect_violations.py --weights best.pt --source 0
```

### 4. See the results
- The terminal prints `[VIOLATION]` or `[ok]` per image, plus a summary.
- With `--save`, annotated images (boxes drawn on) are written to
  `runs/detect/predict/` (each run makes a new `predict`, `predict2`, ... folder).
  Open the newest one to see what the model detected.

---

## Options

| Flag | Default | Effect |
|------|---------|--------|
| `--weights` | (required) | Path to the model, e.g. `best.pt` |
| `--source` | (required) | Image, folder, video path, or `0` for webcam |
| `--conf` | `0.35` | Confidence threshold. Lower = detect more but noisier |
| `--associate` | off | Also flag a detected `Person` who has no overlapping helmet/vest |
| `--save` | off | Save annotated images/video to `runs/detect/predict*/` |

---

## How a violation is decided

The model knows 12 classes:

```
helmet, gloves, vest, boots, goggles, none, Person,
no_helmet, no_goggle, no_gloves, no_boots, no_vest
```

A violation is flagged two ways:

1. **Direct** — the model detects a negative class (`no_helmet`, `no_goggle`,
   `no_gloves`, `no_boots`, `no_vest`) or `none` (a person wearing no PPE at all).
   This is reliable.
2. **Person-level** (`--associate`) — for each detected `Person`, the script
   checks whether a helmet/vest box overlaps them. A person with none is flagged
   as likely missing that item. This catches people the model only tagged as
   `Person`, but it can over-flag when the model misses a PPE item that's
   actually present (a precision/recall tradeoff — see Limitations).

---

## Results

Validation metrics for the current model (merged dataset, YOLO26s):

- Overall: **mAP50 ~ 0.50**, mAP50-95 ~ 0.24
- Positive PPE classes (helmet, vest, gloves, boots, goggles, Person):
  **~0.70-0.78 mAP50** — strong, usable detection.
- Violation classes (`no_helmet`, `no_vest`, etc.): **weak**, because the
  training data contains very few negative examples.

See `examples/` for before/after images on real construction scenes.

---

## Limitations

Worth being upfront about — this is a PoC:

- **Misses people in cluttered / shadowed / distant scenes.** If the model can't
  detect a person, it can't flag their violation. This is a *data* limitation
  (needs more crowded-scene training images), not a model-size one.
- **Violation classes are data-starved.** The model detects *present* PPE well
  but is unreliable at detecting *missing* PPE, because negative examples are
  scarce in the training data.
- **`--associate` can produce false positives.** When the model fails to draw a
  vest/helmet box that's actually there, the person-level check wrongly flags a
  violation. Use it when you want maximum recall and can tolerate false alarms.

---

## Train it yourself

Both notebooks run on free [Google Colab](https://colab.research.google.com)
(set Runtime -> GPU). No local GPU needed.

- **`train_ppe_colab.ipynb`** — trains the base model on the auto-downloading
  Construction-PPE dataset.
- **`retrain_merged_colab.ipynb`** — merges Construction-PPE with a Roboflow PPE
  dataset (and optionally SH17) into one 12-class scheme, then trains a larger
  model. `merge_datasets.py` handles the class remapping by name so datasets
  with different class numbering don't corrupt training.

After training, download `best.pt` from the notebook and use it with
`detect_violations.py` as above.

### Adding more data
Point the merge at more sources to improve the model. The biggest gaps to fill
are **violation examples** (datasets with `no-helmet`/`no-vest` labels) and
**crowded/occluded scenes** (e.g. the SH17 dataset). Edit the `CLASS_MAP` in
`merge_datasets.py` if a new dataset uses class names it doesn't recognize.

---

## License & credits

Built on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
Datasets: Ultralytics Construction-PPE, Roboflow Universe "Construction Site
Safety". Check each dataset's own license before commercial use.