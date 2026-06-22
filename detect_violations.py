"""
detect_violations.py — Run the trained model and flag safety violations.

A "violation" is decided two ways (both reported):

1. DIRECT:  the model detects an explicit negative class
            (no_helmet, no_goggle, no_gloves, no_boots).

2. PERSON-LEVEL (optional, --associate):
            for each detected Person box, check whether a helmet / vest box
            overlaps it. A person with no overlapping helmet is flagged as a
            likely "missing helmet" violation. This catches cases the model
            labelled only as "Person" without an explicit no_* box.

Works on a single image, a folder of images, a video file, or a webcam.

Examples:
    python detect_violations.py --weights runs/detect/ppe_base/weights/best.pt --source test.jpg
    python detect_violations.py --weights best.pt --source ./images/ --associate
    python detect_violations.py --weights best.pt --source video.mp4 --save
    python detect_violations.py --weights best.pt --source 0          # webcam
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

# Class names in the Construction-PPE dataset (index -> name).
NAMES = {
    0: "helmet", 1: "gloves", 2: "vest", 3: "boots", 4: "goggles",
    5: "none", 6: "Person", 7: "no_helmet", 8: "no_goggle",
    9: "no_gloves", 10: "no_boots",
}
VIOLATION_CLASSES = {"no_helmet", "no_goggle", "no_gloves", "no_boots"}
# Required PPE we'll check per-person when --associate is on.
REQUIRED_PPE = {"helmet", "vest"}


def iou_contains(person_box, ppe_box, min_overlap=0.3):
    """Return True if ppe_box meaningfully overlaps person_box.

    Uses intersection-over-the-PPE-area (not standard IoU) because a helmet is
    small relative to a person, so we ask: is most of the PPE inside the person?
    """
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = ppe_box
    ix1, iy1 = max(px1, bx1), max(py1, by1)
    ix2, iy2 = min(px2, bx2), min(py2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ppe_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return (inter / ppe_area) >= min_overlap


def analyze(result, associate=False):
    """Turn one Ultralytics Result into a list of violation strings."""
    violations = []
    persons, ppe_by_class = [], {name: [] for name in REQUIRED_PPE}

    boxes = result.boxes
    if boxes is None:
        return violations

    for cls_id, xyxy, conf in zip(
        boxes.cls.tolist(), boxes.xyxy.tolist(), boxes.conf.tolist()
    ):
        name = NAMES.get(int(cls_id), str(int(cls_id)))

        # 1. Direct negative-class detections.
        if name in VIOLATION_CLASSES:
            violations.append(f"{name} (conf {conf:.2f})")

        # Collect boxes for optional person-level association.
        if name == "Person":
            persons.append(xyxy)
        elif name in ppe_by_class:
            ppe_by_class[name].append(xyxy)

    # 2. Person-level association.
    if associate:
        for i, person in enumerate(persons, start=1):
            missing = [
                ppe for ppe in REQUIRED_PPE
                if not any(iou_contains(person, b) for b in ppe_by_class[ppe])
            ]
            for ppe in missing:
                violations.append(f"Person #{i} missing {ppe}")

    return violations


def parse_args():
    p = argparse.ArgumentParser(description="Detect PPE safety violations.")
    p.add_argument("--weights", required=True,
                   help="Path to trained weights, e.g. runs/detect/ppe_base/weights/best.pt")
    p.add_argument("--source", required=True,
                   help="Image, folder, video path, or webcam index (e.g. 0).")
    p.add_argument("--conf", type=float, default=0.35,
                   help="Confidence threshold.")
    p.add_argument("--associate", action="store_true",
                   help="Also flag persons with no overlapping helmet/vest.")
    p.add_argument("--save", action="store_true",
                   help="Save annotated images/video to runs/detect/predict*/.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    # stream=True yields one Result per frame — memory-safe for video/webcam.
    results = model.predict(
        source=args.source,
        conf=args.conf,
        device=args.device,
        save=args.save,
        stream=True,
        verbose=False,
    )

    total_frames = 0
    flagged_frames = 0
    for idx, result in enumerate(results):
        total_frames += 1
        violations = analyze(result, associate=args.associate)
        src = Path(result.path).name if result.path else f"frame_{idx}"

        if violations:
            flagged_frames += 1
            print(f"\n[VIOLATION] {src}")
            for v in violations:
                print(f"    - {v}")
        else:
            print(f"[ok]        {src} — no violations")

    print("\n=== Summary ===")
    print(f"Frames/images processed : {total_frames}")
    print(f"Flagged with violations : {flagged_frames}")
    if args.save:
        print("Annotated output saved under runs/detect/predict*/")


if __name__ == "__main__":
    main()
