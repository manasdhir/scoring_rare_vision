import json
import os
import streamlit as st
from collections import defaultdict
import base64

ALLOWED_LABELS = {
    "mouth",
    "esophagus",
    "stomach",
    "small intestine",
    "colon",
    "z-line",
    "pylorus",
    "ileocecal valve",
    "active bleeding",
    "angiectasia",
    "blood",
    "erosion",
    "erythema",
    "hematin",
    "lymphangioectasis",
    "polyp",
    "ulcer",
}


def tiou(a, b):
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]) + 1)
    union = (a["end"] - a["start"] + 1) + (b["end"] - b["start"] + 1) - inter
    return inter / union if union > 0 else 0.0


def extract_by_video_label(data):
    out = defaultdict(lambda: defaultdict(list))
    for v in data["videos"]:
        vid = v["video_id"]
        for e in v["events"]:
            for lbl in e["label"]:
                out[vid][lbl].append({"start": e["start"], "end": e["end"]})
    return out


def sanity_check(gt, pred):
    gt_ids = {v["video_id"] for v in gt["videos"]}
    pr_ids = {v["video_id"] for v in pred["videos"]}

    if gt_ids != pr_ids:
        missing = gt_ids - pr_ids
        extra = pr_ids - gt_ids
        msg = "Video ID mismatch."
        if missing:
            msg += f" Missing in prediction: {missing}."
        if extra:
            msg += f" Extra in prediction: {extra}."
        return False, msg

    for v in pred["videos"]:
        for e in v["events"]:
            for l in e["label"]:
                if l not in ALLOWED_LABELS:
                    return False, f"Invalid label '{l}' in video '{v['video_id']}'."

    return True, "All checks passed"


def average_precision(gt_segs, pr_segs, thr):
    matched = set()
    tp = []
    for p in pr_segs:
        hit = False
        for i, g in enumerate(gt_segs):
            if i in matched:
                continue
            if tiou(p, g) >= thr:
                matched.add(i)
                hit = True
                break
        tp.append(1 if hit else 0)

    if not gt_segs:
        return 0.0 if pr_segs else 1.0

    cum_tp = 0
    prev_r = 0.0
    ap = 0.0

    for i, v in enumerate(tp):
        cum_tp += v
        r = cum_tp / len(gt_segs)
        p = cum_tp / (i + 1)
        ap += p * (r - prev_r)
        prev_r = r

    return ap


def compute_map(gt, pr, thr):
    gt_ev = extract_by_video_label(gt)
    pr_ev = extract_by_video_label(pr)
    video_maps = []

    for vid in gt_ev:
        aps = []
        for lbl in ALLOWED_LABELS:
            aps.append(
                average_precision(
                    gt_ev[vid].get(lbl, []),
                    pr_ev[vid].get(lbl, []),
                    thr,
                )
            )
        video_maps.append(sum(aps) / len(aps))

    return sum(video_maps) / len(video_maps)


st.title("ICPR 2026 RARE VISION TEMPORAL mAP EVALUATOR")

# Expandable description for JSON generation
with st.expander("ℹ️ How to generate the required prediction JSON"):
    st.markdown("""
    **Using the scripts provided on GitHub**, you can automatically generate the required JSON file from your model's one-hot encoded CSV output.
    
    **Process Overview:**
    1. **Frame-Level Predictions:** Your model initially outputs a CSV where each row corresponds to a frame, and columns represent the one-hot encoded labels.
    2. **Temporal Grouping:** The provided GitHub script processes this CSV frame-by-frame. When it detects continuous frames where a specific label is active (e.g., `1`), it groups them into a single temporal "event".
    3. **JSON Structuring:** It calculates the `start` and `end` times/frames for these contiguous blocks and formats the final output into the required hierarchical JSON structure (`videos` -> `events` -> `start`, `end`, `label`) necessary for this mAP evaluator.
    """)

gt_b64 = os.environ.get("GROUND_TRUTH_JSON_BASE64")
gt = json.loads(base64.b64decode(gt_b64).decode())

pred_file = st.file_uploader("Upload prediction JSON", type=["json"])

if pred_file:
    pr = json.load(pred_file)
    passed, message = sanity_check(gt, pr)

    if not passed:
        st.error(message)
    else:
        st.success(message)
        st.metric("mAP @ 0.5", round(compute_map(gt, pr, 0.5), 4))
        st.metric("mAP @ 0.95", round(compute_map(gt, pr, 0.95), 4))
