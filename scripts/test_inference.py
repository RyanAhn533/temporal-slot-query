"""
TSQ Inference Test — 학습 없이 모델 구조 검증
==============================================
녹화 영상 → YOLO → TSQ forward → 이벤트 확률 출력.
랜덤 weight이므로 정확도는 의미 없고, 파이프라인 동작 + 속도 측정.

Usage:
    docker exec gracious_edison python3 /workspace/temporal-slot-query/scripts/test_inference.py
"""
import sys
import os
import time
import json
import numpy as np

# 프로젝트 루트 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import cv2

print("=" * 60)
print("  TSQ Inference Pipeline Test")
print("=" * 60)

# ── 1. 모델 로드 ──
print("\n[1/4] Loading TSQ model (random weights)...")
from models.tsq import TemporalSlotQuery, EVENT_NAMES, STATE_NAMES

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TemporalSlotQuery(
    d_model=128, num_slots=8, temporal_len=10,
    max_det=20, num_patches=16, bio_gate=True,
).to(device).eval()

print(f"  Device: {device}")
print(f"  GPU: {torch.cuda.memory_allocated()//1024//1024}MB" if device == "cuda" else "")

# ── 2. YOLO 로드 ──
print("\n[2/4] Loading YOLO...")
from ultralytics import YOLO

yolo_path = "/workspace/Real-Time-Driver-State-Emotion-Monitoring-System/cutin_detection/yolov8n-seg.pt"
if not os.path.exists(yolo_path):
    yolo_path = "yolov8n.pt"
yolo = YOLO(yolo_path)
yolo(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)  # warmup
print("  YOLO loaded")

# ── 3. 영상 처리 ──
print("\n[3/4] Processing video...")
video_path = "/workspace/Real-Time-Driver-State-Emotion-Monitoring-System/recordings/rec_20260316_171105.mp4"
if not os.path.exists(video_path):
    print(f"  Video not found: {video_path}")
    print("  Using synthetic frames instead")
    video_path = None

if video_path:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(fps / 10))
    print(f"  Video: {video_path}")
    print(f"  FPS: {fps}, Frames: {total}, Skip: {skip}")
else:
    cap = None
    skip = 1

tsq_times = []
yolo_times = []
results_log = []

model.reset()

n_frames = 0
max_frames = 100  # 10초분

for i in range(max_frames * skip if cap else max_frames):
    if cap:
        ret, frame = cap.read()
        if not ret:
            break
        if i % skip != 0:
            continue
    else:
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    n_frames += 1
    h, w = frame.shape[:2]

    # YOLO
    t0 = time.time()
    results = yolo(frame, verbose=False)
    yolo_ms = (time.time() - t0) * 1000
    yolo_times.append(yolo_ms)

    # YOLO → detections list
    detections = []
    if results and len(results) > 0:
        r = results[0]
        if r.boxes is not None:
            for j in range(len(r.boxes)):
                detections.append({
                    "cls": int(r.boxes.cls[j]),
                    "conf": float(r.boxes.conf[j]),
                    "bbox": r.boxes.xyxy[j].cpu().numpy().tolist(),
                })

    # 더미 bio (학습 안 했으니 랜덤)
    bio_features = {
        "ppg": {"hr_mean": 75 + np.random.randn() * 5,
                "hr_std": 3 + np.random.rand(),
                "hrv_rmssd": 30 + np.random.randn() * 5,
                "hrv_lf_hf": 1.5 + np.random.rand()},
        "eda": {"scr_count": int(np.random.randint(0, 3)),
                "scr_mean_amp": np.random.rand() * 0.5,
                "scl_mean": 2 + np.random.rand(),
                "scl_slope": np.random.randn() * 0.1,
                "scr_rise": np.random.rand() * 0.3},
        "temp": {"skin_temp": 33 + np.random.rand(),
                 "temp_slope": np.random.randn() * 0.01},
    }

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # TSQ predict
    t0 = time.time()
    result = model.predict_frame(
        detections=detections,
        img_h=h, img_w=w,
        drivable_mask=None,
        lane_mask=None,
        bio_features=bio_features,
        scene_gray=gray,
    )
    tsq_ms = (time.time() - t0) * 1000
    tsq_times.append(tsq_ms)

    results_log.append(result)

    if n_frames % 20 == 0:
        print(f"  Frame {n_frames}: YOLO={yolo_ms:.0f}ms TSQ={tsq_ms:.0f}ms "
              f"det={len(detections)} → {result['dominant_event']}({result['dominant_event_prob']:.2f})")

if cap:
    cap.release()

# ── 4. 결과 ──
print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)

print(f"\nFrames: {n_frames}")
print(f"\nYOLO:")
print(f"  Mean: {np.mean(yolo_times):.1f}ms")
print(f"  P95:  {np.percentile(yolo_times, 95):.1f}ms")

print(f"\nTSQ (Bio-Aware Temporal Slot Query):")
print(f"  Mean: {np.mean(tsq_times):.1f}ms")
print(f"  P95:  {np.percentile(tsq_times, 95):.1f}ms")

print(f"\nTotal per-frame (YOLO + TSQ):")
total_mean = np.mean(yolo_times) + np.mean(tsq_times)
print(f"  Mean: {total_mean:.1f}ms")
print(f"  10Hz budget (100ms): {'PASS' if total_mean < 100 else 'FAIL'}")

print(f"\nGPU: {torch.cuda.memory_allocated()//1024//1024}MB" if device == "cuda" else "")

# 마지막 5프레임 결과
print(f"\nLast 5 predictions (random weights, not trained):")
for r in results_log[-5:]:
    evts = " ".join(f"{k}={v:.2f}" for k, v in r["events"].items())
    sts = " ".join(f"{k}={v:.2f}" for k, v in r["states"].items())
    print(f"  Events: {evts}")
    print(f"  States: {sts}")
    print(f"  Dominant: {r['dominant_event']} ({r['dominant_event_prob']:.2f}) risk={r['risk_level']}")
    print()

print("=" * 60)
print("  Pipeline: OK. Model needs training for meaningful predictions.")
print("=" * 60)
