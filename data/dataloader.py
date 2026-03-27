"""
TSQ Dataset & DataLoader
=========================
K-MER 자체 데이터 + 외부 데이터셋을 TSQ 입력 포맷으로 변환.

Strategy:
  Phase 1: K-MER pseudo-labels (rule-based detector 결과를 라벨로 사용)
  Phase 2: 외부 데이터셋 통합
  Phase 3: 연세대 시뮬레이터 실데이터

Input format (per sample):
  - detections: List[Dict] per frame, T frames
  - drivable_mask: [T, H, W] or None
  - lane_mask: [T, H, W] or None
  - bio_features: [T, 14] or None
  - scene_features: [T, 5] or None
  - event_labels: [5] binary
  - state_labels: [3] binary
"""

import os
import json
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class TSQSample:
    """단일 학습 샘플 (T 프레임 시퀀스)."""

    def __init__(self, detections_seq, bio_seq=None, scene_seq=None,
                 event_labels=None, state_labels=None,
                 img_h=480, img_w=640):
        self.detections_seq = detections_seq  # List[List[Dict]], T frames
        self.bio_seq = bio_seq                # [T, 14] or None
        self.scene_seq = scene_seq            # [T, 5] or None
        self.event_labels = event_labels      # [5] float
        self.state_labels = state_labels      # [3] float
        self.img_h = img_h
        self.img_w = img_w


class PseudoLabelGenerator:
    """
    Rule-based detector 결과를 pseudo-label로 변환.
    학습 데이터가 없을 때 bootstrap용.
    """

    @staticmethod
    def generate_event_labels(detections_seq: List[List[Dict]],
                               scene_features: Optional[np.ndarray] = None) -> np.ndarray:
        """T 프레임 검출 결과 → 5 이벤트 라벨."""
        labels = np.zeros(5, dtype=np.float32)

        all_dets = [d for frame in detections_seq for d in frame]
        vru_count = sum(1 for d in all_dets if d.get("cls") in {0, 1})
        vehicle_count = sum(1 for d in all_dets if d.get("cls") in {2, 3, 5, 7})

        # cutin: 차량이 있고 bbox가 커지는 패턴 (간이)
        if vehicle_count > 0 and len(detections_seq) >= 2:
            first_vehicles = [d for d in detections_seq[0] if d.get("cls") in {2, 3, 5, 7}]
            last_vehicles = [d for d in detections_seq[-1] if d.get("cls") in {2, 3, 5, 7}]
            if first_vehicles and last_vehicles:
                first_max_h = max((d["bbox"][3] - d["bbox"][1]) for d in first_vehicles)
                last_max_h = max((d["bbox"][3] - d["bbox"][1]) for d in last_vehicles)
                if last_max_h > first_max_h * 1.3:  # 30% 이상 커짐
                    labels[0] = 1.0  # cutin

        # low_visibility: scene features에서
        if scene_features is not None:
            avg_brightness = np.mean(scene_features[:, 0])  # normalized 0~1
            if avg_brightness < 0.2:
                labels[1] = 1.0  # low_visibility

        # vru: 보행자/자전거 존재
        if vru_count > len(detections_seq) * 0.3:  # 30% 이상 프레임에 VRU
            labels[2] = 1.0  # vru

        # obstacle: 큰 unknown object (drivable mask 없이는 간이 판정)
        # TODO: drivable mask 기반 판정 추가

        # stable: 다른 이벤트 없으면
        if labels[:4].sum() == 0:
            labels[4] = 1.0  # stable

        return labels

    @staticmethod
    def generate_state_labels(bio_features: Optional[np.ndarray] = None) -> np.ndarray:
        """Bio features → 3 상태 라벨."""
        labels = np.zeros(3, dtype=np.float32)
        if bio_features is None:
            return labels

        # bio_features: [T, 14]
        avg = np.mean(bio_features, axis=0)

        # stress: EDA 높고 HR 높으면
        hr_mean = avg[0]  # hr_mean
        eda_scl = avg[6]  # scl_mean
        if hr_mean > 85 and eda_scl > 3.0:
            labels[0] = 1.0  # stress

        # drowsy: HR variability 낮고 HR 낮으면
        hrv = avg[2]  # hrv_rmssd
        if hrv < 20 and hr_mean < 65:
            labels[1] = 1.0  # drowsy

        # low_attention: (bio만으로는 어려움, placeholder)
        # TODO: 실제로는 gaze/PERCLOS 필요

        return labels


class KMERDataset(Dataset):
    """
    K-MER 시뮬레이터 데이터 → TSQ 학습 포맷.

    데이터 구조:
      data/C039/20260327_1015/
        video_main.mp4  → YOLO 검출 → detections
        ppg.csv         → bio features
        gsr.csv         → bio features
        temp.csv        → bio features

    학습 시: YOLO를 오프라인으로 미리 돌려서 detection cache 생성.
    """

    def __init__(self, data_dir: str, temporal_len: int = 10,
                 max_det: int = 20, cache_dir: str = "cache",
                 split: str = "train", split_ratio: Tuple = (0.7, 0.15, 0.15)):
        self.data_dir = Path(data_dir)
        self.temporal_len = temporal_len
        self.max_det = max_det
        self.cache_dir = Path(cache_dir)
        self.pseudo_gen = PseudoLabelGenerator()

        # 세션 폴더 수집
        sessions = sorted(self.data_dir.glob("C[0-9][0-9][0-9]"))
        if not sessions:
            self.samples = []
            return

        # Train/Val/Test 분리 (세션 단위)
        n = len(sessions)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])

        if split == "train":
            sessions = sessions[:n_train]
        elif split == "val":
            sessions = sessions[n_train:n_train + n_val]
        else:
            sessions = sessions[n_train + n_val:]

        # 샘플 수집 (temporal_len 프레임씩 슬라이딩)
        self.samples = []
        for session in sessions:
            minute_dirs = sorted(session.glob("20*"))
            # temporal_len개씩 슬라이딩 윈도우
            for i in range(0, len(minute_dirs) - temporal_len + 1, temporal_len // 2):
                window = minute_dirs[i:i + temporal_len]
                self.samples.append({
                    "session": session.name,
                    "window": [str(d) for d in window],
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        detections_seq = []
        bio_seq = []
        scene_seq = []

        for minute_dir in sample["window"]:
            minute_dir = Path(minute_dir)

            # Detection cache 로드 (없으면 빈 리스트)
            det_cache = minute_dir / "detections.json"
            if det_cache.exists():
                with open(det_cache) as f:
                    dets = json.load(f)
            else:
                dets = []
            detections_seq.append(dets)

            # Bio features
            bio_feat = self._load_bio(minute_dir)
            bio_seq.append(bio_feat)

            # Scene features (cache에서)
            scene_cache = minute_dir / "scene.json"
            if scene_cache.exists():
                with open(scene_cache) as f:
                    scene = json.load(f)
                scene_seq.append(scene)
            else:
                scene_seq.append([0.5, 0.5, 0.0, 0.0, 0.1])  # default

        bio_arr = np.array(bio_seq, dtype=np.float32)
        scene_arr = np.array(scene_seq, dtype=np.float32)

        # Pseudo labels
        event_labels = self.pseudo_gen.generate_event_labels(detections_seq, scene_arr)
        state_labels = self.pseudo_gen.generate_state_labels(bio_arr)

        # 텐서 변환
        return {
            "detections": detections_seq,  # collate에서 처리
            "bio": torch.from_numpy(bio_arr),  # [T, 14]
            "scene": torch.from_numpy(scene_arr),  # [T, 5]
            "events": torch.from_numpy(event_labels),  # [5]
            "states": torch.from_numpy(state_labels),  # [3]
        }

    def _load_bio(self, minute_dir: Path) -> list:
        """1분 폴더에서 bio features 추출."""
        feat = [0.0] * 14
        # PPG
        ppg_path = minute_dir / "ppg.csv"
        if ppg_path.exists():
            try:
                import csv
                with open(ppg_path) as f:
                    reader = csv.reader(f)
                    next(reader)  # header
                    vals = [float(row[1]) for row in reader if len(row) >= 2]
                if vals:
                    feat[0] = np.mean(vals)  # hr_mean (실제로는 PPG raw)
                    feat[1] = np.std(vals)   # hr_std
            except Exception:
                pass

        # EDA/GSR
        gsr_path = minute_dir / "gsr.csv"
        if gsr_path.exists():
            try:
                import csv
                with open(gsr_path) as f:
                    reader = csv.reader(f)
                    next(reader)
                    vals = [float(row[1]) for row in reader if len(row) >= 2]
                if vals:
                    feat[6] = np.mean(vals)  # scl_mean
                    feat[7] = (vals[-1] - vals[0]) / max(len(vals), 1)  # scl_slope
            except Exception:
                pass

        # Temp
        temp_path = minute_dir / "temp.csv"
        if temp_path.exists():
            try:
                import csv
                with open(temp_path) as f:
                    reader = csv.reader(f)
                    next(reader)
                    vals = [float(row[1]) for row in reader if len(row) >= 2]
                if vals:
                    feat[11] = np.mean(vals)  # skin_temp
            except Exception:
                pass

        return feat


class SyntheticDataset(Dataset):
    """
    합성 데이터셋 — 학습 데이터 없을 때 모델 구조 검증용.
    랜덤 detection + bio + scene + label 생성.
    """

    def __init__(self, num_samples: int = 1000, temporal_len: int = 10,
                 max_det: int = 20, num_classes: int = 80):
        self.num_samples = num_samples
        self.temporal_len = temporal_len
        self.max_det = max_det
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        T = self.temporal_len

        # Random detections
        detections_seq = []
        for t in range(T):
            n_det = random.randint(0, self.max_det)
            dets = []
            for _ in range(n_det):
                x1, y1 = random.random() * 0.7, random.random() * 0.7
                x2, y2 = x1 + random.random() * 0.3, y1 + random.random() * 0.3
                dets.append({
                    "cls": random.randint(0, self.num_classes - 1),
                    "conf": random.random(),
                    "bbox": [x1 * 640, y1 * 480, x2 * 640, y2 * 480],
                })
            detections_seq.append(dets)

        # Random bio
        bio = np.random.randn(T, 14).astype(np.float32) * 0.5 + 0.5

        # Random scene
        scene = np.random.rand(T, 5).astype(np.float32)

        # Random labels (correlated with input for some signal)
        has_vru = any(any(d["cls"] in {0, 1} for d in frame) for frame in detections_seq)
        low_bright = scene[:, 0].mean() < 0.3

        events = np.zeros(5, dtype=np.float32)
        if has_vru:
            events[2] = 1.0  # vru
        if low_bright:
            events[1] = 1.0  # low_vis
        if events[:4].sum() == 0:
            events[4] = 1.0  # stable

        states = np.zeros(3, dtype=np.float32)
        if bio[:, 0].mean() > 0.7:
            states[0] = 1.0  # stress

        return {
            "detections": detections_seq,
            "bio": torch.from_numpy(bio),
            "scene": torch.from_numpy(scene),
            "events": torch.from_numpy(events),
            "states": torch.from_numpy(states),
        }


def tsq_collate_fn(batch):
    """커스텀 collate — detections는 가변 길이라 리스트로."""
    return {
        "detections": [b["detections"] for b in batch],
        "bio": torch.stack([b["bio"] for b in batch]),
        "scene": torch.stack([b["scene"] for b in batch]),
        "events": torch.stack([b["events"] for b in batch]),
        "states": torch.stack([b["states"] for b in batch]),
    }


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=4):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, collate_fn=tsq_collate_fn,
        pin_memory=True, drop_last=True,
    )
