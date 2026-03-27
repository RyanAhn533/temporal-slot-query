"""
Multi-Modal Encoders for Temporal Slot Query
=============================================
각 입력 스트림을 d_model 차원으로 인코딩.

Streams:
  1. Detection Encoder: YOLO bbox+cls+conf → [N, d]
  2. Mask Encoder: drivable+lane mask → [P, d] (patch-based)
  3. Bio Encoder: PPG/EDA/Temp features → [1, d]
  4. Scene Encoder: brightness/contrast/glare → [1, d]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class DetectionEncoder(nn.Module):
    """YOLO 검출 → d_model embedding."""

    def __init__(self, num_classes: int = 80, d_model: int = 128):
        super().__init__()
        # bbox(4) + conf(1) + cls_embed(d_model//4)
        self.cls_embed = nn.Embedding(num_classes, d_model // 4)
        feat_dim = 4 + 1 + d_model // 4
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.num_classes = num_classes

    def forward(self, bboxes: torch.Tensor, confs: torch.Tensor,
                cls_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bboxes: [B, N, 4] normalized (0~1)
            confs: [B, N, 1]
            cls_ids: [B, N] long

        Returns: [B, N, d_model]
        """
        cls_emb = self.cls_embed(cls_ids)  # [B, N, d//4]
        feat = torch.cat([bboxes, confs, cls_emb], dim=-1)
        return self.proj(feat)

    @staticmethod
    def prepare_detections(det_list: List[Dict], img_h: int, img_w: int,
                           max_det: int = 20) -> Dict[str, torch.Tensor]:
        """Python dict → 텐서 변환."""
        bboxes = []
        confs = []
        cls_ids = []

        for d in det_list[:max_det]:
            bbox = d.get("bbox", [0, 0, 0, 0])
            bboxes.append([bbox[0]/img_w, bbox[1]/img_h, bbox[2]/img_w, bbox[3]/img_h])
            confs.append([d.get("conf", 0.5)])
            cls_ids.append(d.get("cls", 0))

        # 패딩
        while len(bboxes) < max_det:
            bboxes.append([0, 0, 0, 0])
            confs.append([0])
            cls_ids.append(0)

        return {
            "bboxes": torch.tensor(bboxes, dtype=torch.float32).unsqueeze(0),
            "confs": torch.tensor(confs, dtype=torch.float32).unsqueeze(0),
            "cls_ids": torch.tensor(cls_ids, dtype=torch.long).unsqueeze(0),
            "n_valid": min(len(det_list), max_det),
        }


class MaskEncoder(nn.Module):
    """
    Drivable + Lane mask → patch embeddings.
    마스크를 P개 패치로 나누고 각 패치의 통계를 인코딩.
    """

    def __init__(self, num_patches: int = 16, d_model: int = 128):
        super().__init__()
        self.num_patches = num_patches
        # 패치별 feature: [drivable_ratio, lane_ratio, position_x, position_y]
        self.proj = nn.Sequential(
            nn.Linear(4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, drivable: torch.Tensor,
                lane: torch.Tensor) -> torch.Tensor:
        """
        Args:
            drivable: [B, H, W] float (0~1)
            lane: [B, H, W] float (0~1)

        Returns: [B, P, d_model]
        """
        B, H, W = drivable.shape
        gh = int(np.sqrt(self.num_patches))
        gw = self.num_patches // gh
        ph, pw = H // gh, W // gw

        patches = []
        for i in range(gh):
            for j in range(gw):
                da_patch = drivable[:, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                ll_patch = lane[:, i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                da_ratio = da_patch.float().mean(dim=(-2, -1))  # [B]
                ll_ratio = ll_patch.float().mean(dim=(-2, -1))  # [B]
                pos_y = torch.full((B,), (i + 0.5) / gh, device=drivable.device)
                pos_x = torch.full((B,), (j + 0.5) / gw, device=drivable.device)
                feat = torch.stack([da_ratio, ll_ratio, pos_x, pos_y], dim=-1)  # [B, 4]
                patches.append(feat)

        patches = torch.stack(patches, dim=1)  # [B, P, 4]
        return self.proj(patches)

    @staticmethod
    def prepare_masks(drivable_mask: Optional[np.ndarray],
                      lane_mask: Optional[np.ndarray],
                      target_h: int = 64, target_w: int = 64) -> Dict[str, torch.Tensor]:
        """numpy mask → 텐서."""
        import cv2
        if drivable_mask is not None:
            da = cv2.resize(drivable_mask, (target_w, target_h),
                           interpolation=cv2.INTER_NEAREST)
            da = torch.from_numpy(da.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            da = torch.ones(1, target_h, target_w)

        if lane_mask is not None:
            ll = cv2.resize(lane_mask, (target_w, target_h),
                           interpolation=cv2.INTER_NEAREST)
            ll = torch.from_numpy(ll.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            ll = torch.zeros(1, target_h, target_w)

        return {"drivable": da, "lane": ll}


class BioEncoder(nn.Module):
    """
    생체신호 features → [1, d_model].

    Input features (from IncrementalBioProcessor):
      PPG: [hr_mean, hr_std, hrv_rmssd, hrv_lf_hf] = 4d
      EDA: [scr_count, scr_mean_amp, scl_mean, scl_slope, scr_rise] = 5d
      Temp: [skin_temp, temp_slope] = 2d
      Quality: [ppg_quality, eda_quality, temp_quality] = 3d
      Total: 14d
    """

    def __init__(self, bio_dim: int = 14, d_model: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(bio_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.bio_dim = bio_dim

    def forward(self, bio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bio_features: [B, bio_dim]

        Returns: [B, 1, d_model]
        """
        return self.proj(bio_features).unsqueeze(1)

    @staticmethod
    def prepare_bio(ppg_features: Optional[Dict] = None,
                    eda_features: Optional[Dict] = None,
                    temp_features: Optional[Dict] = None,
                    bio_dim: int = 14) -> torch.Tensor:
        """Dict → 텐서. 없는 값은 0으로."""
        feat = [0.0] * bio_dim
        idx = 0
        if ppg_features:
            for k in ["hr_mean", "hr_std", "hrv_rmssd", "hrv_lf_hf"]:
                feat[idx] = float(ppg_features.get(k, 0))
                idx += 1
        else:
            idx += 4

        if eda_features:
            for k in ["scr_count", "scr_mean_amp", "scl_mean", "scl_slope", "scr_rise"]:
                feat[idx] = float(eda_features.get(k, 0))
                idx += 1
        else:
            idx += 5

        if temp_features:
            for k in ["skin_temp", "temp_slope"]:
                feat[idx] = float(temp_features.get(k, 0))
                idx += 1
        else:
            idx += 2

        # quality (3d) — default 1.0
        for i in range(3):
            feat[idx + i] = 1.0 if (ppg_features or eda_features or temp_features) else 0.0

        return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)


class SceneEncoder(nn.Module):
    """장면 컨텍스트 → [1, d_model]."""

    def __init__(self, scene_dim: int = 5, d_model: int = 128):
        super().__init__()
        # brightness, contrast, glare_ratio, mean_flow, edge_density
        self.proj = nn.Sequential(
            nn.Linear(scene_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, scene_features: torch.Tensor) -> torch.Tensor:
        """[B, scene_dim] → [B, 1, d_model]"""
        return self.proj(scene_features).unsqueeze(1)

    @staticmethod
    def extract_scene(frame_gray: np.ndarray) -> torch.Tensor:
        """그레이스케일 프레임에서 장면 특징 추출."""
        import cv2
        brightness = float(np.mean(frame_gray)) / 255.0
        contrast = float(np.std(frame_gray)) / 128.0
        glare = float(np.sum(frame_gray > 240)) / max(frame_gray.size, 1)
        edges = cv2.Canny(frame_gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / max(edges.size, 1)
        # mean_flow placeholder (optical flow 없이)
        mean_flow = 0.0

        return torch.tensor(
            [brightness, contrast, glare, mean_flow, edge_density],
            dtype=torch.float32).unsqueeze(0)
