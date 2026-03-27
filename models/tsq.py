"""
Bio-Aware Temporal Slot Query (TSQ)
====================================
전체 모델. Multi-modal input → Event + Driver State prediction.

Architecture:
  Multi-Modal Encoders → Temporal Buffer → Slot Attention (Bio-Gated)
  → Cross-Slot Self-Attention → Dual Heads (Event + State)

Paper target: IV 2027 / ITSC 2027
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from models.encoders import DetectionEncoder, MaskEncoder, BioEncoder, SceneEncoder
from models.slot_attention import SlotAttention


# ── 이벤트/상태 정의 ──
EVENT_NAMES = ["cutin", "low_visibility", "vru", "obstacle", "stable"]
STATE_NAMES = ["stress", "drowsy", "low_attention"]
NUM_EVENTS = len(EVENT_NAMES)
NUM_STATES = len(STATE_NAMES)


class TemporalSlotQuery(nn.Module):
    """
    Bio-Aware Temporal Slot Query.

    전체 forward:
      1. 각 인코더로 입력 토큰 생성
      2. Temporal position encoding 추가
      3. Slot Attention (bio-gated)
      4. Cross-Slot Self-Attention
      5. Dual head: Event (5) + State (3)

    Args:
        d_model: 임베딩 차원 (default 128)
        num_slots: 슬롯 수 (default 8, 이벤트 5 + background 3)
        temporal_len: 시계열 길이 (프레임, default 10 = 1초 @10Hz)
        max_det: 프레임당 최대 검출 수
        num_patches: mask encoder 패치 수
        num_classes: YOLO 클래스 수
        bio_dim: 생체 feature 차원
        scene_dim: 장면 feature 차원
        slot_iters: slot attention 반복 횟수
        sa_layers: self-attention 레이어 수
        bio_gate: bio gating 사용 여부
        dropout: dropout rate
    """

    def __init__(self,
                 d_model: int = 128,
                 num_slots: int = 8,
                 temporal_len: int = 10,
                 max_det: int = 20,
                 num_patches: int = 16,
                 num_classes: int = 80,
                 bio_dim: int = 14,
                 scene_dim: int = 5,
                 slot_iters: int = 3,
                 sa_layers: int = 2,
                 bio_gate: bool = True,
                 dropout: float = 0.1):
        super().__init__()

        self.d_model = d_model
        self.temporal_len = temporal_len
        self.max_det = max_det
        self.num_patches = num_patches

        # ── Encoders ──
        self.det_encoder = DetectionEncoder(num_classes, d_model)
        self.mask_encoder = MaskEncoder(num_patches, d_model)
        self.bio_encoder = BioEncoder(bio_dim, d_model)
        self.scene_encoder = SceneEncoder(scene_dim, d_model)

        # ── Modality Type Embedding ──
        # 0=detection, 1=mask, 2=bio, 3=scene
        self.modality_embed = nn.Embedding(4, d_model)

        # ── Temporal Position Encoding ──
        self.temporal_pe = nn.Parameter(
            torch.randn(1, temporal_len, 1, d_model) * 0.02)

        # ── Slot Attention ──
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            d_model=d_model,
            num_iters=slot_iters,
            hidden_dim=d_model * 2,
            bio_gate=bio_gate,
        )

        # ── Cross-Slot Self-Attention ──
        sa_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.cross_slot_sa = nn.TransformerEncoder(sa_layer, num_layers=sa_layers)

        # ── Dual Heads ──
        self.event_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, NUM_EVENTS),
        )

        self.state_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, NUM_STATES),
        )

        # ── Inference buffer ──
        self._buffer: deque = deque(maxlen=temporal_len)
        self._bio_buffer: deque = deque(maxlen=temporal_len)

        # Parameter count
        self._count_params()

    def _count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[TSQ] Parameters: {total:,} total, {trainable:,} trainable "
              f"({trainable/1e6:.1f}M)")

    def _encode_frame(self, det_tokens: torch.Tensor,
                      mask_tokens: torch.Tensor,
                      bio_token: torch.Tensor,
                      scene_token: torch.Tensor) -> torch.Tensor:
        """
        한 프레임의 모든 토큰을 concat + modality embedding.

        Returns: [B, N_total, d_model]
        """
        B = det_tokens.shape[0]

        # Modality type IDs
        n_det = det_tokens.shape[1]
        n_mask = mask_tokens.shape[1]

        det_type = torch.zeros(B, n_det, dtype=torch.long, device=det_tokens.device)
        mask_type = torch.ones(B, n_mask, dtype=torch.long, device=det_tokens.device)
        bio_type = torch.full((B, 1), 2, dtype=torch.long, device=det_tokens.device)
        scene_type = torch.full((B, 1), 3, dtype=torch.long, device=det_tokens.device)

        type_ids = torch.cat([det_type, mask_type, bio_type, scene_type], dim=1)
        type_emb = self.modality_embed(type_ids)  # [B, N_total, d]

        tokens = torch.cat([det_tokens, mask_tokens, bio_token, scene_token], dim=1)
        return tokens + type_emb

    def forward(self, det_tokens_seq: torch.Tensor,
                mask_tokens_seq: torch.Tensor,
                bio_tokens_seq: torch.Tensor,
                scene_tokens_seq: torch.Tensor,
                bio_embed_seq: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        학습용 forward.

        Args:
            det_tokens_seq: [B, T, max_det, d] — 이미 인코딩된 토큰
            mask_tokens_seq: [B, T, P, d]
            bio_tokens_seq: [B, T, 1, d]
            scene_tokens_seq: [B, T, 1, d]
            bio_embed_seq: [B, T, d] — bio gating용 (마지막 프레임)

        Returns:
            {"event_probs": [B, 5], "state_probs": [B, 3], "slots": [B, K, d]}
        """
        B, T = det_tokens_seq.shape[:2]

        # 프레임별 토큰 concat + modality embed
        all_tokens = []
        for t in range(T):
            frame_tokens = self._encode_frame(
                det_tokens_seq[:, t],
                mask_tokens_seq[:, t],
                bio_tokens_seq[:, t],
                scene_tokens_seq[:, t],
            )  # [B, N_frame, d]

            # Temporal PE
            pe = self.temporal_pe[:, t, :, :]  # [1, 1, d]
            frame_tokens = frame_tokens + pe

            all_tokens.append(frame_tokens)

        # Flatten temporal: [B, T*N_frame, d]
        all_tokens = torch.cat(all_tokens, dim=1)

        # Bio embedding for gating (마지막 프레임)
        bio_emb = None
        if bio_embed_seq is not None:
            bio_emb = bio_embed_seq[:, -1, :]  # [B, d]

        # Slot Attention
        slots = self.slot_attention(all_tokens, bio_emb)  # [B, K, d]

        # Cross-Slot Self-Attention
        slots = self.cross_slot_sa(slots)  # [B, K, d]

        # Pool → Heads
        pooled = slots.mean(dim=1)  # [B, d]

        event_logits = self.event_head(pooled)  # [B, 5]
        state_logits = self.state_head(pooled)  # [B, 3]

        return {
            "event_probs": torch.sigmoid(event_logits),
            "state_probs": torch.sigmoid(state_logits),
            "event_logits": event_logits,
            "state_logits": state_logits,
            "slots": slots,
        }

    @torch.no_grad()
    def predict_frame(self,
                      detections: List[Dict],
                      img_h: int, img_w: int,
                      drivable_mask=None,
                      lane_mask=None,
                      bio_features: Optional[Dict] = None,
                      scene_gray=None) -> Dict:
        """
        추론용. 매 프레임(10Hz) 호출.

        Returns:
            {
                "events": {"cutin": 0.92, ...},
                "states": {"stress": 0.3, ...},
                "dominant_event": "cutin",
                "risk_level": 2,
            }
        """
        self.eval()
        device = next(self.parameters()).device

        # 1. Encode detections
        det_data = DetectionEncoder.prepare_detections(
            detections, img_h, img_w, self.max_det)
        det_tokens = self.det_encoder(
            det_data["bboxes"].to(device),
            det_data["confs"].to(device),
            det_data["cls_ids"].to(device),
        )  # [1, max_det, d]

        # 2. Encode masks
        mask_data = MaskEncoder.prepare_masks(drivable_mask, lane_mask)
        mask_tokens = self.mask_encoder(
            mask_data["drivable"].to(device),
            mask_data["lane"].to(device),
        )  # [1, P, d]

        # 3. Encode bio
        bio_tensor = BioEncoder.prepare_bio(
            bio_features.get("ppg") if bio_features else None,
            bio_features.get("eda") if bio_features else None,
            bio_features.get("temp") if bio_features else None,
        ).to(device)
        bio_token = self.bio_encoder(bio_tensor)  # [1, 1, d]
        bio_embed = bio_token.squeeze(1)  # [1, d]

        # 4. Encode scene
        if scene_gray is not None:
            scene_tensor = SceneEncoder.extract_scene(scene_gray).to(device)
        else:
            scene_tensor = torch.zeros(1, 5, device=device)
        scene_token = self.scene_encoder(scene_tensor)  # [1, 1, d]

        # 5. Frame tokens
        frame_tokens = self._encode_frame(det_tokens, mask_tokens, bio_token, scene_token)

        # 6. Buffer
        self._buffer.append(frame_tokens.squeeze(0))  # [N_frame, d]
        self._bio_buffer.append(bio_embed.squeeze(0))  # [d]

        # 7. Pad if needed
        if len(self._buffer) < self.temporal_len:
            pad_n = self.temporal_len - len(self._buffer)
            padded = [torch.zeros_like(self._buffer[0])] * pad_n + list(self._buffer)
            bio_padded = [torch.zeros_like(self._bio_buffer[0])] * pad_n + list(self._bio_buffer)
        else:
            padded = list(self._buffer)
            bio_padded = list(self._bio_buffer)

        # 8. Stack: [1, T*N_frame, d]
        all_tokens = torch.cat(
            [p.unsqueeze(0) + self.temporal_pe[:, i, :, :]
             for i, p in enumerate(padded)],
            dim=1,
        )

        # 9. Slot Attention + Self-Attention
        bio_emb = bio_padded[-1].unsqueeze(0)  # [1, d]
        slots = self.slot_attention(all_tokens, bio_emb)
        slots = self.cross_slot_sa(slots)

        # 10. Heads
        pooled = slots.mean(dim=1)
        event_probs = torch.sigmoid(self.event_head(pooled)).squeeze(0).cpu().numpy()
        state_probs = torch.sigmoid(self.state_head(pooled)).squeeze(0).cpu().numpy()

        # 11. 결과 포맷
        events = {name: float(event_probs[i]) for i, name in enumerate(EVENT_NAMES)}
        states = {name: float(state_probs[i]) for i, name in enumerate(STATE_NAMES)}

        dominant = max(events, key=events.get)
        max_event_prob = events[dominant]
        max_state_prob = max(states.values())

        risk = 0
        if max_event_prob > 0.8 or max_state_prob > 0.8:
            risk = 3
        elif max_event_prob > 0.5 or max_state_prob > 0.5:
            risk = 2
        elif max_event_prob > 0.3 or max_state_prob > 0.3:
            risk = 1

        return {
            "events": events,
            "states": states,
            "dominant_event": dominant,
            "dominant_event_prob": float(max_event_prob),
            "risk_level": risk,
        }

    def reset(self):
        """새 세션 시작 시."""
        self._buffer.clear()
        self._bio_buffer.clear()


def create_tsq(pretrained: Optional[str] = None, **kwargs) -> TemporalSlotQuery:
    """팩토리."""
    model = TemporalSlotQuery(**kwargs)
    if pretrained:
        state = torch.load(pretrained, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"[TSQ] Loaded pretrained: {pretrained}")
    return model
