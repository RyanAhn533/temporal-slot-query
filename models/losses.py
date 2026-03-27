"""
TSQ Loss Functions
===================
Multi-label event + state classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TSQLoss(nn.Module):
    """
    Combined loss for Temporal Slot Query.

    L = w_event * BCE(event_logits, event_labels)
      + w_state * BCE(state_logits, state_labels)
      + w_slot_diversity * slot_diversity_loss
    """

    def __init__(self, w_event: float = 1.0, w_state: float = 0.5,
                 w_diversity: float = 0.1):
        super().__init__()
        self.w_event = w_event
        self.w_state = w_state
        self.w_diversity = w_diversity
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, outputs: dict, targets: dict) -> dict:
        """
        Args:
            outputs: from TSQ.forward()
                - event_logits: [B, 5]
                - state_logits: [B, 3]
                - slots: [B, K, d]
            targets:
                - events: [B, 5] float (0 or 1)
                - states: [B, 3] float (0 or 1)

        Returns:
            {"loss": total, "event_loss": ..., "state_loss": ..., "diversity_loss": ...}
        """
        event_loss = self.bce(outputs["event_logits"], targets["events"])
        state_loss = self.bce(outputs["state_logits"], targets["states"])
        diversity_loss = self._slot_diversity(outputs["slots"])

        total = (self.w_event * event_loss
                 + self.w_state * state_loss
                 + self.w_diversity * diversity_loss)

        return {
            "loss": total,
            "event_loss": event_loss.item(),
            "state_loss": state_loss.item(),
            "diversity_loss": diversity_loss.item(),
        }

    def _slot_diversity(self, slots: torch.Tensor) -> torch.Tensor:
        """슬롯 간 다양성 촉진. 슬롯들이 서로 다른 것을 담도록."""
        # Cosine similarity between all slot pairs
        slots_norm = F.normalize(slots, dim=-1)  # [B, K, d]
        sim = torch.bmm(slots_norm, slots_norm.transpose(1, 2))  # [B, K, K]

        # 대각선 제외 (자기 자신)
        K = sim.shape[1]
        mask = 1 - torch.eye(K, device=sim.device).unsqueeze(0)
        sim = sim * mask

        # 높은 유사도 = 페널티
        return sim.abs().mean()
