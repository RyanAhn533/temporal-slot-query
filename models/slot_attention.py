"""
Slot Attention + Bio-Aware Gating
==================================
Locatello et al. (2020) 기반 + Bio modulation 확장.

핵심:
  - K개 슬롯이 입력 토큰에 competitive attention (softmax over slots)
  - 각 슬롯이 하나의 "상황/이벤트"를 담당
  - Bio-Aware Gating: 생체신호로 슬롯 sensitivity 동적 조절
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-8


class SlotAttention(nn.Module):
    """
    Slot Attention with Bio-Aware Gating.

    Args:
        num_slots: 슬롯 수 (K). 이벤트 수보다 크게 설정 (여분은 background 흡수)
        d_model: 임베딩 차원
        num_iters: iterative refinement 횟수
        hidden_dim: FFN 히든 차원
        bio_gate: bio gating 활성화 여부
    """

    def __init__(self, num_slots: int = 8, d_model: int = 128,
                 num_iters: int = 3, hidden_dim: int = 256,
                 bio_gate: bool = True):
        super().__init__()
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.d_model = d_model
        self.bio_gate = bio_gate

        # Learnable slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, d_model) * 0.02)
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, num_slots, d_model))

        # Attention projections
        self.norm_slots = nn.LayerNorm(d_model)
        self.norm_inputs = nn.LayerNorm(d_model)

        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        # GRU for slot update
        self.gru = nn.GRUCell(d_model, d_model)

        # FFN
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

        self.scale = d_model ** -0.5

        # Bio-Aware Gating
        if bio_gate:
            self.bio_gate_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )

    def forward(self, inputs: torch.Tensor,
                bio_embedding: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            inputs: [B, N, d_model] — multi-modal tokens (temporal 포함)
            bio_embedding: [B, d_model] — bio encoder output (optional)

        Returns:
            slots: [B, K, d_model]
        """
        B = inputs.shape[0]

        # Slot initialization (reparameterization trick)
        sigma = self.slot_log_sigma.exp()
        slots = self.slot_mu + sigma * torch.randn(B, self.num_slots, self.d_model,
                                                     device=inputs.device)

        # Pre-compute keys and values (input은 iterative하게 안 바뀜)
        inputs_norm = self.norm_inputs(inputs)
        k = self.to_k(inputs_norm)  # [B, N, d]
        v = self.to_v(inputs_norm)  # [B, N, d]

        for it in range(self.num_iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))  # [B, K, d]

            # Dot-product attention: [B, K, N]
            attn_logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale

            # Competitive softmax over SLOTS (dim=1)
            # 각 입력 토큰이 하나의 슬롯에 주로 할당됨
            attn = F.softmax(attn_logits, dim=1)

            # Weighted normalization (sum over inputs)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + EPSILON)

            # Aggregate: [B, K, d]
            updates = torch.einsum("bkn,bnd->bkd", attn, v)

            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.d_model),
                slots_prev.reshape(-1, self.d_model),
            ).reshape(B, self.num_slots, self.d_model)

            # FFN with residual
            slots = slots + self.ffn(self.norm_ffn(slots))

        # Bio-Aware Gating
        if self.bio_gate and bio_embedding is not None:
            gate = self.bio_gate_proj(bio_embedding)  # [B, d]
            gate = gate.unsqueeze(1)  # [B, 1, d]
            slots = slots * gate  # broadcast: [B, K, d] * [B, 1, d]

        return slots

    def get_attention_maps(self, inputs: torch.Tensor,
                           bio_embedding: torch.Tensor = None) -> torch.Tensor:
        """시각화용: 마지막 iteration의 attention map 반환."""
        B = inputs.shape[0]

        sigma = self.slot_log_sigma.exp()
        slots = self.slot_mu + sigma * torch.randn(B, self.num_slots, self.d_model,
                                                     device=inputs.device)
        inputs_norm = self.norm_inputs(inputs)
        k = self.to_k(inputs_norm)
        v = self.to_v(inputs_norm)

        for it in range(self.num_iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))
            attn_logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale
            attn = F.softmax(attn_logits, dim=1)
            attn_normed = attn / (attn.sum(dim=-1, keepdim=True) + EPSILON)
            updates = torch.einsum("bkn,bnd->bkd", attn_normed, v)
            slots = self.gru(
                updates.reshape(-1, self.d_model),
                slots_prev.reshape(-1, self.d_model),
            ).reshape(B, self.num_slots, self.d_model)
            slots = slots + self.ffn(self.norm_ffn(slots))

        # 마지막 attention 반환
        return attn  # [B, K, N]
