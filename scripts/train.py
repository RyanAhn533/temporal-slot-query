"""
TSQ Training Script
====================
Bio-Aware Temporal Slot Query 학습.

Usage:
    # 합성 데이터로 구조 검증
    python scripts/train.py --synthetic --epochs 10

    # K-MER 데이터 학습
    python scripts/train.py --data-dir data/kmer --epochs 100

    # 전체 옵션
    python scripts/train.py --data-dir data/kmer --epochs 100 --lr 1e-4 --batch 32 --device cuda
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.tsq import TemporalSlotQuery, EVENT_NAMES, STATE_NAMES, create_tsq
from models.losses import TSQLoss
from models.encoders import DetectionEncoder, MaskEncoder, BioEncoder, SceneEncoder
from data.dataloader import SyntheticDataset, KMERDataset, create_dataloader


def encode_batch(batch, model, device, max_det=20):
    """배치 데이터를 인코딩된 토큰으로 변환."""
    B = len(batch["detections"])
    T = batch["bio"].shape[1]

    det_tokens_list = []
    for b in range(B):
        frame_tokens = []
        for t in range(T):
            dets = batch["detections"][b][t]
            data = DetectionEncoder.prepare_detections(dets, 480, 640, max_det)
            tokens = model.det_encoder(
                data["bboxes"].to(device),
                data["confs"].to(device),
                data["cls_ids"].to(device),
            )  # [1, max_det, d]
            frame_tokens.append(tokens)
        det_tokens_list.append(torch.cat(frame_tokens, dim=0))  # [T, max_det, d]
    det_tokens = torch.stack(det_tokens_list)  # [B, T, max_det, d]

    bio = batch["bio"].to(device)  # [B, T, 14]
    scene = batch["scene"].to(device)  # [B, T, 5]

    bio_tokens = []
    scene_tokens = []
    for t in range(T):
        bt = model.bio_encoder(bio[:, t, :])  # [B, 1, d]
        st = model.scene_encoder(scene[:, t, :])  # [B, 1, d]
        bio_tokens.append(bt)
        scene_tokens.append(st)

    bio_tokens = torch.stack(bio_tokens, dim=1)  # [B, T, 1, d]
    scene_tokens = torch.stack(scene_tokens, dim=1)  # [B, T, 1, d]

    # Mask tokens (placeholder — 실제로는 YOLO P v2 필요)
    dummy_da = torch.ones(B, 64, 64, device=device)
    dummy_ll = torch.zeros(B, 64, 64, device=device)
    mask_token_single = model.mask_encoder(dummy_da, dummy_ll)  # [B, P, d]
    mask_tokens = mask_token_single.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, P, d]

    # Bio embedding for gating
    bio_embed = bio_tokens.squeeze(2)  # [B, T, d]

    return det_tokens, mask_tokens, bio_tokens, scene_tokens, bio_embed


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_event_loss = 0
    total_state_loss = 0
    n_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()

        det_tokens, mask_tokens, bio_tokens, scene_tokens, bio_embed = \
            encode_batch(batch, model, device)

        outputs = model(det_tokens, mask_tokens, bio_tokens, scene_tokens, bio_embed)

        targets = {
            "events": batch["events"].to(device),
            "states": batch["states"].to(device),
        }

        losses = criterion(outputs, targets)
        losses["loss"].backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += losses["loss"].item()
        total_event_loss += losses["event_loss"]
        total_state_loss += losses["state_loss"]
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "event_loss": total_event_loss / max(n_batches, 1),
        "state_loss": total_state_loss / max(n_batches, 1),
    }


@torch.no_grad()
def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_event_preds = []
    all_event_labels = []
    all_state_preds = []
    all_state_labels = []
    n_batches = 0

    for batch in dataloader:
        det_tokens, mask_tokens, bio_tokens, scene_tokens, bio_embed = \
            encode_batch(batch, model, device)

        outputs = model(det_tokens, mask_tokens, bio_tokens, scene_tokens, bio_embed)

        targets = {
            "events": batch["events"].to(device),
            "states": batch["states"].to(device),
        }

        losses = criterion(outputs, targets)
        total_loss += losses["loss"].item()

        all_event_preds.append(outputs["event_probs"].cpu().numpy())
        all_event_labels.append(targets["events"].cpu().numpy())
        all_state_preds.append(outputs["state_probs"].cpu().numpy())
        all_state_labels.append(targets["states"].cpu().numpy())
        n_batches += 1

    # F1 계산
    event_preds = np.concatenate(all_event_preds)
    event_labels = np.concatenate(all_event_labels)
    state_preds = np.concatenate(all_state_preds)
    state_labels = np.concatenate(all_state_labels)

    event_f1 = compute_f1(event_preds, event_labels)
    state_f1 = compute_f1(state_preds, state_labels)

    return {
        "loss": total_loss / max(n_batches, 1),
        "event_f1": event_f1,
        "state_f1": state_f1,
    }


def compute_f1(preds, labels, threshold=0.5):
    """Multi-label F1."""
    preds_binary = (preds > threshold).astype(float)
    tp = (preds_binary * labels).sum()
    fp = (preds_binary * (1 - labels)).sum()
    fn = ((1 - preds_binary) * labels).sum()
    precision = tp / max(tp + fp, 1e-8)
    recall = tp / max(tp + fn, 1e-8)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return float(f1)


def main():
    parser = argparse.ArgumentParser(description="Train TSQ")
    parser.add_argument("--data-dir", default="data/kmer")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-slots", type=int, default=8)
    parser.add_argument("--temporal-len", type=int, default=10)
    parser.add_argument("--bio-gate", action="store_true", default=True)
    parser.add_argument("--no-bio-gate", dest="bio_gate", action="store_false")
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 60)
    print("  TSQ Training")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch:  {args.batch}")
    print(f"  LR:     {args.lr}")
    print(f"  Bio-Gate: {args.bio_gate}")

    # Dataset
    if args.synthetic:
        print(f"  Data: Synthetic (structure validation)")
        train_ds = SyntheticDataset(num_samples=500, temporal_len=args.temporal_len)
        val_ds = SyntheticDataset(num_samples=100, temporal_len=args.temporal_len)
    else:
        print(f"  Data: {args.data_dir}")
        train_ds = KMERDataset(args.data_dir, temporal_len=args.temporal_len, split="train")
        val_ds = KMERDataset(args.data_dir, temporal_len=args.temporal_len, split="val")

    train_loader = create_dataloader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = create_dataloader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val:   {len(val_ds)} samples")

    # Model
    model = TemporalSlotQuery(
        d_model=args.d_model,
        num_slots=args.num_slots,
        temporal_len=args.temporal_len,
        bio_gate=args.bio_gate,
    ).to(device)

    # Loss + Optimizer
    criterion = TSQLoss(w_event=1.0, w_state=0.5, w_diversity=0.1)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Training loop
    best_f1 = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0

        log = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "event_f1": val_metrics["event_f1"],
            "state_f1": val_metrics["state_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "time_sec": elapsed,
        }
        history.append(log)

        # Print
        print(f"  [{epoch:3d}/{args.epochs}] "
              f"loss={train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} "
              f"event_f1={val_metrics['event_f1']:.3f} "
              f"state_f1={val_metrics['state_f1']:.3f} "
              f"lr={log['lr']:.2e} "
              f"({elapsed:.1f}s)")

        # Save best
        combined_f1 = val_metrics["event_f1"] + val_metrics["state_f1"]
        if combined_f1 > best_f1:
            best_f1 = combined_f1
            save_path = os.path.join(args.save_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "event_f1": val_metrics["event_f1"],
                "state_f1": val_metrics["state_f1"],
                "config": vars(args),
            }, save_path)
            print(f"  * Best model saved (F1={combined_f1:.3f})")

    # Save history
    history_path = os.path.join(args.save_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Best event_f1: {best_f1/2:.3f}")
    print(f"  Checkpoint: {args.save_dir}/best.pt")
    print(f"  History: {history_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
