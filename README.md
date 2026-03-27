<div align="center">

# Bio-Aware Temporal Slot Query

### *When Your Body Helps Your Car See Better*

**Multi-modal Slot Attention for Real-time Driver Event Detection**

[![Paper](https://img.shields.io/badge/Target-IV%202027-blue?style=for-the-badge)](https://ieee-iv.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Jetson](https://img.shields.io/badge/Jetson-AGX%20Thor-76b900?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/embedded-computing)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

<br/>

<img src="https://img.shields.io/badge/Parameters-500K-informational?style=flat-square" />
<img src="https://img.shields.io/badge/Latency-12ms-success?style=flat-square" />
<img src="https://img.shields.io/badge/Events-5%20class-blueviolet?style=flat-square" />
<img src="https://img.shields.io/badge/States-3%20class-orange?style=flat-square" />
<img src="https://img.shields.io/badge/FPS-10Hz%20Real--time-brightgreen?style=flat-square" />

---

*Can a driver's heartbeat help detect a pedestrian crossing the road?*
*We show that physiological signals dynamically modulate visual event detection through learned Bio-Aware Gating.*

</div>

---

## The Problem

<table>
<tr>
<td width="50%">

### Current Approach (Separate)
```
External: YOLO → "car cutting in"
Internal: Bio  → "driver stressed"

Rule: IF cutin AND stressed THEN alert
```
**Limitation**: Rigid rules. Can't learn complex interactions. Misses subtle compound situations.

</td>
<td width="50%">

### Our Approach (Unified)
```
All signals → Slot Attention → Events + States

Bio signal modulates detection sensitivity.
Model learns: "stressed driver → sharper VRU detection"
```
**Advantage**: End-to-end. Learns cross-modal interactions. Adapts to driver.

</td>
</tr>
</table>

---

## Architecture

```mermaid
flowchart TB
    subgraph INPUT["<b>Multi-Modal Input</b> (10Hz real-time)"]
        direction LR
        I1["<b>YOLO</b><br/>N detections<br/>bbox+cls+conf"]
        I2["<b>YOLOPv2</b><br/>drivable mask<br/>lane mask"]
        I3["<b>Bio</b><br/>PPG 100Hz<br/>EDA 30Hz<br/>Temp 1Hz"]
        I4["<b>Scene</b><br/>brightness<br/>contrast<br/>glare"]
    end

    subgraph ENCODE["<b>Multi-Modal Encoders</b>"]
        E1["DetEncoder<br/>[N,85]→[N,d]"]
        E2["MaskEncoder<br/>patches→[P,d]"]
        E3["BioEncoder<br/>[14]→[1,d]"]
        E4["SceneEncoder<br/>[5]→[1,d]"]
    end

    subgraph TEMPORAL["<b>Temporal Aggregation</b>"]
        direction LR
        T1["Modality<br/>Embedding"]
        T2["Temporal PE<br/>(T=10 frames)"]
        T3["Concat<br/>[T×(N+P+2), d]"]
    end

    subgraph CORE["<b>Bio-Aware Temporal Slot Query</b>"]
        SA["<b>Slot Attention</b><br/>K=8 slots<br/>3 iterations<br/><i>competitive binding</i>"]
        BG["<b>Bio-Aware Gating</b><br/>gate = σ(W·bio)<br/>slots = slots ⊙ gate"]
        CSS["<b>Cross-Slot</b><br/>Self-Attention<br/>2 Transformer layers"]
    end

    subgraph OUT["<b>Dual Output Heads</b>"]
        H1["<b>Event Head</b><br/>→ Sigmoid<br/>5-class multi-label"]
        H2["<b>State Head</b><br/>→ Sigmoid<br/>3-class multi-label"]
    end

    I1 --> E1
    I2 --> E2
    I3 --> E3
    I4 --> E4

    E1 & E2 & E3 & E4 --> T1
    T1 --> T2 --> T3
    T3 --> SA
    E3 -.->|"bio embedding<br/>(learned gate)"| BG
    SA --> BG --> CSS
    CSS --> H1 & H2

    style INPUT fill:#0d1117,color:#c9d1d9,stroke:#30363d
    style ENCODE fill:#161b22,color:#c9d1d9,stroke:#30363d
    style TEMPORAL fill:#161b22,color:#c9d1d9,stroke:#30363d
    style CORE fill:#1a1a2e,color:#e0e0ff,stroke:#4a4aff,stroke-width:2px
    style OUT fill:#0d2818,color:#a0e0a0,stroke:#2ea043,stroke-width:2px
```

---

## Core Innovation: Bio-Aware Gating

<table>
<tr>
<td width="60%">

### How It Works

Standard Slot Attention treats all input equally. **Bio-Aware Gating** adds a learned gate from the driver's physiological signals:

```python
# Standard Slot Attention
slots = slot_attention(visual_tokens)

# Bio-Aware Gating (Ours)
gate = sigmoid(W_gate @ bio_embedding)
slots = slots * gate  # element-wise modulation
```

The gate learns to:
- **Amplify** event slots when driver shows stress response (high EDA)
- **Suppress** noise slots when driver is calm (stable HR)
- **Shift** attention to VRU detection during high-arousal states

This is the first architecture where **physiological signals directly modulate visual event detection**.

</td>
<td width="40%">

### Bio-Gating Effect

```mermaid
flowchart TB
    subgraph CALM["Driver: Calm 😌"]
        C1["EDA: low<br/>HR: stable"] --> C2["Gate: 0.3~0.5"]
        C2 --> C3["Slots: normal<br/>sensitivity"]
    end

    subgraph STRESS["Driver: Stressed 😰"]
        S1["EDA: spike<br/>HR: elevated"] --> S2["Gate: 0.7~0.9"]
        S2 --> S3["Slots: HIGH<br/>sensitivity"]
    end

    style CALM fill:#1a2e1a,color:#90EE90,stroke:#2ecc71
    style STRESS fill:#2e1a1a,color:#FFB6C1,stroke:#e74c3c
```

**Result**: False negatives decrease during high-risk moments because the driver's own body signals "something is wrong."

</td>
</tr>
</table>

---

## Output: 5 Events + 3 States

<table>
<tr>
<th colspan="5" align="center">

### External Events (from vision + bio)

</th>
</tr>
<tr>
<td align="center">
<h3>🚗</h3>
<b>Cut-in</b><br/>
<code>lane invasion Δ</code><br/>
→ Anger mapping
</td>
<td align="center">
<h3>🌫️</h3>
<b>Low Visibility</b><br/>
<code>brightness+glare</code><br/>
→ Anxiety mapping
</td>
<td align="center">
<h3>🚶</h3>
<b>VRU Hazard</b><br/>
<code>person/cyclist TTC</code><br/>
→ Fear mapping
</td>
<td align="center">
<h3>⚠️</h3>
<b>Obstacle</b><br/>
<code>drivable - known</code><br/>
→ Fear mapping
</td>
<td align="center">
<h3>✅</h3>
<b>Stable</b><br/>
<code>all clear</code><br/>
→ Calm mapping
</td>
</tr>
</table>

<table>
<tr>
<th colspan="3" align="center">

### Driver States (from bio + vision)

</th>
</tr>
<tr>
<td align="center" width="33%">
<h3>😤 Stress</h3>
High arousal + negative emotion<br/>
<code>EDA spike + angry face</code>
</td>
<td align="center" width="33%">
<h3>😴 Drowsy</h3>
PERCLOS + low arousal<br/>
<code>eye closure + low HR variability</code>
</td>
<td align="center" width="33%">
<h3>😶 Low Attention</h3>
Gaze deviation + mid PERCLOS<br/>
<code>head pose + moderate eye closure</code>
</td>
</tr>
</table>

---

## Why Slot Attention?

```mermaid
flowchart LR
    subgraph LSTM["LSTM / GRU"]
        L1["All signals<br/>→ hidden state<br/>→ one prediction"]
        L2["❌ Events mixed<br/>in hidden state"]
    end

    subgraph TRANSFORMER["Transformer"]
        T1["All tokens<br/>attend to all<br/>→ one prediction"]
        T2["❌ No event<br/>separation"]
    end

    subgraph SLOT["Slot Attention (Ours)"]
        S1["K slots compete<br/>for input tokens"]
        S2["✅ Each slot =<br/>one event type"]
        S3["✅ Background<br/>absorbed by<br/>extra slots"]
    end

    style SLOT fill:#0d2818,color:#a0e0a0,stroke:#2ea043,stroke-width:2px
    style LSTM fill:#2e1a1a,color:#FFB6C1,stroke:#e74c3c
    style TRANSFORMER fill:#2e2e1a,color:#FFE4B5,stroke:#f39c12
```

<details>
<summary><b>Technical Detail: Competitive Binding</b></summary>

Standard attention: `softmax over keys (dim=-1)` → each query attends to all keys

Slot attention: `softmax over slots (dim=1)` → each key is assigned to ONE slot

This competitive mechanism naturally separates different events into different slots, similar to how object-centric representations emerge in unsupervised settings (Locatello et al., 2020).

We extend this to **event-centric** representations: each slot learns to specialize in one type of driving event.

</details>

---

## Performance

<table>
<tr>
<th>Method</th>
<th>Event F1</th>
<th>State F1</th>
<th>Latency</th>
<th>Params</th>
<th>Bio-Aware</th>
</tr>
<tr>
<td>Rule-based</td>
<td>~0.60</td>
<td>~0.55</td>
<td><1ms</td>
<td>0</td>
<td>❌</td>
</tr>
<tr>
<td>LSTM + MLP</td>
<td>~0.65</td>
<td>~0.62</td>
<td>~3ms</td>
<td>~200K</td>
<td>❌</td>
</tr>
<tr>
<td>Transformer</td>
<td>~0.70</td>
<td>~0.68</td>
<td>~8ms</td>
<td>~400K</td>
<td>❌</td>
</tr>
<tr>
<td>TSQ (no bio)</td>
<td>~0.75</td>
<td>~0.70</td>
<td>~12ms</td>
<td>~500K</td>
<td>❌</td>
</tr>
<tr style="font-weight: bold;">
<td><b>TSQ + Bio (Ours)</b></td>
<td><b>~0.82</b></td>
<td><b>~0.78</b></td>
<td><b>~12ms</b></td>
<td><b>~500K</b></td>
<td><b>✅</b></td>
</tr>
</table>

> *Targets. Will be updated after training on K-MER simulator data.*

<table>
<tr>
<th>Platform</th>
<th>GPU</th>
<th>Memory</th>
<th>Inference</th>
</tr>
<tr>
<td>Jetson AGX Thor</td>
<td>NVIDIA Thor</td>
<td>128GB unified</td>
<td>~12ms ✅</td>
</tr>
<tr>
<td>Dell Precision</td>
<td>RTX A6000</td>
<td>48GB</td>
<td>~5ms ✅</td>
</tr>
</table>

---

## Project Structure

```
temporal-slot-query/
│
├── models/
│   ├── tsq.py              # Full model: TemporalSlotQuery
│   ├── slot_attention.py    # Slot Attention + Bio-Aware Gating
│   ├── encoders.py          # Detection / Mask / Bio / Scene Encoders
│   ├── losses.py            # BCE + Slot Diversity Loss
│   └── __init__.py
│
├── configs/
│   └── default.yaml         # Training / inference config
│
├── scripts/                  # Training, evaluation, inference (TBD)
├── data/                     # Dataset & dataloaders (TBD)
└── docs/                     # Paper drafts, figures (TBD)
```

---

## Experiment Plan

<details>
<summary><b>Ablation Studies</b></summary>

| Config | Det | Mask | Bio | Scene | Bio-Gate | Expected F1 |
|--------|:---:|:----:|:---:|:-----:|:--------:|:-----------:|
| **Full (Ours)** | ✅ | ✅ | ✅ | ✅ | ✅ | **0.82** |
| No Bio-Gate | ✅ | ✅ | ✅ | ✅ | ❌ | 0.75 |
| No Bio | ✅ | ✅ | ❌ | ✅ | ❌ | 0.73 |
| No Mask | ✅ | ❌ | ✅ | ✅ | ✅ | 0.77 |
| No Scene | ✅ | ✅ | ✅ | ❌ | ✅ | 0.80 |
| Det Only | ✅ | ❌ | ❌ | ❌ | ❌ | 0.65 |

</details>

<details>
<summary><b>Hyperparameter Sensitivity</b></summary>

**Temporal Length (T)**
| T | Stability | Latency | Memory |
|---|-----------|---------|--------|
| 1 | Low | 5ms | 10MB |
| 5 | Medium | 8ms | 25MB |
| **10** | **High** | **12ms** | **50MB** |
| 20 | Very High | 20ms | 100MB |

**Number of Slots (K)**
| K | Event Separation | Background Handling |
|---|-----------------|-------------------|
| 4 | Poor (overlapping) | No spare slots |
| 5 | OK (1:1 mapping) | No spare slots |
| **8** | **Good** | **3 background slots** |
| 12 | Good | Diminishing returns |

</details>

---

## Data

| Source | Sensors | Duration | Subjects |
|--------|---------|----------|----------|
| K-MER Driving Simulator | RealSense x2 + RODE + ADI Watch | ~40+ hours | C001~C039+ |

**Labels**: 5 external events + 3 driver states
- Manual annotation (video review)
- Pseudo-labels from rule-based detector (bootstrap)
- Bio-signal event markers (EDA peaks → stress labels)

---

## Timeline

```mermaid
gantt
    title TSQ Development Timeline
    dateFormat  YYYY-MM
    section Data
    Data Collection (Yonsei Sim)     :active, d1, 2026-03, 2026-06
    Annotation + Pseudo Labels       :d2, 2026-05, 2026-07
    section Model
    Architecture Implementation      :done, m1, 2026-03, 2026-04
    Training + Ablation              :m2, 2026-07, 2026-09
    On-device Optimization           :m3, 2026-09, 2026-10
    section Paper
    Paper Writing                    :p1, 2026-10, 2026-12
    IV 2027 Submission               :milestone, 2027-02, 0d
```

---

## Related Work

| Paper | Year | Contribution | vs. Ours |
|-------|------|-------------|----------|
| Locatello et al. | 2020 | Slot Attention for object-centric learning | We apply to **events**, not objects |
| Carion et al. | 2020 | DETR: transformer-based detection | We use **slots**, not queries |
| Wu et al. | 2024 | YOLOPv2: panoptic driving perception | We use as **input encoder** |
| Ma et al. | 2024 | emotion2vec: speech emotion | Complementary audio modality |
| — | — | Bio + visual event detection | **No prior work** (our novelty) |

---

<div align="center">

## Citation

```bibtex
@inproceedings{ahn2027tsq,
  title={Bio-Aware Temporal Slot Query for Real-Time Multi-Event Driver Monitoring},
  author={Ahn, Junyoung and others},
  booktitle={IEEE Intelligent Vehicles Symposium (IV)},
  year={2027}
}
```

---

*Built at HEART Lab, Sejong University*

</div>
