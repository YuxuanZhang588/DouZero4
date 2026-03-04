# DouZero 4-Player Doudizhu — Project Roadmap

## Current State (March 2026)

**Model**: ResNet v2  
**Training**: `resnet_v2`, ~18M frames  
**Architecture**: Conv1d ResNet z-encoder + Dueling FC head + LayerNorm  
**Key issue**: Landlord WP ~40% vs random (baseline LSTM: 83%). Farmers are strong (93% vs random). Self-play asymmetry suppresses landlord learning.

---

## Milestone 1 — ResNet + Opponent Pool (In Progress)

**Goal**: Fix the landlord learning bottleneck via training improvements.

**Changes implemented / in progress**:
- [x] BN gamma collapse fix (`affine=False`)
- [x] Dueling Network head (V + A branches)
- [x] LayerNorm after each FC layer
- [ ] Asymmetric `exp_epsilon` (farmers 0.05, landlord 0.01)
- [ ] Independent landlord learning rate (`--learning_rate_landlord 0.0002`)
- [ ] Opponent pool: farmers sample historical checkpoints with prob 0.3

**Success criteria** (evaluate at ~50M frames):
| Metric | Target | Baseline ref |
|--------|--------|-------------|
| Landlord WP vs random | ≥ 70% | 83.2% (LSTM) |
| Landlord ADP vs random | ≥ +2.5 | +4.24 (LSTM) |
| ResNet landlord vs Baseline farmers | ≥ 25% | 8.3% current |
| Baseline landlord vs ResNet farmers | ≤ 40% | 51.5% current |

**Timeline**: ~1 week of training at current speed (~750 fps).

---

## Milestone 2 — Attention z-Encoder (Conditional on M1 failure)

**Trigger**: Landlord WP < 70% at 50M frames despite opponent pool.

**Hypothesis**: Conv1d cannot capture long-range dependencies in the 20-step action history (e.g., who played what 10 turns ago). A Transformer encoder with self-attention solves this directly.

### Architecture change

Replace `_ZEncoder` (Conv1d ResNet) with a Transformer:

```
z: (batch, 20, 52)
  → Linear(52, 64) projection per token    # (batch, 20, 64)
  → + learnable positional embedding        # turn order matters
  → TransformerEncoder(d_model=64, nhead=4, num_layers=3, dim_ff=256)
  → mean-pool over 20 tokens               # (batch, 64)
  → Linear(64, 640)                        # keep same downstream dim
  → (batch, 640)  ← same interface
```

**Advantages over Conv1d**:
- Self-attention sees all 20 turns simultaneously (no stride/receptive-field limit)
- Naturally models "who played what" when combined with player-ID embedding
- ~same parameter count as current ResNet z-encoder

**Additional changes**:
- Add per-token **player-ID embedding** (4 positions) concatenated to card encoding
- z encoding: `(batch, 20, 52+4)` → project to d_model

**Implementation plan**:
1. Add `_ZEncoderTransformer` to `models.py` (swap-in for `_ZEncoder`)
2. Add `--z_encoder` argument (`resnet` / `transformer`)
3. Port weights from best M1 checkpoint (FC layers compatible, z-encoder re-init)
4. New xpid: `attn_v1`

**Success criteria** (evaluate at ~30M frames from init):
| Metric | Target |
|--------|--------|
| Landlord WP vs random | ≥ 75% |
| ResNet landlord vs Baseline farmers | ≥ 35% |

---

## Milestone 3 — Optimized LSTM Baseline (Fallback)

**Trigger**: Landlord WP still < 70% after Transformer at 80M total frames.

**Strategy**: Abandon the from-scratch ResNet/Transformer approach. Instead, take the **existing strong LSTM baseline** (`baseline32652800-lr5e5`, 32.6M pretrain + 5.2M finetune) and push it to its ceiling with systematic optimizations.

### Why this is viable
The baseline landlord already achieves **83.2% WP** and **+4.24 ADP** vs random — competitive performance. The gap is only in the farmers (ResNet farmers slightly edge LSTM farmers). Fine-tuning a proven architecture is lower-risk than training a new one from scratch.

### Optimization plan

**3a. Extended fine-tuning** (easiest)
```
python train.py --xpid lstm_finetune \
  --load_model \          # from baseline32652800-lr5e5
  --learning_rate 0.00001 \   # lower LR for fine-tuning
  --batch_size 64
```
Target: push to 20M+ fine-tune frames (currently only 5.2M).

**3b. Reward shaping**
Current reward: `±2^bomb_num`. Try `logadp` (`±(bomb_num + 1)`) which reduces outlier bomb games and stabilises training variance.

**3c. Larger LSTM**
Current hidden size: 128. Try 256. The x-features are high-dimensional (418/422 dims); a larger LSTM extracts richer sequential patterns.

```python
# In LegacyLandlordModel:
self.lstm = nn.LSTM(208, 256, batch_first=True)   # 128 → 256
self.dense1 = nn.Linear(256 + 418, 512)
```

**3d. Curriculum**
Phase 1 (0–10M): train landlord against random farmers, pin farmers at baseline.
Phase 2 (10M+): full self-play. Gives landlord a solid foundation before facing strong farmers.

**3e. Ensemble at inference**
Average predictions of 3–5 checkpoints from different training stages (majority-vote on actions). No training cost, often +2–5% WP.

**Success criteria** (evaluate at 20M fine-tune frames):
| Metric | Target |
|--------|--------|
| Landlord WP vs random | ≥ 87% | 
| Landlord ADP vs random | ≥ +5.0 |
| Landlord vs ResNet v2 farmers | ≥ 55% |
| Farmers WP vs random | ≥ 95% |

---

## Decision Tree

```
Start
  │
  ▼
[M1] ResNet + Opponent Pool @ 50M frames
  │
  ├─ Landlord WP ≥ 70%?
  │     YES → Continue training to 100M, publish results
  │
  └─ NO
        │
        ▼
      [M2] Transformer z-Encoder @ 80M frames
        │
        ├─ Landlord WP ≥ 75%?
        │     YES → Continue, declare ResNet/Attn approach viable
        │
        └─ NO
              │
              ▼
            [M3] Optimized LSTM Baseline
              → Fine-tune existing strong model to ceiling
              → Systematic hyperparameter sweep
              → Ensemble inference
```

---

## Evaluation Protocol

Standard evaluation always uses `eval_data.pkl` (1000 fixed games):

```powershell
# vs random (absolute quality)
python evaluate.py --landlord <ckpt> --landlord_next random ...

# Cross-match (relative vs baseline)  
python evaluate.py --landlord <ckpt> --landlord_next baseline_next ... --legacy_positions landlord_next,...
python evaluate.py --landlord baseline_landlord ... --landlord_next <ckpt> --legacy_positions landlord
```

Evaluate at: 10M, 20M, 50M, 100M frames.

---

## Checkpoint Index

| xpid | Arch | Frames | Landlord WP | Notes |
|------|------|--------|-------------|-------|
| `baseline32652800-lr5e5` | LSTM | 32.6M pretrain + 5.2M ft | **83.2%** | Strong reference |
| `resnet_lr5e5` | ResNet v1 (BN bug) | 4.8M | 39.6% | BN gamma=0 bug |
| `resnet_2train` | ResNet v1 (BN bug) | 8.2M | 39.6% | Farmer collapse |
| `resnet_v2` | ResNet v2 (fixed) | 17.9M | 40.2% | Opponent pool pending |
| `resnet_v3` *(planned)* | ResNet v2 + pool | 50M target | ? | M1 |
| `attn_v1` *(conditional)* | Transformer | — | ? | M2 |
| `lstm_finetune` *(fallback)* | LSTM | — | ? | M3 |
