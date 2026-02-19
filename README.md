# TGAD — Temporal Graph Anomaly Detection

Edge-level anomaly detection on provenance graphs built from the **DARPA TC CADETS E3** dataset. The system encodes system-call audit logs into heterogeneous temporal graphs and trains a **GINE-based** edge classifier to detect malicious activity at syscall granularity.

---

## Dataset — DARPA TC CADETS E3

The dataset spans **April 2–13, 2018** and is stored in a PostgreSQL database (`tc_cadet_dataset_db`). Each row in `event_table` represents a system-call event between two entities.

| Day | Label | Attack Scenario |
|-----|-------|----------------|
| 2–5, 7–9 | **BENIGN** | Normal system activity |
| **6** | **ATTACK** | §3.1 Nginx backdoor → Drakon in-memory RAT, netrecon, sshd inject; §4.1 Phishing via postfix |
| **10** | **ATTACK** | §4.1 Phishing e-mail re-routed through CADETS postfix |
| **11** | **ATTACK** | §3.8 Nginx backdoor → Drakon, grain, sshd inject |
| **12** | **ATTACK** | §3.13 Nginx → Drakon, tmux, minions, font, XIM, sendmail, micro APT port scans |
| **13** | **ATTACK** | §3.14 Nginx → Drakon, pEja72mA, eWq10bVcx, eraseme, memhelp.so, done.so |

Ground truth sourced from `TC_Ground_Truth_Report_E3_Update.pdf` (Kudu Dynamics / DARPA TC) and encoded in [`label_data.py`](label_data.py) as sets of malicious IPs and node UUIDs per day.

---

## Model — GINEEdgeModel

```
Input: node features x [N, 387],  edge_index [2, E],  edge_relation [E, 7]
                │
      ┌─────────▼──────────┐
      │  GINEConv Layer 1   │  node_in=387 → hidden=128,  edge_dim=7
      │  + ReLU             │
      └─────────┬──────────┘
                │
      ┌─────────▼──────────┐
      │  GINEConv Layer 2   │  hidden=128 → 128,  edge_dim=7
      │  + ReLU             │
      └─────────┬──────────┘
                │
        src_emb ║ dst_emb ║ edge_attr
                │    [hidden*2 + 7 = 263]
      ┌─────────▼──────────┐
      │    Edge MLP         │  263 → 128 → ReLU → Dropout(0.3) → 2
      └─────────┬──────────┘
                │
           logits [E, 2]   →   softmax   →   anomaly score per edge
```

**GINE** (Graph Isomorphism Network with Edge features) extends GIN by injecting edge features at each message-passing step, making the model sensitive to syscall type — essential for separating attack-related system calls from normal ones.

---

## Training Procedure

### 1. Edge Balancing
Attack days have severely imbalanced edges (~1–5% malicious). Benign edges are subsampled at a configurable ratio before training:

$$\text{benign sample} = \min\left(|\text{malicious}| \times r,\ |\text{benign}|\right)$$

Default `benign_ratio = 3` → 1 malicious : 3 benign.

### 2. Loss
`CrossEntropyLoss` over per-edge logits.

### 3. Optimiser
`Adam`, `lr=1e-3`.

### 4. Early Stopping
Validation loss is computed on a balanced held-out graph (first test day) after every epoch. Training halts when val loss does not improve by `min_delta=1e-4` for `patience=10` consecutive epochs. The best checkpoint is saved to `artifact/models/best_model.pt` and reloaded before evaluation.

### 5. Threshold Selection
Instead of a fixed 0.5, the optimal decision threshold is selected by sweeping the PR curve and picking the point that maximises F1:

$$t^* = \arg\max_t \frac{2 \cdot P(t) \cdot R(t)}{P(t) + R(t)}$$

---

## Logging

All output is written to both the console and `artifact/gad.log` via the shared logger in [`logger.py`](logger.py).

### Per-epoch training log
```
2026-02-19 ... INFO - Epoch   0 | train_loss=0.6821  val_loss=0.6504 | train_auc=0.8231  train_f1=0.7412 | val_auc=0.8019  val_f1=0.7204 | patience=0/10
2026-02-19 ... INFO - Epoch   1 | train_loss=0.5934  val_loss=0.5812 | train_auc=0.8876  train_f1=0.8103 | val_auc=0.8541  val_f1=0.7890 | patience=0/10
...
2026-02-19 ... INFO - Early stopping triggered at epoch 23
2026-02-19 ... INFO - Loaded best model from epoch 13 (val_loss=0.3214)
```

### Overall combined report (all test days)
After all test days are evaluated, predictions across every day are concatenated and a single combined `classification_report` is logged.

---

## Evaluation Outputs

After training, for each test day the following are saved to `artifact/plots/`:

| Panel | Content |
|-------|---------|
| PR Curve | Precision vs Recall with best-threshold marker (red dashed) and AP score |
| ROC Curve | FPR vs TPR with AUC score |
| Confusion Matrix | TP / FP / FN / TN with `Benign` / `Malicious` class labels |

---

## Setup

### Prerequisites
- Python 3.10
- CUDA-capable GPU (tested on RTX 3090)
- PostgreSQL with `tc_cadet_dataset_db` loaded
- Conda

### Installation

```bash
conda create -n tgad python=3.10
conda activate tgad

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

# Fix MKL version conflict (if OpenBLAS warnings appear)
conda install "mkl<2025" "intel-openmp<2025" -y
```

---

## Running

### Step 1 — Build graphs
```bash
python3 build_graphs.py
```
Connects to PostgreSQL, encodes node texts with `all-MiniLM-L6-v2`, builds one PyG `Data` object per day (days 2–13), saves to `artifact/graphs/`.

> First run downloads `all-MiniLM-L6-v2` (~90 MB) and caches the 387-dim embeddings to `artifact/node_embeddings.pt`. All subsequent runs load directly from cache.

### Step 2 — Train and evaluate
```bash
python3 train_test.py
```
- Loads and balances training graphs
- Trains `GINEEdgeModel` with early stopping
- Logs per-epoch `train_loss`, `val_loss`, `train_auc`, `train_f1`, `val_auc`, `val_f1` to console and `artifact/gad.log`
- Evaluates on each test day and logs a full `classification_report` (precision / recall / F1 / support per class)
- Logs an overall combined classification report across all test days
- Saves PR curve, ROC curve, and confusion matrix plots to `artifact/plots/`

### Configuration
Edit [`config.py`](config.py) to change:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_days` | `[6, 10]` | Days used for training |
| `test_days` | `[11, 12]` | Days used for evaluation |
| `epochs` | `100` | Max training epochs |
| `benign_ratio` | `3` | Benign:malicious edge sampling ratio |

---

## Key Dependencies

| Package | Version |
|---------|---------|
| `torch` | 2.5.1 |
| `torch-geometric` | 2.7.0 |
| `sentence-transformers` | 5.2.3 |
| `scikit-learn` | 1.7.2 |
| `numpy` | 2.0.1 |
| `matplotlib` | 3.10.8 |
| `psycopg2-binary` | 2.9.11 |

---

## Citation / Acknowledgements

Dataset: DARPA Transparent Computing (TC) Program — CADETS E3  
Ground truth: `TC_Ground_Truth_Report_E3_Update.pdf`, Kudu Dynamics  
Kairos: [`ubc-provenance/kairos`](https://github.com/ubc-provenance/kairos)
