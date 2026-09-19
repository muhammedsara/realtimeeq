<p align="center">
  <img src="figures/fig_system_architecture.png" width="760"/>
</p>

<h1 align="center">Edge-AI Earthquake Early Warning</h1>

<h3 align="center">Real-Time Seismic Detection with Edge-AI: A Balanced Multi-Architecture Comparison</h3>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-FF6F00.svg" alt="TensorFlow 2.15+">
  <img src="https://img.shields.io/badge/TFLite-INT8-brightgreen.svg" alt="TFLite INT8">
  <img src="https://img.shields.io/badge/Android-Deployment-3DDC84.svg" alt="Android">
</p>

<p align="center">
  <strong>Edge-AI · TFLite INT8 · Majority-Vote Ensemble · 35.4M Window Dataset · Android Deployment</strong>
</p>

<p align="center">
  Muhammed Şara
</p>

<p align="center">
  Department of Information Systems Engineering, Kocaeli University, Türkiye
</p>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#main-results">Key Results</a> •
  <a href="#methodology">Methodology</a> •
  <a href="#repository-structure">Structure</a> •
  <a href="#getting-started">Quick Start</a> •
  <a href="#pre-trained-models">Models</a> •
  <a href="#citation">Citation</a>
</p>

---

### Introduction

This repository contains the source code, pre-trained models, and publication figures accompanying the paper **"Real-Time Seismic Detection with Edge-AI: A Balanced Comparison"**.

We propose an edge-to-cloud earthquake early warning (EEW) framework that runs entirely on a smartphone's on-device accelerometer, eliminating dependence on dedicated seismometer networks. Three systematic gaps motivate this work: (1) severe class imbalance in crowdsourced sensor streams, (2) the absence of multi-phase temporal modelling in lightweight mobile architectures, and (3) the lack of architecture-level deployment guidance across the edge-hardware spectrum. The core contribution is a systematic, balanced comparison of five deep learning architectures that addresses all three.

**Key Contributions:**
- **Unified Dataset** — 35.4 million 2-second accelerometer windows from 8 heterogeneous sources (AFAD broadband, TDG shake-table MEMS, STEAD earthquake traces, MyShake-shake, STEAD-Noise, MyShake-Human, UCI HAR, WISDM)
- **Balanced Evaluation** — A 79.74:1 class imbalance corrected to 1:1 via stratified index-based sampling, without modifying source data; a human-activity-inclusive balanced EEW evaluation framework
- **Five Architectures** — Systematic comparison spanning Baseline CNN → CNN-LSTM → CNN-GRU → CNN-BiLSTM-Bahdanau-Attention → Dilated TCN
- **On-Device Deployment** — INT8 TFLite quantisation (44–138 KB) with measured Android inference latency; majority-vote ensemble completes within 14 ms per 2-second window
- **Deployment Recommendations** — Architecture-to-hardware mapping covering microcontrollers, Android smartphones, and edge accelerators

---

### Main Results

**Table 1 — Architecture Comparison on Balanced Test Set** (*N* = 200,000, 1:1 EQ:NonEQ)

| Phase | Model | Params | AUC | F1 | EQ Recall | Gap |
|-------|-------|--------|-----|----|-----------|-----|
| Ph. 1 | Baseline CNN, imbalanced training † | 28K | 0.8868 | 0.8261 | 0.8564 | 0.056 |
| Ph. 1 | Baseline CNN, balanced training ‡ | 28K | 0.9167 | 0.8362 | 0.8402 | 0.109 |
| Ph. 2 | CNN-LSTM | 35K | 0.9521 | 0.8787 | 0.8821 | 0.009 |
| Ph. 2 | CNN-GRU | 17K | 0.9522 | 0.8837 | 0.8795 | 0.007 |
| Ph. 2 | CNN-BiLSTM-Attn | 41K | 0.9617 | 0.8949 | 0.8995 | 0.004 |
| Ph. 3 | TCN | 103K | 0.9688 | 0.9081 | 0.9107 | 0.007 |

> Every row is measured on the balanced test set at that model's decision threshold. Params is the total parameter count (trainable weights plus BatchNorm statistics). Gap = train AUC − val AUC at the final epoch, as raw ΔAUC; lower means better generalisation.  
> † Trained on the imbalanced training split (11.69:1 EQ:NonEQ).  
> ‡ Trained on the balanced 1:1 subset, as every Phase 2–3 model was. This is the like-for-like reference: against it the temporal architectures gain 0.035–0.052 AUC and cut the generalisation gap from 0.109 to 0.004–0.009, a 12–27× reduction. Balancing raises the level the baseline can reach (best validation AUC 0.8772 → 0.9128) at one eighteenth of the training cost; closing the gap is the contribution of temporal modelling, not of balancing.

**Table 2 — On-Device Inference Latency**

| Model | INT8 Size | Android Mean | Android Peak | GPU (INT8) |
|-------|-----------|-------------|-------------|-----------|
| CNN-GRU | 44 KB | 3.2 ms | 8 ms | 0.47 ms |
| CNN-BiLSTM-Attn | 91 KB | 3.6 ms | 12 ms | 0.90 ms |
| TCN | 138 KB | 6.4 ms | 14 ms | 0.37 ms |
| Ensemble (3 models) | — | 13.5 ms | ~33 ms | — |

> All values below the 50 ms EEW latency target. Android measurements on a commercial mid-range device (FP32-builtins variants, 239–486 KB).

<p align="center">
  <img src="figures/fig3_deployment_pareto.png" width="680"/>
</p>

> CNN-BiLSTM-Attention is the **Pareto-optimal** operating point for Android deployment: 99.3% of TCN accuracy at 65.9% of its INT8 model size.

---

### What the pooled AUC does and does not mean

The numbers above are measured on a balanced test set whose positive class is
almost entirely seismometer-recorded: 99,990 of the 100,000 positive windows come
from AFAD and STEAD, and only 10 from a phone. A model could therefore score well
partly by recognising the recording instrument. We measured whether it does, by
splitting the test set according to the provenance of the two classes being
compared.

| Evaluated pair | Baseline | CNN-LSTM | CNN-GRU | CNN-BiLSTM-Attn | TCN |
|---|---|---|---|---|---|
| Pooled test set (table above) | 0.9167 | 0.9521 | 0.9522 | 0.9617 | 0.9688 |
| **Seismometer EQ vs seismometer noise** | 0.9152 | 0.9441 | 0.9441 | 0.9527 | 0.9622 |
| Seismometer EQ vs phone/IMU noise | 0.9218 | 0.9790 | 0.9790 | 0.9918 | 0.9906 |
| **Phone-recorded EQ, 50 Hz** (n = 4,393) | 0.8931 | 0.8585 | 0.9034 | 0.8822 | 0.9212 |
| Phone-recorded EQ, 25 Hz (n = 6,974) | 0.4194 | 0.5842 | 0.5090 | 0.6568 | 0.6110 |

Three things follow, and they matter for anyone planning to deploy this.

**The models separate seismic content, not sensor identity.** In the
same-instrument pairing both classes are seismometer recordings, so an
instrument-recognition shortcut is unavailable — and discrimination holds at
0.944–0.962, within 0.007–0.018 of the pooled value.

**A provenance cue nonetheless exists.** Seismometer earthquakes against phone
negatives is the easiest pairing of all, above the pooled value. The pooled AUC is
therefore an upper bound on field performance rather than an estimate of it.

**Transfer is governed by recording bandwidth.** Seismometer recordings carry
18–24% of their 0.5–20 Hz energy above 12.5 Hz; the 25 Hz phone recordings carry
1.4% and the phone/IMU negatives 0.8%, because everything above a device's own
Nyquist frequency is resampling artefact rather than measurement. Detection works
on 50 Hz phone recordings and fails at 25 Hz. **Participating handsets should
therefore sample at no less than about 50 Hz** — a sampling-rate floor, not a
sensor-quality threshold.

<p align="center">
  <img src="figures/fig_r1_provenance.png" width="920"/>
</p>

---

### Choosing a threshold for deployment

The thresholds in the table above maximise F1 on a 1:1 validation set. That is the
right criterion for comparing architectures and the wrong one for a handset, which
sees no earthquake for months. At those thresholds the per-device false-alarm load
is 927–2,848 events per day, and precision at a prior of 10⁻⁴ is below 0.01 for
every architecture.

Fixing the threshold by a false-alarm budget instead **reverses the ranking**:

| Budget | Best on seismometer EQ | Best on phone EQ (50 Hz) |
|---|---|---|
| 100 false alarms / device-day | **CNN-BiLSTM-Attn 0.756** | **CNN-BiLSTM-Attn 0.610** |
| 10 false alarms / device-day | **CNN-GRU 0.564** | **CNN-GRU 0.429** |

The TCN leads on pooled AUC but falls to 0.525 / 0.326 at 100 and 0.237 / 0.128 at
10 false alarms per device-day: its probability outputs saturate near unity, so the
upper tail that a tight budget selects on is poorly resolved. This is why the
attention model, not the TCN, is the recommendation for on-handset use.

<p align="center">
  <img src="figures/fig_r1_far_tradeoff.png" width="900"/>
</p>

---

### How much of this is training noise

Each deployment candidate was retrained over five seeds, varying nothing but
initialisation and batch order:

| Model | n | Mean AUC | SD | 95% CI | Min–Max |
|---|---|---|---|---|---|
| CNN-GRU | 5 | 0.9521 | 0.0040 | [0.9485, 0.9556] | 0.9466–0.9578 |
| CNN-BiLSTM-Attn | 5 | 0.9588 | 0.0027 | [0.9564, 0.9611] | 0.9547–0.9617 |
| TCN | 5 | 0.9739 | 0.0024 | [0.9718, 0.9759] | 0.9705–0.9761 |

The confidence intervals do not overlap and Welch *t*-tests separate all three
pairs, so the ordering TCN > CNN-BiLSTM-Attn > CNN-GRU is a property of the
architectures. One comparison does **not** survive: a DeLong test on the published
models gives *p* = 0.73 for CNN-LSTM against CNN-GRU, whose AUCs differ by 0.0001
— forty times smaller than the seed standard deviation. Treat those two as
indistinguishable in accuracy and pick CNN-GRU for its 49.7% smaller parameter
count.

Two caveats worth stating. These runs use a comparable but not identical protocol
to the originals (early-stopping patience 6 rather than 10, a 30-epoch cap), so
they estimate the spread rather than reproducing the checkpoints bit-for-bit. And
the single runs in the table above sit at different points within that spread: the
CNN-GRU value on its mean, the CNN-BiLSTM-Attn value at the top of its range, the
TCN value just below its range.

<p align="center">
  <img src="figures/fig_r1_balancing_curves.png" width="900"/>
</p>

> Balancing raises the achievable validation AUC from 0.8772 to 0.9128 and gets
> there after 2.3 h of training against 42.5 h, an eighteen-fold reduction in the
> cost of one architecture trial.

---

### Methodology

```
┌─────────────────────┐   ┌─────────────────────────────┐   ┌──────────────────────────┐
│  8-Source Dataset   │   │  Preprocessing Pipeline     │   │  Architecture Comparison  │
│                     │   │                             │   │                          │
│  AFAD broadband  ───┼──▶│  Resample → 100 Hz          │──▶│  Ph.1  Baseline CNN      │
│  TDG shake-table ───┤   │  Butterworth BP 0.5–20 Hz   │   │  Ph.2  CNN-LSTM          │
│  STEAD (eq)      ───┤   │  Z-score (per-window)       │   │        CNN-GRU ★         │
│  MyShake-shake   ───┤   │  Window: 200 samples (2 s)  │   │        CNN-BiLSTM-Attn ★ │
│  STEAD-Noise     ───┤   │  Stride: 100 samples (1 s)  │   │  Ph.3  Dilated TCN ★     │
│  MyShake-Human   ───┤   │                             │   │                          │
│  UCI HAR         ───┤   │  Balanced Sampling          │   │  ★ = TFLite deployed     │
│  WISDM           ───┘   │  79.74:1 → 1:1 (seed=42)   │   │                          │
└─────────────────────────┘  └───────────────────────────┘   └──────────────────────────┘
                                                                          │
                                                           ┌──────────────▼─────────────┐
                                                           │  Android Ensemble App      │
                                                           │  Majority vote (≥2/3)      │
                                                           │  14 ms / 2-second window   │
                                                           └────────────────────────────┘
```

**Preprocessing Pipeline:**
1. Resample to 100 Hz
2. 4th-order Butterworth band-pass filter (0.5–20 Hz)
3. Per-window z-score normalisation
4. 200-sample (2 s) sliding windows, 100-sample (1 s) stride

**Majority-Vote Ensemble:**

| Model | Detection Threshold |
|-------|-------------------|
| CNN-GRU | 0.30 |
| CNN-BiLSTM-Attn | 0.35 |
| TCN | 0.65 |

Alarm triggered when ≥ 2 of 3 models exceed their respective thresholds.

<p align="center">
  <img src="figures/fig5_mobile_inference.png" width="600"/>
</p>

---

### Repository Structure

```
realtimeeq/
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
├── .gitignore
│
├── src/                                   # Source code
│   ├── models/
│   │   ├── baseline_cnn.py                #   Phase 1 — Baseline 1D-CNN
│   │   ├── hybrid_models.py               #   Phase 2 — CNN-LSTM, CNN-BiLSTM-Attn, CNN-GRU
│   │   └── tcn.py                         #   Phase 3 — Dilated TCN
│   ├── data/
│   │   └── preprocessor.py                #   Signal preprocessing pipeline
│   └── deployment/
│       └── tflite_converter.py            #   TFLite INT8 conversion
│
├── models/                                # Pre-trained TFLite models
│   ├── int8/                              #   INT8 quantised (GPU / high-end edge)
│   │   ├── cnn_gru_int8.tflite            #     45 KB
│   │   ├── cnn_bilstm_attn_int8.tflite    #     91 KB
│   │   └── tcn_int8.tflite                #    138 KB
│   └── android/                           #   FP32-builtins (Android CPU compatible)
│       ├── cnn_gru_builtins.tflite        #    239 KB
│       ├── cnn_bilstm_attn_builtins.tflite#    486 KB
│       └── tcn_builtins.tflite            #    417 KB
│
├── figures/                               # Publication figures (PNG)
│   ├── fig_system_architecture.png        #   End-to-end system architecture
│   ├── fig_tdg_pipeline.png               #   Shake-table acquisition protocol
│   ├── fig1_roc_curves.png
│   ├── fig2_performance_summary.png
│   ├── fig3_deployment_pareto.png
│   ├── fig4_attention_weights.png
│   ├── fig5_mobile_inference.png
│   ├── fig_r1_balancing_curves.png        #   Effect of dataset balancing
│   ├── fig_r1_provenance.png              #   Provenance-stratified evaluation
│   └── fig_r1_far_tradeoff.png            #   Detection rate vs false-alarm budget
│
└── scripts/
    └── reproduce_figures.py               # Regenerate figures from raw results
```

> **Dataset**: The full 35.4 M window HDF5 dataset (~60 GB) is not hosted on GitHub. See [Data Availability](#data-availability) below.

---

### Getting Started

#### Prerequisites

- Python ≥ 3.8
- TensorFlow ≥ 2.15
- CUDA (optional, for GPU training)

#### Installation

```bash
git clone https://github.com/muhammedsara/realtimeeq.git
cd realtimeeq
pip install -r requirements.txt
```

#### Run a Pre-trained Model

```python
import numpy as np
import tensorflow as tf

# Load INT8 model (GPU / server)
interpreter = tf.lite.Interpreter(model_path="models/int8/cnn_gru_int8.tflite")
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 2-second window: (1, 200, 3) — 100 Hz, 3-axis, z-score normalised
window = np.random.randn(1, 200, 3).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], window)
interpreter.invoke()

prob = interpreter.get_tensor(output_details[0]['index'])[0, 0]
print(f"P(earthquake) = {prob:.4f}")
```

#### Use the Preprocessing Pipeline

```python
from src.data.preprocessor import SignalPreprocessor

preprocessor = SignalPreprocessor(
    target_sr=100.0,
    filter_low=0.5,
    filter_high=20.0,
    filter_order=4,
    normalization="zscore",
    window_size=200,
    window_stride=100,
)

# raw_signal: np.ndarray, shape (n_samples, 3), original_sr: Hz
windows = preprocessor.process_and_window(raw_signal, original_sr=50.0)
# windows.shape → (n_windows, 200, 3)
```

#### Train an Architecture

```python
from src.models.hybrid_models import build_cnn_bilstm_attention, compile_hybrid

model = build_cnn_bilstm_attention(input_shape=(200, 3))
model = compile_hybrid(model)
model.summary()

# model.fit(train_gen, validation_data=val_gen, epochs=50, ...)
```

---

### Pre-trained Models

| File | Architecture | Format | Size | AUC | Android Mean |
|------|-------------|--------|------|-----|-------------|
| `int8/cnn_gru_int8.tflite` | CNN-GRU | INT8 | 45 KB | 0.9522 | — |
| `int8/cnn_bilstm_attn_int8.tflite` | CNN-BiLSTM-Attn | Partial INT8 | 91 KB | 0.9617 | — |
| `int8/tcn_int8.tflite` | TCN | INT8 | 138 KB | 0.9688 | — |
| `android/cnn_gru_builtins.tflite` | CNN-GRU | FP32-builtins | 239 KB | 0.9522 | 3.2 ms |
| `android/cnn_bilstm_attn_builtins.tflite` | CNN-BiLSTM-Attn | FP32-builtins | 486 KB | 0.9617 | 3.6 ms |
| `android/tcn_builtins.tflite` | TCN | FP32-builtins | 417 KB | 0.9688 | 6.4 ms |

**Which variant to use:**
- **Android deployment** → use `android/` variants (TFLITE_BUILTINS compatible, no Flex delegate required)
- **GPU server / edge accelerator** → use `int8/` variants
- **Recommended single model** → `cnn_gru_builtins.tflite` (best latency), `cnn_bilstm_attn_builtins.tflite` (best accuracy/size tradeoff)

---

### Evaluation Metrics

| Metric | Definition |
|--------|------------|
| **AUC** | Area under the ROC curve — primary ranking metric, threshold-independent |
| **F1-score** | Harmonic mean of precision and recall at the optimal threshold |
| **EQ Recall** | Fraction of true earthquakes detected — critical for EEW (miss cost > false alarm cost) |
| **Non-EQ Precision** | Fraction of positive predictions that are true earthquakes |
| **Overfitting Gap** | train AUC − val AUC at final epoch; Phase 2–3 models achieve < 0.01 pt (well-regularised); see † for Baseline context |

All metrics are reported on the **balanced test set** (200,000 windows, 1:1 EQ:NonEQ). The original 79.74:1 imbalance means a trivial "always predict earthquake" classifier achieves 98.7% accuracy — making the balanced evaluation essential for meaningful comparison.

<p align="center">
  <img src="figures/fig2_performance_summary.png" width="680"/>
</p>

---

### Data Availability

The shake-table recordings were acquired under the protocol below; the at-rest
windows before and after each playback come from the same sensor as the positive
windows, so a per-device sensor bias cannot separate the two classes.

<p align="center">
  <img src="figures/fig_tdg_pipeline.png" width="620"/>
</p>

The full unified dataset (`unified_dataset_v1.hdf5`, ~60 GB, 35.4 M windows) is available upon request. The balanced sampling index (`balanced_indices_v1.json`, ~20 MB, seed=42) is provided separately and enables exact reproduction without redistributing the raw data.

Data sources used (8 heterogeneous sources, 35.4 M windows total):

The TDG entry counts the 23 shake-table recordings captured at 50 Hz across four
playback scenarios on seven handsets and tablets (four Android, three iOS). Six
further TDG recordings sampled at ~2 Hz violate the Nyquist criterion for the
0.5–20 Hz band; they remain in the training and validation pools, where they are
0.12% and 0.30% of the balanced subsets, and are absent from the test split, so no
reported test-set result depends on them.
| # | Source | Label | Sensor Type |
|---|--------|-------|-------------|
| 1 | AFAD Broadband Network (93,442 records) | EQ | Seismometer |
| 2 | TDG Shake-Table MEMS (23 shake-table records) | EQ | MEMS accelerometer |
| 3 | STEAD earthquake traces (13,000 records) | EQ | Seismometer |
| 4 | MyShake-shake (192 records) | EQ | Smartphone MEMS |
| 5 | STEAD-Noise (15,000 records) | Non-EQ | Seismometer |
| 6 | MyShake-Human (613 records) | Non-EQ | Smartphone MEMS |
| 7 | UCI HAR (10,299 records) | Non-EQ | Body-worn IMU |
| 8 | WISDM (54,000 records) | Non-EQ | Smartphone IMU |

---

### Acknowledgements

This work was conducted at the Department of Information Systems Engineering, **Kocaeli University**, Türkiye.

The authors thank:
- [AFAD](https://deprem.afad.gov.tr/) for providing broadband seismometer recordings from the Turkish national network
- [STEAD](https://github.com/smousavi05/STEAD) (Mousavi et al., 2019) for the publicly available seismic dataset
- TDG (Teknik Destek Grubu) Vibration and Dynamics Laboratory for shake-table MEMS recordings
- UCI Machine Learning Repository for the HAR dataset
- WISDM Lab, Fordham University for the activity recognition dataset
- [MyShake](https://myshake.berkeley.edu/) (Kong et al., 2016) for the smartphone seismic and human-activity datasets

---

### Citation

If you use this work in your research, please cite it as follows:

```bibtex
@article{sara2025eew,
  title   = {Real-Time Seismic Detection with Edge-AI: A Balanced Comparison},
  author  = {{\c{S}}ara, Muhammed and others},
  year    = {2025},
  note    = {Manuscript in preparation (under review)}
}
```


---

### Contact

For questions or bug reports, please open an issue or contact **muhammedsaraa@gmail.com**.

---

### License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
