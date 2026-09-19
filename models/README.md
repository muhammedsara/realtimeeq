# Pre-trained TFLite Models

All models trained on the balanced training subset (2 M windows drawn from the
35.4 M window archive, seed=42, 1:1 EQ:NonEQ) and evaluated on the balanced test
set (200,000 windows, disjoint at recording level).
Input: `(1, 200, 3)` — 2-second window @ 100 Hz, 3-axis accelerometer, z-score normalised.
Output: `float32` in `[0, 1]` — P(earthquake).

---

## `int8/` — INT8 Quantised (GPU / Server / Edge Accelerator)

| File | Architecture | INT8 Size | AUC | GPU Latency |
|------|-------------|-----------|-----|------------|
| `cnn_gru_int8.tflite` | CNN-GRU | 44 KB | 0.9522 | 0.47 ms |
| `cnn_bilstm_attn_int8.tflite` | CNN-BiLSTM-Attn | 91 KB | 0.9617 | 0.90 ms* |
| `tcn_int8.tflite` | TCN (dilated) | 138 KB | 0.9688 | 0.37 ms |

> \* Partial INT8: attention layer uses FP32 fallback (Flex delegate). GPU latency measured on NVIDIA RTX 5000 Ada, median over 200 runs.

## `android/` — FP32-Builtins (Android CPU Compatible)

| File | Architecture | Size | AUC | Android Mean | Android Peak |
|------|-------------|------|-----|-------------|-------------|
| `cnn_gru_builtins.tflite` | CNN-GRU | 239 KB | 0.9522 | 3.2 ms | 8 ms |
| `cnn_bilstm_attn_builtins.tflite` | CNN-BiLSTM-Attn | 486 KB | 0.9617 | 3.6 ms | 12 ms |
| `tcn_builtins.tflite` | TCN (dilated) | 417 KB | 0.9688 | 6.4 ms | 14 ms |

> Android measurements on a commercial mid-range Android device. Built with `unroll=True` to eliminate `TensorListReserve` Flex ops — compatible with `TFLITE_BUILTINS` only, no Flex delegate required.

---

## Which Model to Use

| Use Case | Recommended Model |
|----------|------------------|
| Android app | `cnn_gru_builtins.tflite` (lowest latency) |
| Android app (max accuracy) | `cnn_bilstm_attn_builtins.tflite` (Pareto-optimal) |
| Server / GPU pipeline | `tcn_int8.tflite` (highest AUC, fastest GPU) |
| Ultra-constrained MCU | `cnn_gru_builtins.tflite` (239 KB, FP32-builtins) — see note below |
| Majority-vote ensemble | All three `android/` variants, threshold GRU≥0.30, BiLSTM≥0.35, TCN≥0.65 |

> **MCU note:** The INT8 variants (`int8/`) contain attention Flex ops and dynamic-shape GRU cells that require TFLite Micro with Flex support — not available on standard MCU runtimes (e.g., bare ESP32). For MCU deployment, use `cnn_gru_builtins.tflite` (239 KB FP32-builtins, fully static graph) and verify it fits within your device's available SRAM.

---

## Loading Models

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="android/cnn_gru_builtins.tflite")
interpreter.allocate_tensors()

inp  = interpreter.get_input_details()
out  = interpreter.get_output_details()

# window: (1, 200, 3), float32, z-score normalised per window
window = np.random.randn(1, 200, 3).astype(np.float32)
interpreter.set_tensor(inp[0]["index"], window)
interpreter.invoke()
prob = float(interpreter.get_tensor(out[0]["index"])[0, 0])
print(f"P(earthquake) = {prob:.4f}")
```
