"""
Dilated Temporal Convolutional Network (TCN) — Edge-AI Based Seismic Detection Framework
=========================================================================================
Phase 3 architecture. Fully parallelisable dilated causal convolutions.

Architecture:
    Input(200, 3) → 1×1 projection
                  → [TCN Residual Block × 4] (dilations: 1, 2, 4, 8)
                  → GlobalAvgPool → Dense → sigmoid

Receptive field: 1 + 2 × (k−1) × Σ(dilations) = 1 + 2 × 2 × 15 = 61 timesteps = 610 ms @ 100 Hz

Key implementation note:
    padding='causal' causes CuDNN crashes on some GPU/XLA configurations.
    This implementation uses manual causal padding (ZeroPadding1D + padding='valid'),
    which is mathematically equivalent and avoids the CuDNN code path entirely.

Reference:
    Bai et al. (2018) "An Empirical Evaluation of Generic Convolutional and
    Recurrent Networks for Sequence Modeling"
"""

import tensorflow as tf
from tensorflow.keras import layers, regularizers


def _causal_conv1d(x, filters: int, kernel_size: int, dilation: int, l2: float, name: str):
    """
    Dilated causal convolution via manual left-padding + valid conv.

    Equivalent to Conv1D with padding='causal', but avoids CuDNN kernel conflicts
    that occur with dilated causal convolutions on some GPU configurations.

    Proof of equivalence:
        Causal: y[t] = Σ_{k=0}^{K-1} w[k] · x[t − d·k]
        Manual: x' = [0 × d*(K-1), x[0], ..., x[T-1]]
                y[t] = Σ_{k=0}^{K-1} w[k] · x'[t − d·k + d*(K-1)]  ← identical
    """
    pad = dilation * (kernel_size - 1)
    x = layers.ZeroPadding1D(padding=(pad, 0), name=f"{name}_pad")(x)
    x = layers.Conv1D(
        filters, kernel_size,
        dilation_rate=dilation,
        padding="valid",
        kernel_regularizer=regularizers.l2(l2),
        name=f"{name}_conv",
    )(x)
    return x


def _tcn_block(x, filters: int, kernel_size: int, dilation: int,
               dropout: float, l2: float, block_id: int):
    """
    TCN residual block: two dilated causal convolutions with residual connection.

    Structure:
        CausalConv → BN → ReLU → Dropout  (×2)
        + Residual (1×1 projection if channel mismatch)
    """
    residual = x
    nm = f"tcn_b{block_id}"

    x = _causal_conv1d(x, filters, kernel_size, dilation, l2, f"{nm}_c1")
    x = layers.BatchNormalization(name=f"{nm}_bn1")(x)
    x = layers.ReLU(name=f"{nm}_relu1")(x)
    x = layers.Dropout(dropout, name=f"{nm}_drop1")(x)

    x = _causal_conv1d(x, filters, kernel_size, dilation, l2, f"{nm}_c2")
    x = layers.BatchNormalization(name=f"{nm}_bn2")(x)
    x = layers.ReLU(name=f"{nm}_relu2")(x)
    x = layers.Dropout(dropout, name=f"{nm}_drop2")(x)

    if residual.shape[-1] != filters:
        residual = layers.Conv1D(filters, 1, name=f"{nm}_proj")(residual)

    x = layers.Add(name=f"{nm}_add")([x, residual])
    x = layers.ReLU(name=f"{nm}_out")(x)
    return x


def build_tcn(
    input_shape: tuple = (200, 3),
    filters: int = 64,
    kernel_size: int = 3,
    dilations: list = None,
    dropout: float = 0.3,
    dense_units: int = 32,
    l2: float = 1e-4,
    name: str = "TCN_Dilated",
) -> tf.keras.Model:
    """
    Dilated Temporal Convolutional Network for seismic detection.

    Parameters
    ----------
    input_shape : tuple
        (time_steps, channels) = (200, 3).
    filters : int
        Feature channels in each TCN block (64 by default → 103K params).
    kernel_size : int
        Convolutional kernel size.
    dilations : list
        Dilation rates per block. Default: [1, 2, 4, 8].
    dropout : float
        Dropout rate within each block.
    dense_units : int
        Classification head units.
    l2 : float
        L2 regularisation coefficient.

    Returns
    -------
    tf.keras.Model
    """
    if dilations is None:
        dilations = [1, 2, 4, 8]

    inp = tf.keras.Input(shape=input_shape, name="accelerometer_input")
    x = layers.Conv1D(filters, 1, padding="same", name="input_proj")(inp)

    for i, d in enumerate(dilations):
        x = _tcn_block(x, filters, kernel_size, d, dropout, l2, block_id=i + 1)

    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(
        dense_units, activation="relu",
        kernel_regularizer=regularizers.l2(l2),
        name="dense_hidden",
    )(x)
    x = layers.Dropout(dropout, name="dropout_dense")(x)
    out = layers.Dense(1, activation="sigmoid", name="eq_probability")(x)

    return tf.keras.Model(inputs=inp, outputs=out, name=name)


def compile_tcn(model: tf.keras.Model, learning_rate: float = 1e-3) -> tf.keras.Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc", curve="ROC"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


if __name__ == "__main__":
    model = compile_tcn(build_tcn())
    model.summary()
    print(f"\nTrainable params: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
