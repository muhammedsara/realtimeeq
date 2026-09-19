"""
Baseline 1D-CNN — Edge-AI Based Seismic Detection Framework
============================================================
Lightweight mobile-compatible 1D-CNN baseline (Phase 1).

Architecture:
    Input(200, 3) → [Conv1D → BN → ReLU → MaxPool → Dropout] × 4
                  → GlobalAvgPool → Dense → sigmoid

Target:
    - Input : (batch, 200, 3)  — 2 s window @ 100 Hz, 3-axis accelerometer
    - Output: (batch, 1)       — sigmoid P(earthquake)
    - Params: 15K–50K          — TFLite INT8 budget

References:
    Kong et al. (2019) "MyShake" — Smartphone seismology for EEW
    Perol et al. (2018) "ConvNetQuake" — 1D CNN for seismic detection
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers


def build_baseline_cnn(
    input_shape: tuple = (200, 3),
    filters: list = None,
    kernel_sizes: list = None,
    pool_sizes: list = None,
    dropout_rate: float = 0.4,
    dense_units: int = 32,
    l2_reg: float = 1e-4,
    name: str = "BaselineCNN_1D",
) -> keras.Model:
    """
    Lightweight 1D-CNN baseline model.

    Parameters
    ----------
    input_shape : tuple
        Window shape (time_steps, channels) = (200, 3).
    filters : list
        Filters per Conv1D block. Default: [16, 32, 64, 64].
    kernel_sizes : list
        Kernel sizes per block. Default: [7, 5, 5, 3].
    pool_sizes : list
        MaxPool sizes per block. Default: [2, 2, 2, 2].
    dropout_rate : float
        Spatial dropout rate (0.3–0.5 recommended).
    dense_units : int
        Hidden Dense layer units.
    l2_reg : float
        L2 regularisation coefficient.

    Returns
    -------
    keras.Model
    """
    if filters is None:
        filters = [16, 32, 64, 64]
    if kernel_sizes is None:
        kernel_sizes = [7, 5, 5, 3]
    if pool_sizes is None:
        pool_sizes = [2, 2, 2, 2]

    assert len(filters) == len(kernel_sizes) == len(pool_sizes)

    inputs = layers.Input(shape=input_shape, name="accelerometer_input")
    x = inputs

    for i, (n_filters, k_size, p_size) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
        x = layers.Conv1D(
            n_filters, k_size, padding="same",
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"conv1d_{i+1}",
        )(x)
        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Activation("relu", name=f"relu_{i+1}")(x)
        x = layers.MaxPooling1D(pool_size=p_size, name=f"maxpool_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)
    x = layers.Dense(
        dense_units, activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_hidden",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_dense")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="eq_probability")(x)

    return Model(inputs=inputs, outputs=outputs, name=name)


def compile_model(model: keras.Model, learning_rate: float = 1e-3) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc", curve="ROC"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


if __name__ == "__main__":
    model = compile_model(build_baseline_cnn())
    model.summary()
