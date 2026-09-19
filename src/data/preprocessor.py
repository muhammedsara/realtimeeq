"""
Signal Preprocessing Pipeline — Edge-AI Based Seismic Detection Framework
==============================
Filtreleme, normalizasyon, resampling, windowing ve kalite kontrol.
Tüm veri kaynaklarına uygulanacak ortak pipeline.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from typing import List, Optional, Tuple
import warnings


class SignalPreprocessor:
    """
    Sismik sinyal ön işleme pipeline'ı.

    Pipeline sırası:
        1. Resampling (hedef fs'ye)
        2. DC offset kaldırma (detrend)
        3. Bandpass filtre (0.5-20 Hz)
        4. Normalizasyon (z-score veya min-max)
        5. Windowing (kayan pencere)
        6. Kalite kontrol (SNR, PGA, clipping)
    """

    def __init__(
        self,
        target_sr: float = 100.0,
        filter_low: float = 0.5,
        filter_high: float = 20.0,
        filter_order: int = 4,
        normalization: str = "zscore",  # "zscore", "minmax", "none"
        window_size: int = 200,         # samples (2s @ 100Hz)
        window_stride: int = 100,       # samples (1s stride, %50 overlap)
    ):
        self.target_sr = target_sr
        self.filter_low = filter_low
        self.filter_high = filter_high
        self.filter_order = filter_order
        self.normalization = normalization
        self.window_size = window_size
        self.window_stride = window_stride

    # ─── 1. Resampling ──────────────────────────────────────────────────────

    def resample(self, data: np.ndarray, original_sr: float) -> np.ndarray:
        """
        Veriyi hedef örnekleme hızına yeniden örnekler.
        
        Parameters
        ----------
        data : np.ndarray
            Shape: (n_samples, n_channels)
        original_sr : float
            Orijinal örnekleme hızı (Hz).

        Returns
        -------
        np.ndarray
            Yeniden örneklenmiş veri.
        """
        if abs(original_sr - self.target_sr) < 1e-3:
            return data

        n_samples = data.shape[0]
        target_n = int(n_samples * self.target_sr / original_sr)

        if data.ndim == 1:
            return scipy_signal.resample(data, target_n)

        resampled = np.zeros((target_n, data.shape[1]), dtype=np.float32)
        for ch in range(data.shape[1]):
            resampled[:, ch] = scipy_signal.resample(data[:, ch], target_n)

        return resampled

    # ─── 2. Detrend (DC Offset Kaldırma) ─────────────────────────────────────

    @staticmethod
    def detrend(data: np.ndarray) -> np.ndarray:
        """Doğrusal trendi kaldırır."""
        from scipy.signal import detrend
        if data.ndim == 1:
            return detrend(data).astype(np.float32)
        return np.column_stack([
            detrend(data[:, ch]) for ch in range(data.shape[1])
        ]).astype(np.float32)

    # ─── 3. Bandpass Filtre ──────────────────────────────────────────────────

    def bandpass_filter(self, data: np.ndarray, sr: float = None) -> np.ndarray:
        """
        Butterworth bandpass filtresi uygular.
        
        Varsayılan: 0.5–20 Hz (sismik frekans bandı).
        """
        if sr is None:
            sr = self.target_sr

        nyquist = sr / 2.0
        low = self.filter_low / nyquist
        high = min(self.filter_high / nyquist, 0.99)

        if low >= high or low <= 0:
            warnings.warn(f"Geçersiz filtre parametreleri: low={low}, high={high}")
            return data

        try:
            b, a = scipy_signal.butter(self.filter_order, [low, high], btype="band")
            
            if data.ndim == 1:
                return scipy_signal.filtfilt(b, a, data).astype(np.float32)

            filtered = np.zeros_like(data, dtype=np.float32)
            for ch in range(data.shape[1]):
                filtered[:, ch] = scipy_signal.filtfilt(b, a, data[:, ch])
            return filtered

        except Exception as e:
            warnings.warn(f"Filtre hatası: {e}")
            return data

    # ─── 4. Normalizasyon ────────────────────────────────────────────────────

    def normalize(self, data: np.ndarray, method: str = None) -> np.ndarray:
        """
        Veriyi normalleştirir.
        
        Methods:
        - "zscore": (x - μ) / σ (pencere bazlı)
        - "minmax": (x - min) / (max - min)
        - "none": normalizasyon yok
        """
        if method is None:
            method = self.normalization

        if method == "none":
            return data

        if method == "zscore":
            mean = np.mean(data, axis=0, keepdims=True)
            std = np.std(data, axis=0, keepdims=True)
            std = np.where(std < 1e-10, 1.0, std)
            return ((data - mean) / std).astype(np.float32)

        if method == "minmax":
            dmin = np.min(data, axis=0, keepdims=True)
            dmax = np.max(data, axis=0, keepdims=True)
            drange = dmax - dmin
            drange = np.where(drange < 1e-10, 1.0, drange)
            return ((data - dmin) / drange).astype(np.float32)

        return data

    # ─── 5. Windowing ────────────────────────────────────────────────────────

    def create_windows(
        self,
        data: np.ndarray,
        window_size: int = None,
        stride: int = None,
    ) -> np.ndarray:
        """
        Kayan pencere segmentasyonu.

        Parameters
        ----------
        data : np.ndarray
            Shape: (n_samples, n_channels)
        window_size : int
            Pencere boyutu (sample).
        stride : int
            Adım boyutu (sample).

        Returns
        -------
        np.ndarray
            Shape: (n_windows, window_size, n_channels)
        """
        if window_size is None:
            window_size = self.window_size
        if stride is None:
            stride = self.window_stride

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_samples, n_channels = data.shape
        if n_samples < window_size:
            # Padding (sıfır ile doldur)
            padded = np.zeros((window_size, n_channels), dtype=np.float32)
            padded[:n_samples] = data
            return padded.reshape(1, window_size, n_channels)

        n_windows = (n_samples - window_size) // stride + 1
        windows = np.zeros((n_windows, window_size, n_channels), dtype=np.float32)

        for i in range(n_windows):
            start = i * stride
            windows[i] = data[start: start + window_size]

        return windows

    # ─── 6. Kalite Kontrol ───────────────────────────────────────────────────

    @staticmethod
    def calculate_snr(data: np.ndarray, noise_ratio: float = 0.3) -> float:
        """
        Sinyal/Gürültü Oranı (SNR) hesapla.
        İlk %noise_ratio'luk kısım gürültü, geri kalanı sinyal kabul edilir.
        """
        n_samples = data.shape[0]
        noise_end = int(n_samples * noise_ratio)
        
        if data.ndim > 1:
            # Vector magnitude
            data_vm = np.sqrt(np.sum(data ** 2, axis=1))
        else:
            data_vm = np.abs(data)

        noise_power = np.mean(data_vm[:noise_end] ** 2) + 1e-10
        signal_power = np.mean(data_vm[noise_end:] ** 2)

        snr = 10 * np.log10(signal_power / noise_power)
        return float(snr)

    @staticmethod
    def calculate_pga(data: np.ndarray) -> float:
        """Peak Ground Acceleration (g cinsinden veya m/s²)."""
        if data.ndim > 1:
            vm = np.sqrt(np.sum(data ** 2, axis=1))
        else:
            vm = np.abs(data)
        return float(np.max(vm))

    @staticmethod
    def detect_clipping(data: np.ndarray, threshold_ratio: float = 0.99) -> bool:
        """Sinyal doyma (clipping) tespiti."""
        max_val = np.max(np.abs(data))
        if max_val < 1e-10:
            return True  # tamamen sıfır → bozuk
        clip_threshold = max_val * threshold_ratio
        n_clipped = np.sum(np.abs(data) >= clip_threshold)
        # %1'den fazla sample cliplenmiş ise
        return (n_clipped / data.size) > 0.01

    @staticmethod
    def calculate_rms(data: np.ndarray) -> float:
        """Root Mean Square hesapla."""
        return float(np.sqrt(np.mean(data ** 2)))

    # ─── Full Pipeline ───────────────────────────────────────────────────────

    def process(
        self,
        data: np.ndarray,
        original_sr: float,
        apply_filter: bool = True,
        apply_normalize: bool = True,
    ) -> np.ndarray:
        """
        Tam ön işleme pipeline'ı uygular:
        resample → detrend → filter → normalize

        Parameters
        ----------
        data : np.ndarray
            Ham sinyal (n_samples, n_channels)
        original_sr : float
            Orijinal örnekleme hızı.

        Returns
        -------
        np.ndarray
            İşlenmiş sinyal.
        """
        # 1. Resample
        processed = self.resample(data, original_sr)

        # 2. Detrend
        processed = self.detrend(processed)

        # 3. Bandpass filter
        if apply_filter:
            processed = self.bandpass_filter(processed)

        # 4. Normalize
        if apply_normalize:
            processed = self.normalize(processed)

        return processed.astype(np.float32)

    def process_and_window(
        self,
        data: np.ndarray,
        original_sr: float,
        apply_filter: bool = True,
        apply_normalize: bool = True,
    ) -> np.ndarray:
        """
        Process + windowing: Tam pipeline → pencereler.

        Returns
        -------
        np.ndarray
            Shape: (n_windows, window_size, n_channels)
        """
        processed = self.process(data, original_sr, apply_filter, apply_normalize)
        return self.create_windows(processed)
