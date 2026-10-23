"""Constants and ROM tables shared by the C code (via gen_tables.py) and the Python models."""
import numpy as np

N_FFT = 64
PILOT_SPACING = 4                      # comb pilots on k = 0, 4, 8, ...
N_PILOT = N_FFT // PILOT_SPACING       # 16
N_DATA = N_FFT - N_PILOT               # 48
PILOT_IDX = np.arange(0, N_FFT, PILOT_SPACING)
DATA_IDX = np.array([k for k in range(N_FFT) if k % PILOT_SPACING])
QAM16_THR_Q15 = 20724                  # round(2/sqrt(10) * 2^15), must match ofdm.h


def twiddles_q15():
    """W_N^n = exp(-j*2*pi*n/N) for n = 0..N/2-1, rounded to Q15."""
    n = np.arange(N_FFT // 2)
    ang = 2 * np.pi * n / N_FFT
    re = np.floor(np.cos(ang) * 32767 + 0.5).astype(np.int64)
    im = np.floor(-np.sin(ang) * 32767 + 0.5).astype(np.int64)
    return re, im


def pilot_signs():
    """Fixed pseudo-random BPSK (+1/-1) pilot sequence."""
    rng = np.random.default_rng(2026)
    return rng.choice(np.array([-1, 1]), N_PILOT).astype(np.int64)
