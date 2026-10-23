"""Python models of the OFDM link.

The Q15 receiver here is a bit-exact mirror of c/fft.c and c/ofdm.c: same twiddles,
same rounding (add half an LSB, then arithmetic shift right) and same saturation.
numpy int64 holds the intermediates; that matches C because the C code is written so
its int32 intermediates never overflow.
"""
import numpy as np

from tables import (N_FFT, PILOT_SPACING, N_DATA, PILOT_IDX, DATA_IDX,
                    QAM16_THR_Q15, twiddles_q15, pilot_signs)

TW_RE, TW_IM = twiddles_q15()
PILOTS = pilot_signs()
LOG2_N = N_FFT.bit_length() - 1


# ---------------------------------------------------------------- transmitter / channel

def map_symbols(bits, bps):
    """Gray QPSK / 16-QAM with unit average energy. bits: (..., n*bps) -> (..., n)."""
    b = bits.reshape(*bits.shape[:-1], -1, bps).astype(float)
    sign = 2 * b - 1
    if bps == 2:
        return (sign[..., 0] + 1j * sign[..., 1]) / np.sqrt(2)
    level = lambda s, inner: s * np.where(inner > 0, 1, 3)
    return (level(sign[..., 0], b[..., 1]) + 1j * level(sign[..., 2], b[..., 3])) / np.sqrt(10)


def tx_frame(bits, bps, cp):
    """bits: (nsym, N_DATA*bps) -> serial baseband stream with cyclic prefixes."""
    X = np.zeros((bits.shape[0], N_FFT), complex)
    X[:, PILOT_IDX] = PILOTS
    X[:, DATA_IDX] = map_symbols(bits, bps)
    x = np.fft.ifft(X, axis=1) * np.sqrt(N_FFT)          # unitary IFFT, unit power
    return np.concatenate([x[:, N_FFT - cp:], x], axis=1).reshape(-1)


def rayleigh_taps(rng, taps, tau):
    """Exponential power-delay profile (uniform if tau <= 0), normalised to unit power."""
    l = np.arange(taps)
    pdp = np.exp(-l / tau) if tau > 0 else np.ones(taps)
    pdp /= pdp.sum()
    return np.sqrt(pdp / 2) * (rng.standard_normal(taps) + 1j * rng.standard_normal(taps))


def agc_quantize(y, backoff_db):
    """AGC sets the complex RMS backoff_db below full scale, then a 16-bit ADC (Q15)."""
    g = 32768 * 10 ** (-backoff_db / 20) / np.sqrt(np.mean(np.abs(y) ** 2))
    q = lambda v: np.clip(np.floor(v * g + 0.5), -32768, 32767).astype(np.int64)
    return q(y.real), q(y.imag), g


# ---------------------------------------------------------------- Q15 receiver (bit-exact)

def sat16(v):
    return np.clip(v, -32768, 32767)


def _bit_reverse_perm():
    return np.array([int(format(i, f"0{LOG2_N}b")[::-1], 2) for i in range(N_FFT)])


def fft_q15(re, im):
    """Radix-2 DIT FFT in Q15, scaled by 1/2 per stage (1/N overall). Works on (..., N)."""
    perm = _bit_reverse_perm()
    re = re[..., perm].astype(np.int64)
    im = im[..., perm].astype(np.int64)
    m = 2
    while m <= N_FFT:
        half, step = m // 2, N_FFT // m
        for j in range(half):
            wr, wi = int(TW_RE[j * step]), int(TW_IM[j * step])
            top = np.arange(j, N_FFT, m)
            bot = top + half
            ar, ai, br, bi = re[..., top], im[..., top], re[..., bot], im[..., bot]
            tr = (br * wr - bi * wi + (1 << 14)) >> 15
            ti = (br * wi + bi * wr + (1 << 14)) >> 15
            re[..., top] = sat16((ar + tr + 1) >> 1)
            im[..., top] = sat16((ai + ti + 1) >> 1)
            re[..., bot] = sat16((ar - tr + 1) >> 1)
            im[..., bot] = sat16((ai - ti + 1) >> 1)
        m *= 2
    return re, im


def _twiddle(n):
    """exp(-j*2*pi*n/N) for any integer array n, from the half-circle table."""
    n = np.asarray(n) % N_FFT
    sign = np.where(n < N_FFT // 2, 1, -1)
    idx = n % (N_FFT // 2)
    return sign * TW_RE[idx], sign * TW_IM[idx]


def ls_estimate_q15(Yr, Yi, est="dft"):
    """LS estimate on +-1 pilots (a sign flip), then linear or DFT-based interpolation."""
    ar = sat16(Yr[..., PILOT_IDX] * PILOTS)
    ai = sat16(Yi[..., PILOT_IDX] * PILOTS)
    if est == "linear":
        br, bi = np.roll(ar, -1, axis=-1), np.roll(ai, -1, axis=-1)
        Hr, Hi = np.zeros_like(Yr), np.zeros_like(Yi)
        for d in range(PILOT_SPACING):
            Hr[..., PILOT_IDX + d] = ((PILOT_SPACING - d) * ar + d * br + 2) >> 2
            Hi[..., PILOT_IDX + d] = ((PILOT_SPACING - d) * ai + d * bi + 2) >> 2
        return Hr, Hi
    n_p = len(PILOT_IDX)
    # 16-point IDFT, Hp * conj(W): rows l, columns p
    pl = np.arange(n_p)[:, None] * np.arange(n_p)[None, :]
    wr, wi = _twiddle(PILOT_SPACING * pl)
    hr = sat16((ar @ wr.T + ai @ wi.T + (1 << 18)) >> 19)
    hi = sat16((ai @ wr.T - ar @ wi.T + (1 << 18)) >> 19)
    # 64-point DFT of the 16 taps: rows k, columns l
    wr, wi = _twiddle(np.arange(N_FFT)[:, None] * np.arange(n_p)[None, :])
    Hr = sat16((hr @ wr.T - hi @ wi.T + (1 << 14)) >> 15)
    Hi = sat16((hr @ wi.T + hi @ wr.T + (1 << 14)) >> 15)
    return Hr, Hi


def demap_q15(Yr, Yi, Hr, Hi, bps):
    """Division-free demapper: decisions on Y*conj(H) against thresholds scaled by |H|^2."""
    yr, yi, hr, hi = Yr[..., DATA_IDX], Yi[..., DATA_IDX], Hr[..., DATA_IDX], Hi[..., DATA_IDX]
    zr = yr * hr + yi * hi
    zi = yi * hr - yr * hi
    if bps == 2:
        bits = np.stack([zr > 0, zi > 0], axis=-1)
    else:
        thr = ((hr * hr + hi * hi) * QAM16_THR_Q15) >> 15
        bits = np.stack([zr > 0, np.abs(zr) < thr, zi > 0, np.abs(zi) < thr], axis=-1)
    return bits.reshape(*bits.shape[:-2], N_DATA * bps).astype(np.uint8)


def rx_q15(td_re, td_im, bps, est="dft"):
    """td_*: (nsym, N) int16 samples after CP removal."""
    Yr, Yi = fft_q15(td_re, td_im)
    Hr, Hi = ls_estimate_q15(Yr, Yi, est)
    return (Yr, Yi), (Hr, Hi), demap_q15(Yr, Yi, Hr, Hi, bps)
