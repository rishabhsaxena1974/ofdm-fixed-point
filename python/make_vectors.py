"""Write test vectors for c/bitexact.c: a received Q15 frame plus the Python model's
FFT outputs, channel estimates and decided bits."""
import argparse
from pathlib import Path

import numpy as np

from tables import N_FFT, N_DATA
from ref_model import tx_frame, rayleigh_taps, agc_quantize, rx_q15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsym", type=int, default=500)
    ap.add_argument("--cp", type=int, default=16)
    ap.add_argument("--mod", choices=["qpsk", "16qam"], default="16qam")
    ap.add_argument("--ebn0", type=float, default=15.0)
    ap.add_argument("--taps", type=int, default=8)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--backoff", type=float, default=12.0)
    ap.add_argument("--est", choices=["dft", "linear"], default="dft")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "vectors"))
    a = ap.parse_args()

    bps = 2 if a.mod == "qpsk" else 4
    rng = np.random.default_rng(a.seed)
    bits = rng.integers(0, 2, (a.nsym, N_DATA * bps), dtype=np.uint8)
    x = tx_frame(bits, bps, a.cp)
    y = np.convolve(x, rayleigh_taps(rng, a.taps, a.tau))[: len(x)]
    n0 = 1 / (bps * 10 ** (a.ebn0 / 10))
    y += np.sqrt(n0 / 2) * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
    qr, qi, _ = agc_quantize(y, a.backoff)

    td_r = qr.reshape(a.nsym, -1)[:, a.cp:]
    td_i = qi.reshape(a.nsym, -1)[:, a.cp:]
    (Yr, Yi), (Hr, Hi), bits_hat = rx_q15(td_r, td_i, bps, a.est)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    interleave = lambda r, i: np.stack([r, i], axis=-1).astype("<i2").tobytes()
    (out / "rx_q15.bin").write_bytes(interleave(qr, qi))
    (out / "exp_fft.bin").write_bytes(interleave(Yr, Yi))
    (out / "exp_h.bin").write_bytes(interleave(Hr, Hi))
    (out / "exp_bits.bin").write_bytes(bits_hat.astype(np.uint8).tobytes())
    # estimator ids match the EST_* enum in ofdm.h
    (out / "meta.txt").write_text(f"{a.nsym} {a.cp} {bps} {0 if a.est == 'linear' else 1}\n")

    clipped = np.mean((np.abs(qr) >= 32767) | (np.abs(qi) >= 32767))
    print(f"wrote {a.nsym} symbols to {out}")
    print(f"reference-model BER {np.mean(bits_hat != bits):.3e}, clipped samples {clipped:.2e}")


if __name__ == "__main__":
    main()
