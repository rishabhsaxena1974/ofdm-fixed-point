"""Run the C simulator for every experiment, compare with theory, and write plots and a
summary to results/."""
import io
import os
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc

from tables import N_FFT, N_DATA
from ref_model import tx_frame, rayleigh_taps, agc_quantize, fft_q15

ROOT = Path(__file__).resolve().parent.parent
EXE = ROOT / "build" / ("ofdm_sim.exe" if os.name == "nt" else "ofdm_sim")
RES = ROOT / "results"
BPS = {"qpsk": 2, "16qam": 4}


def sim(**kw):
    args = [str(EXE)] + [f"{k}={v}" for k, v in kw.items()]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    return np.genfromtxt(io.StringIO(out), delimiter=",", names=True, ndmin=1)


# ---------------------------------------------------------------- theory

def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))


def ber_awgn(mod, g):
    """g = Eb/N0 (linear). Exact Gray-mapped QPSK and 16-QAM."""
    if mod == "qpsk":
        return Q(np.sqrt(2 * g))
    a = np.sqrt(0.8 * g)
    return (3 * Q(a) + 2 * Q(3 * a) - Q(5 * a)) / 4


def ber_rayleigh(mod, ebn0_db):
    """Average the AWGN BER over the exponential SNR distribution of a Rayleigh subcarrier."""
    out = []
    for gm in 10 ** (np.atleast_1d(ebn0_db) / 10):
        f = lambda g: ber_awgn(mod, g) * np.exp(-g / gm) / gm
        out.append(quad(f, 0, np.inf, limit=200)[0])
    return np.array(out)


def snr_at_ber(snr, ber, target=1e-3):
    """Interpolate log10(BER) to find the Eb/N0 that reaches the target."""
    ok = ber > 0
    return float(np.interp(-np.log10(target), -np.log10(ber[ok]), snr[ok]))


def style(ax, title, xlabel="Eb/N0 (dB)"):
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("BER")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------- experiments

def fig_awgn():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for mod, color in (("qpsk", "C0"), ("16qam", "C3")):
        stop = 10 if mod == "qpsk" else 14
        ideal = sim(mod=mod, chan="awgn", est="ideal", snr=f"0:1:{stop}", bits=4_000_000)
        dft = sim(mod=mod, chan="awgn", est="dft", snr=f"0:1:{stop}", bits=4_000_000)
        s = np.linspace(0, stop, 200)
        ax.plot(s, ber_awgn(mod, 10 ** (s / 10)), color=color, lw=1, label=f"{mod.upper()} theory")
        ax.plot(ideal["ebn0_db"], ideal["ber_float"], "o", color=color, mfc="none", label=f"{mod.upper()} sim, ideal CSI")
        ax.plot(dft["ebn0_db"], dft["ber_float"], "x--", color=color, lw=0.8, label=f"{mod.upper()} sim, LS + DFT estimate")
    ax.set_ylim(1e-6, 0.5)
    style(ax, "AWGN: simulation vs. theory")
    fig.tight_layout()
    fig.savefig(RES / "1_awgn_validation.png", dpi=150)


def fig_rayleigh():
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for mod, color in (("qpsk", "C0"), ("16qam", "C3")):
        run = lambda est: sim(mod=mod, chan="rayleigh", taps=8, tau=2, est=est, snr="0:3:30", bits=3_000_000)
        ideal, dft, lin = run("ideal"), run("dft"), run("linear")
        s = np.linspace(0, 30, 61)
        ax.plot(s, ber_rayleigh(mod, s), color=color, lw=1, label=f"{mod.upper()} Rayleigh theory")
        ax.plot(ideal["ebn0_db"], ideal["ber_float"], "o", color=color, mfc="none", label=f"{mod.upper()} ideal CSI")
        ax.plot(dft["ebn0_db"], dft["ber_float"], "x--", color=color, lw=0.8, label=f"{mod.upper()} LS + DFT")
        ax.plot(lin["ebn0_db"], lin["ber_float"], "^:", color=color, lw=0.8, ms=4, label=f"{mod.upper()} LS + linear")
    style(ax, "8-tap Rayleigh multipath (exp. PDP, CP = 16)")
    fig.tight_layout()
    fig.savefig(RES / "2_rayleigh_multipath.png", dpi=150)


def fig_cp(cp=16, snr=30):
    lengths = [1, 2, 4, 8, 12, 16, 17, 18, 20, 24, 28, 32]
    ber = {"ideal": [], "dft": [], "linear": []}
    for L in lengths:
        for est in ber:
            r = sim(mod="16qam", chan="rayleigh", taps=L, tau=0, cp=cp, est=est,
                    snr=f"{snr}:1:{snr}", bits=4_000_000)
            ber[est].append(r["ber_float"][0])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(lengths, ber["ideal"], "o-", label="ideal CSI")
    ax.plot(lengths, ber["dft"], "x--", label="LS + DFT interpolation")
    ax.plot(lengths, ber["linear"], "^:", label="LS + linear interpolation")
    ax.axvline(cp + 1, color="k", ls=":", lw=1, label=f"channel length = CP + 1 = {cp + 1}")
    style(ax, f"16-QAM, Eb/N0 = {snr} dB: CP vs. channel delay spread",
          xlabel="channel length L (taps, uniform power)")
    fig.tight_layout()
    fig.savefig(RES / "3_cp_vs_delay_spread.png", dpi=150)
    return lengths, ber


def fig_fixed_vs_float():
    awgn = sim(mod="16qam", chan="awgn", est="dft", snr="4:0.5:16", bits=6_000_000, errors=2000)
    ray = sim(mod="16qam", chan="rayleigh", taps=8, tau=2, est="dft", snr="0:3:30", bits=3_000_000)
    loss = (snr_at_ber(awgn["ebn0_db"], awgn["ber_q15"])
            - snr_at_ber(awgn["ebn0_db"], awgn["ber_float"]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, r, title in ((axes[0], awgn, "AWGN"), (axes[1], ray, "8-tap Rayleigh")):
        ax.plot(r["ebn0_db"], r["ber_float"], "o-", mfc="none", label="double-precision receiver")
        ax.plot(r["ebn0_db"], r["ber_q15"], "x--", label="Q15 receiver (12 dB backoff)")
        style(ax, f"16-QAM, LS + DFT estimation, {title}")
    axes[0].text(0.03, 0.05, f"implementation loss @ BER 1e-3: {loss:.3f} dB",
                 transform=axes[0].transAxes, fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / "4_fixed_vs_float.png", dpi=150)
    return loss


def rx_sqnr(backoff_db, nsym=400, seed=3):
    """SQNR at the Q15 FFT output against an infinite-precision receiver with the same gain.
    Captures ADC clipping, ADC quantisation and FFT rounding together (noise-free input)."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (nsym, N_DATA * 4), dtype=np.uint8)
    x = tx_frame(bits, 4, 16)
    y = np.convolve(x, rayleigh_taps(rng, 8, 2.0))[: len(x)]
    qr, qi, g = agc_quantize(y, backoff_db)
    Yr, Yi = fft_q15(qr.reshape(nsym, -1)[:, 16:], qi.reshape(nsym, -1)[:, 16:])
    ref = np.fft.fft((y * g).reshape(nsym, -1)[:, 16:], axis=1) / N_FFT
    err = (Yr + 1j * Yi) - ref
    return 10 * np.log10(np.sum(np.abs(ref) ** 2) / np.sum(np.abs(err) ** 2))


def fig_backoff(ebn0=12):
    backoffs = np.arange(0, 67, 3)
    ber_q, ber_f = [], []
    for bo in backoffs:
        r = sim(mod="16qam", chan="awgn", est="dft", backoff=bo, snr=f"{ebn0}:1:{ebn0}", bits=2_000_000, errors=3000)
        ber_q.append(r["ber_q15"][0])
        ber_f.append(r["ber_float"][0])
    sqnr = [rx_sqnr(bo) for bo in backoffs]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(backoffs, ber_q, "x-", label="Q15 receiver")
    axes[0].axhline(np.mean(ber_f), color="k", ls=":", label="double-precision receiver")
    style(axes[0], f"16-QAM, AWGN, Eb/N0 = {ebn0} dB", xlabel="AGC backoff below full scale (dB)")
    axes[1].plot(backoffs, sqnr, "o-")
    axes[1].set_xlabel("AGC backoff below full scale (dB)")
    axes[1].set_ylabel("SQNR at FFT output (dB)")
    axes[1].set_title("Fixed-point SQNR: clipping vs. quantisation")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "5_backoff_sweep.png", dpi=150)
    return backoffs, sqnr, ber_q


def main():
    RES.mkdir(exist_ok=True)
    fig_awgn()
    fig_rayleigh()
    lengths, ber_cp = fig_cp()
    loss = fig_fixed_vs_float()
    backoffs, sqnr, ber_q = fig_backoff()

    best = int(np.argmax(sqnr))
    lines = [
        f"Q15 implementation loss, 16-QAM AWGN, BER 1e-3: {loss:.3f} dB",
        f"SQNR at FFT output, 12 dB backoff: {rx_sqnr(12):.1f} dB",
        f"Best SQNR {sqnr[best]:.1f} dB at {backoffs[best]} dB backoff",
        "Backoff sweep (dB, SQNR dB, BER Q15): "
        + "; ".join(f"{b},{s:.1f},{q:.2e}" for b, s, q in zip(backoffs, sqnr, ber_q)),
        "CP=16 sweep, 16-QAM 30 dB (L, BER ideal, BER DFT, BER linear): "
        + "; ".join(f"{L},{i:.2e},{d:.2e},{l:.2e}" for L, i, d, l
                    in zip(lengths, ber_cp["ideal"], ber_cp["dft"], ber_cp["linear"])),
    ]
    (RES / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
