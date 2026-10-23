# Fixed-Point OFDM Baseband Transceiver

A 64-subcarrier OFDM link written in C, with two receivers that run side by side on the
same channel and noise realisations:

- a **double-precision** receiver (the "algorithm" reference), and
- a **Q15 fixed-point** receiver (what would run on a DSP core or in firmware),

plus a **Python model of the Q15 receiver that is bit-exact with the C code**, which is how
PHY firmware is usually verified against a golden model.

```
 bits -> Gray QPSK/16-QAM -> insert pilots -> 64-pt IFFT -> add CP
                                                              |
                                          multipath Rayleigh + AWGN
                                                              |
        +-----------------------------------------------------+
        |                                                     |
   double precision                                 AGC -> 16-bit ADC (Q15)
   remove CP -> FFT                                 remove CP -> Q15 FFT (1/2 per stage)
   LS pilots -> interpolate H                       LS pilots -> interpolate H (Q15)
   ZF: X = Y / H -> demap                           division-free demapper
```

## System parameters

| Parameter | Value |
|---|---|
| FFT size | 64 |
| Pilots | comb, every 4th subcarrier (16 pilots, BPSK +-1), 48 data subcarriers |
| Cyclic prefix | 16 samples (configurable) |
| Modulation | Gray-mapped QPSK, 16-QAM, unit average energy |
| Channel | AWGN, or L-tap Rayleigh block fading with exponential or uniform power-delay profile |
| Channel estimation | LS on pilots + linear **or** DFT-based interpolation |
| Equalisation | one-tap zero forcing |
| Fixed-point format | 16-bit Q15 samples, int32 butterflies, 64-bit accumulators in the estimator and demapper |

## Fixed-point design decisions

- **Scaled FFT.** Each radix-2 butterfly halves its output (round, then shift right by 1),
  so the 64-point FFT is scaled by 1/64 overall and can never overflow. Twiddles are Q15,
  products are Q30 rounded back to Q15. Worst case `|b*w| <= 32768 * 32767 * sqrt(2) < 2^31`,
  so the int32 intermediates cannot overflow either.
- **Pilots are +-1**, so the LS estimate `Y/P` is just a sign flip: no division, no multiplier.
- **Linear interpolation** uses weights (4-d)/4 and d/4, which are shifts and adds.
- **DFT-based interpolation**: the 16 pilot estimates are a 16-point DFT of the channel
  impulse response. A 16-point IDFT gives h[0..15] and a 64-point DFT of those taps gives H
  on every subcarrier. It is exact for any channel shorter than 64/4 = 16 taps, which linear
  interpolation is not.
- **Division-free demapper.** Instead of computing `X = Y/H`, the receiver forms
  `Z = Y * conj(H) = |H|^2 X + noise` and compares Z against decision thresholds scaled by
  `|H|^2` (for 16-QAM the inner/outer threshold is `2/sqrt(10) * |H|^2`). Divisions are slow
  or absent on DSP cores; this needs only multiplies and compares.
- **AGC backoff.** The ADC input is scaled so the RMS sits a set number of dB below full scale.
  Too little backoff clips OFDM peaks (high PAPR); too much wastes bits to quantisation.

## Results

All numbers come from `python/experiments.py`, which writes `results/summary.txt`.

**1. AWGN validation** (`results/1_awgn_validation.png`): with ideal channel knowledge the
simulated BER lies on the closed-form curves for QPSK and 16-QAM. LS + DFT estimation costs
about 3 dB, which is expected: each pilot estimate is as noisy as a data subcarrier, so the
zero-forcing output carries roughly twice the noise.

**2. Multipath** (`results/2_rayleigh_multipath.png`, 8-tap exponential profile): ideal CSI
matches Rayleigh theory. Linear interpolation hits an error floor (16-QAM about 3e-2, QPSK about 1e-2)
because H rotates too fast between pilots. DFT interpolation removes the floor.

**3. Cyclic prefix vs. delay spread** (`results/3_cp_vs_delay_spread.png`, 16-QAM, 30 dB,
CP = 16, uniform profile). Two separate limits show up:

| Channel length L | ideal CSI | LS + DFT | LS + linear |
|---|---|---|---|
| 8  | 4.7e-4 | 8.8e-4 | 1.0e-1 |
| 16 | 5.3e-4 | 9.9e-4 | 3.7e-1 |
| 17 | 5.0e-4 | 1.4e-1 | 3.8e-1 |
| 18 | 2.6e-3 | 1.9e-1 | 4.0e-1 |
| 24 | 3.8e-2 | 3.3e-1 | 4.3e-1 |

- DFT estimation fails from L = 17: the pilots sample H every 4 subcarriers, so they can
  only resolve 64/4 = 16 taps (frequency-domain Nyquist).
- Even ideal CSI degrades from L = 18: the channel is longer than CP + 1, so the previous
  symbol leaks in (ISI) and orthogonality between subcarriers is lost (ICI).

**4. Fixed point vs. floating point** (`results/4_fixed_vs_float.png`, 16-QAM, LS + DFT):
the Q15 curves sit on top of the double-precision curves. Implementation loss at BER 1e-3
in AWGN: **0.002 dB**, within the simulation's resolution.

**5. AGC backoff sweep** (`results/5_backoff_sweep.png`): SQNR at the Q15 FFT output against
an infinite-precision receiver peaks at **60.4 dB at 12 dB backoff**. Below about 9 dB, clipping of
OFDM peaks dominates; above it, SQNR falls 1 dB per dB of backoff as the signal sinks
towards the LSB. BER stays at the floating-point value from 9 dB to about 30 dB of backoff.

**Bit-exactness**: `bitexact` runs the C Q15 receiver on vectors from the Python model and
compares every FFT bin, channel estimate and decided bit. Result for 500 symbols, both
estimators, QPSK and 16-QAM: **0 mismatches** (32 000 FFT bins, 32 000 channel estimates,
96 000 bits for 16-QAM).

## Build and run

Needs gcc and Python 3 with numpy, scipy and matplotlib.

```bash
make            # generates c/tables_q15.c and builds build/ofdm_sim, build/bitexact
make test       # bit-exact check, both estimators
make results    # all experiments -> results/
```

Without make (Windows):

```bash
python python/gen_tables.py
gcc -O2 -std=c99 -Wall -Wextra -o build/ofdm_sim.exe c/sim.c c/ofdm.c c/fft.c c/tables_q15.c -lm
gcc -O2 -std=c99 -Wall -Wextra -o build/bitexact.exe c/bitexact.c c/ofdm.c c/fft.c c/tables_q15.c -lm
python python/make_vectors.py && build/bitexact.exe vectors
python python/experiments.py
```

Single simulation run (CSV on stdout):

```bash
build/ofdm_sim mod=16qam chan=rayleigh taps=8 tau=2 est=dft backoff=12 snr=0:3:30
```

## Code map

| File | What it does |
|---|---|
| `c/ofdm.h` | constants, types, interfaces |
| `c/fft.c` | radix-2 DIT FFT in double and in Q15, unitary IFFT |
| `c/ofdm.c` | RNG, mapping, transmitter, LS + linear/DFT estimation, both receivers |
| `c/sim.c` | Monte-Carlo BER of both receivers on the same realisations |
| `c/bitexact.c` | compares the C Q15 receiver with the Python model |
| `c/tables_q15.c` | twiddle and pilot ROM tables, generated |
| `python/tables.py`, `gen_tables.py` | single source of the ROM tables |
| `python/ref_model.py` | transmitter/channel model and bit-exact Q15 receiver model |
| `python/make_vectors.py` | writes test vectors for `bitexact` |
| `python/experiments.py` | runs every experiment, theory curves, plots |

## Limitations and next steps

- Perfect timing and frequency sync are assumed. Next: PSS-style correlation for timing and
  CP-based carrier frequency offset estimation.
- No channel coding. Next: convolutional code + Viterbi, then compare coded BER.
- The DFT estimator keeps all 16 taps, so it interpolates but does not denoise. Zeroing taps
  beyond the expected delay spread would recover part of the 3 dB estimation loss.
- Block fading only (channel constant over a 10-symbol frame); no Doppler.
- Port the Q15 receiver to a Cortex-M (CMSIS-DSP) and measure cycles per symbol.
