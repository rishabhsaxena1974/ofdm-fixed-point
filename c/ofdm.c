/* ofdm.c - mapping, OFDM modulation, channel estimation and the float / Q15 receivers */
#include <math.h>
#include <string.h>
#include "ofdm.h"

#define PI      3.14159265358979323846
#define QPSK_A  0.70710678118654752   /* 1/sqrt(2) */
#define QAM16_A 0.31622776601683794   /* 1/sqrt(10) */

/* ---------------------------------------------------------------- random numbers */

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

void rng_seed(uint64_t s) { rng_state = s ? s : 0x9E3779B97F4A7C15ULL; }

uint64_t rng_u64(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 2685821657736338717ULL;
}

static double rng_uniform(void)   /* open interval (0, 1) */
{
    return ((double)(rng_u64() >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

double rng_gauss(void)
{
    return sqrt(-2.0 * log(rng_uniform())) * cos(2.0 * PI * rng_uniform());
}

/* ---------------------------------------------------------------- mapping */

/* Gray mapping. QPSK: one bit per axis gives the sign. 16-QAM: per axis the first bit
 * gives the sign and the second picks inner (1) or outer (3) level: -3,-1,+1,+3 = 00,01,11,10 */
void map_symbols(const uint8_t *b, cplx *s, int nsym, int bps)
{
    for (int i = 0; i < nsym; i++, b += bps) {
        if (bps == 2) {
            s[i].re = (b[0] ? 1 : -1) * QPSK_A;
            s[i].im = (b[1] ? 1 : -1) * QPSK_A;
        } else {
            s[i].re = (b[0] ? 1 : -1) * (b[1] ? 1 : 3) * QAM16_A;
            s[i].im = (b[2] ? 1 : -1) * (b[3] ? 1 : 3) * QAM16_A;
        }
    }
}

static void demap_float(const cplx *x, uint8_t *b, int nsym, int bps)
{
    for (int i = 0; i < nsym; i++, b += bps) {
        if (bps == 2) {
            b[0] = x[i].re > 0;
            b[1] = x[i].im > 0;
        } else {
            b[0] = x[i].re > 0;
            b[1] = fabs(x[i].re) < 2 * QAM16_A;
            b[2] = x[i].im > 0;
            b[3] = fabs(x[i].im) < 2 * QAM16_A;
        }
    }
}

/* ---------------------------------------------------------------- transmitter */

void ofdm_tx_symbol(const cplx *data, cplx *out, int cp)
{
    cplx X[N_FFT];
    int d = 0;
    for (int k = 0; k < N_FFT; k++) {
        if (k % PILOT_SPACING == 0) {
            X[k].re = PILOT_SIGN[k / PILOT_SPACING];
            X[k].im = 0.0;
        } else {
            X[k] = data[d++];
        }
    }
    ifft_unitary(X);
    for (int n = 0; n < cp; n++)
        out[n] = X[N_FFT - cp + n];
    memcpy(out + cp, X, sizeof X);
}

/* ---------------------------------------------------------------- float receiver */

/* Pilots are +-1, so the LS estimate Y/P is a sign flip.
 * EST_LINEAR: linear interpolation between pilots; the DFT is periodic in k, so the last
 *             gap wraps round to pilot 0. Breaks down when H rotates fast (long channels).
 * EST_DFT:    the 16 pilot estimates are a 16-point DFT of the channel impulse response,
 *             so a 16-point IDFT recovers h[0..15] and a 64-point DFT of h gives H on every
 *             subcarrier. Exact for any channel shorter than N / PILOT_SPACING = 16 taps. */
static void ls_estimate_float(const cplx *Y, int est, cplx *H)
{
    cplx Hp[N_PILOT], h[N_PILOT];
    for (int p = 0; p < N_PILOT; p++) {
        Hp[p].re = Y[p * PILOT_SPACING].re * PILOT_SIGN[p];
        Hp[p].im = Y[p * PILOT_SPACING].im * PILOT_SIGN[p];
    }
    if (est == EST_LINEAR) {
        for (int p = 0; p < N_PILOT; p++) {
            cplx a = Hp[p], b = Hp[(p + 1) % N_PILOT];
            for (int d = 0; d < PILOT_SPACING; d++) {
                H[p * PILOT_SPACING + d].re = ((PILOT_SPACING - d) * a.re + d * b.re) / PILOT_SPACING;
                H[p * PILOT_SPACING + d].im = ((PILOT_SPACING - d) * a.im + d * b.im) / PILOT_SPACING;
            }
        }
        return;
    }
    for (int l = 0; l < N_PILOT; l++) {
        h[l].re = h[l].im = 0.0;
        for (int p = 0; p < N_PILOT; p++) {
            double ang = 2.0 * PI * p * l / N_PILOT;
            h[l].re += (Hp[p].re * cos(ang) - Hp[p].im * sin(ang)) / N_PILOT;
            h[l].im += (Hp[p].re * sin(ang) + Hp[p].im * cos(ang)) / N_PILOT;
        }
    }
    for (int k = 0; k < N_FFT; k++) {
        H[k].re = H[k].im = 0.0;
        for (int l = 0; l < N_PILOT; l++) {
            double ang = -2.0 * PI * k * l / N_FFT;
            H[k].re += h[l].re * cos(ang) - h[l].im * sin(ang);
            H[k].im += h[l].re * sin(ang) + h[l].im * cos(ang);
        }
    }
}

void rx_float(const cplx *td, const cplx *h_ideal, int est, uint8_t *bits, int bps)
{
    cplx Y[N_FFT], H[N_FFT], eq[N_DATA];
    const double s = 1.0 / sqrt((double)N_FFT);

    memcpy(Y, td, sizeof Y);
    fft_float(Y);
    for (int k = 0; k < N_FFT; k++) {
        Y[k].re *= s;
        Y[k].im *= s;
    }
    if (h_ideal)
        memcpy(H, h_ideal, sizeof H);
    else
        ls_estimate_float(Y, est, H);

    /* one-tap zero-forcing equaliser: X = Y / H */
    int d = 0;
    for (int k = 0; k < N_FFT; k++) {
        if (k % PILOT_SPACING == 0)
            continue;
        double e = H[k].re * H[k].re + H[k].im * H[k].im;
        eq[d].re = (Y[k].re * H[k].re + Y[k].im * H[k].im) / e;
        eq[d].im = (Y[k].im * H[k].re - Y[k].re * H[k].im) / e;
        d++;
    }
    demap_float(eq, bits, N_DATA, bps);
}

/* ---------------------------------------------------------------- Q15 receiver */

static int64_t abs64(int64_t v) { return v < 0 ? -v : v; }

/* exp(-j*2*pi*n/N) for any n, from the half-circle table: W^(n+N/2) = -W^n */
static void twiddle(int n, int32_t *wr, int32_t *wi)
{
    n &= N_FFT - 1;
    if (n < N_FFT / 2) { *wr = TW_RE[n];               *wi = TW_IM[n]; }
    else               { *wr = -TW_RE[n - N_FFT / 2];  *wi = -TW_IM[n - N_FFT / 2]; }
}

static void ls_estimate_q15(const cq15 *Y, int est, cq15 *H)
{
    cq15 Hp[N_PILOT], h[N_PILOT];

    for (int p = 0; p < N_PILOT; p++) {
        cq15 y = Y[p * PILOT_SPACING];
        Hp[p].re = PILOT_SIGN[p] > 0 ? y.re : sat16(-y.re);
        Hp[p].im = PILOT_SIGN[p] > 0 ? y.im : sat16(-y.im);
    }

    if (est == EST_LINEAR) {
        /* weights (4-d)/4 and d/4: shifts and adds, no multiplier needed */
        for (int p = 0; p < N_PILOT; p++) {
            cq15 a = Hp[p], b = Hp[(p + 1) % N_PILOT];
            for (int d = 0; d < PILOT_SPACING; d++) {
                H[p * PILOT_SPACING + d].re = (int16_t)(((PILOT_SPACING - d) * a.re + d * b.re + 2) >> 2);
                H[p * PILOT_SPACING + d].im = (int16_t)(((PILOT_SPACING - d) * a.im + d * b.im + 2) >> 2);
            }
        }
        return;
    }

    /* 16-point IDFT: h[l] = 1/16 * sum_p Hp[p] * conj(W64^(4pl)). Products are Q15 x Q15,
     * accumulated in 64 bits, then one rounding shift removes Q15 (>>15) and the 1/16 (>>4). */
    for (int l = 0; l < N_PILOT; l++) {
        int64_t ar = 0, ai = 0;
        for (int p = 0; p < N_PILOT; p++) {
            int32_t wr, wi;
            twiddle(PILOT_SPACING * p * l, &wr, &wi);
            ar += (int64_t)Hp[p].re * wr + (int64_t)Hp[p].im * wi;
            ai += (int64_t)Hp[p].im * wr - (int64_t)Hp[p].re * wi;
        }
        h[l].re = sat16((int32_t)((ar + (1 << 18)) >> 19));
        h[l].im = sat16((int32_t)((ai + (1 << 18)) >> 19));
    }
    /* 64-point DFT of the 16 taps (zero padding implied) */
    for (int k = 0; k < N_FFT; k++) {
        int64_t ar = 0, ai = 0;
        for (int l = 0; l < N_PILOT; l++) {
            int32_t wr, wi;
            twiddle(k * l, &wr, &wi);
            ar += (int64_t)h[l].re * wr - (int64_t)h[l].im * wi;
            ai += (int64_t)h[l].re * wi + (int64_t)h[l].im * wr;
        }
        H[k].re = sat16((int32_t)((ar + (1 << 14)) >> 15));
        H[k].im = sat16((int32_t)((ai + (1 << 14)) >> 15));
    }
}

void rx_q15(const cq15 *td, int est, uint8_t *bits, int bps, cq15 *fft_out, cq15 *h_out)
{
    cq15 Y[N_FFT], H[N_FFT];

    memcpy(Y, td, sizeof Y);
    fft_q15(Y);
    ls_estimate_q15(Y, est, H);

    /* Division-free demapper. Z = Y*conj(H) = |H|^2 * X + noise, so instead of computing
     * X = Z/|H|^2 we compare Z against decision thresholds scaled by |H|^2.
     * The 64-bit accumulators play the role of a DSP's guard bits. */
    uint8_t *b = bits;
    for (int k = 0; k < N_FFT; k++) {
        if (k % PILOT_SPACING == 0)
            continue;
        int64_t zr = (int64_t)Y[k].re * H[k].re + (int64_t)Y[k].im * H[k].im;
        int64_t zi = (int64_t)Y[k].im * H[k].re - (int64_t)Y[k].re * H[k].im;
        if (bps == 2) {
            b[0] = zr > 0;
            b[1] = zi > 0;
        } else {
            int64_t e   = (int64_t)H[k].re * H[k].re + (int64_t)H[k].im * H[k].im;
            int64_t thr = (e * QAM16_THR_Q15) >> 15;
            b[0] = zr > 0;
            b[1] = abs64(zr) < thr;
            b[2] = zi > 0;
            b[3] = abs64(zi) < thr;
        }
        b += bps;
    }

    if (fft_out) memcpy(fft_out, Y, sizeof Y);
    if (h_out)   memcpy(h_out, H, sizeof H);
}
