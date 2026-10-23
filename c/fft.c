/* fft.c - radix-2 decimation-in-time FFTs in double precision and Q15 */
#include <math.h>
#include "ofdm.h"

#define PI 3.14159265358979323846

static int bit_reverse(int i)
{
    int r = 0;
    for (int b = 0; b < LOG2_N; b++) {
        r = (r << 1) | (i & 1);
        i >>= 1;
    }
    return r;
}

void fft_float(cplx *x)
{
    for (int i = 0; i < N_FFT; i++) {
        int j = bit_reverse(i);
        if (j > i) { cplx t = x[i]; x[i] = x[j]; x[j] = t; }
    }
    for (int m = 2; m <= N_FFT; m <<= 1) {
        int half = m >> 1;
        for (int j = 0; j < half; j++) {
            double wr = cos(-2.0 * PI * j / m);
            double wi = sin(-2.0 * PI * j / m);
            for (int k = 0; k < N_FFT; k += m) {
                cplx a = x[k + j], b = x[k + j + half];
                double tr = b.re * wr - b.im * wi;
                double ti = b.re * wi + b.im * wr;
                x[k + j].re        = a.re + tr;
                x[k + j].im        = a.im + ti;
                x[k + j + half].re = a.re - tr;
                x[k + j + half].im = a.im - ti;
            }
        }
    }
}

/* IFFT via the conjugation trick: ifft(X) = conj(fft(conj(X))) / N */
void ifft_unitary(cplx *x)
{
    const double s = 1.0 / sqrt((double)N_FFT);
    for (int i = 0; i < N_FFT; i++)
        x[i].im = -x[i].im;
    fft_float(x);
    for (int i = 0; i < N_FFT; i++) {
        x[i].re *= s;
        x[i].im *= -s;
    }
}

int16_t sat16(int32_t v)
{
    if (v > 32767)  return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

/* Overflow budget: |b*w| <= 32768 * 32767 * sqrt(2) < 2^31, so the products and their
 * sum fit in int32. Halving after every butterfly keeps each stage inside int16. */
void fft_q15(cq15 *x)
{
    for (int i = 0; i < N_FFT; i++) {
        int j = bit_reverse(i);
        if (j > i) { cq15 t = x[i]; x[i] = x[j]; x[j] = t; }
    }
    for (int m = 2; m <= N_FFT; m <<= 1) {
        int half = m >> 1, step = N_FFT / m;
        for (int j = 0; j < half; j++) {
            int32_t wr = TW_RE[j * step], wi = TW_IM[j * step];
            for (int k = 0; k < N_FFT; k += m) {
                cq15 a = x[k + j], b = x[k + j + half];
                /* Q15 x Q15 -> Q30, round back to Q15 */
                int32_t tr = (b.re * wr - b.im * wi + (1 << 14)) >> 15;
                int32_t ti = (b.re * wi + b.im * wr + (1 << 14)) >> 15;
                x[k + j].re        = sat16((a.re + tr + 1) >> 1);
                x[k + j].im        = sat16((a.im + ti + 1) >> 1);
                x[k + j + half].re = sat16((a.re - tr + 1) >> 1);
                x[k + j + half].im = sat16((a.im - ti + 1) >> 1);
            }
        }
    }
}
