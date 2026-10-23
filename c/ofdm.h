/* ofdm.h - shared definitions for the OFDM baseband transceiver */
#ifndef OFDM_H
#define OFDM_H

#include <stdint.h>

#define N_FFT          64
#define LOG2_N         6
#define PILOT_SPACING  4                          /* comb pilots on k = 0, 4, 8, ... */
#define N_PILOT        (N_FFT / PILOT_SPACING)    /* 16 */
#define N_DATA         (N_FFT - N_PILOT)          /* 48 */
#define MAX_CP         64
#define MAX_TAPS       64
#define SYMS_PER_FRAME 10
#define QAM16_THR_Q15  20724                      /* 2/sqrt(10) in Q15: 16-QAM inner/outer threshold */

typedef struct { double re, im; } cplx;
typedef struct { int16_t re, im; } cq15;

/* ---- ROM tables, generated into tables_q15.c by python/gen_tables.py ---- */
extern const int16_t TW_RE[N_FFT / 2], TW_IM[N_FFT / 2];   /* exp(-j*2*pi*n/N) in Q15 */
extern const int8_t  PILOT_SIGN[N_PILOT];                   /* BPSK pilot sequence */

/* ---- random numbers (xorshift64*, Box-Muller) ---- */
void     rng_seed(uint64_t s);
uint64_t rng_u64(void);
double   rng_gauss(void);

/* ---- transforms (fft.c) ---- */
void    fft_float(cplx *x);       /* forward, unnormalised, in place */
void    ifft_unitary(cplx *x);    /* inverse, scaled by 1/sqrt(N) */
int16_t sat16(int32_t v);
void    fft_q15(cq15 *x);         /* forward, scaled by 1/2 per stage (1/N overall) */

/* ---- modulation and receivers (ofdm.c) ---- */
void map_symbols(const uint8_t *bits, cplx *sym, int nsym, int bps);
void ofdm_tx_symbol(const cplx *data, cplx *out, int cp);   /* N_DATA symbols -> N+cp samples */

/* Channel estimation: LS on the pilots, then interpolation across the data subcarriers. */
enum { EST_LINEAR = 0,   /* linear interpolation between neighbouring pilots */
       EST_DFT    = 1 }; /* IDFT pilots to a 16-tap impulse response, DFT back to 64 bins */

/* Both receivers take the N samples left after CP removal and write N_DATA*bps bits.
 * rx_float uses h_ideal (true frequency response) when non-NULL, else the estimator est. */
void rx_float(const cplx *td, const cplx *h_ideal, int est, uint8_t *bits, int bps);
/* fft_out / h_out (optional) expose intermediate results for bit-exact testing. */
void rx_q15(const cq15 *td, int est, uint8_t *bits, int bps, cq15 *fft_out, cq15 *h_out);

#endif
