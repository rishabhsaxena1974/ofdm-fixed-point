/* sim.c - Monte-Carlo BER of the float and Q15 receivers on identical channel and noise
 * realisations, so the gap between them is the fixed-point implementation loss.
 *
 * usage: ofdm_sim [key=value ...]
 *   mod=qpsk|16qam  chan=awgn|rayleigh  taps=8  tau=2 (<=0: uniform PDP)  cp=16
 *   est=dft|linear|ideal (ideal = true H for the float receiver, DFT estimation for Q15)
 *   backoff=12 (AGC target, dB below ADC full scale)
 *   snr=start:step:stop (Eb/N0 in dB)  seed=1
 *   minbits=1000000  bits=2000000  errors=300: run at least minbits (so fading is averaged
 *   over many channels) and at most bits, stopping early once both receivers hit errors */
/* Output: CSV on stdout. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ofdm.h"

#define FRAME_LEN (SYMS_PER_FRAME * (N_FFT + MAX_CP))

typedef struct {
    int bps, awgn, taps, cp, ideal, est;
    double tau, backoff_db, snr_start, snr_step, snr_stop;
    long min_bits, max_bits, min_errors;
    unsigned long seed;
} params;

static cplx tx[FRAME_LEN], rx[FRAME_LEN];
static cq15 rxq[FRAME_LEN];

static int parse(params *p, int argc, char **argv)
{
    *p = (params){ .bps = 4, .taps = 8, .cp = 16, .est = EST_DFT, .tau = 2.0, .backoff_db = 12.0,
                   .snr_start = 0, .snr_step = 2, .snr_stop = 20,
                   .min_bits = 1000000, .max_bits = 2000000, .min_errors = 300, .seed = 1 };
    for (int i = 1; i < argc; i++) {
        char *eq = strchr(argv[i], '=');
        if (!eq) return -1;
        *eq = '\0';
        const char *k = argv[i], *v = eq + 1;
        if      (!strcmp(k, "mod"))     p->bps = !strcmp(v, "qpsk") ? 2 : !strcmp(v, "16qam") ? 4 : -1;
        else if (!strcmp(k, "chan"))    p->awgn = !strcmp(v, "awgn");
        else if (!strcmp(k, "taps"))    p->taps = atoi(v);
        else if (!strcmp(k, "tau"))     p->tau = atof(v);
        else if (!strcmp(k, "cp"))      p->cp = atoi(v);
        else if (!strcmp(k, "est")) {
            p->ideal = !strcmp(v, "ideal");
            p->est = !strcmp(v, "linear") ? EST_LINEAR : EST_DFT;
        }
        else if (!strcmp(k, "backoff")) p->backoff_db = atof(v);
        else if (!strcmp(k, "minbits")) p->min_bits = atol(v);
        else if (!strcmp(k, "bits"))    p->max_bits = atol(v);
        else if (!strcmp(k, "errors"))  p->min_errors = atol(v);
        else if (!strcmp(k, "seed"))    p->seed = strtoul(v, NULL, 10);
        else if (!strcmp(k, "snr")) {
            if (sscanf(v, "%lf:%lf:%lf", &p->snr_start, &p->snr_step, &p->snr_stop) != 3) return -1;
        } else return -1;
    }
    if (p->bps < 0 || p->cp < 0 || p->cp > MAX_CP || p->taps < 1 || p->taps > MAX_TAPS || p->snr_step <= 0)
        return -1;
    return 0;
}

static int16_t quantize(double v)
{
    double r = floor(v + 0.5);
    if (r > 32767.0)  return 32767;
    if (r < -32768.0) return -32768;
    return (int16_t)r;
}

static long count_errors(const uint8_t *a, const uint8_t *b, int n)
{
    long e = 0;
    for (int i = 0; i < n; i++)
        e += a[i] != b[i];
    return e;
}

int main(int argc, char **argv)
{
    params p;
    if (parse(&p, argc, argv)) {
        fprintf(stderr, "bad arguments, see the header of sim.c\n");
        return 1;
    }
    rng_seed(p.seed);

    const int sym_len = N_FFT + p.cp, frame_len = SYMS_PER_FRAME * sym_len;
    const int bits_per_ofdm = N_DATA * p.bps, nbits = SYMS_PER_FRAME * bits_per_ofdm;
    const int L = p.awgn ? 1 : p.taps;
    uint8_t bits[SYMS_PER_FRAME * N_DATA * 4], hat[N_DATA * 4];
    cplx data[N_DATA], h[MAX_TAPS], hf[N_FFT];
    double pdp[MAX_TAPS], pdp_sum = 0;

    for (int l = 0; l < L; l++) {
        pdp[l] = p.tau > 0 ? exp(-l / p.tau) : 1.0;
        pdp_sum += pdp[l];
    }
    for (int l = 0; l < L; l++)
        pdp[l] /= pdp_sum;

    printf("ebn0_db,bits,err_float,err_q15,ber_float,ber_q15\n");
    for (double snr = p.snr_start; snr <= p.snr_stop + 1e-9; snr += p.snr_step) {
        /* unit signal power per subcarrier, so N0 = 1 / (bps * Eb/N0) */
        const double sigma = sqrt(1.0 / (p.bps * pow(10.0, snr / 10.0)) / 2.0);
        long nb = 0, err_f = 0, err_q = 0;

        while (nb < p.max_bits &&
               (nb < p.min_bits || err_f < p.min_errors || err_q < p.min_errors)) {
            for (int i = 0; i < nbits; i++)
                bits[i] = (uint8_t)(rng_u64() >> 63);
            for (int s = 0; s < SYMS_PER_FRAME; s++) {
                map_symbols(bits + s * bits_per_ofdm, data, N_DATA, p.bps);
                ofdm_tx_symbol(data, tx + s * sym_len, p.cp);
            }

            /* block fading: one Rayleigh realisation per frame */
            for (int l = 0; l < L; l++) {
                if (p.awgn) {
                    h[l].re = 1.0;
                    h[l].im = 0.0;
                } else {
                    double a = sqrt(pdp[l] / 2.0);
                    h[l].re = a * rng_gauss();
                    h[l].im = a * rng_gauss();
                }
            }
            memset(hf, 0, sizeof hf);
            memcpy(hf, h, L * sizeof(cplx));
            fft_float(hf);

            /* multipath convolution + AWGN */
            for (int n = 0; n < frame_len; n++) {
                double re = 0, im = 0;
                for (int l = 0; l < L && l <= n; l++) {
                    re += h[l].re * tx[n - l].re - h[l].im * tx[n - l].im;
                    im += h[l].re * tx[n - l].im + h[l].im * tx[n - l].re;
                }
                rx[n].re = re + sigma * rng_gauss();
                rx[n].im = im + sigma * rng_gauss();
            }

            /* float receiver */
            for (int s = 0; s < SYMS_PER_FRAME; s++) {
                rx_float(rx + s * sym_len + p.cp, p.ideal ? hf : NULL, p.est, hat, p.bps);
                err_f += count_errors(hat, bits + s * bits_per_ofdm, bits_per_ofdm);
            }

            /* Q15 receiver: AGC puts the RMS backoff_db below full scale, then a 16-bit ADC */
            double pw = 0;
            for (int n = 0; n < frame_len; n++)
                pw += rx[n].re * rx[n].re + rx[n].im * rx[n].im;
            const double g = 32768.0 * pow(10.0, -p.backoff_db / 20.0) / sqrt(pw / frame_len);
            for (int n = 0; n < frame_len; n++) {
                rxq[n].re = quantize(rx[n].re * g);
                rxq[n].im = quantize(rx[n].im * g);
            }
            for (int s = 0; s < SYMS_PER_FRAME; s++) {
                rx_q15(rxq + s * sym_len + p.cp, p.est, hat, p.bps, NULL, NULL);
                err_q += count_errors(hat, bits + s * bits_per_ofdm, bits_per_ofdm);
            }
            nb += nbits;
        }
        printf("%.2f,%ld,%ld,%ld,%.6e,%.6e\n", snr, nb, err_f, err_q,
               (double)err_f / nb, (double)err_q / nb);
        fflush(stdout);
    }
    return 0;
}
