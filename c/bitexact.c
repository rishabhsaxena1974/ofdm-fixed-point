/* bitexact.c - run the C Q15 receiver on vectors written by python/make_vectors.py and
 * compare every FFT output, channel estimate and decided bit with the Python model.
 *
 * usage: bitexact [vector_dir]      exit code 0 = bit-exact */
#include <stdio.h>
#include <stdlib.h>
#include "ofdm.h"

static void *load(const char *dir, const char *name, size_t size)
{
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    void *buf = malloc(size);
    if (!f || !buf || fread(buf, 1, size, f) != size) {
        fprintf(stderr, "cannot read %lu bytes from %s\n", (unsigned long)size, path);
        exit(2);
    }
    fclose(f);
    return buf;
}

static long diff_q15(const cq15 *a, const cq15 *b, int n)
{
    long d = 0;
    for (int i = 0; i < n; i++)
        d += a[i].re != b[i].re || a[i].im != b[i].im;
    return d;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "vectors";
    char path[512];
    int nsym, cp, bps, est;

    snprintf(path, sizeof path, "%s/meta.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f || fscanf(f, "%d %d %d %d", &nsym, &cp, &bps, &est) != 4) {
        fprintf(stderr, "cannot read %s\n", path);
        return 2;
    }
    fclose(f);

    const int sym_len = N_FFT + cp, nbits = N_DATA * bps;
    cq15 *rx       = load(dir, "rx_q15.bin",   (size_t)nsym * sym_len * sizeof(cq15));
    cq15 *exp_fft  = load(dir, "exp_fft.bin",  (size_t)nsym * N_FFT * sizeof(cq15));
    cq15 *exp_h    = load(dir, "exp_h.bin",    (size_t)nsym * N_FFT * sizeof(cq15));
    uint8_t *exp_b = load(dir, "exp_bits.bin", (size_t)nsym * nbits);

    long bad_fft = 0, bad_h = 0, bad_bits = 0;
    for (int s = 0; s < nsym; s++) {
        cq15 fft[N_FFT], h[N_FFT];
        uint8_t bits[N_DATA * 4];
        rx_q15(rx + s * sym_len + cp, est, bits, bps, fft, h);
        bad_fft += diff_q15(fft, exp_fft + s * N_FFT, N_FFT);
        bad_h   += diff_q15(h, exp_h + s * N_FFT, N_FFT);
        for (int i = 0; i < nbits; i++)
            bad_bits += bits[i] != exp_b[s * nbits + i];
    }

    printf("%s estimator, %d symbols (%d FFT bins, %d channel estimates, %d bits)\n",
           est == EST_LINEAR ? "linear" : "DFT", nsym, nsym * N_FFT, nsym * N_FFT, nsym * nbits);
    printf("mismatches: fft=%ld  h=%ld  bits=%ld\n", bad_fft, bad_h, bad_bits);
    int pass = !bad_fft && !bad_h && !bad_bits;
    printf("%s\n", pass ? "PASS: C receiver is bit-exact with the Python model" : "FAIL");
    return pass ? 0 : 1;
}
