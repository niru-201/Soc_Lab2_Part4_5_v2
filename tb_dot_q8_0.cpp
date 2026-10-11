/*
 * Testbench for dot_q8_0. 
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dot_q8_0.h"
#include "vec_dot_q8_0.h"

/* FP16 bit patterns */
#define H_ZERO     0x0000
#define H_ONE      0x3C00
#define H_TWO      0x4000
#define H_HALF     0x3800
#define H_QUARTER  0x3400
#define H_MTWO     0xC000
#define H_MIN_SUB  0x0001   /* 2^-24, smallest denormal */
#define H_MAX      0x7BFF   /* 65504 */

static block_q8_0 X[DOT_MAX_NB], Y[DOT_MAX_NB];   /* llama.cpp layout - reference */
static int8_t     xq[DOT_MAX_N], yq[DOT_MAX_N];   /* design layout */
static half       xd[DOT_MAX_NB], yd[DOT_MAX_NB];
static float      xf[DOT_MAX_N], yf[DOT_MAX_N];

static int    n_tests, n_fail, n_exact;
static double max_abs_err, max_rel_err;

static void fill(block_q8_0 *b, int nb, uint16_t d, int8_t q)
{
    for (int i = 0; i < nb; i++) {
        b[i].d = d;
        for (int j = 0; j < QK8_0; j++)
            b[i].qs[j] = q;
    }
}

/* Copy the raw FP16 bits. Building the half from a float instead would go
 * through Vitis bit-accurate FP model, which flushes denormals to zero before
 * the design sees them. */
static half half_from_bits(uint16_t b)
{
    half h;
    memcpy(&h, &b, sizeof h);
    return h;
}

/* Runs reference and design on X/Y. nb_dut differs from nb only in the clamp test. */
static void run(const char *name, int nb, int nb_dut, bool verbose)
{
    for (int ib = 0; ib < nb; ib++) {
        xd[ib] = half_from_bits(X[ib].d);
        yd[ib] = half_from_bits(Y[ib].d);
        memcpy(&xq[ib * DOT_QK], X[ib].qs, DOT_QK);
        memcpy(&yq[ib * DOT_QK], Y[ib].qs, DOT_QK);
    }

    float ref;
    ggml_vec_dot_q8_0_q8_0(nb * QK8_0, &ref, 0, X, 0, Y, 0, 1);
    float dut = dot_q8_0(nb_dut, xq, xd, yq, yd);

    /* Tolerance is relative to sum(block terms), so sums that
     * cancel to 0 are not flagged. 1e-5 covers FP32 rounding of the sum in any
     * order over up to 152 blocks. */
    double mag = 0;
    for (int ib = 0; ib < nb; ib++) {
        int sumi = 0;
        for (int j = 0; j < QK8_0; j++)
            sumi += X[ib].qs[j] * Y[ib].qs[j];
        mag += fabs(sumi * (double)ggml_compute_fp16_to_fp32(X[ib].d)
                         * (double)ggml_compute_fp16_to_fp32(Y[ib].d));
    }

    double err  = fabs((double)dut - ref);
    double rel  = ref != 0 ? 100.0 * err / fabs(ref) : (err == 0 ? 0.0 : 100.0);
    bool   pass = err <= 1e-5 * mag;   /* false for NaN */

    n_tests++;
    n_fail  += !pass;
    n_exact += (dut == ref);
    if (err > max_abs_err) max_abs_err = err;
    if (rel > max_rel_err) max_rel_err = rel;

    if (verbose || !pass)
        printf("  [%s] %s: ref %.9g, dut %.9g, abs err %.2e, rel err %.2e%%\n",
               pass ? "PASS" : "FAIL", name, ref, dut, err, rel);
}

/* random data generated like llama.cpp does it */

/* xorshift64*: same sequence on every platform and in both co-sim runs */
static uint64_t rng_state;

static uint64_t rng_u64(void)
{
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1Dull;
}

static double rng_unif(void)
{
    return (rng_u64() >> 11) * (1.0 / 9007199254740992.0);
}

static double rng_gauss(void)
{
    double u1 = rng_unif() + 1e-300, u2 = rng_unif();
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* ggml_compute_fp32_to_fp16 (round to nearest even), bit-level so it keeps denormals */
static uint16_t fp32_to_fp16(float f)
{
    uint32_t w, bits;
    float scale_to_inf, scale_to_zero, base;
    const uint32_t c_inf = 0x77800000, c_zero = 0x08800000;
    memcpy(&scale_to_inf, &c_inf, 4);
    memcpy(&scale_to_zero, &c_zero, 4);
    memcpy(&w, &f, 4);

    base = (fabsf(f) * scale_to_inf) * scale_to_zero;
    uint32_t shl1_w = w + w;
    uint32_t sign   = w & 0x80000000;
    uint32_t bias   = shl1_w & 0xFF000000;
    if (bias < 0x71000000)
        bias = 0x71000000;

    uint32_t b2 = (bias >> 1) + 0x07800000;
    float f2;
    memcpy(&f2, &b2, 4);
    base = f2 + base;
    memcpy(&bits, &base, 4);
    uint32_t nonsign = ((bits >> 13) & 0x7C00) + (bits & 0x0FFF);
    return (uint16_t)((sign >> 16) | (shl1_w > 0xFF000000 ? 0x7E00 : nonsign));
}

/* quantize_row_q8_0_ref from ggml-quants.c */
static void quantize(const float *v, block_q8_0 *b, int n)
{
    for (int i = 0; i < n / QK8_0; i++) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; j++)
            amax = fmaxf(amax, fabsf(v[i * QK8_0 + j]));
        float d  = amax / 127;
        float id = d ? 1.0f / d : 0.0f;
        b[i].d = fp32_to_fp16(d);
        for (int j = 0; j < QK8_0; j++)
            b[i].qs[j] = (int8_t)roundf(v[i * QK8_0 + j] * id);
    }
}


/* same hand-chosen inputs as Part 3 */
static void test_basic(void)
{
    printf("\n Basic hand chosen tests: \n");

    /* all ones, unit scales */
    fill(X, 1, H_ONE, 1);   fill(Y, 1, H_ONE, 1);   run("ones, n=32", 1, 1, true);
    fill(X, 128, H_ONE, 1); fill(Y, 128, H_ONE, 1); run("ones, n=4096", 128, 128, true);

    /* int8 extremes; (-128)*(-128)*32 = 2^19 is the largest per-block integer sum */
    fill(X, 1, H_ONE, 127);  fill(Y, 1, H_ONE, 127);  run("127 * 127", 1, 1, true);
    fill(X, 1, H_ONE, -128); fill(Y, 1, H_ONE, -128); run("-128 * -128", 1, 1, true);
    fill(X, 1, H_ONE, 127);  fill(Y, 1, H_ONE, -128); run("127 * -128", 1, 1, true);

    /* different value at every position */
    for (int j = 0; j < QK8_0; j++)
        X[0].qs[j] = Y[0].qs[j] = (int8_t)(j - 16);
    X[0].d = Y[0].d = H_ONE;
    run("ramp", 1, 1, true);

    /* alternating signs cancel exactly */
    fill(X, 4, H_ONE, 100); fill(Y, 4, H_ONE, 100);
    for (int ib = 0; ib < 4; ib++)
        for (int j = 0; j < QK8_0; j += 2)
            Y[ib].qs[j] = -100;
    run("alternating signs", 4, 4, true);

    /* zero scale with non-zero quants */
    fill(X, 2, H_ZERO, 77); fill(Y, 2, H_ONE, 55); run("zero scale", 2, 2, true);

    /* each block uses its own scale, incl. a negative one */
    fill(&X[0], 1, H_HALF, 1);    fill(&Y[0], 1, H_TWO, 1);
    fill(&X[1], 1, H_QUARTER, 2); fill(&Y[1], 1, H_TWO, 2);
    fill(&X[2], 1, H_MTWO, 1);    fill(&Y[2], 1, H_ONE, 1);
    fill(&X[3], 1, H_ONE, 3);     fill(&Y[3], 1, H_ONE, 3);
    run("mixed scales", 4, 4, true);

    /* denormal FP16 scale goes through the renormalization path */
    fill(X, 1, H_MIN_SUB, 1); fill(Y, 1, H_ONE, 1); run("denormal scale", 1, 1, true);

    /* largest FP16 scales with the largest integer sum stay finite */
    fill(X, 1, H_MAX, -128); fill(Y, 1, H_MAX, -128); run("max scales", 1, 1, true);
}

/* loop bounds and BRAM addressing */
static void test_lengths(void)
{
    printf("\n Testing different lengths: \n");
    fill(X, DOT_MAX_NB, H_ONE, 5);
    fill(Y, DOT_MAX_NB, H_ONE, 7);

    run("nb = 0", 0, 0, true);
    run("nb = MAX (4864 elements)", DOT_MAX_NB, DOT_MAX_NB, true);
    run("nb = MAX + 10 is clamped", DOT_MAX_NB, DOT_MAX_NB + 10, true);

    /* only the last block is non-zero: last BRAM addresses */
    fill(X, DOT_MAX_NB, H_ONE, 0);
    for (int j = 0; j < QK8_0; j++)
        X[DOT_MAX_NB - 1].qs[j] = (int8_t)(j + 1);
    run("only last block non-zero", DOT_MAX_NB, DOT_MAX_NB, true);
}

/* Gaussian data quantized like llama.cpp; sigma 1e-4 gives denormal scales,
 * Used 0.02 for weights, 1 for activations */
static void test_random(void)
{
    printf("\n Random (llama.cpp quantizer): \n");
    const int    lens[]   = { 32, 896, 4864 };
    const double sigmas[] = { 1e-4, 0.02, 1.0 };

    for (int li = 0; li < 3; li++) {
        for (int si = 0; si < 3; si++) {
            int fails_before = n_fail;
            for (int t = 0; t < 3; t++) {
                rng_state = 1 + 1000 * li + 100 * si + t;
                for (int i = 0; i < lens[li]; i++) {
                    xf[i] = (float)(sigmas[si] * rng_gauss());
                    yf[i] = (float)(sigmas[si] * rng_gauss());
                }
                quantize(xf, X, lens[li]);
                quantize(yf, Y, lens[li]);
                run("random", lens[li] / QK8_0, lens[li] / QK8_0, false);
            }
            printf("  n=%d, sigma=%g: %d/3 pass\n", lens[li], sigmas[si], 3 - (n_fail - fails_before));
        }
    }
}

/* To stress the per-block scales */
static void test_special(void)
{
    printf("\n Special distributions (n=896): \n");
    const int n = 896, nb = n / QK8_0;

    for (int t = 0; t < 3; t++) {
        rng_state = 5000 + t;

        /* one large outlier per block sets the scale, the rest quantize to small values */
        for (int i = 0; i < n; i++) {
            xf[i] = (float)(0.01 * rng_gauss()) + ((i % QK8_0) == t ? 5.0f : 0.0f);
            yf[i] = (float)rng_gauss();
        }
        quantize(xf, X, n); quantize(yf, Y, n);
        run("outliers", nb, nb, true);

        /* Mostly zeros, including all-zero blocks with scale 0 */
        for (int i = 0; i < n; i++) {
            xf[i] = rng_unif() < 0.9 ? 0.0f : (float)rng_gauss();
            yf[i] = rng_unif() < 0.9 ? 0.0f : (float)rng_gauss();
        }
        quantize(xf, X, n); quantize(yf, Y, n);
        run("sparse", nb, nb, true);

        /* block magnitude grows from 1e-6 to 1e-2, to test both denormal and normal scales */
        for (int i = 0; i < n; i++) {
            double mag = pow(10.0, -6.0 + 4.0 * (i / QK8_0) / (nb - 1));
            xf[i] = (float)(mag * rng_gauss());
            yf[i] = (float)(mag * rng_gauss());
        }
        quantize(xf, X, n); quantize(yf, Y, n);
        run("denormal to normal sweep", nb, nb, true);
    }
}

int main()
{
    test_basic();
    test_lengths();
    test_random();
    test_special();

    printf("\n%d tests, %d failed, %d bit-exact\n", n_tests, n_fail, n_exact);
    printf("max absolute error: %.6e\n", max_abs_err);
    printf("max relative error: %.6e %%\n", max_rel_err);
    printf("%s\n", n_fail ? "* TEST FAILED *" : "* TEST PASSED *");
    return n_fail ? 1 : 0;
}
