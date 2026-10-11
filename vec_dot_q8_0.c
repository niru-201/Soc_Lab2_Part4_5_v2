/*
 * Scalar path of ggml_vec_dot_q8_0_q8_0 from ggml/src/ggml-cpu/arch/arm/quants.c,
 * with the NEON, SVE and i8mm paths removed.
 */
#include "vec_dot_q8_0.h"

#define UNUSED(x) (void)(x)

static float fp32_from_bits(uint32_t w)
{
    union { uint32_t u; float f; } v;
    v.u = w;
    return v.f;
}

static uint32_t fp32_to_bits(float f)
{
    union { float f; uint32_t u; } v;
    v.f = f;
    return v.u;
}

/*
 * ggml_compute_fp16_to_fp32 from ggml-impl.h. The CPU backend uses a
 * 64K lookup table built from this function; all give the same
 * result, so we removed the table.
 */
float ggml_compute_fp16_to_fp32(ggml_half h)
{
    const uint32_t w     = (uint32_t) h << 16;
    const uint32_t sign  = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float    exp_scale  = fp32_from_bits(UINT32_C(0x7800000));    /* 2^-112 */
    const float    normalized = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask   = UINT32_C(126) << 23;
    const float    denormalized = fp32_from_bits((two_w >> 17) | magic_mask) - 0.5f;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized) : fp32_to_bits(normalized));
    return fp32_from_bits(result);
}

void ggml_vec_dot_q8_0_q8_0(int n, float *s, size_t bs, const void *vx, size_t bx,
                            const void *vy, size_t by, int nrc)
{
    const int qk = QK8_0;
    const int nb = n / qk;
    const block_q8_0 *x = (const block_q8_0 *) vx;
    const block_q8_0 *y = (const block_q8_0 *) vy;
    float sumf = 0;
    int ib, j;

    /* only used by the ARM i8mm path (nrc == 2) */
    UNUSED(bs);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(nrc);

    for (ib = 0; ib < nb; ++ib) {
        int sumi = 0;

        for (j = 0; j < qk; j++)
            sumi += x[ib].qs[j] * y[ib].qs[j];

        sumf += sumi * (ggml_compute_fp16_to_fp32(x[ib].d) * ggml_compute_fp16_to_fp32(y[ib].d));
    }

    *s = sumf;
}
