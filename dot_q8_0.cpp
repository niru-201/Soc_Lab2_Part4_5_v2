#include <string.h>

#include "dot_q8_0.h"

/*
 * FP16 to FP32 with integer logic. Every FP16 value is a normal FP32 number,
 * so denormals are renormalized here instead.
 */
static float half_to_float(half h)
{
    uint16_t b;
    memcpy(&b, &h, sizeof b);

    uint32_t sign = (uint32_t)(b & 0x8000) << 16;
    uint32_t exp  = (b >> 10) & 0x1F;
    uint32_t man  = b & 0x3FF;
    uint32_t out;

    if (exp == 0x1F) {                  /* Inf / NaN */
        out = sign | 0x7F800000 | (man << 13);
    } else if (exp != 0) {              /* normal: rebias 15 to 127 */
        out = sign | ((exp + 112) << 23) | (man << 13);
    } else if (man == 0) {              /* + or -0 */
        out = sign;
    } else {
        /* denormal: shift the leading 1 into the implicit-bit position.
         * Fixed 4-step leading-zero count instead of a data-dependent loop. */
        uint32_t w = man << 6;          /* mantissa bit 9 -> bit 15 */
        uint32_t n = 0;
        if (!(w & 0xFF00)) { n += 8; w <<= 8; }
        if (!(w & 0xF000)) { n += 4; w <<= 4; }
        if (!(w & 0xC000)) { n += 2; w <<= 2; }
        if (!(w & 0x8000)) { n += 1; w <<= 1; }
        out = sign | ((112 - n) << 23) | (((w >> 5) & 0x3FF) << 13);
    }

    float f;
    memcpy(&f, &out, sizeof f);
    return f;
}

float dot_q8_0(int nb, const int8_t xq[DOT_MAX_N], const half xd[DOT_MAX_NB],
                       const int8_t yq[DOT_MAX_N], const half yd[DOT_MAX_NB])
{
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return
#pragma HLS INTERFACE mode=ap_none port=nb

#pragma HLS INTERFACE mode=bram port=xq depth=4864
#pragma HLS INTERFACE mode=bram port=yq depth=4864
#pragma HLS INTERFACE mode=bram port=xd depth=152
#pragma HLS INTERFACE mode=bram port=yd depth=152

    // Don't read past the end of the BRAMs
    if (nb > DOT_MAX_NB)
        nb = DOT_MAX_NB;

    float sumf = 0.0f;

BLOCK_LOOP:
    for (int ib = 0; ib < nb; ib++) {
        // n = 896 .. 4864 - for latency estimate in the report
#pragma HLS LOOP_TRIPCOUNT min=28 max=152
        int sumi = 0;

    ELEM_LOOP:
        for (int j = 0; j < DOT_QK; j++)
            sumi += xq[ib * DOT_QK + j] * yq[ib * DOT_QK + j];

        // same operation order as llama.cpp, so expecting results to be bit-exact
        sumf += (float)sumi * (half_to_float(xd[ib]) * half_to_float(yd[ib]));
    }

    return sumf;
}
