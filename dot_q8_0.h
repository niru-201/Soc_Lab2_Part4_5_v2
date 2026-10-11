#ifndef DOT_Q8_0_H
#define DOT_Q8_0_H

#include <stdint.h>
#include <hls_half.h>

#define DOT_QK     32
#define DOT_MAX_N  4864                   /* largest dot length in Qwen2.5-0.5B */
#define DOT_MAX_NB (DOT_MAX_N / DOT_QK)   /* 152 blocks */

/* nb = number of 32-element blocks; values above DOT_MAX_NB are clamped */
float dot_q8_0(int nb, const int8_t xq[DOT_MAX_N], const half xd[DOT_MAX_NB],
                       const int8_t yq[DOT_MAX_N], const half yd[DOT_MAX_NB]);

#endif
