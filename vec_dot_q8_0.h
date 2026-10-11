#ifndef VEC_DOT_Q8_0_H
#define VEC_DOT_Q8_0_H

#include <stddef.h>
#include <stdint.h>

/* Also included from the C++ HLS testbench in Part 4 */
#ifdef __cplusplus
extern "C" {
#endif

#define QK8_0 32

/* FP16 stored as raw bits, as in ggml-common.h */
typedef uint16_t ggml_half;

typedef struct {
    ggml_half d;
    int8_t    qs[QK8_0];
} block_q8_0;

float ggml_compute_fp16_to_fp32(ggml_half h);

void ggml_vec_dot_q8_0_q8_0(int n, float *s, size_t bs, const void *vx, size_t bx,
                            const void *vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif

#endif
