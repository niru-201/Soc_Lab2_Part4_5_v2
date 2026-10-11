# Parts 4 & 5: Q8_0 dot product for Vitis HLS

| File | Role |
|---|---|
| `dot_q8_0.cpp`, `dot_q8_0.h` | Design. Top function `dot_q8_0`. |
| `tb_dot_q8_0.cpp` | Testbench |
| `vec_dot_q8_0.c`, `vec_dot_q8_0.h` | Reference model: the Part 3 kernel |
| `run_hls.tcl` | C simulation, C synthesis, C/RTL co-simulation |

## Run (ECE-LRC)

```sh
module load xilinx/2022
vitis_hls -f run_hls.tcl      # results in hls_dot/solution1
vitis_hls -p hls_dot          # open in the GUI (reports, waveform)
```

## Design

- The interface is split into arrays: int8 quants `xq`/`yq` (BRAM, depth 4864) and FP16 scales `xd`/`yd` (BRAM, depth 152).
  - `nb` is `ap_none`; the block-level handshake is `ap_ctrl_hs`.
  - `nb` above 152 is clamped.
- Operations and their order are the same as llama.cpp, so results are bit-exact.
- The scales are converted FP16→FP32 with integer logic that renormalizes FP16 denormals. The Floating-Point IP would flush them to zero.

## Testbench

Each test compares the design with the reference on the same data.

**Test groups:**
- **hand-chosen:** the same cases as Part 3.
- **lengths:** nb = 0, nb = max, clamp, last block only.
- **random:** Gaussian data quantized with llama.cpp's quantizer, n = 32/896/4864, σ = 1e-4/0.02/1.
- **special:** outliers, sparse, and denormal→normal scales in one vector.

**Output and pass rule:**
- The testbench prints the max absolute error, the max relative error (%), and the bit-exact count.
- A test passes if |dut − ref| ≤ 1e-5 · Σ|block terms|. This allows FP32 rounding in any summation order, and it doesn't flag sums that cancel to ~0.
