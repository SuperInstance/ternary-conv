# ternary-conv

**Convolution where every weight is {-1, 0, +1} — the operation that makes ternary networks see.**

[![crate](https://img.shields.io/badge/crates.io-ternary--conv-orange)](https://crates.io)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Why This Exists

Convolution is the backbone of computer vision. It's also where neural networks burn the most compute: every pixel touches every kernel weight, every layer. When you constrain those weights to {-1, 0, +1}, convolution stops being floating-point multiplication and becomes integer sign operations — something a microcontroller can do faster than a GPU does float math.

But you can't just round float convolution outputs and call it ternary. The accumulation, the boundary handling, the rounding — they all interact. You need convolution that was *born* ternary.

## The Key Insight

A ternary convolution kernel multiplies each element by {-1, 0, or +1}. In hardware, that's a sign flip, a zero, or a pass-through — no multiplier needed. The accumulation is just integer addition. The final rounding (back to a trit) is a sign check. The entire operation is:

```
output[i,j] = sign(Σ kernel[k,l] × input[i+k, j+l])
```

Where every `×` is a trit multiplication (9 cases, compiler-visible) and `sign` maps to {-1, 0, +1}. No floating point anywhere in the pipeline.

## Quick Start

```toml
[dependencies]
ternary-conv = "0.1"
```

```rust
use ternary_conv::*;

// 1D convolution — signal processing
let signal = vec![1, 1, -1, 0, 1, -1];
let kernel = vec![1, 0, -1]; // difference detector
let output = conv1d(&signal, &kernel);

// 2D convolution — image filtering
let image = TernaryMatrix::from_vec(5, 5, vec![
     1,  1,  1,  1,  1,
     1,  0,  0,  0,  1,
     1,  0, -1,  0,  1,
     1,  0,  0,  0,  1,
     1,  1,  1,  1,  1,
]);

let edges    = conv2d(&image, &kernel_edge_detect());
let blurred  = conv2d(&image, &kernel_blur());
let sharp    = conv2d(&image, &kernel_sharpen());
let embossed = conv2d(&image, &kernel_emboss());

// Dilated (atrous) convolution — larger receptive field, same parameters
let dilated = conv2d_dilated(&image, &kernel_edge_detect(), 2);

// Depthwise separable — mobile-friendly factorization
let depth = kernel_blur();
let point = TernaryMatrix::from_vec(1, 1, vec![1]);
let sep = depthwise_separable_conv2d(&image, &depth, &point);
```

## Architecture

```
               ┌──────────────────┐
               │  TernaryMatrix   │  (row-major, i8, {-1, 0, +1})
               └────────┬─────────┘
                        │
          ┌─────────────┼──────────────┐
          │             │              │
    ┌─────▼─────┐ ┌────▼──────┐ ┌────▼──────────┐
    │  conv1d   │ │  conv2d   │ │  conv2d_       │
    │  (same)   │ │  (same)   │ │  dilated       │
    └───────────┘ └────┬──────┘ └───────────────┘
                       │
                ┌──────▼──────────┐
                │ depthwise_      │
                │ separable_conv2d│
                │ (spatial + 1×1) │
                └─────────────────┘
```

| Operation | Input | Output | Use Case |
|-----------|-------|--------|----------|
| `conv1d` | Signal + kernel | Same-length signal | Time series, NLP |
| `conv2d` | Image + kernel | Same-size image | Vision, feature extraction |
| `conv2d_dilated` | Image + kernel + dilation | Same-size image | Segmentation, multi-scale |
| `depthwise_separable_conv2d` | Image + spatial + pointwise | Same-size image | Mobile inference |

## Preset Kernels

All kernels are valid ternary matrices — no floating-point coefficients:

| Kernel | 3×3 Pattern | Effect |
|--------|------------|--------|
| **Edge Detect** | Prewit-like horizontal gradient | Detect edges |
| **Blur** | Cross-shaped averaging | Smooth, preserve ternary |
| **Sharpen** | Laplacian-like | Enhance details |
| **Emboss** | Diagonal difference | 3D relief effect |
| **Identity** | 1×1 pass-through | No-op verification |

```rust
let k = kernel_edge_detect();
//  1  1  1
//  0  0  0
// -1 -1 -1
```

## Accumulation and Rounding

Ternary products are in {-1, 0, +1}, but their sum can be arbitrarily large. The rounding rule is:

```
sum < 0 → -1    sum = 0 → 0    sum > 0 → +1
```

This is the ternary sign function. It's the natural projection from ℤ back to Z₃, and it's computationally trivial: one comparison.

**Why not modular arithmetic?** Because we're accumulating *integer sums* of trit products, not operating in Z₃ directly. The sum carries meaningful magnitude information that modular arithmetic would destroy. Sign rounding preserves the direction while discarding magnitude — exactly what you want for a ternary output.

## API Reference

### Core Types

```rust
struct TernaryMatrix {
    // Row-major, elements are i8 in {-1, 0, +1}
}

impl TernaryMatrix {
    fn zeros(rows: usize, cols: usize) -> Self;
    fn from_vec(rows: usize, cols: usize, data: Vec<i8>) -> Self;
    fn get(&self, r: usize, c: usize) -> i8;
    fn set(&mut self, r: usize, c: usize, v: i8);
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
}
```

### Convolution Operations

```rust
fn conv1d(signal: &[i8], kernel: &[i8]) -> Vec<i8>;
fn conv2d(input: &TernaryMatrix, kernel: &TernaryMatrix) -> TernaryMatrix;
fn conv2d_dilated(input: &TernaryMatrix, kernel: &TernaryMatrix, dilation: usize) -> TernaryMatrix;
fn depthwise_separable_conv2d(
    input: &TernaryMatrix,
    depth_kernel: &TernaryMatrix,
    point_kernel: &TernaryMatrix,  // must be 1×1
) -> TernaryMatrix;
```

### Preset Kernels

```rust
fn kernel_edge_detect() -> TernaryMatrix;
fn kernel_blur() -> TernaryMatrix;
fn kernel_sharpen() -> TernaryMatrix;
fn kernel_emboss() -> TernaryMatrix;
fn kernel_identity() -> TernaryMatrix;
```

### Arithmetic

```rust
fn trit_mul(a: i8, b: i8) -> i8;  // Z₃ multiplication via match
```

## Real-World Example: Ternary Edge Detection on a Satellite

A low-earth-orbit satellite captures 50-megapixel images and needs to detect coastline edges in real-time to trigger high-resolution capture. The downlink bandwidth is limited — processing must happen on-board.

Traditional approach: float32 Sobel filter → 200 MB/s bandwidth for the raw image → 800 MB/s for the filter kernels → power budget exceeded.

Ternary approach: quantize image to {-1, 0, +1} (dark/neutral/bright) → apply `kernel_edge_detect()` → 6.25 MB/s bandwidth (16× less) → runs on a radiation-hardened microcontroller drawing 50 mW.

```rust
// On-board processing
let quantized = ternarize_satellite_image(&raw_image);
let edges = conv2d(&quantized, &kernel_edge_detect());
let coastline_pixels = count_nonzero(&edges);
if coastline_pixels > threshold {
    trigger_high_res_capture();
}
```

## Performance Characteristics

- **conv1d**: O(signal_len × kernel_len) — each output is a dot product of length k
- **conv2d**: O(H × W × kH × kW) — four nested loops, same-mode with zero padding
- **Dilated**: Same complexity, larger effective receptive field: `k_eff = k + (k-1) × (dilation-1)`
- **Depthwise separable**: Reduces from O(k² × C²) to O(k² × C + C²) — dramatic savings for multi-channel

Memory: O(H × W) for the input and output matrices. Kernels are tiny (typically 3×3 = 9 bytes). No scratch space needed beyond the output.

## Ecosystem Connections

This crate operates on `TernaryMatrix` and produces outputs for downstream layers:

- [`ternary-matmul`](https://github.com/SuperInstance/ternary-matmul) — the matmul engine (fully-connected layers)
- [`ternary-pool`](https://github.com/SuperInstance/ternary-pool) — downsampling after convolution
- [`ternary-norm`](https://github.com/SuperInstance/ternary-norm) — normalize feature maps
- [`ternary-activation`](https://github.com/SuperInstance/ternary-activation) — non-linearities between conv layers
- [`ternary-kernel-launch`](https://github.com/SuperInstance/ternary-kernel-launch) — dispatch these ops to GPU

## Open Questions

- **Winograd for ternary**: Winograd's algorithm reduces the number of multiplications for small fixed kernels (3×3 → 2×2). Since ternary "multiplications" are already nearly free, the win is marginal — but the reduced accumulation count still helps.
- **Im2col + GEMM decomposition**: Standard CNN implementations reshape convolution into matrix multiplication. Combined with `ternary-matmul`'s XNOR path, this could be very fast for binary-ternary filters.
- **3D convolution**: Video and volumetric data need 3D kernels. The current API is 1D and 2D only.

## Testing

```bash
cargo test
```

Covers: exhaustive trit multiplication, 1D identity preservation, 2D identity pass-through, edge detection directional responses, dilation correctness, depthwise separable equivalence with identity pointwise, kernel application on known inputs, all kernels are valid ternary, and Display formatting.

## License

MIT
