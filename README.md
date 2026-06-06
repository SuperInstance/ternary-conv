# ternary-conv

**Ternary convolution operations for signals and images with {-1, 0, +1} arithmetic.**

[![crate](https://img.shields.io/badge/crates.io-ternary--conv-orange)](https://crates.io)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

## Overview

`ternary-conv` provides convolution operations designed for ternary-valued signals and images — data where every element is in the set {-1, 0, +1}. This constraint enables efficient, hardware-friendly computation with explicit Z₃ arithmetic throughout.

The crate supports:

| Operation | Description |
|-----------|-------------|
| **1D Convolution** | Same-mode with zero-padding |
| **2D Convolution** | Image/signal filtering with arbitrary kernels |
| **Dilated Convolution** | Atrous convolution with configurable dilation rate |
| **Depthwise Separable** | Spatial + pointwise factorization |
| **Preset Kernels** | Edge detect, blur, sharpen, emboss — all ternary |

## Why Ternary Convolution?

Ternary convolution is central to **Quantized Neural Networks (QNNs)** where weights and activations are constrained to {-1, 0, +1}. Benefits include:

- **Memory efficiency**: 2 bits per weight vs 32 for float32 — a 16× reduction
- **Compute efficiency**: Ternary multiplications reduce to lookup tables or XNOR+popcount
- **Hardware friendly**: No floating-point units needed; suitable for edge/FPGA deployment
- **Surprisingly effective**: Networks like TWN (Ternary Weight Networks) lose only 1-3% accuracy vs full precision

Beyond neural networks, ternary convolution appears in:
- **Digital signal processing** with quantized filters
- **Cellular automata** and discrete dynamical systems
- **Image processing** with simplified kernel operations
- **Mathematical morphology** on ternary-valued images

## Z₃ Arithmetic

All element-wise operations use explicit match arms — no modular arithmetic shortcuts:

```rust
pub fn trit_mul(a: i8, b: i8) -> i8 {
    match (a, b) {
        (-1, -1) =>  1,  (-1, 0) =>  0,  (-1, 1) => -1,
        ( 0, -1) =>  0,  ( 0, 0) =>  0,  ( 0, 1) =>  0,
        ( 1, -1) => -1,  ( 1, 0) =>  0,  ( 1, 1) =>  1,
        _ => unreachable!(),
    }
}
```

The convolution accumulates products in regular integers and rounds the result back to the nearest trit: negative → -1, zero → 0, positive → 1.

## Quick Start

```rust
use ternary_conv::*;

// 1D convolution
let signal = vec![1, 1, -1, 0, 1, -1];
let kernel = vec![1, 0, -1]; // simple difference detector
let output = conv1d(&signal, &kernel);

// 2D convolution with a ternary image
let image = TernaryMatrix::from_vec(5, 5, vec![
     1,  1,  1,  1,  1,
     1,  0,  0,  0,  1,
     1,  0, -1,  0,  1,
     1,  0,  0,  0,  1,
     1,  1,  1,  1,  1,
]);

// Edge detection
let edges = conv2d(&image, &kernel_edge_detect());

// Blur
let blurred = conv2d(&image, &kernel_blur());

// Sharpen
let sharp = conv2d(&image, &kernel_sharpen());

// Emboss (3D-like effect)
let embossed = conv2d(&image, &kernel_emboss());

// Dilated convolution (atrous)
let dilated = conv2d_dilated(&image, &kernel_edge_detect(), 2);

// Depthwise separable convolution
let depth = kernel_blur();
let point = TernaryMatrix::from_vec(1, 1, vec![1]); // identity pointwise
let sep = depthwise_separable_conv2d(&image, &depth, &point);
```

## Operation Details

### 1D Convolution

Computes same-mode convolution: the output has the same length as the input. Zero-padding is applied at boundaries. The kernel is centered at each position.

```
output[i] = round_to_trit(Σ signal[i+p-pad] × kernel[p])
```

### 2D Convolution

Standard 2D convolution with zero-padding (same-mode). Supports arbitrary kernel sizes. For a kernel of size (kh, kw), the padding is (kh/2, kw/2).

### Dilated Convolution

Also known as *atrous convolution*, this inserts `dilation - 1` zeros between kernel elements, effectively increasing the receptive field without increasing the parameter count. A dilation of 1 is standard convolution.

This is particularly useful in:
- **Semantic segmentation** (DeepLab-style architectures)
- **Multi-scale feature extraction** without pooling
- **Ternary CNNs** where larger receptive fields are needed cheaply

### Depthwise Separable Convolution

Factorizes a standard convolution into:
1. **Depthwise**: spatial convolution applied per channel
2. **Pointwise**: 1×1 convolution across channels

This reduces the parameter count from `k² × C_in × C_out` to `k² × C_in + C_in × C_out`. In the ternary case, the pointwise convolution is simply a Z₃ scalar multiplication.

### Preset Kernels

All kernels are valid ternary matrices:

| Kernel | Pattern | Use |
|--------|---------|-----|
| **Edge Detect** | Horizontal Prewit-like | Detect horizontal edges |
| **Blur** | Cross-shaped averaging | Smooth while preserving ternary values |
| **Sharpen** | Laplacian-like | Enhance edges and details |
| **Emboss** | Diagonal difference | Create 3D relief effect |
| **Identity** | 1×1 pass-through | No-op verification |

## Accumulation and Rounding

Since ternary products are in {-1, 0, +1} but the sum of many products can be arbitrarily large, the crate uses a simple rounding rule:

- Sum < 0 → **-1**
- Sum = 0 → **0**
- Sum > 0 → **+1**

This is the ternary analog of the sign function and preserves the algebraic structure while being computationally trivial.

## Testing

```bash
cargo test
```

The test suite covers:
- Exhaustive trit multiplication verification
- 1D basic convolution and identity kernel preservation
- 2D identity kernel pass-through
- Edge detection produces expected directional responses
- Dilation correctly skips positions and produces different results
- Depthwise separable matches standard convolution with identity pointwise
- Kernel application correctness on known inputs
- All preset kernels contain only valid trits

## Use in Research

If you use this crate in academic work, the key design choices to cite are:

1. **Ternary Weight Networks** (Li et al., 2016) — {-1, 0, +1} quantization
2. **Trained Ternary Quantization** (Zhu et al., 2016) — learned ternary weights
3. **XNOR-Net** (Rastegari et al., 2016) — binary/ternary fast paths
4. **Atrous convolution** (Chen et al., 2017) — dilated convolution for segmentation

## License

MIT
