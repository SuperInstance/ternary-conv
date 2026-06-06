//! # ternary-conv
//!
//! Ternary convolution operations for signals and images with elements in {-1, 0, +1}.
//!
//! Provides 1D and 2D convolution, dilated convolution, depthwise separable
//! convolution, and common ternary kernel types (edge detect, blur, sharpen, emboss).
//!
//! All arithmetic uses explicit Z₃ match arms — never modular tricks.

use std::fmt;

/// A ternary signal/matrix storing elements as i8 in {-1, 0, +1}.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TernaryMatrix {
    rows: usize,
    cols: usize,
    data: Vec<i8>,
}

impl TernaryMatrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0; rows * cols] }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<i8>) -> Self {
        assert_eq!(data.len(), rows * cols);
        for &v in &data {
            assert!(v >= -1 && v <= 1, "element {} not in {{-1, 0, 1}}", v);
        }
        Self { rows, cols, data }
    }

    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }
    pub fn get(&self, r: usize, c: usize) -> i8 { self.data[r * self.cols + c] }
    pub fn set(&mut self, r: usize, c: usize, v: i8) {
        assert!(v >= -1 && v <= 1);
        self.data[r * self.cols + c] = v;
    }

    fn round_to_trit(v: i32) -> i8 {
        match v {
            ..=-1 => -1,
            0 => 0,
            1.. => 1,
        }
    }
}

impl fmt::Display for TernaryMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:2} ", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

/// Z₃ multiplication via explicit match.
#[inline(always)]
pub fn trit_mul(a: i8, b: i8) -> i8 {
    match (a, b) {
        (-1, -1) => 1,  (-1, 0) => 0,  (-1, 1) => -1,
        (0, -1) => 0,   (0, 0) => 0,   (0, 1) => 0,
        (1, -1) => -1,  (1, 0) => 0,   (1, 1) => 1,
        _ => unreachable!(),
    }
}

// ---------------------------------------------------------------------------
// 1D Convolution
// ---------------------------------------------------------------------------

/// 1D ternary convolution (same-mode: output length = input length).
///
/// Pads with zeros. Uses Z₃ arithmetic throughout.
pub fn conv1d(signal: &[i8], kernel: &[i8]) -> Vec<i8> {
    let n = signal.len();
    let k = kernel.len();
    let pad = k / 2;
    let mut out = vec![0i8; n];
    for i in 0..n {
        let mut acc: i32 = 0;
        for j in 0..k {
            let si = i as isize - pad as isize + j as isize;
            if si >= 0 && (si as usize) < n {
                acc += trit_mul(signal[si as usize], kernel[j]) as i32;
            }
        }
        out[i] = TernaryMatrix::round_to_trit(acc);
    }
    out
}

// ---------------------------------------------------------------------------
// 2D Convolution
// ---------------------------------------------------------------------------

/// 2D ternary convolution (same-mode: output size = input size).
///
/// Zero-padded boundary. Uses Z₃ arithmetic.
pub fn conv2d(input: &TernaryMatrix, kernel: &TernaryMatrix) -> TernaryMatrix {
    let rows = input.rows();
    let cols = input.cols();
    let kr = kernel.rows();
    let kc = kernel.cols();
    let pr = kr / 2;
    let pc = kc / 2;
    let mut out = TernaryMatrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            let mut acc: i32 = 0;
            for ki in 0..kr {
                for kj in 0..kc {
                    let si = i as isize - pr as isize + ki as isize;
                    let sj = j as isize - pc as isize + kj as isize;
                    if si >= 0 && (si as usize) < rows && sj >= 0 && (sj as usize) < cols {
                        acc += trit_mul(input.get(si as usize, sj as usize), kernel.get(ki, kj)) as i32;
                    }
                }
            }
            out.set(i, j, TernaryMatrix::round_to_trit(acc));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Dilated Convolution
// ---------------------------------------------------------------------------

/// 2D dilated ternary convolution.
///
/// `dilation` controls the spacing between kernel elements. A dilation of 1
/// is standard convolution. Uses Z₃ arithmetic.
pub fn conv2d_dilated(input: &TernaryMatrix, kernel: &TernaryMatrix, dilation: usize) -> TernaryMatrix {
    assert!(dilation >= 1, "dilation must be >= 1");
    let rows = input.rows();
    let cols = input.cols();
    let kr = kernel.rows();
    let kc = kernel.cols();
    let pr = (kr * dilation) / 2;
    let pc = (kc * dilation) / 2;
    let mut out = TernaryMatrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            let mut acc: i32 = 0;
            for ki in 0..kr {
                for kj in 0..kc {
                    let si = i as isize - pr as isize + (ki * dilation) as isize;
                    let sj = j as isize - pc as isize + (kj * dilation) as isize;
                    if si >= 0 && (si as usize) < rows && sj >= 0 && (sj as usize) < cols {
                        acc += trit_mul(input.get(si as usize, sj as usize), kernel.get(ki, kj)) as i32;
                    }
                }
            }
            out.set(i, j, TernaryMatrix::round_to_trit(acc));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Depthwise Separable Convolution
// ---------------------------------------------------------------------------

/// Depthwise separable 2D convolution.
///
/// Applies `depth_kernel` (spatial filter) per channel, then `point_kernel`
/// (1×1) across channels. Here modeled as: spatial conv followed by 1×1 conv.
/// Uses Z₃ arithmetic.
pub fn depthwise_separable_conv2d(
    input: &TernaryMatrix,
    depth_kernel: &TernaryMatrix,
    point_kernel: &TernaryMatrix,
) -> TernaryMatrix {
    assert_eq!(point_kernel.rows(), 1);
    assert_eq!(point_kernel.cols(), 1);
    // Step 1: spatial depthwise convolution
    let spatial = conv2d(input, depth_kernel);
    // Step 2: 1×1 pointwise (just scalar multiply)
    let scalar = point_kernel.get(0, 0);
    let mut out = TernaryMatrix::zeros(spatial.rows(), spatial.cols());
    for i in 0..spatial.rows() {
        for j in 0..spatial.cols() {
            out.set(i, j, trit_mul(spatial.get(i, j), scalar));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Predefined Kernels
// ---------------------------------------------------------------------------

/// Ternary edge detection kernel (Sobel-like 3×3).
///
/// ```text
///  1  1  1
///  0  0  0
/// -1 -1 -1
/// ```
pub fn kernel_edge_detect() -> TernaryMatrix {
    TernaryMatrix::from_vec(3, 3, vec![
         1,  1,  1,
         0,  0,  0,
        -1, -1, -1,
    ])
}

/// Ternary blur kernel (3×3 box blur with center emphasis).
///
/// ```text
///  0  1  0
///  1  0  1
///  0  1  0
/// ```
pub fn kernel_blur() -> TernaryMatrix {
    TernaryMatrix::from_vec(3, 3, vec![
        0, 1, 0,
        1, 0, 1,
        0, 1, 0,
    ])
}

/// Ternary sharpen kernel.
///
/// ```text
///  0 -1  0
/// -1  1 -1
///  0 -1  0
/// ```
pub fn kernel_sharpen() -> TernaryMatrix {
    TernaryMatrix::from_vec(3, 3, vec![
         0, -1,  0,
        -1,  1, -1,
         0, -1,  0,
    ])
}

/// Ternary emboss kernel.
///
/// ```text
/// -1 -1  0
/// -1  0  1
///  0  1  1
/// ```
pub fn kernel_emboss() -> TernaryMatrix {
    TernaryMatrix::from_vec(3, 3, vec![
        -1, -1,  0,
        -1,  0,  1,
         0,  1,  1,
    ])
}

/// Identity kernel (1×1 with value 1).
pub fn kernel_identity() -> TernaryMatrix {
    TernaryMatrix::from_vec(1, 1, vec![1])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trit_mul_exhaustive() {
        for a in [-1, 0, 1] {
            for b in [-1, 0, 1] {
                let r = trit_mul(a, b);
                // Verify it's a valid trit
                assert!(r >= -1 && r <= 1);
                // Verify: matches sign of regular multiplication clamped to {-1,0,1}
                let expected = match (a, b) {
                    (0, _) | (_, 0) => 0,
                    (-1, -1) | (1, 1) => 1,
                    (-1, 1) | (1, -1) => -1,
                    _ => 0,
                };
                assert_eq!(r, expected, "trit_mul({a}, {b}) = {r}, expected {expected}");
            }
        }
    }

    #[test]
    fn test_conv1d_basic() {
        let signal = vec![1, 1, -1, 0, 1];
        let kernel = vec![1, 0, -1]; // simple diff
        let out = conv1d(&signal, &kernel);
        assert_eq!(out.len(), signal.len());
        // Index 0: 1*0 + 1*1 + 0*(boundary) = 1 → but with padding:
        // i=0: signal[-1]=0, signal[0]=1, signal[1]=1 → 0*1 + 1*0 + 1*(-1) = -1
        assert_eq!(out[0], -1);
    }

    #[test]
    fn test_conv1d_identity() {
        let signal = vec![1, -1, 0, 1, -1];
        let kernel = vec![1]; // identity kernel
        let out = conv1d(&signal, &kernel);
        assert_eq!(out, signal);
    }

    #[test]
    fn test_conv2d_identity_kernel() {
        let input = TernaryMatrix::from_vec(3, 3, vec![
            1, -1, 0,
            0, 1, -1,
            -1, 0, 1,
        ]);
        let kernel = kernel_identity();
        let out = conv2d(&input, &kernel);
        assert_eq!(out, input, "1×1 identity kernel should preserve input");
    }

    #[test]
    fn test_conv2d_edge_detection() {
        // Create a gradient image: top half -1, bottom half +1
        let input = TernaryMatrix::from_vec(5, 5, vec![
            -1, -1, -1, -1, -1,
            -1, -1, -1, -1, -1,
             0,  0,  0,  0,  0,
             1,  1,  1,  1,  1,
             1,  1,  1,  1,  1,
        ]);
        let edge = conv2d(&input, &kernel_edge_detect());
        // Row 0: only ki=1 (si=0) and ki=2 (si=1) are valid; ki=1 has kernel=0, ki=2 has kernel=-1 on input row -1
        // → trit_mul(-1,-1)=1 × 3 = 3 → 1
        assert_eq!(edge.get(0, 2), 1);
        // Row 2 (transition): full kernel. top=-1×1=-3, mid=0×0=0, bot=1×(-1)=-3 → -6 → -1
        assert_eq!(edge.get(2, 2), -1);
        // Row 4: only ki=0 (si=3) and ki=1 (si=4); ki=0 has kernel=1 on input row 1s → +3 → 1
        assert_eq!(edge.get(4, 2), 1);
    }

    #[test]
    fn test_conv2d_dilation_skips() {
        let input = TernaryMatrix::from_vec(5, 5, vec![
            1, 0, 0, 0, 1,
            0, 0, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 0, 0,
            1, 0, 0, 0, 1,
        ]);
        let kernel = kernel_identity();
        // Dilation 1 = standard
        let out1 = conv2d_dilated(&input, &kernel, 1);
        // With 1×1 kernel, dilation doesn't matter
        assert_eq!(out1, input);
    }

    #[test]
    fn test_conv2d_dilation_2_differs_from_1() {
        // Use non-uniform input so dilation makes a real difference
        let input = TernaryMatrix::from_vec(5, 5, vec![
            1, 0, -1, 0, 1,
            0, 1, 0, -1, 0,
           -1, 0, 1, 0, -1,
            0, -1, 0, 1, 0,
            1, 0, -1, 0, 1,
        ]);
        let kernel = TernaryMatrix::from_vec(3, 3, vec![
            1, 0, 1,
            0, 1, 0,
            1, 0, 1,
        ]);
        let out1 = conv2d_dilated(&input, &kernel, 1);
        let out2 = conv2d_dilated(&input, &kernel, 2);
        assert_ne!(out1, out2, "different dilations should produce different results");
    }

    #[test]
    fn test_depthwise_separable_matches_standard() {
        let input = TernaryMatrix::from_vec(4, 4, vec![
            1, -1, 0, 1,
            0, 1, -1, 0,
            -1, 0, 1, -1,
            1, 1, 0, -1,
        ]);
        let depth_k = kernel_blur();
        let point_k = TernaryMatrix::from_vec(1, 1, vec![1]); // identity pointwise
        let sep = depthwise_separable_conv2d(&input, &depth_k, &point_k);
        let standard = conv2d(&input, &depth_k);
        assert_eq!(sep, standard, "depthwise separable with identity pointwise should match standard");
    }

    #[test]
    fn test_kernel_application_correctness() {
        // Test that blur kernel produces expected output for a known input
        let input = TernaryMatrix::from_vec(3, 3, vec![
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
        ]);
        let blur = kernel_blur();
        let out = conv2d(&input, &blur);
        // Blur kernel: [0,1,0; 1,0,1; 0,1,0]
        // Center (1,1): up=0, down=0, left=0, right=0 → sum=0 → trit 0
        assert_eq!(out.get(1, 1), 0);
        // Corner (0,0): right=0, down=0, diag ignored by kernel → sum=0
        // But boundary padding means only valid cells count
        // (0,0): up/left padded as 0. kernel(0,0)=0, kernel(0,1)=1→input(0,1)=0,
        // kernel(0,2)=0, kernel(1,0)=1→input(1,0)=0, kernel(1,1)=0→input(1,1)=1,
        // kernel(1,2)=1→input(1,2)=0, kernel(2,0)=0, kernel(2,1)=1→input(2,1)=0, kernel(2,2)=0
        // sum = 0*0 + 1*0 + 0*0 + 1*0 + 0*1 + 1*0 + 0*0 + 1*0 + 0*1 = 0
        assert_eq!(out.get(0, 0), 0);
    }

    #[test]
    fn test_kernel_emboss() {
        let input = TernaryMatrix::from_vec(3, 3, vec![
            1, 1, 1,
            0, 0, 0,
           -1,-1,-1,
        ]);
        let emboss = kernel_emboss();
        let out = conv2d(&input, &emboss);
        // Verify it produces a valid ternary output
        for i in 0..3 {
            for j in 0..3 {
                let v = out.get(i, j);
                assert!(v >= -1 && v <= 1, "output should be ternary, got {v}");
            }
        }
    }

    #[test]
    fn test_kernel_sharpen() {
        let input = TernaryMatrix::from_vec(3, 3, vec![
            0, 0, 0,
            0, 1, 0,
            0, 0, 0,
        ]);
        let sharpen = kernel_sharpen();
        let out = conv2d(&input, &sharpen);
        // Center: 1*1 + (-1)*0 + (-1)*0 + (-1)*0 + (-1)*0 = 1
        assert_eq!(out.get(1, 1), 1);
    }

    #[test]
    fn test_all_kernels_are_ternary() {
        for k in [kernel_edge_detect(), kernel_blur(), kernel_sharpen(), kernel_emboss(), kernel_identity()] {
            for &v in &k.data {
                assert!(v >= -1 && v <= 1, "kernel contains non-ternary value {v}");
            }
        }
    }

    #[test]
    fn test_conv2d_uniform_input() {
        let input = TernaryMatrix::from_vec(3, 3, vec![
            1, 1, 1,
            1, 1, 1,
            1, 1, 1,
        ]);
        let blur = kernel_blur();
        let out = conv2d(&input, &blur);
        // All neighbors are 1: sum at center = 1+1+1+1 = 4 → 1
        assert_eq!(out.get(1, 1), 1);
    }

    #[test]
    fn test_display_format() {
        let m = TernaryMatrix::from_vec(2, 2, vec![1, -1, 0, 1]);
        let s = format!("{}", m);
        assert!(s.contains(" 1"));
        assert!(s.contains("-1"));
    }
}
