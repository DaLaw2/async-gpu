//! Quantization utilities — shared pack/unpack for INT8 and INT4.

/// Quantize f32 values to INT8 (symmetric per-tensor).
/// Returns (quantized_i8, scale).
pub fn quantize_int8_per_tensor(data: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = data.iter().fold(0.0f32, |mx, &v| mx.max(v.abs()));
    let scale = if max_abs < 1e-12 {
        1.0
    } else {
        max_abs / 127.0
    };
    let inv_scale = 1.0 / scale;
    let quantized = data
        .iter()
        .map(|&v| (v * inv_scale).round().clamp(-128.0, 127.0) as i8)
        .collect();
    (quantized, scale)
}

/// Pack INT8 values into u32 (4 per u32, little-endian byte order).
///
/// The input length is padded to a multiple of 4 (with zero bytes) if needed.
pub fn pack_int8_to_u32(values: &[i8]) -> Vec<u32> {
    let n_u32 = values.len().div_ceil(4);
    let mut packed = vec![0u32; n_u32];
    for (i, &v) in values.iter().enumerate() {
        let word = i / 4;
        let byte = i % 4;
        packed[word] |= (v as u8 as u32) << (byte * 8);
    }
    packed
}

/// Quantize f32 column to INT8 (symmetric per-column).
/// Returns (quantized_i8, scale).
pub fn quantize_int8_per_column(data: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = data.iter().fold(0.0f32, |mx, &v| mx.max(v.abs()));
    let scale = if max_abs < 1e-12 {
        1.0
    } else {
        max_abs / 127.0
    };
    let inv_scale = 1.0 / scale;
    let quantized = data
        .iter()
        .map(|&v| (v * inv_scale).round().clamp(-128.0, 127.0) as i8)
        .collect();
    (quantized, scale)
}

/// Quantize f32 values to INT4 per-group (symmetric, `group_size` elements per group).
/// Returns `(packed_u32, scales)` where packed has 8 INT4 values per u32.
///
/// Values are shifted to unsigned `[0, 15]` before packing (bias = +8).
/// `data.len()` must be divisible by 8.
pub fn quantize_int4_per_group(data: &[f32], group_size: usize) -> (Vec<u32>, Vec<f32>) {
    let len = data.len();
    let n_groups = len.div_ceil(group_size);
    let n_packed = len / 8;

    let mut packed = vec![0u32; n_packed];
    let mut scales = vec![0.0f32; n_groups];

    for g in 0..n_groups {
        let start = g * group_size;
        let end = (start + group_size).min(len);

        // Find max absolute value in group
        let mut max_abs = 0.0f32;
        for &v in &data[start..end] {
            max_abs = max_abs.max(v.abs());
        }
        let scale = if max_abs < 1e-12 { 1.0 } else { max_abs / 7.0 };
        scales[g] = scale;

        // Quantize and pack
        let inv_scale = 1.0 / scale;
        for (i, &v) in data[start..end].iter().enumerate() {
            let idx = start + i;
            let q = ((v * inv_scale).round() as i32).clamp(-8, 7);
            let q_unsigned = (q + 8) as u32; // shift to [0, 15]
            let word = idx / 8;
            let bit_pos = (idx % 8) * 4;
            packed[word] |= (q_unsigned & 0xF) << bit_pos;
        }
    }

    (packed, scales)
}

/// Unpack INT4 from u32: extract the `index`-th 4-bit value (0..7).
///
/// Returns the signed value in `[-8, 7]` (bias-subtracted).
pub fn unpack_int4(packed: u32, index: usize) -> i8 {
    debug_assert!(index < 8, "INT4 index must be 0..7");
    let bits = (packed >> (index * 4)) & 0xF;
    (bits as i8) - 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_per_tensor_roundtrip() {
        let data = vec![0.0, 1.0, -1.0, 0.5];
        let (q, scale) = quantize_int8_per_tensor(&data);
        assert_eq!(q.len(), 4);
        assert!((scale - 1.0 / 127.0).abs() < 1e-6);
        assert_eq!(q[0], 0);
        assert_eq!(q[1], 127);
        assert_eq!(q[2], -127);
    }

    #[test]
    fn test_pack_int8_to_u32() {
        let values: Vec<i8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let packed = pack_int8_to_u32(&values);
        assert_eq!(packed.len(), 2);
        // First u32: bytes [1, 2, 3, 4]
        assert_eq!(packed[0], 0x04030201);
        // Second u32: bytes [5, 6, 7, 8]
        assert_eq!(packed[1], 0x08070605);
    }

    #[test]
    fn test_pack_int8_padding() {
        let values: Vec<i8> = vec![1, 2, 3];
        let packed = pack_int8_to_u32(&values);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0], 0x00030201);
    }

    #[test]
    fn test_int4_per_group_and_unpack() {
        let data = vec![0.0, 1.0, -1.0, 0.5, 0.25, -0.5, 0.75, -0.75];
        let (packed, scales) = quantize_int4_per_group(&data, 128);
        assert_eq!(packed.len(), 1);
        assert_eq!(scales.len(), 1);

        // Verify round-trip through unpack
        for i in 0..8 {
            let q = unpack_int4(packed[0], i);
            let reconstructed = q as f32 * scales[0];
            assert!(
                (reconstructed - data[i]).abs() < scales[0] + 1e-6,
                "mismatch at {i}: expected ~{}, got {reconstructed}",
                data[i]
            );
        }
    }

    #[test]
    fn test_unpack_int4_range() {
        // Pack known values: [-8, -7, ..., 7]
        let mut word = 0u32;
        for i in 0..8 {
            let val = i as i32 - 4; // [-4, -3, ..., 3]
            let unsigned = (val + 8) as u32;
            word |= (unsigned & 0xF) << (i * 4);
        }
        for i in 0..8 {
            let v = unpack_int4(word, i);
            assert_eq!(v, i as i8 - 4);
        }
    }
}
