//! Reed-Solomon error-correcting code over GF(2⁸).
//!
//! This is a method-agnostic byte-level ECC layer. It wraps the payload bytes
//! before encoding, and unwraps after decoding. Corrects up to ⌊nsym/2⌋ byte
//! errors, which is exactly what logprob drift produces (a bin shift corrupts a
//! run of consecutive bits).
//!
//! # Usage
//!
//! ```ignore
//! let encoded = rs_encode(&payload, 10);   // adds 10 parity bytes
//! let decoded = rs_decode(&encoded, 10)?; // corrects up to 5 byte errors
//! ```

use crate::error::{Error, Result};

// GF(2⁸) arithmetic -- uses the primitive polynomial x⁸ + x⁴ + x³ + x² + 1
// (0x11D). This is the same field used by QR codes, RAID 6, etc.

const GF_SIZE: usize = 256;

static GF_LOG: [u8; GF_SIZE] = make_gf_log();
static GF_EXP: [u8; GF_SIZE * 2] = make_gf_exp();

const fn make_gf_log() -> [u8; 256] {
    let mut log = [0u8; 256];
    let mut x = 1u16;
    let mut i = 0;
    while i < 255 {
        log[x as u8 as usize] = i as u8;
        x = (x as u16) << 1;
        if x & 0x100 != 0 {
            x ^= 0x11D;
        }
        i += 1;
    }
    log[0] = 255; // log(0) is undefined; mark it
    log
}

const fn make_gf_exp() -> [u8; 512] {
    let mut exp = [0u8; 512];
    let mut x = 1u16;
    let mut i = 0;
    while i < 255 {
        exp[i as usize] = x as u8;
        x = (x as u16) << 1;
        if x & 0x100 != 0 {
            x ^= 0x11D;
        }
        i += 1;
    }
    // Duplicate for easier multiplication
    let mut i = 255;
    while i < 512 {
        exp[i] = exp[i - 255];
        i += 1;
    }
    exp
}

fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let log_a = GF_LOG[a as usize];
    let log_b = GF_LOG[b as usize];
    GF_EXP[log_a as usize + log_b as usize]
}

fn gf_inv(a: u8) -> u8 {
    if a == 0 {
        return 0;
    }
    GF_EXP[255 - GF_LOG[a as usize] as usize]
}

// Reed-Solomon encoder (systematic)

/// Generate the generator polynomial coefficients for a given number of parity
/// symbols. The generator polynomial is ∏(x - αⁱ) for i = 0..nsym.
fn generator_poly(nsym: usize) -> Vec<u8> {
    // g(x) starts as 1
    let mut g = vec![1u8];
    for i in 0..nsym {
        // Multiply by (x + αⁱ). Since g is in low-order-first format (index 0 = x^0),
    // the factor (x + αⁱ) = αⁱ + x is represented as [αⁱ, 1].
        let alpha = GF_EXP[i];
        let mut new = vec![0u8; g.len() + 1];
        for j in 0..g.len() {
            new[j] ^= gf_mul(g[j], alpha);  // g[j] * αⁱ
            new[j + 1] ^= g[j];              // g[j] * x
        }
        g = new;
    }
    g
}

/// Reed-Solomon systematic encode.
///
/// Takes `data` bytes and returns a new vector with `nsym` parity bytes appended.
/// Uses low-order-first polynomial convention (index 0 = coefficient of x^0).
pub fn rs_encode(data: &[u8], nsym: usize) -> Vec<u8> {
    if data.is_empty() || nsym == 0 {
        return data.to_vec();
    }
    let gen_poly = generator_poly(nsym); // gen_poly[i] = coeff of x^i (low-order), gen_poly[nsym] = 1
    let k = data.len();
    let n = k + nsym;

    // Message polynomial (high-order): m(x) = data[0]*x^(k-1) + ... + data[k-1]
    // We need parity(x) = (m(x) * x^nsym) mod g(x).
    // Internally use low-order arrays (index i = coefficient of x^i).
    let mut p = vec![0u8; n];
    for i in 0..k {
        p[nsym + i] = data[k - 1 - i]; // coefficient of x^(nsym+i) in m(x)*x^nsym
    }

    // Synthetic division: for i from n-1 down to nsym, if coefficient c = p[i]
    // is nonzero, subtract c * x^(i-nsym) * g(x) (cancels the term at degree i).
    for i in (nsym..n).rev() {
        let c = p[i];
        if c == 0 {
            continue;
        }
        for j in 0..gen_poly.len() {
            p[i - nsym + j] ^= gf_mul(gen_poly[j], c);
        }
    }

    // p[0..nsym] = remainder (parity) in low-order: index j = coefficient of x^j.
    // Systematic codeword (high-order byte array): [data..., parity...] where
    //   c(x) = data[0]*x^(n-1) + ... + data[k-1]*x^nsym
    //          + parity[0]*x^(nsym-1) + ... + parity[nsym-1]
    // So parity byte at position k+j = coefficient of x^(nsym-1-j) = p[nsym-1-j].
    let mut out = vec![0u8; n];
    out[..k].copy_from_slice(data);
    for j in 0..nsym {
        out[k + j] = p[nsym - 1 - j];
    }
    out
}

// Reed-Solomon decoder

/// Compute syndromes for the received message. Returns `nsym` syndrome values.
/// The message is in high-order-first format (index 0 = coefficient of x^(n-1)).
/// Syndromes are S_j = c(α^j) for j = 0..nsym-1.
pub fn syndromes(msg: &[u8], nsym: usize) -> Vec<u8> {
    let mut syn = vec![0u8; nsym];
    for i in 0..nsym {
        let alpha = GF_EXP[i];
        let mut s = 0u8;
        for &b in msg.iter() {
            s = gf_mul(s, alpha) ^ b;
        }
        syn[i] = s;
    }
    syn
}

/// Berlekamp-Massey algorithm to find the error locator polynomial.
pub fn berlekamp_massey(syndromes: &[u8]) -> Vec<u8> {
    let n = syndromes.len();
    let mut c = vec![0u8; n];
    let mut b = vec![0u8; n];
    c[0] = 1;
    b[0] = 1;
    let mut len = 0usize;
    let mut m = 1usize;
    let mut lc = 1u8;

    for r in 0..n {
        // Compute discrepancy d
        let mut d = syndromes[r];
        for i in 1..=len {
            d ^= gf_mul(c[i], syndromes[r - i]);
        }

        if d == 0 {
            m += 1;
        } else {
            let t = c.clone();
            // c = c - d * lc⁻¹ * xᵐ * b
            let coef = gf_mul(d, gf_inv(lc));
            for i in 0..n {
                if i + m < n {
                    c[i + m] ^= gf_mul(coef, b[i]);
                }
            }
            if 2 * len <= r {
                len = r + 1 - len;
                b = t;
                lc = d;
                m = 1;
            } else {
                m += 1;
            }
        }
    }

    c.truncate(len + 1);
    c
}

/// Find the roots of the error locator polynomial by evaluating at each
/// possible location. Returns the error positions (indices into the message
/// from the start). Position `i` (0-indexed from start) corresponds to error
/// location X = α^(msg_len - 1 - i) in the high-order-first convention.
pub fn find_errors(error_locator: &[u8], msg_len: usize) -> Vec<usize> {
    let mut errors = Vec::new();
    for i in 0..msg_len {
        // Evaluate Λ(x) at x = X^(-1) = α^(-(msg_len - 1 - i))
        let exponent = (msg_len - 1 - i) % 255;
        let alpha = if exponent == 0 { 1u8 } else { GF_EXP[255 - exponent] };
        let mut val = 0u8;
        let mut power = 1u8;
        for &c in error_locator.iter() {
            val ^= gf_mul(c, power);
            power = gf_mul(power, alpha);
        }
        if val == 0 {
            errors.push(i);
        }
    }
    errors
}

/// Forney algorithm to compute error magnitudes.
/// `msg_len` is the full length of the received message (data + parity).
/// Error positions are 0-indexed from the start of the message.
pub fn forney(syndromes: &[u8], error_locator: &[u8], error_positions: &[usize], msg_len: usize) -> Vec<u8> {
    let mut magnitudes = vec![0u8; error_positions.len()];

    // Compute error evaluator polynomial: Ω(x) = S(x) * Λ(x) mod x^nsym
    let nsym = syndromes.len();
    let mut omega = vec![0u8; nsym];
    for i in 0..nsym {
        for j in 0..=i.min(error_locator.len() - 1) {
            if i - j < syndromes.len() {
                omega[i] ^= gf_mul(syndromes[i - j], error_locator[j]);
            }
        }
    }

    for (idx, &pos) in error_positions.iter().enumerate() {
        // Position `pos` (0-indexed from start) has error location X = α^(msg_len - 1 - pos)
        let degree = (msg_len - 1 - pos) % 255;
        let x_alpha = if degree == 0 { 1u8 } else { GF_EXP[255 - degree] }; // X⁻¹ = α^(-degree)
        let x_factor = if degree == 0 { 1u8 } else { GF_EXP[degree] };      // X = α^degree

        // Evaluate Λ'(x) at x = X⁻¹: Λ'(x) = Σ_{k odd} Λ_k * x^(k-1)
        let mut lambda_prime = 0u8;
        for i in (1..error_locator.len()).step_by(2) {
            let coef = error_locator[i];
            let exp = (degree * (i - 1)) % 255;
            let alpha_pow = if exp == 0 { 1u8 } else { GF_EXP[255 - exp] }; // α^(-degree*(i-1))
            lambda_prime ^= gf_mul(coef, alpha_pow);
        }

        // Evaluate Ω(x) at x = X⁻¹
        let mut omega_val = 0u8;
        let mut power = 1u8;
        for &c in omega.iter() {
            omega_val ^= gf_mul(c, power);
            power = gf_mul(power, x_alpha);
        }

        // Error magnitude: e_i = X * Ω(X⁻¹) / Λ'(X⁻¹)
        if lambda_prime != 0 {
            magnitudes[idx] = gf_mul(gf_mul(x_factor, omega_val), gf_inv(lambda_prime));
        }
    }

    magnitudes
}

/// Reed-Solomon decode: correct errors in the received message.
///
/// Returns the corrected message with parity bytes stripped. Returns an error
/// if the message is too short or has too many errors to correct.
pub fn rs_decode(data: &[u8], nsym: usize) -> Result<Vec<u8>> {
    if data.len() < nsym || nsym == 0 {
        return Err(Error::Steganography(
            "RS decode: data too short or nsym = 0".into(),
        ));
    }
    let msg_len = data.len() - nsym;

    let syn = syndromes(data, nsym);
    if syn.iter().all(|&s| s == 0) {
        // No errors
        return Ok(data[..msg_len].to_vec());
    }

    let error_locator = berlekamp_massey(&syn);
    let degree = error_locator.len().saturating_sub(1);
    let max_errors = nsym / 2;

    // If the error locator degree exceeds the correction capacity, reject.
    if degree > max_errors {
        return Err(Error::Steganography(format!(
            "RS decode: error locator degree {} exceeds max {}",
            degree, max_errors
        )));
    }

    if error_locator.is_empty() || (error_locator.len() == 1 && error_locator[0] == 1) {
        return Ok(data[..msg_len].to_vec());
    }

    let error_positions = find_errors(&error_locator, data.len());
    if error_positions.is_empty() {
        return Err(Error::Steganography(
            "RS decode: error locator has no roots in field".into(),
        ));
    }

    if error_positions.len() > max_errors {
        return Err(Error::Steganography(format!(
            "RS decode: too many errors ({} > {})",
            error_positions.len(),
            max_errors
        )));
    }

    let magnitudes = forney(&syn, &error_locator, &error_positions, data.len());

    let mut corrected = data.to_vec();
    for (&pos, &mag) in error_positions.iter().zip(magnitudes.iter()) {
        corrected[pos] ^= mag;
    }

    Ok(corrected[..msg_len].to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf_arithmetic() {
        // Check that multiplication is commutative and associative
        for a in [0u8, 1, 2, 10, 100, 200] {
            for b in [0u8, 1, 3, 50, 150, 255] {
                assert_eq!(gf_mul(a, b), gf_mul(b, a), "commutative: {} * {}", a, b);
                assert_eq!(gf_mul(gf_mul(a, b), 3u8), gf_mul(a, gf_mul(b, 3u8)), "associative");
            }
        }
        // a * a⁻¹ = 1
        for a in [1u8, 5, 17, 99, 200] {
            let inv = gf_inv(a);
            assert_eq!(gf_mul(a, inv), 1, "a * a⁻¹ = 1 for a={}", a);
        }
        // gf_inv(0) = 0
        assert_eq!(gf_inv(0), 0);
    }

    #[test]
    fn test_rs_roundtrip_no_errors() {
        let data = b"Hello, World!";
        let nsym = 10;
        let encoded = rs_encode(data, nsym);
        assert_eq!(encoded.len(), data.len() + nsym);
        assert_eq!(&encoded[..data.len()], data);

        let decoded = rs_decode(&encoded, nsym).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rs_corrects_errors() {
        let data = b"Test message with some content!";
        let nsym = 10;
        let mut encoded = rs_encode(data, nsym);

        // Corrupt 4 bytes (within correction capacity of 5)
        encoded[3] ^= 0xFF;
        encoded[7] ^= 0xAA;
        encoded[12] ^= 0x55;
        encoded[15] ^= 0x01;

        let decoded = rs_decode(&encoded, nsym).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rs_corrects_max_errors() {
        let data = b"RS test with max errors capability";
        let nsym = 8; // corrects up to 4 errors
        let mut encoded = rs_encode(data, nsym);

        // Corrupt exactly 4 bytes
        encoded[2] ^= 0x11;
        encoded[5] ^= 0x22;
        encoded[10] ^= 0x33;
        encoded[14] ^= 0x44;

        let decoded = rs_decode(&encoded, nsym).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_rs_too_many_errors() {
        let data = b"Short";
        let nsym = 4; // corrects up to 2 errors
        let mut encoded = rs_encode(data, nsym);

        // Corrupt 3 bytes (beyond capacity)
        encoded[0] ^= 0xFF;
        encoded[1] ^= 0xFF;
        encoded[2] ^= 0xFF;

        let result = rs_decode(&encoded, nsym);
        assert!(result.is_err());
    }

    #[test]
    fn test_rs_empty_data() {
        let data = b"";
        let encoded = rs_encode(data, 10);
        assert!(encoded.is_empty());

        // rs_decode of just parity bytes
        let encoded2 = rs_encode(b"Hi", 6);
        assert_eq!(encoded2.len(), 8);
        let decoded = rs_decode(&encoded2, 6).unwrap();
        assert_eq!(decoded, b"Hi");
    }

    #[test]
    fn test_rs_burst_error() {
        // Burst errors (consecutive bytes) are common with logprob drift
        let data = b"Burst error test pattern for steganography!";
        let nsym = 12; // corrects up to 6 errors
        let mut encoded = rs_encode(data, nsym);

        // Corrupt 5 consecutive bytes
        for i in 5..10 {
            encoded[i] ^= 0xAB;
        }

        let decoded = rs_decode(&encoded, nsym).unwrap();
        assert_eq!(decoded, data);
    }
}