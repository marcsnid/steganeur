//! Interval-based arithmetic coding for steganography.
//!
//! Both encode and decode use the SAME interval-narrowing logic and count the
//! "fixed" leading bits, making them perfect inverses.

use crate::error::Result;

/// Precision (in bits). 16 bits matches the reference implementation.
pub const PRECISION: u32 = 16;
pub const MAX_VAL: u64 = 1u64 << PRECISION;
pub const MAX_FREQ: u64 = MAX_VAL - 1;
pub const DEFAULT_MAX_TOKENS: usize = 10_000;

#[derive(Debug, Clone)]
pub struct FreqTable {
    pub cum: Vec<u64>,
    pub total: u64,
}

impl FreqTable {
    pub fn from_probs(probs: &[f64], max_total: u64) -> Self {
        let n = probs.len();
        if n == 0 { return FreqTable { cum: vec![0], total: 0 }; }
        let sum: f64 = probs.iter().sum();
        if sum <= 0.0 { return Self::uniform(n, max_total); }
        let mut freq = Vec::with_capacity(n);
        let mut total: u64 = 0;
        for &p in probs {
            let scaled = ((p / sum) * max_total as f64).round() as u64;
            let clamped = scaled.max(1).min(max_total.saturating_sub(total + (n - freq.len() - 1) as u64));
            total = total.saturating_add(clamped);
            freq.push(clamped);
        }
        while total > max_total && !freq.is_empty() {
            if freq.last().copied().unwrap_or(0) > 1 {
                if let Some(last) = freq.last_mut() { *last -= 1; total -= 1; }
            } else { break; }
        }
        let mut cum = Vec::with_capacity(n + 1);
        let mut running = 0u64;
        cum.push(0);
        for &f in &freq { running += f; cum.push(running); }
        FreqTable { cum, total: running }
    }

    pub fn uniform(n: usize, max_total: u64) -> Self {
        if n == 0 { return FreqTable { cum: vec![0], total: 0 }; }
        let each = (max_total / n as u64).max(1);
        let mut cum = Vec::with_capacity(n + 1);
        let mut running = 0u64;
        cum.push(0);
        for _ in 0..n { running += each; cum.push(running.min(max_total)); }
        if running < max_total && n > 0 { cum[n] = max_total; }
        let total = cum.last().copied().unwrap_or(0);
        FreqTable { cum, total }
    }

    pub fn len(&self) -> usize { self.cum.len().saturating_sub(1) }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

pub fn is_sentence_end(s: &str) -> bool {
    let t = s.trim(); t == "." || t == "!" || t == "?"
}

pub fn int_to_bits_msb(val: u64, precision: u32) -> Vec<u8> {
    (0..precision).rev().map(|i| ((val >> i) & 1) as u8).collect()
}

pub fn bits_msb_to_int(bits: &[u8]) -> u64 {
    bits.iter().fold(0u64, |v, &b| (v << 1) | (b as u64))
}

pub fn num_same_from_beg(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

pub fn subdivide(low: u64, high: u64, table: &FreqTable) -> Vec<u64> {
    let range = high - low;
    (0..table.len()).map(|i| low + (table.cum[i + 1] * range) / table.total).collect()
}

fn find_selection(cum: &[u64], value: u64) -> usize {
    match cum.binary_search(&value) {
        Ok(i) => i + 1,
        Err(i) => i,
    }
}

pub fn encode_message_to_tokens(
    message: &[u8],
    num_msg_bits: usize,
    mut get_dist: impl FnMut(&[u32]) -> Result<(FreqTable, Vec<u32>, Vec<String>)>,
    end_token: Option<u32>,
    max_tokens: usize,
    block_size: usize,
) -> Result<(Vec<u32>, usize, Vec<String>)> {
    let message_bits: Vec<u8> = (0..num_msg_bits)
        .map(|i| (message[i / 8] >> (7 - (i % 8))) & 1).collect();
    let mut cur_interval: [u64; 2] = [0, MAX_VAL];
    let mut tokens: Vec<u32> = Vec::new();
    let mut token_strings: Vec<String> = Vec::new();
    let mut i = 0usize;
    let stop_at = num_msg_bits + 32;

    for _ in 0..max_tokens {
        // Error-resilient block reset: every block_size tokens, reset the
        // interval so a drift-induced bin flip in one block doesn't cascade
        // into subsequent blocks.  block_size == 0 means no reset (legacy).
        if block_size > 0 && !tokens.is_empty() && tokens.len() % block_size == 0 {
            cur_interval = [0, MAX_VAL];
        }
        let (table, ids, strings) = get_dist(&tokens)?;
        if table.total == 0 || table.is_empty() { break; }
        let cum = subdivide(cur_interval[0], cur_interval[1], &table);
        let remaining = message_bits.len().saturating_sub(i);
        let bits_to_use = remaining.min(PRECISION as usize);
        let mut padded = message_bits.get(i..i + bits_to_use).unwrap_or(&[]).to_vec();
        padded.resize(PRECISION as usize, 0);
        let message_idx = bits_msb_to_int(&padded);
        let selection = find_selection(&cum, message_idx);
        let token = ids[selection];
        let token_str = strings.get(selection).cloned().unwrap_or_default();
        let new_low = if selection > 0 { cum[selection - 1] } else { cur_interval[0] };
        let new_high = cum[selection];
        let low_bits = int_to_bits_msb(new_low, PRECISION);
        let high_bits = int_to_bits_msb(new_high - 1, PRECISION);
        let n_fixed = num_same_from_beg(&low_bits, &high_bits);
        i += n_fixed;
        let mut nlb = low_bits[n_fixed..].to_vec();
        nlb.resize(PRECISION as usize, 0);
        let mut nhb = high_bits[n_fixed..].to_vec();
        nhb.resize(PRECISION as usize, 1);
        cur_interval[0] = bits_msb_to_int(&nlb);
        cur_interval[1] = bits_msb_to_int(&nhb) + 1;
        tokens.push(token);
        token_strings.push(token_str);
        if let Some(et) = end_token && token == et { break; }
        if i >= stop_at {
            if token_strings.last().is_some_and(|s| is_sentence_end(s)) { break; }
            if tokens.len() > stop_at / 8 + 50 { break; }
        }
        if i >= num_msg_bits + PRECISION as usize * 2 { break; }
    }
    Ok((tokens, i.min(num_msg_bits), token_strings))
}

pub fn decode_tokens_to_message(
    tokens: &[u32],
    mut get_dist: impl FnMut(&[u32]) -> Result<(FreqTable, Vec<u32>, Vec<String>)>,
    block_size: usize,
) -> Result<(Vec<u8>, usize)> {
    let mut cur_interval: [u64; 2] = [0, MAX_VAL];
    let mut message_bits: Vec<u8> = Vec::new();
    let mut context: Vec<u32> = Vec::new();

    for (idx, &token) in tokens.iter().enumerate() {
        // Error-resilient block reset: matches the encoder's reset points.
        if block_size > 0 && idx > 0 && idx % block_size == 0 {
            cur_interval = [0, MAX_VAL];
        }
        let (table, ids, _strings) = get_dist(&context)?;
        if table.total == 0 || table.is_empty() { break; }
        let cum = subdivide(cur_interval[0], cur_interval[1], &table);
        let selection = match ids.iter().position(|&t| t == token) {
            Some(s) => s,
            None => break,
        };
        let new_low = if selection > 0 { cum[selection - 1] } else { cur_interval[0] };
        let new_high = cum[selection];
        let low_bits = int_to_bits_msb(new_low, PRECISION);
        let high_bits = int_to_bits_msb(new_high - 1, PRECISION);
        let n_fixed = num_same_from_beg(&low_bits, &high_bits);
        if n_fixed > 0 { message_bits.extend_from_slice(&low_bits[..n_fixed]); }
        let mut nlb = low_bits[n_fixed..].to_vec();
        nlb.resize(PRECISION as usize, 0);
        let mut nhb = high_bits[n_fixed..].to_vec();
        nhb.resize(PRECISION as usize, 1);
        cur_interval[0] = bits_msb_to_int(&nlb);
        cur_interval[1] = bits_msb_to_int(&nhb) + 1;
        context.push(token);
    }

    let mut bytes = Vec::with_capacity(message_bits.len().div_ceil(8));
    for chunk in message_bits.chunks(8) {
        let mut b = 0u8;
        for (j, &bit) in chunk.iter().enumerate() { b |= bit << (7 - j); }
        bytes.push(b);
    }
    Ok((bytes, message_bits.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_fixed_dist() {
        let probs = vec![0.7, 0.3];
        let ids = vec![0u32, 1u32];
        let get_dist = |_ctx: &[u32]| -> Result<(FreqTable, Vec<u32>, Vec<String>)> {
            Ok((FreqTable::from_probs(&probs, MAX_FREQ), ids.clone(), vec!["a".into(), "b".into()]))
        };

        let msg_bytes: Vec<u8> = vec![0x48, 0x65]; // "He"
        let msg_bits = 16usize;
        let (tokens, consumed, _s) =
            encode_message_to_tokens(&msg_bytes, msg_bits, get_dist, None, 200, 0).unwrap();
        assert!(!tokens.is_empty());
        assert!(consumed <= msg_bits);

        let get_dist2 = |_ctx: &[u32]| -> Result<(FreqTable, Vec<u32>, Vec<String>)> {
            Ok((FreqTable::from_probs(&probs, MAX_FREQ), ids.clone(), vec!["a".into(), "b".into()]))
        };
        let (decoded, decoded_bits) = decode_tokens_to_message(&tokens, get_dist2, 0).unwrap();
        assert!(decoded_bits >= msg_bits, "decoded {} < msg {}", decoded_bits, msg_bits);
        for j in 0..msg_bits {
            let orig = (msg_bytes[j / 8] >> (7 - (j % 8))) & 1;
            let dec = (decoded[j / 8] >> (7 - (j % 8))) & 1;
            assert_eq!(orig, dec, "bit {} differs", j);
        }
    }

    /// Regression: the arithmetic decoder must preserve leading 0x00 bytes
    /// in the payload. The old `skip_while(|&b| b == 0)` hack in main.rs
    /// stripped them, which silently corrupted binary payloads (e.g. ciphertext
    /// whose first byte is 0x00). This test pins the correct behavior so any
    /// future regression is caught.
    #[test]
    fn test_leading_zero_bytes_preserved() {
        let n = 16usize;
        let ids: Vec<u32> = (0..n as u32).collect();
        let get_dist = |_ctx: &[u32]| -> Result<(FreqTable, Vec<u32>, Vec<String>)> {
            Ok((FreqTable::uniform(n, MAX_FREQ), ids.clone(), vec![]))
        };

        for payload in [
            vec![0x00u8, 0xAB],
            vec![0x00, 0x00, 0xCD],
            vec![0x01, 0x00],
            vec![0x4D, 0x00],
        ] {
            let msg_bits = payload.len() * 8;
            let (tokens, _, _) =
                encode_message_to_tokens(&payload, msg_bits, get_dist, None, 500, 0).unwrap();
            let (decoded, _) = decode_tokens_to_message(&tokens, get_dist, 0).unwrap();
            // The leading payload bytes must be intact (trailing bytes are
            // read-ahead zero padding, which framing is responsible for).
            assert_eq!(
                &decoded[..payload.len()],
                &payload[..],
                "payload {:02x?} did not round-trip; got {:02x?}",
                payload,
                decoded,
            );
        }
    }

    #[test]
    fn test_roundtrip_uniform() {
        let n = 10usize;
        let ids: Vec<u32> = (0..n as u32).collect();
        let get_dist = |_ctx: &[u32]| -> Result<(FreqTable, Vec<u32>, Vec<String>)> {
            Ok((FreqTable::uniform(n, MAX_FREQ), ids.clone(), vec![]))
        };
        let msg_bits = 16usize;
        let msg_bytes: Vec<u8> = vec![0x48, 0x65];
        let (tokens, _, _) =
            encode_message_to_tokens(&msg_bytes, msg_bits, get_dist, None, 200, 0).unwrap();
        assert!(!tokens.is_empty());
        let (decoded, decoded_bits) = decode_tokens_to_message(&tokens, get_dist, 0).unwrap();
        assert!(decoded_bits >= msg_bits);
        for j in 0..msg_bits {
            let orig = (msg_bytes[j / 8] >> (7 - (j % 8))) & 1;
            let dec = (decoded[j / 8] >> (7 - (j % 8))) & 1;
            assert_eq!(orig, dec, "bit {} differs", j);
        }
    }
}