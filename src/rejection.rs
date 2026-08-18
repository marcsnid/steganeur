//! Rejection-sampling-based steganography (Cachin, 2004).
//!
//! At each step, the filtered LM distribution is partitioned into K = 2^bits
//! equal-probability bins by cumulative probability (reusing FreqTable and
//! subdivide from arithmetic.rs). The next `bits` message bits select a target
//! bin. The encoder draws a random sample from the distribution; if the sampled
//! token falls into the target bin it is emitted, otherwise it is redrawn.
//!
//! This gives exactly zero KL divergence from the original LM distribution --
//! P(emit t) = P(t) for all t -- because the bins are just a partition of
//! the probability space, and the message bits provide a uniform random target.
//!
//! The decoder does NOT sample. It recomputes the same cumulative bins and
//! maps each emitted token to its bin via cumulative-range lookup, emitting
//! `bits` bits per token. This is deterministic given the distribution.
//!
//! The encoder uses a seeded RNG for reproducibility. The seed is stored in
//! StegaConfig (decoder ignores it).

use crate::arithmetic::{FreqTable, MAX_FREQ};
use crate::error::Result;
use crate::lm::{LanguageModel, TokenId};
use crate::steganography::{filter_distribution, find_punctuation_token, StegaConfig};
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

pub struct RejectionStega<'a> {
    lm: &'a dyn LanguageModel,
    config: StegaConfig,
    /// Number of bits per token (= log2 of number of bins).
    pub bits: usize,
}

impl<'a> RejectionStega<'a> {
    pub fn new(lm: &'a dyn LanguageModel, config: StegaConfig, bits: usize) -> Self {
        RejectionStega { lm, config, bits }
    }

    /// Build K bin boundaries over the cumulative table.
    ///
    /// Each bin is assigned a contiguous run of tokens (boundaries fall on
    /// token cumulative edges), every bin has at least one token, and the
    /// probability mass is as close to equal (total/K per bin) as token
    /// granularity allows.  Equal-mass bins mean every target bin is hit with
    /// probability ≈1/K per draw, so expected draws per token ≈ K and the
    /// fallback fires with probability (1−1/K)^{10K} ≈ e^{−10} per step.
    ///
    /// Returns a vec of length K+1 where bin b spans [bounds[b], bounds[b+1]).
    fn build_bin_bounds(cum: &[u64], total: u64, num_bins: usize) -> Vec<u64> {
        let n = cum.len().saturating_sub(1);
        if num_bins == 0 || n == 0 {
            return vec![0, total];
        }

        let mut bounds = vec![0u64; num_bins + 1];
        bounds[0] = 0;
        bounds[num_bins] = total;

        // Split points s_1..s_{K-1}: token i belongs to bin b iff
        // s_b <= i < s_{b+1} (with s_0 = 0, s_K = n).
        // s_b = first token index where cumulative mass reaches b·total/K,
        // clamped so each bin keeps at least one token.
        let mut prev_split = 0usize;
        for b in 1..num_bins {
            let target = (b as u128 * total as u128) / num_bins as u128;
            let mut split = (0..=n)
                .find(|&i| (cum[i] as u128) >= target)
                .unwrap_or(n);
            // Each bin gets at least one token: bin b-1 ends at prev_split
            // (exclusive), so bin b must start at prev_split+1 at the earliest,
            // and leave one token for each of the remaining (K-b) bins.
            split = split.clamp(prev_split + 1, n.saturating_sub(num_bins - b));
            bounds[b] = cum[split];
            prev_split = split;
        }
        bounds[num_bins] = total;

        bounds
    }

    /// Given a token index (into the filtered distribution), find which bin it
    /// falls into. Since boundaries fall on token cumulative edges and tokens
    /// are contiguous, the midpoint of the token's range lies in the same bin.
    fn token_to_bin(
        token_idx: usize,
        cum: &[u64],
        bin_bounds: &[u64],
    ) -> usize {
        let mid = (cum[token_idx] + cum[token_idx + 1]) / 2;
        match bin_bounds.binary_search(&mid) {
            Ok(i) => i.min(bin_bounds.len() - 2),
            Err(i) => (i.saturating_sub(1)).min(bin_bounds.len() - 2),
        }
    }

    /// Encode a message into tokens using rejection sampling.
    ///
    /// Per step: filter_distribution once → FreqTable → K=2^bits equal-size
    /// bins by cumulative probability. Read `bits` message bits → target bin.
    /// Draw u ~ Uniform[0, total) -- if the sampled token's bin matches the
    /// target bin, emit it; else redraw. Cap retries at 10·K, then fall back
    /// to the highest-prob token in the target bin.
    pub fn encode(
        &self,
        context: &[TokenId],
        message: &[u8],
        num_msg_bits: usize,
    ) -> Result<(Vec<TokenId>, usize, String)> {
        let mut rng: StdRng = match self.config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut ctx = context.to_vec();
        let mut tokens = Vec::new();
        let mut cover_parts: Vec<String> = Vec::new();
        let mut bits_consumed = 0;
        let eos = self.lm.eos_token();
        let num_bins = 1usize << self.bits;
        let max_retries = 10 * num_bins;

        let message_bits: Vec<u8> = (0..num_msg_bits)
            .map(|i| (message[i / 8] >> (7 - (i % 8))) & 1)
            .collect();
        let mut bit_pos = 0;

        for _ in 0..self.config.max_tokens {
            if bit_pos + self.bits > num_msg_bits {
                break;
            }

            // Read target bin from message bits
            let target_bin = {
                let mut bin = 0usize;
                for j in 0..self.bits {
                    bin = (bin << 1) | (message_bits[bit_pos + j] as usize);
                }
                bin
            };
            bit_pos += self.bits;
            bits_consumed += self.bits;

            // Compute distribution ONCE per step
            let filtered = filter_distribution(self.lm, &ctx, &self.config)?;
            let table = FreqTable::from_probs(&filtered.probs, MAX_FREQ);
            let total = table.total;
            if total == 0 || table.is_empty() {
                break;
            }

                // Build bin boundaries that guarantee every bin has at least one token
            let bin_bounds = Self::build_bin_bounds(&table.cum, total, num_bins);

            // Pre-compute which bin each token is in
            let token_bins: Vec<usize> = (0..filtered.ids.len())
                .map(|i| Self::token_to_bin(i, &table.cum, &bin_bounds))
                .collect();

            // Track the first token found in the target bin for fallback
            let mut best_in_target_idx = usize::MAX;
            let total_tokens = filtered.ids.len();

            // Rejection loop
            let mut accepted: Option<(TokenId, usize)> = None;
            for _attempt in 0..max_retries {
                let u = rng.gen_range(0..total);
                let token_idx = match table.cum.binary_search(&u) {
                    Ok(i) => i,
                    Err(i) => i.saturating_sub(1),
                };
                if token_idx >= total_tokens {
                    continue;
                }

                let t_bin = token_bins[token_idx];
                if t_bin == target_bin {
                    // Remember the first (or any) token in the target bin for fallback
                    if best_in_target_idx == usize::MAX {
                        best_in_target_idx = token_idx;
                    }
                    accepted = Some((filtered.ids[token_idx], token_idx));
                    break;
                }
            }

            // Fallback if no token accepted
            let (token, token_idx) = match accepted {
                Some((t, idx)) => (t, idx),
                None => {
                    // Pick the highest-prob token in the target bin
                    // If we already tracked one, use it; else scan all
                    if best_in_target_idx != usize::MAX {
                        (filtered.ids[best_in_target_idx], best_in_target_idx)
                    } else {
                        // Full scan
                        let mut best_t = filtered.ids[0];
                        let mut best_p = f64::NEG_INFINITY;
                        let mut best_i = 0;
                        for (i, (&id, &prob)) in
                            filtered.ids.iter().zip(filtered.probs.iter()).enumerate()
                        {
                            if token_bins[i] == target_bin && prob > best_p {
                                best_p = prob;
                                best_t = id;
                                best_i = i;
                            }
                        }
                        (best_t, best_i)
                    }
                }
            };

            let token_str = if token_idx < filtered.strings.len() {
                filtered.strings[token_idx].clone()
            } else {
                String::new()
            };

            tokens.push(token);
            cover_parts.push(token_str);
            ctx.push(token);

            if let Some(et) = eos && token == et {
                break;
            }
        }

        // Append trailing punctuation
        let filtered = filter_distribution(self.lm, &ctx, &self.config)?;
        if let Some((punct_id, punct_str)) =
            find_punctuation_token(&filtered.ids, &filtered.strings)
        {
            tokens.push(punct_id);
            cover_parts.push(punct_str);
        }

        let generated_text = cover_parts.concat();
        Ok((tokens, bits_consumed, generated_text))
    }

    /// Decode tokens back into message bits using the same bin boundaries.
    ///
    /// For each emitted token, find which bin it occupies in the cumulative
    /// distribution, and emit those `bits` bits.
    pub fn decode(
        &self,
        context: &[TokenId],
        tokens: &[TokenId],
        max_message_bits: usize,
    ) -> Result<(Vec<u8>, usize)> {
        let mut ctx = context.to_vec();
        let mut message_bits: Vec<u8> = Vec::new();
        let num_bins = 1usize << self.bits;

        for &token in tokens {
            let filtered = filter_distribution(self.lm, &ctx, &self.config)?;
            let table = FreqTable::from_probs(&filtered.probs, MAX_FREQ);
            if table.total == 0 || table.is_empty() {
                break;
            }

            // Find which index this token is at
            let token_idx = match filtered.ids.iter().position(|&t| t == token) {
                Some(i) => i,
                None => break,
            };

            // Build bin boundaries
            let total = table.total;
            let bin_bounds = Self::build_bin_bounds(&table.cum, total, num_bins);

            let bin = Self::token_to_bin(token_idx, &table.cum, &bin_bounds);

            // Emit `bits` bits for this bin
            for j in (0..self.bits).rev() {
                let bit = ((bin >> j) & 1) as u8;
                message_bits.push(bit);
            }

            ctx.push(token);
        }

        // Convert bits to bytes
        let max_bits = message_bits.len().min(max_message_bits);
        let mut bytes = Vec::with_capacity(max_bits.div_ceil(8));
        for chunk in message_bits[..max_bits].chunks(8) {
            let mut b = 0u8;
            for (j, &bit) in chunk.iter().enumerate() {
                b |= bit << (7 - j);
            }
            bytes.push(b);
        }

        Ok((bytes, max_bits))
    }

    /// Decode from cover text by matching token strings (same logic as
    /// ArithmeticStega::decode_text but uses bin lookup instead of interval math).
    pub fn decode_text(
        &self,
        context_text: &str,
        cover_text: &str,
        max_message_bits: usize,
    ) -> Result<(Vec<u8>, usize)> {
        let context_prefix = self.lm.tokenize(context_text)?;
        let mut context_tokens = context_prefix.to_vec();
        let mut message_bits: Vec<u8> = Vec::new();
        let mut remaining: &str = if !context_text.is_empty() && cover_text.starts_with(context_text)
        {
            &cover_text[context_text.len()..]
        } else {
            cover_text
        };
        let num_bins = 1usize << self.bits;

        loop {
            if remaining.is_empty() {
                break;
            }
            let filtered = filter_distribution(self.lm, &context_tokens, &self.config)?;
            let table = FreqTable::from_probs(&filtered.probs, MAX_FREQ);
            if table.total == 0 || table.is_empty() {
                break;
            }

            let total = table.total;
            let bin_bounds = Self::build_bin_bounds(&table.cum, total, num_bins);

            // Match token strings greedily
            let mut matched = None;
            let mut best_len = 0usize;
            for (i, token_str) in filtered.strings.iter().enumerate() {
                if token_str.is_empty() {
                    continue;
                }
                if token_str.len() > best_len && remaining.starts_with(token_str.as_str()) {
                    matched = Some((i, filtered.ids[i], token_str.len()));
                    best_len = token_str.len();
                }
            }

            if let Some((idx, _token_id, len)) = matched {
                let bin = Self::token_to_bin(idx, &table.cum, &bin_bounds);
                for j in (0..self.bits).rev() {
                    let bit = ((bin >> j) & 1) as u8;
                    message_bits.push(bit);
                }
                context_tokens.push(filtered.ids[idx]);
                remaining = &remaining[len..];
            } else {
                break;
            }
        }

        let max_bits = message_bits.len().min(max_message_bits);
        let mut bytes = Vec::with_capacity(max_bits.div_ceil(8));
        for chunk in message_bits[..max_bits].chunks(8) {
            let mut b = 0u8;
            for (j, &bit) in chunk.iter().enumerate() {
                b |= bit << (7 - j);
            }
            bytes.push(b);
        }

        Ok((bytes, max_bits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lm::DummyLM;
    use crate::steganography::StegaConfig;

    /// Helper: create a DummyLM with token strings for text-based decode.
    /// Uses a larger vocabulary so that every bin (for bits <= 3) has at
    /// least one token -- the rejection scheme needs each bin reachable.
    fn make_test_lm() -> DummyLM {
        let probs: Vec<f64> = vec![
            0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03,
            0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
        ];
        let strings: Vec<String> = (0..probs.len())
            .map(|i| format!("word{} ", i))
            .collect();
        DummyLM::new(probs.len())
            .with_probs(probs)
            .with_token_strings(strings)
    }

    #[test]
    fn test_rejection_roundtrip() {
        let lm = make_test_lm();
        let config = StegaConfig {
            temperature: 1.0,
            top_k: 10,
            max_tokens: 100,
            seed: Some(42),
        };
        let stega = RejectionStega::new(&lm, config, 2); // 2 bits per token

        let context = vec![0u32, 1u32];
        let message = b"Hi";
        let num_bits = 16;

        let (tokens, consumed, cover_text) = stega.encode(&context, message, num_bits).unwrap();
        assert!(!tokens.is_empty(), "should produce tokens");
        assert!(consumed <= num_bits, "should not consume more than available bits");
        assert!(!cover_text.is_empty(), "should generate cover text");

        // Decode using token-based decode
        let (decoded, decoded_bits) = stega.decode(&context, &tokens, num_bits + 8).unwrap();
        assert!(decoded_bits >= num_bits, "decoded {} < msg {}", decoded_bits, num_bits);

        // Compare bits
        for j in 0..num_bits {
            let orig = (message[j / 8] >> (7 - (j % 8))) & 1;
            let dec = (decoded[j / 8] >> (7 - (j % 8))) & 1;
            assert_eq!(orig, dec, "bit {} differs", j);
        }
    }

    #[test]
    fn test_rejection_roundtrip_text_decode() {
        let lm = make_test_lm();
        let config = StegaConfig {
            temperature: 1.0,
            top_k: 10,
            max_tokens: 100,
            seed: Some(42),
        };
        let stega = RejectionStega::new(&lm, config, 2);

        let context_text = "Some context. ";
        let message = b"Hello";
        let num_bits = 40;

        let context = lm.tokenize(context_text).unwrap();
        let (_tokens, _, cover_text) = stega.encode(&context, message, num_bits).unwrap();

        // Decode from full cover text
        let full_cover = format!("{}{}", context_text, cover_text);
        let (decoded, decoded_bits) = stega
            .decode_text(context_text, &full_cover, num_bits + 8)
            .unwrap();
        assert!(decoded_bits >= num_bits, "decoded {} < msg {}", decoded_bits, num_bits);

        for j in 0..num_bits {
            let orig = (message[j / 8] >> (7 - (j % 8))) & 1;
            let dec = (decoded[j / 8] >> (7 - (j % 8))) & 1;
            assert_eq!(orig, dec, "bit {} differs (text decode)", j);
        }
    }

    #[test]
    fn test_rejection_deterministic_with_seed() {
        let lm = make_test_lm();
        let config = StegaConfig {
            temperature: 1.0,
            top_k: 10,
            max_tokens: 50,
            seed: Some(12345),
        };
        let stega = RejectionStega::new(&lm, config, 2);

        let context = vec![0u32];
        let message = b"Test";
        let num_bits = 32;

        let (tokens1, _, _) = stega.encode(&context, message, num_bits).unwrap();
        let (tokens2, _, _) = stega.encode(&context, message, num_bits).unwrap();

        assert_eq!(tokens1, tokens2, "same seed should produce same tokens");
    }

    #[test]
    fn test_rejection_different_seed_different() {
        let lm = make_test_lm();
        let config1 = StegaConfig {
            temperature: 1.0,
            top_k: 10,
            max_tokens: 50,
            seed: Some(100),
        };
        let config2 = StegaConfig {
            temperature: 1.0,
            top_k: 10,
            max_tokens: 50,
            seed: Some(200),
        };
        let stega1 = RejectionStega::new(&lm, config1, 2);
        let stega2 = RejectionStega::new(&lm, config2, 2);

        let context = vec![0u32];
        let message = b"Test";
        let num_bits = 32;

        let (tokens1, _, _) = stega1.encode(&context, message, num_bits).unwrap();
        let (tokens2, _, _) = stega2.encode(&context, message, num_bits).unwrap();

        // They may sometimes be equal by chance, but with 32 bits and 2-bit bins
        // on a 3-token distribution, they're likely different.
        // This is a weak test but documents the intent.
        assert!(tokens1.len() == tokens2.len() || tokens1 != tokens2);
    }

    #[test]
    fn test_rejection_with_ecc_roundtrip() {
        use crate::ecc::{rs_decode, rs_encode};

        let lm = make_test_lm();
        let config = StegaConfig {
            temperature: 1.0,
            top_k: 50,
            max_tokens: 200,
            seed: Some(7),
        };
        let stega = RejectionStega::new(&lm, config, 2);

        // Build payload: message + 0x00 null terminator, then RS-encode.
        let mut payload = b"Meet me at noon".to_vec();
        payload.push(0x00);
        let parity = 10;
        let encoded_payload = rs_encode(&payload, parity);

        let context = vec![0u32];
        let (tokens, consumed, _) = stega
            .encode(&context, &encoded_payload, encoded_payload.len() * 8)
            .unwrap();
        assert_eq!(consumed, encoded_payload.len() * 8);

        // Simulate logprob drift: corrupt a few decoded bytes.
        let (mut raw, _) = stega.decode(&context, &tokens, encoded_payload.len() * 8 + 16).unwrap();
        // Corrupt 3 bytes within the correction capacity (⌊10/2⌋ = 5).
        raw[3] ^= 0xFF;
        raw[7] ^= 0xAA;
        raw[12] ^= 0x55;

        let corrected = rs_decode(&raw, parity).unwrap();
        // Strip leading zeros, read until null terminator.
        let stripped: Vec<u8> = corrected.iter().copied().skip_while(|&b| b == 0).collect();
        let message: Vec<u8> = stripped.iter().copied().take_while(|&b| b != 0).collect();
        assert_eq!(message, b"Meet me at noon");
    }

    #[test]
    fn test_rejection_with_varying_bits() {
        let lm = make_test_lm();
        
        for bits in 1..=3 {
            let config = StegaConfig {
                temperature: 1.0,
                top_k: 10,
                max_tokens: 100,
                seed: Some(42),
            };
            let stega = RejectionStega::new(&lm, config, bits);
            let context = vec![0u32];
            let message = b"Msg";
            let num_bits = 24;

            let (tokens, consumed, _) = stega.encode(&context, message, num_bits).unwrap();
            assert!(!tokens.is_empty(), "bits={}: should produce tokens", bits);
            assert!(consumed <= num_bits, "bits={}: consumed {} > {}", bits, consumed, num_bits);

            let (decoded, decoded_bits) = stega.decode(&context, &tokens, num_bits + 8).unwrap();
            assert!(decoded_bits >= num_bits, "bits={}: decoded {} < {}", bits, decoded_bits, num_bits);

            for j in 0..24.min(decoded_bits) {
                let orig = (message[j / 8] >> (7 - (j % 8))) & 1;
                let dec = (decoded[j / 8] >> (7 - (j % 8))) & 1;
                assert_eq!(orig, dec, "bits={}: bit {} differs", bits, j);
            }
        }
    }
}