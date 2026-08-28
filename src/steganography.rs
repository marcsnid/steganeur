use crate::arithmetic::{self, FreqTable, DEFAULT_MAX_TOKENS};
use crate::bitstream::{BitReader, BitWriter};
use crate::error::Result;
use crate::lm::{LanguageModel, TokenId};
use crate::rejection::RejectionStega;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

// ============================================================================
// Shared distribution filtering
// ============================================================================

/// A filtered next-token distribution: probabilities, token IDs, and strings,
/// with EOS / special / replacement / prefix-overlap duplicates removed.
#[derive(Debug, Clone)]
pub struct FilteredDist {
    pub probs: Vec<f64>,
    pub ids: Vec<TokenId>,
    pub strings: Vec<String>,
}

/// Filter a raw LM distribution: remove EOS, special tokens (<|...|>),
/// replacement characters (U+FFFD), empty strings, and tokens whose
/// strings are prefixes of another (keeps the longer). Duplicate strings
/// keep the lower token ID for determinism.
pub fn filter_distribution(
    lm: &dyn LanguageModel,
    ctx: &[TokenId],
    config: &StegaConfig,
) -> Result<FilteredDist> {
    let dist = lm.predict(ctx)?;
    let mut probs = dist.get_probs(config.temperature, config.top_k);
    let mut token_ids = dist.get_token_ids(config.temperature, config.top_k);
    let mut token_strings: Vec<String> = dist
        .get_token_strings(config.temperature, config.top_k)
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    let eos = lm.eos_token();
    let has_strings = token_strings.iter().any(|s| !s.is_empty());

    // Remove EOS, special tokens, empty strings, replacement chars
    let mut i = 0;
    while i < token_ids.len() {
        let id = token_ids[i];
        let s = token_strings.get(i).map(|s| s.as_str()).unwrap_or("");
        let is_eos = eos.map(|e| id == e).unwrap_or(false);
        let is_special = has_strings
            && (s.contains("<|") || s.is_empty() || s.contains('\u{fffd}'));
        if is_eos || is_special {
            token_ids.remove(i);
            probs.remove(i);
            token_strings.remove(i);
        } else {
            i += 1;
        }
    }

    // Overlap filter: remove tokens where one string is a prefix of another,
    // or where two tokens have the same string. For duplicate strings, keep
    // the one with the LOWER token ID (deterministic).
    if has_strings {
        let mut keep = vec![true; token_ids.len()];
        for i in 0..token_ids.len() {
            if !keep[i] {
                continue;
            }
            let si = &token_strings[i];
            if si.is_empty() {
                continue;
            }
            for j in (i + 1)..token_ids.len() {
                if !keep[j] {
                    continue;
                }
                let sj = &token_strings[j];
                if sj.is_empty() {
                    continue;
                }
                if sj.starts_with(si.as_str()) || si.starts_with(sj.as_str()) {
                    if si == sj && token_ids[i] != token_ids[j] {
                        if token_ids[i] < token_ids[j] {
                            keep[j] = false;
                        } else {
                            keep[i] = false;
                            break;
                        }
                    } else {
                        keep[j] = false;
                    }
                }
            }
            if !keep[i] {
                continue;
            }
        }
        let mut i = 0;
        while i < token_ids.len() {
            if !keep[i] {
                token_ids.remove(i);
                probs.remove(i);
                token_strings.remove(i);
                keep.remove(i);
            } else {
                i += 1;
            }
        }
    }

    Ok(FilteredDist {
        probs,
        ids: token_ids,
        strings: token_strings,
    })
}

pub(crate) fn find_punctuation_token<S: AsRef<str>>(
    token_ids: &[TokenId],
    token_strings: &[S],
) -> Option<(TokenId, String)> {
    for (i, s) in token_strings.iter().enumerate() {
        let r = s.as_ref();
        let trimmed = r.trim();
        if (trimmed == "." || trimmed == "!" || trimmed == "?") && !r.contains('\n') && !r.contains('\r') {
            return Some((token_ids[i], r.to_string()));
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct StegaConfig {
    pub temperature: f64,
    pub top_k: usize,
    pub max_tokens: usize,
    /// Optional seed for encoder-side RNG (e.g., rejection sampling).
    /// Decoder ignores this field.
    pub seed: Option<u64>,
}

impl Default for StegaConfig {
    fn default() -> Self {
        StegaConfig {
            temperature: 1.0,
            top_k: 300,
            max_tokens: 512,
            seed: None,
        }
    }
}

pub struct ArithmeticStega<'a> {
    lm: &'a dyn LanguageModel,
    config: StegaConfig,
    /// Interval reset period (error-resilient arithmetic coding).
    /// Every `block_size` tokens the interval resets to [0, MAX_VAL), so a
    /// drift-induced bin flip in one block cascades only within that block,
    /// not through the whole message.  0 = no reset.
    block_size: usize,
}

impl<'a> ArithmeticStega<'a> {
    pub fn new(lm: &'a dyn LanguageModel, config: StegaConfig, block_size: usize) -> Self {
        ArithmeticStega { lm, config, block_size }
    }

    fn get_filtered_dist(&self, full_ctx: &[TokenId]) -> Result<(FreqTable, Vec<TokenId>)> {
        let fd = self.get_filtered_dist_full(full_ctx)?;
        Ok((fd.0, fd.1))
    }

    fn get_filtered_dist_full(
        &self,
        full_ctx: &[TokenId],
    ) -> Result<(FreqTable, Vec<TokenId>, Vec<String>)> {
        let filtered = filter_distribution(self.lm, full_ctx, &self.config)?;
        let table = FreqTable::from_probs(&filtered.probs, arithmetic::MAX_FREQ);
        Ok((table, filtered.ids, filtered.strings))
    }

    pub fn encode(
        &self,
        context: &[TokenId],
        message: &[u8],
        num_msg_bits: usize,
    ) -> Result<(Vec<TokenId>, usize, String)> {
        let ctx = context.to_vec();
        let eos = self.lm.eos_token();

        let (mut tokens, bits_consumed, mut strings) = arithmetic::encode_message_to_tokens(
            message, num_msg_bits,
            |tokens| {
                let full_ctx = [ctx.as_slice(), tokens].concat();
                log::debug!(
                    "batch encode: ctx_len={}, tokens_len={}, bit_pos may differ",
                    ctx.len(), tokens.len()
                );
                let (table, token_ids, token_strings) = self.get_filtered_dist_full(&full_ctx)?;
                Ok((table, token_ids, token_strings))
            },
            eos, self.config.max_tokens.max(DEFAULT_MAX_TOKENS), self.block_size,
        )?;

        let needs_punct = strings.last().is_none_or(|s| !crate::arithmetic::is_sentence_end(s));
        if needs_punct {
            let full_ctx = [ctx.as_slice(), tokens.as_slice()].concat();
            if let Ok((_, token_ids, token_strings)) = self.get_filtered_dist_full(&full_ctx)
                && let Some((punct_id, punct_str)) = find_punctuation_token(&token_ids, &token_strings) {
                    tokens.push(punct_id);
                    strings.push(punct_str);
                }
        }
        let generated_text = strings.concat();
        Ok((tokens, bits_consumed, generated_text))
    }

    pub fn decode(
        &self,
        context: &[TokenId],
        tokens: &[TokenId],
        max_message_bits: usize,
    ) -> Result<(Vec<u8>, usize)> {
        let ctx = context.to_vec();
        let (bytes, mut num_bits) = arithmetic::decode_tokens_to_message(tokens, |prefix| {
            let full_ctx = [ctx.as_slice(), prefix].concat();
            let (table, token_ids) = self.get_filtered_dist(&full_ctx)?;
            Ok((table, token_ids, vec![]))
        }, self.block_size)?;
        num_bits = num_bits.min(max_message_bits);
        Ok((bytes, num_bits))
    }

    pub fn decode_text(
        &self,
        context_text: &str,
        cover_text: &str,
        max_message_bits: usize,
    ) -> Result<(Vec<u8>, usize)> {
        let ctx = self.lm.tokenize(context_text)?;
        self.decode_text_inner(&ctx, context_text, cover_text, max_message_bits)
    }

    fn decode_text_inner(
        &self,
        context_prefix: &[TokenId],
        context_text: &str,
        cover_text: &str,
        max_message_bits: usize,
    ) -> Result<(Vec<u8>, usize)> {
        let mut context_tokens = context_prefix.to_vec();
        let mut remaining: &str = if !context_text.is_empty() && cover_text.starts_with(context_text) {
            &cover_text[context_text.len()..]
        } else {
            cover_text
        };

        let mut cur_interval: [u64; 2] = [0, arithmetic::MAX_VAL];
        let mut message_bits: Vec<u8> = Vec::new();
        let mut token_count: usize = 0;

        loop {
            if remaining.is_empty() { break; }
            // Error-resilient block reset: matches the encoder's reset points.
            if self.block_size > 0 && token_count > 0 && token_count % self.block_size == 0 {
                cur_interval = [0, arithmetic::MAX_VAL];
            }
            token_count += 1;
            let full_ctx = context_tokens.clone();
            let (table, token_ids, token_strings) = self.get_filtered_dist_full(&full_ctx)?;

            let mut matched = None;
            let mut best_len = 0usize;
            for (i, token_str) in token_strings.iter().enumerate() {
                if token_str.is_empty() { continue; }
                if token_str.len() > best_len && remaining.starts_with(token_str.as_str()) {
                    matched = Some((i, token_ids[i], token_str.len()));
                    best_len = token_str.len();
                }
            }

            if let Some((idx, token_id, len)) = matched {
                let cum = arithmetic::subdivide(cur_interval[0], cur_interval[1], &table);
                let new_low = if idx > 0 { cum[idx - 1] } else { cur_interval[0] };
                let new_high = cum[idx];
                let low_bits = arithmetic::int_to_bits_msb(new_low, arithmetic::PRECISION);
                let high_bits = arithmetic::int_to_bits_msb(new_high - 1, arithmetic::PRECISION);
                let n_fixed = arithmetic::num_same_from_beg(&low_bits, &high_bits);
                if n_fixed > 0 { message_bits.extend_from_slice(&low_bits[..n_fixed]); }
                let mut nlb = low_bits[n_fixed..].to_vec();
                nlb.resize(arithmetic::PRECISION as usize, 0);
                let mut nhb = high_bits[n_fixed..].to_vec();
                nhb.resize(arithmetic::PRECISION as usize, 1);
                cur_interval[0] = arithmetic::bits_msb_to_int(&nlb);
                cur_interval[1] = arithmetic::bits_msb_to_int(&nhb) + 1;
                context_tokens.push(token_id);
                remaining = &remaining[len..];
            } else { break; }
        }

        let mut bytes = Vec::with_capacity(message_bits.len().div_ceil(8));
        for chunk in message_bits.chunks(8) {
            let mut b = 0u8;
            for (j, &bit) in chunk.iter().enumerate() { b |= bit << (7 - j); }
            bytes.push(b);
        }
        let num_bits = message_bits.len().min(max_message_bits);
        Ok((bytes, num_bits))
    }
}

// ============================================================================
// Streaming arithmetic encode/decode
// ============================================================================
//
// These structs allow incremental encoding and decoding: message bytes are
// fed in chunks and cover text is produced as tokens are generated; cover
// text is fed in chunks and message bytes are recovered as tokens are matched.
// This enables streaming I/O at the CLI level (pipe a large file in, get cover
// text streaming out) without buffering the entire message and cover text.
//
// The caller is responsible for framing: feed framed bytes (from `frame_chunk`
// / `frame_end`) to the encoder, and feed the encoder's output to the
// `UnframeStream` on decode.

/// Streaming arithmetic encoder. Encodes framed bytes into cover text,
/// maintaining interval state across calls.
pub struct ArithmeticStreamEncoder<'a> {
    lm: &'a dyn LanguageModel,
    config: StegaConfig,
    block_size: usize,
    /// Context tokens (context + generated tokens so far).
    context: Vec<TokenId>,
    /// Current arithmetic interval.
    cur_interval: [u64; 2],
    /// Bit buffer: message bits accumulated from framed bytes.
    message_bits: Vec<u8>,
    /// Bit position: how many bits have been consumed.
    bit_pos: usize,
    /// Cover text parts generated so far (for output).
    pub cover_parts: Vec<String>,
    /// Whether `finish` has been called (no more input).
    finished: bool,
}

impl<'a> ArithmeticStreamEncoder<'a> {
    pub fn new(lm: &'a dyn LanguageModel, config: StegaConfig, block_size: usize, context: &[TokenId]) -> Self {
        ArithmeticStreamEncoder {
            lm,
            config,
            block_size,
            context: context.to_vec(),
            cur_interval: [0, arithmetic::MAX_VAL],
            message_bits: Vec::new(),
            bit_pos: 0,
            cover_parts: Vec::new(),
            finished: false,
        }
    }

    /// Feed framed bytes and return any new cover text generated.
    ///
    /// The bytes should be the output of `frame_chunk` (or `frame_end` for
    /// the final call, followed by `finish`).
    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<String> {
        if self.finished {
            return Ok(String::new());
        }
        // Convert bytes to bits and append.
        for &byte in bytes {
            for j in (0..8).rev() {
                self.message_bits.push((byte >> j) & 1);
            }
        }
        self.encode_step(false)
    }

    /// Signal that no more bytes are coming. Flushes the arithmetic interval
    /// and returns the final cover text (including punctuation if needed).
    pub fn finish(&mut self) -> Result<String> {
        if self.finished {
            return Ok(String::new());
        }
        self.finished = true;
        let mut cover = self.encode_step(true)?;

        // Add punctuation if the last token doesn't end with sentence punctuation.
        if self.cover_parts.last().is_none_or(|s| !arithmetic::is_sentence_end(s)) {
            let filtered = filter_distribution(self.lm, &self.context, &self.config)?;
            if let Some((punct_id, punct_str)) = find_punctuation_token(&filtered.ids, &filtered.strings) {
                self.context.push(punct_id);
                self.cover_parts.push(punct_str.clone());
                cover.push_str(&punct_str);
            }
        }
        Ok(cover)
    }

    /// Run the encode loop, generating as many tokens as possible.
    ///
    /// If `flush` is false, leave a 32-bit reserve (don't consume the last
    /// 32 bits, because more real bits may arrive). If `flush` is true,
    /// encode until the stop condition (32 bits past the end with zero
    /// padding).
    fn encode_step(&mut self, flush: bool) -> Result<String> {
        let eos = self.lm.eos_token();
        let max_tokens = self.config.max_tokens.max(arithmetic::DEFAULT_MAX_TOKENS);
        let new_cover_start = self.cover_parts.len();

        // During normal operation, reserve 32 bits for the flush.
        // During flush, encode 32 bits past the end (zero-padded).
        let stop_at = if flush {
            self.message_bits.len() + 32
        } else {
            self.message_bits.len().saturating_sub(32)
        };

        for _ in 0..max_tokens {
            // Error-resilient block reset.
            if self.block_size > 0 && !self.cover_parts.is_empty() && self.cover_parts.len() % self.block_size == 0 {
                self.cur_interval = [0, arithmetic::MAX_VAL];
            }

            let filtered = filter_distribution(self.lm, &self.context, &self.config)?;
            if filtered.probs.is_empty() {
                break;
            }
            let table = arithmetic::FreqTable::from_probs(&filtered.probs, arithmetic::MAX_FREQ);
            if table.total == 0 || table.is_empty() {
                break;
            }

            let cum = arithmetic::subdivide(self.cur_interval[0], self.cur_interval[1], &table);

            // Read PRECISION bits from the bit buffer, padding with zeros.
            let remaining = self.message_bits.len().saturating_sub(self.bit_pos);
            let bits_to_use = remaining.min(arithmetic::PRECISION as usize);
            let mut padded = self.message_bits
                .get(self.bit_pos..self.bit_pos + bits_to_use)
                .unwrap_or(&[])
                .to_vec();
            padded.resize(arithmetic::PRECISION as usize, 0);
            let message_idx = arithmetic::bits_msb_to_int(&padded);

            let selection = arithmetic::find_selection(&cum, message_idx);
            let token = filtered.ids[selection];
            let token_str = filtered.strings.get(selection).cloned().unwrap_or_default();

            let new_low = if selection > 0 { cum[selection - 1] } else { self.cur_interval[0] };
            let new_high = cum[selection];
            let low_bits = arithmetic::int_to_bits_msb(new_low, arithmetic::PRECISION);
            let high_bits = arithmetic::int_to_bits_msb(new_high - 1, arithmetic::PRECISION);
            let n_fixed = arithmetic::num_same_from_beg(&low_bits, &high_bits);
            self.bit_pos += n_fixed;

            let mut nlb = low_bits[n_fixed..].to_vec();
            nlb.resize(arithmetic::PRECISION as usize, 0);
            let mut nhb = high_bits[n_fixed..].to_vec();
            nhb.resize(arithmetic::PRECISION as usize, 1);
            self.cur_interval[0] = arithmetic::bits_msb_to_int(&nlb);
            self.cur_interval[1] = arithmetic::bits_msb_to_int(&nhb) + 1;

            self.context.push(token);
            self.cover_parts.push(token_str);

            if let Some(et) = eos && token == et {
                break;
            }

            // Stop conditions: match the batch encoder's structure (check
            // at the BOTTOM of the loop, after generating a token).
            if self.bit_pos >= stop_at {
                if flush {
                    if self.cover_parts.last().is_some_and(|s| arithmetic::is_sentence_end(s)) {
                        break;
                    }
                    if self.cover_parts.len() > stop_at / 8 + 50 {
                        break;
                    }
                } else {
                    // Non-flush: stop after reaching the reserve threshold.
                    break;
                }
            }
            if flush && self.bit_pos >= self.message_bits.len() + arithmetic::PRECISION as usize * 2 {
                break;
            }
        }

        // Return only the cover text generated in this step.
        Ok(self.cover_parts[new_cover_start..].concat())
    }
}

/// Streaming arithmetic decoder. Matches cover text against token strings,
/// recovers bits from the interval narrowing, and returns decoded bytes.
pub struct ArithmeticStreamDecoder<'a> {
    lm: &'a dyn LanguageModel,
    config: StegaConfig,
    block_size: usize,
    /// Context tokens (context + matched tokens so far).
    context: Vec<TokenId>,
    /// Current arithmetic interval.
    cur_interval: [u64; 2],
    /// Recovered message bits.
    message_bits: Vec<u8>,
    /// Token count (for block resets).
    token_count: usize,
    /// Buffered cover text awaiting token matching.
    text_buffer: String,
    /// Whether `finish` has been called (no more input coming).
    finishing: bool,
    /// Number of bits already returned as complete bytes.
    bits_returned: usize,
}

impl<'a> ArithmeticStreamDecoder<'a> {
    pub fn new(lm: &'a dyn LanguageModel, config: StegaConfig, block_size: usize, context: &[TokenId]) -> Self {
        ArithmeticStreamDecoder {
            lm,
            config,
            block_size,
            context: context.to_vec(),
            cur_interval: [0, arithmetic::MAX_VAL],
            message_bits: Vec::new(),
            token_count: 0,
            text_buffer: String::new(),
            finishing: false,
            bits_returned: 0,
        }
    }

    /// Feed cover text and return any recovered bytes.
    ///
    /// The returned bytes are the raw framed payload (not unframed). The
    /// caller should feed them to an `UnframeStream` to extract message
    /// chunks.
    pub fn push_text(&mut self, text: &str) -> Result<Vec<u8>> {
        self.text_buffer.push_str(text);
        self.decode_step()
    }

    /// Signal that no more cover text is coming. Process any remaining
    /// buffered text and return final recovered bytes.
    pub fn finish(&mut self) -> Result<Vec<u8>> {
        self.finishing = true;
        self.decode_step()?;
        // Return all remaining bits, including the partial last byte
        // (padded with zeros). These are read-ahead padding bits that the
        // unframer will ignore after the end-of-stream marker.
        let remaining = &self.message_bits[self.bits_returned..];
        let mut bytes = Vec::with_capacity(remaining.len().div_ceil(8));
        for chunk in remaining.chunks(8) {
            let mut b = 0u8;
            for (j, &bit) in chunk.iter().enumerate() {
                b |= bit << (7 - j);
            }
            bytes.push(b);
        }
        self.bits_returned = self.message_bits.len();
        Ok(bytes)
    }

    fn decode_step(&mut self) -> Result<Vec<u8>> {
        loop {
            if self.text_buffer.is_empty() {
                break;
            }
            // Error-resilient block reset.
            if self.block_size > 0 && self.token_count > 0 && self.token_count % self.block_size == 0 {
                self.cur_interval = [0, arithmetic::MAX_VAL];
            }

            let filtered = filter_distribution(self.lm, &self.context, &self.config)?;
            if filtered.probs.is_empty() {
                break;
            }
            let table = arithmetic::FreqTable::from_probs(&filtered.probs, arithmetic::MAX_FREQ);
            if table.total == 0 || table.is_empty() {
                break;
            }

            // Find the token whose string matches the beginning of the buffer.
            let mut matched = None;
            let mut best_len = 0usize;
            for (i, token_str) in filtered.strings.iter().enumerate() {
                if token_str.is_empty() {
                    continue;
                }
                if token_str.len() > best_len && self.text_buffer.starts_with(token_str.as_str()) {
                    matched = Some((i, filtered.ids[i], token_str.len()));
                    best_len = token_str.len();
                }
            }

            if let Some((idx, token_id, len)) = matched {
                let cum = arithmetic::subdivide(self.cur_interval[0], self.cur_interval[1], &table);
                let new_low = if idx > 0 { cum[idx - 1] } else { self.cur_interval[0] };
                let new_high = cum[idx];
                let low_bits = arithmetic::int_to_bits_msb(new_low, arithmetic::PRECISION);
                let high_bits = arithmetic::int_to_bits_msb(new_high - 1, arithmetic::PRECISION);
                let n_fixed = arithmetic::num_same_from_beg(&low_bits, &high_bits);
                if n_fixed > 0 {
                    self.message_bits.extend_from_slice(&low_bits[..n_fixed]);
                }
                let mut nlb = low_bits[n_fixed..].to_vec();
                nlb.resize(arithmetic::PRECISION as usize, 0);
                let mut nhb = high_bits[n_fixed..].to_vec();
                nhb.resize(arithmetic::PRECISION as usize, 1);
                self.cur_interval[0] = arithmetic::bits_msb_to_int(&nlb);
                self.cur_interval[1] = arithmetic::bits_msb_to_int(&nhb) + 1;

                self.context.push(token_id);
                self.token_count += 1;
                self.text_buffer.drain(..len);
            } else {
                // No token matched. This could mean:
                // 1. Not enough text yet (need more input) -- buffer and wait.
                // 2. The remaining text is trailing whitespace/garbage.
                // During push_text, we can't distinguish these, so we break
                // and wait for more input. During finish (no more input
                // coming), skip unmatched characters.
                if self.finishing {
                    let skip = self.text_buffer.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                    if skip >= self.text_buffer.len() {
                        self.text_buffer.clear();
                    } else {
                        self.text_buffer.drain(..skip);
                    }
                } else {
                    break;
                }
            }
        }

        // Convert recovered bits to complete bytes only. Partial bits
        // (not a multiple of 8) are kept for the next call. This prevents
        // returning garbage bytes when bit recovery doesn't align to byte
        // boundaries across streaming calls.
        let total_bits = self.message_bits.len();
        let complete_bits = (total_bits / 8) * 8;
        let new_bits = &self.message_bits[self.bits_returned..complete_bits];
        let mut bytes = Vec::with_capacity(new_bits.len() / 8);
        for chunk in new_bits.chunks(8) {
            let mut b = 0u8;
            for (j, &bit) in chunk.iter().enumerate() {
                b |= bit << (7 - j);
            }
            bytes.push(b);
        }
        self.bits_returned = complete_bits;
        Ok(bytes)
    }
}

pub struct BlockStega<'a> {
    lm: &'a dyn LanguageModel,
    config: StegaConfig,
    block_bits: usize,
    token_to_bin: Vec<usize>,
}

impl<'a> BlockStega<'a> {
    pub fn new(lm: &'a dyn LanguageModel, config: StegaConfig, block_bits: usize) -> Result<Self> {
        let vocab_size = lm.vocab_size();
        let num_bins = 1usize << block_bits;
        let mut rng = StdRng::seed_from_u64(0x53544147414E4F00);
        let mut token_to_bin = vec![0usize; vocab_size];
        for bin in token_to_bin.iter_mut() { *bin = rng.gen_range(0..num_bins); }
        Ok(BlockStega { lm, config, block_bits, token_to_bin })
    }

    pub fn encode(&self, context: &[TokenId], message: &[u8], num_msg_bits: usize) -> Result<(Vec<TokenId>, usize, String)> {
        let mut ctx = context.to_vec();
        let mut reader = BitReader::new(message);
        let mut tokens = Vec::new();
        let mut cover_parts: Vec<String> = Vec::new();
        let mut bits_consumed = 0;
        let eos = self.lm.eos_token();
        for _ in 0..self.config.max_tokens {
            if bits_consumed + self.block_bits > num_msg_bits {
                break;
            }
            let block = reader.read_bits(self.block_bits)? as usize;
            bits_consumed += self.block_bits;
            let filtered = filter_distribution(self.lm, &ctx, &self.config)?;
            let mut best_token = None;
            let mut best_prob = f64::NEG_INFINITY;
            let mut best_str = String::new();
            for (i, (&id, &prob)) in filtered.ids.iter().zip(filtered.probs.iter()).enumerate() {
                let id_usize = id as usize;
                if id_usize >= self.token_to_bin.len() {
                    continue;
                }
                if self.token_to_bin[id_usize] != block {
                    continue;
                }
                if prob > best_prob {
                    best_prob = prob;
                    best_token = Some(id);
                    best_str = filtered.strings[i].clone();
                }
            }
            let token = match best_token {
                Some(t) => t,
                None => {
                    let mut fallback_id = 0u32;
                    let mut fallback_prob = f64::NEG_INFINITY;
                    for (i, (&id, &prob)) in filtered.ids.iter().zip(filtered.probs.iter()).enumerate() {
                        if prob > fallback_prob {
                            fallback_prob = prob;
                            fallback_id = id;
                            best_str = filtered.strings[i].clone();
                        }
                    }
                    fallback_id
                }
            };
            tokens.push(token);
            cover_parts.push(best_str);
            ctx.push(token);
            if let Some(et) = eos && token == et {
                break;
            }
        }
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

    pub fn decode(&self, _context: &[TokenId], tokens: &[TokenId], _max_message_bits: usize) -> Result<(Vec<u8>, usize)> {
        let mut writer = BitWriter::new();
        for &token in tokens {
            let id = token as usize;
            if id < self.token_to_bin.len() { writer.write_bits(self.token_to_bin[id] as u64, self.block_bits)?; }
            else { writer.write_bits(0, self.block_bits)?; }
        }
        let bytes = writer.finalize();
        let n = bytes.len() * 8;
        Ok((bytes, n))
    }

    pub fn decode_text(
        &self,
        context_text: &str,
        cover_text: &str,
        _max_message_bits: usize,
    ) -> Result<(Vec<u8>, usize)> {
        let context_prefix = self.lm.tokenize(context_text)?;
        let mut context_tokens = context_prefix.to_vec();
        let mut writer = BitWriter::new();
        let mut remaining: &str = if !context_text.is_empty() && cover_text.starts_with(context_text) {
            &cover_text[context_text.len()..]
        } else {
            cover_text
        };
        for _ in 0..self.config.max_tokens {
            if remaining.is_empty() {
                break;
            }
            let filtered = filter_distribution(self.lm, &context_tokens, &self.config)?;
            let mut matched = None;
            let mut best_len = 0usize;
            for (i, token_str) in filtered.strings.iter().enumerate() {
                if token_str.is_empty() {
                    continue;
                }
                if token_str.len() >= best_len && remaining.starts_with(token_str.as_str()) {
                    let id = filtered.ids[i] as usize;
                    if id < self.token_to_bin.len() && token_str.len() > best_len {
                        matched = Some((id, token_str.len()));
                        best_len = token_str.len();
                    }
                }
            }
            if let Some((id, len)) = matched {
                let bin = self.token_to_bin[id];
                writer.write_bits(bin as u64, self.block_bits)?;
                context_tokens.push(id as TokenId);
                remaining = &remaining[len..];
            } else {
                let skip = remaining.chars().next().map(|c| c.len_utf8()).unwrap_or(1);
                remaining = &remaining[skip.min(remaining.len())..];
            }
        }
        let bytes = writer.finalize();
        let n2 = bytes.len() * 8;
        Ok((bytes, n2))
    }
}

pub struct HuffmanStega<'a> {
    lm: &'a dyn LanguageModel,
    config: StegaConfig,
    max_code_len: usize,
}

impl<'a> HuffmanStega<'a> {
    pub fn new(lm: &'a dyn LanguageModel, config: StegaConfig, max_code_len: usize) -> Self {
        assert!(max_code_len <= 63, "max_code_len must be ≤ 63 (u64 storage)");
        HuffmanStega { lm, config, max_code_len }
    }

    // =================================================================
    // Length-limited Huffman via package-merge (Larmore-Hirschberg)
    // =================================================================

    /// Package-merge: compute optimal codeword lengths subject to max_len ≤ cap.
    fn length_limited_lengths(probs: &[f64], cap: usize) -> Vec<usize> {
        let n = probs.len();
        if n == 1 {
            return vec![1usize];
        }
        // Minimum cap is ceil(log2(n)).  Clamp if necessary.
        let min_l = (n as f64).log2().ceil() as usize;
        let max_level = cap.max(min_l);

        // Leaves: (weight, symbol_set)
        let leaves: Vec<(f64, Vec<usize>)> =
            (0..n).map(|i| (probs[i], vec![i])).collect();
        let mut packages: Vec<(f64, Vec<usize>)> = Vec::new();

        for level in 1..=max_level {
            // Merge leaves + packages from previous level, sort ascending.
            let mut current: Vec<(f64, Vec<usize>)> = Vec::with_capacity(leaves.len() + packages.len());
            current.extend_from_slice(&leaves);
            current.extend(packages.drain(..));
            current.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            if level == max_level {
                // Select the (2n - 2) smallest items.
                let selected = &current[..current.len().min(2 * n - 2)];
                let mut lengths = vec![0usize; n];
                for (_, syms) in selected {
                    for &s in syms {
                        lengths[s] += 1;
                    }
                }
                return lengths;
            }

            // Pair consecutive items into packages for the next level.
            let mut j = 0;
            while j + 1 < current.len() {
                let (wa, sa) = &current[j];
                let (wb, sb) = &current[j + 1];
                let mut merged = sa.clone();
                merged.extend_from_slice(sb);
                packages.push((wa + wb, merged));
                j += 2;
            }
            // Odd leftover is dropped (still selectable at level L).
        }
        unreachable!("package-merge loop must return at max_level");
    }

    /// Assign canonical prefix-free codes from optimal lengths.
    fn canonical_codes(lengths: &[usize], token_ids: &[TokenId]) -> Vec<(u64, usize)> {
        let n = lengths.len();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            lengths[a]
                .cmp(&lengths[b])
                .then(token_ids[a].cmp(&token_ids[b]))
        });

        let mut codes = vec![(0u64, 0usize); n];
        let mut code: u64 = 0;
        let mut prev_len = 0;
        for &i in &order {
            let len = lengths[i];
            if prev_len > 0 {
                code = (code + 1) << (len - prev_len);
            }
            codes[i] = (code, len);
            prev_len = len;
        }
        codes
    }

    fn build_huffman_tree(&self, probs: &[f64], token_ids: &[TokenId]) -> Vec<(u64, usize)> {
        if probs.is_empty() {
            return vec![];
        }
        let max_len = self.max_code_len;
        let lengths = Self::length_limited_lengths(probs, max_len);
        Self::canonical_codes(&lengths, token_ids)
    }

    pub fn encode(&self, context: &[TokenId], message: &[u8], num_msg_bits: usize) -> Result<(Vec<TokenId>, usize, String)> {
        let mut ctx = context.to_vec();
        let mut reader = BitReader::new(message);
        let mut tokens = Vec::new();
        let mut cover_parts: Vec<String> = Vec::new();
        let mut bits_consumed = 0;
        let eos = self.lm.eos_token();

        // Safety cap: at most enough tokens to encode all message bits
        // at the minimum code length (1 bit per token).
        let max_tokens = (num_msg_bits + 1).min(self.config.max_tokens.max(DEFAULT_MAX_TOKENS));
        for _ in 0..max_tokens {
            if bits_consumed >= num_msg_bits {
                break;
            }
            let filtered = filter_distribution(self.lm, &ctx, &self.config)?;
            if filtered.probs.is_empty() { break; }
            let huff_codes = self.build_huffman_tree(&filtered.probs, &filtered.ids);
            if huff_codes.is_empty() { break; }
            let mut matched = None;
            let mut code_buf = 0u64;
            let mut code_len = 0;
            // Read up to max_code_len bits, allowing the last code to finish
            // by reading into the bit-stream's zero-padding after the message.
            while code_len < self.max_code_len {
                let bit = reader.read_bit()?;
                code_buf = (code_buf << 1) | bit as u64;
                bits_consumed += 1;
                code_len += 1;
                for (i, &(code, len)) in huff_codes.iter().enumerate() {
                    if len == code_len && code == code_buf { matched = Some((i, code_len)); break; }
                }
                if matched.is_some() { break; }
            }
            let (token_idx, _) = match matched {
                Some((idx, len)) => (idx, len),
                None => unreachable!(
                    "length-limited Huffman code is complete; bit-walker must match within {} bits",
                    self.max_code_len
                ),
            };
            let token = filtered.ids[token_idx];
            let token_str = filtered.strings.get(token_idx).cloned().unwrap_or_default();
            tokens.push(token); cover_parts.push(token_str); ctx.push(token);
            if let Some(et) = eos && token == et { break; }
        }
        let filtered = filter_distribution(self.lm, &ctx, &self.config)?;
        if let Some((punct_id, punct_str)) = find_punctuation_token(&filtered.ids, &filtered.strings) {
            tokens.push(punct_id); cover_parts.push(punct_str);
        }
        let generated_text = cover_parts.concat();
        Ok((tokens, bits_consumed, generated_text))
    }

    pub fn decode(&self, context: &[TokenId], tokens: &[TokenId], _max_message_bits: usize) -> Result<(Vec<u8>, usize)> {
        let mut ctx = context.to_vec();
        let mut writer = BitWriter::new();
        for &token in tokens {
            let filtered = filter_distribution(self.lm, &ctx, &self.config)?;
            let codes = self.build_huffman_tree(&filtered.probs, &filtered.ids);
            if let Some(pos) = filtered.ids.iter().position(|&t| t == token)
                && pos < codes.len() {
                    let (code, code_len) = codes[pos];
                    if code_len > 0 { writer.write_bits(code, code_len)?; }
                }
            ctx.push(token);
        }
        let bytes = writer.finalize();
        let n3 = bytes.len() * 8;
        Ok((bytes, n3))
    }

    pub fn decode_text(
        &self,
        context_text: &str,
        cover_text: &str,
        _max_message_bits: usize,
    ) -> Result<(Vec<u8>, usize)> {
        let context_prefix = self.lm.tokenize(context_text)?;
        let mut context_tokens = context_prefix.to_vec();
        let mut writer = BitWriter::new();
        let mut remaining: &str = if !context_text.is_empty() && cover_text.starts_with(context_text) {
            &cover_text[context_text.len()..]
        } else {
            cover_text
        };
        for _ in 0..self.config.max_tokens {
            if remaining.is_empty() {
                break;
            }
            let filtered = filter_distribution(self.lm, &context_tokens, &self.config)?;
            let codes = self.build_huffman_tree(&filtered.probs, &filtered.ids);
            // Greedy longest-match on token strings, like Block/rejection
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
                if idx < codes.len() {
                    let (code, code_len) = codes[idx];
                    if code_len > 0 {
                        writer.write_bits(code, code_len)?;
                    }
                }
                context_tokens.push(filtered.ids[idx]);
                remaining = &remaining[len..];
            } else {
                break;
            }
        }
        let bytes = writer.finalize();
        let n2 = bytes.len() * 8;
        Ok((bytes, n2))
    }
}

pub enum StegaMethod<'a> {
    Arithmetic(ArithmeticStega<'a>),
    Block(BlockStega<'a>),
    Huffman(HuffmanStega<'a>),
    Rejection(RejectionStega<'a>),
}

impl<'a> StegaMethod<'a> {
    pub fn encode(&self, context: &[TokenId], message: &[u8], num_msg_bits: usize) -> Result<(Vec<TokenId>, usize, String)> {
        match self { StegaMethod::Arithmetic(s) => s.encode(context, message, num_msg_bits), StegaMethod::Block(s) => s.encode(context, message, num_msg_bits), StegaMethod::Huffman(s) => s.encode(context, message, num_msg_bits), StegaMethod::Rejection(s) => s.encode(context, message, num_msg_bits) }
    }
    pub fn decode(&self, context: &[TokenId], tokens: &[TokenId], max_message_bits: usize) -> Result<(Vec<u8>, usize)> {
        match self { StegaMethod::Arithmetic(s) => s.decode(context, tokens, max_message_bits), StegaMethod::Block(s) => s.decode(context, tokens, max_message_bits), StegaMethod::Huffman(s) => s.decode(context, tokens, max_message_bits), StegaMethod::Rejection(s) => s.decode(context, tokens, max_message_bits) }
    }
    pub fn decode_text(&self, context_text: &str, cover_text: &str, max_message_bits: usize) -> Result<(Vec<u8>, usize)> {
        match self { StegaMethod::Arithmetic(s) => s.decode_text(context_text, cover_text, max_message_bits), StegaMethod::Block(s) => s.decode_text(context_text, cover_text, max_message_bits), StegaMethod::Huffman(s) => s.decode_text(context_text, cover_text, max_message_bits), StegaMethod::Rejection(s) => s.decode_text(context_text, cover_text, max_message_bits) }
    }
}

#[cfg(test)]
mod stream_tests {
    use super::*;
    use crate::framing::{frame_chunk, frame_end, frame_message, UnframeStream};
    use crate::lm::{DummyLM, LanguageModel};

    fn make_lm(vocab: usize) -> DummyLM {
        let probs: Vec<f64> = (0..vocab)
            .map(|i| 1.0 / (i as f64 + 1.0))
            .collect();
        let strings: Vec<String> = (0..vocab)
            .map(|i| format!("t{} ", i))
            .collect();
        DummyLM::new(vocab)
            .with_probs(probs)
            .with_token_strings(strings)
    }

    #[test]
    fn test_stream_vs_batch_identical_output() {
        let vocab = 64;
        let lm = make_lm(vocab);
        let config = StegaConfig {
            temperature: 2.0,
            top_k: 50,
            max_tokens: 500,
            seed: None,
        };
        let ctx = lm.tokenize("ctx").unwrap();

        let secret = b"Hello streaming world test message";

        // Batch encode
        let payload = frame_message(secret);
        let msg = crate::bitstream::Message::from_bytes(payload.clone());
        let stega = ArithmeticStega::new(&lm, config.clone(), 16);
        let (batch_tokens, _, batch_cover) = stega.encode(&ctx, msg.data(), msg.num_bits()).unwrap();

        // Streaming encode
        let mut encoder = ArithmeticStreamEncoder::new(&lm, config, 16, &ctx);
        let mut cover = String::new();
        cover.push_str(&encoder.push_bytes(&frame_chunk(secret)).unwrap());
        cover.push_str(&encoder.push_bytes(&frame_end()).unwrap());
        cover.push_str(&encoder.finish().unwrap());

        assert_eq!(cover, batch_cover, "streaming cover text differs from batch cover text");
        assert_eq!(encoder.cover_parts.len(), batch_tokens.len(), "token count differs");
    }

    #[test]
    fn test_stream_encode_decode_single_chunk() {
        let vocab = 64;
        let lm = make_lm(vocab);
        let config = StegaConfig {
            temperature: 2.0,
            top_k: 50,
            max_tokens: 500,
            seed: None,
        };
        let ctx = lm.tokenize("ctx").unwrap();

        let secret = b"Hello streaming world";

        let mut encoder = ArithmeticStreamEncoder::new(&lm, config.clone(), 0, &ctx);
        let mut cover = String::new();
        cover.push_str(&encoder.push_bytes(&frame_chunk(secret)).unwrap());
        cover.push_str(&encoder.push_bytes(&frame_end()).unwrap());
        cover.push_str(&encoder.finish().unwrap());
        assert!(!cover.is_empty());

        let mut decoder = ArithmeticStreamDecoder::new(&lm, config, 0, &ctx);
        let recovered = decoder.push_text(&cover).unwrap();
        let final_bytes = decoder.finish().unwrap();
        let all_bytes = [recovered.as_slice(), final_bytes.as_slice()].concat();

        let mut unframer = UnframeStream::new();
        let chunks = unframer.push(&all_bytes).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], secret);
        assert!(unframer.is_done());
    }

    #[test]
    fn test_stream_encode_decode_multi_chunk() {
        let vocab = 64;
        let lm = make_lm(vocab);
        let config = StegaConfig {
            temperature: 2.0,
            top_k: 50,
            max_tokens: 1000,
            seed: None,
        };
        let ctx = lm.tokenize("ctx").unwrap();

        let chunk1 = b"First chunk of data";
        let chunk2 = b"Second chunk here";
        let chunk3 = b"Third and final";

        let mut encoder = ArithmeticStreamEncoder::new(&lm, config.clone(), 0, &ctx);
        let mut cover = String::new();
        cover.push_str(&encoder.push_bytes(&frame_chunk(chunk1)).unwrap());
        cover.push_str(&encoder.push_bytes(&frame_chunk(chunk2)).unwrap());
        cover.push_str(&encoder.push_bytes(&frame_chunk(chunk3)).unwrap());
        cover.push_str(&encoder.push_bytes(&frame_end()).unwrap());
        cover.push_str(&encoder.finish().unwrap());

        let mut decoder = ArithmeticStreamDecoder::new(&lm, config, 0, &ctx);
        let recovered = decoder.push_text(&cover).unwrap();
        let final_bytes = decoder.finish().unwrap();
        let all_bytes = [recovered.as_slice(), final_bytes.as_slice()].concat();

        let mut unframer = UnframeStream::new();
        let chunks = unframer.push(&all_bytes).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], chunk1);
        assert_eq!(chunks[1], chunk2);
        assert_eq!(chunks[2], chunk3);
        assert!(unframer.is_done());
    }

    #[test]
    fn test_stream_encode_decode_binary_with_nulls() {
        let vocab = 64;
        let lm = make_lm(vocab);
        let config = StegaConfig {
            temperature: 2.0,
            top_k: 50,
            max_tokens: 500,
            seed: None,
        };
        let ctx = lm.tokenize("ctx").unwrap();

        let secret: Vec<u8> = vec![0x00, 0xAB, 0x00, 0x00, 0xCD, 0xFF, 0x00, 0x42];

        let mut encoder = ArithmeticStreamEncoder::new(&lm, config.clone(), 0, &ctx);
        let mut cover = String::new();
        cover.push_str(&encoder.push_bytes(&frame_chunk(&secret)).unwrap());
        cover.push_str(&encoder.push_bytes(&frame_end()).unwrap());
        cover.push_str(&encoder.finish().unwrap());

        let mut decoder = ArithmeticStreamDecoder::new(&lm, config, 0, &ctx);
        let recovered = decoder.push_text(&cover).unwrap();
        let final_bytes = decoder.finish().unwrap();
        let all_bytes = [recovered.as_slice(), final_bytes.as_slice()].concat();

        let mut unframer = UnframeStream::new();
        let chunks = unframer.push(&all_bytes).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], secret);
        assert!(unframer.is_done());
    }

    #[test]
    fn test_stream_decode_increments() {
        // Feed cover text one character at a time to the decoder.
        let vocab = 64;
        let lm = make_lm(vocab);
        let config = StegaConfig {
            temperature: 2.0,
            top_k: 50,
            max_tokens: 500,
            seed: None,
        };
        let ctx = lm.tokenize("ctx").unwrap();

        let secret = b"Incremental decode test";

        let mut encoder = ArithmeticStreamEncoder::new(&lm, config.clone(), 0, &ctx);
        let mut cover = String::new();
        cover.push_str(&encoder.push_bytes(&frame_chunk(secret)).unwrap());
        cover.push_str(&encoder.push_bytes(&frame_end()).unwrap());
        cover.push_str(&encoder.finish().unwrap());

        // Decode: feed one character at a time.
        let mut decoder = ArithmeticStreamDecoder::new(&lm, config, 0, &ctx);
        let mut all_bytes = Vec::new();
        for ch in cover.chars() {
            let bytes = decoder.push_text(&ch.to_string()).unwrap();
            all_bytes.extend(bytes);
        }
        let final_bytes = decoder.finish().unwrap();
        all_bytes.extend(final_bytes);

        let mut unframer = UnframeStream::new();
        let chunks = unframer.push(&all_bytes).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], secret);
        assert!(unframer.is_done());
    }
}

#[cfg(test)]
mod huffman_tests {
    use super::*;

    /// Check that Kraft equality Σ 2^(-length_i) == 1.0 within tolerance.
    fn kraft_sum(lengths: &[usize]) -> f64 {
        lengths.iter().map(|&l| 2.0_f64.powi(-(l as i32))).sum()
    }

    /// Check that no codeword is a prefix of another.
    fn is_prefix_free(codes: &[(u64, usize)]) -> bool {
        for i in 0..codes.len() {
            let (ci, li) = codes[i];
            if li == 0 { continue; }
            for j in 0..codes.len() {
                if i == j { continue; }
                let (cj, lj) = codes[j];
                if lj == 0 { continue; }
                // Check if ci is a prefix of cj
                let shorter_len = li.min(lj);
                if (ci >> (li.saturating_sub(shorter_len))) == (cj >> (lj.saturating_sub(shorter_len))) {
                    return false;
                }
            }
        }
        true
    }

    fn build_and_verify(probs: &[f64], token_ids: &[TokenId], max_len: usize) {
        let lengths = HuffmanStega::length_limited_lengths(probs, max_len);
        let codes = HuffmanStega::canonical_codes(&lengths, token_ids);

        // 1. Kraft equality
        let k = kraft_sum(&lengths);
        assert!(
            (k - 1.0).abs() < 1e-9,
            "Kraft sum = {:.12}, expected 1.0",
            k
        );

        // 2. Prefix-free
        assert!(is_prefix_free(&codes), "codes are not prefix-free");

        // 3. Max length
        for &l in &lengths {
            assert!(l <= max_len, "length {} > max {}", l, max_len);
        }

        // 4. Determinism
        let codes2 = HuffmanStega::canonical_codes(&lengths, token_ids);
        assert_eq!(codes, codes2, "canonical_codes not deterministic");
    }

    #[test]
    fn test_n1() {
        let probs = vec![1.0];
        let ids = vec![42u32];
        let lengths = HuffmanStega::length_limited_lengths(&probs, 10);
        assert_eq!(lengths, vec![1]);
        let codes = HuffmanStega::canonical_codes(&lengths, &ids);
        assert_eq!(codes, vec![(0, 1)]);
    }

    #[test]
    fn test_n2() {
        let probs = vec![0.7, 0.3];
        let ids = vec![10, 20];
        build_and_verify(&probs, &ids, 8);
        // Both should have length 1
        let lengths = HuffmanStega::length_limited_lengths(&probs, 8);
        assert_eq!(lengths, vec![1, 1]);
    }

    #[test]
    fn test_uniform_power_of_2() {
        for k in 1..=4 {
            let n = 1 << k;
            let prob = 1.0 / n as f64;
            let probs: Vec<f64> = vec![prob; n];
            let ids: Vec<u32> = (0..n as u32).collect();
            build_and_verify(&probs, &ids, 10);
            let lengths = HuffmanStega::length_limited_lengths(&probs, 10);
            for &l in &lengths {
                assert_eq!(l, k, "uniform n={} should have all length {}", n, k);
            }
        }
    }

    #[test]
    fn test_skewed_distribution() {
        // A distribution where natural Huffman depth > 8
        let probs = vec![
            0.90, 0.03, 0.02, 0.01, 0.005, 0.005, 0.004, 0.004,
            0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002,
            0.001, 0.001, 0.001, 0.001,
        ];
        let ids: Vec<u32> = (0..probs.len() as u32).collect();
        // With L=8 (old default), this would fail. With L=16 it works.
        build_and_verify(&probs, &ids, 16);
    }

    #[test]
    fn test_completeness_always_match() {
        // Enumerate all 2^L bit patterns and verify each matches a codeword
        let probs = vec![0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05];
        let ids: Vec<u32> = (0..probs.len() as u32).collect();
        #[allow(non_snake_case)]
        let L = 6;
        let lengths = HuffmanStega::length_limited_lengths(&probs, L);
        let codes = HuffmanStega::canonical_codes(&lengths, &ids);

        for pattern in 0..(1u64 << L) {
            let mut match_count = 0;
            for &(code, len) in &codes {
                if len == 0 { continue; }
                // Take the top `len` bits of the pattern
                let pattern_prefix = pattern >> (L - len);
                if pattern_prefix == code {
                    match_count += 1;
                }
            }
            assert!(
                match_count >= 1,
                "bit pattern 0x{:x} matched no codeword",
                pattern
            );
            assert!(
                match_count <= 1,
                "bit pattern 0x{:x} matched {} codewords (not prefix-free)",
                pattern, match_count
            );
        }
    }

    #[test]
    fn test_determinism() {
        let probs = vec![0.4, 0.3, 0.2, 0.1];
        let ids = vec![100, 200, 300, 400];
        let c1 = HuffmanStega::canonical_codes(
            &HuffmanStega::length_limited_lengths(&probs, 8), &ids);
        let c2 = HuffmanStega::canonical_codes(
            &HuffmanStega::length_limited_lengths(&probs, 8), &ids);
        assert_eq!(c1, c2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_minimum_L_boundary() {
        // With n=300 and L=9 (the minimum), should still produce a valid code.
        let mut probs: Vec<f64> = Vec::with_capacity(300);
        let mut remaining = 1.0;
        for _i in 0..299 {
            let p = remaining * 0.1;
            probs.push(p);
            remaining -= p;
        }
        probs.push(remaining);
        let ids: Vec<u32> = (0..300).collect();
        build_and_verify(&probs, &ids, 9);
    }

    #[test]
    fn test_clamp_up() {
        // User passes cap=4 which is below ceil(log2(300))=9.
        // The code must clamp up to 9 and produce a valid complete code.
        let mut probs: Vec<f64> = Vec::with_capacity(300);
        let mut remaining = 1.0;
        for _i in 0..299 {
            let p = remaining * 0.1;
            probs.push(p);
            remaining -= p;
        }
        probs.push(remaining);
        let ids: Vec<u32> = (0..300).collect();
        // This calls length_limited_lengths with cap=4, which clamps to 9.
        let lengths = super::HuffmanStega::length_limited_lengths(&probs, 4);
        assert!(lengths.iter().all(|&l| l <= 9), "all lengths must be ≤ 9");
        let k: f64 = lengths.iter().map(|&l| 2.0_f64.powi(-(l as i32))).sum();
        assert!((k - 1.0).abs() < 1e-9, "Kraft sum = {:.12}", k);
        let codes = super::HuffmanStega::canonical_codes(&lengths, &ids);
        assert!(is_prefix_free(&codes), "codes must be prefix-free");
    }

    #[test]
    fn test_roundtrip_skewed() {
        
        use crate::lm::DummyLM;

        let probs = vec![
            0.90, 0.03, 0.02, 0.01, 0.005, 0.005, 0.004, 0.004,
            0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002,
            0.001, 0.001, 0.001, 0.001,
        ];
        let strings: Vec<String> = (0..probs.len()).map(|i| format!("t{} ", i)).collect();
        let lm = DummyLM::new(probs.len())
            .with_probs(probs)
            .with_token_strings(strings);

        let config = StegaConfig {
            temperature: 1.0,
            top_k: 50,
            max_tokens: 500,
            seed: None,
        };

        let ctx = lm.tokenize("ctx").unwrap();
        let msg = b"Hello Huffman roundtrip test message!";
        let msg_bits = msg.len() * 8;

        // Check Kraft for the ACTUAL distribution the encoder sees
        let fd = filter_distribution(&lm, &ctx, &config).unwrap();
        let lengths = super::HuffmanStega::length_limited_lengths(&fd.probs, 16);
        let k: f64 = lengths.iter().map(|&l| 2.0_f64.powi(-(l as i32))).sum();
        assert!((k - 1.0).abs() < 1e-9, "Kraft sum = {:.12}", k);

        let stega = HuffmanStega::new(&lm, config, 16);
        let (tokens, consumed, _) = stega.encode(&ctx, msg, msg_bits).unwrap();
        assert!(consumed >= msg_bits, "encode must consume at least all message bits (got {})", consumed);
        assert!(consumed <= msg_bits + 16, "encode consumed {} bits, expected ≤ {} (msg + max_code_len)", consumed, msg_bits + 16);

        let (decoded, dbits) = stega.decode(&ctx, &tokens, msg_bits + 16).unwrap();
        assert!(dbits >= msg_bits, "decode must recover at least message bits");

        // Compare decoded bytes directly (stronger than bit-by-bit because
        // it catches padding-bit corruption that bleeds into message bytes).
        assert_eq!(&decoded[..msg.len()], msg, "decoded message differs from original");
    }
}