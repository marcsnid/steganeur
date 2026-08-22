//! Chunked varint framing for binary-safe message transport.
//!
//! The steganographic codecs (arithmetic, block, huffman, rejection) all
//! operate on raw byte slices, but they always emit slightly more bits than
//! the payload contains (the encoder reads past the end into zero padding).
//! The framing layer's job is to tell the decoder where the real message
//! ends so the trailing padding can be discarded.
//!
//! ## Format
//!
//! ```text
//! [varint N1][N1 bytes][varint N2][N2 bytes]...[varint 0]
//! ```
//!
//! - Each chunk is a varint length followed by that many payload bytes.
//! - A varint length of 0 marks end of stream.
//! - Varints use LEB128 (7 bits per byte, MSB = continuation bit).
//!
//! ## Properties
//!
//! - **Binary-safe.** The decoder reads by count, not by scanning for a
//!   sentinel, so `0x00` bytes inside the message are just data. This is the
//!   fix for the old null-terminator framing, which truncated at the first
//!   `0x00` and silently corrupted ciphertext.
//! - **Streaming-capable.** Each chunk is self-delimited. A streaming encoder
//!   can emit chunks as bytes arrive; a streaming decoder can emit each
//!   chunk's bytes as soon as it recovers them. The trailing `varint(0)`
//!   signals end of stream.
//! - **No fixed int size.** Varints are 1 byte for lengths < 128, 2 bytes
//!   for < 16384, scaling logarithmically. There is no cap on message size.
//! - **Minimal overhead.** For a batch message under 128 bytes: 1 byte for
//!   the length varint + 1 byte for the end-of-stream marker = 2 bytes flat.

use crate::error::{Error, Result};

/// Frame a single message as one chunk followed by end-of-stream.
///
/// Produces: `[varint(msg.len())][msg bytes][0x00]`
///
/// This is the batch framing (one chunk). For streaming, call
/// [`encode_varint`] and extend with chunk bytes for each chunk, then push
/// `0x00` at the end.
pub fn frame_message(message: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(message.len() + 4);
    encode_varint(&mut out, message.len() as u64);
    out.extend_from_slice(message);
    out.push(0x00); // end of stream
    out
}

/// Unframe a payload back into message bytes.
///
/// Reads chunks until the end-of-stream marker (`varint 0`). Concatenates all
/// chunk payloads. Bytes past the end-of-stream marker are ignored (they are
/// read-ahead zero padding from the steganographic decoder).
pub fn unframe_payload(payload: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut pos = 0;
    loop {
        let (len, consumed) = decode_varint(&payload[pos..])
            .ok_or_else(|| Error::Steganography("framing: varint extends past payload".into()))?;
        pos += consumed;
        if len == 0 {
            break; // end of stream
        }
        let end = pos
            .checked_add(len as usize)
            .ok_or_else(|| Error::Steganography("framing: chunk length overflow".into()))?;
        if end > payload.len() {
            return Err(Error::Steganography(format!(
                "framing: chunk length {} exceeds remaining payload ({} bytes)",
                len,
                payload.len() - pos
            )));
        }
        out.extend_from_slice(&payload[pos..end]);
        pos = end;
    }
    Ok(out)
}

/// Encode an unsigned integer as LEB128 varint into `out`.
pub fn encode_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let mut byte = (value & 0x7F) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80; // continuation bit
            out.push(byte);
        } else {
            out.push(byte);
            break;
        }
    }
}

/// Decode an LEB128 varint from `data`. Returns `(value, bytes_consumed)`.
///
/// Returns `None` if the varint is incomplete (no terminating byte) or
/// longer than 64 bits.
pub fn decode_varint(data: &[u8]) -> Option<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0u32;
    for (i, &byte) in data.iter().enumerate() {
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((value, i + 1));
        }
        shift += 7;
        if shift >= 64 {
            return None; // varint too long
        }
    }
    None // incomplete varint
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_unframe_text() {
        let msg = b"Meet me at noon";
        let framed = frame_message(msg);
        let unframed = unframe_payload(&framed).unwrap();
        assert_eq!(unframed, msg);
    }

    #[test]
    fn test_frame_unframe_binary_with_nulls() {
        // Ciphertext-like: multiple 0x00 bytes, including leading and internal.
        let msg = [0x00u8, 0xAB, 0x00, 0x00, 0xCD, 0x00, 0xFF, 0x00, 0x00];
        let framed = frame_message(&msg);
        let unframed = unframe_payload(&framed).unwrap();
        assert_eq!(unframed, msg);
    }

    #[test]
    fn test_frame_unframe_all_zeros() {
        let msg = [0x00u8; 32];
        let framed = frame_message(&msg);
        let unframed = unframe_payload(&framed).unwrap();
        assert_eq!(unframed, msg);
    }

    #[test]
    fn test_trailing_padding_ignored() {
        // Simulates decoder read-ahead: framed payload + trailing zero bytes.
        let msg = b"Hello";
        let mut framed = frame_message(msg);
        framed.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        let unframed = unframe_payload(&framed).unwrap();
        assert_eq!(unframed, msg);
    }

    #[test]
    fn test_varint_encoding() {
        let mut out = vec![];
        encode_varint(&mut out, 0);
        assert_eq!(out, vec![0x00]);

        let mut out = vec![];
        encode_varint(&mut out, 127);
        assert_eq!(out, vec![0x7F]);

        let mut out = vec![];
        encode_varint(&mut out, 128);
        assert_eq!(out, vec![0x80, 0x01]);

        let mut out = vec![];
        encode_varint(&mut out, 200);
        assert_eq!(out, vec![0xC8, 0x01]);

        let mut out = vec![];
        encode_varint(&mut out, 100_000);
        assert_eq!(out, vec![0xA0, 0x8D, 0x06]);

        // Round-trip all varint sizes
        for &val in &[0u64, 1, 127, 128, 255, 256, 16383, 16384, 100_000, u32::MAX as u64, u64::MAX] {
            let mut out = vec![];
            encode_varint(&mut out, val);
            let (decoded, consumed) = decode_varint(&out).expect("decode failed");
            assert_eq!(decoded, val, "varint round-trip failed for {}", val);
            assert_eq!(consumed, out.len(), "consumed mismatch for {}", val);
        }
    }

    #[test]
    fn test_large_message() {
        let msg: Vec<u8> = (0..50_000).map(|i| (i % 256) as u8).collect();
        let framed = frame_message(&msg);
        let unframed = unframe_payload(&framed).unwrap();
        assert_eq!(unframed, msg);
    }

    #[test]
    fn test_multiple_chunks_streaming() {
        // Streaming format: multiple chunks then end-of-stream.
        let chunk1 = b"Hello ";
        let chunk2 = b"World!";
        let mut framed = Vec::new();
        encode_varint(&mut framed, chunk1.len() as u64);
        framed.extend_from_slice(chunk1);
        encode_varint(&mut framed, chunk2.len() as u64);
        framed.extend_from_slice(chunk2);
        framed.push(0x00);

        let unframed = unframe_payload(&framed).unwrap();
        assert_eq!(unframed, b"Hello World!");
    }

    #[test]
    fn test_incomplete_varint() {
        // Continuation bit set but no terminating byte.
        assert_eq!(decode_varint(&[0x80]), None);
        assert_eq!(decode_varint(&[]), None);
    }

    #[test]
    fn test_chunk_exceeds_payload() {
        // Varint says 100 bytes but only 5 remain.
        let mut framed = vec![];
        encode_varint(&mut framed, 100);
        framed.extend_from_slice(b"short");
        assert!(unframe_payload(&framed).is_err());
    }

    // ========================================================================
    // End-to-end: framing + steganographic encode/decode round-trip.
    // Verifies that binary payloads with 0x00 bytes survive the full path.
    // ========================================================================

    #[test]
    fn test_e2e_arithmetic_binary_with_nulls() {
        use crate::bitstream::Message;
        use crate::lm::{DummyLM, LanguageModel};
        use crate::steganography::{ArithmeticStega, StegaConfig};

        let vocab = 64usize;
        let strings: Vec<String> = (0..vocab)
            .map(|i| format!("t{} ", i))
            .collect();
        let probs: Vec<f64> = (0..vocab)
            .map(|i| 1.0 / (i as f64 + 1.0))
            .collect();
        let lm = DummyLM::new(vocab)
            .with_probs(probs)
            .with_token_strings(strings);

        let config = StegaConfig {
            temperature: 2.0,
            top_k: 50,
            max_tokens: 500,
            seed: None,
        };
        let stega = ArithmeticStega::new(&lm, config, 0);

        // Binary payload with null bytes: leading, internal, and trailing.
        let secret: Vec<u8> = vec![0x00, 0xAB, 0x00, 0x00, 0xCD, 0xFF, 0x00, 0x42];
        let payload = frame_message(&secret);
        let msg = Message::from_bytes(payload.clone());

        let ctx = lm.tokenize("ctx").unwrap();
        let (tokens, bits_consumed, _text) =
            stega.encode(&ctx, msg.data(), msg.num_bits()).unwrap();
        assert!(bits_consumed >= msg.num_bits(),
            "encode consumed {} < {} bits", bits_consumed, msg.num_bits());

        let (raw_decoded, _) = stega.decode(&ctx, &tokens, msg.num_bits() + 64).unwrap();
        let decoded = unframe_payload(&raw_decoded).unwrap();

        assert_eq!(decoded, secret, "binary payload with nulls did not round-trip");
    }

    #[test]
    fn test_e2e_block_binary_with_nulls() {
        use crate::bitstream::Message;
        use crate::lm::{DummyLM, LanguageModel};
        use crate::steganography::{BlockStega, StegaConfig};

        let vocab = 64usize;
        let strings: Vec<String> = (0..vocab)
            .map(|i| format!("t{} ", i))
            .collect();
        let probs: Vec<f64> = (0..vocab)
            .map(|i| 1.0 / (i as f64 + 1.0))
            .collect();
        let lm = DummyLM::new(vocab)
            .with_probs(probs)
            .with_token_strings(strings);

        let config = StegaConfig {
            temperature: 2.0,
            top_k: 50,
            max_tokens: 500,
            seed: None,
        };
        let stega = BlockStega::new(&lm, config, 2).unwrap();

        let secret: Vec<u8> = vec![0x00, 0xFF, 0x00, 0x01, 0x00, 0x02];
        let payload = frame_message(&secret);
        let msg = Message::from_bytes(payload.clone());

        let ctx = lm.tokenize("ctx").unwrap();
        let (tokens, bits_consumed, _text) =
            stega.encode(&ctx, msg.data(), msg.num_bits()).unwrap();
        assert!(bits_consumed >= msg.num_bits());

        let (raw_decoded, _) = stega.decode(&ctx, &tokens, msg.num_bits() + 16).unwrap();
        let decoded = unframe_payload(&raw_decoded).unwrap();

        assert_eq!(decoded, secret, "block: binary payload with nulls did not round-trip");
    }
}
