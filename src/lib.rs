//! # Steganeur -- Neural Linguistic Steganography
//!
//! Implementation of "Neural Linguistic Steganography" (Ziegler, Deng, Rush, 2019).
//!
//! This library provides:
//!
//! - **Arithmetic coding** with fixed precision for steganography
//! - **Four steganography methods**: Arithmetic, Block, Huffman, and Rejection (Cachin, 2004)
//! - **Reed-Solomon ECC layer** (`ecc`) for tolerating logprob drift
//! - **Language model trait** for pluggable backends
//! - **llama.cpp integration** over HTTP (optional, feature `llamacpp`)
//!
//! ## Quick Start
//!
//! ```no_run
//! use steganeur::{
//!     bitstream::Message,
//!     steganography::{ArithmeticStega, StegaConfig},
//!     lm::DummyLM,
//! };
//!
//! // Create a language model (use DummyLM for testing)
//! let lm = DummyLM::new(100);
//!
//! // Configure steganography
//! let config = StegaConfig {
//!     temperature: 1.0,
//!     top_k: 50,
//!     max_tokens: 200,
//!     seed: None,
//! };
//!
//! let stega = ArithmeticStega::new(&lm, config, 16);
//!
//! // Generate a random message
//! let mut rng = rand::thread_rng();
//! let msg = Message::random(&mut rng, 64);
//!
//! // Encode message into tokens
//! let ctx = vec![0u32, 1u32];
//! let (tokens, bits_consumed, _cover_text) = stega.encode(&ctx, msg.data(), msg.num_bits()).unwrap();
//!
//! // Decode tokens back into message
//! let (decoded, decoded_bits) = stega.decode(&ctx, &tokens, msg.num_bits()).unwrap();
//! ```

pub mod arithmetic;
pub mod bitstream;
pub mod ecc;
pub mod error;
pub mod framing;
pub mod lm;
pub mod rejection;
pub mod steganography;

pub use error::{Error, Result};