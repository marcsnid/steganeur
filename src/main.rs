//! Stagano -- Neural Linguistic Steganography CLI
//!
//! Command-line interface for encoding and decoding secret messages
//! in natural language cover text using arithmetic coding with language models.
//!
//! ## Usage
//!
//! Encode a message into cover text:
//! ```bash
//! echo "Secret message" | steganeur encode \
//!     --context "News context..." \
//!     --method arithmetic \
//!     --llama-url http://127.0.0.1:8080
//! ```
//!
//! Decode cover text back into the message:
//! ```bash
//! echo "Generated cover text" | steganeur decode \
//!     --context "News context..." \
//!     --method arithmetic \
//!     --llama-url http://127.0.0.1:8080
//! ```

use clap::{Parser, Subcommand};
use steganeur::bitstream::Message;
use steganeur::ecc::{rs_decode, rs_encode};
use steganeur::framing::{frame_message, unframe_payload};
use steganeur::lm::{DummyLM, LanguageModel};
use steganeur::rejection::RejectionStega;
use steganeur::steganography::{
    ArithmeticStega, BlockStega, HuffmanStega, StegaConfig, StegaMethod,
};
use std::io::{Read, Write};

#[derive(Parser)]
#[command(name = "steganeur")]
#[command(about = "Neural linguistic steganography", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Encode a secret message into natural language cover text
    Encode {
        /// Context text to condition the language model on
        #[arg(short, long)]
        context: String,

        /// Secret message to encode (read from stdin if not provided)
        #[arg(short = 'i', long)]
        message: Option<String>,

        /// Message file path (alternative to --message)
        #[arg(short = 'f', long)]
        message_file: Option<String>,

        /// Steganography method: arithmetic, block, huffman, rejection
        #[arg(short, long, default_value = "arithmetic")]
        method: String,

        /// Temperature for softmax (lower = more deterministic)
        #[arg(short, long, default_value_t = 1.0)]
        temperature: f64,

        /// Top-k truncation (0 = no truncation)
        #[arg(long, default_value_t = 300)]
        top_k: usize,

        /// Block bits (for block method)
        #[arg(long, default_value_t = 2)]
        block_bits: usize,

        /// Max code length (for huffman method)
        #[arg(long, default_value_t = 16)]
        max_code_len: usize,

        /// Rejection sampling bits per token (for rejection method)
        #[arg(long, default_value_t = 2)]
        rejection_bits: usize,

        /// Arithmetic interval reset period -- every N tokens the interval
        /// resets, so a drift-induced bin flip cascades only within that
        /// block. 0 = no reset. Default 16.
        #[arg(long, default_value_t = 16)]
        arith_block_size: usize,

        /// Seed for deterministic RNG (rejection method, encoder only)
        #[arg(long)]
        seed: Option<u64>,

        /// Enable Reed-Solomon ECC
        #[arg(long, default_value_t = false)]
        ecc: bool,

        /// Reed-Solomon parity bytes (ECC robustness)
        #[arg(long, default_value_t = 10)]
        ecc_parity: usize,

        /// Max tokens to generate (safety cap, auto-sized by default).
        /// The encoder stops automatically once the message is consumed, so
        /// this is rarely needed. Use only to override the auto-sizing.
        #[arg(long)]
        max_tokens: Option<usize>,

        /// llama.cpp server URL (e.g., http://127.0.0.1:8080)
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        llama_url: String,

        /// Model name on the server (auto-detected if not set)
        #[arg(long)]
        model: Option<String>,

        /// Vocab size for llama.cpp model
        #[arg(long, default_value_t = 32000)]
        vocab_size: usize,

        /// EOS token ID for llama.cpp model
        #[arg(long, default_value_t = 2)]
        eos_token: u32,

        /// Context size for llama.cpp model
        #[arg(long, default_value_t = 2048)]
        n_ctx: usize,

        /// Use dummy LM instead of real model (for testing)
        #[arg(long, default_value_t = false)]
        dummy: bool,

        /// Dummy LM vocabulary size
        #[arg(long, default_value_t = 100)]
        dummy_vocab: usize,

        /// Print encoding statistics to stderr
        #[arg(long, default_value_t = false)]
        stats: bool,

        /// Read stdin as raw bytes without stripping a trailing newline.
        /// Use this when piping binary data (e.g. ciphertext) via stdin.
        /// --message-file is always raw.
        #[arg(long, default_value_t = false)]
        raw: bool,

    },
    Decode {
        /// Context text used during encoding
        #[arg(short, long)]
        context: String,

        /// Cover text to decode (read from stdin if not provided)
        #[arg(short = 'i', long)]
        cover: Option<String>,

        /// Cover text file path (alternative to --cover)
        #[arg(short = 'f', long)]
        cover_file: Option<String>,

        /// Steganography method: arithmetic, block, huffman, rejection
        #[arg(short, long, default_value = "arithmetic")]
        method: String,

        /// Temperature used during encoding
        #[arg(short, long, default_value_t = 1.0)]
        temperature: f64,

        /// Top-k truncation used during encoding
        #[arg(long, default_value_t = 300)]
        top_k: usize,

        /// Block bits (for block method)
        #[arg(long, default_value_t = 2)]
        block_bits: usize,

        /// Max code length (for huffman method)
        #[arg(long, default_value_t = 16)]
        max_code_len: usize,

        /// Rejection sampling bits per token (for rejection method)
        #[arg(long, default_value_t = 2)]
        rejection_bits: usize,

        /// Arithmetic interval reset period (must match encode)
        #[arg(long, default_value_t = 16)]
        arith_block_size: usize,

        /// Seed for deterministic RNG (rejection method, encoder only)
        #[arg(long)]
        seed: Option<u64>,

        /// Enable Reed-Solomon ECC (must match encode)
        #[arg(long, default_value_t = false)]
        ecc: bool,

        /// Reed-Solomon parity bytes (must match encode)
        #[arg(long, default_value_t = 10)]
        ecc_parity: usize,

        /// Max message bits to recover (auto-sized by default).
        /// The decoder stops at the end-of-stream marker, so this is rarely needed.
        #[arg(long)]
        max_bits: Option<usize>,

        /// llama.cpp server URL
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        llama_url: String,

        /// Model name on the server (auto-detected if not set)
        #[arg(long)]
        model: Option<String>,

        /// Vocab size for llama.cpp model
        #[arg(long, default_value_t = 32000)]
        vocab_size: usize,

        /// EOS token ID for llama.cpp model
        #[arg(long, default_value_t = 2)]
        eos_token: u32,

        /// Context size for llama.cpp model
        #[arg(long, default_value_t = 2048)]
        n_ctx: usize,

        /// Use dummy LM for testing
        #[arg(long, default_value_t = false)]
        dummy: bool,

        /// Dummy LM vocabulary size
        #[arg(long, default_value_t = 100)]
        dummy_vocab: usize,

        /// Print decoding statistics to stderr
        #[arg(long, default_value_t = false)]
        stats: bool,

        /// Treat the decoded message as text: validate UTF-8 and append a
        /// trailing newline. Without this flag, raw message bytes are written
        /// to stdout as-is (binary-safe).
        #[arg(long, default_value_t = false)]
        text: bool,

    },
    Demo {
        /// Steganography method
        #[arg(long, default_value = "arithmetic")]
        method: String,

        /// Temperature
        #[arg(long, default_value_t = 1.0)]
        temperature: f64,

        /// llama.cpp server URL
        #[arg(long, default_value = "http://127.0.0.1:8080")]
        llama_url: String,

        /// Model name on the server (auto-detected if not set)
        #[arg(long)]
        model: Option<String>,

        /// Vocab size
        #[arg(long, default_value_t = 32000)]
        vocab_size: usize,

        /// EOS token ID
        #[arg(long, default_value_t = 2)]
        eos_token: u32,

        /// Context size
        #[arg(long, default_value_t = 2048)]
        n_ctx: usize,
    },
}

#[allow(unused_variables)]
#[allow(clippy::too_many_arguments)]
fn create_lm(
    dummy: bool,
    dummy_vocab: usize,
    llama_url: &str,
    model: Option<&str>,
    vocab_size: usize,
    eos_token: u32,
    n_ctx: usize,
) -> Box<dyn LanguageModel> {
    if dummy {
        return Box::new(DummyLM::new(dummy_vocab));
    }

    #[cfg(feature = "llamacpp")]
    {
        let result = if let Some(m) = model {
            steganeur::lm::LlamaCppLM::with_model(llama_url, Some(m), vocab_size, Some(eos_token), n_ctx)
        } else {
            steganeur::lm::LlamaCppLM::new(llama_url, vocab_size, Some(eos_token), n_ctx)
        };
        match result {
            Ok(lm) => {
                return Box::new(lm);
            }
            Err(e) => {
                eprintln!("Warning: Failed to connect to llama.cpp at {}: {}", llama_url, e);
                eprintln!("Falling back to dummy LM. Use --dummy to suppress this warning.");
            }
        }
    }

    #[cfg(not(feature = "llamacpp"))]
    {
        eprintln!("Note: llamacpp feature not enabled. Rebuild with default features.");
    }

    Box::new(DummyLM::new(dummy_vocab))
}

fn create_stega_method<'a>(
    lm: &'a dyn LanguageModel,
    config: StegaConfig,
    method: &str,
    block_bits: usize,
    max_code_len: usize,
    rejection_bits: usize,
    arith_block_size: usize,
) -> Result<StegaMethod<'a>, Box<dyn std::error::Error>> {
    match method {
        "arithmetic" => Ok(StegaMethod::Arithmetic(ArithmeticStega::new(lm, config, arith_block_size))),
        "block" => {
            let block = BlockStega::new(lm, config, block_bits)?;
            Ok(StegaMethod::Block(block))
        }
        "huffman" => Ok(StegaMethod::Huffman(HuffmanStega::new(lm, config, max_code_len))),
        "rejection" => Ok(StegaMethod::Rejection(RejectionStega::new(lm, config, rejection_bits))),
        _ => Err(format!(
            "Unknown method: {}. Use arithmetic, block, huffman, or rejection.",
            method
        )
        .into()),
    }
}

fn encode_command(cmd: &CommandEncodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let lm = create_lm(cmd.dummy, cmd.dummy_vocab, &cmd.llama_url, cmd.model.as_deref(), cmd.vocab_size, cmd.eos_token, cmd.n_ctx);

    let config = StegaConfig {
        temperature: cmd.temperature,
        top_k: cmd.top_k,
        max_tokens: cmd.max_tokens.unwrap_or(steganeur::arithmetic::DEFAULT_MAX_TOKENS),
        seed: cmd.seed,
    };

    let stega = create_stega_method(
        lm.as_ref(),
        config,
        &cmd.method,
        cmd.block_bits,
        cmd.max_code_len,
        cmd.rejection_bits,
        cmd.arith_block_size,
    )?;

    // Get the secret message as raw bytes.
    // --message: CLI arg (text, .as_bytes()).
    // --message-file: file read as raw bytes (binary-safe).
    // stdin: raw bytes via read_to_end. By default a single trailing newline
    //        is stripped (so `echo` works). --raw skips the strip for binary.
    let message_bytes = if let Some(msg) = &cmd.message {
        msg.as_bytes().to_vec()
    } else if let Some(path) = &cmd.message_file {
        std::fs::read(path)?
    } else {
        let mut buf = Vec::new();
        std::io::stdin().read_to_end(&mut buf)?;
        if !cmd.raw {
            // Strip a single trailing \n or \r\n so `echo` and `printf` work.
            if buf.last() == Some(&b'\n') {
                buf.pop();
                if buf.last() == Some(&b'\r') {
                    buf.pop();
                }
            }
        }
        buf
    };

    if message_bytes.is_empty() {
        return Err("Cannot encode an empty message.".into());
    }

    // Build the payload using chunked varint framing.
    //
    // Format: [varint(msg.len())][msg bytes][0x00]
    //
    // The varint length prefix tells the decoder exactly how many bytes to
    // read, and the trailing 0x00 (varint 0) marks end of stream. This is
    // binary-safe (0x00 in the message body is just data, not a sentinel),
    // has no fixed int size or cap, and the overhead is 2 bytes flat for any
    // message under 128 bytes.
    let payload = frame_message(&message_bytes);

    // Optional ECC: encode payload with Reed-Solomon
    let payload = if cmd.ecc && cmd.ecc_parity > 0 {
        rs_encode(&payload, cmd.ecc_parity)
    } else {
        payload
    };

    let msg_message = Message::from_bytes(payload.clone());
    let num_bits = msg_message.num_bits();

    // Tokenize context
    let context = lm.tokenize(&cmd.context)?;

    // Encode
    let (tokens, bits_consumed, generated_text) = stega.encode(&context, msg_message.data(), num_bits)?;

    // Warn if the message could not be fully encoded.
    if bits_consumed < num_bits {
        eprintln!(
            "Warning: only {} of {} message bits were encoded ({} tokens).",
            bits_consumed, num_bits, tokens.len()
        );
    }

    // Build the cover text from the token strings the encoder actually selected,
    // not from server-side detokenization. The decoder matches token strings
    // against the cover text, so the cover text must be the exact concatenation
    // of those strings (appended to the context). Server detokenization
    // normalizes spacing differently from the raw token strings, which causes
    // decode drift on longer messages.
    //
    // Fall back to server detokenization only when token strings are unavailable
    // (e.g. the DummyLM, which has no strings).
    let cover_text = if !generated_text.is_empty() {
        format!("{}{}", cmd.context, generated_text)
    } else {
        let full_tokens: Vec<u32> = [context.as_slice(), tokens.as_slice()].concat();
        lm.detokenize(&full_tokens)?
    };

    let num_tokens = tokens.len();
    let bits_per_word = if num_tokens > 0 {
        bits_consumed as f64 / num_tokens as f64
    } else {
        0.0
    };

    // Print cover text (stdout, for human reading)
    // Output ONLY the cover text to stdout
    println!("{}", cover_text);
    // Stats go to stderr, only when requested via --stats
    if cmd.stats {
        eprintln!();
        eprintln!("--- Stats ---");
        eprintln!("Message: {} bytes", message_bytes.len());
        if cmd.ecc {
            eprintln!("ECC: RS(parity={}) on payload", cmd.ecc_parity);
        }
        eprintln!("Payload: {} bits", num_bits);
        eprintln!("Bits consumed: {}", bits_consumed);
        eprintln!("Tokens generated: {}", num_tokens);
        eprintln!("Bits/word: {:.2}", bits_per_word);
    }

    Ok(())
}

struct CommandEncodeArgs {
    context: String,
    message: Option<String>,
    message_file: Option<String>,
    method: String,
    temperature: f64,
    top_k: usize,
    block_bits: usize,
    max_code_len: usize,
    rejection_bits: usize,
    seed: Option<u64>,
    ecc: bool,
    ecc_parity: usize,
    max_tokens: Option<usize>,
    arith_block_size: usize,
    llama_url: String,
    model: Option<String>,
    vocab_size: usize,
    eos_token: u32,
    n_ctx: usize,
    dummy: bool,
    dummy_vocab: usize,
    stats: bool,
    raw: bool,
}

fn decode_command(cmd: &CommandDecodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let lm = create_lm(cmd.dummy, cmd.dummy_vocab, &cmd.llama_url, cmd.model.as_deref(), cmd.vocab_size, cmd.eos_token, cmd.n_ctx);

    let config = StegaConfig {
        temperature: cmd.temperature,
        top_k: cmd.top_k,
        max_tokens: cmd.max_bits.unwrap_or(steganeur::arithmetic::DEFAULT_MAX_TOKENS),
        seed: cmd.seed,
    };

    let stega = create_stega_method(
        lm.as_ref(),
        config,
        &cmd.method,
        cmd.block_bits,
        cmd.max_code_len,
        cmd.rejection_bits,
        cmd.arith_block_size,
    )?;

    let cover_text = if let Some(cover) = &cmd.cover {
        cover.clone()
    } else if let Some(path) = &cmd.cover_file {
        std::fs::read_to_string(path)?
    } else {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        // Remove trailing newline(s) but preserve trailing spaces
        buf.trim_end_matches(&['\n', '\r'][..]).to_string()
    };

    // Decode using string matching against the cover text.
    // The max_bits default is derived from the input size so long or ECC'd
    // messages are never silently truncated (the end-of-stream marker gates
    // extraction anyway).
    //
    // Text-only decode: match token strings against the cover text.

    let default_bits = cover_text.len() * 8;
    let (raw_bytes, decoded_bits) =
        stega.decode_text(&cmd.context, &cover_text, cmd.max_bits.unwrap_or(default_bits))?;

    // The payload format is: [message bytes] + [0x00 null terminator] (+ parity if ECC).
    // If --ecc, first run Reed-Solomon decode to correct any logprob-drift bit
    // errors, then strip the parity bytes.
    let raw_decoded: Vec<u8> = if cmd.ecc && cmd.ecc_parity > 0 {
        match rs_decode(&raw_bytes, cmd.ecc_parity) {
            Ok(corrected) => corrected,
            Err(e) => {
                eprintln!("Warning: RS decode failed ({}). Falling back to raw bytes.", e);
                raw_bytes.clone()
            }
        }
    } else {
        raw_bytes.clone()
    };

    // Extract the message from the framed payload.
    //
    // The payload uses chunked varint framing:
    //   [varint N][N bytes][0x00]
    // The decoder reads N bytes, sees the 0x00 end-of-stream marker, and
    // stops. Trailing bytes (read-ahead zero padding from the steganographic
    // decoder) are ignored. This is binary-safe: 0x00 bytes inside the
    // message are read by count, not by sentinel scanning.
    let message_bytes = unframe_payload(&raw_decoded)?;

    // Write the decoded message to stdout.
    //
    // Default: raw bytes via write_all (binary-safe, no trailing newline).
    // --text: validate UTF-8 and append a trailing newline (for shell
    //         pipelines that expect text output).
    if cmd.text {
        let decoded_text = String::from_utf8(message_bytes.clone())
            .map_err(|e| format!("Decoded message is not valid UTF-8: {}. \
                                   Drop --text for raw byte output.", e))?;
        println!("{}", decoded_text);
    } else {
        std::io::stdout().write_all(&message_bytes)?;
    }
    if cmd.stats {
        eprintln!();
        eprintln!("--- Stats ---");
        eprintln!("Decoded bits: {}", decoded_bits);
        eprintln!("Raw bytes: {}", raw_bytes.len());
        if cmd.ecc {
            eprintln!("ECC: RS(parity={}) applied", cmd.ecc_parity);
        }
        eprintln!("Raw hex: {}", raw_decoded.iter().map(|b| format!("{:02x}", b)).collect::<String>());
        eprintln!("Message bytes: {}", message_bytes.len());
        eprintln!("Hex: {}", message_bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>());
    }

    Ok(())
}

struct CommandDecodeArgs {
    context: String,
    cover: Option<String>,
    cover_file: Option<String>,
    method: String,
    temperature: f64,
    top_k: usize,
    block_bits: usize,
    max_code_len: usize,
    rejection_bits: usize,
    seed: Option<u64>,
    ecc: bool,
    ecc_parity: usize,
    max_bits: Option<usize>,
    arith_block_size: usize,
    llama_url: String,
    model: Option<String>,
    vocab_size: usize,
    eos_token: u32,
    n_ctx: usize,
    dummy: bool,
    dummy_vocab: usize,
    stats: bool,
    text: bool,
}

fn demo_command(method_name: &str, temperature: f64, llama_url: &str, model: Option<&str>, vocab_size: usize, eos_token: u32, n_ctx: usize) -> Result<(), Box<dyn std::error::Error>> {
    let (_llama_url, _vocab_size, _eos_token, _n_ctx) = (llama_url, vocab_size, eos_token, n_ctx);
    println!("Method: {}", method_name);
    println!("Temperature: {}", temperature);
    println!("LLM server: {}", _llama_url);
    println!();

    let lm: Box<dyn LanguageModel> = {
        #[cfg(feature = "llamacpp")]
        {
            let result = if let Some(m) = model {
                steganeur::lm::LlamaCppLM::with_model(_llama_url, Some(m), _vocab_size, Some(_eos_token), _n_ctx)
            } else {
                steganeur::lm::LlamaCppLM::new(_llama_url, _vocab_size, Some(_eos_token), _n_ctx)
            };
            match result {
                Ok(lm) => {
                    println!("Connected to llama.cpp server.");
                    Box::new(lm)
                }
                Err(e) => {
                    eprintln!("Warning: Could not connect to llama.cpp ({}). Using dummy LM.", e);
                    Box::new(DummyLM::new(100))
                }
            }
        }
        #[cfg(not(feature = "llamacpp"))]
        {
            eprintln!("llamacpp feature not enabled. Using dummy LM.");
            Box::new(DummyLM::new(100))
        }
    };

    let config = StegaConfig {
        temperature,
        top_k: 300,
        max_tokens: 128,
        seed: None,
    };

    let stega: StegaMethod = match method_name {
        "block" => {
            let block = BlockStega::new(lm.as_ref(), config.clone(), 2);
            match block {
                Ok(b) => StegaMethod::Block(b),
                Err(e) => {
                    eprintln!("Block init failed: {}. Falling back to arithmetic.", e);
                    StegaMethod::Arithmetic(ArithmeticStega::new(lm.as_ref(), config, 16))
                }
            }
        }
        "huffman" => StegaMethod::Huffman(HuffmanStega::new(lm.as_ref(), config, 8)),
        "rejection" => StegaMethod::Rejection(RejectionStega::new(lm.as_ref(), config, 2)),
        _ => StegaMethod::Arithmetic(ArithmeticStega::new(lm.as_ref(), config, 16)),
    };

    println!("Enter a context prompt (or press Enter for default):");
    let mut context_text = String::new();
    std::io::stdin().read_line(&mut context_text)?;
    let context_text = context_text.trim();
    let context_text = if context_text.is_empty() {
        "Kim Jong Il was the enigmatic leader of the most enigmatic country on earth. Much about Kim's life was, and is, shrouded in mystery."
    } else {
        context_text
    };
    println!("Using context: {}", context_text);
    println!();

    let context_tokens = lm.tokenize(context_text)?;

    loop {
        println!("Enter a secret message (or 'quit' to exit):");
        let mut message = String::new();
        std::io::stdin().read_line(&mut message)?;
        let message = message.trim();
        if message == "quit" || message == "q" {
            break;
        }

        if message.is_empty() {
            continue;
        }

        let msg_bytes = message.as_bytes().to_vec();
        let msg = Message::from_bytes(msg_bytes.clone());
        let (tokens, consumed, generated_text) = stega.encode(&context_tokens, msg.data(), msg.num_bits())?;
        let cover_text = if !generated_text.is_empty() {
            format!("{}{}", context_text, generated_text)
        } else {
            let full_tokens: Vec<u32> = [context_tokens.as_slice(), tokens.as_slice()].concat();
            lm.detokenize(&full_tokens)?
        };

        println!();
        println!("=== Cover Text ===");
        println!("{}", cover_text);
        println!();

        let (decoded_bytes, _decoded_bits) = stega.decode(&context_tokens, &tokens, msg.num_bits() + 8)?;
        let decoded_text = String::from_utf8_lossy(&decoded_bytes);
        println!("=== Decoded Message ===");
        println!("{}", decoded_text);
        println!();

        let bits_per_word = if !tokens.is_empty() {
            consumed as f64 / tokens.len() as f64
        } else {
            0.0
        };
        println!("Bits/word: {:.2}, Tokens: {}, Bits: {}", bits_per_word, tokens.len(), consumed);
        println!();
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Command::Encode {
            context,
            message,
            message_file,
            method,
            temperature,
            top_k,
            block_bits,
            max_code_len,
            rejection_bits,
            arith_block_size,
            seed,
            ecc,
            ecc_parity,
            max_tokens,
            llama_url,
            model,
            vocab_size,
            eos_token,
            n_ctx,
            dummy,
            dummy_vocab,
            stats,
            raw,
        } => encode_command(&CommandEncodeArgs {
            context,
            message,
            message_file,
            method,
            temperature,
            top_k,
            block_bits,
            max_code_len,
            rejection_bits,
            arith_block_size,
            seed,
            ecc,
            ecc_parity,
            max_tokens,
            llama_url,
            model,
            vocab_size,
            eos_token,
            n_ctx,
            dummy,
            dummy_vocab,
            stats,
            raw,
        }),
        Command::Decode {
            context,
            cover,
            cover_file,
            method,
            temperature,
            top_k,
            block_bits,
            max_code_len,
            rejection_bits,
            arith_block_size,
            seed,
            ecc,
            ecc_parity,
            max_bits,
            llama_url,
            model,
            vocab_size,
            eos_token,
            n_ctx,
            dummy,
            dummy_vocab,
            stats,
            text,
        } => decode_command(&CommandDecodeArgs {
            context,
            cover,
            cover_file,
            method,
            temperature,
            top_k,
            block_bits,
            max_code_len,
            rejection_bits,
            arith_block_size,
            seed,
            ecc,
            ecc_parity,
            max_bits,
            llama_url,
            model,
            vocab_size,
            eos_token,
            n_ctx,
            dummy,
            dummy_vocab,
            stats,
            text,
        }),
        Command::Demo {
            method,
            temperature,
            llama_url,
            model,
            vocab_size,
            eos_token,
            n_ctx,
        } => demo_command(&method, temperature, &llama_url, model.as_deref(), vocab_size, eos_token, n_ctx),
    }
}