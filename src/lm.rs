//! Language model trait and implementations.
//!
//! Defines the `LanguageModel` trait that all steganography algorithms depend on.
//! A language model provides probability distributions over tokens given a context.

use crate::error::Result;

/// A token ID in the language model's vocabulary.
pub type TokenId = u32;

/// A single (token, log-probability) pair.
#[derive(Debug, Clone)]
pub struct TokenProb {
    pub token: TokenId,
    pub log_prob: f64,
}

/// The conditional distribution produced by a language model at a single step.
#[derive(Debug, Clone)]
pub struct LmDistribution {
    /// Token IDs and their log-probabilities (or unnormalized scores).
    pub tokens: Vec<TokenProb>,
    /// Token strings (optional, for text-based decoding).
    pub token_strings: Vec<String>,
}

impl LmDistribution {
    /// Number of tokens in the distribution.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the distribution is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Get token IDs in sorted order (matching get_probs output).
    pub fn get_token_ids(&self, temperature: f64, top_k: usize) -> Vec<TokenId> {
        self.get_sorted(temperature, top_k).into_iter().map(|(id, _, _)| id).collect()
    }

    /// Get token strings in sorted order (matching get_probs output).
    pub fn get_token_strings(&self, temperature: f64, top_k: usize) -> Vec<&str> {
        self.get_sorted(temperature, top_k).into_iter().map(|(_, s, _)| s).collect()
    }

    /// Sort tokens by score, returning (id, string, score) triples.
    fn get_sorted(&self, temperature: f64, top_k: usize) -> Vec<(TokenId, &str, f64)> {
        if self.tokens.is_empty() {
            return vec![];
        }

        let mut sorted: Vec<(TokenId, &str, f64)> = self
            .tokens
            .iter()
            .enumerate()
            .map(|(i, tp)| (tp.token, self.token_strings.get(i).map_or("", |s| s.as_str()), tp.log_prob / temperature.max(1e-8)))
            .collect();

        sorted.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        let effective_k = if top_k > 0 && top_k < sorted.len() {
            top_k
        } else {
            sorted.len()
        };
        sorted.truncate(effective_k);
        sorted
    }

    /// Extract probabilities, optionally applying temperature and top-k filtering.
    ///
    /// Returns a vector of probabilities (softmax-normalized, summing to 1).
    /// `temperature` -- softmax temperature (higher = more uniform, lower = more peaked).
    /// `top_k` -- if > 0, keep only the top-k tokens by probability.
    pub fn get_probs(&self, temperature: f64, top_k: usize) -> Vec<f64> {
        let sorted = self.get_sorted(temperature, top_k);
        if sorted.is_empty() {
            return vec![];
        }

        // Softmax
        let max_log = sorted.iter().map(|(_, _, lp)| *lp).fold(f64::NEG_INFINITY, f64::max);

        let mut probs = Vec::with_capacity(sorted.len());
        let mut sum = 0.0;
        for &(_, _, lp) in &sorted {
            let p = (lp - max_log).exp();
            probs.push(p);
            sum += p;
        }

        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for p in probs.iter_mut() {
                *p *= inv_sum;
            }
        } else {
            let uniform = 1.0 / sorted.len() as f64;
            for p in probs.iter_mut() {
                *p = uniform;
            }
        }

        probs
    }
}

pub trait LanguageModel {
    /// Get the distribution over the next token given a context of token IDs.
    fn predict(&self, context: &[TokenId]) -> Result<LmDistribution>;

    /// Get the vocabulary size.
    fn vocab_size(&self) -> usize;

    /// Get the end-of-sequence token ID, if any.
    fn eos_token(&self) -> Option<TokenId>;

    /// Tokenize a string into token IDs.
    fn tokenize(&self, text: &str) -> Result<Vec<TokenId>>;

    /// Tokenize without modifying LM state (for decode).
    fn tokenize_static(&self, text: &str) -> Result<Vec<TokenId>> {
        self.tokenize(text)
    }

    /// Detokenize token IDs back into a string.
    fn detokenize(&self, tokens: &[TokenId]) -> Result<String>;
}

/// A simple dummy language model for testing.
pub struct DummyLM {
    vocab_size: usize,
    fixed_probs: Option<Vec<f64>>,
    eos: Option<TokenId>,
    /// Optional token strings. If present, `token i` renders as `strings[i]`.
    /// Used to exercise the text-based decode path without a real server.
    strings: Option<Vec<String>>,
}

impl DummyLM {
    pub fn new(vocab_size: usize) -> Self {
        DummyLM {
            vocab_size,
            fixed_probs: None,
            eos: None,
            strings: None,
        }
    }

    pub fn with_probs(mut self, probs: Vec<f64>) -> Self {
        assert_eq!(probs.len(), self.vocab_size);
        self.fixed_probs = Some(probs);
        self
    }

    pub fn with_eos(mut self, eos: TokenId) -> Self {
        self.eos = Some(eos);
        self
    }

    /// Provide renderable strings for each token id (enables text-based decode).
    pub fn with_token_strings(mut self, strings: Vec<String>) -> Self {
        assert_eq!(strings.len(), self.vocab_size);
        self.strings = Some(strings);
        self
    }
}

impl LanguageModel for DummyLM {
    fn predict(&self, _context: &[TokenId]) -> Result<LmDistribution> {
        let (tokens, token_strings) = match &self.fixed_probs {
            Some(probs) => {
                let ts: Vec<String> = match &self.strings {
                    Some(s) => s.clone(),
                    None => (0..self.vocab_size).map(|_| String::new()).collect(),
                };
                let tps: Vec<TokenProb> = probs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| TokenProb {
                        token: i as TokenId,
                        log_prob: p.ln(),
                    })
                    .collect();
                (tps, ts)
            }
            None => {
                let ts: Vec<String> = match &self.strings {
                    Some(s) => s.clone(),
                    None => (0..self.vocab_size).map(|_| String::new()).collect(),
                };
                let tps: Vec<TokenProb> = (0..self.vocab_size)
                    .map(|i| TokenProb {
                        token: i as TokenId,
                        log_prob: -(self.vocab_size as f64).ln(),
                    })
                    .collect();
                (tps, ts)
            }
        };
        Ok(LmDistribution { tokens, token_strings })
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_token(&self) -> Option<TokenId> {
        self.eos
    }

    fn tokenize(&self, text: &str) -> Result<Vec<TokenId>> {
        Ok(text
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| (i % self.vocab_size) as TokenId)
            .collect())
    }

    fn detokenize(&self, tokens: &[TokenId]) -> Result<String> {
        match &self.strings {
            Some(strings) => {
                let mut text = String::new();
                for &id in tokens {
                    if (id as usize) < strings.len() {
                        text.push_str(&strings[id as usize]);
                    }
                }
                Ok(text)
            }
            None => Ok(tokens
                .iter()
                .map(|t| format!("t{}", t))
                .collect::<Vec<_>>()
                .join(" ")),
        }
    }
}

// ============================================================================
// llama.cpp API client
// ============================================================================

/// A language model that connects to a running llama.cpp server via HTTP.
///
/// ## Usage
///
/// 1. Start llama.cpp server:
///    ```bash
///    llama-server -m path/to/model.gguf --host 127.0.0.1 --port 8080
///    ```
///
/// 2. Use this model in steganography:
///    ```no_run,ignore
///    // Requires feature "llamacpp"
///    let lm = steganeur::lm::LlamaCppLM::new("http://127.0.0.1:8080", 32000, 2, 128_000)?;
///    let dist = lm.predict(&[1, 2, 3])?;
///    ```
#[cfg(feature = "llamacpp")]
pub mod llamacpp;

#[cfg(feature = "llamacpp")]
pub use llamacpp::LlamaCppLM;