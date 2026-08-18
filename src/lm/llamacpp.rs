//! LLM backend client using the `/v1/completions` endpoint with raw token-ID prompts.
//!
//! Connects to any server providing the OpenAI `/v1/completions` endpoint,
//! such as [llama.cpp](https://github.com/ggerganov/llama.cpp).
//!
//! Sends raw token IDs as the prompt (a JSON array of integers) and reads
//! `top_logprobs` from the response. `cache_prompt` is always `false` so
//! each call recomputes from scratch -- a recipient on a different server
//! cannot share the sender's KV cache.
//!
//! ## Quick Start
//! ## Quick Start
//!
//! ```bash
//! llama-server -m model.gguf --host 0.0.0.0 --port 11434
//! curl http://127.0.0.1:11434/v1/models
//! ```

use crate::error::{Error, Result};
use crate::lm::{LanguageModel, LmDistribution, TokenProb, TokenId};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;

/// Language model client for OpenAI-compatible completions APIs.
///
/// Supports:
/// - `/v1/completions` for next-token probabilities (with token-ID prompt)
/// - `/v1/models` for model discovery
/// - `/v1/tokenize` and `/v1/detokenize` for tokenization (llama.cpp native)
/// - `/api/tokenize` and `/api/detokenize` (Ollama fallback)
pub struct LlamaCppLM {
    base_url: String,
    client: reqwest::blocking::Client,
    /// Model name used in API requests (e.g., "Qwen3.6-27B-Uncensored-4-GGUF").
    model: String,
    /// Discovered model name from server (if auto-detected).
    discovered_model: Option<String>,
    vocab_size: usize,
    eos_token_id: Option<TokenId>,
    _n_ctx: usize,
    /// Accumulated token ID→string mapping from previous predictions.
    /// Used by `detokenize()` for cover-text rendering.
    token_strings: RefCell<HashMap<TokenId, String>>,
}

impl LlamaCppLM {
    /// Create a new client.
    ///
    /// - `base_url`: server address, e.g. `"http://192.168.0.122:11434"`.
    /// - `vocab_size`: vocabulary size (e.g. 152064 for Qwen3, 256000 for Gemma-4).
    /// - `eos_token_id`: end-of-sequence token ID (e.g. `151643` for Qwen, `107` for Gemma).
    /// - `n_ctx`: maximum context length (e.g., 8192).
    pub fn new(
        base_url: &str,
        vocab_size: usize,
        eos_token_id: Option<TokenId>,
        n_ctx: usize,
    ) -> Result<Self> {
        Self::with_model(base_url, None, vocab_size, eos_token_id, n_ctx)
    }

    /// Create with an explicit model name (skips auto-discovery).
    pub fn with_model(
        base_url: &str,
        model_name: Option<&str>,
        vocab_size: usize,
        eos_token_id: Option<TokenId>,
        n_ctx: usize,
    ) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| Error::Lm(format!("Failed to create HTTP client: {}", e)))?;

        let base = base_url.trim_end_matches('/').to_string();

        // Try to discover a model from the server
        let discovered = Self::discover_model(&client, &base).ok();

        // Use the provided model name, or the URL-path model, or auto-discovered
        let model = model_name
            .map(|s| s.to_string())
            .or_else(|| {
                base.split('/')
                    .next_back()
                    .filter(|s| !s.is_empty() && !s.contains(':') && *s != "http:" && *s != "https:")
                    .map(|s| s.to_string())
            })
            .or_else(|| discovered.clone())
            .unwrap_or_else(|| "default".to_string());

        log::info!(
            "No server-side cache reuse (cache_prompt=false). Every call recomputes from scratch."
        );

        Ok(LlamaCppLM {
            base_url: base,
            client,
            model,
            discovered_model: discovered,
            vocab_size,
            eos_token_id,
            _n_ctx: n_ctx,
            token_strings: RefCell::new(HashMap::new()),
        })
    }

    /// Query `/v1/models` and return the first available model ID.
    fn discover_model(client: &reqwest::blocking::Client, base_url: &str) -> Result<String> {
        #[derive(Deserialize)]
        struct ModelsResponse {
            data: Vec<ModelEntry>,
        }
        #[derive(Deserialize)]
        struct ModelEntry {
            id: String,
        }

        let resp = client
            .get(format!("{}/v1/models", base_url))
            .send()
            .map_err(|e| Error::Lm(format!("Failed to query /v1/models: {}", e)))?;

        if !resp.status().is_success() {
            return Err(Error::Lm(format!("/v1/models returned {}", resp.status())));
        }

        let parsed: ModelsResponse = resp
            .json()
            .map_err(|e| Error::Lm(format!("Failed to parse /v1/models: {}", e)))?;

        parsed
            .data
            .into_iter()
            .next()
            .map(|m| m.id)
            .ok_or_else(|| Error::Lm("No models available on server".into()))
    }

    /// Get the effective model name: use the provided one, or fall back to discovered.
    fn effective_model(&self) -> &str {
        if self.model != "default" {
            &self.model
        } else if let Some(ref discovered) = self.discovered_model {
            discovered
        } else {
            &self.model
        }
    }

    // ------------------------------------------------------------------
    // Tokenization: try multiple backends
    // ------------------------------------------------------------------

    fn try_tokenize(&self, text: &str) -> Result<Vec<TokenId>> {
        if let Ok(ids) = self.tokenize_v1(text) {
            return Ok(ids);
        }
        if let Ok(ids) = self.tokenize_ollama(text) {
            return Ok(ids);
        }
        if let Ok(ids) = self.tokenize_echo(text) {
            return Ok(ids);
        }
        Err(Error::Lm("No tokenization backend available".into()))
    }

    fn tokenize_v1(&self, text: &str) -> Result<Vec<TokenId>> {
        let endpoints = [
            format!("{}/v1/tokenize", self.base_url),
            format!("{}/tokenize", self.base_url),
        ];
        let mut last_err = None;
        for url in &endpoints {
            match self.tokenize_at_url(url, text) {
                Ok(ids) => return Ok(ids),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| Error::Lm("No tokenize endpoint available".into())))
    }

    fn tokenize_at_url(&self, url: &str, text: &str) -> Result<Vec<TokenId>> {
        #[derive(Serialize)]
        struct Req {
            content: String,
            model: String,
        }
        #[derive(Deserialize)]
        struct Resp {
            tokens: Vec<TokenId>,
        }

        let resp = self
            .client
            .post(url)
            .json(&Req {
                content: text.to_string(),
                model: self.effective_model().to_string(),
            })
            .send()
            .map_err(|e| Error::Lm(format!("tokenize HTTP error: {}", e)))?;

        if !resp.status().is_success() {
            return Err(Error::Lm(format!("tokenize status {}", resp.status())));
        }

        let parsed: Resp = resp
            .json()
            .map_err(|e| Error::Lm(format!("tokenize parse error: {}", e)))?;
        Ok(parsed.tokens)
    }

    fn tokenize_ollama(&self, text: &str) -> Result<Vec<TokenId>> {
        #[derive(Serialize)]
        struct Req {
            model: String,
            prompt: String,
        }
        #[derive(Deserialize)]
        struct Resp {
            tokens: Vec<TokenId>,
        }

        let resp = self
            .client
            .post(format!("{}/api/tokenize", self.base_url))
            .json(&Req {
                model: self.effective_model().to_string(),
                prompt: text.to_string(),
            })
            .send()
            .map_err(|e| Error::Lm(format!("tokenize_ollama HTTP error: {}", e)))?;

        if !resp.status().is_success() {
            return Err(Error::Lm(format!("tokenize_ollama status {}", resp.status())));
        }

        let parsed: Resp = resp
            .json()
            .map_err(|e| Error::Lm(format!("tokenize_ollama parse error: {}", e)))?;
        Ok(parsed.tokens)
    }

    /// Tokenize via completions API with echo=true.
    fn tokenize_echo(&self, text: &str) -> Result<Vec<TokenId>> {
        #[derive(Serialize)]
        struct Req {
            model: String,
            prompt: String,
            max_tokens: u32,
            echo: bool,
            logprobs: usize,
        }
        #[derive(Deserialize)]
        struct Resp {
            choices: Vec<Choice>,
        }
        #[derive(Deserialize)]
        struct Choice {
            logprobs: Option<Logprobs>,
        }
        #[derive(Deserialize)]
        struct Logprobs {
            content: Vec<ContentEntry>,
        }
        #[derive(Deserialize)]
        struct ContentEntry {
            id: TokenId,
        }

        let req = Req {
            model: self.effective_model().to_string(),
            prompt: text.to_string(),
            max_tokens: 0,
            echo: true,
            logprobs: 1,
        };

        let resp = self
            .client
            .post(format!("{}/v1/completions", self.base_url))
            .json(&req)
            .send()
            .map_err(|e| Error::Lm(format!("tokenize_echo HTTP error: {}", e)))?;

        if !resp.status().is_success() {
            return Err(Error::Lm(format!("tokenize_echo status {}", resp.status())));
        }

        let parsed: Resp = resp
            .json()
            .map_err(|e| Error::Lm(format!("tokenize_echo parse error: {}", e)))?;

        let ids: Vec<TokenId> = parsed
            .choices
            .first()
            .and_then(|c| c.logprobs.as_ref())
            .map(|lp| lp.content.iter().map(|e| e.id).collect())
            .unwrap_or_default();

        if ids.is_empty() {
            return Err(Error::Lm("No token IDs in echo response".into()));
        }
        Ok(ids)
    }

    fn try_detokenize(&self, tokens: &[TokenId]) -> Result<String> {
        if let Ok(text) = self.detokenize_v1(tokens) {
            return Ok(text);
        }
        if let Ok(text) = self.detokenize_ollama(tokens) {
            return Ok(text);
        }
        Err(Error::Lm("No detokenization backend available".into()))
    }

    fn detokenize_v1(&self, tokens: &[TokenId]) -> Result<String> {
        let endpoints = [
            format!("{}/v1/detokenize", self.base_url),
            format!("{}/detokenize", self.base_url),
        ];
        let mut last_err = None;
        for url in &endpoints {
            match self.detokenize_at_url(url, tokens) {
                Ok(text) => return Ok(text),
                Err(e) => last_err = Some(e),
            }
        }
        Err(last_err.unwrap_or_else(|| Error::Lm("No detokenize endpoint available".into())))
    }

    fn detokenize_at_url(&self, url: &str, tokens: &[TokenId]) -> Result<String> {
        #[derive(Serialize)]
        struct Req {
            tokens: Vec<TokenId>,
            model: String,
        }
        #[derive(Deserialize)]
        struct Resp {
            content: String,
        }

        let resp = self
            .client
            .post(url)
            .json(&Req {
                tokens: tokens.to_vec(),
                model: self.effective_model().to_string(),
            })
            .send()
            .map_err(|e| Error::Lm(format!("detokenize HTTP error: {}", e)))?;

        if !resp.status().is_success() {
            return Err(Error::Lm(format!("detokenize status {}", resp.status())));
        }

        let parsed: Resp = resp
            .json()
            .map_err(|e| Error::Lm(format!("detokenize parse error: {}", e)))?;
        Ok(parsed.content)
    }

    fn detokenize_ollama(&self, tokens: &[TokenId]) -> Result<String> {
        #[derive(Serialize)]
        struct Req {
            model: String,
            tokens: Vec<TokenId>,
        }
        #[derive(Deserialize)]
        struct Resp {
            response: String,
        }

        let resp = self
            .client
            .post(format!("{}/api/detokenize", self.base_url))
            .json(&Req {
                model: self.effective_model().to_string(),
                tokens: tokens.to_vec(),
            })
            .send()
            .map_err(|e| Error::Lm(format!("detokenize_ollama HTTP error: {}", e)))?;

        if !resp.status().is_success() {
            return Err(Error::Lm(format!("detokenize_ollama status {}", resp.status())));
        }

        let parsed: Resp = resp
            .json()
            .map_err(|e| Error::Lm(format!("detokenize_ollama parse error: {}", e)))?;
        Ok(parsed.response)
    }

    // ------------------------------------------------------------------
    // Next-token probabilities via /v1/completions (token-ID prompt)
    //
    // Sends raw token IDs as the prompt, reads top_logprobs.

    /// Get next-token logprobs by calling `/v1/completions` with token IDs.
    ///
    /// The `prompt` field is a JSON array of integers (token IDs). The server
    /// avoids re-tokenization and processes the IDs directly.
    ///
    /// Returns triples `(token_id, token_string, logprob)` sorted by logprob
    /// descending (highest probability first).
    fn get_logprobs_completions(&self, prompt_tokens: &[TokenId]) -> Result<Vec<(TokenId, String, f64)>> {
        if prompt_tokens.is_empty() {
            return Err(Error::Lm("Cannot get logprobs for empty prompt".into()));
        }

        // The effective top_logprobs count: use the server's max or config's top_k.
        // llama.cpp's /v1/completions uses `logprobs` as integer = n_probs.
        // Cap at some reasonable maximum; 300 is the typical default.
        let n_probs: usize = 300.min(self.vocab_size);

        // cache_prompt is always false. A recipient decoding on a different
        // server cannot rely on the sender's warm KV cache, so every call
        // must recompute the forward pass from scratch.

        #[derive(Serialize)]
        struct Req<'a> {
            model: &'a str,
            prompt: &'a [TokenId],
            max_tokens: u32,
            temperature: f64,
            logprobs: usize,
            echo: bool,
            cache_prompt: bool,
        }

        #[derive(Deserialize)]
        struct Resp {
            choices: Vec<Choice>,
        }
        #[derive(Deserialize)]
        struct Choice {
            logprobs: Option<Logprobs>,
        }
        #[derive(Deserialize)]
        struct Logprobs {
            #[serde(default)]
            content: Vec<ContentEntry>,
        }
        #[derive(Deserialize)]
        struct ContentEntry {
            #[serde(default)]
            top_logprobs: Vec<TopLogprob>,
        }
        #[derive(Deserialize)]
        struct TopLogprob {
            #[serde(default)]
            id: TokenId,
            #[serde(default)]
            token: String,
            #[serde(default)]
            logprob: f64,
        }

        let req = Req {
            model: self.effective_model(),
            prompt: prompt_tokens,
            max_tokens: 1,
            temperature: 1.0,
            logprobs: n_probs,
            echo: false,
            cache_prompt: false,
        };

        let resp = self
            .client
            .post(format!("{}/v1/completions", self.base_url))
            .json(&req)
            .send()
            .map_err(|e| Error::Lm(format!("get_logprobs_completions HTTP error: {}", e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(Error::Lm(format!(
                "Server {} returned {}: {}",
                self.base_url, status, body
            )));
        }

        let parsed: Resp = resp
            .json()
            .map_err(|e| Error::Lm(format!("Parse error: {}", e)))?;

        let top_probs: &[TopLogprob] = parsed
            .choices
            .first()
            .and_then(|c| c.logprobs.as_ref())
            .and_then(|lp| lp.content.first())
            .map(|entry| &entry.top_logprobs[..])
            .unwrap_or(&[]);

        if top_probs.is_empty() {
            return Err(Error::Lm("Empty logprobs in completions response".into()));
        }

        let mut result: Vec<(TokenId, String, f64)> = top_probs
            .iter()
            .map(|alt| (alt.id, alt.token.clone(), alt.logprob))
            .collect();

        result.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        Ok(result)
    }
}

impl LanguageModel for LlamaCppLM {
    fn predict(&self, context: &[TokenId]) -> Result<LmDistribution> {
        if context.is_empty() {
            let uniform = -(self.vocab_size as f64).ln();
            let tokens: Vec<TokenProb> = (0..self.vocab_size.min(1000) as u32)
                .map(|t| TokenProb { token: t, log_prob: uniform })
                .collect();
            return Ok(LmDistribution { tokens, token_strings: vec![] });
        }

        // Call /v1/completions with the token IDs directly.
        // No text reconstruction, no base-splitting.
        let triples = self.get_logprobs_completions(context)?;

        let token_strings: Vec<String> = triples.iter().map(|(_, s, _)| s.clone()).collect();

        // Cache token strings for later detokenize calls
        {
            let mut token_map = self.token_strings.borrow_mut();
            for (id, s, _) in &triples {
                token_map.entry(*id).or_insert_with(|| s.clone());
            }
        }

        let token_probs: Vec<TokenProb> = triples
            .into_iter()
            .map(|(id, _token_str, log_prob)| TokenProb {
                token: id,
                log_prob,
            })
            .collect();

        Ok(LmDistribution { tokens: token_probs, token_strings })
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn eos_token(&self) -> Option<TokenId> {
        self.eos_token_id
    }

    fn tokenize(&self, text: &str) -> Result<Vec<TokenId>> {
        // The completions endpoint does not need base-context tracking.
        // We just tokenize and return.
        if let Ok(ids) = self.try_tokenize(text) {
            Ok(ids)
        } else {
            Ok(text.as_bytes().iter().map(|&b| b as TokenId).collect())
        }
    }

    fn tokenize_static(&self, text: &str) -> Result<Vec<TokenId>> {
        if let Ok(ids) = self.try_tokenize(text) {
            return Ok(ids);
        }
        Ok(text.as_bytes().iter().map(|&b| b as TokenId).collect())
    }

    fn detokenize(&self, tokens: &[TokenId]) -> Result<String> {
        if let Ok(text) = self.try_detokenize(tokens) {
            return Ok(text);
        }
        // Reconstruct from cached token strings.
        let token_map = self.token_strings.borrow();
        let mut text = String::new();
        for &id in tokens {
            if let Some(s) = token_map.get(&id) {
                text.push_str(s);
            }
        }
        if !text.is_empty() {
            return Ok(text);
        }
        Err(Error::Lm("Cannot detokenize: no backend available".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::steganography::{ArithmeticStega, StegaConfig};
    use crate::bitstream::Message;

    #[test]
    fn test_lm_connectivity() {
        let client = match reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
        {
            Ok(c) => c,
            Err(_) => return,
        };
        match LlamaCppLM::discover_model(&client, "http://192.168.0.122:11434") {
            Ok(model) => println!("Discovered model: {}", model),
            Err(e) => println!("Discovery failed (expected if server is down): {}", e),
        }
    }

    /// Load-test: encode a message, then decode from the cover text alone
    /// (no tokens file) 20 times. Each iteration uses a fresh LM (separate
    /// process simulation). The cover text is the only decode path.
    ///
    /// This test is #[ignore] by default -- run against a live server with:
    ///   cargo test --release -- --ignored test_decode_from_text_stability
    #[test]
    #[ignore]
    fn test_decode_from_text_stability() {
        let _ = env_logger::try_init();

        let base_url = "http://127.0.0.1:11434";
        let model = "Qwen3.6-27B-GGUF";
        let vocab_size = 152064;
        let eos_token = 151643;
        let context_text = "The neighborhood had changed over the years. Old Mr. Henderson still kept his garden immaculate, and every morning he could be seen tending to his roses before the sun was fully up.";
        let secret = "A short secret.";

        let lm = LlamaCppLM::with_model(base_url, Some(model), vocab_size, Some(eos_token), 8192)
            .expect("Failed to connect to server");

        let config = StegaConfig {
            temperature: 2.0,
            top_k: 300,
            max_tokens: 200,
            seed: None,
        };

        let stega = ArithmeticStega::new(&lm, config, 16);

        let context = lm.tokenize(context_text).expect("Tokenize failed");

        let mut payload = secret.as_bytes().to_vec();
        payload.push(0x00);
        let msg = Message::from_bytes(payload);
        let num_bits = msg.num_bits();

        let (generated_tokens, bits_consumed, generated_text) = stega
            .encode(&context, msg.data(), num_bits)
            .expect("Encode failed");

        let cover_text = format!("{}{}", context_text, generated_text);
        eprintln!(
            "Encoded {} bits into {} tokens, cover text length {}",
            bits_consumed,
            generated_tokens.len(),
            cover_text.len()
        );

        // Decode from text 20 times with fresh LM per iteration.
        let mut last_result: Option<Vec<u8>> = None;
        for iteration in 0..20 {
            let decode_lm = LlamaCppLM::with_model(
                base_url, Some(model), vocab_size, Some(eos_token), 8192,
            )
            .expect("Failed to connect to server for decode");

            let decode_config = StegaConfig {
                temperature: 2.0,
                top_k: 300,
                max_tokens: 200,
                seed: None,
            };
            let decode_stega = ArithmeticStega::new(&decode_lm, decode_config, 16);

            let (decoded_bytes, _decoded_bits) = decode_stega
                .decode_text(context_text, &cover_text, num_bits + 32)
                .expect("Text decode failed");

            let message: Vec<u8> = decoded_bytes
                .iter()
                .copied()
                .skip_while(|&b| b == 0)
                .take_while(|&b| b != 0)
                .collect();

            if iteration == 0 {
                last_result = Some(message.clone());
                eprintln!("Iteration 0: decoded {} bytes", message.len());
            } else {
                let prev = last_result.as_ref().unwrap();
                if &message != prev {
                    panic!(
                        "Drift at iteration {}: got {:?}, expected {:?}",
                        iteration,
                        &message[..message.len().min(20)],
                        &prev[..prev.len().min(20)]
                    );
                }
            }
        }

        let final_bytes = last_result.unwrap();
        let decoded_text = String::from_utf8_lossy(&final_bytes);
        assert_eq!(decoded_text.as_ref(), secret);
        eprintln!("All 20 text decode iterations passed! Decoded: {}", decoded_text);
    }
}