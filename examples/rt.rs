use steganeur::lm::{LanguageModel, LlamaCppLM};
use steganeur::arithmetic::{FreqTable, MAX_FREQ, PRECISION, MAX_VAL, subdivide, int_to_bits_msb, num_same_from_beg, bits_msb_to_int};

fn filter(lm: &dyn LanguageModel, ctx: &[u32], temp: f64, top_k: usize) -> (FreqTable, Vec<u32>) {
    let dist = lm.predict(ctx).unwrap();
    let probs = dist.get_probs(temp, top_k);
    let ids = dist.get_token_ids(temp, top_k);
    let strings: Vec<String> = dist.get_token_strings(temp, top_k).iter().map(|s| s.to_string()).collect();
    let mut f_ids = Vec::new();
    let mut f_probs = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        let s = strings.get(i).map(|s| s.as_str()).unwrap_or("");
        if !(id == 151643 || s.contains("<|") || s.is_empty() || s.contains('\u{fffd}')) {
            f_ids.push(id);
            f_probs.push(probs[i]);
        }
    }
    // Overlap filter
    let mut keep = vec![true; f_ids.len()];
    for i in 0..f_ids.len() {
        if !keep[i] { continue; }
        for j in (i+1)..f_ids.len() {
            if !keep[j] { continue; }
            let si = lm.detokenize(&[f_ids[i]]).unwrap();
            let sj = lm.detokenize(&[f_ids[j]]).unwrap();
            if sj.starts_with(&si) || si.starts_with(&sj) {
                keep[j] = false;
            }
        }
    }
    let mut ff_ids = Vec::new();
    let mut ff_probs = Vec::new();
    for (i, &id) in f_ids.iter().enumerate() {
        if keep[i] { ff_ids.push(id); ff_probs.push(f_probs[i]); }
    }
    (FreqTable::from_probs(&ff_probs, MAX_FREQ), ff_ids)
}

fn main() {
    let lm = LlamaCppLM::with_model(
        "http://127.0.0.1:11434",
        Some("Qwen3.6-27B-GGUF"),
        152064, Some(151643), 2048,
    ).unwrap();

    let ctx_text = "Once upon a time";
    let ctx = lm.tokenize(ctx_text).unwrap();

    let payload = b"Meet me at noon\x00".to_vec();
    let msg_bits: Vec<u8> = (0..payload.len()*8).map(|i| (payload[i/8] >> (7-(i%8))) & 1).collect();
    let mut cur_int: [u64; 2] = [0, MAX_VAL];
    let mut enc_ctx = ctx.clone();
    let mut tokens = Vec::new();
    let mut i = 0;

    // ENCODE with tracing
    for step in 0..5 {
        let (table, ids) = filter(&lm, &enc_ctx, 2.0, 300);
        let cum = subdivide(cur_int[0], cur_int[1], &table);
        let remaining = msg_bits.len() - i;
        let bits_to_use = remaining.min(PRECISION as usize);
        let mut padded = msg_bits.get(i..i+bits_to_use).unwrap_or(&[]).to_vec();
        padded.resize(PRECISION as usize, 0);
        let msg_idx = bits_msb_to_int(&padded);
        let selection = match cum.binary_search(&msg_idx) {
            Ok(idx) => idx + 1,
            Err(idx) => idx,
        };
        let tok = ids[selection];
        let new_low = if selection > 0 { cum[selection-1] } else { cur_int[0] };
        let new_high = cum[selection];
        let low_bits = int_to_bits_msb(new_low, PRECISION);
        let high_bits = int_to_bits_msb(new_high - 1, PRECISION);
        let n_fixed = num_same_from_beg(&low_bits, &high_bits);
        let fixed_bits = &low_bits[..n_fixed];
        
        eprintln!("ENC step {}: tok={} sel={} n_fixed={} bits={:?} table_len={}",
            step, tok, selection, n_fixed, fixed_bits, table.len());
        
        i += n_fixed;
        let mut nlb = low_bits[n_fixed..].to_vec(); nlb.resize(PRECISION as usize, 0);
        let mut nhb = high_bits[n_fixed..].to_vec(); nhb.resize(PRECISION as usize, 1);
        cur_int[0] = bits_msb_to_int(&nlb);
        cur_int[1] = bits_msb_to_int(&nhb) + 1;
        enc_ctx.push(tok);
        tokens.push(tok);
    }

    // RESET
    lm.tokenize(ctx_text).unwrap();

    // DECODE with tracing
    let mut cur_int2: [u64; 2] = [0, MAX_VAL];
    let mut dec_ctx = ctx.clone();
    
    for step in 0..5 {
        let (table, ids) = filter(&lm, &dec_ctx, 2.0, 300);
        let cum = subdivide(cur_int2[0], cur_int2[1], &table);
        let tok = tokens[step];
        let idx = ids.iter().position(|&t| t == tok).unwrap();
        let new_low = if idx > 0 { cum[idx-1] } else { cur_int2[0] };
        let new_high = cum[idx];
        let low_bits = int_to_bits_msb(new_low, PRECISION);
        let high_bits = int_to_bits_msb(new_high - 1, PRECISION);
        let n_fixed = num_same_from_beg(&low_bits, &high_bits);
        let fixed_bits = &low_bits[..n_fixed];
        
        eprintln!("DEC step {}: tok={} idx={} n_fixed={} bits={:?} table_len={}",
            step, tok, idx, n_fixed, fixed_bits, table.len());
        
        let mut nlb = low_bits[n_fixed..].to_vec(); nlb.resize(PRECISION as usize, 0);
        let mut nhb = high_bits[n_fixed..].to_vec(); nhb.resize(PRECISION as usize, 1);
        cur_int2[0] = bits_msb_to_int(&nlb);
        cur_int2[1] = bits_msb_to_int(&nhb) + 1;
        dec_ctx.push(tok);
        
        // Check if interval matches
        eprintln!("  interval match? {}", cur_int == cur_int2);
    }
}
