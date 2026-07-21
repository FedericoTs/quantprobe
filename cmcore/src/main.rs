// cmcore: an evolvable context-mixing lossless codec.
//
//   cmcore c <in> <out>   compress
//   cmcore d <in> <out>   decompress
//
// Architecture (PAQ-family, bitwise):
//   * The input is coded one bit at a time, MSB-first within each byte.
//   * Several context models each predict P(next bit = 1).
//   * A logistic mixer combines them with online-learned weights.
//   * Two chained SSE/APM stages recalibrate the probability.
//   * A binary arithmetic coder turns the final probability into bits.
//   * The model LEARNS ONLINE from bits already seen -- so no weights are
//     stored; the decompressor regenerates identical state (Hutter principle).
//
// CORRECTNESS: encoder and decoder run the SAME predict/update code, so any
// model is exactly reversible. Round-trip is guaranteed by construction; the
// model only affects the compressed *size*.
//
// EVOLUTION: everything inside the EVOLVE-BLOCK (the Predictor) is what the LLM
// mutates. The coder, I/O, and driver outside the block are fixed.

use std::env;
use std::fs;
use std::process::exit;

// --------------------------------------------------------------------------
// Logistic helpers (fixed infrastructure)
// --------------------------------------------------------------------------
const SQUASH_T: [i32; 33] = [
    1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994, 3348,
    3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094,
];

/// Map a stretched value d in [-2047, 2047] to a 12-bit probability [0, 4095].
#[inline]
fn squash(d: i32) -> i32 {
    if d >= 2047 {
        return 4095;
    }
    if d <= -2047 {
        return 0;
    }
    let w = d & 127;
    let i = ((d >> 7) + 16) as usize;
    (SQUASH_T[i] * (128 - w) + SQUASH_T[i + 1] * w + 64) >> 7
}

/// Build the inverse of squash: stretch[p] for p in [0, 4095].
fn build_stretch() -> Vec<i32> {
    let mut tbl = vec![0i32; 4096];
    let mut pi = 0usize;
    for x in -2047..=2047 {
        let v = squash(x) as usize;
        for j in pi..=v {
            tbl[j] = x;
        }
        pi = v + 1;
    }
    for j in pi..4096 {
        tbl[j] = 2047;
    }
    tbl
}

// ==========================================================================
// EVOLVE-BLOCK-START :: Predictor
// The LLM evolves this region. Keep `Predictor::new`, `predict`, and `update`
// signatures stable so the fixed driver below can call them.
// ==========================================================================

/// log2 of each context-model table size (entries). 22 => 4M entries per model.
const TABLE_BITS: u32 = 22;
/// Match model: hash the last MATCH_MIN bytes; MATCH_BITS sizes its hash table.
const MATCH_MIN: usize = 4;
const MATCH_BITS: u32 = 22;
/// Second, long-range match model (catches long repeated passages).
const MATCH_MIN2: usize = 7;
const MATCH_BITS2: u32 = 22;
/// Hashed context models: orders 0..6 (7) + 3 sparse + word + cap + prev-word
/// + xml-structure + column + run + 2x match-predicted-byte + 3x indirect = 21.
const N_CM: usize = 21;
/// Mixture-of-experts: KMIX context-gated first-level mixers feed a final mixer.
const KMIX: usize = 6;
const MCTX: [usize; KMIX] = [512, 256, 256, 256, 256, 256];
/// Online neural-network mixer (a small MLP), used as an extra expert.
const NN_H: usize = 24; // hidden units (learning rate via env CM_NNLR)
/// Echo-state reservoir: a fixed-random recurrent state giving the MLP cheap
/// temporal memory (the LSTM benefit without training recurrent weights).
const RES: usize = 16; // reservoir size
const RES_FEAT: usize = 4; // per-byte input features to the reservoir

fn rnd_weights(size: usize, scale: f32, salt: u32) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let h = (i as u32).wrapping_add(salt).wrapping_mul(2654435761) ^ 0x9E3779B1;
            (((h >> 8) & 0xffff) as f32 / 32767.5 - 1.0) * scale
        })
        .collect()
}

#[inline]
fn hashk(hist: u64, k: u32, salt: u32) -> u32 {
    let bytes = hist & ((1u64 << (8 * k)) - 1);
    ((bytes.wrapping_mul(0x9E3779B97F4A7C15)) >> 32) as u32 ^ salt
}

#[inline]
fn hash2(a: u32, b: u32, salt: u32) -> u32 {
    let mut h = a.wrapping_mul(0x9E3779B1) ^ b.wrapping_mul(0x85EBCA77) ^ salt;
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B3C6D);
    h ^ (h >> 12)
}

struct Predictor {
    stretch: Vec<i32>,
    dt: Vec<i64>, // adaptive step dt[n] = 65536 / (n + 2)
    mask: usize,

    // hashed context -> nonstationary bit-history state -> per-model StateMap
    state: Vec<Vec<u8>>, // N_CM tables: context -> bit-history state (0..255)
    check: Vec<Vec<u8>>, // per-slot checksum byte (hash-collision detection)
    smp: Vec<Vec<u16>>,  // per-model StateMap: state -> P(bit=1), 16-bit
    smc: Vec<Vec<u8>>,   // per-model StateMap adaptation counts
    nex: Vec<u8>,        // state transition table nex[state*2 + bit]
    ind2: Vec<u8>,       // indirect model: last byte after each order-2 context
    ind3: Vec<u8>,       // indirect model: last byte that followed each order-3 context
    ind4: Vec<u8>,       // indirect model: last byte after each order-4 context
    bases: [u32; N_CM],  // per-model context hash, refreshed each byte
    idx: [usize; N_CM],  // table index used by predict, consumed by update
    cst: [u8; N_CM],     // cached bit-history state per model (for update)
    cchk: [u8; N_CM],    // cached checksum per model (for update)

    // mixture-of-experts logistic mixing: KMIX context-gated first-level mixers
    // feed a learned final mixer
    n_in: usize,
    wmix: Vec<Vec<i32>>, // KMIX first-level mixers (each MCTX[k] * n_in weights)
    wf: Vec<i32>,        // final mixer: 256 contexts x (KMIX inputs + bias)
    mixlr: i32,          // mixer learning rate (env CM_MIXLR)
    apmw: i32,           // SSE blend weight 0..4 (env CM_APMW)
    smcap: usize,        // StateMap adaptation count cap (env CM_SMCAP)
    tx: Vec<i32>,
    mctx: [usize; KMIX], // selected context per first-level mixer
    mpk: [i32; KMIX],    // each first-level mixer's prediction
    fk: [i32; KMIX],     // stretched first-level predictions (final mixer inputs)
    mcf: usize,
    pr: i32, // final mixer prediction

    // online neural mixer (small MLP) -- an extra expert for the final mixer
    nn_w1: Vec<f32>, // NN_H * n_in (input->hidden weights)
    nn_b1: Vec<f32>, // NN_H hidden biases
    nn_w2: Vec<f32>, // NN_H hidden->output weights
    nn_b2: f32,      // output bias
    nn_h: Vec<f32>,  // hidden activations (cache for backprop)
    nn_x: Vec<f32>,  // scaled inputs (cache)
    nn_pf: f32,      // output probability (cache)
    nn_p: i32,       // stretched output (final-mixer input)
    nn_lr: f32,      // neural-mixer learning rate (env CM_NNLR)
    nn_in: usize,    // MLP input dim = n_in + RES (context features + reservoir)

    // echo-state reservoir (fixed recurrent weights, only the MLP readout learns)
    res: Vec<f32>,     // RES reservoir state (updated once per byte)
    res_wr: Vec<f32>,  // RES*RES fixed recurrent weights
    res_wi: Vec<f32>,  // RES*RES_FEAT fixed input weights

    // running context
    c0: u32,        // in-progress byte with leading-1 sentinel (1..255)
    bpos: u32,      // bits emitted in the current byte (0..7)
    hist: u64,      // last 8 bytes, most recent in low byte
    word_hash: u32, // hash of the current word (letters since last boundary)
    prev_word: u32, // hash of the previous completed word
    case_hist: u32, // 1 bit per recent byte: was it an uppercase letter?
    xstate: u32,    // packed XML/wiki structural state (in-tag, link/template depth)
    col: u32,       // column: bytes since the last newline (capped)
    run_len: u32,   // run length of the most recent byte value (capped)

    // match model
    buf: Vec<u8>,
    ht: Vec<u32>,
    match_ptr: usize,
    match_len: u32,
    mprob: Vec<u16>,
    mcnt: Vec<u8>,
    m_idx: usize,
    m_active: bool,
    m_expected: i32,

    // second, long-range match model
    ht2: Vec<u32>,
    match_ptr2: usize,
    match_len2: u32,
    mprob2: Vec<u16>,
    mcnt2: Vec<u8>,
    m2_idx: usize,
    m2_active: bool,
    m2_expected: i32,

    // two chained SSE/APM stages (256 contexts x 33 interpolation points)
    apm1: Vec<u16>,
    apm2: Vec<u16>,
    apm3: Vec<u16>,
    apm1_idx: usize,
    apm2_idx: usize,
    apm3_idx: usize,
}

fn apm_init() -> Vec<u16> {
    let mut t = vec![0u16; 256 * 33];
    for cx in 0..256 {
        for j in 0..33 {
            t[cx * 33 + j] = (squash((j as i32 - 16) * 128) * 16) as u16;
        }
    }
    t
}

/// Nonstationary bit-history state machine. A state packs two bounded counts
/// (n0, n1) of recent 0s and 1s; seeing a bit increments its count and discounts
/// the opposite count (so the state tracks *recent* statistics). nex[s*2 + bit]
/// gives the next state.
fn build_nex(thr: i32) -> Vec<u8> {
    let mut nex = vec![0u8; 512];
    for s in 0..256usize {
        let n0 = (s >> 4) as i32;
        let n1 = (s & 15) as i32;
        let a0 = (n0 + 1).min(15);
        let b0 = if n1 > thr { (n1 / 2) + 1 } else { n1 };
        nex[s * 2] = ((a0 << 4) | b0) as u8;
        let a1 = if n0 > thr { (n0 / 2) + 1 } else { n0 };
        let b1 = (n1 + 1).min(15);
        nex[s * 2 + 1] = ((a1 << 4) | b1) as u8;
    }
    nex
}

/// StateMap prior: probability implied by a state's (n0, n1) counts.
fn sm_init() -> Vec<u16> {
    let mut v = vec![0u16; 256];
    for s in 0..256 {
        let n0 = (s >> 4) as i32;
        let n1 = (s & 15) as i32;
        v[s] = (((n1 * 2 + 1) * 65536) / ((n0 + n1) * 2 + 2)) as u16;
    }
    v
}

impl Predictor {
    fn new() -> Self {
        let tbits = std::env::var("CM_TBITS").ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(TABLE_BITS);
        let size = 1usize << tbits;
        let n_in = N_CM + 3; // context models + 2 matches + bias
        let mut dt = vec![0i64; 256];
        for n in 0..256 {
            dt[n] = 65536 / (n as i64 + 2);
        }
        let envi = |k: &str, d: i64| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
        let mixlr = envi("CM_MIXLR", 12) as i32;
        let apmw = envi("CM_APMW", 2) as i32;
        let smcap = envi("CM_SMCAP", 255) as usize;
        let smthr = envi("CM_SMTHR", 3) as i32;
        Predictor {
            stretch: build_stretch(),
            dt,
            mask: size - 1,
            state: (0..N_CM).map(|_| vec![0u8; size]).collect(),
            check: (0..N_CM).map(|_| vec![0u8; size]).collect(),
            smp: (0..N_CM).map(|_| sm_init()).collect(),
            smc: (0..N_CM).map(|_| vec![0u8; 256]).collect(),
            nex: build_nex(smthr),
            ind2: vec![0u8; size],
            ind3: vec![0u8; size],
            ind4: vec![0u8; size],
            bases: [0; N_CM],
            idx: [0; N_CM],
            cst: [0; N_CM],
            cchk: [0; N_CM],
            n_in,
            wmix: (0..KMIX).map(|k| vec![0i32; MCTX[k] * n_in]).collect(),
            wf: vec![0i32; 256 * (KMIX + 2)],
            mixlr,
            apmw,
            smcap,
            tx: vec![0i32; n_in],
            mctx: [0; KMIX],
            mpk: [2048; KMIX],
            fk: [0; KMIX],
            mcf: 0,
            nn_w1: (0..NN_H * (n_in + RES)).map(|t| (((t * 7 + 3) % 11) as f32 - 5.0) * 0.02).collect(),
            nn_b1: vec![0.0f32; NN_H],
            nn_w2: (0..NN_H).map(|j| (((j * 5 + 3) % 7) as f32 - 3.0) * 0.03).collect(),
            nn_b2: 0.0,
            nn_h: vec![0.0f32; NN_H],
            nn_x: vec![0.0f32; n_in + RES],
            nn_pf: 0.5,
            nn_p: 0,
            nn_lr: std::env::var("CM_NNLR").ok().and_then(|s| s.parse().ok()).unwrap_or(0.003f32),
            nn_in: n_in + RES,
            res: vec![0.0f32; RES],
            res_wr: rnd_weights(RES * RES, 0.18, 0x111),
            res_wi: rnd_weights(RES * RES_FEAT, 0.5, 0x222),
            pr: 2048,
            c0: 1,
            bpos: 0,
            hist: 0,
            word_hash: 0,
            prev_word: 0,
            case_hist: 0,
            xstate: 0,
            col: 0,
            run_len: 0,
            buf: Vec::new(),
            ht: vec![0u32; 1usize << MATCH_BITS],
            match_ptr: 0,
            match_len: 0,
            mprob: vec![32768u16; 256],
            mcnt: vec![0u8; 256],
            m_idx: 0,
            m_active: false,
            m_expected: 0,
            ht2: vec![0u32; 1usize << MATCH_BITS2],
            match_ptr2: 0,
            match_len2: 0,
            mprob2: vec![32768u16; 256],
            mcnt2: vec![0u8; 256],
            m2_idx: 0,
            m2_active: false,
            m2_expected: 0,
            apm1: apm_init(),
            apm2: apm_init(),
            apm3: apm_init(),
            apm1_idx: 0,
            apm2_idx: 0,
            apm3_idx: 0,
        }
    }

    /// Refresh each context model's base hash from the byte history (once per byte).
    fn refresh_bases(&mut self) {
        let h = self.hist;
        self.bases[0] = 0; // order-0 (just the in-progress byte bits)
        self.bases[1] = hashk(h, 1, 0x11);
        self.bases[2] = hashk(h, 2, 0x22);
        self.bases[3] = hashk(h, 3, 0x33);
        self.bases[4] = hashk(h, 4, 0x44);
        self.bases[5] = hashk(h, 5, 0x55);
        self.bases[6] = hashk(h, 6, 0x66);
        let b1 = (h & 0xff) as u32;
        let b2 = ((h >> 8) & 0xff) as u32;
        let b3 = ((h >> 16) & 0xff) as u32;
        let b4 = ((h >> 24) & 0xff) as u32;
        self.bases[7] = hash2(b1, b3, 0x77); // sparse: lags 1 & 3
        self.bases[8] = hash2(b2, b4, 0x88); // sparse: lags 2 & 4
        self.bases[9] = hash2(b1, b4, 0x99); // sparse: lags 1 & 4
        self.bases[10] = self.word_hash ^ 0xABCD; // word model
        self.bases[11] = hash2(self.case_hist & 0xfff, (h & 0xff) as u32, 0xCA5E); // capitalization
        self.bases[12] = self.prev_word ^ 0x5678; // previous-word (word bigram)
        self.bases[13] = hash2(self.xstate, (h & 0xff) as u32, 0x111); // xml/wiki structure
        self.bases[14] = hash2(self.col, (h & 0xff) as u32, 0xC01); // column model
        self.bases[15] = hash2(self.run_len, (h & 0xff) as u32, 0x4E); // run model
        // match-predicted-byte context: lets a model refine (not just trust) the match
        let mpred = if self.match_len > 0 && self.match_ptr < self.buf.len() {
            self.buf[self.match_ptr] as u32
        } else {
            256
        };
        self.bases[16] = hash2(mpred, (h & 0xff) as u32, 0x9A7);
        let mpred2 = if self.match_len2 > 0 && self.match_ptr2 < self.buf.len() {
            self.buf[self.match_ptr2] as u32
        } else {
            256
        };
        self.bases[17] = hash2(mpred2, (h & 0xff) as u32, 0xB13);
        // indirect order-3 model: record the byte that followed the previous
        // order-3 context, then predict the next byte from the current one.
        let last = (self.hist & 0xff) as u8;
        let prev2 = hashk(self.hist >> 8, 2, 0x22) as usize & self.mask;
        self.ind2[prev2] = last;
        self.bases[18] = hash2(self.ind2[self.bases[2] as usize & self.mask] as u32, (h & 0xff) as u32, 0xC2);
        let prev3 = hashk(self.hist >> 8, 3, 0x33) as usize & self.mask;
        self.ind3[prev3] = last;
        self.bases[19] = hash2(self.ind3[self.bases[3] as usize & self.mask] as u32, (h & 0xff) as u32, 0xC3);
        let prev4 = hashk(self.hist >> 8, 4, 0x44) as usize & self.mask;
        self.ind4[prev4] = last;
        self.bases[20] = hash2(self.ind4[self.bases[4] as usize & self.mask] as u32, (h & 0xff) as u32, 0xC4);
    }

    #[inline]
    fn predict(&mut self) -> u32 {
        let c0 = self.c0;
        for i in 0..N_CM {
            let h = self.bases[i] ^ c0.wrapping_mul(0x9E3779B1);
            let id = (h as usize) & self.mask;
            let chk = (h >> 24) as u8;
            self.idx[i] = id;
            self.cchk[i] = chk;
            // on a hash collision (checksum mismatch) treat the context as fresh
            let st = if self.check[i][id] == chk { self.state[i][id] } else { 0 };
            self.cst[i] = st;
            self.tx[i] = self.stretch[(self.smp[i][st as usize] >> 4) as usize];
        }

        // match model 1 (short) input
        self.m_active = false;
        let mut m_in = 0i32;
        if self.match_len > 0 && self.match_ptr < self.buf.len() {
            let predicted = self.buf[self.match_ptr] as u32;
            let exp_bit = ((predicted >> (7 - self.bpos)) & 1) as i32;
            let ctx = self.match_len.min(127) as usize * 2 + exp_bit as usize;
            self.m_idx = ctx;
            self.m_expected = exp_bit;
            self.m_active = true;
            m_in = self.stretch[(self.mprob[ctx] >> 4) as usize];
        }
        // match model 2 (long) input
        self.m2_active = false;
        let mut m2_in = 0i32;
        if self.match_len2 > 0 && self.match_ptr2 < self.buf.len() {
            let predicted = self.buf[self.match_ptr2] as u32;
            let exp_bit = ((predicted >> (7 - self.bpos)) & 1) as i32;
            let ctx = self.match_len2.min(127) as usize * 2 + exp_bit as usize;
            self.m2_idx = ctx;
            self.m2_expected = exp_bit;
            self.m2_active = true;
            m2_in = self.stretch[(self.mprob2[ctx] >> 4) as usize];
        }
        self.tx[N_CM] = m_in; // match-1 input
        self.tx[N_CM + 1] = m2_in; // match-2 input
        self.tx[N_CM + 2] = 256; // bias input

        // mixture-of-experts: KMIX context-gated first-level mixers, then a
        // learned final mixer combines their predictions.
        let mflag = if self.m_active { 256 } else { 0 };
        let h2 = (((self.hist & 0xffff) as u32).wrapping_mul(0x9E3779B1) >> 24) as usize;
        let h3 = (((self.hist & 0xffffff) as u32).wrapping_mul(0x85EBCA77) >> 24) as usize;
        let mstate = (self.match_len.min(15) as usize) | ((self.match_len2.min(15) as usize) << 4);
        let ctxs = [
            ((c0 as usize) & 0xff) | mflag,   // mixer 0: in-progress byte + match flag
            (self.hist & 0xff) as usize,      // mixer 1: previous byte
            h2,                               // mixer 2: last two bytes
            (self.word_hash as usize) & 0xff, // mixer 3: current word
            h3,                               // mixer 4: last three bytes
            mstate,                           // mixer 5: match-length state
        ];
        for k in 0..KMIX {
            let ctx = ctxs[k];
            self.mctx[k] = ctx;
            let base = ctx * self.n_in;
            let mut dot: i64 = 0;
            for i in 0..self.n_in {
                dot += self.tx[i] as i64 * self.wmix[k][base + i] as i64;
            }
            let mut mp = squash((dot >> 16) as i32);
            if mp < 1 { mp = 1; } else if mp > 4094 { mp = 4094; }
            self.mpk[k] = mp;
            self.fk[k] = self.stretch[mp as usize];
        }
        // neural expert: a small online MLP over the context inputs + reservoir
        for i in 0..self.n_in {
            self.nn_x[i] = self.tx[i] as f32 * (1.0 / 256.0);
        }
        for i in 0..RES {
            self.nn_x[self.n_in + i] = self.res[i];
        }
        let mut o = self.nn_b2;
        for j in 0..NN_H {
            let wbase = j * self.nn_in;
            let mut s = self.nn_b1[j];
            for i in 0..self.nn_in {
                s += self.nn_w1[wbase + i] * self.nn_x[i];
            }
            let hj = s.tanh();
            self.nn_h[j] = hj;
            o += self.nn_w2[j] * hj;
        }
        let pf = 1.0 / (1.0 + (-o).exp());
        self.nn_pf = pf;
        let mut pq = (pf * 4096.0) as i32;
        if pq < 1 { pq = 1; } else if pq > 4094 { pq = 4094; }
        self.nn_p = self.stretch[pq as usize];

        let mcf = (c0 as usize) & 0xff;
        self.mcf = mcf;
        let fbase = mcf * (KMIX + 2);
        let mut dotf: i64 = 256i64 * self.wf[fbase + KMIX + 1] as i64; // bias term
        for k in 0..KMIX {
            dotf += self.fk[k] as i64 * self.wf[fbase + k] as i64;
        }
        dotf += self.nn_p as i64 * self.wf[fbase + KMIX] as i64; // neural expert
        let mut mp = squash((dotf >> 16) as i32);
        if mp < 1 {
            mp = 1;
        } else if mp > 4094 {
            mp = 4094;
        }
        self.pr = mp;

        // SSE/APM stage 1 (keyed on the in-progress byte)
        let s1 = self.stretch[mp as usize];
        let w1 = s1 & 127;
        let i1 = ((c0 as usize) & 0xff) * 33 + ((s1 >> 7) + 16) as usize;
        self.apm1_idx = i1;
        let mut a1 = (self.apm1[i1] as i32 * (128 - w1) + self.apm1[i1 + 1] as i32 * w1) >> 11;
        if a1 < 1 {
            a1 = 1;
        } else if a1 > 4094 {
            a1 = 4094;
        }

        // SSE/APM stage 2 (chained, keyed on the previous byte)
        let prev = (self.hist & 0xff) as usize;
        let s2 = self.stretch[a1 as usize];
        let w2 = s2 & 127;
        let i2 = prev * 33 + ((s2 >> 7) + 16) as usize;
        self.apm2_idx = i2;
        let mut a2 = (self.apm2[i2] as i32 * (128 - w2) + self.apm2[i2 + 1] as i32 * w2) >> 11;
        if a2 < 1 { a2 = 1; } else if a2 > 4094 { a2 = 4094; }

        // SSE/APM stage 3 (chained, keyed on the last two bytes)
        let ctx3 = (((self.hist & 0xffff) as u32).wrapping_mul(0x9E3779B1) >> 24) as usize;
        let s3 = self.stretch[a2 as usize];
        let w3 = s3 & 127;
        let i3 = ctx3 * 33 + ((s3 >> 7) + 16) as usize;
        self.apm3_idx = i3;
        let a3 = (self.apm3[i3] as i32 * (128 - w3) + self.apm3[i3 + 1] as i32 * w3) >> 11;

        let mut p = (a3 * self.apmw + mp * (4 - self.apmw)) >> 2;
        if p < 1 {
            p = 1;
        } else if p > 4094 {
            p = 4094;
        }
        p as u32
    }

    #[inline]
    fn update(&mut self, bit: u32) {
        let target: i64 = if bit == 1 { 65535 } else { 0 };

        // context-model updates: adapt the per-state probability, then advance
        // the bit-history state of the context we just used.
        for i in 0..N_CM {
            let st = self.cst[i] as usize;
            let n = self.smc[i][st] as usize;
            let cur = self.smp[i][st] as i64;
            let nv = cur + (((target - cur) * self.dt[n]) >> 16);
            self.smp[i][st] = nv.clamp(0, 65535) as u16;
            if n < self.smcap {
                self.smc[i][st] = (n + 1) as u8;
            }
            let id = self.idx[i];
            self.state[i][id] = self.nex[st * 2 + bit as usize];
            self.check[i][id] = self.cchk[i];
        }

        // match StateMap update; drop the match if its predicted bit was wrong
        if self.m_active {
            let id = self.m_idx;
            let n = self.mcnt[id] as usize;
            let cur = self.mprob[id] as i64;
            let nv = cur + (((target - cur) * self.dt[n]) >> 16);
            self.mprob[id] = nv.clamp(0, 65535) as u16;
            if n < 255 {
                self.mcnt[id] = (n + 1) as u8;
            }
            if (bit as i32) != self.m_expected {
                self.match_len = 0;
            }
        }
        if self.m2_active {
            let id = self.m2_idx;
            let n = self.mcnt2[id] as usize;
            let cur = self.mprob2[id] as i64;
            let nv = cur + (((target - cur) * self.dt[n]) >> 16);
            self.mprob2[id] = nv.clamp(0, 65535) as u16;
            if n < 255 {
                self.mcnt2[id] = (n + 1) as u8;
            }
            if (bit as i32) != self.m2_expected {
                self.match_len2 = 0;
            }
        }

        // SSE/APM updates: nudge both interpolation endpoints toward the bit
        let t = target;
        let j1 = self.apm1_idx;
        self.apm1[j1] = (self.apm1[j1] as i64 + ((t - self.apm1[j1] as i64) >> 7)) as u16;
        self.apm1[j1 + 1] = (self.apm1[j1 + 1] as i64 + ((t - self.apm1[j1 + 1] as i64) >> 7)) as u16;
        let j2 = self.apm2_idx;
        self.apm2[j2] = (self.apm2[j2] as i64 + ((t - self.apm2[j2] as i64) >> 7)) as u16;
        self.apm2[j2 + 1] = (self.apm2[j2 + 1] as i64 + ((t - self.apm2[j2 + 1] as i64) >> 7)) as u16;
        let j3 = self.apm3_idx;
        self.apm3[j3] = (self.apm3[j3] as i64 + ((t - self.apm3[j3] as i64) >> 7)) as u16;
        self.apm3[j3 + 1] = (self.apm3[j3 + 1] as i64 + ((t - self.apm3[j3 + 1] as i64) >> 7)) as u16;

        // mixture-of-experts updates: each first-level mixer trains on its own
        // error; the final mixer trains on the combined error.
        let target12 = (bit as i32) << 12;
        for k in 0..KMIX {
            let errk = (target12 - self.mpk[k]) * self.mixlr;
            let base = self.mctx[k] * self.n_in;
            for i in 0..self.n_in {
                let nw = self.wmix[k][base + i] + ((self.tx[i] * errk) >> 16);
                self.wmix[k][base + i] = nw.clamp(-(1 << 20), 1 << 20);
            }
        }
        let errf = (target12 - self.pr) * self.mixlr;
        let fbase = self.mcf * (KMIX + 2);
        for k in 0..KMIX {
            let nw = self.wf[fbase + k] + ((self.fk[k] * errf) >> 16);
            self.wf[fbase + k] = nw.clamp(-(1 << 20), 1 << 20);
        }
        let nwn = self.wf[fbase + KMIX] + ((self.nn_p * errf) >> 16);
        self.wf[fbase + KMIX] = nwn.clamp(-(1 << 20), 1 << 20);
        let nb = self.wf[fbase + KMIX + 1] + ((256 * errf) >> 16);
        self.wf[fbase + KMIX + 1] = nb.clamp(-(1 << 20), 1 << 20);

        // neural mixer backprop (logistic output + cross-entropy)
        let d = self.nn_pf - bit as f32;
        for j in 0..NN_H {
            let hj = self.nn_h[j];
            let dh = d * self.nn_w2[j] * (1.0 - hj * hj);
            self.nn_w2[j] -= self.nn_lr * d * hj;
            let wbase = j * self.nn_in;
            for i in 0..self.nn_in {
                self.nn_w1[wbase + i] -= self.nn_lr * dh * self.nn_x[i];
            }
            self.nn_b1[j] -= self.nn_lr * dh;
        }
        self.nn_b2 -= self.nn_lr * d;

        // advance the in-progress byte
        self.c0 = (self.c0 << 1) | bit;
        if self.c0 >= 256 {
            let byte = (self.c0 & 0xff) as u8;
            self.c0 = 1;
            self.bpos = 0;
            self.hist = (self.hist << 8) | byte as u64;

            // echo-state reservoir update (once per byte); fixed recurrent weights,
            // leaky integration, tanh nonlinearity -> cheap temporal features
            let u = [
                (byte as f32) * (1.0 / 128.0) - 1.0,
                if byte.is_ascii_alphabetic() { 1.0 } else { -1.0 },
                if byte == b' ' || byte == b'\n' { 1.0 } else { -1.0 },
                1.0,
            ];
            let mut nr = [0.0f32; RES];
            for a in 0..RES {
                let mut s = 0.0f32;
                let rb = a * RES;
                for b2 in 0..RES {
                    s += self.res_wr[rb + b2] * self.res[b2];
                }
                let ib = a * RES_FEAT;
                for f in 0..RES_FEAT {
                    s += self.res_wi[ib + f] * u[f];
                }
                nr[a] = s.tanh();
            }
            for a in 0..RES {
                self.res[a] = 0.7 * self.res[a] + 0.3 * nr[a];
            }

            // word model: extend hash on letters, reset at a word boundary
            let lc = byte | 0x20; // ascii-lowercase fold
            if lc >= b'a' && lc <= b'z' {
                self.word_hash = self.word_hash.wrapping_add(lc as u32 + 1).wrapping_mul(2654435761);
            } else {
                if self.word_hash != 0 {
                    self.prev_word = self.word_hash; // remember the word that just ended
                }
                self.word_hash = 0;
            }
            let upper = (byte >= b'A' && byte <= b'Z') as u32;
            self.case_hist = (self.case_hist << 1) | upper;
            // track XML/wiki nesting: in-tag flag (bit 0), link depth (bits 1-3),
            // template depth (bits 4-6)
            match byte {
                b'<' => self.xstate |= 1,
                b'>' => self.xstate &= !1,
                b'[' => {
                    let d = (((self.xstate >> 1) & 7) + 1).min(7);
                    self.xstate = (self.xstate & !(7 << 1)) | (d << 1);
                }
                b']' => {
                    let d = ((self.xstate >> 1) & 7).saturating_sub(1);
                    self.xstate = (self.xstate & !(7 << 1)) | (d << 1);
                }
                b'{' => {
                    let d = (((self.xstate >> 4) & 7) + 1).min(7);
                    self.xstate = (self.xstate & !(7 << 4)) | (d << 4);
                }
                b'}' => {
                    let d = ((self.xstate >> 4) & 7).saturating_sub(1);
                    self.xstate = (self.xstate & !(7 << 4)) | (d << 4);
                }
                _ => {}
            }
            // column model: bytes since the last newline
            if byte == b'\n' {
                self.col = 0;
            } else {
                self.col = (self.col + 1).min(1023);
            }
            // run model: how many times the most recent byte value has repeated
            if byte as u64 == ((self.hist >> 8) & 0xff) {
                self.run_len = (self.run_len + 1).min(255);
            } else {
                self.run_len = 0;
            }

            // continue or reset both matches against the just-finished byte
            if self.match_len > 0
                && self.match_ptr < self.buf.len()
                && self.buf[self.match_ptr] == byte
            {
                self.match_ptr += 1;
                self.match_len = (self.match_len + 1).min(65535);
            } else {
                self.match_len = 0;
            }
            if self.match_len2 > 0
                && self.match_ptr2 < self.buf.len()
                && self.buf[self.match_ptr2] == byte
            {
                self.match_ptr2 += 1;
                self.match_len2 = (self.match_len2 + 1).min(65535);
            } else {
                self.match_len2 = 0;
            }
            self.buf.push(byte);
            let pos = self.buf.len();
            if pos >= MATCH_MIN {
                let key = self.hist & ((1u64 << (8 * MATCH_MIN as u32)) - 1);
                let h = (key.wrapping_mul(0x9E3779B97F4A7C15) >> (64 - MATCH_BITS)) as usize;
                if self.match_len == 0 {
                    let cand = self.ht[h] as usize;
                    if cand != 0 && cand < pos {
                        self.match_ptr = cand;
                        self.match_len = MATCH_MIN as u32;
                    }
                }
                self.ht[h] = pos as u32;
            }
            if pos >= MATCH_MIN2 {
                let key = self.hist & ((1u64 << (8 * MATCH_MIN2 as u32)) - 1);
                let h = (key.wrapping_mul(0xD6E8FEB86659FD93) >> (64 - MATCH_BITS2)) as usize;
                if self.match_len2 == 0 {
                    let cand = self.ht2[h] as usize;
                    if cand != 0 && cand < pos {
                        self.match_ptr2 = cand;
                        self.match_len2 = MATCH_MIN2 as u32;
                    }
                }
                self.ht2[h] = pos as u32;
            }

            self.refresh_bases();
        } else {
            self.bpos += 1;
        }
    }
}

// ==========================================================================
// EVOLVE-BLOCK-END
// ==========================================================================

// --------------------------------------------------------------------------
// Binary arithmetic coder (fixed infrastructure)
// --------------------------------------------------------------------------
struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Encoder {
    fn new() -> Self {
        Encoder { x1: 0, x2: 0xFFFF_FFFF, out: Vec::new() }
    }
    #[inline]
    fn encode(&mut self, bit: u32, p: u32) {
        // p = P(bit == 1), 12-bit (1..4094)
        let range = self.x2 - self.x1;
        let xmid = self.x1
            + (range >> 12) * p
            + (((range & 0xFFF) * p) >> 12);
        if bit == 1 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.out.push((self.x2 >> 24) as u8);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
        }
    }
    fn flush(mut self) -> Vec<u8> {
        // emit 4 bytes of x1 so the decoder always has a value within range
        for s in [24, 16, 8, 0] {
            self.out.push((self.x1 >> s) as u8);
        }
        self.out
    }
}

struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    inp: &'a [u8],
    pos: usize,
}

impl<'a> Decoder<'a> {
    fn new(inp: &'a [u8]) -> Self {
        let mut d = Decoder { x1: 0, x2: 0xFFFF_FFFF, x: 0, inp, pos: 0 };
        for _ in 0..4 {
            d.x = (d.x << 8) | d.next() as u32;
        }
        d
    }
    #[inline]
    fn next(&mut self) -> u8 {
        let b = if self.pos < self.inp.len() { self.inp[self.pos] } else { 0 };
        self.pos += 1;
        b
    }
    #[inline]
    fn decode(&mut self, p: u32) -> u32 {
        let range = self.x2 - self.x1;
        let xmid = self.x1
            + (range >> 12) * p
            + (((range & 0xFFF) * p) >> 12);
        let bit = if self.x <= xmid { 1 } else { 0 };
        if bit == 1 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xFF00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 0xFF;
            self.x = (self.x << 8) | self.next() as u32;
        }
        bit
    }
}

// --------------------------------------------------------------------------
// Driver (fixed infrastructure)
// --------------------------------------------------------------------------
fn cm_compress(input: &[u8]) -> Vec<u8> {
    let mut pred = Predictor::new();
    let mut enc = Encoder::new();
    for &byte in input {
        let mut b = 7i32;
        while b >= 0 {
            let bit = ((byte >> b) & 1) as u32;
            let p = pred.predict();
            enc.encode(bit, p);
            pred.update(bit);
            b -= 1;
        }
    }
    let coder = enc.flush();
    let mut out = Vec::with_capacity(coder.len() + 8);
    out.extend_from_slice(&(input.len() as u64).to_be_bytes());
    out.extend_from_slice(&coder);
    out
}

fn cm_decompress(blob: &[u8]) -> Vec<u8> {
    let n = u64::from_be_bytes(blob[0..8].try_into().unwrap()) as usize;
    let mut pred = Predictor::new();
    let mut dec = Decoder::new(&blob[8..]);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut byte = 0u8;
        for _ in 0..8 {
            let p = pred.predict();
            let bit = dec.decode(p);
            pred.update(bit);
            byte = (byte << 1) | (bit as u8);
        }
        out.push(byte);
    }
    out
}

// Optional dictionary pre-pass (WRT-style), enabled by env CM_DICT=1. Common
// lowercase words are replaced by single byte codes drawn from byte values absent
// from the input, extending the model's effective context reach. The word list is
// embedded (counts once toward the decompressor); the output is self-describing
// via a leading flag byte, so decode needs no env.
const DICT: &[&str] = &[
    "the", "of", "and", "to", "in", "is", "that", "for", "it", "as", "was", "with",
    "be", "by", "on", "not", "he", "this", "are", "or", "his", "from", "at", "which",
    "but", "have", "an", "had", "they", "you", "were", "their", "one", "all", "we",
    "can", "her", "has", "there", "been", "if", "more", "when", "will", "would", "who",
    "so", "no", "said", "what", "up", "its", "about", "into", "than", "them", "only",
    "other", "new", "some", "could", "time", "these", "two", "may", "then", "do",
    "first", "any", "my", "now", "such", "like", "our", "over", "man", "me", "even",
    "most", "made", "after", "also", "did", "many", "before", "must", "through",
    "back", "years", "where", "much", "your", "way", "well", "down", "should",
    "because", "each", "just", "those", "people", "how", "too", "little", "state",
    "good", "very", "make", "world", "still", "see", "own", "men", "work", "long",
    "here", "get", "both", "between", "life", "being", "under", "never", "day", "same",
    "another", "know", "while", "last", "might", "us", "great", "old", "year", "off",
    "come", "since", "against", "go", "came", "right", "used", "take", "three",
];

fn compress(input: &[u8]) -> Vec<u8> {
    if std::env::var("CM_DICT").ok().as_deref() == Some("0") {
        let mut out = Vec::with_capacity(input.len() / 3 + 16);
        out.push(0u8); // dictionary disabled (CM_DICT=0)
        out.extend_from_slice(&cm_compress(input));
        return out;
    }
    let mut used = [false; 256];
    for &b in input {
        used[b as usize] = true;
    }
    let free: Vec<u8> = (0u16..256).filter(|&b| !used[b as usize]).map(|b| b as u8).collect();
    if free.len() < 2 {
        let mut out = Vec::with_capacity(input.len() / 3 + 16);
        out.push(0u8);
        out.extend_from_slice(&cm_compress(input));
        return out;
    }
    let cap_code = free[0]; // flag: capitalize the first letter of the next word code
    let n = (free.len() - 1).min(DICT.len()).min(254);
    let mut wcode: std::collections::HashMap<&[u8], u8> = std::collections::HashMap::new();
    for i in 0..n {
        wcode.insert(DICT[i].as_bytes(), free[1 + i]);
    }
    let mut tr = Vec::with_capacity(input.len());
    let mut i = 0;
    while i < input.len() {
        let b = input[i];
        if b.is_ascii_alphabetic() {
            let start = i;
            while i < input.len() && input[i].is_ascii_alphabetic() {
                i += 1;
            }
            let word = &input[start..i];
            let all_lower = word.iter().all(|c| c.is_ascii_lowercase());
            let cap = word.len() >= 2
                && word[0].is_ascii_uppercase()
                && word[1..].iter().all(|c| c.is_ascii_lowercase());
            if all_lower {
                match wcode.get(word) {
                    Some(&code) => tr.push(code),
                    None => tr.extend_from_slice(word),
                }
            } else if cap {
                let mut lw = word.to_vec();
                lw[0] = lw[0].to_ascii_lowercase();
                match wcode.get(lw.as_slice()) {
                    Some(&code) => {
                        tr.push(cap_code);
                        tr.push(code);
                    }
                    None => tr.extend_from_slice(word),
                }
            } else {
                tr.extend_from_slice(word);
            }
        } else {
            tr.push(b);
            i += 1;
        }
    }
    let cm = cm_compress(&tr);
    let mut out = Vec::with_capacity(cm.len() + n + 3);
    out.push(1u8); // dictionary used
    out.push(cap_code);
    out.push(n as u8);
    out.extend_from_slice(&free[1..1 + n]);
    out.extend_from_slice(&cm);
    out
}

fn decompress(blob: &[u8]) -> Vec<u8> {
    if blob[0] == 0 {
        return cm_decompress(&blob[1..]);
    }
    let cap_code = blob[1];
    let n = blob[2] as usize;
    let codes = &blob[3..3 + n];
    let mut map: [Option<&'static [u8]>; 256] = [None; 256];
    for i in 0..n {
        map[codes[i] as usize] = Some(DICT[i].as_bytes());
    }
    let tr = cm_decompress(&blob[3 + n..]);
    let mut out = Vec::with_capacity(tr.len() * 2);
    let mut j = 0;
    while j < tr.len() {
        let b = tr[j];
        if b == cap_code && j + 1 < tr.len() {
            j += 1;
            match map[tr[j] as usize] {
                Some(w) => {
                    out.push(w[0].to_ascii_uppercase());
                    out.extend_from_slice(&w[1..]);
                }
                None => {
                    out.push(cap_code);
                    out.push(tr[j]);
                }
            }
        } else if let Some(w) = map[b as usize] {
            out.extend_from_slice(w);
        } else {
            out.push(b);
        }
        j += 1;
    }
    out
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 || (args[1] != "c" && args[1] != "d") {
        eprintln!("usage: cmcore c|d <in> <out>");
        exit(2);
    }
    let input = fs::read(&args[2]).unwrap_or_else(|e| {
        eprintln!("read {}: {}", args[2], e);
        exit(1);
    });
    let output = if args[1] == "c" {
        compress(&input)
    } else {
        decompress(&input)
    };
    fs::write(&args[3], &output).unwrap_or_else(|e| {
        eprintln!("write {}: {}", args[3], e);
        exit(1);
    });
}
