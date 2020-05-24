use num::{Complex, Zero};
use rand::Rng;
use rustfft::FFTplanner;
use std::collections::VecDeque;
use std::default::Default;

pub struct OfdmConfig {
    /// Number of frequency channels (number of OFDM symbols per OFDM channel)
    pub num_channels: usize,
    /// The cyclic prefix of num_channels/cyclic_prefix_frac is added to the OFDM symbol
    pub cyclic_prefix_frac: usize,
}

pub struct PktDetectConfig {
    /// Number of samples used in the two averaging windows (should be
    /// even)
    pub num_samps: usize,
    /// Detect threshold. Says packet is detected if (avg of second
    /// window) >= (mean of first window) + thresh * (std dev in first
    /// window)
    pub thresh: f32,
}

/// Configuration for how the packet is constructed
pub struct PktConfig {
    /// Number of data bits per packet
    num_data_bits: usize,
    /// Number of actual bits, including extra bits for ECC
    num_bits: usize,
    /// The pilot symbol (exactly OfdmConfig::num_channels long)
    pilot: Vec<Complex<f32>>,
}

pub struct Config {
    pub ofdm: OfdmConfig,
    pub pkt_detect: PktDetectConfig,
    pub pkt: PktConfig,
}

impl OfdmConfig {
    /// Number of samples in the cyclic prefix
    pub fn prefix_len(&self) -> usize {
        assert_eq!(self.num_channels % self.cyclic_prefix_frac, 0);
        self.num_channels / self.cyclic_prefix_frac
    }

    /// Number of samples in the entire symbol, including cyclic prefix
    pub fn symbol_len(&self) -> usize {
        self.num_channels + self.prefix_len()
    }
}

impl Config {
    /// Total number of samples per packet
    pub fn samps_per_pkt(&self) -> usize {
        assert_eq!(self.pkt.num_bits % self.ofdm.num_channels, 0);
        self.ofdm.symbol_len() * self.pkt.num_bits / self.ofdm.num_channels
    }
}

impl Default for Config {
    fn default() -> Self {
        let num_channels = 128;
        // Let pilot be a (deterministic) random number
        let mut rng = rand_pcg::Pcg32::new(0, 0);
        let pilot = (0..num_channels)
            .map(|_| Complex::<f32>::new(rng.gen(), rng.gen()))
            .collect::<Vec<_>>();

        Self {
            ofdm: OfdmConfig {
                num_channels,
                cyclic_prefix_frac: 4,
            },
            pkt_detect: PktDetectConfig {
                num_samps: 128,
                thresh: 2.,
            },
            pkt: PktConfig {
                num_data_bits: 256,
                num_bits: 512,
                pilot,
            },
        }
    }
}

/// A moving window of the sum and std. dev. of values
/// inside. Recalculates the value to maintain precision
pub struct WindowSum {
    /// Number of values in the window
    win_size: usize,
    /// The values currently in the window
    vals: VecDeque<f32>,
    /// Current sum of values in the window
    sum: f32,
    /// Current sum of square of values in the window
    sum_sq: f32,
    /// Number of samples since the last time we recomputed the sum
    /// (for numeric precision)
    samps_since_last: usize,
}

impl WindowSum {
    fn new(win_size: usize) -> Self {
        Self {
            win_size,
            vals: VecDeque::with_capacity(win_size),
            sum: 0.,
            sum_sq: 0.,
            samps_since_last: 0,
        }
    }

    /// Push a new value into the window. If window already has
    /// win_size values, returns the outgoing value
    fn push(&mut self, v: f32) -> Option<f32> {
        // Add in the new value
        self.sum += v;
        self.sum_sq += v * v;
        self.vals.push_back(v);
        self.samps_since_last += 1;

        if let Some(vo) = self.vals.front() {
            self.sum -= vo;
        }

        // Pop if already full
        let res = if self.vals.len() == self.win_size {
            Some(self.vals.pop_front().unwrap())
        } else {
            None
        };

        // Recompute if necessary
        if self.samps_since_last > 65535 {
            self.sum = self.vals.iter().sum();
            self.sum_sq = self.vals.iter().map(|v| v * v).sum();
            self.samps_since_last = 0;
        }

        res
    }

    fn avg(&self) -> f32 {
        if self.vals.len() == 0 {
            0.
        } else {
            self.sum / self.vals.len() as f32
        }
    }

    fn std_dev(&self) -> f32 {
        if self.vals.len() == 0 {
            0.
        } else {
            (self.sum_sq / self.vals.len() as f32 - self.avg() * self.avg()).sqrt()
        }
    }

    /// Have >= `win_size` values been pushed?
    #[allow(dead_code)]
    fn full(&self) -> bool {
        assert!(self.vals.len() <= self.win_size);
        self.vals.len() == self.win_size
    }
}

/// Takes a constant stream of packets and tells us exactly where a
/// packet began
pub struct PktDetector<'c> {
    config: &'c Config,
    /// Two sliding windows to do thresholding
    windows: (WindowSum, WindowSum),
}

impl<'c> PktDetector<'c> {
    pub fn new(config: &'c Config) -> Self {
        assert_eq!(config.pkt_detect.num_samps % 2, 0);
        let n = config.pkt_detect.num_samps / 2;
        Self {
            config,
            windows: (WindowSum::new(n), WindowSum::new(n)),
        }
    }

    /// Indicator which we use to decide if a packet has started
    fn indicator(&self) -> f32 {
        if self.windows.0.std_dev() == 0. {
            if self.windows.1.avg() > 0. {
                std::f32::MAX
            } else {
                0.
            }
        } else {
            (self.windows.1.avg() - self.windows.0.avg()) / self.windows.0.std_dev()
        }
    }

    /// Push a sample into the windows. Return true if we have enough
    /// samples to start detecting packets
    fn push_samp(&mut self, samp: Complex<f32>) -> bool {
        // Put value in window
        if let Some(samp) = self.windows.1.push(samp.norm()) {
            if let Some(_) = self.windows.0.push(samp) {
                // We need enough seed samples before we can start
                // detecting packets
                return true;
            }
        }
        return false;
    }

    /// Push samples one-by-one. When it returns true, send the next
    /// >= `config.pkt_detect.thresh` samples to get the exact start
    /// id of the packet. `pkt_start_index` must be called with the
    /// samples immediately following the one where it returned
    /// true. Following call to `pkt_start_index`, `push` can be
    /// called again with the samples that come after
    /// `pkt_start_index`
    pub fn push(&mut self, samp: Complex<f32>) -> bool {
        if !self.push_samp(samp) {
            return false;
        }
        // Has the indicator met the threshold?
        self.indicator() > self.config.pkt_detect.thresh
    }

    /// Given at-least `config.pkt_detect.num_samples` samples after
    /// `Self::push` returned true, returns the sample index at which
    /// the packet started
    pub fn pkt_start_index(&mut self, samps: &[Complex<f32>]) -> usize {
        // Warning: Ideally if indicator peaks at the current sample,
        // we should return -1, but we return 0. This is an unlikely
        // event in which case our answer will be 1 sample off. OFDM
        // is more than capable of handling this much offset
        let (mut max_indicator, mut max_idx) = (self.indicator(), 0);
        // Start pushing the samples and find the index at which
        // indicator peaks
        for (i, samp) in samps
            .iter()
            .take(self.config.pkt_detect.num_samps)
            .enumerate()
        {
            self.push_samp(*samp);
            if self.indicator() > max_indicator {
                max_indicator = self.indicator();
                max_idx = i;
            }
        }
        max_idx
    }
}

/// A symbol that has been received
struct RxOfdmSymbol<'c> {
    config: &'c Config,
    /// Decoded symbols
    symbols: Vec<Complex<f32>>,
}

/// Symbol made ready to transmit
struct TxOfdmSymbol<'c> {
    config: &'c Config,
    samples: Vec<Complex<f32>>,
}

impl<'a> TxOfdmSymbol<'a> {
    /// Construct a symbol given a set of input symbols to transmit
    /// Warning: This implementation does some unnecessary copies and
    /// other stupid things
    pub fn new(symbols: &[Complex<f32>], config: &'a Config) -> Self {
        assert_eq!(symbols.len(), config.ofdm.num_channels);

        // Perform IFFT
        let mut planner = FFTplanner::new(true);
        let fft = planner.plan_fft(config.ofdm.num_channels);
        let mut scratch_inp: Vec<_> = symbols.into();
        let mut fft_out = vec![Complex::zero(); config.ofdm.num_channels];
        fft.process(&mut scratch_inp, &mut fft_out);

        Self {
            config,
            samples: fft_out,
        }
    }

    /// Return samples with a cyclic prefix added
    pub fn samps_with_cyclic_prefix(&self) -> Vec<Complex<f32>> {
        assert_eq!(self.samples.len(), self.config.ofdm.num_channels);
        let mut res = Vec::new();
        let num_channels = self.config.ofdm.num_channels;
        let prefix_len = self.config.ofdm.prefix_len();
        res.extend(&self.samples[num_channels - prefix_len - 1..]);
        res.extend(&self.samples);

        res
    }
}

impl<'a> RxOfdmSymbol<'a> {
    /// Decode symbol from samples
    pub fn new(samples: &[Complex<f32>], config: &'a Config) -> Self {
        assert_eq!(samples.len(), config.ofdm.num_channels);

        // Take FFT
        let mut planner = FFTplanner::new(false);
        let fft = planner.plan_fft(config.ofdm.num_channels);
        let mut scratch_inp: Vec<_> = samples.into();
        let mut fft_out = vec![Complex::zero(); config.ofdm.num_channels];
        fft.process(&mut scratch_inp, &mut fft_out);

        Self {
            config,
            symbols: fft_out,
        }
    }
}

pub fn bpsk_encode(data: &[bool]) -> Vec<Complex<f32>> {
    data.iter()
        .map(|b| {
            if *b {
                Complex::new(1., 0.)
            } else {
                Complex::new(0., 1.)
            }
        })
        .collect::<Vec<_>>()
}

/// Construct a packet containing the given data. Data must be exactly
/// ConfigPkt::num_data_bits long
pub fn construct_pkt(data: &[bool], config: &Config) -> Vec<Complex<f32>> {
    assert_eq!(data.len(), config.pkt.num_data_bits);
    assert_eq!(config.pkt.num_bits % config.pkt.num_data_bits, 0);

    let mut res = Vec::with_capacity(config.samps_per_pkt());
    // Add the pilot symbol
    let pilot = TxOfdmSymbol::new(&config.pkt.pilot, config);
    res.append(&mut pilot.samps_with_cyclic_prefix());

    // Add the data symbols to create a sequence of symbols that we'll
    // repeat to fill `num_bits`
    let mut data_samps = Vec::new();
    let num_data_symbols = config.pkt.num_data_bits / config.ofdm.num_channels;
    for symbol_id in 0..num_data_symbols {
        // Do simple BPSK modulation for maximum robustness
        let start = symbol_id * config.ofdm.num_channels;
        let bpsk = bpsk_encode(&data[start..start + config.ofdm.num_channels]);
        let symbol = TxOfdmSymbol::new(&bpsk, config);
        data_samps.append(&mut symbol.samps_with_cyclic_prefix());
    }

    // Repeat data for redundancy
    let redundancy = config.pkt.num_bits / config.pkt.num_data_bits;
    for _ in 0..redundancy {
        res.extend(&data_samps);
    }

    res
}

trait InputSampleStream: Iterator<Item = Complex<f32>> {}

/// Periodically transmits packets to test the system
struct TestTx {
    /// Packet we are currently transmitting. If None, we are
    /// currently transmitting silence
    cur_pkt: Option<Vec<Complex<f32>>>,
    /// Position in whatever we are currently transmitting (either
    /// cur_pkt or silence)
    pos: usize,
    /// The packet we'll transmit repeatedly
    pkt: Vec<Complex<f32>>,
    /// Number of more packets to transmit
    num_pkts_left: usize,
    /// Number of samples of silence between transmissions
    silence_samps: usize,
}

impl TestTx {
    /// A test transmitter that will transmit `num_pkts` separated by
    /// `silence_samps` number of samples of silence
    pub fn new(silence_samps: usize, num_pkts: usize, config: &Config) -> Self {
        // Construct a random packet for us to transmit repeatedly
        let mut rng = rand_pcg::Pcg32::new(0, 0);
        let pkt_data: Vec<bool> = (0..config.pkt.num_data_bits).map(|_| rng.gen()).collect();
        let pkt = construct_pkt(&pkt_data, config);

        Self {
            cur_pkt: None,
            pos: 0,
            pkt,
            num_pkts_left: num_pkts,
            silence_samps,
        }
    }
}

impl Iterator for TestTx {
    type Item = Complex<f32>;
    fn next(&mut self) -> Option<Complex<f32>> {
        if self.num_pkts_left == 0 {
            return None;
        }

        if let Some(ref cur_pkt) = self.cur_pkt.as_ref() {
            // Transmit self.cur_pkt
            let res = Some(cur_pkt[self.pos]);
            if self.pos >= cur_pkt.len() - 1 {
                self.cur_pkt = None;
                self.pos = 0;
                self.num_pkts_left -= 1;
            } else {
                self.pos += 1
            }
            res
        } else {
            // If silence period has ended, transmit the next packet
            if self.pos >= self.silence_samps {
                self.cur_pkt = Some(self.pkt.clone());
                self.pos = 0;
            } else {
                self.pos += 1;
            }
            // Transmit silence
            Some(Complex::zero())
        }
    }
}

/// A wrapper around a transmitter that adds some noise etc. This
/// implements `InputSampleStream`, and hence can be given to the
/// receiver
pub struct TestLoopback<T: Iterator<Item = Complex<f32>>> {
    tx: T,
}

impl<T: Iterator<Item = Complex<f32>>> TestLoopback<T> {
    pub fn new(tx: T) -> Self {
        Self { tx }
    }
}

impl<T> Iterator for TestLoopback<T>
where
    T: Iterator<Item = Complex<f32>>,
{
    type Item = Complex<f32>;
    fn next(&mut self) -> Option<Complex<f32>> {
        self.tx.next()
    }
}

impl<T> InputSampleStream for TestLoopback<T> where T: Iterator<Item = Complex<f32>> {}

fn rx_loop<I: InputSampleStream>(inp: &mut I, config: &Config) {
    let mut pkt_detector = PktDetector::new(config);
    while let Some(sample) = inp.next() {
        // Do packet detection
        if !pkt_detector.push(sample) {
            continue;
        }

        // Collect enough samples that they definitely contain the
        // packet (if what we detected is indeed a packet)
        let max_samples = config.samps_per_pkt() + config.pkt_detect.num_samps;
        let mut pkt_samps = Vec::with_capacity(max_samples);
        let mut inp_done = false;
        for _ in 0..max_samples {
            if let Some(sample) = inp.next() {
                pkt_samps.push(sample);
            } else {
                inp_done = true;
                break;
            }
        }
        if inp_done {
            break;
        }

        // Now find exactly where the packet started
        let pkt_start_index = pkt_detector.pkt_start_index(&pkt_samps);
        // Remove off the stuff before the packet starts
        let pkt_samps = pkt_samps.split_off(pkt_start_index);

        println!("{:?}", pkt_samps);
        println!("Start index: {}", pkt_start_index);

        // Try to decode this packet
        break;
    }
}

fn main() {
    let config = Config::default();
    let tx = TestTx::new(1024, 4, &config);
    let mut lo = TestLoopback::new(tx);
    rx_loop(&mut lo, &config);
}
