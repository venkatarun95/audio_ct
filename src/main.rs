use num::{Complex, Zero};
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
    /// Threshold in terms of the ratio of the two windows. Warning:
    /// Better to do it in terms of mean + thresh * std_dev? If window
    /// size is large enough, then this is unlikely to be broken by
    /// chance
    pub thresh: f32,
}

pub struct Config {
    pub ofdm: OfdmConfig,
    pub pkt_detect: PktDetectConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            ofdm: OfdmConfig {
                num_channels: 128,
                cyclic_prefix_frac: 4,
            },
            pkt_detect: PktDetectConfig {
                num_samps: 128,
                thresh: 2.,
            },
        }
    }
}

/// A moving window of the sum of values inside. Recalculates the
/// value to maintain precision
pub struct WindowSum {
    /// Number of values in the window
    win_size: usize,
    /// The values currently in the window
    vals: VecDeque<f32>,
    /// Current sum of values in the window
    sum: f32,
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
            samps_since_last: 0,
        }
    }

    /// Push a new value into the window. If window already has
    /// win_size values, returns the outgoing value
    fn push(&mut self, v: f32) -> Option<f32> {
        // Add in the new value
        self.sum += v;
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
            self.samps_since_last = 0;
        }

        res
    }

    fn sum(&self) -> f32 {
        self.sum
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
pub struct PktDetect<'c> {
    config: &'c Config,
    /// Two sliding windows to do thresholding
    windows: (WindowSum, WindowSum),
}

impl<'c> PktDetect<'c> {
    pub fn new(config: &'c Config) -> Self {
        assert_eq!(config.pkt_detect.num_samps % 2, 0);
        let n = config.pkt_detect.num_samps / 2;
        Self {
            config,
            windows: (WindowSum::new(n), WindowSum::new(n)),
        }
    }

    pub fn push(&mut self, samp: Complex<f32>) -> bool {
        // Put value in window
        if let Some(samp) = self.windows.1.push(samp.norm_sqr()) {
            if let Some(_) = self.windows.0.push(samp) {
                // We need enough seed samples before we can start
                // detecting packets
                return false;
            }
        }

        // Has the indicator met the threshold?
        let indicator = self.windows.1.sum() / self.windows.0.sum();
        indicator > self.config.pkt_detect.thresh
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
    pub fn from_symbols(symbols: &[Complex<f32>], config: &'a Config) -> Self {
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
    pub fn samples_with_cyclic_prefix(&self) -> Vec<Complex<f32>> {
        assert_eq!(self.samples.len(), self.config.ofdm.num_channels);
        let mut res = Vec::new();
        let num_channels = self.config.ofdm.num_channels;
        let prefix_len = num_channels / self.config.ofdm.cyclic_prefix_frac;
        res.extend(&self.samples[num_channels - prefix_len..]);
        res.extend(&self.samples);

        res
    }
}

impl<'a> RxOfdmSymbol<'a> {
    /// Decode symbol from samples
    pub fn from_samples(samples: &[Complex<f32>], config: &'a Config) -> Self {
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

trait InputSampleStream: Iterator<Item = Complex<f32>> {}

fn rx_loop<I: InputSampleStream>(inp: I) {
    for sample in inp {
        // Do packet detection
    }
}

fn main() {
    println!("Hello, world!");
}
