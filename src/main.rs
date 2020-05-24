mod pkt_detect;
mod config;

use pkt_detect::PktDetector;
use config::*;

use num::{Complex, Zero};
use rand::Rng;
use rustfft::FFTplanner;

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
    rng: rand_pcg::Pcg32,
}

impl<T: Iterator<Item = Complex<f32>>> TestLoopback<T> {
    pub fn new(tx: T) -> Self {
        Self {
            tx,
            rng: rand_pcg::Pcg32::new(1, 1),
        }
    }
}

impl<T> Iterator for TestLoopback<T>
where
    T: Iterator<Item = Complex<f32>>,
{
    type Item = Complex<f32>;
    fn next(&mut self) -> Option<Complex<f32>> {
        if let Some(val) = self.tx.next() {
            Some(val + 0.1 * Complex::<f32>::new(self.rng.gen(), self.rng.gen()))
        } else {
            None
        }
    }
}

impl<T> InputSampleStream for TestLoopback<T> where T: Iterator<Item = Complex<f32>> {}

fn rx_loop<I: InputSampleStream>(inp: &mut I, config: &Config) {
    let mut pkt_detector = PktDetector::new(config);
    while let Some(sample) = inp.next() {
        // Do packet detection
        let pkt_samps = if let Some(pkt_samps) = pkt_detector.push(sample) {
            pkt_samps
        } else {
            continue;
        };

        println!("Packet detected");

        // Try to decode this packet
    }
}

fn main() {
    let config = Config::default();
    let tx = TestTx::new(1024, 4, &config);
    let mut lo = TestLoopback::new(tx);
    rx_loop(&mut lo, &config);
}
