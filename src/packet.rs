use crate::config::Config;
use crate::pkt_detect::PktDetector;
use crate::test::InputSampleStream;

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

        let norm = (fft_out.len() as f32).sqrt();
        for x in &mut fft_out {
            *x /= norm;
        }

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
        res.extend(&self.samples[num_channels - prefix_len..]);
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

        let norm = (fft_out.len() as f32).sqrt();
        for x in &mut fft_out {
            *x /= norm;
        }

        Self {
            config,
            symbols: fft_out,
        }
    }

    /// Divide out the entire symbol by the given array
    pub fn normalize(&mut self, denom: &[Complex<f32>]) {
        assert_eq!(denom.len(), self.config.ofdm.num_channels);

        for (s, d) in self.symbols.iter_mut().zip(denom) {
            *s /= d;
        }
    }

    /// Should be called after normalization to decode the bits
    pub fn decode_bpsk(&self) -> Vec<bool> {
        self.symbols.iter().map(|x| x.re > 0.).collect()
    }

    pub fn symbols(&self) -> &[Complex<f32>] {
        &self.symbols
    }
}

mod tests {
    use super::{RxOfdmSymbol, TxOfdmSymbol};
    use crate::config::Config;

    use num::Complex;
    use rand::Rng;

    #[test]
    /// Encode and decode a symbol using OFDM and ensure the result
    /// is the same
    fn ofdm_encode_decode() {
        // Prepare the data
        let config = Config::default();
        let mut rng = rand_pcg::Pcg32::new(0, 0);
        let symbols = (0..config.ofdm.num_channels)
            .map(|_| Complex::<f32>::new(rng.gen(), rng.gen()))
            .collect::<Vec<_>>();

        // Simulate the tx and rx
        let txed = TxOfdmSymbol::new(&symbols, &config);
        let rxed = txed.samps_with_cyclic_prefix();
        let rx = RxOfdmSymbol::new(&rxed[config.ofdm.prefix_len() + 1..], &config);

        // See if they are equal
        for (s, r) in symbols.iter().zip(rx.symbols()) {
            assert!((s - r).norm() <= 1e-6);
        }
    }

    /// Encode a symbol, then decode with varying amounts of delay and
    /// see if it still works
    #[test]
    fn ofdm_encode_decode_delay() {
        // Prepare the data
        let config = Config::default();
        let mut rng = rand_pcg::Pcg32::new(0, 0);
        let symbols1 = (0..config.ofdm.num_channels)
            .map(|_| Complex::<f32>::new(rng.gen(), rng.gen()))
            .collect::<Vec<_>>();
        let symbols2 = (0..config.ofdm.num_channels)
            .map(|_| Complex::<f32>::new(rng.gen(), rng.gen()))
            .collect::<Vec<_>>();

        // Simulate the tx and rx
        let txed1 = TxOfdmSymbol::new(&symbols1, &config);
        let rxed1 = txed1.samps_with_cyclic_prefix();
        let txed2 = TxOfdmSymbol::new(&symbols2, &config);
        let rxed2 = txed2.samps_with_cyclic_prefix();

        // Add different amounts of delay
        for delay in 0..config.ofdm.prefix_len() {
            let rx1 = RxOfdmSymbol::new(&rxed1[delay..delay + config.ofdm.num_channels], &config);
            let rx2 = RxOfdmSymbol::new(&rxed2[delay..delay + config.ofdm.num_channels], &config);

            // There should be a constant channel difference for all the symbols. See if they are equal
            for i in 0..config.ofdm.num_channels {
                let (s1, s2) = (symbols1[i], symbols2[i]);
                let (r1, r2) = (rx1.symbols()[i], rx2.symbols()[i]);
                assert!((s1 / r1 - s2 / r2).norm() < 1e-4);
            }
        }
    }
}

pub fn encode_bpsk(data: &[bool]) -> Vec<Complex<f32>> {
    data.iter()
        .map(|b| {
            if *b {
                Complex::new(1., 0.)
            } else {
                Complex::new(-1., 0.)
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
        let bpsk = encode_bpsk(&data[start..start + config.ofdm.num_channels]);
        let symbol = TxOfdmSymbol::new(&bpsk, config);
	assert_eq!(data_samps.len(), symbol_id * config.ofdm.symbol_len());
        data_samps.append(&mut symbol.samps_with_cyclic_prefix());
    }

    // Repeat data for redundancy
    let redundancy = config.pkt.num_bits / config.pkt.num_data_bits;
    for _ in 0..redundancy {
        res.extend(&data_samps);
    }

    res
}

pub fn rx_loop<I: InputSampleStream>(inp: &mut I, config: &Config) {
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
        let prefix_len = config.ofdm.prefix_len();
        let symbol_len = config.ofdm.symbol_len();
        let num_channels = config.ofdm.num_channels;

        // First get the pilot, so we can equalize the rest of the
        // symbols relative to it
        let mut pilot = RxOfdmSymbol::new(&pkt_samps[prefix_len..symbol_len], config);
        pilot.normalize(&config.pkt.pilot);
        // println!(
        //     "Pilot {:.3?}",
        //     pilot
        //         .symbols()
        //         .iter()
        //         .map(|x| x.to_polar())
        //         .collect::<Vec<_>>()
        // );

        // Now get the rest of the OFDM symbols and decode them
        let mut decoded = Vec::with_capacity(config.pkt.num_bits);
        assert_eq!(config.pkt.num_bits % num_channels, 0);
        for symbol_id in 0..config.pkt.num_bits / num_channels {
            let pos = (symbol_id + 1) * symbol_len + prefix_len;
            let mut symbol = RxOfdmSymbol::new(&pkt_samps[pos..pos + num_channels], config);
            symbol.normalize(pilot.symbols());
            println!(
                "{:?}",
                symbol
                    .decode_bpsk()
                    .iter()
                    .map(|x| *x as u8)
                    .collect::<Vec<_>>()
            );
            decoded.extend(symbol.decode_bpsk());
        }

        // This is the packet we wanted to transmit
        let mut rng = rand_pcg::Pcg32::new(0, 0);
        let pkt_data: Vec<bool> = (0..config.pkt.num_data_bits).map(|_| rng.gen()).collect();
        // Check the error rate using this packet
        let num_wrong_bits: usize = pkt_data
            .iter()
            .zip(decoded)
            .map(|(b1, b2)| (*b1 != b2) as usize)
            .sum();
        println!(
            "Tot data bits: {}, erreanous bits: {}",
            config.pkt.num_data_bits, num_wrong_bits
        );
    }
}
