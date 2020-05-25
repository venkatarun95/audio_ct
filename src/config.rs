use num::Complex;
use rand::Rng;
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
    pub num_data_bits: usize,
    /// Number of actual bits, including extra bits for ECC
    pub num_bits: usize,
    /// The pilot symbol (exactly OfdmConfig::num_channels long)
    pub pilot: Vec<Complex<f32>>,
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
        self.ofdm.symbol_len() * (self.pkt.num_bits / self.ofdm.num_channels + 1)
    }

    /// Check whether the configuration parameters make sense. Any
    /// constraints on the parameter should be added to this
    /// function. Only works in debug mode
    pub fn make_sense(&self) {
	assert!(self.pkt_detect.thresh >= 0.);
	// Everything can't be cyclic prefix
	assert!(self.ofdm.cyclic_prefix_frac > 1);
	// Total number of bits should fit evenly into the symbols
	assert_eq!(self.pkt.num_bits % self.ofdm.num_channels, 0);
	// Note: num_bits has more for FEC
	assert!(self.pkt.num_data_bits <= self.pkt.num_bits);
	// Pilot is just an OFDM symbol
	assert_eq!(self.pkt.pilot.len(), self.ofdm.num_channels);
	// So that packet detection is always based on the predictable
	// behavior of the preamble and not on some random OFDM symbol
	assert!(self.pkt_detect.num_samps <= self.ofdm.num_channels);
    }
}

impl Default for Config {
    fn default() -> Self {
        let num_channels = 128;
        // Let pilot be a (deterministic) random number
        let mut rng = rand_pcg::Pcg32::new(0, 0);
        let pilot_fft = (0..num_channels)
            .map(|_| Complex::<f32>::new(rng.gen(), rng.gen()))
            //.map(|_| Complex::new(1., 2. * std::f32::consts::PI * rng.gen::<f32>()))
            //.map(|_| Complex::new(1f32, 0f32))
            .collect::<Vec<_>>();

        // Perform IFFT
        let mut planner = rustfft::FFTplanner::new(true);
        let fft = planner.plan_fft(num_channels);
        let mut scratch_inp: Vec<_> = pilot_fft.clone();
        let mut pilot = vec![Complex::new(0., 0.); num_channels];
        fft.process(&mut scratch_inp, &mut pilot);

        Self {
            ofdm: OfdmConfig {
                num_channels,
                cyclic_prefix_frac: 4,
            },
            pkt_detect: PktDetectConfig {
                num_samps: 128,
                thresh: 3.,
            },
            pkt: PktConfig {
                num_data_bits: 256,
                num_bits: 512,
                pilot,
            },
        }
    }
}
