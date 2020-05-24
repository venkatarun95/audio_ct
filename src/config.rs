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

