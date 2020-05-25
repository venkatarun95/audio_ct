use crate::config::Config;
use crate::packet::construct_pkt;

use num::{Complex, Zero};
use rand::Rng;

pub trait InputSampleStream: Iterator<Item = Complex<f32>> {}

/// Periodically transmits packets to test the system
pub struct TestTx {
    /// Packet we are currently transmitting. If None, we are
    /// currently transmitting silence
    cur_pkt: Option<Vec<Complex<f32>>>,
    /// Position in whatever we are currently transmitting (either
    /// cur_pkt or silence)
    pos: usize,
    /// The packet we'll transmit repeatedly
    pkt: Vec<Complex<f32>>,
    /// Number of more packets to transmit
    num_pkts_left: Option<usize>,
    /// Number of samples of silence between transmissions
    silence_samps: usize,
}

impl TestTx {
    /// A test transmitter that will transmit `num_pkts` separated by
    /// `silence_samps` number of samples of silence
    pub fn new(silence_samps: usize, num_pkts: Option<usize>, config: &Config) -> Self {
        // Construct a random packet for us to transmit repeatedly
        let mut rng = rand_pcg::Pcg32::new(0, 0);
        let pkt_data: Vec<bool> = (0..config.pkt.num_data_bits).map(|_| rng.gen()).collect();
        println!(
            "Txed:\n{:?}",
            pkt_data.iter().map(|x| *x as u8).collect::<Vec<_>>()
        );
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
        if Some(0) == self.num_pkts_left {
            return None;
        }

        if let Some(ref cur_pkt) = self.cur_pkt.as_ref() {
            // Transmit self.cur_pkt
            let res = Some(cur_pkt[self.pos]);
            if self.pos >= cur_pkt.len() - 1 {
                self.cur_pkt = None;
                self.pos = 0;
		if let Some(n) = self.num_pkts_left.as_mut() {
                    *n -= 1;
		}
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
