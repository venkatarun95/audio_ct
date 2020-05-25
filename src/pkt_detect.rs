use crate::config::Config;

use num::Complex;
use std::collections::VecDeque;

/// A moving window of the sum and std. dev. of the norm of the values
/// inside. Recalculates the value to maintain precision
pub struct WindowSum {
    /// Number of values in the window
    win_size: usize,
    /// The values currently in the window
    vals: VecDeque<Complex<f32>>,
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
    fn push(&mut self, v: Complex<f32>) -> Option<Complex<f32>> {
        // Add in the new value
        self.sum += v.norm();
        self.sum_sq += v.norm_sqr();
        self.vals.push_back(v);
        self.samps_since_last += 1;

        assert!(self.vals.len() <= self.win_size + 1);
        // Pop if already full
        let res = if self.vals.len() == self.win_size + 1 {
            let vo = self.vals.pop_front().unwrap();
            self.sum -= vo.norm();
            self.sum_sq -= vo.norm_sqr();
            Some(vo)
        } else {
            None
        };

        // Recompute if necessary
        if self.samps_since_last > 65535 {
            self.sum = self.vals.iter().map(|v| v.norm()).sum();
            self.sum_sq = self.vals.iter().map(|v| v.norm_sqr()).sum();
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

    #[allow(dead_code)]
    fn sum_sq(&self) -> f32 {
        self.sum_sq
    }

    #[allow(dead_code)]
    fn var(&self) -> f32 {
        if self.vals.len() == 0 {
            0.
        } else {
            self.sum_sq / self.vals.len() as f32 - self.avg() * self.avg()
        }
    }

    /// Give access to the underlying `VecDeqeue`
    fn deque(&self) -> &VecDeque<Complex<f32>> {
        &self.vals
    }
}

/// State machine state for the PktDetector
enum PktDetectorState {
    Silent,
    /// We are currently recording samples for a packet
    PktInProgress {
        /// We'll add samples to this one-by-one as they are pushed
        pkt_samps: Vec<Complex<f32>>,
        /// Position in `pkt_samps` that is currently our best
        /// candidate for the start of the packet
        start_pos: usize,
        /// Position (in `pkt_samps`) of the last indicator calculated
        cur_pos: usize,
        /// Maximum eligible indicator value for this packet
        max_indicator: f32,
    },
}

/// Takes a constant stream of packets and tells us exactly where a
/// packet began
pub struct PktDetector<'c> {
    config: &'c Config,
    /// Two sliding windows to do thresholding
    windows: (WindowSum, WindowSum),
    /// Total number of samples seen so far
    _samp_id: usize,
    /// State of the state machine for packet detector
    state: PktDetectorState,
}

impl<'c> PktDetector<'c> {
    pub fn new(config: &'c Config) -> Self {
        assert_eq!(config.pkt_detect.num_samps % 2, 0);
        let n = config.pkt_detect.num_samps / 2;
        Self {
            config,
            windows: (WindowSum::new(n), WindowSum::new(n)),
            _samp_id: 0,
            state: PktDetectorState::Silent,
        }
    }

    /// Indicator which we use to decide if a packet has started
    fn indicator(&self) -> f32 {
        // Computing the standard deviation from both windows keeps
        // things smooth er than if we only include variance from
        // windows.0, and makes sure tests pass
        let std_dev = (0.5 * (self.windows.0.var() + self.windows.1.var())).sqrt();
	//println!("{} {} {} {}", self._samp_id, std_dev, self.windows.1.avg(), self.windows.0.avg());

        if std_dev == 0. {
            if self.windows.0.avg() != self.windows.1.avg() {
                // Super unlikely event: both windows were perfectly
                // constant, except for a jump. If there is an edge,
                // this is definitely it
                std::f32::MAX
            } else {
                // Will be taken on almost every real case
                0.
            }
        } else {
            (self.windows.1.avg() - self.windows.0.avg()) / std_dev
        }
    }

    /// Push a sample into the windows. Return true if we have enough
    /// samples to start detecting packets
    fn push_samp(&mut self, samp: Complex<f32>) -> bool {
        self._samp_id += 1;
        // Put value in window
        if let Some(samp) = self.windows.1.push(samp) {
            if let Some(_) = self.windows.0.push(samp) {
                // We need enough seed samples before we can start
                // detecting packets
                return true;
            }
        }
        return false;
    }

    /// Push all samples one-by-one. Once it thinks it has received
    /// all samples from a packet, returns a vec containing all the
    /// samples
    pub fn push(&mut self, samp: Complex<f32>) -> Option<Vec<Complex<f32>>> {
        if !self.push_samp(samp) {
            return None;
        }
        // The maximum number of samples per packet, including some
        // slack for detecting the packet
        let max_samples = self.config.samps_per_pkt() + self.config.pkt_detect.num_samps;
        let indicator = self.indicator();
        match self.state {
            PktDetectorState::Silent => {
		//println!("{} Silent: {} {}", self._samp_id, indicator, samp);
                // if the indicator has met the threshold, change state
                if indicator >= self.config.pkt_detect.thresh {
                    let mut pkt_samps = Vec::with_capacity(max_samples);
                    let (s1, s2) = self.windows.1.deque().as_slices();
                    pkt_samps.extend(s1);
                    pkt_samps.extend(s2);
                    assert!(pkt_samps.len() <= max_samples);
                    // The current `indicator` value corresponds to
                    // the 0th sample here
                    self.state = PktDetectorState::PktInProgress {
                        pkt_samps,
                        start_pos: 0,
                        cur_pos: 0,
                        max_indicator: indicator,
                    };
                }
                None
            }
            PktDetectorState::PktInProgress {
                ref mut pkt_samps,
                ref mut start_pos,
                ref mut cur_pos,
                ref mut max_indicator,
            } => {
		//println!("{} InProgress: {} {}", self._samp_id, indicator, samp);
                if pkt_samps.len() < max_samples {
                    pkt_samps.push(samp);
                    *cur_pos += 1;
                    // See if we need up update the indicator. We only
                    // check for the first few samples, enough to
                    // cover a window
                    if pkt_samps.len() <= self.config.pkt_detect.num_samps
                        && indicator > *max_indicator
                    {
                        *max_indicator = indicator;
                        *start_pos = *cur_pos;
                    }
                    None
                } else {
                    // Remove the portion of `pkt_samples` before pkt starts
                    let mut res = pkt_samps.split_off(*start_pos).clone();
                    res.truncate(self.config.samps_per_pkt());
                    self.state = PktDetectorState::Silent;
                    Some(res)
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::{PktDetector, WindowSum};
    use crate::config::Config;

    use num::Complex;
    use rand::Rng;
    use std::default::Default;

    #[test]
    fn window_sum() {
        fn cplx(i: usize) -> Complex<f32> {
            Complex::new(i as f32, 0.)
        }

        let l = 10;
        let mut win = WindowSum::new(l);
        // Add numbers 0..100
        for i in 0..100 {
            win.push(cplx(i));
            if i > l {
                let correct = i * (i + 1) / 2 - (i - l) * (i - l + 1) / 2;
                assert_eq!((win.avg() * l as f32).round() as usize, correct);
            }
        }

        // Add alternating 0s and 1s
        for alt in 5..30 {
            let mut win = WindowSum::new(l);
            let mut correct = 0;
            for i in 0..100 {
                let val = (i / alt) % 2;
                win.push(cplx(val));
                correct += val;
                if i >= l {
                    correct -= ((i - l) / alt) % 2;
                }
                assert_eq!(
                    (win.avg() * win.deque().len() as f32).round() as usize,
                    correct
                );
            }
        }
    }

    /// Takes a model of variation and tests
    fn pkt_detect_base(variation: &mut dyn FnMut(usize) -> Complex<f32>) {
        let config = Config::default();
        let mut detector = PktDetector::new(&config);
        let n = config.samps_per_pkt();
        let mut num_pkts_detected = 0;
        for i in 0..n * 9 {
            let samp = if i % (2 * n) == n {
                Complex::new(2., 0.)
            } else if (i / n) % 2 == 1 {
                Complex::new(0., 1.) + variation(i)
            } else {
                Complex::new(0., 0.) + variation(i)
            };

            if let Some(pkt_samps) = detector.push(samp) {
                num_pkts_detected += 1;
                assert_eq!(pkt_samps.len(), config.samps_per_pkt());
                assert_eq!(pkt_samps[0], Complex::new(2., 0.));
                for x in &pkt_samps[1..] {
                    assert!((x - Complex::new(0., 1.)).norm() <= 0.11);
                }
            }
        }
        assert_eq!(num_pkts_detected, 4);
    }

    #[test]
    fn pkt_detect_const() {
        let mut variation = |_| Complex::new(0., 0.);
        pkt_detect_base(&mut variation);
        let mut variation = |_| Complex::new(0.05, 0.05);
        pkt_detect_base(&mut variation);
    }

    #[test]
    fn pkt_detect_variation_deterministic() {
        // Deterministic
        let mut variation = |i: usize| -> Complex<f32> {
            if i % 2 == 1 {
                Complex::new(0.1, 0.)
            } else {
                Complex::new(0., 0.)
            }
        };
        pkt_detect_base(&mut variation);
    }

    #[test]
    fn pkt_detect_variation_random() {
        // Random
        let mut rng = rand_pcg::Pcg32::new(1, 1);
        let mut variation = |_| -> Complex<f32> { 0.01f32 * Complex::new(rng.gen(), rng.gen()) };
        pkt_detect_base(&mut variation);
    }
}
