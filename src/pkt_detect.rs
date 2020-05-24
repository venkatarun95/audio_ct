use crate::config::Config;

use std::collections::VecDeque;
use num::Complex;

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

	assert!(self.vals.len() <= self.win_size);
	if self.vals.len() == self.win_size {
            if let Some(vo) = self.vals.front() {
		self.sum -= vo.norm();
		self.sum_sq -= vo.norm_sqr();
            }
	}

        // Pop if already full
        let res = if self.vals.len() == self.win_size {
            Some(self.vals.pop_front().unwrap())
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

    fn std_dev(&self) -> f32 {
        if self.vals.len() == 0 {
            0.
        } else {
            (self.sum_sq / self.vals.len() as f32 - self.avg() * self.avg()).sqrt()
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
            state: PktDetectorState::Silent,
        }
    }

    /// Indicator which we use to decide if a packet has started
    fn indicator(&self) -> f32 {
        if self.windows.0.std_dev() == 0. {
            if self.windows.1.avg() > 0. {
                // The minimum value that'll pass the threshold
                self.config.pkt_detect.thresh
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
                // if the indicator has met the threshold, change state
                if indicator >= self.config.pkt_detect.thresh {
                    let mut pkt_samps = Vec::with_capacity(max_samples);
                    let (s1, s2) = self.windows.1.deque().as_slices();
                    pkt_samps.push(*self.windows.0.deque().back().unwrap());
                    pkt_samps.extend(s1);
                    pkt_samps.extend(s2);
                    assert!(pkt_samps.len() <= max_samples);
                    // The current `indicator` value corresponds to the 0th sample here
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
                    let res = pkt_samps.split_off(*start_pos).clone();
                    self.state = PktDetectorState::Silent;
                    Some(res)
                }
            }
        }
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

