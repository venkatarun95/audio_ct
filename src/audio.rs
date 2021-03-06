use crate::config::Config;
use crate::test::InputSampleStream;

use failure::Error;
use num::{Complex, Zero};
use portaudio as pa;
use std::sync::mpsc::{Receiver, Sender};

const CHANNELS: i32 = 1;
const FRAMES: u32 = 256;
const INTERLEAVED: bool = true;

fn run<'c, T>(
    mut modulator: Modulate<'c, T>,
    rx_sender: Sender<f32>,
    sample_rate: f32,
) -> Result<(), pa::Error>
where
    T: Iterator<Item = Complex<f32>>,
{
    let pa = pa::PortAudio::new()?;

    println!("PortAudio");
    println!("version: {}", pa.version());
    println!("version text: {:?}", pa.version_text());
    println!("host count: {}", pa.host_api_count()?);

    let default_host = pa.default_host_api()?;
    println!("default host: {:#?}", pa.host_api_info(default_host));

    let def_input = pa.default_input_device()?;
    let input_info = pa.device_info(def_input)?;
    println!("Default input device info: {:#?}", &input_info);

    // Construct the input stream parameters.
    let latency = input_info.default_low_input_latency;
    let input_params = pa::StreamParameters::<f32>::new(def_input, CHANNELS, INTERLEAVED, latency);

    let def_output = pa.default_output_device()?;
    let output_info = pa.device_info(def_output)?;
    println!("Default output device info: {:#?}", &output_info);

    // Construct the output stream parameters.
    let latency = output_info.default_low_output_latency;
    let output_params =
        pa::StreamParameters::<f32>::new(def_output, CHANNELS, INTERLEAVED, latency);

    // Check that the stream format is supported.
    pa.is_duplex_format_supported(input_params, output_params, sample_rate as f64)?;

    // Construct the settings with which we'll open our duplex stream.
    let settings =
        pa::DuplexStreamSettings::new(input_params, output_params, sample_rate as f64, FRAMES);

    let mut stream = pa.open_blocking_stream(settings)?;

    stream.start()?;

    // We'll use this function to wait for read/write availability.
    fn wait_for_stream<F>(f: F, name: &str) -> u32
    where
        F: Fn() -> Result<pa::StreamAvailable, pa::error::Error>,
    {
        loop {
            match f() {
                Ok(available) => match available {
                    pa::StreamAvailable::Frames(frames) => return frames as u32,
                    pa::StreamAvailable::InputOverflowed => println!("Input stream has overflowed"),
                    pa::StreamAvailable::OutputUnderflowed => {
                        println!("Output stream has underflowed")
                    }
                },
                Err(err) => panic!(
                    "An error occurred while waiting for the {} stream: {}",
                    name, err
                ),
            }
        }
    };

    // Now start the main read/write loop! In this example, we pass
    // the input buffer directly to the output buffer, so watch out
    // for feedback.
    loop {
        // How many frames are available on the input stream?
        let in_frames = wait_for_stream(|| stream.read_available(), "Read");

        // If there are frames available, let's take them and add them
        // to our buffer.
        if in_frames > 0 {
            let input_samples = stream.read(in_frames)?;
            for samp in input_samples {
                rx_sender.send(*samp).unwrap();
            }
        }

        // How many frames are available for writing on the output stream?
        let out_frames = wait_for_stream(|| stream.write_available(), "Write");

        // If there are frames available for writing and we have some
        // to write, then write!
        if out_frames > 0 {
            // If we have more than enough frames for writing, take
            // them from the start of the buffer. Otherwise if we
            // have less, just take what we can for now.
            let write_frames = out_frames;
            let n_write_samples = write_frames as usize * CHANNELS as usize;
	    let mut flag = false;
            stream.write(write_frames, |output| {
                for i in 0..n_write_samples {
                    if let Some(samp) = modulator.next() {
                        output[i] = samp;
                    } else {
                        println!("Tx samples finished. Exiting");
			flag = true;
                        break;
                    }
                }
            })?;
	    if flag {
		break;
	    }
        }
    }
    Ok(())
}

pub fn start_audio<'c>(
    tx_receiver: Receiver<Complex<f32>>,
    config: &'c Config,
) -> Result<(std::thread::JoinHandle<()>, AudioSampleStream<'c>), Error> {
    // For the microphone
    let (rx_sender, rx_receiver) = std::sync::mpsc::channel::<f32>();
    let sample_rate = config.audio.sample_rate;
    let config_c = config.clone();

    let handle = std::thread::spawn(move || {
        // Use the modulator to go from baseband to carrier frequency
        let modulator = Modulate::new(tx_receiver.iter(), sample_rate, &config_c);
        run(modulator, rx_sender, sample_rate).unwrap();
    });

    return Ok((
        handle,
        AudioSampleStream::new(rx_receiver, sample_rate as f32, config),
    ));
}

/// Stream of samples from an audio device
pub struct AudioSampleStream<'c> {
    channel: Receiver<f32>,
    demod: Demodulate<'c>,
}

impl<'c> AudioSampleStream<'c> {
    fn new(channel: Receiver<f32>, sample_rate: f32, config: &'c Config) -> Self {
        Self {
            channel,
            demod: Demodulate::new(sample_rate, config),
        }
    }
}

impl<'c> InputSampleStream for AudioSampleStream<'c> {}

impl<'c> Iterator for AudioSampleStream<'c> {
    type Item = Complex<f32>;

    fn next(&mut self) -> Option<Complex<f32>> {
        loop {
            let in_samp = if let Ok(samp) = self.channel.recv() {
                samp
            } else {
                return None;
            };
            let out = self.demod.push(in_samp);
            if out.is_some() {
                return out;
            }
        }
    }
}

/// Upconvert signal from baseband to carrier frequency
struct Modulate<'c, T>
where
    T: Iterator<Item = Complex<f32>>,
{
    config: &'c Config,
    /// Our source of baseband samples
    src: T,
    /// Number of carrier samples to skip per baseband sample
    to_skip: u64,
    /// Sample rate of the audio signal
    sample_rate: f32,
    /// Total number of (audio) samples transmitted so far
    num_samps: u64,
    /// If true, input is done and we'll always return None
    done: bool,
    /// The two samples we are currently in between sending
    cur_samps: (Complex<f32>, Complex<f32>),
    /// The current sample we are sending
    cur_ewma: Complex<f32>,
}

impl<'c, T> Modulate<'c, T>
where
    T: Iterator<Item = Complex<f32>>,
{
    fn new(src: T, sample_rate: f32, config: &'c Config) -> Self {
        // For now, sample_rate has to be a multiple of bandwidth. Can
        // remove restriction later
        let to_skip = sample_rate / config.audio.bandwidth;
        assert!((to_skip.round() - to_skip).abs() <= 1e-3);
        let to_skip = 1; //to_skip.round() as u64;
        Self {
            config,
            src,
            to_skip,
            sample_rate,
            num_samps: 0,
            done: false,
            cur_samps: (Complex::zero(), Complex::zero()),
            cur_ewma: Complex::zero(),
        }
    }
}

impl<'c, T> Iterator for Modulate<'c, T>
where
    T: Iterator<Item = Complex<f32>>,
{
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if self.done {
            return None;
        }

        // See if we need to update the current sample
        if self.num_samps % self.to_skip == 0 {
            if let Some(x) = self.src.next() {
                self.cur_samps = (self.cur_samps.1, x);
            } else {
		self.done = true;
		return None;
	    }
        }

        let pi2 = 2. * std::f32::consts::PI;
        let e = Complex::new(
            0.,
            pi2 * self.num_samps as f32 / self.sample_rate * self.config.audio.center_frequency,
        )
        .exp();
        self.num_samps += 1;

        // Low-pass filter

	// Frequency-domain sinc
        //let f = (self.num_samps % self.to_skip) as f32 / self.to_skip as f32;
        //let samp = prev * (1. - f) + cur * f;

	// EWMA (equivalent to RC-filter)
	// let alpha = 1. / (1. + self.to_skip as f32);
	// self.cur_ewma = alpha * self.cur_samps.1 + (1. - alpha) * self.cur_ewma;
	// let samp = self.cur_ewma;

	let samp = self.cur_samps.1;
        Some(e.re * samp.re + e.im * samp.im)
    }
}

/// Convert to baseband from carrier frequency
struct Demodulate<'c> {
    config: &'c Config,
    /// Number of carrier samples to skip per baseband sample
    to_skip: usize,
    /// Sample rate of the audio signal
    sample_rate: f32,
    /// Total number of (audio) samples transmitted so far
    num_samps: u64,
    /// Average of the sample so far
    samp_avg: Complex<f32>,
}

impl<'c> Demodulate<'c> {
    fn new(sample_rate: f32, config: &'c Config) -> Self {
        // For now, sample_rate has to be a multiple of bandwidth. Can
        // remove restriction later
        let to_skip = sample_rate / config.audio.bandwidth;
        assert!((to_skip.round() - to_skip).abs() <= 1e-3);
        let to_skip = to_skip.round() as usize;
        Self {
            config,
            to_skip,
            sample_rate,
            num_samps: 0,
            samp_avg: Complex::zero(),
        }
    }

    /// Takes an audio sample, and if appropriate, returns a baseband
    /// sample
    fn push(&mut self, samp: f32) -> Option<Complex<f32>> {
        // Add to the average
        let pi2 = 2. * std::f32::consts::PI;
        let e = Complex::new(
            0.,
            pi2 * self.num_samps as f32 / self.sample_rate * self.config.audio.center_frequency,
        )
        .exp();
        self.samp_avg += samp * e;

        // Should we output something?
        let res = if self.num_samps % self.to_skip as u64 == 0 {
            let res = Some(self.samp_avg / self.to_skip as f32);
            self.samp_avg = Complex::zero();
            res
        } else {
            None
        };
        self.num_samps += 1;
        res
    }
}

#[cfg(test)]
mod tests {
    use super::{Demodulate, Modulate};
    use crate::config::Config;

    use num::Complex;
    use rand::Rng;

    #[test]
    fn mod_demod() {
        let config = Config::default();
        let sample_rate = 44100.;

        // The samples we'll transmit
        let mut rng = rand_pcg::Pcg32::new(1, 1);
        let samples: Vec<_> = (0..100_000)
            .map(|_| Complex::new(rng.gen(), rng.gen()))
            .collect();

        let modulate = Modulate::new(samples.clone().into_iter(), sample_rate, &config);
        let mut demodulate = Demodulate::new(sample_rate, &config);

        let mut pos = 0;
        let mut channel = None;
        for x in modulate {
            if let Some(out) = demodulate.push(x) {
                if channel.is_none() {
                    channel = Some(out / samples[pos]);
                }
                let channel = channel.unwrap();

                //println!("{:?} {:?}", out.to_polar(), samples[pos].to_polar());
                //println!("{} {}", out / samples[pos], channel);
                assert!((out / samples[pos] - channel).norm() <= 1e-3);
                pos += 1;
            }
        }
    }
}
