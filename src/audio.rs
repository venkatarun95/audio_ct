use crate::config::Config;
use crate::test::InputSampleStream;

use failure::Error;
use num::complex::Complex;
use portaudio as pa;
use std::sync::mpsc::Receiver;

pub struct Audio {
    _pa: pa::PortAudio,
}

impl Audio {
    pub fn new<'c>(
        tx: Receiver<Complex<f32>>,
        config: &'c Config,
    ) -> Result<(Self, AudioSampleStream<'c>), Error> {
        // Size of the buffer in which we'll send/receive data
        const FRAMES: u32 = 256;

        let pa = pa::PortAudio::new()?;

	println!("All devices:");
	for d in pa.devices()? {
	    println!("{:#?}", d);
	}

	// Let's use device 5 for now (Warning: not portable)
	let device = pa::DeviceIndex(5);
	
        // Construct the output stream settings
        let output_dev = device; //pa.default_output_device()?;
        let output_info = pa.device_info(output_dev)?;
        println!("Output Device: {:#?}", output_info);
        let latency = output_info.default_high_output_latency;
        let output_params = pa::StreamParameters::<f32>::new(output_dev, 1, false, latency);
        let sample_rate = output_info.default_sample_rate;

        // Construct the input stream settings
        let input_dev = device; //pa.default_input_device()?;
        let input_info = pa.device_info(input_dev)?;
        println!("Input Device: {:#?}", input_info);
        let latency = input_info.default_high_input_latency;
        let input_params = pa::StreamParameters::<f32>::new(input_dev, 1, false, latency);
        // Both input and sample rates should be equal
        assert_eq!(input_info.default_sample_rate, sample_rate);

        // Check whether duplex format is supported
        pa.is_duplex_format_supported(input_params, output_params, sample_rate)?;
        let duplex_settings =
            pa::DuplexStreamSettings::new(input_params, output_params, sample_rate, FRAMES);

        // For the microphone
        let (rx_sender, rx_receiver) = std::sync::mpsc::channel::<f32>();

        // Total number of samples sent so far
        let mut num_sent_samps = 0;
	let center_frequency = config.audio.center_frequency;

        let callback = move |pa::DuplexStreamCallbackArgs {
                                 in_buffer,
                                 out_buffer,
                                 frames,
                                 ..
                             }| {
            assert_eq!(frames, FRAMES as usize);

            for input_sample in in_buffer {
                rx_sender.send(*input_sample).unwrap();
            }
            for output_sample in out_buffer {
                let samp = if let Ok(samp) = tx.recv() {
                    samp
                } else {
                    return pa::Complete;
                };
                let pi2 = 2. * std::f32::consts::PI;
                let e = Complex::new(
                    0.,
                    pi2 * num_sent_samps as f32 / sample_rate as f32
                        * center_frequency,
                )
                    .exp();
		num_sent_samps += 1;
                *output_sample = samp.re * e.re + samp.im * e.im
            }

            pa::Continue
        };

        let mut stream = pa.open_non_blocking_stream(duplex_settings, callback)?;

	//stream.start()?;
	std::thread::spawn(move || {
	    stream.start().unwrap();
	    while stream.is_active().unwrap() {
		//std::thread::sleep(std::time::Duration::from_secs(1));
	    }
	});
	
        Ok((
            Self { _pa: pa },
            AudioSampleStream::new(rx_receiver, sample_rate as f32, config),
        ))
    }
}

/// Stream of samples from an audio device
pub struct AudioSampleStream<'c> {
    channel: Receiver<f32>,
    /// Number of samples per second that we'll receive from the microphone
    sample_rate: f32,
    /// Total number of baseband samples received so far
    samps_so_far: u64,
    config: &'c Config,
}

impl<'c> AudioSampleStream<'c> {
    fn new(channel: Receiver<f32>, sample_rate: f32, config: &'c Config) -> Self {
        Self {
            channel,
            sample_rate,
            samps_so_far: 0,
            config,
        }
    }
}

impl<'c> InputSampleStream for AudioSampleStream<'c> {}

impl<'c> Iterator for AudioSampleStream<'c> {
    type Item = Complex<f32>;

    fn next(&mut self) -> Option<Complex<f32>> {
        // Sample rate needs to be an integer multiple of bandwidth
        let samps_to_skip = self.sample_rate / self.config.audio.bandwidth;
        assert!((samps_to_skip.round() - samps_to_skip) < 1e-3);
        let samps_to_skip = samps_to_skip.round() as usize;

        let mut avg = Complex::new(0., 0.);
        for _ in 0..samps_to_skip {
            let samp = if let Ok(samp) = self.channel.recv() {
                samp
            } else {
                return None;
            };
            let pi2 = 2. * std::f32::consts::PI;
            avg += samp
                * Complex::new(
                    0.,
                    pi2 * self.config.audio.center_frequency *
			self.samps_so_far as f32 / self.sample_rate,
                )
                .exp();
        }
        avg /= samps_to_skip as f32;
        Some(avg)
    }
}
