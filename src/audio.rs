use crate::config::Config;
use crate::test::InputSampleStream;

use failure::Error;
use num::complex::Complex;
use portaudio as pa;
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender};

const SAMPLE_RATE: f64 = 44_100.0;
const CHANNELS: i32 = 1;
const FRAMES: u32 = 256;
const INTERLEAVED: bool = true;

fn run(
    tx_receiver: Receiver<Complex<f32>>,
    rx_sender: Sender<f32>,
    sample_rate: f32,
    center_frequency: f32,
    bandwidth: f32,
) -> Result<(), pa::Error> {
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

    // The sample we are currently sending through the speaker
    let mut tx_samp = Complex::new(0., 0.);
    // Total number of samples sent so far
    let mut num_sent_samples = 0u64;

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
	    let mut avg = 0.;
	    for samp in input_samples {
		rx_sender.send(*samp).unwrap();
		avg += samp.abs();
	    }
	    //println!("{}", avg / in_frames as f32);
            //println!("Read {:?} frames from the input stream.", in_frames);
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
            let samps_to_skip = (sample_rate / bandwidth).round() as usize;

            stream.write(write_frames, |output| {
                for i in 0..n_write_samples {
                    // See if we need a new sample
                    if num_sent_samples % samps_to_skip as u64 == 0 {
                        tx_samp = tx_receiver.recv().unwrap();
                    }

                    let pi2 = 2. * std::f32::consts::PI;
                    let e = Complex::new(
                        0.,
                        pi2 * (num_sent_samples / CHANNELS as u64) as f32 / sample_rate
                            * center_frequency,
                    )
                    .exp();
                    num_sent_samples += 1;
                    output[i] = e.re * tx_samp.re + e.im * tx_samp.im;
                }
                //println!("Wrote {:?} frames to the output stream.", out_frames);
            })?;
        }
    }
}

pub fn start_audio<'c>(
    tx: Receiver<Complex<f32>>,
    config: &'c Config,
) -> Result<(std::thread::JoinHandle<()>, AudioSampleStream<'c>), Error> {
    // For the microphone
    let (rx_sender, rx_receiver) = std::sync::mpsc::channel::<f32>();
    let sample_rate = 44100.;
    let center_frequency = config.audio.center_frequency;
    let bandwidth = config.audio.bandwidth;

    let handle = std::thread::spawn(move || {
        run(tx, rx_sender, sample_rate, center_frequency, bandwidth);
    });

    return Ok((
        handle,
        AudioSampleStream::new(rx_receiver, sample_rate as f32, config),
    ));

    /*
    let handle = std::thread::spawn(move || {
        // Size of the buffer in which we'll send/receive data
        const FRAMES: u32 = 256;

        let pa = pa::PortAudio::new().unwrap();

        println!("All devices:");
        for d in pa.devices().unwrap() {
            println!("{:#?}", d);
        }

        // Let's use device 5 for now (Warning: not portable)
        let device = pa::DeviceIndex(5);

        // Construct the output stream settings
        let output_dev = device; //pa.default_output_device()?;
        let output_info = pa.device_info(output_dev).unwrap();
        println!("Output Device: {:#?}", output_info);
        let latency = output_info.default_high_output_latency;
        let output_params = pa::StreamParameters::<f32>::new(output_dev, 1, false, latency);
        assert_eq!(sample_rate, output_info.default_sample_rate);

        // Construct the input stream settings
        let input_dev = device; //pa.default_input_device()?;
        let input_info = pa.device_info(input_dev).unwrap();
        println!("Input Device: {:#?}", input_info);
        let latency = input_info.default_high_input_latency;
        let input_params = pa::StreamParameters::<f32>::new(input_dev, 1, false, latency);
        // Both input and sample rates should be equal
        assert_eq!(input_info.default_sample_rate, sample_rate);

        // Check whether duplex format is supported
        pa.is_duplex_format_supported(input_params, output_params, sample_rate)
            .unwrap();
        let settings =
            pa::DuplexStreamSettings::new(input_params, output_params, sample_rate, FRAMES);
        // Total number of samples sent so far
        let mut num_sent_samps = 0;

        let mut stream = pa.open_blocking_stream(settings).unwrap();

        stream.start().unwrap();

        // We'll use this function to wait for read/write availability.
        fn wait_for_stream<F>(f: F, name: &str) -> u32
        where
            F: Fn() -> Result<pa::StreamAvailable, pa::error::Error>,
        {
            println!("Trying {}", name);
            loop {
                match f() {
                    Ok(available) => match available {
                        pa::StreamAvailable::Frames(frames) => return frames as u32,
                        pa::StreamAvailable::InputOverflowed => {
                            println!("Input stream has overflowed")
                        }
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

        // Now start the main read/write loop! In this example, we
        // pass the input buffer directly to the output buffer, so
        // watch out for feedback.
        loop {
            // How many frames are available on the input stream?
            let in_frames = wait_for_stream(|| stream.read_available(), "Read");
            println!("{} in_frames", in_frames);

            // If there are frames available, let's take them and
            // add them to our buffer.
            if in_frames > 0 {
                let input_samples = stream.read(in_frames).unwrap();
                for samp in input_samples {
                    rx_sender.send(*samp).unwrap();
                }
                //buffer.extend(input_samples.into_iter());
                println!("Read {:?} frames from the input stream.", in_frames);
            }

            // How many frames are available for writing on the output stream?
            let out_frames = wait_for_stream(|| stream.write_available(), "Write");

            // If there are frames available for writing then write!
            if out_frames > 0 {
                stream
                    .write(out_frames, |output| {
                        println!("Writing: {} {}", out_frames, output.len());
                        for i in 0..out_frames as usize {
                            let samp = if let Ok(samp) = tx.recv() {
                                samp
                            } else {
                                break;
                            };
                            let pi2 = 2. * std::f32::consts::PI;
                            let e = Complex::new(
                                0.,
                                pi2 * num_sent_samps as f32 / sample_rate as f32 * center_frequency,
                            )
                            .exp();
                            num_sent_samps += 1;
                            output[i] = samp.re * e.re + samp.im * e.im;
                        }
                        println!("Wrote {:?} frames to the output stream.", out_frames);
                    })
                    .unwrap();
                println!("Written");
            }
        }

        /*

                let callback = move |pa::DuplexStreamCallbackArgs {
                                         in_buffer,
                                         out_buffer,
                                         frames,
                                         ..
                                     }| {
                    assert_eq!(frames, FRAMES as usize);
                    //println!("{:.2?}", in_buffer);

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
                            pi2 * num_sent_samps as f32 / sample_rate as f32 * center_frequency,
                        )
                        .exp();
                        num_sent_samps += 1;
                        *output_sample = samp.re * e.re + samp.im * e.im
                    }

                    pa::Continue
                };

                let mut stream = pa.open_non_blocking_stream(duplex_settings, callback)?;

                stream.start()?;
                // std::thread::spawn(move || {
                //     stream.start().unwrap();
                //     while stream.is_active().unwrap() {
                // 	//std::thread::sleep(std::time::Duration::from_secs(1));
                //     }
                // });
                std::mem::forget(stream);
        */
    });
    Ok((
        handle,
        AudioSampleStream::new(rx_receiver, sample_rate as f32, config),
    ))*/
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
                    pi2 * self.config.audio.center_frequency * self.samps_so_far as f32
                        / self.sample_rate,
                )
                .exp();
        }
        avg /= samps_to_skip as f32;
        Some(avg)
    }
}
