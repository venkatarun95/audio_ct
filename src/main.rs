mod audio;
mod config;
mod packet;
mod pkt_detect;
mod test;

use audio::Audio;
use config::Config;
use packet::rx_loop;

use std::env;

fn main() {
    let config = Config::default();
    config.make_sense();

    let args: Vec<_> = env::args().collect();
    if args.len() == 1 || args[1] == "test" {
        let tx = test::TestTx::new(1024, Some(4), &config);
        let mut lo = test::TestLoopback::new(tx);
        rx_loop(&mut lo, &config);
    } else if args[1] == "audio" {
	// Start up the audio devices
        let (sender, receiver) = std::sync::mpsc::channel();
        let (_, mut rx) = Audio::new(receiver, &config).unwrap();

        // Sender
	let config_c = config.clone();
        std::thread::spawn(move || {
            let tx = test::TestTx::new(1024, None, &config_c);
            for samp in tx {
                sender.send(samp).unwrap();
            }
        });

	// Receive loop
	rx_loop(&mut rx, &config);
    } else {
        eprintln!("Usage: ./ofdm [test|audio]");
    }
}
