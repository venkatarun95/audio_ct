mod config;
mod packet;
mod pkt_detect;
mod test;

use config::Config;
use packet::rx_loop;

fn main() {
    let config = Config::default();
    config.make_sense();
    let tx = test::TestTx::new(1024, 4, &config);
    let mut lo = test::TestLoopback::new(tx);
    rx_loop(&mut lo, &config);
}
