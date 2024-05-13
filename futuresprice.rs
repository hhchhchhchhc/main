use binance::api::*;
use binance::config::*;
use binance::futures::market::FuturesMarket;
use binance::futures::model::MarkPrices;

fn main() {
    let config = Config::default();
    let market: FuturesMarket = Binance::new_with_config(None, None, &config);

    let result = market.get_mark_prices();
    match result {
        Ok(MarkPrices::AllMarkPrices(mark_prices)) => {
            if let Some(mark_price) = mark_prices.iter().find(|mp| mp.symbol == "ETHUSDC") {
                println!("ETHUSDC Perpetual Mark Price: {}", mark_price.mark_price);
            } else {
                println!("No mark price found for ETHUSDC");
            }
        }
        Err(e) => println!("Error getting mark prices: {:?}", e),
    }
}
