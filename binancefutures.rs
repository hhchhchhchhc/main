use binance::api::*;
use binance::config::*;
use binance::futures::account::*;

fn main() {
    let api_key = Some("".into());
    let secret_key = Some("".into());
    let config = Config::default();
    let account: FuturesAccount = Binance::new_with_config(api_key, secret_key, &config);

    let result = account.account_information();
    match result {
        Ok(account_info) => println!("{:?}", account_info),
        Err(e) => println!("Error getting account information: {:?}", e),
    }

    let result = account.account_balance();
    match result {
        Ok(balances) => {
            println!("Account balances: {:?}", balances);
            if let Some(USDC_balance) = balances.iter().find(|b| b.asset == "USDC") {
                println!("USDC balance: {:?}", USDC_balance);
            } else {
                println!("No USDC balance found");
            }
        }
        Err(e) => println!("Error getting account balances: {:?}", e),
    }
}
