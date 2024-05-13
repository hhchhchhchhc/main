use binance::api::*;
use binance::config::*;
use binance::futures::account::*;
use binance::account::OrderSide; // 添加这行

fn main() {
    let api_key = Some("dtZ8mQh0LWuytNqzfvDMN6sbJRTxpUbNJ9WKkT8YFRltLkir1ezmD1vCVv8Hz56p".into());
    let secret_key = Some("jufQS4dkzXTG3xjq7D9RbJzEc9h9jOdwBxY4RhwixWsl7wTWYIZI8DUd1DJKBPDl".into());
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
            if let Some(usdc_balance) = balances.iter().find(|b| b.asset == "USDC") {
                println!("USDC balance: {:?}", usdc_balance);
            } else {
                println!("No USDC balance found");
            }
        }
        Err(e) => println!("Error getting account balances: {:?}", e),
    }

    // 限价单在价格 2900 委托购买 125 USDC 的 ETHUSDC，指定多头持仓
    let order_request = CustomOrderRequest {
        symbol: "ETHUSDC".into(),
        side: OrderSide::Buy, // 使用导入的 OrderSide
        position_side: Some(PositionSide::Long),
        order_type: OrderType::Limit,
        time_in_force: Some(TimeInForce::GTC),
        qty: Some(1.0),
        reduce_only: None,
        price: Some(2900.0),
        stop_price: None,
        close_position: None,
        activation_price: None,
        callback_rate: None,
        working_type: None,
        price_protect: None,
    };
    let result = account.custom_order(order_request);
    match result {
        Ok(transaction) => println!("Successful limit buy order: {:?}", transaction),
        Err(e) => println!("Error with limit buy order: {:?}", e),
    }
}
