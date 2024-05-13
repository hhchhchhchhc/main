
use csv::WriterBuilder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::File;
use parquet::record::RowAccessor;
use std::f64::NAN;
use csv;

#[derive(Debug, Clone)]
struct DataItem {
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

impl DataItem {
    fn new(open: f64, high: f64, low: f64, close: f64, volume: f64) -> Self {
        DataItem {
            open,
            high,
            low,
            close,
            volume,
        }
    }
}

fn sma(prices: &[f64], window: usize) -> Vec<f64> {
    prices.windows(window)
        .map(|w| w.iter().sum::<f64>() / w.len() as f64)
        .collect()
}

fn ema(prices: &[f64], window: usize, smoothing: f64) -> Vec<f64> {
    let mut ema = Vec::with_capacity(prices.len());
    let multiplier = smoothing / (window as f64 + 1.0);
    let mut prev_ema = prices[0];

    ema.push(prev_ema);

    for price in prices.iter().skip(1) {
        let curr_ema = (price - prev_ema) * multiplier + prev_ema;
        ema.push(curr_ema);
        prev_ema = curr_ema;
    }

    ema
}

fn factor_price_volume_corr(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let prices: Vec<f64> = df.iter().map(|item| item.close).collect();
        let volumes: Vec<f64> = df.iter().map(|item| item.volume).collect();

        let price_changes: Vec<f64> = prices.windows(2).map(|w| w[1] - w[0]).collect();
        let volume_changes: Vec<f64> = volumes.windows(2).map(|w| w[1] - w[0]).collect();

        price_changes.windows(n)
            .zip(volume_changes.windows(n))
            .map(|(price_window, volume_window)| {
                let sum_prod: f64 = price_window.iter().zip(volume_window.iter()).map(|(p, v)| p * v).sum();
                let sum_price_sq: f64 = price_window.iter().map(|p| p.powi(2)).sum();
                let sum_volume_sq: f64 = volume_window.iter().map(|v| v.powi(2)).sum();
                sum_prod / (sum_price_sq.sqrt() * sum_volume_sq.sqrt())
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_sma_crossover(df: &[DataItem], n1: usize, n2: usize) -> Vec<f64> {
    let closes: Vec<f64> = df.iter().map(|item| item.close).collect();
    
    let sma1 = sma(&closes, n1);
    let sma2 = sma(&closes, n2);
    
    sma1.iter()
        .zip(sma2.iter())
        .map(|(s1, s2)| if s1 > s2 { 1.0 } else if s1 < s2 { -1.0 } else { 0.0 })
        .collect()
}

fn factor_price_high_low(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let highs: Vec<f64> = df.iter().map(|item| item.high).collect();
        let lows: Vec<f64> = df.iter().map(|item| item.low).collect();

        highs.windows(n)
            .zip(lows.windows(n))
            .map(|(high_window, low_window)| {
                let max_high = high_window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min_low = low_window.iter().cloned().fold(f64::INFINITY, f64::min);
                (max_high == df[df.len() - 1].high) as i64 as f64 - (min_low == df[df.len() - 1].low) as i64 as f64
            })
            .collect()
    } else if n == df.len() {
        // 当窗口大小等于数据长度时,直接返回 0.0
        vec![0.0]
    } else {
        vec![]
    }
}

fn factor_volume_high_low(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let volumes: Vec<f64> = df.iter().map(|item| item.volume).collect();

        volumes.windows(n)
            .map(|volume_window| {
                let max_volume = volume_window.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let min_volume = volume_window.iter().cloned().fold(f64::INFINITY, f64::min);
                (max_volume == df[df.len() - 1].volume) as i64 as f64 - (min_volume == df[df.len() - 1].volume) as i64 as f64
            })
            .collect()
    } else if n == df.len() {
        // 当窗口大小等于数据长度时,直接返回 0.0
        vec![0.0]
    } else {
        vec![]
    }
}

fn factor_abnormal_volume(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let volumes: Vec<f64> = df.iter().map(|item| item.volume).collect();

        volumes.windows(n)
            .map(|volume_window| {
                let mean_volume = volume_window.iter().sum::<f64>() / volume_window.len() as f64;
                df[df.len() - 1].volume / mean_volume
            })
            .collect()
    } else if n == df.len() {
        // 当窗口大小等于数据长度时,直接返回 1.0
        vec![1.0]
    } else {
        vec![]
    }
}

fn factor_price_volatility(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let closes: Vec<f64> = df.iter().map(|item| item.close).collect();

        closes.windows(n)
            .map(|close_window| {
                let changes: Vec<f64> = close_window.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
                let mean_change = changes.iter().sum::<f64>() / changes.len() as f64;
                let variance = changes.iter().map(|c| (c - mean_change).powi(2)).sum::<f64>() / changes.len() as f64;
                variance.sqrt()
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_volume_volatility(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let volumes: Vec<f64> = df.iter().map(|item| item.volume).collect();

        volumes.windows(n)
            .map(|volume_window| {
                let changes: Vec<f64> = volume_window.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
                let mean_change = changes.iter().sum::<f64>() / changes.len() as f64;
                let variance = changes.iter().map(|c| (c - mean_change).powi(2)).sum::<f64>() / changes.len() as f64;
                variance.sqrt()
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_order_imbalance(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let mut buy_volumes: Vec<f64> = Vec::new();
        let mut sell_volumes: Vec<f64> = Vec::new();

        for item in df {
            if item.close > item.open {
                buy_volumes.push(item.volume);
                sell_volumes.push(0.0);
            } else if item.close < item.open {
                buy_volumes.push(0.0);
                sell_volumes.push(item.volume);
            } else {
                buy_volumes.push(0.0);
                sell_volumes.push(0.0);
            }
        }

        buy_volumes.windows(n)
            .zip(sell_volumes.windows(n))
            .map(|(buy_window, sell_window)| {
                let total_volume: f64 = buy_window.iter().sum::<f64>() + sell_window.iter().sum::<f64>();
                (buy_window.iter().sum::<f64>() - sell_window.iter().sum::<f64>()) / total_volume
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_return_skewness(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let closes: Vec<f64> = df.iter().map(|item| item.close).collect();

        closes.windows(n)
            .map(|close_window| {
                let changes: Vec<f64> = close_window.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
                let mean_change = changes.iter().sum::<f64>() / changes.len() as f64;
                let variance = changes.iter().map(|c| (c - mean_change).powi(2)).sum::<f64>() / changes.len() as f64;
                let std_dev = variance.sqrt();
                let skewness = changes.iter().map(|c| ((c - mean_change) / std_dev).powi(3)).sum::<f64>() / changes.len() as f64;
                skewness
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_return_kurtosis(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let closes: Vec<f64> = df.iter().map(|item| item.close).collect();

        closes.windows(n)
            .map(|close_window| {
                let changes: Vec<f64> = close_window.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();
                let mean_change = changes.iter().sum::<f64>() / changes.len() as f64;
                let variance = changes.iter().map(|c| (c - mean_change).powi(2)).sum::<f64>() / changes.len() as f64;
                let std_dev = variance.sqrt();
                let kurtosis = changes.iter().map(|c| ((c - mean_change) / std_dev).powi(4)).sum::<f64>() / changes.len() as f64;
                kurtosis
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_price_cog(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let highs: Vec<f64> = df.iter().map(|item| item.high).collect();
        let lows: Vec<f64> = df.iter().map(|item| item.low).collect();

        highs.windows(n)
            .zip(lows.windows(n))
            .map(|(high_window, low_window)| {
                let mean_high = high_window.iter().sum::<f64>() / high_window.len() as f64;
                let mean_low = low_window.iter().sum::<f64>() / low_window.len() as f64;
                (mean_high + mean_low) / 2.0
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_vwap(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let closes: Vec<f64> = df.iter().map(|item| item.close).collect();
        let volumes: Vec<f64> = df.iter().map(|item| item.volume).collect();

        closes.windows(n)
            .zip(volumes.windows(n))
            .map(|(close_window, volume_window)| {
                let sum_price_volume: f64 = close_window.iter().zip(volume_window.iter()).map(|(p, v)| p * v).sum();
                let sum_volume: f64 = volume_window.iter().sum();
                sum_price_volume / sum_volume
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_money_flow(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let highs: Vec<f64> = df.iter().map(|item| item.high).collect();
        let lows: Vec<f64> = df.iter().map(|item| item.low).collect();
        let closes: Vec<f64> = df.iter().map(|item| item.close).collect();
        let volumes: Vec<f64> = df.iter().map(|item| item.volume).collect();

        highs.windows(n)
            .zip(lows.windows(n))
            .zip(closes.windows(n))
            .zip(volumes.windows(n))
            .map(|(((high_window, low_window), close_window), volume_window)| {
                let typical_price: Vec<f64> = high_window.iter().zip(low_window.iter()).zip(close_window.iter())
                    .map(|((h, l), c)| (h + l + c) / 3.0)
                    .collect();
                let money_flow: Vec<f64> = typical_price.iter().zip(volume_window.iter()).map(|(p, v)| p * v).collect();
                money_flow.iter().sum::<f64>() / volume_window.iter().sum::<f64>()
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_psy_line(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let closes: Vec<f64> = df.iter().map(|item| item.close).collect();

        closes.windows(n)
            .map(|close_window| {
                let num_up_days = close_window.iter().zip(close_window.iter().skip(1)).filter(|&(x, y)| x > y).count() as f64 / n as f64;
                num_up_days
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_pos_neg_volume(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let pos_volume: Vec<f64> = df.windows(n)
            .map(|window| window.iter().filter(|item| item.close > item.open).map(|item| item.volume).sum())
            .collect();
        let neg_volume: Vec<f64> = df.windows(n)
            .map(|window| window.iter().filter(|item| item.close < item.open).map(|item| item.volume).sum())
            .collect();
    
        if pos_volume.len() == neg_volume.len() {
            pos_volume.iter().zip(neg_volume.iter()).map(|(pos, neg)| if *neg != 0.0 { pos / neg } else { 0.0 }).collect()
        } else {
            vec![]
        }
    } else {
        vec![]
    }
}


fn factor_avg_candle_body(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        df.windows(n)
            .map(|window| window.iter().map(|item| (item.close - item.open).abs()).sum::<f64>() / n as f64)
            .collect()
    } else {
        vec![]
    }
}

fn factor_avg_candle_shadow(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        df.windows(n)
            .map(|window| {
                let upper_shadow: f64 = window.iter().map(|item| item.high - item.close.max(item.open)).sum();
                let lower_shadow: f64 = window.iter().map(|item| item.close.min(item.open) - item.low).sum();
                (upper_shadow + lower_shadow) / n as f64
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_bullish_candle_ratio(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        df.windows(n)
            .map(|window| window.iter().filter(|item| item.close > item.open).count() as f64 / n as f64)
            .collect()
    } else {
        vec![]
    }
}

fn factor_information_ratio(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let returns: Vec<f64> = df.windows(2).map(|window| (window[1].close / window[0].close - 1.0)).collect();
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_dev = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f64>().sqrt() / (returns.len() - 1) as f64;
        returns.windows(n).map(|window| window.iter().sum::<f64>() / n as f64 / std_dev).collect()
    } else {
        vec![]
    }
}

fn factor_rs_ratio(df: &[DataItem], n1: usize, n2: usize) -> Vec<f64> {
    if n1 < df.len() && n2 < df.len() {
        let sma1: Vec<f64> = sma(&df.iter().map(|item| item.close).collect::<Vec<f64>>(), n1);
        let sma2: Vec<f64> = sma(&df.iter().map(|item| item.close).collect::<Vec<f64>>(), n2);
        
        sma1.iter().zip(sma2.iter()).map(|(s1, s2)| if *s2 != 0.0 { s1 / s2 } else { 0.0 }).collect()
    } else {
        vec![]
    }
}

fn factor_turnover_rate(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        df.windows(n)
            .map(|window| {
                let sum_volume = window.iter().map(|item| item.volume).sum::<f64>();
                if sum_volume != 0.0 {
                    window.iter().map(|item| item.volume).sum::<f64>() / sum_volume
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_volume_burst(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        let max_volume: Vec<f64> = df.windows(n).map(|window| window.iter().map(|item| item.volume).fold(f64::NEG_INFINITY, f64::max)).collect();
        df.iter().map(|item| item.volume).zip(max_volume.iter()).map(|(volume, max)| if *max != 0.0 { volume / max } else { 0.0 }).collect()
    } else {
        vec![]
    }
}

fn factor_price_breakout(df: &[DataItem], n: usize) -> Vec<f64> {
    if n < df.len() {
        df.windows(n + 1)
            .map(|window| {
                if window[0].close != 0.0 {
                    (window[n].close - window[0].close) / window[0].close
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_price_oscillation(df: &[DataItem], n: usize) -> Vec<f64> {
    if n <= df.len() {
        df.windows(n)
            .map(|window| {
                let max_high = window.iter().map(|item| item.high).fold(f64::NEG_INFINITY, f64::max);
                let min_low = window.iter().map(|item| item.low).fold(f64::INFINITY, f64::min);
                let mean_close = window.iter().map(|item| item.close).sum::<f64>() / n as f64;
                if mean_close != 0.0 {
                    (max_high - min_low) / mean_close
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_volume_price_divergence(df: &[DataItem], n: usize) -> Vec<f64> {
    if n <= df.len() {
        let price_change: Vec<f64> = df.windows(n).map(|window| (window[n - 1].close / window[0].close - 1.0)).collect();
        let volume_change: Vec<f64> = df.windows(n).map(|window| (window[n - 1].volume / window[0].volume - 1.0)).collect();
        price_change.iter().zip(volume_change.iter()).map(|(p, v)| p - v).collect()
    } else {
        vec![]
    }
}

fn factor_smart_money_flow(df: &[DataItem], n: usize) -> Vec<f64> {
    if n <= df.len() {
        df.windows(n)
            .map(|window| {
                let typical_price: Vec<f64> = window.iter().map(|item| (item.high + item.low + item.close) / 3.0).collect();
                let money_flow: Vec<f64> = typical_price.iter().zip(window.iter()).map(|(price, item)| {
                    if item.high - item.low != 0.0 {
                        price * item.volume * (item.close - item.open) / (item.high - item.low)
                    } else {
                        0.0
                    }
                }).collect();
                money_flow.iter().sum::<f64>()
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_momentum_reversal(df: &[DataItem], n: usize) -> Vec<f64> {
    if n <= df.len() {
        let returns: Vec<f64> = df.windows(2).map(|window| (window[1].close / window[0].close - 1.0)).collect();
        let sign: Vec<f64> = returns.iter().map(|r| r.signum()).collect();
        sign.windows(n)
            .map(|window| (window[n - 1] != window[0]) as i64 as f64)
            .collect()
    } else {
        vec![]
    }
}

fn factor_jump_risk(df: &[DataItem], n: usize, threshold: f64) -> Vec<f64> {
    if n <= df.len() {
        df.windows(n)
            .map(|window| {
                let jumps: Vec<bool> = window.iter().map(|item| {
                    if item.open != 0.0 {
                        (item.high - item.low) / item.open > threshold
                    } else {
                        false
                    }
                }).collect();
                jumps.iter().filter(|&&jump| jump).count() as f64 / n as f64
            })
            .collect()
    } else {
        vec![]
    }
}

fn factor_abnormal_trading_pattern(df: &[DataItem], n: usize, threshold: f64) -> Vec<f64> {
    if n <= df.len() {
        let volume_mean: Vec<f64> = df.windows(n).map(|window| window.iter().map(|item| item.volume).sum::<f64>() / n as f64).collect();
        let volume_std: Vec<f64> = df.windows(n)
            .map(|window| {
                let mean = window.iter().map(|item| item.volume).sum::<f64>() / n as f64;
                window.iter().map(|item| (item.volume - mean).powi(2)).sum::<f64>().sqrt() / (n as f64)
            })
            .collect();
        
        df.iter().map(|item| item.volume).zip(volume_mean.iter()).zip(volume_std.iter())
            .map(|((volume, mean), std)| {
                if *std != 0.0 {
                    ((volume - mean) / std > threshold) as i64 as f64
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        vec![]
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open("/1mBTCUSDT240418.parquet")?;
    let reader: Box<dyn FileReader> = Box::new(SerializedFileReader::new(file)?);
    let mut iter = reader.get_row_iter(None)?;

    let mut data = Vec::new();

    while let Some(record) = iter.next() {
        let record = record?;
        let open = record.get_double(0)?;
        let high = record.get_double(1)?;
        let low = record.get_double(2)?;
        let close = record.get_double(3)?;
        let volume = record.get_double(4)?;

        data.push(DataItem::new(open, high, low, close, volume));
    }


    let periods = vec![5, 10, 20, 60];
    let period_pairs = vec![(5, 10), (10, 20), (20, 60)];

let mut features = Vec::new();

for &n in &periods {
    features.push((format!("price_volume_corr_{}", n), n, factor_price_volume_corr(&data, n)));
    features.push((format!("price_high_low_{}", n), n, factor_price_high_low(&data, n)));
    features.push((format!("volume_high_low_{}", n), n, factor_volume_high_low(&data, n)));
    features.push((format!("price_volatility_{}", n), n, factor_price_volatility(&data, n)));
    features.push((format!("volume_volatility_{}", n), n, factor_volume_volatility(&data, n)));
    features.push((format!("order_imbalance_{}", n), n, factor_order_imbalance(&data, n)));
    features.push((format!("return_skewness_{}", n), n, factor_return_skewness(&data, n)));
    features.push((format!("return_kurtosis_{}", n), n, factor_return_kurtosis(&data, n)));
    features.push((format!("abnormal_volume_{}", n), n, factor_abnormal_volume(&data, n)));
    features.push((format!("price_cog_{}", n), n, factor_price_cog(&data, n)));
    features.push((format!("vwap_{}", n), n, factor_vwap(&data, n)));
    features.push((format!("money_flow_{}", n), n, factor_money_flow(&data, n)));
    features.push((format!("psy_line_{}", n), n, factor_psy_line(&data, n)));
    features.push((format!("pos_neg_volume_{}", n), n, factor_pos_neg_volume(&data, n)));
    features.push((format!("avg_candle_body_{}", n), n, factor_avg_candle_body(&data, n)));
    features.push((format!("avg_candle_shadow_{}", n), n, factor_avg_candle_shadow(&data, n)));
    features.push((format!("bullish_candle_ratio_{}", n), n, factor_bullish_candle_ratio(&data, n)));
    features.push((format!("information_ratio_{}", n), n, factor_information_ratio(&data, n)));
    features.push((format!("turnover_rate_{}", n), n, factor_turnover_rate(&data, n)));
    features.push((format!("volume_burst_{}", n), n, factor_volume_burst(&data, n)));
    features.push((format!("price_breakout_{}", n), n, factor_price_breakout(&data, n)));
    features.push((format!("price_oscillation_{}", n), n, factor_price_oscillation(&data, n)));
    features.push((format!("volume_price_divergence_{}", n), n, factor_volume_price_divergence(&data, n)));
    features.push((format!("smart_money_flow_{}", n), n, factor_smart_money_flow(&data, n)));
    features.push((format!("momentum_reversal_{}", n), n, factor_momentum_reversal(&data, n)));
    features.push((format!("jump_risk_005_{}", n), n, factor_jump_risk(&data, n, 0.05)));
    features.push((format!("jump_risk_01_{}", n), n, factor_jump_risk(&data, n, 0.1)));
    features.push((format!("abnormal_trading_pattern_2_{}", n), n, factor_abnormal_trading_pattern(&data, n, 2.0)));
    features.push((format!("abnormal_trading_pattern_3_{}", n), n, factor_abnormal_trading_pattern(&data, n, 3.0)));
}

for &(n1, n2) in &period_pairs {
    features.push((format!("sma_crossover_{}_{}", n1, n2), n1, factor_sma_crossover(&data, n1, n2)));
    features.push((format!("rs_ratio_{}_{}", n1, n2), n1, factor_rs_ratio(&data, n1, n2)));
}


  let mut feature_data: Vec<Vec<f64>> = vec![vec![]; features.len()];

    for (i, (_feature_name, _period, feature_value)) in features.iter().enumerate() {
        for value in feature_value {
            feature_data[i].push(*value);
        }
    }

    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(File::create("/hy-tmp/factors3.csv")?);

    let mut record = csv::ByteRecord::new();

    // Write header row
    record.push_field(b"feature_name");
    for (feature_name, _, _) in &features {
        record.push_field(feature_name.as_bytes());
    }
    writer.write_record(&record)?;

    // Write data rows
    for i in 0..feature_data[0].len() {
        record.clear();
        record.push_field(&i.to_string().into_bytes());
        for feature_values in &feature_data {
            record.push_field(&feature_values[i].to_string().into_bytes());
        }
        writer.write_record(&record)?;
    }

    writer.flush()?;

    Ok(())

}
