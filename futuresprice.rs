use csv::WriterBuilder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::File;
use parquet::record::RowAccessor;
use csv;
use std::sync::Arc;
use rayon::prelude::*;

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

fn cal_return_factors(data: &[DataItem], return_ma_windows: &[usize]) -> (std::collections::HashMap<String, Vec<f64>>, Vec<String>) {
    let mut features_data = std::collections::HashMap::new();
    let mut feature_names = Vec::new();

    for &d in return_ma_windows {
        let col = format!("return_ma_{}", d);
        let mut returns: Vec<f64> = data.windows(d+1)
            .map(|w| w[d].close / w[0].close - 1.0)
            .collect();
        let return_ma: Vec<f64> = returns.windows(d)
            .map(|w| w.iter().sum::<f64>() / d as f64)
            .collect();
        returns.drain(..d);
        features_data.insert(col.clone(), return_ma);
        feature_names.push(col);
    }

    (features_data, feature_names)
}

fn cal_price_field(data: &[DataItem], field: &str, windows: &[usize]) -> (std::collections::HashMap<String, Vec<f64>>, Vec<String>) {
    let mut features_data = std::collections::HashMap::new();
    let mut feature_names = Vec::new();

    let price_data: Vec<f64> = match field {
        "Open" => data.iter().map(|item| item.open).collect(),
        "High" => data.iter().map(|item| item.high).collect(),
        "Low" => data.iter().map(|item| item.low).collect(),
        "Close" => data.iter().map(|item| item.close).collect(),
        _ => panic!("Invalid price field"),
    };

    let close_data: Vec<f64> = data.iter().map(|item| item.close).collect();

    for &d in windows {
        // ROC
        let col = format!("{}_roc_{}", field, d);
        let roc: Vec<f64> = price_data.windows(d+1)
            .map(|w| w[d] / close_data[close_data.len() - w.len() + d])
            .collect();
        features_data.insert(col.clone(), roc);
        feature_names.push(col);

        // MA
        let col = format!("{}_ma_{}", field, d);
        let ma: Vec<f64> = price_data.windows(d)
            .map(|w| {
                let sum = w.iter().sum::<f64>();
                let ma = sum / d as f64;
                ma / close_data[close_data.len() - w.len()]
            })
            .collect();
        features_data.insert(col.clone(), ma);
        feature_names.push(col);

        // STD
        let col = format!("{}_std_{}", field, d);
        let std_dev: Vec<f64> = price_data.windows(d)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / d as f64;
                let variance = w.iter().map(|x| (*x - mean).powi(2)).sum::<f64>() / (d - 1) as f64;
                let std = variance.sqrt();
                std / close_data[close_data.len() - w.len()]
            })
            .collect();
        features_data.insert(col.clone(), std_dev);
        feature_names.push(col);

        // Beta
        let col = format!("{}_beta_{}", field, d);
        let beta: Vec<f64> = price_data.windows(d+1)
            .map(|w| (w[d] - w[0]) / d as f64 / w[0])
            .collect();
        features_data.insert(col.clone(), beta);
        feature_names.push(col);

        // MAX
        let col = format!("{}_max_{}", field, d);
        let max: Vec<f64> = price_data.windows(d)
            .map(|w| {
                let max_price = w.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                max_price / close_data[close_data.len() - w.len()]
            })
            .collect();
        features_data.insert(col.clone(), max);
        feature_names.push(col);

        // MIN
        let col = format!("{}_min_{}", field, d);
        let min: Vec<f64> = price_data.windows(d)
            .map(|w| {
                let min_price = w.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                min_price / close_data[close_data.len() - w.len()]
            })
            .collect();
        features_data.insert(col.clone(), min);
        feature_names.push(col);

        // Quantile
        let col = format!("{}_q80_{}", field, d);
        let q80: Vec<f64> = price_data.windows(d)
            .map(|w| {
                let mut prices: Vec<f64> = w.to_vec();
                prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let q80_price = prices[(d as f64 * 0.8) as usize];
                q80_price / close_data[close_data.len() - w.len()]
            })
            .collect();
        features_data.insert(col.clone(), q80);
        feature_names.push(col);

        let col = format!("{}_q20_{}", field, d);
        let q20: Vec<f64> = price_data.windows(d)
            .map(|w| {
                let mut prices: Vec<f64> = w.to_vec();
                prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let q20_price = prices[(d as f64 * 0.2) as usize];
                q20_price / close_data[close_data.len() - w.len()]
            })
            .collect();
        features_data.insert(col.clone(), q20);
        feature_names.push(col);
    }

    (features_data, feature_names)
}

fn cal_vol_field(data: &[DataItem], field: &str, windows: &[usize]) -> (std::collections::HashMap<String, Vec<f64>>, Vec<String>) {
    let mut features_data = std::collections::HashMap::new();
    let mut feature_names = Vec::new();

    let vol_data: Vec<f64> = data.iter().map(|item| item.volume).collect();

    for &d in windows {
        // MA
        let col = format!("{}_ma_{}", field, d);
        let ma: Vec<f64> = vol_data.windows(d)
            .map(|w| {
                let sum = w.iter().sum::<f64>();
                let ma = sum / d as f64;
                ma / (vol_data[vol_data.len() - w.len()] + 1e-8)
            })
            .collect();
        features_data.insert(col.clone(), ma);
        feature_names.push(col);

        // STD
        let col = format!("{}_std_{}", field, d);
        let std_dev: Vec<f64> = vol_data.windows(d)
            .map(|w| {
                let mean = w.iter().sum::<f64>() / d as f64;
                let variance = w.iter().map(|x| (*x - mean).powi(2)).sum::<f64>() / (d - 1) as f64;
                let std = variance.sqrt();
                std / (vol_data[vol_data.len() - w.len()] + 1e-8)
            })
            .collect();
        features_data.insert(col.clone(), std_dev);
        feature_names.push(col);

        // VSUMP
        let col = format!("v_sump_{}", d);
        let v_change: Vec<f64> = vol_data.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        let sump: Vec<f64> = v_change.windows(d)
            .map(|w| w.iter().filter(|&&x| x > 0.0).sum::<f64>())
            .collect();
        let sumn: Vec<f64> = v_change.windows(d)
            .map(|w| w.iter().filter(|&&x| x < 0.0).sum::<f64>().abs())
            .collect();
        let v_sump: Vec<f64> = sump.iter().zip(sumn.iter())
            .map(|(&p, &n)| p / (p + n + 1e-8))
            .collect();
        features_data.insert(col.clone(), v_sump.clone());
        feature_names.push(col);

        // VSUMN
        let col = format!("v_sumn_{}", d);
        let v_sumn: Vec<f64> = v_sump.iter().map(|&x| 1.0 - x).collect();
        features_data.insert(col.clone(), v_sumn);
        feature_names.push(col);
    }

    (features_data, feature_names)
}

fn cal_price_vol_factors(data: &[DataItem], windows: &[usize]) -> (std::collections::HashMap<String, Vec<f64>>, Vec<String>) {
    let mut features_data = std::collections::HashMap::new();
    let mut feature_names = Vec::new();

    for &d in windows {
        // WVMA
        let col = format!("wvma_{}", d);
        let ret_vol: Vec<f64> = data.windows(2)
            .map(|w| (w[1].close / w[0].close - 1.0).abs() * w[1].volume)
            .collect();
        let ret_vol_ma: Vec<f64> = ret_vol.windows(d)
            .map(|w| w.iter().sum::<f64>() / d as f64)
            .collect();
        let wvma: Vec<f64> = ret_vol.windows(d)
            .zip(ret_vol_ma.iter())
            .map(|(w, &ma)| {
                let variance = w.iter().map(|&x| (x - ma).powi(2)).sum::<f64>() / (d - 1) as f64;
                variance.sqrt() / (ma + 1e-8)
            })
            .collect();
        features_data.insert(col.clone(), wvma);
        feature_names.push(col);

        // CORR
        let col = format!("p_v_corr_{}", d);
        let price: Vec<f64> = data.iter().map(|item| item.close).collect();
        let volume: Vec<f64> = data.iter().map(|item| (item.volume + 1.0).ln()).collect();
        let corr: Vec<f64> = price.windows(d)
            .zip(volume.windows(d))
            .map(|(price_window, volume_window)| {
                let (price_mean, volume_mean) = (price_window.iter().sum::<f64>() / d as f64, volume_window.iter().sum::<f64>() / d as f64);
                let (price_std, volume_std) = (
                    price_window.iter().map(|&x| (x - price_mean).powi(2)).sum::<f64>().sqrt(),
                    volume_window.iter().map(|&x| (x - volume_mean).powi(2)).sum::<f64>().sqrt()
                );
                let cov = price_window.iter().zip(volume_window.iter())
                    .map(|(&p, &v)| (p - price_mean) * (v - volume_mean))
                    .sum::<f64>();
                cov / (price_std * volume_std)
            })
            .collect();
        features_data.insert(col.clone(), corr);
        feature_names.push(col);

        // CORD
        let col = format!("p_v_cord_{}", d);
        let price_change: Vec<f64> = data.windows(2)
            .map(|w| w[1].close / w[0].close)
            .collect();
        let vol_change: Vec<f64> = data.windows(2)
            .map(|w| (w[1].volume / w[0].volume).ln())
            .collect();
        let cord: Vec<f64> = price_change.windows(d)
            .zip(vol_change.windows(d))
            .map(|(price_window, volume_window)| {
                let (price_mean, volume_mean) = (price_window.iter().sum::<f64>() / d as f64, volume_window.iter().sum::<f64>() / d as f64);
                let (price_std, volume_std) = (
                    price_window.iter().map(|&x| (x - price_mean).powi(2)).sum::<f64>().sqrt(),
                    volume_window.iter().map(|&x| (x - volume_mean).powi(2)).sum::<f64>().sqrt()
                );
                let cov = price_window.iter().zip(volume_window.iter())
                    .map(|(&p, &v)| (p - price_mean) * (v - volume_mean))
                    .sum::<f64>();
                cov / (price_std * volume_std)
            })
            .collect();
        features_data.insert(col.clone(), cord);
        feature_names.push(col);
    }

    (features_data, feature_names)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file = std::fs::File::open("../1mBTCUSDT240418.parquet")?;
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

    let windows = vec![5, 10, 20, 30, 60];
    let return_ma_windows = vec![5];
    let price_fields = vec!["Open", "High", "Low", "Close"];
    let vol_fields = vec!["Volume"];

    let data = Arc::new(data);

    // 并行计算各类因子
    let price_features: Vec<_> = price_fields.par_iter()
        .map(|field| {
            let data = data.clone();
            cal_price_field(&data, field, &windows)
        })
        .collect();

    let vol_features: Vec<_> = vol_fields.par_iter()
        .map(|field| {
            let data = data.clone();
            cal_vol_field(&data, field, &windows)
        })
        .collect();

    let price_vol_features = {
        let data = data.clone();
        cal_price_vol_factors(&data, &windows)
    };

    let return_features = {
        let data = data.clone();
        cal_return_factors(&data, &return_ma_windows)
    };

    let mut features_data: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
    let mut feature_names: Vec<String> = Vec::new();

    for (data, names) in price_features.into_iter().chain(vol_features.into_iter()).chain(std::iter::once(price_vol_features)).chain(std::iter::once(return_features)) {
        features_data.extend(data);
        feature_names.extend(names);
    }

    let mut features: Vec<Vec<f64>> = Vec::new();
    for name in &feature_names {
        features.push(features_data.get(name).unwrap().clone());
    }

    let mut writer = WriterBuilder::new()
        .has_headers(true)
        .from_writer(File::create("/hy-tmp/features0.csv")?);

    writer.write_record(&feature_names)?;

    for i in 0..features[0].len() {
        let mut record = Vec::new();
        for feature in &features {
            record.push(feature[i].to_string());
        }
        writer.write_record(&record)?;
    }

    writer.flush()?;

    Ok(())
}
