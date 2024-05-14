from scipy.stats import linregress
import pandas as pd
import numpy as np
from scipy import signal
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.stattools import coint
from scipy.stats import skew, kurtosis

class FT:

    def __init__(self, df, windows=[5, 10, 20, 30, 60], return_ma_windows=[5], price_fields=['Open', 'High', 'Low', 'Close'], vol_fields=['Volume']):
        self.df = df
        self.windows = windows
        self.return_ma_windows = return_ma_windows
        self.price_fields = price_fields
        self.vol_fields = vol_fields  
        self.feature_names = []

    def calculate(self):
        features_data = {}
        
        # 计算价格类因子
        for field in self.price_fields:
            features_data=self.cal_price_field(field, features_data)
        
        # 计算交易量类因子
        for field in self.vol_fields:
            features_data=self.cal_vol_field(field, features_data)
        
        # 计算价量组合类因子 
        features_data=self.cal_price_vol_factors(features_data)
        
        # 计算收益率类因子
        features_data=self.cal_return_factors(features_data)
        
        # 计算其他类型因子
        features_data=self.cal_other_factors(features_data)
        
        self.features = pd.DataFrame(features_data, index=self.df.index)
        return self.features

    def cal_return_factors(self, features_data):
        for d in self.return_ma_windows:
            col = f'return_ma_{d}'
            features_data[col] = self.df['Close'].pct_change(d).rolling(d).mean()
            self.feature_names.append(col)
        return features_data
    def cal_price_field(self, field, features_data):
        for d in self.windows:
            # ROC
            col = f'{field}_roc_{d}' 
            features_data[col] = self.df[field].shift(d) / self.df[field]
            self.feature_names.append(col)
            
            # MA
            col = f'{field}_ma_{d}'
            features_data[col] = self.df[field].rolling(d).mean() 
            self.feature_names.append(col)
            
            # STD
            col = f'{field}_std_{d}'
            features_data[col] = self.df[field].rolling(d).std()
            self.feature_names.append(col)
            
            # Beta
            col = f'{field}_beta_{d}'
            features_data[col] = self.beta(self.df[field], d)
            self.feature_names.append(col)
            
            # 20% Quantile
            col = f'{field}_q20_{d}'
            features_data[col] = self.df[field].rolling(d).quantile(0.2)
            self.feature_names.append(col)
            
            # 80% Quantile
            col = f'{field}_q80_{d}'
            features_data[col] = self.df[field].rolling(d).quantile(0.8)
            self.feature_names.append(col)
            
            # Min
            col = f'{field}_min_{d}'
            features_data[col] = self.df[field].rolling(d).min()
            self.feature_names.append(col)
            
            # Max
            col = f'{field}_max_{d}'
            features_data[col] = self.df[field].rolling(d).max()
            self.feature_names.append(col)

        return features_data
    def cal_vol_field(self, field, features_data):
        for d in self.windows:
            # ROC
            col = f'{field}_roc_{d}'
            features_data[col] = (self.df[field].shift(d) - self.df[field]) / self.df[field]
            self.feature_names.append(col)
            
            # MA  
            col = f'{field}_ma_{d}'
            features_data[col] = self.df[field].rolling(d).mean()
            self.feature_names.append(col)
            
            # STD
            col = f'{field}_std_{d}'
            features_data[col] = self.df[field].rolling(d).std()
            self.feature_names.append(col)

        return features_data
    def cal_price_vol_factors(self, features_data):
        for d in self.windows:
            # 价格/成交量
            col = f'price_volume_roc_{d}'
            features_data[col] = (self.df['Close'].shift(d) - self.df['Close']) / (self.df['Volume'].shift(d) - self.df['Volume'])
            self.feature_names.append(col)

            col = f'price_volume_ma_{d}'  
            features_data[col] = self.df['Close'].rolling(d).mean() / self.df['Volume'].rolling(d).mean()
            self.feature_names.append(col)
            
            # 成交额
            col = f'amount_ma_{d}'
            features_data[col] = (self.df['Close'] * self.df['Volume']).rolling(d).mean()
            self.feature_names.append(col)
            
            # 加权成交量
            col = f'wvma_{d}'
            features_data[col] = self.wvma(d)
            self.feature_names.append(col)
            
            # 交易量动量（正/负）
            col = f'v_sump_{d}'
            features_data[col] = self.df['Volume'][self.df['Close'] > self.df['Close'].shift(1)].rolling(d).sum()
            self.feature_names.append(col)
            
            col = f'v_sumn_{d}'
            features_data[col] = self.df['Volume'][self.df['Close'] < self.df['Close'].shift(1)].rolling(d).sum() 
            self.feature_names.append(col)
            
            # 心理线
            col = f'psy_line_{d}'
            features_data[col] = self.psy_line(d)
            self.feature_names.append(col)

        return features_data
    def cal_other_factors(self, features_data):
        max_length = max(len(self.df), max(len(v) for v in features_data.values()))
        for d in self.windows:
            # Fractal Dimension
            col = 'fractal_dimension_hausdorff_%d' % d
            features_data[col] = self.factor_fractal_dimension(d)
            self.feature_names.append(col)
            col = 'fractal_dimension_hurst_%d' % d
            features_data[col] = self.factor_fractal_dimension(d,'hurst')
            self.feature_names.append(col)

            # Smart Money Flow  
            col = 'smart_money_flow_%d' % d
            features_data[col] = np.log(self.df['High']-self.df['Low']) * np.log(self.df['Volume']) * (self.df['Close']-self.df['Open'])/self.df['Open']
            self.feature_names.append(col)

            # Momentum Reversal
            col = 'momentum_reversal_%d' % d
            features_data[col] = self.df['Close'].pct_change(d) * self.df['Close'].pct_change(1) 
            self.feature_names.append(col)

            # Jump Risk
            col = 'jump_risk_005_%d' % d
            jump = (self.df['High'] - self.df['Low']) / self.df['Open'] > 0.05
            features_data[col] = jump.rolling(d).mean()
            self.feature_names.append(col)

            col = 'jump_risk_01_%d' % d  
            jump = (self.df['High'] - self.df['Low']) / self.df['Open'] > 0.1
            features_data[col] = jump.rolling(d).mean()
            self.feature_names.append(col)

            # Pos Neg Volume
            col = 'pos_neg_volume_%d' % d
            pos_vol = self.df['Volume'][self.df['Close']>self.df['Open']].rolling(d).sum()
            neg_vol = self.df['Volume'][self.df['Close']<=self.df['Open']].rolling(d).sum()
            features_data[col] = pos_vol / (pos_vol + neg_vol + 1e-8) 
            self.feature_names.append(col)

            # Average Candle Body/Shadow
            col = 'avg_candle_body_%d' % d
            body = abs(self.df['Close'] - self.df['Open']) 
            features_data[col] = body.rolling(d).mean()
            self.feature_names.append(col)

            col = 'avg_candle_shadow_%d' % d
            wick = (self.df['High'] - self.df[['Open','Close']].max(axis=1)) + (self.df[['Open','Close']].min(axis=1) - self.df['Low'])
            features_data[col] = wick.rolling(d).mean()
            self.feature_names.append(col)

            # Information Ratio
            col = 'information_ratio_%d' % d
            ret = self.df['Close'].pct_change()
            features_data[col] = ret.rolling(d).mean() / ret.rolling(d).std()
            self.feature_names.append(col)

            # Volume Burst
            col = 'volume_burst_%d' % d
            features_data[col] = self.df['Volume'] / self.df['Volume'].rolling(d).max()
            self.feature_names.append(col)

            # Price Breakout  
            col = 'price_breakout_%d' % d
            price_change = (self.df['Close'] - self.df['Close'].shift(d)) / self.df['Close'].shift(d)
            features_data[col] = price_change
            self.feature_names.append(col)


            # Price Oscillation
            col = 'price_oscillation_%d' % d  
            osc_range = (self.df['High'].rolling(d).max() - self.df['Low'].rolling(d).min()) / self.df['Close'].rolling(d).mean()
            features_data[col] = osc_range  
            self.feature_names.append(col)

            # Volume Price Divergence  
            col = 'volume_price_divergence_%d' % d
            p_change = self.df['Close'].pct_change(d)
            v_change = self.df['Volume'].pct_change(d) 
            features_data[col] = p_change - v_change
            self.feature_names.append(col)

            # VWAP
            col = 'vwap_%d' % d
            vwap = (self.df['Volume'] * (self.df['High']+self.df['Low']+self.df['Close'])/3).rolling(d).sum() / self.df['Volume'].rolling(d).sum()
            features_data[col] = self.df['Close'] / vwap
            self.feature_names.append(col)

        for d1,d2 in [(5,10),(10,20),(20,60)]:  
            # RS Ratio
            col = 'rs_ratio_%d_%d' % (d1,d2)
            rs = self.df['Close'].pct_change(d1).rolling(d1).mean() / self.df['Close'].pct_change(d2).rolling(d2).mean()
            features_data[col] = rs
            self.feature_names.append(col)

            # SMA Crossover
            col = 'sma_crossover_%d_%d' % (d1, d2)
            sma1 = self.df['Close'].rolling(d1).mean()
            sma2 = self.df['Close'].rolling(d2).mean()
            features_data[col] = np.where(sma1>sma2, 1, -1)
            self.feature_names.append(col)

        # Return distribution 
        for d in self.windows:
            ret = self.df['Close'].pct_change(d)

            col = 'return_skewness_%d' % d
            features_data[col] = ret.rolling(d).skew()
            self.feature_names.append(col)

            col = 'return_kurtosis_%d' % d  
            features_data[col] = ret.rolling(d).kurt()
            self.feature_names.append(col)

        # Candle type
        for d in self.windows:
            green = (self.df['Close'] > self.df['Open']).astype(int)

            col = 'bullish_candle_ratio_%d' % d
            bull_count = green.rolling(d).sum()
            features_data[col] = bull_count / d
            self.feature_names.append(col)

        # Turnover
        for d in self.windows:  
            col = 'turnover_rate_%d' % d
            turnover = self.df['Volume'].rolling(d).sum() / self.df['Volume'].rolling(d).mean() 
            features_data[col] = turnover
            self.feature_names.append(col)

        # Abnormal trading pattern
        for d in self.windows:
            vol_ma = self.df['Volume'].rolling(d).mean()
            vol_std = self.df['Volume'].rolling(d).std()
            col = 'abnormal_trading_pattern_2_%d' % d
            features_data[col] = (self.df['Volume'] > vol_ma + 2*vol_std).astype(int)
            self.feature_names.append(col)

            col = 'abnormal_trading_pattern_3_%d' % d
            features_data[col] = (self.df['Volume'] > vol_ma + 3*vol_std).astype(int)
            self.feature_names.append(col)

        # Price volume correlation  
        for d in self.windows:
            ret = np.log(self.df['Close']).diff()
            vol = np.log(self.df['Volume'])

            col = 'price_volume_corr_%d' % d
            features_data[col] = ret.rolling(d).corr(vol)
            self.feature_names.append(col)
            
        # Price/Volume Cointegration
        for d in self.windows:
            col = 'price_volume_cointegration_%d' % d
            features_data[col] = self.factor_price_volume_cointegration(d)
            self.feature_names.append(col)

        # Abnormal Volume
        for d in self.windows:
            col = 'abnormal_volume_%d' % d
            features_data[col] = self.factor_abnormal_volume(d)
            self.feature_names.append(col)

        # Volume High/Low
        for d in self.windows:
            col = 'volume_high_low_%d' % d
            features_data[col] = self.factor_volume_high_low(d) 
            self.feature_names.append(col)

        # Price High/Low
        for d in self.windows:
            col = 'price_high_low_%d' % d
            features_data[col] = self.factor_price_high_low(d)
            self.feature_names.append(col)

        # Price COG
        for d in self.windows:
            col = 'price_cog_%d' % d
            features_data[col] = self.factor_price_cog(d)
            self.feature_names.append(col)

        # Price/Volume Correlation
        for d in self.windows:
            col = 'p_v_corr0_%d' % d
            features_data[col] = self.factor_price_volume_corr0(d)
            self.feature_names.append(col)

        for d in self.windows:
            col = 'p_v_corr1_%d' % d
            features_data[col] = self.factor_price_volume_corr1(d)
            self.feature_names.append(col)

        for d in self.windows:
            col = 'p_v_cord_%d' % d
            features_data[col] = self.factor_price_volume_cord(d)
            self.feature_names.append(col)
        # Money Flow
        for d in self.windows:
            col = 'money_flow_%d' % d
            features_data[col] = self.factor_money_flow(d)
            self.feature_names.append(col)

        # Order Imbalance
        for d in self.windows:
            col = 'order_imbalance_%d' % d
            features_data[col] = self.factor_order_imbalance(d)
            self.feature_names.append(col)

        # Price/Volume Volatility
        for d in self.windows:
            col = 'price_volatility_%d' % d
            features_data[col] = self.factor_price_volatility(d)
            self.feature_names.append(col)
            
            col = 'volume_volatility_%d' % d
            features_data[col] = self.factor_volume_volatility(d)
            self.feature_names.append(col)
        for k, v in features_data.items():
            if len(v) < max_length:
                features_data[k] = np.concatenate((np.nan*np.zeros(max_length - len(v)),v.values.flatten()))
        #features_data['feature_name'] = self.feature_names
        return features_data

    def hurst_exponent(self, series, window):
        if len(series) < window:
            return np.nan
        lags = range(2, window)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]*2.0

    def factor_price_volume_corr0(self, n):
        price = self.df['Close'].pct_change()
        volume = self.df['Volume'].pct_change()
        corr = price.rolling(n).corr(volume)
        return corr

    def factor_price_volume_corr1(self, n):
        return self.df['Close'].rolling(d).corr(np.log(self.df['Volume']+1))
    def factor_price_volume_cord(self, n):
        price_change = self.df['Close'] / self.df['Close'].shift(1)
        vol_change = self.df['Volume'] / self.df['Volume'].shift(1) 
        return price_change.rolling(d).corr(np.log(vol_change+1))

    def factor_price_volume_cointegration(self, n):
        price = self.df['Close'].replace(np.inf,np.nan).replace(-np.inf,np.nan).fillna(method='ffill').fillna(0)
        volume = self.df['Volume'].replace(np.inf,np.nan).replace(-np.inf,np.nan).fillna(method='ffill').fillna(0)
        _, pvalue, _ = coint(price[-n:], volume[-n:])
        return pd.Series(np.full(self.df.shape[0], pvalue), index=self.df.index)
        
    def factor_sma_crossover(self, n1, n2):
        sma1 = self.df['Close'].rolling(n1).mean()
        sma2 = self.df['Close'].rolling(n2).mean()
        crossover = (sma1 > sma2).astype(int) - (sma1 < sma2).astype(int)
        return crossover

    def factor_price_high_low(self, n):
        high = (self.df['High'] == self.df['High'].rolling(n).max()).astype(int)
        low = (self.df['Low'] == self.df['Low'].rolling(n).min()).astype(int)
        return high - low
    
    def factor_volume_high_low(self, n):  
        high = (self.df['Volume'] == self.df['Volume'].rolling(n).max()).astype(int)
        low = (self.df['Volume'] == self.df['Volume'].rolling(n).min()).astype(int)
        return high - low
    
    def factor_price_volatility(self, n):
        return self.df['Close'].pct_change().rolling(n).std()

    def factor_volume_volatility(self, n):
        return self.df['Volume'].pct_change().rolling(n).std() 
    
    def factor_order_imbalance(self, n):
        buy_volume = self.df['Volume'][self.df['Close'] > self.df['Open']]
        sell_volume = self.df['Volume'][self.df['Close'] < self.df['Open']]
        buy_volume = buy_volume.fillna(0)
        sell_volume = sell_volume.fillna(0)
        imbalance = (buy_volume - sell_volume).rolling(n).sum() / self.df['Volume'].rolling(n).sum()
        return imbalance

    def factor_abnormal_volume(self, n):
        return self.df['Volume'] / self.df['Volume'].rolling(n).mean()

    def factor_price_cog(self, n):
        high = self.df['High'].rolling(n).mean() 
        low = self.df['Low'].rolling(n).mean()
        return (high + low) / 2

    def factor_vwap(self, n):
        vwap = (self.df['Close'] * self.df['Volume']).rolling(n).sum() / self.df['Volume'].rolling(n).sum()
        return self.df['Close'] / vwap

    def factor_money_flow(self, n):
        typical_price = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        money_flow = typical_price * self.df['Volume']
        mf_ratio = money_flow.rolling(n).sum() / self.df['Volume'].rolling(n).sum()
        return mf_ratio

    def factor_pos_neg_volume(self, n):
        pos_volume = self.df['Volume'][self.df['Close'] > self.df['Close'].shift(1)].rolling(n).sum()
        neg_volume = self.df['Volume'][self.df['Close'] < self.df['Close'].shift(1)].rolling(n).sum()
        return pos_volume / neg_volume

    def factor_avg_candle_body(self, n):
        candle_body = abs(self.df['Close'] - self.df['Open'])
        return candle_body.rolling(n).mean()

    def factor_avg_candle_shadow(self, n):
        upper_shadow = self.df['High'] - self.df[['Close', 'Open']].max(axis=1)
        lower_shadow = self.df[['Close', 'Open']].min(axis=1) - self.df['Low']
        total_shadow = (upper_shadow + lower_shadow).rolling(n).mean()
        return total_shadow

    def factor_information_ratio(self, n):
        returns = self.df['Close'].pct_change()
        ir = returns.rolling(n).mean() / returns.rolling(n).std()
        return ir

    def factor_rs_ratio(self, n1, n2):
        rs = self.df['Close'].rolling(n1).mean() / self.df['Close'].rolling(n2).mean()
        return rs

    def factor_volume_burst(self, n):
        volume_burst = self.df['Volume'] / self.df['Volume'].rolling(n).max()
        return volume_burst

    def factor_price_breakout(self, n):
        breakout = (self.df['Close'] - self.df['Close'].shift(n)) / self.df['Close'].shift(n)
        return breakout

    def factor_price_oscillation(self, n):
        osc_range = (self.df['High'].rolling(n).max() - self.df['Low'].rolling(n).min()) / self.df['Close'].rolling(n).mean()
        return osc_range

    def factor_volume_price_divergence(self, n):
        price_change = self.df['Close'].pct_change(n)
        volume_change = self.df['Volume'].pct_change(n)
        divergence = price_change - volume_change
        return divergence

    def factor_smart_money_flow(self, n):
        smart_money_flow = self.df['Close'] * self.df['Volume'] * (self.df['Close'] - self.df['Open']) / (self.df['High'] - self.df['Low'])
        return smart_money_flow.rolling(n).sum()

    def factor_momentum_reversal(self, n):
        return_sign = np.sign(self.df['Close'].pct_change(n))
        reversal = (return_sign != return_sign.shift(1)).astype(int)
        return reversal

    def factor_jump_risk(self, n, threshold=0.05):  
        jump = (self.df['High'] - self.df['Low']) / self.df['Open'] > threshold
        jump_ma = jump.rolling(n).mean()
        return jump_ma
    
    def factor_fractal_dimension(self, n, method='hausdorff'):
        price = self.df['Close'].values
        rs = np.log(price[1:] / price[:-1])
        if method == 'hausdorff':
            fd = 2 - np.log(np.sum(np.absolute(rs))) / np.log(n)
        elif method == 'hurst':
            fd = 2 - self.hurst_exponent(rs, n)
        else:
            raise ValueError(f"Unsupported method: {method}")
        return pd.Series(np.full(self.df.shape[0], fd), index=self.df.index, name=f'FractalDimension_{n}_{method}')

    def factor_abnormal_trading_pattern(self, n, threshold=2):
        volume_mean = self.df['Volume'].rolling(n).mean()
        volume_std = self.df['Volume'].rolling(n).std()
        abnormal_volume = (self.df['Volume'] > volume_mean + threshold * volume_std).astype(int)
        return abnormal_volume
        
    def psy_line(self, n):
        psy = self.df['Close'].diff().apply(lambda x: 1 if x > 0 else 0).rolling(n).sum() / n
        return psy

    def wvma(self, n):
        vol_ma = self.df['Volume'].rolling(n).mean()
        wvma = (self.df['Volume'] * self.df['Close']).rolling(n).sum() / vol_ma
        return wvma

    def beta(self, series, n):
        returns = series.pct_change()
        market_returns = self.df['Close'].pct_change()
        cov = returns.rolling(n).cov(market_returns)
        market_var = market_returns.rolling(n).var()
        beta = cov / market_var
        return beta

import pandas as pd
import pandas as pd
# 读取原始数据
sym='BTC'
df = pd.read_parquet(f'1mfutures{sym}USDT18.parquet')
import datetime
# 创建 Alpha158 实例
ft = FT(df)

# 计算因子特征
features = ft.calculate()

# 将因子特征保存为新的 Parquet 文件
output_path = f'/hy-tmp/features03_1mfutures{sym}USDT18.parquet'  # 替换为你要保存的文件路径
features.to_parquet(output_path)
