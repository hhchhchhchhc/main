import pandas as pd
import numpy as np
from scipy import signal
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.tsa.stattools import coint
from scipy.stats import skew, kurtosis
import networkx as nx
open = df['Open']
high = df['High'] 
low = df['Low']
close = df['Close']
volume = df['Volume']

factor_df = pd.DataFrame()
factor_names = []

# 价格与交易量相关性因子
def factor_price_volume_corr(df, n):
    price = df['Close'].pct_change()
    volume = df['Volume'].pct_change()
    corr = price.rolling(n).corr(volume)
    return pd.Series(corr, name=f'PriceVolumeCorr_{n}')

# 价格与交易量的协整性因子
def factor_price_volume_cointegration(df, n):
    price = df['Close'].replace(np.inf,np.nan).replace(-np.inf,np.nan).fillna(method='ffill').fillna(0)
    volume = df['Volume'].replace(np.inf,np.nan).replace(-np.inf,np.nan).fillna(method='ffill').fillna(0)
    _, pvalue, _ = coint(price[-n:], volume[-n:])
    return pd.Series(pvalue, index=[df.index[-1]], name=f'PriceVolumeCointegration_{n}')

# 均线交叉因子
def factor_sma_crossover(df, n1, n2):
    sma1 = df['Close'].rolling(n1).mean()
    sma2 = df['Close'].rolling(n2).mean()
    crossover = (sma1 > sma2).astype(int) - (sma1 < sma2).astype(int)
    return pd.Series(crossover, name=f'SMACrossover_{n1}_{n2}')

# 价格新高新低因子
def factor_price_high_low(df, n):
    high = (df['High'] == df['High'].rolling(n).max()).astype(int)
    low = (df['Low'] == df['Low'].rolling(n).min()).astype(int)
    return pd.Series(high - low, name=f'PriceHighLow_{n}')

# 成交量新高新低因子
def factor_volume_high_low(df, n):
    high = (df['Volume'] == df['Volume'].rolling(n).max()).astype(int)
    low = (df['Volume'] == df['Volume'].rolling(n).min()).astype(int)
    return pd.Series(high - low, name=f'VolumeHighLow_{n}')

# 价格波动率因子
def factor_price_volatility(df, n):
    return pd.Series(df['Close'].pct_change().rolling(n).std(), name=f'PriceVolatility_{n}')

# 成交量波动率因子
def factor_volume_volatility(df, n):  
    return pd.Series(df['Volume'].pct_change().rolling(n).std(), name=f'VolumeVolatility_{n}')
  
# 订单失衡因子
def factor_order_imbalance(df, n):
    buy_volume = df['Volume'][df['Close'] > df['Open']]  
    sell_volume = df['Volume'][df['Close'] < df['Open']]
    buy_volume = buy_volume.fillna(0)
    sell_volume = sell_volume.fillna(0)
    imbalance = (buy_volume - sell_volume).rolling(n).sum() / df['Volume'].rolling(n).sum()
    return pd.Series(imbalance, name=f'OrderImbalance_{n}')

# 收益率偏度因子
def factor_return_skewness(df, n):
    return pd.Series(df['Close'].pct_change().rolling(n).apply(skew), name=f'ReturnSkewness_{n}')

# 收益率峰度因子
def factor_return_kurtosis(df, n):
    return pd.Series(df['Close'].pct_change().rolling(n).apply(kurtosis), name=f'ReturnKurtosis_{n}')

# 异常交易量因子
def factor_abnormal_volume(df, n):
    return pd.Series(df['Volume'] / df['Volume'].rolling(n).mean(), name=f'AbnormalVolume_{n}')
    
# 价格重心因子
def factor_price_cog(df, n):
    high = df['High'].rolling(n).mean()
    low = df['Low'].rolling(n).mean()
    return pd.Series((high + low) / 2, name=f'PriceCOG_{n}')
    
# 成交量加权价格因子
def factor_vwap(df, n):
    vwap = (df['Close'] * df['Volume']).rolling(n).sum() / df['Volume'].rolling(n).sum()
    return pd.Series(df['Close'] / vwap, name=f'VWAP_{n}')
    
# 资金流向因子
def factor_money_flow(df, n):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume'] 
    mf_ratio = money_flow.rolling(n).sum() / df['Volume'].rolling(n).sum()
    return pd.Series(mf_ratio, name=f'MoneyFlow_{n}')

# 心理线因子
def factor_psy_line(df, n):
    psy_line = df['Close'].rolling(n).apply(lambda x: sum(x > x.shift(1)) / n)
    return pd.Series(psy_line, name=f'PSY_{n}')

# 正负成交量因子
def factor_pos_neg_volume(df, n):
    pos_volume = df['Volume'][df['Close'] > df['Close'].shift(1)].rolling(n).sum()
    neg_volume = df['Volume'][df['Close'] < df['Close'].shift(1)].rolling(n).sum()
    return pd.Series(pos_volume / neg_volume, name=f'PosNegVolume_{n}')

# 平均K线实体因子
def factor_avg_candle_body(df, n):
    candle_body = abs(df['Close'] - df['Open'])
    return pd.Series(candle_body.rolling(n).mean(), name=f'AvgCandleBody_{n}')

# 平均K线上下影线因子
def factor_avg_candle_shadow(df, n):
    upper_shadow = df['High'] - df[['Close', 'Open']].max(axis=1)
    lower_shadow = df[['Close', 'Open']].min(axis=1) - df['Low']
    total_shadow = (upper_shadow + lower_shadow).rolling(n).mean()
    return pd.Series(total_shadow, name=f'AvgCandleShadow_{n}')

# 阳线比例因子
def factor_bullish_candle_ratio(df, n):
    bullish_ratio = df['Close'].rolling(n).apply(lambda x: sum(x > x.shift(1)) / n)
    return pd.Series(bullish_ratio, name=f'BullishCandleRatio_{n}')

# 信息比率因子  
def factor_information_ratio(df, n):
    returns = df['Close'].pct_change()
    ir = returns.rolling(n).mean() / returns.rolling(n).std()
    return pd.Series(ir, name=f'InformationRatio_{n}')

# 相对强弱比率因子
def factor_rs_ratio(df, n1, n2):
    rs = df['Close'].rolling(n1).mean() / df['Close'].rolling(n2).mean()
    return pd.Series(rs, name=f'RSRatio_{n1}_{n2}')

# 换手率因子
def factor_turnover_rate(df, n):
    turnover_rate = df['Volume'].rolling(n).sum() / df['Volume'].rolling(n).mean()
    return pd.Series(turnover_rate, name=f'TurnoverRate_{n}')

# 爆量因子
def factor_volume_burst(df, n):
    volume_burst = df['Volume'] / df['Volume'].rolling(n).max()
    return pd.Series(volume_burst, name=f'VolumeBurst_{n}')

# 价格突破因子
def factor_price_breakout(df, n):
    breakout = (df['Close'] - df['Close'].shift(n)) / df['Close'].shift(n)
    return pd.Series(breakout, name=f'PriceBreakout_{n}')

# 震荡幅度因子
def factor_price_oscillation(df, n):
    osc_range = (df['High'].rolling(n).max() - df['Low'].rolling(n).min()) / df['Close'].rolling(n).mean()
    return pd.Series(osc_range, name=f'PriceOscillation_{n}')

# 异常成交量因子
def factor_abnormal_volume(df, n):
    abnormal_volume = df['Volume'] / df['Volume'].rolling(n).mean()
    return pd.Series(abnormal_volume, name=f'AbnormalVolume_{n}')

# 量价背离因子
def factor_volume_price_divergence(df, n):
    price_change = df['Close'].pct_change(n)
    volume_change = df['Volume'].pct_change(n)
    divergence = price_change - volume_change
    return pd.Series(divergence, name=f'VolumePriceDivergence_{n}')

# 智能资金流向因子
def factor_smart_money_flow(df, n):
    smart_money_flow = df['Close'] * df['Volume'] * (df['Close'] - df['Open']) / (df['High'] - df['Low'])
    return pd.Series(smart_money_flow.rolling(n).sum(), name=f'SmartMoneyFlow_{n}')

# 相关市场动量因子
def factor_related_market_momentum(df, related_df, n):
    related_return = related_df['Close'].pct_change(n)
    return pd.Series(related_return, name=f'RelatedMarketMomentum_{n}')

# 相关市场波动率因子
def factor_related_market_volatility(df, related_df, n):  
    related_volatility = related_df['Close'].pct_change().rolling(n).std()
    return pd.Series(related_volatility, name=f'RelatedMarketVolatility_{n}')
    
# 相关市场流动性因子
def factor_related_market_liquidity(df, related_df, n):
    related_liquidity = related_df['Volume'].rolling(n).mean()
    return pd.Series(related_liquidity, name=f'RelatedMarketLiquidity_{n}')

# 机器学习预测因子
def factor_ml_prediction(df, n, target_col='Close', feature_cols=['Open', 'High', 'Low', 'Volume']):
    X = df[feature_cols].values
    y = df[target_col].pct_change(n).shift(-n).values

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X[:-n], y[:-n])
    prediction = model.predict(X[-n:])

    return pd.Series(prediction, index=df.index[-n:], name=f'MLPrediction_{n}')

# 跨品种动量因子
def factor_cross_asset_momentum(df, related_df, n):  
    return_diff = df['Close'].pct_change(n) - related_df['Close'].pct_change(n)
    return pd.Series(return_diff, name=f'CrossAssetMomentum_{n}')
    
# 跨品种相关性因子
def factor_cross_asset_correlation(df, related_df, n):
    correlation = df['Close'].pct_change().rolling(n).corr(related_df['Close'].pct_change())
    return pd.Series(correlation, name=f'CrossAssetCorrelation_{n}')

# 行业轮动因子
def factor_sector_rotation(df, sector_data, n):
    sector_return = sector_data.pct_change(n)
    sector_rank = sector_return.rank(axis=1, ascending=False)
    return pd.Series(sector_rank.mean(axis=1), name=f'SectorRotation_{n}')

# 经济政策不确定性因子
def factor_economic_policy_uncertainty(df, epu_data, n):
    epu_change = epu_data.pct_change(n)
    return pd.Series(epu_change, name=f'EconomicPolicyUncertainty_{n}')

# 投资者情绪因子
def factor_investor_sentiment(df, sentiment_data, n):
    sentiment_change = sentiment_data.pct_change(n)
    return pd.Series(sentiment_change, name=f'InvestorSentiment_{n}')

# 复杂网络中心性因子
def factor_complex_network_centrality(df, network_data, n):
    network = nx.from_pandas_adjacency(network_data)
    centrality = nx.eigenvector_centrality(network)
    centrality_series = pd.Series(centrality)
    return pd.Series(centrality_series.rolling(n).mean(), name=f'ComplexNetworkCentrality_{n}')

# 动量反转因子
def factor_momentum_reversal(df, n):  
    return_sign = np.sign(df['Close'].pct_change(n))
    reversal = (return_sign != return_sign.shift(1)).astype(int)
    return pd.Series(reversal, name=f'MomentumReversal_{n}')

# 隐含波动率因子
def factor_implied_volatility(df, option_data, n):
    iv_change = option_data['ImpliedVolatility'].pct_change(n)
    return pd.Series(iv_change, name=f'ImpliedVolatility_{n}')

# 隐含流动性因子
def factor_implied_liquidity(df, option_data, n):  
    iv_spread = option_data['AskImpliedVolatility'] - option_data['BidImpliedVolatility']
    iv_spread_change = iv_spread.pct_change(n)
    return pd.Series(iv_spread_change, name=f'ImpliedLiquidity_{n}')
    
# 信息不对称因子(续)
def factor_information_asymmetry(df, order_data, n):
    volume_imbalance = order_data['BuyVolume'] / (order_data['BuyVolume'] + order_data['SellVolume'])
    volume_imbalance_change = volume_imbalance.pct_change(n)
    return pd.Series(volume_imbalance_change, name=f'InformationAsymmetry_{n}')

# 限价单流量因子
def factor_limit_order_flow(df, order_data, n):
    order_flow = order_data['LimitBuyVolume'] - order_data['LimitSellVolume']
    order_flow_ma = order_flow.rolling(n).mean()
    return pd.Series(order_flow_ma, name=f'LimitOrderFlow_{n}')

# 情绪偏差因子
def factor_sentiment_bias(df, news_data, n):
    sentiment_score = news_data['SentimentScore']
    sentiment_bias = sentiment_score - sentiment_score.rolling(n).mean()
    return pd.Series(sentiment_bias, name=f'SentimentBias_{n}')

# 跳跃风险因子
def factor_jump_risk(df, n, threshold=0.05):
    jump = (df['High'] - df['Low']) / df['Open'] > threshold
    jump_ma = jump.rolling(n).mean()
    return pd.Series(jump_ma, name=f'JumpRisk_{n}_{threshold}')

# 新闻情感因子
def factor_news_sentiment(df, sentiment_data, n):
    sentiment_score = sentiment_data.rolling(n).mean()
    return pd.Series(sentiment_score, name=f'NewsSentiment_{n}')

# 社交媒体关注度因子
def factor_social_media_attention(df, social_data, n):
    attention_score = social_data.rolling(n).sum()
    return pd.Series(attention_score, name=f'SocialMediaAttention_{n}')

# 卫星图像因子
def factor_satellite_image(df, image_data, n):
    image_feature = image_data.rolling(n).mean()
    return pd.Series(image_feature, name=f'SatelliteImage_{n}')

# 深度学习特征因子
def factor_deep_learning_feature(df, n, feature_cols=['Open', 'High', 'Low', 'Close', 'Volume']):
    X = df[feature_cols].values.reshape(-1, n, len(feature_cols))

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(n, len(feature_cols))),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X[:-1], df['Close'].pct_change().shift(-1).values[:-1], epochs=50, batch_size=32, verbose=0)

    feature = model.predict(X[-1:]).squeeze()
    return pd.Series(feature, index=[df.index[-1]], name=f'DeepLearningFeature_{n}')

# 异常交易模式因子
def factor_abnormal_trading_pattern(df, n, threshold=2):
    volume_mean = df['Volume'].rolling(n).mean()
    volume_std = df['Volume'].rolling(n).std()
    abnormal_volume = (df['Volume'] > volume_mean + threshold * volume_std).astype(int)
    return pd.Series(abnormal_volume, name=f'AbnormalTradingPattern_{n}_{threshold}')

# 分形维度因子
def factor_fractal_dimension(df, n, method='hausdorff'):
    price = df['Close'].values
    rs = np.log(price[1:] / price[:-1])
    if method == 'hausdorff':
        fd = 2 - np.log(np.sum(np.absolute(rs))) / np.log(n)
    elif method == 'hurst':
        fd = 2 - hurst_exponent(rs)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return pd.Series(fd, index=[df.index[-1]], name=f'FractalDimension_{n}_{method}')

def hurst_exponent(time_series):
    lags = range(2, 100)
    tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0

# 泛在量化因子
def factor_ubiquitous_quantification(df, ubiq_data, n):
    ubiq_factor = ubiq_data.rolling(n).mean()
    return pd.Series(ubiq_factor, name=f'UbiquitousQuantification_{n}')

# 代际传递因子
def factor_generational_transmission(df, parent_data, n):
    gt_factor = parent_data.shift(n).rolling(n).corr(df['Close'])
    return pd.Series(gt_factor, name=f'GenerationalTransmission_{n}')

# 生物节律因子
def factor_biological_rhythm(df, rhythm_data, n):
    rhythm_factor = rhythm_data.rolling(n).apply(lambda x: signal.correlate(x, df['Close'][-n:]).mean())
    return pd.Series(rhythm_factor, name=f'BiologicalRhythm_{n}')

# 集群行为因子
def factor_swarm_behavior(df, swarm_data, n):
    swarm_factor = swarm_data.rolling(n).apply(lambda x: np.mean([cosine_similarity(x[-1], y) for y in x]))
    return pd.Series(swarm_factor, name=f'SwarmBehavior_{n}')
    
    
# 在这里将所有因子计算结果添加到factor_df中
for n in [5, 10, 20, 60]:
    print('--------------------------------------------------------------------------------------------',1)
    factor_df[f'PriceVolumeCorr_{n}'] = factor_price_volume_corr(df, n)
    factor_names.append(f'PriceVolumeCorr_{n}')
    
    print('--------------------------------------------------------------------------------------------',2)
    factor_df[f'PriceVolumeCointegration_{n}'] = factor_price_volume_cointegration(df, n)
    factor_names.append(f'PriceVolumeCointegration_{n}')
    
    print('--------------------------------------------------------------------------------------------',3)
    factor_df[f'PriceHighLow_{n}'] = factor_price_high_low(df, n)
    factor_names.append(f'PriceHighLow_{n}')
    
    factor_df[f'VolumeHighLow_{n}'] = factor_volume_high_low(df, n)  
    factor_names.append(f'VolumeHighLow_{n}')
    
    factor_df[f'PriceVolatility_{n}'] = factor_price_volatility(df, n)
    factor_names.append(f'PriceVolatility_{n}')
    
    factor_df[f'VolumeVolatility_{n}'] = factor_volume_volatility(df, n)
    factor_names.append(f'VolumeVolatility_{n}')
    
    factor_df[f'OrderImbalance_{n}'] = factor_order_imbalance(df, n)
    factor_names.append(f'OrderImbalance_{n}')
    
    print('--------------------------------------------------------------------------------------------',4)
    factor_df[f'AbnormalVolume_{n}'] = factor_abnormal_volume(df, n)
    factor_names.append(f'AbnormalVolume_{n}')
    
    factor_df[f'PriceCOG_{n}'] = factor_price_cog(df, n) 
    factor_names.append(f'PriceCOG_{n}')
    
    factor_df[f'VWAP_{n}'] = factor_vwap(df, n)
    factor_names.append(f'VWAP_{n}')
    
    print('--------------------------------------------------------------------------------------------',5)
    factor_df[f'MoneyFlow_{n}'] = factor_money_flow(df, n)
    factor_names.append(f'MoneyFlow_{n}')
    
    print('--------------------------------------------------------------------------------------------',50)
    factor_df[f'PosNegVolume_{n}'] = factor_pos_neg_volume(df, n)
    factor_names.append(f'PosNegVolume_{n}')
    
    print('--------------------------------------------------------------------------------------------',51)
    factor_df[f'AvgCandleBody_{n}'] = factor_avg_candle_body(df, n)
    factor_names.append(f'AvgCandleBody_{n}')
    
    print('--------------------------------------------------------------------------------------------',52)
    factor_df[f'AvgCandleShadow_{n}'] = factor_avg_candle_shadow(df, n)
    factor_names.append(f'AvgCandleShadow_{n}')
    
    print('--------------------------------------------------------------------------------------------',54)
    factor_df[f'InformationRatio_{n}'] = factor_information_ratio(df, n)
    factor_names.append(f'InformationRatio_{n}')
    
    print('--------------------------------------------------------------------------------------------',55)
    factor_df[f'TurnoverRate_{n}'] = factor_turnover_rate(df, n)
    factor_names.append(f'TurnoverRate_{n}')
    
    print('--------------------------------------------------------------------------------------------',6)
    factor_df[f'VolumeBurst_{n}'] = factor_volume_burst(df, n)
    factor_names.append(f'VolumeBurst_{n}')
    
    factor_df[f'PriceBreakout_{n}'] = factor_price_breakout(df, n)
    factor_names.append(f'PriceBreakout_{n}')
    
    factor_df[f'PriceOscillation_{n}'] = factor_price_oscillation(df, n)
    factor_names.append(f'PriceOscillation_{n}')
    
    factor_df[f'AbnormalVolume_{n}'] = factor_abnormal_volume(df, n)
    factor_names.append(f'AbnormalVolume_{n}')
    
    factor_df[f'VolumePriceDivergence_{n}'] = factor_volume_price_divergence(df, n)
    factor_names.append(f'VolumePriceDivergence_{n}')
    
    factor_df[f'SmartMoneyFlow_{n}'] = factor_smart_money_flow(df, n)
    factor_names.append(f'SmartMoneyFlow_{n}')
    
    """factor_df[f'MLPrediction_{n}'] = factor_ml_prediction(df, n)
    factor_names.append(f'MLPrediction_{n}')"""
    
    """if related_df is not None:
        factor_df[f'RelatedMarketMomentum_{n}'] = factor_related_market_momentum(df, related_df, n)
        factor_names.append(f'RelatedMarketMomentum_{n}')
        
        factor_df[f'RelatedMarketVolatility_{n}'] = factor_related_market_volatility(df, related_df, n)
        factor_names.append(f'RelatedMarketVolatility_{n}')
        
        factor_df[f'RelatedMarketLiquidity_{n}'] = factor_related_market_liquidity(df, related_df, n)
        factor_names.append(f'RelatedMarketLiquidity_{n}')
        
        factor_df[f'CrossAssetMomentum_{n}'] = factor_cross_asset_momentum(df, related_df, n)
        factor_names.append(f'CrossAssetMomentum_{n}')
        
        factor_df[f'CrossAssetCorrelation_{n}'] = factor_cross_asset_correlation(df, related_df, n)
        factor_names.append(f'CrossAssetCorrelation_{n}')

    if sector_data is not None:
        factor_df[f'SectorRotation_{n}'] = factor_sector_rotation(df, sector_data, n)
        factor_names.append(f'SectorRotation_{n}')
        
    if epu_data is not None:
        factor_df[f'EconomicPolicyUncertainty_{n}'] = factor_economic_policy_uncertainty(df, epu_data, n)
        factor_names.append(f'EconomicPolicyUncertainty_{n}')
        
    if sentiment_data is not None:
        factor_df[f'InvestorSentiment_{n}'] = factor_investor_sentiment(df, sentiment_data, n)
        factor_names.append(f'InvestorSentiment_{n}')
        
    if network_data is not None:
        factor_df[f'ComplexNetworkCentrality_{n}'] = factor_complex_network_centrality(df, network_data, n)
        factor_names.append(f'ComplexNetworkCentrality_{n}')
        
    if option_data is not None:  
        factor_df[f'ImpliedVolatility_{n}'] = factor_implied_volatility(df, option_data, n)
        factor_names.append(f'ImpliedVolatility_{n}')
        
        factor_df[f'ImpliedLiquidity_{n}'] = factor_implied_liquidity(df, option_data, n)
        factor_names.append(f'ImpliedLiquidity_{n}')
        
    if order_data is not None:
        factor_df[f'InformationAsymmetry_{n}'] = factor_information_asymmetry(df, order_data, n)
        factor_names.append(f'InformationAsymmetry_{n}')
        
        factor_df[f'LimitOrderFlow_{n}'] = factor_limit_order_flow(df, order_data, n)
        factor_names.append(f'LimitOrderFlow_{n}')
        
    if news_data is not None:
        factor_df[f'SentimentBias_{n}'] = factor_sentiment_bias(df, news_data, n)
        factor_names.append(f'SentimentBias_{n}')
        
        factor_df[f'NewsSentiment_{n}'] = factor_news_sentiment(df, news_data, n)
        factor_names.append(f'NewsSentiment_{n}')
        
    if ubiq_data is not None:
        factor_df[f'UbiquitousQuantification_{n}'] = factor_ubiquitous_quantification(df, ubiq_data, n)
        factor_names.append(f'UbiquitousQuantification_{n}')

    if parent_data is not None:
        factor_df[f'GenerationalTransmission_{n}'] = factor_generational_transmission(df, parent_data, n)
        factor_names.append(f'GenerationalTransmission_{n}')
        
    if rhythm_data is not None:
        factor_df[f'BiologicalRhythm_{n}'] = factor_biological_rhythm(df, rhythm_data, n)
        factor_names.append(f'BiologicalRhythm_{n}')
        
    if swarm_data is not None:
        factor_df[f'SwarmBehavior_{n}'] = factor_swarm_behavior(df, swarm_data, n)
        factor_names.append(f'SwarmBehavior_{n}')"""
        
    factor_df[f'JumpRisk_{n}_0.05'] = factor_jump_risk(df, n, threshold=0.05)
    factor_names.append(f'JumpRisk_{n}_0.05')
    
    factor_df[f'JumpRisk_{n}_0.1'] = factor_jump_risk(df, n, threshold=0.1)
    factor_names.append(f'JumpRisk_{n}_0.1')
    
    factor_df[f'FractalDimension_{n}_hausdorff'] = factor_fractal_dimension(df, n, method='hausdorff')
    factor_names.append(f'FractalDimension_{n}_hausdorff')
    
    factor_df[f'FractalDimension_{n}_hurst'] = factor_fractal_dimension(df, n, method='hurst')
    factor_names.append(f'FractalDimension_{n}_hurst')
    
    print('--------------------------------------------------------------------------------------------',7)
    factor_df[f'AbnormalTradingPattern_{n}_2'] = factor_abnormal_trading_pattern(df, n, threshold=2)
    factor_names.append(f'AbnormalTradingPattern_{n}_2')
    
    factor_df[f'AbnormalTradingPattern_{n}_3'] = factor_abnormal_trading_pattern(df, n, threshold=3)
    factor_names.append(f'AbnormalTradingPattern_{n}_3')

for n1, n2 in [(5, 10), (10, 20), (20, 60)]:
    factor_df[f'SMACrossover_{n1}_{n2}'] = factor_sma_crossover(df, n1, n2)
    factor_names.append(f'SMACrossover_{n1}_{n2}')
    
    factor_df[f'RSRatio_{n1}_{n2}'] = factor_rs_ratio(df, n1, n2)
    factor_names.append(f'RSRatio_{n1}_{n2}')

"""if social_data is not None:
    for n in [5, 10, 20, 60]:
        factor_df[f'SocialMediaAttention_{n}'] = factor_social_media_attention(df, social_data, n)
        factor_names.append(f'SocialMediaAttention_{n}')
        
if image_data is not None:
    for n in [5, 10, 20, 60]:
        factor_df[f'SatelliteImage_{n}'] = factor_satellite_image(df, image_data, n)
        factor_names.append(f'SatelliteImage_{n}')"""

"""for n in [10, 20, 30, 50]:
    factor_df[f'DeepLearningFeature_{n}'] = factor_deep_learning_feature(df, n)
    factor_names.append(f'DeepLearningFeature_{n}')"""
    
for n in [5, 10, 20, 60]:  
    factor_df[f'MomentumReversal_{n}'] = factor_momentum_reversal(df, n)
    factor_names.append(f'MomentumReversal_{n}')

print(f"Generated {len(factor_names)} factors in total.")
#newdf.to_parquet(f'./generate_factors3_1mfutures{sym}USDT.parquet') 
factor_df.to_parquet(f'./generate_factors3_1mfutures{sym}USDT.parquet') 
