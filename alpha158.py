import pandas as pd
import numpy as np
class Alpha158:
    def __init__(self, data, windows=[5, 10, 20, 30, 60], return_ma_windows=[5], price_fields=['Open', 'High', 'Low', 'Close'], vol_fields=['Volume']):
        self.data = data
        self.windows = windows
        self.return_ma_windows = return_ma_windows
        self.price_fields = price_fields
        self.vol_fields = vol_fields
        self.feature_names = []
        
    def calculate(self):
        features_data = {}

        # 计算价格类因子
        for field in self.price_fields:
            self.cal_price_field(field, features_data)
                
        # 计算交易量类因子 
        for field in self.vol_fields:
            self.cal_vol_field(field, features_data)

        # 计算价量组合类因子
        self.cal_price_vol_factors(features_data)
                
        # 计算收益率类因子
        self.cal_return_factors(features_data)

        self.features = pd.DataFrame(features_data, index=self.data.index)
        return self.features

    def cal_return_factors(self, features_data):
        for d in self.return_ma_windows:
            col = 'return_ma_%d' % d
            features_data[col] = self.data['Close'].pct_change(d).rolling(d).mean()
            self.feature_names.append(col)
            
    def cal_price_field(self, field, features_data):
        for d in self.windows:
            # ROC
            col = '%s_roc_%d' % (field, d)
            features_data[col] = self.data[field].shift(d) / self.data['Close']
            self.feature_names.append(col)

            # MA
            col = '%s_ma_%d' % (field, d)  
            features_data[col] = self.data[field].rolling(d).mean() / self.data['Close']
            self.feature_names.append(col)

            # STD
            col = '%s_std_%d' % (field, d)
            features_data[col] = self.data[field].rolling(d).std() / self.data['Close'] 
            self.feature_names.append(col)

            # Beta
            col = '%s_beta_%d' % (field, d)
            features_data[col] = (self.data[field] - self.data[field].shift(d)) / d / self.data['Close']
            self.feature_names.append(col)
            
            # MAX
            col = '%s_max_%d' % (field, d)
            features_data[col] = self.data[field].rolling(d).max() / self.data['Close']
            self.feature_names.append(col)
            
            # MIN
            col = '%s_min_%d' % (field, d)
            features_data[col] = self.data[field].rolling(d).min() / self.data['Close']
            self.feature_names.append(col)

            # Quantile
            col = '%s_q80_%d' % (field, d)
            features_data[col] = self.data[field].rolling(d).quantile(0.8) / self.data['Close']
            self.feature_names.append(col)

            col = '%s_q20_%d' % (field, d)
            features_data[col] = self.data[field].rolling(d).quantile(0.2) / self.data['Close']
            self.feature_names.append(col)

    def cal_vol_field(self, field, features_data):
        for d in self.windows:
            # MA
            col = '%s_ma_%d' % (field, d)
            features_data[col] = self.data[field].rolling(d).mean() / (self.data[field] + 1e-8)
            self.feature_names.append(col)

            # STD
            col = '%s_std_%d' % (field, d)
            features_data[col] = self.data[field].rolling(d).std() / (self.data[field] + 1e-8) 
            self.feature_names.append(col)

            # VSUMP
            col = 'v_sump_%d' % d
            v_change = self.data[field] - self.data[field].shift(1)
            sump = v_change[v_change>0].rolling(d).sum()
            sumn = v_change[v_change<0].rolling(d).sum().abs()
            features_data[col] = sump / (sump + sumn + 1e-8)   
            self.feature_names.append(col)

            # VSUMN
            col = 'v_sumn_%d' % d 
            features_data[col] = 1 - features_data['v_sump_%d' % d]
            self.feature_names.append(col)

    def cal_price_vol_factors(self, features_data):
        for d in self.windows:
            # WVMA
            col = 'wvma_%d' % d
            ret_vol = (self.data['Close'] / self.data['Close'].shift(1) - 1).abs() * self.data['Volume']
            ret_vol_ma = ret_vol.rolling(d).mean() 
            features_data[col] = ret_vol.rolling(d).std() / (ret_vol_ma + 1e-8)
            self.feature_names.append(col)

            # CORR
            col = 'p_v_corr_%d' % d
            features_data[col] = self.data['Close'].rolling(d).corr(np.log(self.data['Volume']+1))
            self.feature_names.append(col)
            
            # CORD
            col = 'p_v_cord_%d' % d
            price_change = self.data['Close'] / self.data['Close'].shift(1)
            vol_change = self.data['Volume'] / self.data['Volume'].shift(1) 
            features_data[col] = price_change.rolling(d).corr(np.log(vol_change+1))
            self.feature_names.append(col)
