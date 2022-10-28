# LongShortCoint
Método de Long Short por Cointegração Quant

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="whitegrid")
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data as pdr
import datetime
import yfinance as yf
yf.pdr_override()
def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs

start = datetime.datetime(2015, 1, 1)
end = datetime.datetime(2020, 1, 1)

tickers = ['CYRE3.SA', 'JHSF3.SA', 'MRVE3.SA', 'GFSA3.SA', 'HBOR3.SA', 'EVEN3.SA']


df = pdr.get_data_yahoo(tickers, start, end)['Close']
df.tail()
scores, pvalues, pairs = find_cointegrated_pairs(df)
import seaborn
fig, ax = plt.subplots(figsize=(10,10))
seaborn.heatmap(pvalues, xticklabels=tickers, yticklabels=tickers, cmap='RdYlGn_r' 
                , mask = (pvalues >= 0.05)
                )
print(pairs)


![Image](https://user-images.githubusercontent.com/67772460/198444034-1721949e-394c-4a2c-8c5d-afd79c805c2f.png)


S1 = df['GFSA3.SA']
S2 = df['MRVE3.SA']

score, pvalue, _ = coint(S1, S2)
pvalue
S1 = sm.add_constant(S1)
results = sm.OLS(S2, S1).fit()
S1 = S1['GFSA3.SA']
b = results.params['GFSA3.SA']

spread = S2 - b * S1
spread.plot(figsize=(12,6))
plt.axhline(spread.mean(), color='black')
plt.xlim('2015-01-01', '2020-01-01')
plt.legend(['Spread']);


![Image](https://user-images.githubusercontent.com/67772460/198444249-38ff3f94-d3e9-4186-a19e-a58014215320.png)


ratio = S1/S2
ratio.plot(figsize=(12,6))
plt.axhline(ratio.mean(), color='black')
plt.xlim('2015-01-01', '2020-01-01')
plt.legend(['Price Ratio']);


![Image](https://user-images.githubusercontent.com/67772460/198444384-d02bc208-6cc7-4a50-981e-507a3c4167a6.png)


def zscore(series):
    return (series - series.mean()) / np.std(series)


zscore(ratio).plot(figsize=(12,6))
plt.axhline(zscore(ratio).mean())
plt.axhline(1.0, color='red')
plt.axhline(-1.0, color='green')
plt.xlim('2015-01-01', '2020-01-01')
plt.show()


![Image](https://user-images.githubusercontent.com/67772460/198444483-baa402a6-e241-46b9-b299-631dcaa7a2c7.png)


ratios = df['GFSA3.SA'] / df['MRVE3.SA'] 
print(len(ratios) * .70 )
train = ratios[:881]
test = ratios[881:]
ratios_mavg5 = train.rolling(window=5, center=False).mean()
ratios_mavg60 = train.rolling(window=60, center=False).mean()
std_60 = train.rolling(window=60, center=False).std()
zscore_60_5 = (ratios_mavg5 - ratios_mavg60)/std_60
plt.figure(figsize=(12, 6))
plt.plot(train.index, train.values)
plt.plot(ratios_mavg5.index, ratios_mavg5.values)
plt.plot(ratios_mavg60.index, ratios_mavg60.values)
plt.legend(['Ratio', '5d Ratio MA', '60d Ratio MA'])

plt.ylabel('Ratio')
plt.show()


![Image](https://user-images.githubusercontent.com/67772460/198444570-a4641f80-be03-4f9a-af7a-89d3ef6a1b7c.png)


plt.figure(figsize=(12,6))
zscore_60_5.plot()
plt.xlim('2015-01-01', '2020-01-01')
plt.axhline(0, color='black')
plt.axhline(1.0, color='red', linestyle='--')
plt.axhline(-1.0, color='green', linestyle='--')
plt.legend(['Rolling Ratio z-Score', 'Mean', '+1', '-1'])
plt.show()


![Image](https://user-images.githubusercontent.com/67772460/198444596-51f3284c-dfbd-445b-ae2d-9573f58655da.png)

