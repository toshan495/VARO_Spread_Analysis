/*
Can you build as a python script/notebook that analyzes the Brent-WTI spread, utilizing the API from Yahoo Finance for the period 2010-now. We are interested in figuring out whether this spread is mean-reverting. Please provide us with a short presentation (max. 4 slides), summarizing the data and your conclusion on mean reversion. If you find any other insights that you think are interesting, highlight them as well. 
*/

/*
Data cleaning code
gen ratio=Brent_Adj_Close/WTI_Adj_Close
drop if ratio<0 | ratio>1.33
tsset Date
tsfill
ipolate Brent_Adj_Close Date,gen(brent)
ipolate WTI_Adj_Close Date,gen(wti)
drop Brent_Adj_Close WTI_Adj_Close
gen spread=brent-wti
replace ratio=brent/wti if ratio==.
*/

import excel "C:\Users\tosha\Documents\Git\pythonProject\VISA_COMPANIES\brent_wti_data.xlsx", sheet("Clean") firstrow
gen spread=brent-wti
gen ratio=brent/wti
des
tsset Date

tsline brent wti spread
tsline ratio
tsline D.spread
scatter brent wti

dfuller brent,lags (1)
dfuller wti,lags (1)
dfuller spread,lags (1)

ac spread # auto correlation graph
pac D.spread $ partial auto correlation graph

arch brent,arch(1/3) # arch modelling
arch D.spread,arch(1) garch(1)

#######################################
PYTHON CODE

#full=spread
#full.index = pd.DatetimeIndex(full.index).to_period('D')
train=spread.loc['2010-01-04':'2019-12-31']
#train.index = pd.DatetimeIndex(train.index).to_period('D')
test=spread.loc['2020-01-02':'2023-07-31']
#test.index = pd.DatetimeIndex(test.index).to_period('D')
######################################
full = pd.DataFrame({'Data': spread.values})
msk = (full.index < len(full)-896)
train = full[msk].copy()
print(df_train)
test = full[~msk].copy()
print(df_test)

################################################################

##ARIMA Modelling:
#no. of significant lags in PACF plot (p) = 1
#difference (d) = 1
#no. of significant lags in ACF plot (q) = 1

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA

##############################################################

df = spread
df.index = df.index.to_timestamp()
df.index = pd.to_datetime(df.index)
df.index = df.index.floor('T')
df.index = df.index.to_pydatetime()
data.index = pd.DatetimeIndex(data.index).to_period('D')

train=df.loc['2010-01-04':'2019-12-31']
test=df.loc['2020-01-02':'2023-07-31']

import pmdarima as pm
auto_arima = pm.auto_arima(df, stepwise=False, seasonal=False)
print(auto_arima.summary())

res = ARIMA(df, order=(1,1,4)).fit()
fig, ax = plt.subplots(figsize=(12,3))
ax = spread.plot(ax=ax)
plot_predict(res,start='2020-01-02',end='2023-07-31', ax=ax)
ax.set_ylim(-2,28)
plt.show()

#CHANGE POINT DETECTION USING KERNEL CHANGE DETECTION ALGORITHM
import pandas as pd
import numpy as np
import ruptures as rpt
import datetime
import matplotlib.dates as mdates

df = spread
#data=df
#df.index = df.index.to_timestamp()
#df.index = pd.to_datetime(df.index)
#df.index = df.index.floor('T')
#df.index = df.index.to_pydatetime()
#data.index = pd.DatetimeIndex(data.index).to_period('D')

algo = rpt.KernelCPD(kernel='linear', min_size=100) # kernel options: linear,rbf,cosine
algo.fit(df.values)
result = algo.predict(n_bkps=8)

plt.figure(figsize=(12, 3))
plt.plot(df.to_numpy(), color='green')

def date2yday(x):
    """Convert matplotlib datenum to days since 2018-01-01."""
    y = x - mdates.date2num(datetime.datetime(2010, 1, 4))
    return y
def yday2date(x):
    """Return a matplotlib datenum for *x* days after 2018-01-01."""
    y = x + mdates.date2num(datetime.datetime(2010, 1, 4))
    return y
secax_x = plt.secondary_xaxis('top', functions=(date2yday, yday2date))
secax_x.set_xlabel('Date')

for bkp in result:
    plt.axvline(x=bkp, color='k', linestyle='--')
plt.title('Change point detection with linear kernel')
plt.xlabel('Days')
plt.ylabel('Price')
plt.grid(True)
plt.show()
#######################################
