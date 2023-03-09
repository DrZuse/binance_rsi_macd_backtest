import datetime
import numpy as np
import os
import pandas as pd

from configurations import setup_logger, basic_parameters as bp
from multiprocessing import Pool, cpu_count

logger = setup_logger('read_csvs')

names = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']



#----- Lists of files 

csv_dir = '../big_dataframes/binance/spot/daily/klines/BTCUSDT/1s/'
theOrderBookFiles = sorted(os.listdir(csv_dir))

def read_csv_inparalel(file):
    return pd.read_csv(csv_dir+file, names=names, usecols=['Open time', 'Open', 'High', 'Low', 'Close'])
    

logger.info('start read csvs')
with Pool(cpu_count()) as p:
    #files = p.map(read_csv_inparalel, theOrderBookFiles)
    files = p.map(read_csv_inparalel, theOrderBookFiles[-1:]) # TEST 3 files
logger.info('finish read csvs')

df = pd.concat(files, ignore_index = True)

#df['time UTC'] = pd.to_datetime(df['Open time'], unit='ms', origin='unix')

logger.info(df.shape)

def computeRSI (data, time_window):
    diff = data.diff(1).dropna()        # diff in one field(one day)

    #this preservers dimensions off diff values
    up_chg = 0 * diff
    down_chg = 0 * diff
    
    # up change is equal to the positive difference, otherwise equal to zero
    up_chg[diff > 0] = diff[ diff>0 ]
    
    # down change is equal to negative deifference, otherwise equal to zero
    down_chg[diff < 0] = diff[ diff < 0 ]
    
    # check pandas documentation for ewm
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
    # values are related to exponential decay
    # we set com=time_window-1 so we get decay alpha=1/time_window
    up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
    
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    return rsi




df['RSI'] = computeRSI(df['Close'], 14)

# # Calculate MACD values using the pandas_ta library
# df.ta.macd(close='close', fast=12, slow=26, signal=9, append=True)
# Get the 26-day EMA of the closing price
k = df['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
# Get the 12-day EMA of the closing price
d = df['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
# Subtract the 26-day EMA from the 12-Day EMA to get the MACD
macd = k - d
# Get the 9-Day EMA of the MACD for the Trigger line
macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
# Calculate the difference between the MACD - Trigger for the Convergence/Divergence value
macd_h = macd - macd_s
# Add all of our new values for the MACD to the dataframe
#df['macd'] = df.index.map(macd)
df['macd_h'] = df.index.map(macd_h)
#df['macd_s'] = df.index.map(macd_s)

df.dropna(inplace=True)

#logger.info(df.head())


df.loc[(df['RSI'] <= 25) & (df['RSI'].shift(1) <= 25), '2RSI<25'] = True


logger.info(df.tail(50))

dfnp = np.array(df.values)
dfnp.shape
logger.info(dfnp.shape)
logger.info(dfnp)

wait_rsi25_signal = False
profits = 0
losses = 0

for i, dfnp_row_i in enumerate(dfnp):
    
    #print(dfnp_row_i)
    #print(i)
    if dfnp_row_i[7] == True: # 2RSI<25
        time_2rsi_lower_25 = dfnp_row_i[0]
        logger.info(f'2RSI<25: {dfnp_row_i[7]} {datetime.datetime.utcfromtimestamp(time_2rsi_lower_25/1000)}')

        wait_rsi25_signal = True

        for j, dfnp_row_j in enumerate(dfnp[(i+1):]):
            #print(j)
            rsi = dfnp_row_j[5]
            print(f'rsi: {rsi}')
            if rsi > 25:
                print('rsi > 25')

                for k, dfnp_row_k in enumerate(dfnp[(i+1)+j:]):
                    macd = dfnp_row_k[6]
                    #print(f'macd: {dfnp_row_k[6]}')
                    if macd >= 0:
                        print(f'macd: {dfnp_row_k[6]}')
                        print('buy!')

                        for l, dfnp_row_l in enumerate(dfnp[(i+1)+j+(k+1):]):
                            print(dfnp_row_l)
                            print(f'open: {dfnp_row_l[1]} | high: {dfnp_row_l[2]} | low: {dfnp_row_l[3]} | close: {dfnp_row_l[4]}')
                            buy_price = dfnp_row_l[1]
                            max_high = dfnp_row_l[2]
                            max_low = dfnp_row_l[3]
                            loss = 0
                            profit = 0
                            print(f'buy_price: {buy_price}')

                            for m, dfnp_row_m in enumerate(dfnp[(i+1)+j+(k+1)+(l+1):]):


                                if dfnp_row_m[3] > max_high:
                                    max_high = dfnp_row_m[3]
                                    profit = (max_high - buy_price) * 100 / buy_price

                                if dfnp_row_m[4] < max_low:
                                    max_low = dfnp_row_m[4]
                                    loss = (buy_price - max_low) * 100 / buy_price

                                if profit >= 0.02:
                                    profits += 1

                                if loss >= 0.02:
                                    losses += 1

                                if profit >= 0.02 or loss >= 0.02:
                                    print(f'profit: {profit} || loss: {loss}')
                                    print(f'max_high: {max_high} || max_low: {max_low}')
                                    print(f'time {datetime.datetime.utcfromtimestamp(dfnp_row_m[0]/1000)}')

                                    break
                            break
                        break
                break

logger.info(f'profits: {profits} || losses: {losses}')