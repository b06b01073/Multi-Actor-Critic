# -*- coding: utf-8 -*-

import argparse
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta,datetime
from tqdm import tqdm

def Prophetic(file):  
    #read file
    df=pd.read_csv(file,parse_dates=True,index_col=0)
        
    #concate prophetic expert action
    df = pd.concat([df,pd.Series(name = 'phtAction', dtype = object)],axis=1) #add column 'phtAction'
    calendar = np.unique(df.Date) #list all the trading date
    for i in tqdm(range(len(calendar))): #for each trading day
     
        #todayMarket = df[mask]['Open'] #extract today's open price
        todayOpen = df['Open'][i] #extract today's open price
        todayClose = df['Close'][i] #extract today's open price
        #phtAction = todayMarket.copy().rename("phtAction") #to record today's prophetic expert action
        
        # greedy : open < close => long, open > close => short
        if todayOpen < todayClose:
            df['phtAction'][i] = 1
        else:
            df['phtAction'][i] = -1
        
        df.update(df['phtAction']) #save result

    # df.to_csv('prophetic_0616.csv',index=False)
    df.to_csv('prophetic.csv',index=False)
    
    # nColumns=['open','close','high','low','volume',
    #           'MACD','EMA_7','EMA_21','EMA_56','RSI',
    #           'BB_up','BB_mid','BB_mid','BB_low','slowK','slowD']

    # Date,type,Month,Open,High,Low,Close,Price change,Price change Ratio,Volume,Final Price,# of OC,Final BBP,Final BSP,Hitorical HP,Historical LP,MA5,MA10,MA20,BBAND5UP,BBAND5LOW,BBAND10UP,BBAND10LOW,BBAND20UP,BBAND20LOW
    nColumns = ['Open','High','Low','Close','Volume','MA5','MA10','MA20','BBAND5UP','BBAND5LOW','BBAND10UP','BBAND10LOW','BBAND20UP','BBAND20LOW']
    for cn in nColumns:
        newLabel = "norm{}".format(cn.capitalize())
        df[newLabel] = df[cn]
        df[newLabel] = 2*((df[newLabel] - df[newLabel].min()) / (df[newLabel].max() - df[newLabel].min())) -1
        
    #deal with the datetime column
    df=df.reset_index()#['date']=pd.to_datetime(df['date'])
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    return df

# fn = 'IC_tech_oriDT_0627.csv'
fn  = 'TX_TI.csv'
Prophetic(fn)
