import pandas as pd
import matplotlib.pyplot as plt
import math
from argparse import ArgumentParser
from datetime import datetime

def baseline(args,start,end):

    history_data = pd.read_csv(args.data_path, index_col=0)
    date_mask = (history_data['Date'] >= start) & (history_data['Date'] <= end)
    
    history_data = history_data[date_mask].reset_index(drop=True)

    
    first_open_price = history_data.iloc[0]['Open']
    shares = args.asset / 46000
    L_earnings = [args.asset]
    S_earnings = [args.asset]

    for i in range(len(history_data)):
        close_price = history_data.iloc[i]['Close']
        L_earnings.append(shares * 50 * (close_price - first_open_price)+args.asset)
        S_earnings.append(shares * 50 * (first_open_price - close_price)+args.asset)

    #plt.plot(earnings)
    #plt.title('Buy and hold')
    #plt.savefig('img/baseline.jpg')
    return(L_earnings,S_earnings)
def MoneytoLot(args,assets):
    '''
    Change Money to Lot
    '''
    TotalCost=args.FutureCost+args.FutureMaintainCost
    return math.floor(assets/TotalCost)

def TC(args,Lot,DDay):
    '''
    Calculate Tax and Fee of all Lot
    input: # of Lot
    output: Tax and Fee
    '''
    Tax=args.FutureTax*Lot
    Fee=0
    if DDay:
        Fee=(args.FutureFee+args.FutureDfee)*Lot
    else:
        Fee=args.FutureFee*Lot
    return Tax+Fee
def baselinewithasset(args):
    history_data=pd.read_csv(args.data_path, index_col=0)
    date_mask = (history_data['Date'] >= args.start) & (history_data['Date'] <= args.end)
    history_data=history_data[date_mask].reset_index(drop=True)
    first_open_price = history_data.iloc[0]['Open']
    assets=args.asset
    Lot=MoneytoLot(args,assets)
    Fee=TC(args,Lot,False)
    print(Lot,Fee)
    cur_trade_day=0
    earnings=[]
    TotalCost=Fee
    while cur_trade_day< len(history_data):
        temp=pd.Timestamp(history_data['Date'][cur_trade_day])
        FirstDayMonth=datetime(temp.year,temp.month,1)
        close_price=history_data.iloc[cur_trade_day]['Close']
       
        #over maintain future cost
        if close_price-first_open_price<-215: 
            print("Over maintain future cost in "+ str(temp))
            
        
        # Due Day calculation
        if temp.isocalendar()[1]-FirstDayMonth.isocalendar()[1]+1 == 3 and temp.isocalendar()[2]==2:
            assets=assets+((close_price-first_open_price)*Lot-TotalCost)
            Lot=MoneytoLot(args,assets)
            Fee=TC(args,Lot,True)
            TotalCost=Fee
            earnings.append((close_price-first_open_price)*Lot)
            first_open_price=history_data.iloc[cur_trade_day+1]['Open']
        else:
            earnings.append((close_price-first_open_price)*Lot) 
        cur_trade_day+=1
    
    plt.plot(earnings)
    plt.title('Buy and hold with assets')
    plt.savefig('img/baselinewithassets.jpg')
    

    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str, default='TX_data/TX_TI.csv')
    parser.add_argument('--start', '-s', type=str, default='2010-01-04')
    parser.add_argument('--end', '-e', type=str, default='2022-12-30')
    parser.add_argument('--asset', '-a', type=float, default=1000000)
    parser.add_argument('--withasset', '-wa', type=bool, default=True)
    # future cost
    parser.add_argument('--FutureCost', '-FC', type=float, default=46000)
    parser.add_argument('--FutureMaintainCost','-FMC',type=float,default=35250)
    parser.add_argument('--FutureTax', '-FT', type=float, default=0.00002)
    parser.add_argument('--FutureFee', '-FF', type=float, default=12)
    parser.add_argument('--FutureDfee', '-FDF', type=float, default=8)
    args = parser.parse_args()
    if args.withasset:
        baselinewithasset(args)
    else:
        baseline(args)