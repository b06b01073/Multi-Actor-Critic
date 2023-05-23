import pandas as pd
import matplotlib.pyplot as plt
import math
from argparse import ArgumentParser


def baseline(args):
    history_data = pd.read_csv(args.data_path, index_col=0)
    date_mask = (history_data['Date'] >= args.start) & (history_data['Date'] <= args.end)
    
    history_data = history_data[date_mask].reset_index(drop=True)

    
    first_open_price = history_data.iloc[0]['Open']
    shares = args.asset / first_open_price
    earnings = []

    for i in range(len(history_data)):
        close_price = history_data.iloc[i]['Close']
        earnings.append(shares * close_price - args.asset)

    plt.plot(earnings)
    plt.title('Buy and hold')
    plt.savefig('img/baseline.jpg')


def MoneytoLot(args):
    '''
    Change Money to Lot
    '''
    TotalCost=args.FutureCost+args.FutureMaintainCost
    return math.floor(args.asset/TotalCost)

def TotalCost(args,Lot):
    '''
    Calculate Tax and Fee of all Lot
    input: # of Lot
    output: Tax and Fee
    '''
    Tax=args.FutureTax*Lot
    Fee=(args.FutureFee+args.FutureDFee)*Lot
    return Tax+Fee
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str, default='TX_data/TX_TI.csv')
    parser.add_argument('--start', '-s', type=str, default='2010-01-04')
    parser.add_argument('--end', '-e', type=str, default='2022-12-30')
    parser.add_argument('--asset', '-a', type=float, default=100000)
    parser.add_argument('--withasset', '-wa', type=bool, default=True)
    # future cost
    parser.add_argument('--FutureCost', '-FC', type=float, default=46000)
    parser.add_argument('--FutureMaintainCost','-FMC',type=float,default=35250)
    parser.add_argument('--FutureTax', '-FT', type=float, default=0.00002)
    parser.add_argument('--FutureFee', '-FF', type=float, default=12)
    parser.add_argument('--FutureDfee', '-FDF', type=float, default=8)
    args = parser.parse_args()

    baseline(args)