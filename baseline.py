import pandas as pd
import matplotlib.pyplot as plt

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



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str, default='dataset/^GSPC_2000-01-01_2022-12-31.csv')
    parser.add_argument('--start', '-s', type=str, default='2022-12-15')
    parser.add_argument('--end', '-e', type=str, default='2022-12-30')
    parser.add_argument('--asset', '-a', type=float, default=30000)
    
    args = parser.parse_args()

    baseline(args)