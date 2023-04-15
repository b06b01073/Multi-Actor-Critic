import argparse
import yfinance as yf
import os


def crawl(args):
    ''' Crawl the data from yahoo finance using the yfinance package

    Crawl the data from yahoo finance using the yfinance package and save the crawled data in a csv file 

    Arguments:
        args(argparse.Namespace): an instance of argparse.Namespace class, use `python crawl.py --help` for more info 
    '''
    company = yf.Ticker(args.ticker)
    history_price = company.history(start=args.start, end=args.end, interval=args.interval).reset_index()

    cleanse_hour(history_price)
    log_crawled_data(history_price)

    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    history_price.to_csv(f'{dataset_path}/{args.ticker}_{args.start}_{args.end}.csv')

def cleanse_hour(history_price):
    '''remove the hour stamp in the Date column in the dataframe

    For example:
        2000-01-03 00:00:00-05:00 -> 2000-01-03

    Arguments:
        history_price(pd dataframe): the crawled data in the csv file
    '''
    history_price['Date'] = history_price['Date'].apply(lambda x: str(x).split()[0])

def log_crawled_data(history_price):
    ''' Log the metadata of the crawled data

    Arguments:
        history_price(pd dataframe): the crawled data in dataframe
    '''
    print(f'Total trading days: {len(history_price.index)}')
    print(f'First trading day: {history_price.iloc[0]["Date"]}')
    print(f'Last trading day: {history_price.iloc[len(history_price.index) - 1]["Date"]}')
    print(f'Crawled data: {history_price.columns.values}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', '-t', help='Choose the ticker of the company of interest. Default: ^GSPC(S&P500)', type=str, default='^GSPC')
    parser.add_argument('--interval', '-i', help='Data interval. Valid intervals are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo.\n Default: 1d', type=str, default='1d')
    parser.add_argument('--start', '-s', help='Format: YYYY-MM-DD. Download data start from the specified start time. Default: 2000-01-01', type=str, default='2000-01-01')
    parser.add_argument('--end', '-e', help='Format: YYYY-MM-DD. Download data til the specified end time. Default: 2022-12-31', type=str, default='2022-12-31')

    args = parser.parse_args()

    crawl(args)