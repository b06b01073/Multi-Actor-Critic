def get_MA(history_price, interval):
    ''' calculate the moving average(MA) of the dataframe in the given windowsize(interval)

    Arguments:
        history_price(pd dataframe): the history price data
        interval(int): the window size of the MA 
    '''
    history_price[f'MA{interval}'] = history_price['Open'].rolling(window=interval).mean()

def get_std(history_price, interval):
    ''' calculate the standard deviation(STD) of the dataframe in the given windowsize(interval)

    Arguments:
        history_price(pd dataframe): the history price data
        interval(int): the window size of the STD 
    '''
    history_price[f'STD{interval}'] = history_price['Open'].rolling(window=interval).std()