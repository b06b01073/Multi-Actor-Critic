import pandas as pd
import mplfinance as mpf
from collections import defaultdict
import os

class StockMarket:
    '''A class to represent the stock market 

    Attributes:
        data_interval(int): the number of entries in one state(default=10, state consists of the data from the last `data_interval` trade days)
        history_data(pd dataframe): the entire stock history data from the crawled csv file
        dataset(pd dataframe): a subset of the dataframe, which is the dataframe in a specific interval
        init_state(pd dataframe): the first state from the dataset
        cur_trade_day(int): the current trade day, the first trade day is 0
        terminated(bool): True when the current trade day is the last trade day
        info(dict): store the value of the information of the environment(debug only)'
        cur_state(pd dataframe): the current state
        first_trade_date(str): date in yyyy-mm-dd format(logging only), the first trade date in the dataset
        last_trade_date(str): date in yyyy-mm-dd format(logging only), the last trade date in the dataset      
    '''
    def __init__(self, csv_path, start, end, data_interval=10):
        ''' initialze the env

        The attribute is commented at the beginning of the class

        Arguments:
            csv_path(str): the path to the csv file that stores the stock history data(should be in the dataset directory)
            start(str): YYYY-MM-DD format(note that the format must be followd strictly, 2010-06-01 is not the same as 2010-6-1), the first day(note that the first day may not be the same as the first trading day)
            end(str): YYYY-MM-DD format(note that the format must be followd strictly, 2010-06-01 is not the same as 2010-6-1), the last day(note that the last day may not be the same as the last trading day)
            data_interval(int): data_interval(int): the number of entries in one state(default=10, state consists of the data from the last `data_interval` trade days)
        '''

        self.data_interval = data_interval
        self.history_data = pd.read_csv(csv_path, index_col=0)
        self.dataset, self.init_state = self.__get_dataset(start, end)
        self.cur_trade_day = 0
        self.terminated = False
        self.info = defaultdict()
        self.cur_state = None

        self.first_trade_date = self.dataset.iloc[0]['Date'] # for logging only 
        self.last_trade_date = self.dataset.iloc[len(self.dataset) - 1]['Date'] # for logging only
        self.__log_init_info()
        self.__plot_candles()

    def __get_dataset(self, start, end):
        ''' get the dataset and return the first state of the environment

        The dataset is the subset of self.history_data(the entire dataset) which falls in the range [start, end]

        Arguments:
            start(str): YYYY-MM-DD format(note that the format must be followd strictly, 2010-06-01 is not the same as 2010-6-1), the first day(note that the first day may not be the same as the first trading day)
            end(str): YYYY-MM-DD format(note that the format must be followd strictly, 2010-06-01 is not the same as 2010-6-1), the last day(note that the last day may not be the same as the last trading day)

        Returns:
            It first return the dataset that is in the desired range(specified by the start and end argument), then, it return the first state
        '''

        first_date_index = self.history_data[self.history_data['Date'] >= start].index[0]
        init_state = self.history_data[first_date_index-self.data_interval+1:first_date_index+1]
        date_mask = (self.history_data['Date'] >= start) & (self.history_data['Date'] <= end)
        return self.history_data[date_mask].reset_index(drop=True), init_state.reset_index(drop=True)
    
    def reset(self):
        ''' Reset the internal state of the environment

        Returns:
            return the first state, and the info
        '''
        self.cur_trade_day = 0
        self.cur_state = self.init_state
        self.terminated = False

        return self.init_state, self.__set_info()

    def step(self, action):
        ''' The environment recieves the action from agent and returns the reward and the next state

        The environment calculate the reward function and decide the state transition based on the cur_trade_day and the action from the agent

        Arguments:
            action(float): should be in the [-1, 1], the action from the agent

        Returns:
            self.cur_state(dataframe): the next state
            reward(float): the reward of the action
            self.terminated(bool): whether the state is a terminated state(the last trade day)
            info: info 
        '''

        assert not self.terminated, 'The environment has terminated(passed the last trade day), please call the reset method and start the next episode'
        assert action >= -1.0 and action <= 1.0, f'action out of range(should be in [-1, 1] but recieved {action})'
            
        # calcualte reward
        reward = self.__reward_function(action)

        # state transition
        self.cur_state = self.__state_transition()
        self.cur_trade_day += 1
        if self.cur_trade_day >= len(self.dataset):
            self.terminated = True

        # update info
        info = None if self.terminated else self.__set_info()

        return self.cur_state, reward, self.terminated, info

    def __reward_function(self, action):
        '''calculate the reward based on the given action and the stock price increase rate
        
        Arguments:
            action(float): the agent's action(should be in [-1, 1])

        Return:
            return the calculated reward
        '''
        cur_day_data = self.dataset.iloc[self.cur_trade_day]
        increase_rate = (cur_day_data['Close'] - cur_day_data['Open']) / cur_day_data['Open']

        return increase_rate * 100 * action 


    def __state_transition(self):
        '''state transition using the sliding window approach

        The first row(oldest) of data is dropped, and a new row(latest) of data is appended

        Returns:
            return the next state
        '''

        next_day = self.cur_trade_day + 1
        if next_day >= len(self.dataset):  # there is no next day(cur day is the last day)
            return None
        
        new_state = pd.concat([self.cur_state, pd.DataFrame([self.dataset.iloc[next_day]])], ignore_index=True).tail(-1)

        return new_state


    def __set_info(self):
        ''' update the info attribute
        
        Return:
            return the updated info
        '''

        self.info['cur_trade_day'] = self.cur_trade_day
        self.info['cur_date'] = self.dataset.iloc[self.cur_trade_day]['Date']

        return self.info

    def __log_init_info(self):
        ''' print the first and the last trade day
        '''
        print(f'first trade date: {self.first_trade_date}, last trade date: {self.last_trade_date}, total entries: {len(self.dataset)}')

    def __plot_candles(self, filepath='trend.jpg'):
        '''plot the candle sticks of the self.dataset dataframe

        Arguments:
            filepath(string): the path that the plot is going to be stored
        '''
        img_path = 'img'
        if not os.path.exists(img_path):
            os.makedirs(img_path)


        plt_frame = self.dataset
        plt_frame.index = pd.DatetimeIndex(plt_frame['Date']) 
        mpf.plot(plt_frame, type='candle', style='yahoo', volume=True, savefig=os.path.join(img_path, filepath), warn_too_much_data=4000)

        print(f'Data plotted in {filepath}')

def make(csv_path, start, end):
    ''' create the stock market environment

    initialize the stock market environment by passing the csv_path, start and end to it

    Arguments:
        

    Return:
        returns the StockMarket class instance
    '''
    return StockMarket(csv_path, start, end)

if __name__ == '__main__':
    make('dataset/^GSPC_2000-01-01_2022-12-31.csv', start='2022-12-15', end='2022-12-30')
