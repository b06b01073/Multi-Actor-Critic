import pandas as pd
import mplfinance as mpf
from collections import defaultdict
import math
from datetime import datetime
import os
import numpy as np

class StockMarket:
    '''A class to represent the stock market 

    Attributes:
        data_interval(int): the number of entries in one state(default=2, state consists of the data from the last `data_interval` trade days)
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
    def __init__(self, args):
        ''' initialze the env

        The attribute is commented at the beginning of the class

        Arguments:
            csv_path(str): the path to the csv file that stores the stock history data(should be in the dataset directory)
            start(str): YYYY-MM-DD format(note that the format must be followd strictly, 2010-06-01 is not the same as 2010-6-1), the first day(note that the first day may not be the same as the first trading day)
            end(str): YYYY-MM-DD format(note that the format must be followd strictly, 2010-06-01 is not the same as 2010-6-1), the last day(note that the last day may not be the same as the last trading day)
            data_interval(int): data_interval(int): the number of entries in one state(default=2, state consists of the data from the last `data_interval` trade days)
        '''

        self.data_interval = args.data_interval
        self.history_data = pd.read_csv(args.csv_path, index_col=0)
        self.dataset, self.init_state = self.__get_dataset(args.start, args.end)
        self.cur_trade_day = 0
        self.terminated = False
        self.info = defaultdict()
        self.cur_state = None

        self.first_trade_date = self.dataset.iloc[0]['Date'] # for logging only 
        self.last_trade_date = self.dataset.iloc[len(self.dataset) - 1]['Date'] # for logging only
        self.__log_init_info()
        self.__plot_candles()
        
        self.FutureCost=args.FutureCost
        self.FutureFee=args.FutureFee
        self.FutureDFee=args.FutureDFee
        self.FutureTax=args.FutureTax

        # Behavior Cloning
        file = 'TX_data/prophetic.csv'
        self.df=pd.read_csv(file,parse_dates=True,index_col=0)
        self.is_BClone = args.is_BClone

        ### DSR parameters ### (DSR: Differential Sharp Ratio)
        self.R_max = args.Reward_max_clip
        self.At0 = 0
        self.Bt0 = 0
        self.eta = 1/100000 
        self.SRt0 = 0
        # 
    def __get_dataset(self, start, end):
        ''' get the dataset and return the first state of the environment

        Pitfall: Do not use the data other than open price in the last entry in the init state

        The dataset is the subset of self.history_data(the entire dataset) which falls in the range [start, end]

        Arguments:
            start(str): YYYY-MM-DD format(note that the format must be followd strictly, 2010-06-01 is not the same as 2010-6-1), the first day(note that the first day may not be the same as the first trading day)
            end(str): YYYY-MM-DD format(note that the format must be followd strictly, 2010-06-01 is not the same as 2010-6-1), the last day(note that the last day may not be the same as the last trading day)

        Returns:
            It first return the dataset that is in the desired range(specified by the start and end argument), then, it return the first state
        '''

        first_date_index = self.history_data[self.history_data['Date'] >= start].index[0]
        print(first_date_index)
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

    def step(self, action, invested_asset):
        ''' The environment recieves the action from agent and returns the reward and the next state

        The environment calculate the reward function and decide the state transition based on the cur_trade_day and the action from the agent

        Arguments:
            action(float): should be in the [-1, 1], the action from the agent
            invested_asset(float): the agent's investment

        Returns:
            self.cur_state(dataframe): the next state
            reward(float): the reward of the action
            self.terminated(bool): whether the state is a terminated state(the last trade day)
            earning(float): the agent's earning
            info: info 
        '''

        assert not self.terminated, 'The environment has terminated(passed the last trade day), please call the reset method and start the next episode'
        assert action >= -1.0 and action <= 1.0, f'action out of range(should be in [-1, 1] but recieved {action})'
        assert invested_asset > 0

        # Behavior Cloning
        # no attribute self.is_BClone
        if self.is_BClone == True:
            action_bc = self.df['phtAction'][self.cur_trade_day]
            # if action_bc==0 and self.is_PER_replay:
            #     action_bc=random.choice([1,-1]) #radnomly choose an action
            # if action_bc==-1:
            #     action_bc = np.array([1., 0.])
            # elif action_bc==1:
            #     action_bc = np.array([0., 1.])
        else:
            action_bc = None

        # calcualte reward
        reward, earning = self.__reward_function(action, invested_asset)

        # state transition
        self.cur_state = self.__state_transition()
        self.cur_trade_day += 1
        if self.cur_trade_day >= len(self.dataset):
            self.terminated = True

        # update info
        info = None if self.terminated else self.__set_info()

        return action_bc, self.cur_state, reward, self.terminated, earning, info

    def __reward_function(self, action, invested_asset):
        '''calculate the reward based on the given action and the stock price increase rate
        
        Arguments:
            action(float): the agent's action(should be in [-1, 1])
            invested_asset(float): the agent's investment 

        Return:
            (i)reward according to the reward function specified in spec
            (ii)the agent's earning
        '''
        assert invested_asset > 0

        cur_day_data = self.dataset.iloc[self.cur_trade_day]
        close_price, open_price = cur_day_data['Close'], cur_day_data['Open']
        increase = (close_price - open_price) > 0

        
        B = 0
        if action > 0: 
            B = 1
        elif action == 0:
            B = 0
        else:
            B = -1
        price_change = close_price - open_price
        Lot = self.money_to_lot(invested_asset)
        final_price = Lot * price_change
        earning = final_price * 50 * B
        TransactionFee = self.FeeCalculation(Lot)

        return np.clip(float(cur_day_data['Price change Ratio'][:-1]) * action, a_min=-0.6, a_max=0.6), earning - TransactionFee


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

    def money_to_lot(self, invested_asset):
        '''
        calculate the number of Lot Future 
        input(float): invested money
        output(float): Lot(s) of future (only purchase the Lot of Future that the Cost is less than invested_asset)
        '''
        future_cost = 23000
        return invested_asset // future_cost
    
    def FeeCalculation(self,Lot):
        '''
        calculate the transaction Fee include the Tax and Fee
        input: Lots of Future
        output: Transaction Fee
        '''
        Tax=0.00002
        Fee=12
        DFee=8
        TotalCost=0
        FutureCost=23000*Lot
        FutureTax=FutureCost*Tax
        temp=pd.Timestamp(self.history_data['Date'][self.cur_trade_day])
        FirstDayMonth=datetime(temp.year,temp.month,1)
        if temp.isocalendar()[1]-FirstDayMonth.isocalendar()[1]+1 == 3 and temp.isocalendar()[2]==2:
            TotalCost=FutureTax+(Fee+DFee)*Lot
        else:
            TotalCost=FutureTax+Fee*Lot*2
        return math.ceil(TotalCost)
    

    def __len__(self):
        '''
        return the len of the dataset
        '''
        return self.dataset.shape[0]
    
def make(is_BClone, csv_path, start, end, FutureCost, FutureFee, FutureDFee, FutureTax, data_interval):
    ''' create the stock market environment

    initialize the stock market environment by passing the csv_path, start and end to it

    Arguments:
        csv_path(str): the csv file path of the stock history data

        start(str): Format: YYYY-MM-DD. the data start from the specified start time. Note that the start time is not equivalent to first trade day.

        end(str): Format: YYYY-MM-DD. the data end at the specified end time. Note that the end time is not equivalent to last trade day.

    Return:
        returns the StockMarket class instance
    '''
    return StockMarket(is_BClone, csv_path, start, end, FutureCost, FutureFee, FutureDFee, FutureTax, data_interval)
