import os
import logging
import tempfile
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding
from sklearn.preprocessing import scale
import talib

from args import Args

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)

args = Args(ticker='USDT', agent='ddqn') #should eventually get rid of this
ticker = args.ticker

class DataLoader:

    def __init__(self, trading_days=args.trading_days, ticker=ticker, normalize=True, train=True):
        self.ticker = ticker
        self.trading_days = trading_days
        self.normalize = normalize
        self.min_values = None
        self.max_values = None
        
        if train == True:
            self.data_path = args.train_path
        else: 
            self.data_path = args.val_path

        self.price_data = None
        self.data = self.load_data()
        self.preprocess_data()
        self.step = 0
        self.offset = None
        self.train = train

    
    def load_closing_price(self):
        df = pd.read_csv(self.data_path).set_index('Date')
        df = df.dropna(subset=['Open', 'High']).drop(['Adj Close', 'Open'], axis=1)
        df.columns = ['high', 'low', 'close', 'volume']
        df = df[['close']]
        log.info(df.info())
        return df


    def load_data(self): #data from yahoo!finance
        log.info('loading data for {}...'.format(self.ticker))
        df = pd.read_csv(self.data_path).set_index('Date')
        df = df.dropna(subset=['Open', 'High']).drop(['Adj Close', 'Open'], axis=1)
        df.columns = ['high', 'low', 'close', 'volume']
        log.info('got data for {}...'.format(self.ticker))
        log.info(df.info())
        return df


    def preprocess_data(self):
        #percentage change of past 1, 7, and 14 days
        self.data['ret'] = self.data.close.pct_change()
        self.data['ret_7'] = self.data.close.pct_change(7)
        self.data['ret_14'] = self.data.close.pct_change(14)
        
        #techncial indicators 
        self.data['rsi'] = talib.STOCHRSI(self.data.close)[1]
        self.data['macd'] = talib.MACD(self.data.close)[1]
        self.data['atr'] = talib.ATR(self.data.high, self.data.low, self.data.close)
        self.data['obv'] = talib.OBV(self.data.close, self.data.volume)
        
        up, mid, low = talib.BBANDS(self.data.close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.data['bbands'] = (self.data.close - low) / (up - low)

        slowk, slowd = talib.STOCH(self.data.high, self.data.low, self.data.close)
        self.data['stoch'] = slowd - slowk

        self.data = (self.data.replace((np.inf, -np.inf), np.nan))
        
        self.price_data = (self.data[['close']].dropna())
        self.data = (self.data.drop(['high', 'low', 'close', 'volume'], axis=1).dropna())

        # normalize data except for daily ret
        ret_cpy = self.data.ret.copy()
        self.data = pd.DataFrame(scale(self.data), columns=self.data.columns, index=self.data.index)
        features = self.data.columns.drop('ret')
        self.data['ret'] = ret_cpy
        self.data = self.data.loc[:, ['ret'] + list(features)]
        
        self.min_values = self.data.min()
        self.max_values = self.data.max()

        log.info(self.data.info())

    def reset(self):
        if not self.train:
            self.offset = 0
            self.step = 0
        else: 
            high = len(self.data.index) - self.trading_days
            self.offset = np.random.randint(low=0, high=high)
            self.step = 0

    def take_step(self):
        curr_state = self.data.iloc[self.offset + self.step].values #state
        curr_price = self.price_data.iloc[self.offset + self.step].values
        self.step += 1
        done = self.step > self.trading_days
        return curr_state, curr_price, done


class TradingSimulator:

    def __init__(self, steps, transaction_cost, time_cost):
        # invariant for object life
        self.transaction_cost = transaction_cost
        self.time_cost = time_cost
        self.steps = steps

        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.capitals = np.ones(self.steps)
        self.shares = np.zeros(self.steps)
        self.prices = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.profits = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.markets = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.capitals.fill(1)
        self.shares.fill(0)
        self.prices.fill(0)
        self.costs.fill(0)
        self.profits.fill(0)
        self.trades.fill(0)
        self.markets.fill(0)

    def take_step(self, action, market, curr_price):
        prev_step = max(0, self.step - 1)
        prev_capital = self.capitals[prev_step]
        prev_shares = self.shares[prev_step]
        self.markets[self.step] = market #daily pct change 
        self.prices[self.step] = curr_price
        
        prev_action = self.actions[prev_step]
        self.actions[self.step] = action

        n_trades = action - prev_action
        self.trades[self.step] = n_trades

        transaction_costs = abs(n_trades) * self.transaction_cost
        time_cost = 0 if n_trades else self.time_cost
        self.costs[self.step] = transaction_costs + time_cost
        reward = (prev_action - 1) * market - self.costs[prev_step]
        self.profits[self.step] = reward

        if self.step != 0:
            if self.actions[self.step] == 0: # sell
                self.shares[self.step] = prev_shares * args.sell_frac
                self.capitals[self.step] = prev_capital + self.shares[self.step] * curr_price - self.shares[self.step] * transaction_costs

            elif self.actions[self.step] == 2: # buy
                self.shares[self.step] = prev_shares + (prev_capital * args.buy_frac / curr_price)
                self.capitals[self.step] = prev_capital * (1 - args.buy_frac) - self.shares[self.step] * transaction_costs
            
            else: #hold
                self.shares[self.step] = prev_shares
                self.capitals[self.step] = prev_capital
            # self.capitals[self.step] = prev_capital * (1 + self.profits[self.step]) # capital calculation might be wrong??
        # print("action: ", action, "curr share: ", self.shares[self.step], "curr capital: ", self.capitals[self.step])
        
        info = {'reward'  : reward,
                'capital'   : self.capitals[self.step],
                'costs'   : self.costs[self.step],
                'profit'  : self.profits[self.step],
                'price'   : self.prices[self.step],
                'shares'  : self.shares[self.step],
                }

        self.step += 1
        return reward, info

    def result(self):
        return pd.DataFrame({'action'         : self.actions,  # current action
                             'capital'          : self.capitals,  # starting capital
                             'market'         : self.markets,
                             'profit'         : self.profits,
                             'cost'           : self.costs,  # eod costs
                             'trade'          : self.trades,
                             'price'          : self.prices,
                             'shares'         : self.shares,
                             })  # eod trade)


class TradingEnvironment:
   
    def __init__(self,
                 trading_days=args.trading_days,
                 transaction_cost=args.transaction_cost,
                 time_cost=args.time_cost,
                 ticker=ticker,
                 train=True
                 ):
        self.trading_days = trading_days
        self.transaction_cost = transaction_cost
        self.ticker = ticker
        self.time_cost = time_cost
        self.train = train
        self.data_loader = DataLoader(trading_days=self.trading_days,
                                      ticker=ticker, train=self.train)
        self.simulator = TradingSimulator(steps=self.trading_days,
                                          transaction_cost=self.transaction_cost,
                                          time_cost=self.time_cost)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.data_loader.min_values.to_numpy(),
                                            self.data_loader.max_values.to_numpy())
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        observation, curr_price, done = self.data_loader.take_step()
        reward, info = self.simulator.take_step(action=action,
                                                market=observation[0], curr_price=curr_price) 
        return observation, reward, done, done, info

    def reset(self):
        self.data_loader.reset()
        self.simulator.reset()
        return self.data_loader.take_step()[0]