import os



class Args():

    def __init__(self, ticker='BTC', agent='dqn', max_episodes = 300, buy_frac = 0.2, sell_frac = 0.5):
        self.ticker = ticker
        self.agent = agent
        self.max_episodes = max_episodes
        self.buy_frac = buy_frac
        self.sell_frac = sell_frac
        
        self.train_path = os.path.join('./datasets', self.ticker, 'train.csv')
        self.val_path = os.path.join('./datasets', self.ticker, 'val.csv')
        results_path = os.path.join('./results/', ticker, agent)
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        self.results_path = results_path

        self.trading_days = 365
        self.transaction_cost = 1e-3
        self.time_cost = 1e-4

        self.discount_factor = .99
        self.update_frequency = 50

        self.architecture = (256, 256)  # units per layer
        self.lr = 0.0001  # learning rate
        self.l2_reg = 1e-6  # L2 regularization
        self.replay_capacity = int(1e6)
        self.batch_size = 32
        self.eps_start = 1.0
        self.eps_end = .01
        self.eps_decay_steps = 250
        self.epsilon_exponential_decay = .99
