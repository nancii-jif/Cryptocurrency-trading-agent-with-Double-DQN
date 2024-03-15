# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import os

from time import time
from collections import deque
from random import sample

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import tensorflow as tf

from agent import DDQNAgent, DQNAgent
from env import TradingEnvironment
from args import Args

np.random.seed(42)
tf.random.set_seed(42)
sns.set_style('whitegrid')

"""
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')
"""

def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)

args = Args(ticker='BTC', agent='ddqn', max_episodes=300, buy_frac = 0.5, sell_frac = 0.5)

ticker = args.ticker #'ETH', 'MATIC', 'DOGE's
train_data = args.train_path
val_data = args.val_path
results_path = args.results_path


training_env = TradingEnvironment(trading_days=args.trading_days, transaction_cost=args.transaction_cost,
                               time_cost=args.time_cost, train=True)
training_env.seed(42)

eval_env = TradingEnvironment(trading_days=args.trading_days, transaction_cost=args.transaction_cost,
                               time_cost=args.time_cost, train=False)

state_dim = training_env.observation_space.shape[0]
n_actions = training_env.action_space.n
max_episode_steps = training_env.trading_days
max_episodes = args.max_episodes

tf.keras.backend.clear_session()

if args.agent == 'dqn':
    q_agent = DQNAgent(state_dim=state_dim,
                 n_actions=n_actions,
                 lr=args.lr,
                 discount_factor=args.discount_factor,
                 eps_start=args.eps_start,
                 eps_end=args.eps_end,
                 eps_decay_steps=args.eps_decay_steps,
                 epsilon_exponential_decay=args.epsilon_exponential_decay,
                 replay_capacity=args.replay_capacity,
                 architecture=args.architecture,
                 l2_reg=args.l2_reg,
                 update_frequency=args.update_frequency,
                 batch_size=args.batch_size)
else: 
    q_agent = DDQNAgent(state_dim=state_dim,
                 n_actions=n_actions,
                 lr=args.lr,
                 discount_factor=args.discount_factor,
                 eps_start=args.eps_start,
                 eps_end=args.eps_end,
                 eps_decay_steps=args.eps_decay_steps,
                 epsilon_exponential_decay=args.epsilon_exponential_decay,
                 replay_capacity=args.replay_capacity,
                 architecture=args.architecture,
                 l2_reg=args.l2_reg,
                 update_frequency=args.update_frequency,
                 batch_size=args.batch_size)


q_agent.online_network.summary()
total_steps = 0
episode_time, assets, sharpes, losses, episode_eps = [], [], [], [], []

def track_results(episode, asset_ma_50, asset_ma_10,
                  sharpe_ma_50, sharpe_ma_10,
                  total, epsilon):

    template = 'ep: {:>3d} --- {} --- Excess Return: {:>4.3f} ({:>4.3f}) --- Sharpe Ratio: {:>4.2f} ({:>4.2f}) --- epsilon: {:>4.3f}'
    entry = template.format(episode, format_time(total),
                          asset_ma_50 - 1, asset_ma_10 - 1,
                          sharpe_ma_50, sharpe_ma_10,
                          epsilon)
    print(entry)
    print(" ")

def trainer(max_episodes):
    start = time()
    results = []
    for episode in range(1, max_episodes + 1):
        this_state = training_env.reset()
        for _ in range(max_episode_steps):
            action = q_agent.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
            next_state, reward, _, done, _ = training_env.step(action)

            q_agent.memorize_transition(this_state,
                                    action,
                                    reward,
                                    next_state,
                                    0.0 if done else 1.0)
            if q_agent.train:
                q_agent.experience_replay()
            if done: 
                break
            this_state = next_state

        result = training_env.simulator.result() # .result() is not the state
        result['asset'] = result.capital + result.shares * result.price 

        final = result.iloc[-1]
        total_asset = final.asset
        assets.append(total_asset)

        sharpe = (np.mean(result.asset) - 1) / np.std(result.asset) 
        sharpes.append(sharpe)

        loss = q_agent.losses[-1]
        losses.append(loss)

        if episode % 10 == 0:
            track_results(episode,
                        np.mean(assets[-50:]),
                        np.mean(assets[-10:]),
                        np.mean(sharpes[-50:]),
                        np.mean(sharpes[-10:]),
                        time() - start, q_agent.epsilon)
            
    results = pd.DataFrame({'Episode': list(range(1, episode + 1)),
                        'Asset': assets,
                        'Sharpe' : sharpes,
                        'Losses' : losses,
                        }).set_index('Episode')
    return results

def evaluator():
    this_state = eval_env.reset()
    for _ in range(max_episode_steps):
        action = q_agent.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
        next_state, reward, _, done, _ = eval_env.step(action)

        q_agent.memorize_transition(this_state, action, reward, next_state, 0.0 if done else 1.0)
        if q_agent.train:
            q_agent.experience_replay()
        if done: 
            break
        this_state = next_state

    result = eval_env.simulator.result() # .result() is not the state
    result['asset'] = result.capital + result.shares * result.price 
    
    final = result.iloc[-1]
    total_asset = final.asset
    sharpe = (np.mean(result.asset) - 1) / np.std(result.asset) 

    return result, total_asset, sharpe

train_results = []
train_results = trainer(max_episodes)
train_results.to_csv(os.path.join(results_path, 'train_results.csv'), index=False)

val_results, final_val_asset, final_val_sharpe = evaluator()
val_results.to_csv(os.path.join(results_path, 'val_results.csv'), index=False)



with sns.axes_style('white'):
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(14, 4), sharey=False)  # Disable sharing y-axis
    
    # Plot training Sharpe performance
    sharpe_df = (train_results[['Sharpe']].rolling(50).mean())
    sharpe_df.plot(ax=axes[0, 0], title='Train Sharpe Ratio', ylabel='Sharpe Ratio', legend=False)

    loss_df = (train_results[['Losses']].rolling(10).mean())
    loss_df.plot(ax=axes[1, 0], title='Train Loss', ylabel='Loss', legend=False)

     # Plot training asset performance
    asset_df = (train_results[['Asset']].sub(1).rolling(50).mean())
    asset_df.plot(ax=axes[0, 1], title='Train Excess Return', ylabel='Excess Return', legend=False)
    
    val_asset_df = (val_results[['asset']].sub(1).rolling(10).mean())
    val_asset_df.plot(ax=axes[1, 1], title='Validation Excess Return', xlabel='Day', ylabel='Excess Return', legend=False)
    
    sns.despine()
    fig.tight_layout()
    
    # Save the plot
    fig.savefig(os.path.join(results_path, 'performance.png'), dpi=500)
    plt.close(fig)
