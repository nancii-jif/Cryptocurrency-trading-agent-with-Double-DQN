# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from time import time
from collections import deque
from random import sample

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')

from tensorflow import keras

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2



class DDQNAgent:
    def __init__(self, state_dim,
                 n_actions,
                 lr,
                 discount_factor,
                 eps_start,
                 eps_end,
                 eps_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 architecture,
                 l2_reg,
                 update_frequency,
                 batch_size):

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.lr = lr
        self.discount_factor = discount_factor
        self.architecture = architecture
        self.l2_reg = l2_reg

        self.online_network = self.build_network()
        self.target_network = self.build_network(trainable=False)
        self.update_target()

        self.epsilon = eps_start
        self.eps_decay_steps = eps_decay_steps
        self.epsilon_decay = (eps_start - eps_end) / eps_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.losses = []
        self.idx = tf.range(batch_size)
        self.train = True

    def build_network(self, trainable=True):
        layers = []
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None,
                                activation='relu',
                                kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        layers.append(Dropout(.1))
        layers.append(Dense(units=self.n_actions,
                            trainable=trainable,
                            name='Output'))
        model = Sequential(layers)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.lr))
        return model

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights())

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_actions)
        q = self.online_network.predict(state)
        return np.argmax(q, axis=1).squeeze()

    def memorize_transition(self, state, action, reward, next_state, not_done):
        if not_done:
            self.episode_reward += reward
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.eps_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((state, action, reward, next_state, not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        next_q = self.online_network.predict_on_batch(next_states)
        next_actions = tf.argmax(next_q, axis=1)
        next_target_q = self.target_network.predict_on_batch(next_states)
        target_q = tf.gather_nd(next_target_q, 
                                       tf.stack((self.idx, tf.cast(next_actions, tf.int32)), axis=1))
        targets = rewards + not_done * self.discount_factor * target_q
        q = self.online_network.predict_on_batch(states)

        idxs = tf.stack((self.idx, actions), axis=1)
        q = tf.tensor_scatter_nd_update(q, idxs, targets)

        loss = self.online_network.train_on_batch(x=states, y=q)
        self.losses.append(loss)

        if self.total_steps % self.update_frequency == 0:
            self.update_target()



class DQNAgent:
    def __init__(self, state_dim,
                 n_actions,
                 lr,
                 discount_factor,
                 eps_start,
                 eps_end,
                 eps_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 architecture,
                 l2_reg,
                 update_frequency,
                 batch_size):

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.lr = lr
        self.discount_factor = discount_factor
        self.architecture = architecture
        self.l2_reg = l2_reg

        self.online_network = self.build_network()

        self.epsilon = eps_start
        self.eps_decay_steps = eps_decay_steps
        self.epsilon_decay = (eps_start - eps_end) / eps_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.losses = []
        self.idx = tf.range(batch_size)
        self.train = True

    def build_network(self, trainable=True):
        layers = []
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None,
                                activation='relu',
                                kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        layers.append(Dropout(.1))
        layers.append(Dense(units=self.n_actions,
                            trainable=trainable,
                            name='Output'))
        model = Sequential(layers)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.lr))
        return model

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_actions)
        q = self.online_network.predict(state)
        return np.argmax(q, axis=1).squeeze()

    def memorize_transition(self, state, action, reward, next_state, not_done):
        if not_done:
            self.episode_reward += reward
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.eps_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((state, action, reward, next_state, not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        next_q = self.online_network.predict_on_batch(next_states)
        next_actions = tf.argmax(next_q, axis=1)
        target_q = tf.gather_nd(next_q, tf.stack((self.idx, tf.cast(next_actions, tf.int32)), axis=1))
        
        targets = rewards + not_done * self.discount_factor * target_q
        q = self.online_network.predict_on_batch(states)
        idxs = tf.stack((self.idx, actions), axis=1)
        q = tf.tensor_scatter_nd_update(q, idxs, targets)

        loss = self.online_network.train_on_batch(x=states, y=q)
        self.losses.append(loss)



#TO-DO
class RandomAgent:
    def __init__(self, state_dim,
                 n_actions,
                 lr,
                 discount_factor,
                 eps_start,
                 eps_end,
                 eps_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 architecture,
                 l2_reg,
                 update_frequency,
                 batch_size):

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.lr = lr
        self.discount_factor = discount_factor
        self.architecture = architecture
        self.l2_reg = l2_reg

        self.online_network = self.build_network()

        self.epsilon = eps_start
        self.eps_decay_steps = eps_decay_steps
        self.epsilon_decay = (eps_start - eps_end) / eps_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.losses = []
        self.idx = tf.range(batch_size)
        self.train = True

    def build_network(self, trainable=True):
        layers = []
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None,
                                activation='relu',
                                kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        layers.append(Dropout(.1))
        layers.append(Dense(units=self.n_actions,
                            trainable=trainable,
                            name='Output'))
        model = Sequential(layers)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.lr))
        return model

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.n_actions)
        q = self.online_network.predict(state)
        return np.argmax(q, axis=1).squeeze()

    def memorize_transition(self, state, action, reward, next_state, not_done):
        if not_done:
            self.episode_reward += reward
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.eps_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((state, action, reward, next_state, not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        next_q = self.online_network.predict_on_batch(next_states)
        next_actions = tf.argmax(next_q, axis=1)
        target_q = tf.gather_nd(next_q, tf.stack((self.idx, tf.cast(next_actions, tf.int32)), axis=1))
        
        targets = rewards + not_done * self.discount_factor * target_q
        q = self.online_network.predict_on_batch(states)
        idxs = tf.stack((self.idx, actions), axis=1)
        q = tf.tensor_scatter_nd_update(q, idxs, targets)

        loss = self.online_network.train_on_batch(x=states, y=q)
        self.losses.append(loss)
