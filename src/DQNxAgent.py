from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from collections import deque
import random
import numpy as np
import sys
# from environment.core import *
from environment.networkx_env import get_services_position
from numpy.random import RandomState
import tensorflow as tf

class DQNAgent:
    def __init__(self, dim):
        self.dim = dim

        self.memory = deque(maxlen=200)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.model = self._build_model()
        self.prng = RandomState(3)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.dim, input_dim=self.dim*self.dim,activation='linear'))
        model.add(Dense(self.dim*2, activation='linear'))
        model.add(Dense(self.dim*self.dim*2))
        model.compile(loss='mse', optimizer=Adam(lr=0.8))

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def round_action(self,action):
        differentiable_round = tf.maximum(action - 0.499, 0)
        differentiable_round = differentiable_round * 10000
        return tf.minimum(differentiable_round, 1).numpy()

    def act(self, state):
        v = self.prng.rand()
        if v <= self.epsilon:
            rew = self.prng.randint(2, size=(self.dim,self.dim,2))
        else:
            # print("FROM MEMORY")
            rew = self.model.predict(state.reshape(1,self.dim*self.dim))
            rew = rew.reshape(self.dim,self.dim,2)
        return self.get_action_from_reward(rew)

    def act_from_memory(self, state):
        """
        For DEBUG issues

        :param state:
        :return:
        """
        # print("FROM MEMORY")
        rew = self.model.predict(state.reshape(1, self.dim * self.dim))
        rew = rew.reshape(self.dim, self.dim, 2)
        # print("REWARD FROM LA MEMORY")
        # print(rew)
        return self.get_action_from_reward(rew)

    def get_reward(self,state):
        rew = self.model.predict(state.reshape(1, self.dim * self.dim))
        rew = rew.reshape(self.dim, self.dim, 2)
        return rew

    def get_action_from_reward(self,rew):
        act = [np.argmax(rew[i][j]) for i in range(self.dim) for j in range(self.dim)]
        return np.array(act).reshape((self.dim, self.dim))

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            s = state.reshape((1,self.dim*self.dim,))
            sp = next_state.reshape((1,self.dim*self.dim,))
            reward_real = reward

            if not done:
                reward_real = (reward_real + self.gamma * np.amax(self.model.predict(sp))) #reward_learned

            reward_model = self.model.predict(s).reshape((self.dim,self.dim,2))
            pos_action = get_services_position(state)
            for row in range(len(reward_real)):
                for col in range(len(reward_real[row])):
                    if row in pos_action:
                        reward_model[row][col] = reward_real[row][col]
                    else:
                        reward_model[row][col] = [0.,0.]

            states.append(s)
            targets_f.append(reward_model.reshape(1,self.dim*self.dim*2))

        states = np.array(states).reshape(-1,self.dim*self.dim)
        targets_f= np.array(targets_f).reshape(-1, self.dim*self.dim*2)

        history = self.model.fit(states, targets_f, epochs=1, verbose=0)

        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def one_replay(self,state,next_state,reward,done):
        """
        For Debug issues

        :param state:
        :param next_state:
        :param reward:
        :param done:
        :return:
        """
        s = state.reshape((1, self.dim * self.dim,))
        sp = next_state.reshape((1, self.dim * self.dim,))
        rew = reward.reshape(1, self.dim * self.dim * 2)
        history = self.model.fit(s, rew, epochs=10, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)