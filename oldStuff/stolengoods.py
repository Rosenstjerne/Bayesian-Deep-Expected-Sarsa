#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 14:25:18 2020

@author: hossein
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
import gym
from collections import deque
import numpy as np
import random as rnd
import datetime
from log_metric import ExSARSAMetric

#tf.config.set_visible_devices([], 'GPU')

REPLAY_MEMORY_SIZE = 10000
MIN_REPLAY_MEMORY_SIZE = 100
MINIBATCH_SIZE = 16
UPDATE_TARGET_AFTER = 1000
DISCOUNT = 0.95
MAX_ACTION = 35000
LOGGING = True
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
exsarsa_reward_log_dir = 'logs/gradient_tape/' + current_time + '/exsarsa_reward2'


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        """
        in each step in training phase, the agent will do an action, to do the action,
        agent needs the Q_VALUES which is the QNetwork (obviousely it's maximum)
        so for each action. the QNetwork.predict() will be called
        On the other hand after each action the QNetwork will tries to fit it self just
        by one input! There are two problem on this:
                1- Fitting a NN just by one input is not a good idea. We know it is better
                   to fitting on a BATCH of data (A sequence of data)
                2- If we do predict-fit-predict-fit we do not have consistency on predicts
        So it is better to use two QNetwork:
                1-model    2-target_model
        and also using experience replay
        The agent use for target_model for predicts and the 
        model we be trained on BATCH thanks to the experience reply
        eperience replay is also for planning (like DYNAQ on RL specialization)
        we will copy model to target_model after UPDATE_AFTER actions
        """
        self.model = self.create_QNetwork()
        self.target_model = self.create_QNetwork()
        self.target_model.set_weights(self.model.get_weights())
        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tau = 1
        self.target_update_after_counter = 0

        if LOGGING:
            self.exsarsa_reward_writer = tf.summary.create_file_writer(exsarsa_reward_log_dir)
            self.exsarsa_reward_metric = ExSARSAMetric()

    def create_QNetwork(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size))
        model.add(Activation('relu'))

        model.add(Dense(128))
        model.add(Activation('relu'))

        model.add(Dense(self.action_size))
        model.add(Activation('linear'))

        model.compile(optimizer=Adam(lr=0.001, decay=0.00001), loss="mse", metrics=['accuracy'])

        return model

    def get_qvalues(self, state):
        return self.model.predict(state)

    def softmax(self, qvalues):
        preferences = qvalues / self.tau
        max_preference = np.amax(qvalues, axis=1) / self.tau
        reshaped_max_preference = max_preference.reshape((-1, 1))

        # Compute the numerator, i.e., the exponential of the preference - the max preference.
        exp_preferences = np.exp(preferences - reshaped_max_preference)
        # Compute the denominator, i.e., the sum over the numerator along the actions axis.
        sum_of_exp_preferences = np.sum(exp_preferences, axis=1)

        reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))
        action_probs = exp_preferences / reshaped_sum_of_exp_preferences
        action_probs = action_probs.squeeze()
        return action_probs

    def act(self, state):
        qvalues = self.get_qvalues(state)
        actions_probability = self.softmax(qvalues)
        action = np.random.choice(self.action_size, p=actions_probability.squeeze())
        return action

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, terminal):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        for i in range(2):
            minibatch = rnd.sample(self.replay_memory, MINIBATCH_SIZE)
            current_states = np.array([transition[0] for transition in minibatch])
            current_qvalues_list = self.model.predict(current_states.squeeze())

            next_states = np.array([transition[3] for transition in minibatch])
            next_qvalues_list = self.target_model.predict(next_states.squeeze())
            next_actions_prob = self.softmax(next_qvalues_list)
            x_train = []
            y_train = []

            for index, (current_state, action, reward, next_state, done) in enumerate(minibatch):
                if not done:
                    future_reward = np.inner(next_qvalues_list[index], next_actions_prob[index])
                    desired_q = reward + DISCOUNT * future_reward
                else:
                    desired_q = reward

                current_q_values = current_qvalues_list[index]
                current_q_values[action] = desired_q

                x_train.append(current_state)
                y_train.append(current_q_values)

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            x_train = np.reshape(x_train, [len(minibatch), self.state_size])
            y_train = np.reshape(y_train, [len(minibatch), self.action_size])

            self.model.fit(x_train, y_train, batch_size=MINIBATCH_SIZE, verbose=0)
            self.target_update_after_counter += 1

        if self.target_update_after_counter > UPDATE_TARGET_AFTER and terminal:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_after_counter = 0
            print("*Target model updated*")


def cartpole():
    env = gym.make("CartPole-v0")
    observation_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n
    exsarsa_agent = DQNAgent(observation_space_size, action_space_size)
    episode_num = 0
    action_num = 0
    task_done = deque(maxlen=20)
    while True:
        episode_num += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space_size])
        t = 0
        while True:
            env.render()
            action = exsarsa_agent.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, observation_space_size])
            transition = (state, action, reward, state_next, terminal)
            exsarsa_agent.update_replay_memory(transition)
            exsarsa_agent.train(terminal)
            state = state_next
            t += 1
            action_num += 1
            if sum(task_done)/(len(task_done)+1)>195:
                env.close()
            if terminal:
                print("Episode {} finished after {} timesteps".format(episode_num, t))
                task_done.append(t)
                if LOGGING:
                    exsarsa_agent.exsarsa_reward_metric.update_state(t)
                    with exsarsa_agent.exsarsa_reward_writer.as_default():
                        tf.summary.scalar('exsarsa_reward', exsarsa_agent.exsarsa_reward_metric.result(), step=episode_num)
                    exsarsa_agent.exsarsa_reward_metric.reset_states()
                break


def main():
    cartpole()


if __name__ == "__main__":
    main()