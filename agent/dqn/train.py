import os
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from collections import deque
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from environment.assembly import Assembly
from environment.panelblock import *


class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.fc1 = Dense(256, activation='elu', kernel_initializer='he_normal')
        self.fc2 = Dense(256, activation='elu', kernel_initializer='he_normal')
        self.fc3 = Dense(128, activation='elu', kernel_initializer='he_normal')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        q = self.fc_out(x)
        return q

class DQNAgent:
    def __init__(self, env, state_size, action_size, num_episode, load_model=False, render=False):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

        self.num_episode = num_episode

        self.load_model = load_model
        self.render = render

        self.model_path = '../../model/dqn/queue-%d' % action_size
        self.summary_path = '../../summary/dqn/queue-%d' % action_size

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        if not os.path.exists(self.summary_path):
            os.makedirs(self.summary_path)

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.001
        self.batch_size = 64
        self.train_start = 1000
        self.update_target_rate = 10000

        self.memory = deque(maxlen=2000)

        if load_model:
            self.model = keras.models.load_model(self.model_path)
        else:
            self.model = DQN(action_size)
        self.target_model = DQN(action_size)
        self.optimizer = Adam(lr=self.learning_rate)
        self.update_target_model()

        self.writer = tf.summary.create_file_writer(self.summary_path)
        self.avg_q_max, self.avg_loss = 0, 0

    def draw_tensorboard(self, reward, lead_time, step, episode):
        with self.writer.as_default():
            tf.summary.scalar('Average Loss', self.avg_loss / float(step), step=episode)
            tf.summary.scalar('Average Max Q', self.avg_q_max / float(step), step=episode)
            tf.summary.scalar('Length', step, step=episode)
            tf.summary.scalar('Total Reward', reward, step=episode)
            tf.summary.scalar('Average Lead time', lead_time, step=episode)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model(state)
            return np.argmax(q_value[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.array([sample[0][0] for sample in mini_batch])
        actions = np.array([sample[1] for sample in mini_batch])
        rewards = np.array(sample[2] for sample in mini_batch)
        next_states = np.array([sample[3][0] for sample in mini_batch])
        dones = np.array([sample[4] for sample in mini_batch])

        model_params = self.model.trainable_variables
        with tf.GradientTape() as tape:
            predicts = self.model(states)
            one_hot_action = tf.one_hot(actions, self.action_size)
            predicts = tf.reduce_sum(one_hot_action * predicts, axis=1)

            target_predicts = self.target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + (1 - dones) * self.discount_factor * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))

            self.avg_loss += loss.numpy()

        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))

    def run(self):
        total_step = 0
        for e in range(self.num_episode):
            done = False
            step = 0
            episode_reward = 0

            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])

            while not done:
                if agent.render:
                    self.env.render()
                total_step += 1
                step += 1

                action = self.get_action(state)
                next_state, reward, done = self.env.step(action)
                next_state = np.reshape(next_state, [1, state_size])

                self.avg_q_max += np.amax(self.model(np.float32(state))[0])

                self.append_sample(state, action, reward, next_state, done)

                episode_reward += reward
                state = next_state

                if len(self.memory) >= self.train_start:
                    self.train_model()

                    if total_step % self.update_target_rate == 0:
                        self.update_target_model()

                if done:
                    lead_time = self.env.model['Sink'].last_arrival
                    if total_step > self.train_start:
                        self.draw_tensorboard(episode_reward, lead_time, step, e)

                    self.avg_q_max, self.avg_loss = 0, 0

            if e % 250 == 0:
                self.model.save_weights(self.model_path, save_format='tf')
                print("Saved Model at episode %d" % e)


if __name__ == "__main__":
    panel_blocks, num_of_processes = import_panel_block_schedule('../../environment/data/PBS_assy_sequence_gen_000.csv')

    num_episode = 10000

    len_of_queue = 10
    state_size = num_of_processes * len_of_queue + num_of_processes
    action_size = len_of_queue

    load_model = False

    event_path = '../../environment/simulation_result'
    if not os.path.exists(event_path):
        os.makedirs(event_path)

    assembly = Assembly(num_of_processes, len_of_queue, event_path + '/event_PBS.csv', inbound_panel_blocks=panel_blocks)
    agent = DQNAgent(assembly, state_size, action_size, num_episode)

    agent.run()