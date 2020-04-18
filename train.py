import os
import time
import random
import numpy as np
from game import Paddle
from random import sample
from collections import deque

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, str(int(time.time())))

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class DQNAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.tau = 0.001
        self.alpha = 0.001
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.batch_size = 32

        self.memory = deque(maxlen=2000)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{int(time.time())}")

        self.model = self.create_model(plot=True)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self, plot=False):

        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='relu'))

        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))

        if plot:
            model.summary()
            plot_model(model, to_file='model.png', show_shapes=True, dpi=100)

        return model

    def update_target_model(self):
        model_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()

        for i in range(len(model_weights)):
            target_weights[i] = self.tau * model_weights[i] + (1 - self.tau) * target_weights[i]

        self.target_model.set_weights(target_weights)

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)

        state = np.array(state).reshape(-1, self.state_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values)

    def replay(self, terminal_state):
        if len(self.memory) < 1000:
            return None

        minibatch = sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

        qs = self.model.predict(states)
        next_qs = self.target_model.predict(next_states)

        for i in range(len(dones)):
            new_q = rewards[i] + (1 - int(dones[i])) * self.gamma * np.max(next_qs[i])
            action = actions[i]
            qs[i][action] = new_q

        self.model.fit(states, qs, batch_size=self.batch_size, verbose=0, shuffle=False,
                       callbacks=[self.tensorboard] if terminal_state else None)

        self.update_target_model()

    def load(self, name):
        self.model = load_model(name)
        self.target_model.set_weights(self.model.get_weights())

    def save(self, name):
        self.model.save(name)


def main():
    episode_start = 1
    episode_total = 1000
    ep_rewards = deque(maxlen=10)

    env = Paddle()
    agent = DQNAgent(5, 3)

    if not os.path.isdir('models'):
        os.makedirs('models')

    for e in range(episode_start, episode_total + 1):
        agent.tensorboard.step = e
        episode_reward = 0
        state = env.reset()

        done = False
        while not done:

            action = agent.act(state)
            new_state, reward, done = env.step(action)

            episode_reward += reward

            agent.memorize(state, action, reward, new_state, done)
            agent.replay(done)

            state = new_state

        ep_rewards.append(episode_reward)

        if not e % 10 or e == 1:
            average_reward = sum(ep_rewards) / len(ep_rewards)
            min_reward = min(ep_rewards)
            max_reward = max(ep_rewards)

            agent.tensorboard.update_stats(reward_avg=average_reward,
                                           reward_min=min_reward,
                                           reward_max=max_reward,
                                           epsilon=agent.epsilon)

            agent.save(f'models\\avg_{average_reward}_e_{e}_epsilon_{agent.epsilon:.5f}.h5')

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay
            agent.epsilon = max(agent.epsilon, agent.epsilon_min)

        print(f'Episode: {e}/{episode_total} Reward: {episode_reward} Epsilon {agent.epsilon:.5f}')

    agent.save(f'models\\final_model_avg_{average_reward}_e_{e}_epsilon_{agent.epsilon:.5f}.h5')


if __name__ == '__main__':
    main()
