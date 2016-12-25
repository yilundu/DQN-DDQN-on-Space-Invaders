import gym
import numpy as np
import random
import keras
import cv2
from keras.models import load_model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Activation, Dropout, Flatten, Dense
from collections import deque

# List of hyper-parameters and constants
DECAY_RATE = 0.99
BUFFER_SIZE = 40000
MINIBATCH_SIZE = 64
TOT_FRAME = 3000000
EPSILON_DECAY = 1000000
MIN_OBSERVATION = 5000
FINAL_EPSILON = 0.05
INITIAL_EPSILON = 0.1
NUM_ACTIONS = 3
TAU = 0.01
# Number of frames to throw into network
NUM_FRAMES = 3

class ReplayBuffer:
    """Constructs a buffer object that stores the past moves
    and samples a set of subsamples"""

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()

    def add(self, s, a, r, d, s2):
        """Add an experience to the buffer"""
        # S represents current state, a is action,
        # r is reward, d is whether it is the end, 
        # and s2 is next state
        experience = (s, a, r, d, s2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self, batch_size):
        """Samples a total of elements equal to batch_size from buffer
        if buffer contains enough elements. Otherwise return all elements"""

        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        # Maps each experience in batch in batches of states, actions, rewards
        # and new states
        s_batch, a_batch, r_batch, d_batch, s2_batch = map(np.array, zip(*batch))

        return s_batch, a_batch, r_batch, d_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

class DeepQ:
    """Constructs the desired deep q learning network"""
    def __init__(self):
        self.construct_q_network()
        pass

    def construct_q_network(self):
        # Uses the network architecture found in DeepMind paper
        self.model = Sequential()
        self.model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Dense(NUM_ACTIONS))
        self.model.compile(loss='mse', optimizer=Adam(lr=0.00001))

        # Creates a target network as described in DeepMind paper
        self.target_model = Sequential()
        self.target_model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(84, 84, NUM_FRAMES)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Convolution2D(64, 3, 3))
        self.target_model.add(Activation('relu'))
        self.target_model.add(Flatten())
        self.target_model.add(Dense(512))
        self.target_model.add(Dense(NUM_ACTIONS))
        self.target_model.compile(loss='mse', optimizer=Adam(lr=0.00001))
        self.target_model.set_weights(self.model.get_weights())

        print "Successfully constructed networks."

    def predict_movement(self, data, epsilon):
        """Predict movement of game controler where is epsilon
        probability randomly move."""
        q_actions = self.model.predict(data.reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
        opt_policy = np.argmax(q_actions)
        rand_val = np.random.random()
        if rand_val < epsilon:
            opt_policy = np.random.randint(0, NUM_ACTIONS)
        return opt_policy, q_actions[0, opt_policy]

    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        """Trains network to fit given parameters"""
        batch_size = s_batch.shape[0]
        targets = np.zeros((batch_size, NUM_ACTIONS))

        for i in xrange(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] += DECAY_RATE * np.max(fut_action)

        loss = self.model.train_on_batch(s_batch, targets)

        # Print the loss every 10 iterations.
        if observation_num % 10 == 0:
            print "We had a loss equal to ", loss

    def save_network(self, path):
        # Saves model at specified path as h5 file
        self.model.save(path)
        print "Successfully saved network."

    def load_network(self, path):
        self.model = load_model(path)
        print "Succesfully loaded network."

    def target_train(self):
        model_weights = self.model.get_weights()
        target_model_weights = self.target_model.get_weights()
        for i in xrange(len(model_weights)):
            target_model_weights[i] = TAU * model_weights[i] + (1 - TAU) * target_model_weights[i]
        self.target_model.set_weights(target_model_weights)

class SpaceInvader:

    def __init__(self):
        self.env = gym.make('SpaceInvaders-v0')
        self.env.reset()
        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.deep_q = DeepQ()
        # A buffer that keeps the last 3 images
        self.process_buffer = []
        # Initialize buffer with the first frame
        s1, r1, _, _ = self.env.step(0)
        s2, r2, _, _ = self.env.step(0)
        s3, r3, _, _ = self.env.step(0)
        self.process_buffer = [s1, s2, s3]

    def load_network(self, path):
        self.deep_q.load_network(path)

    def convert_process_buffer(self):
        """Converts the list of NUM_FRAMES images in the process buffer
        into one training sample"""
        black_buffer = map(lambda x: cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (84, 90)), self.process_buffer)
        black_buffer = map(lambda x: x[1:85, :, np.newaxis], black_buffer)
        return np.concatenate(black_buffer, axis=2)

    def train(self, num_frames):
        observation_num = 0
        curr_state = self.convert_process_buffer()
        epsilon = INITIAL_EPSILON
        alive_frame = 0
        total_reward = 0

        while observation_num < num_frames:
            if observation_num % 1000 == 999:
                print "Executing loop %d" %observation_num

            # Slowly decay the learning rate
            if epsilon > FINAL_EPSILON:
                epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY

            initial_state = self.convert_process_buffer()
            self.process_buffer = []

            predict_movement, predict_q_value = self.deep_q.predict_movement(curr_state, epsilon)

            reward, done = 0, False
            for i in xrange(3):
                temp_observation, temp_reward, temp_done, _ = self.env.step(predict_movement)
                reward += temp_reward
                self.process_buffer.append(temp_observation)
                done = done | temp_done

            if observation_num % 10 == 0:
                print "We predicted a q value of ", predict_q_value

            if done:
                print "Lived with maximum time ", alive_frame
                print "Earned a total of reward equal to ", total_reward
                self.env.reset()
                alive_frame = 0
                total_reward = 0

            new_state = self.convert_process_buffer()
            self.replay_buffer.add(initial_state, predict_movement, reward, done, new_state)
            total_reward += reward

            if self.replay_buffer.size() > MIN_OBSERVATION:
                s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(MINIBATCH_SIZE)
                self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
                self.deep_q.target_train()

            # Save the network every 100000 iterations
            if observation_num % 10000 == 9999:
                print "Saving Network"
                self.deep_q.save_network("saved.h5")

            alive_frame += 1
            observation_num += 1

    def simulate(self):
        """Simulates game"""
        done = False
        self.env.reset()
        tot_award = 0
        # self.env.render()
        while not done:
            state = self.convert_process_buffer()
            predict_movement = self.deep_q.predict_movement(state, 0.5)[0]
            self.process_buffer = []
            for i in xrange(3):
                self.env.render()
                observation, reward, done, _ = self.env.step(predict_movement)
                tot_award += reward
                self.process_buffer.append(observation)
        state = self.convert_process_buffer()
        print tot_award


if __name__ == "__main__":
    print "Haven't finished implementing yet...'"
    space_invader = SpaceInvader()
    space_invader.load_network("saved.h5")
    space_invader.simulate()
    # space_invader.train(TOT_FRAME)
