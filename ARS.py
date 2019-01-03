import os
import numpy as np
import gym
from gym import wrappers
import pybullet_envs
# setting hyper parameters


class HyperParameters():            # You can tweak this parameters they are permanent for every
                                    # episode
    def __init__(self):
        self.num_steps = 1000
        self.episode_length = 1000
        self.learning_rate = 0.02
        self.num_directions = 16
        self.best_directions = 16
        assert self.best_directions <= self.num_directions
        self.noise = 0.03
        self.seed = 1
        self.env_name = 'HalfCheetahBulletEnv-v0' # We choose HalfCheetah for spead reasons
                                                  # if you have available fast enough server
                                                  # or cluster you can try with 'Humanoid-v1'
                                                  # or 'Humanoid-v2' depending on version of lib

# normalise states
class Normalise():
    def __init__(self, num_inputs):
        self.num_states = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.variants = np.zeros(num_inputs)

    def observe(self, y):
        self.num_states += 1.
        last_mean = self.mean.copy()
        self.mean += (y - self.mean) / self.num_states

        self.mean_diff += (y - last_mean) * (y - self.mean)
        self.variants = (self.mean_diff / self.num_states).clip(min=1e-2)

    def norm(self, num_inputs):
        ob = self.mean
        obs_std = np.sqrt(self.variants)
        return (num_inputs - ob) / obs_std


# building the AI
class Perceptron():           # AI perceptron class

    def __init__(self, input_len, output_len):
        self.theta = np.zeros((output_len, input_len))

    def evaluate(self, n_input, delta=None, direction=None):
        if direction is None:
            return self.theta.dot(n_input)
        elif direction == "positive":
            return (self.theta + hp.noise * delta).dot(n_input)
        else:
            return (self.theta - hp.noise * delta).dot(n_input)

    def sample_deltas(self):
        return [np.random.rand(self.theta.shape[0], self.theta.shape[1]) for _ in range(hp.num_directions)]

    def update(self, rollout, std_reward):
        step = np.zeros(self.theta.shape)
        for reward_pos, reward_neg, d in rollout:
            step += (reward_pos - reward_neg) * d
        self.theta += hp.learning_rate / (hp.best_directions * std_reward) * step

# Explore function that is run to explore best directions
def explore(environment, norm, percept, direction=None, delta=None):
    state = environment.reset()
    done = False
    num_action_played = 0.
    accumulated_reward = 0
    while done != True and num_action_played <= hp.episode_length:
        norm.observe(state)
        state = norm.norm(state)
        action = percept.evaluate(state, delta, direction)
        state, reward, done, _ = environment.step(action)
        reward = max(min(reward, 1), -1)
        accumulated_reward += reward
        num_action_played += 1
    return accumulated_reward

# Training function with all needed steps
def trainin(enviroment, percept, norm, hp):
    for step in range(hp.num_steps):
        # Initialising the perturbations deltas positive and negative rewards
        deltas = per.sample_deltas()
        positive_rewards = [0] * hp.num_directions
        negative_rewards = [0] * hp.num_directions

        # Positive rewards in positive direction
        for v in range(hp.num_directions):
            positive_rewards[v] = explore(enviroment, norm, percept, direction="positive", delta=deltas[v])

        # Rewards in the negative direction
        for v in range(hp.num_directions):
            negative_rewards[v] = explore(enviroment, norm, percept, direction="negative", delta=deltas[v])
        # All the rewards and its std
        the_rewards = np.array(positive_rewards + negative_rewards)
        std_reward = the_rewards.std()

        # Sorting the rollout by the max(r_pos, r_neg)
        scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        orders = sorted(scores.keys(), key = lambda x:scores[x])[0:hp.best_directions]

        rollout = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in orders]

        # Perceptron update
        percept.update(rollout, std_reward)

        # Printing the rewards after update.
        reward_eval = explore(enviroment, norm, per)
        print("Step: ", step, " Reward: ", reward_eval)

# Making OS paths for storing the video of training cheetah
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir("exp", "ars")
monitor_dir = mkdir(work_dir, "monitor")

# Creation of objects and setting parameters
hp = HyperParameters()
np.random.seed(hp.seed)
env = gym.make(hp.env_name)
env = wrappers.Monitor(env, monitor_dir, force = True)
num_inputs = env.observation_space.shape[0]
num_outputs = env.action_space.shape[0]
per = Perceptron(num_inputs, num_outputs)
norm = Normalise(num_inputs)
# Actual training starts here
trainin(env, per, norm, hp)
