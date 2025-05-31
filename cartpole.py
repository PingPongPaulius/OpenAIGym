import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rng
import copy
from collections import deque

class Layer:

    def __init__(self, inp_size, out_size, r=False):
        self.weights = tf.Variable(np.random.rand(inp_size, out_size).astype(np.float32), dtype=tf.float32)
        self.r_shape = r
    
    def feed_forward(self, x):
        x = tf.cast(x, tf.float32)
        if self.r_shape:
            x = tf.reshape(x, (1,4))
        matrix = tf.matmul(x, self.weights)
        self.output = self.activation_function(matrix)
        return self.output

    def activation_function(self, matrix):
        constant = tf.constant(1.0)
        return constant / (constant + tf.math.exp(-matrix))

def state_to_tensor(states):
    return tf.convert_to_tensor(states, dtype=tf.float32)

def get_action(state, Q, Epsilon):
    if rng.random() < max(Epsilon, 0.01):
        return env.action_space.sample()
    else:
        for layer in Q:
            state = layer.feed_forward(state)
        return tf.math.argmax(state, axis=1).numpy().item()

##gym.register_envs(ale_py)
env = gym.make("CartPole-v1", render_mode="human")
#env = gym.make("CartPole-v1")

i_layer = Layer(4, 32, True)
h_layer = Layer(32, 2, False)
o_layer = Layer(2, 2, False)

D = deque(maxlen=100000)
Q = [i_layer, h_layer, o_layer]
NUM_EPISODES = 1000
FRAME_QUEUE_SIZE = 4
Epsilon = 1.0
Gamma = tf.constant(0.95, dtype=tf.float32)
SAMPLE_SIZE = 16
LR = 0.0001

performance = []

state, info = env.reset()

for episode in range(NUM_EPISODES):
    
    running = True
    state, info = env.reset()
    score = 0
    while running:
        
        curr_states = state_to_tensor(state)
        action = get_action(curr_states, Q, Epsilon)
        state, reward, terminated, truncated, info = env.step(action) 
        if terminated or truncated:
            running = False
        D.append((curr_states, action, reward, state_to_tensor(state), running))
        score += 1
    
    Epsilon *= 0.955
        
    if(len(D) >= SAMPLE_SIZE):
        minibatch = rng.sample(list(D), SAMPLE_SIZE)

        i = 0
        for state, action, reward, next_state, running in minibatch:
            
            mod = tf.constant(1.0) if running else tf.constant(0.0)
            r = tf.constant(reward)

            # TODO Handle MAX VALUES
            weights = []
            with tf.GradientTape(persistent=True) as tape:
                for layer in Q:
                    tape.watch(layer.weights)
                
                weights = []
                state_q_values = state
                next_q_values = next_state
                tape.watch(state_q_values)
                tape.watch(next_q_values)
                for layer in Q:
                    state_q_values = layer.feed_forward(state_q_values)
                    weights.append(layer.weights)
                for layer in Q:
                    next_q_values = layer.feed_forward(next_q_values)
                
                Q_error = tf.reduce_max(next_q_values)
                discounted_error = tf.multiply(Gamma, Q_error)
                error = tf.multiply(discounted_error, mod)
                y = tf.add(r, error)
                q_pred = tf.reduce_sum(state_q_values * tf.one_hot(action, depth=2), axis=1)
                loss = -tf.reduce_mean(tf.square(tf.subtract(y,q_pred)))

                gradients = tape.gradient(loss, weights)

                for index, l_w_a in enumerate(Q):
                    Q[index].weights = tf.Variable(tf.add(Q[index].weights, gradients[index]*LR), dtype=tf.float32)


    performance.append(score)
    if loss is None:
        loss = 500000
    print("Episode Fisnished", episode, " Reward", score, " Loss: ", loss)


plt.plot([i for i in range(len(performance))], performance)
plt.show()

env.close()

