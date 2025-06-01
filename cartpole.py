import gymnasium as gym
import ale_py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rng
import copy
from collections import deque

def DQN():
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(8, input_shape=(4,), activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

    return model

def state_to_tensor(states):
    return tf.convert_to_tensor(states, dtype=tf.float32)

def get_action(state, DQN, Epsilon):
    if rng.random() < max(Epsilon, 0.01):
        return env.action_space.sample()
    else:
        state = tf.reshape(state, (1,4))
        output = DQN.predict(state, verbose=0)
        return np.argmax(output)

##gym.register_envs(ale_py)
#env = gym.make("CartPole-v1", render_mode="human")
env = gym.make("CartPole-v1")

D = deque(maxlen=10000000)
Q = DQN()
NUM_EPISODES = 1000
FRAME_QUEUE_SIZE = 4
Epsilon = 1.0
Gamma = tf.constant(0.95, dtype=tf.float32)
SAMPLE_SIZE = 32
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
    
        if(len(D) >= SAMPLE_SIZE):
            minibatch = rng.sample(list(D), SAMPLE_SIZE)

            for s, a, r, next_state, cont in minibatch:
                
                
                s = tf.reshape(s, (1,4))
                next_state = tf.reshape(next_state, (1,4))
                y = reward + Gamma * np.max(Q.predict(next_state)) if cont else r
                q = Q.predict(s, verbose=0)
                q[0][a] = y
                Q.fit(s, q, epochs=1, verbose=0)

    Epsilon *= 0.995


    performance.append(score)
    print("Episode Fisnished", episode, " Reward", score)


plt.plot([i for i in range(len(performance))], performance)
plt.show()

env.close()

