import gymnasium as gym
import ale_py
import pandas as pd
import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random as rng
import copy
from collections import deque

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((100,100)),
    T.ToTensor()
    ])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):

    def __init__(self, num_actions):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        self.fc = nn.Linear(3872, 256)
        self.output = nn.Linear(256, num_actions)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        x = F.relu(self.fc(x))
        return self.output(x)

def to_gray_to_binary(rgb):
    return (np.dot(rgb[:,:,:3], [0.2989, 0.5870, 0.1140])>128).astype(np.uint8)

def process_image(rgb):
    binary = to_gray_to_binary(pixels)
    binary = binary[34:194, :]
    return binary, transform(torch.tensor(binary)).squeeze(0)


gym.register_envs(ale_py)
#env = gym.make("ALE/Pong-v5", render_mode="human")
env = gym.make("ALE/Pong-v5")

C_max = 10
C = C_max
D = deque(maxlen=10000000)
Q = NeuralNet(env.action_space.n).to(DEVICE)
Q_star = copy.deepcopy(Q)
NUM_EPISODES = 100
FRAME_QUEUE_SIZE = 4
Epsilon = 1.0
Gamma = 0.99
SAMPLE_SIZE = 32

performance = []
ADAM = torch.optim.Adam(Q_star.parameters(), lr=1e-4)

pixels, info = env.reset()
image, tensor = process_image(pixels)
frame_queue = []
state = None

for episode in range(NUM_EPISODES):
    
    running = True
    pixels, info = env.reset()
    score = 0
    while running:

        if (rng.random() <= max(Epsilon, 0.01)):
            action = env.action_space.sample()
        else:
            output = Q.forward(frames)
            action = torch.argmax(output)

        pixels, reward, terminated, truncated, info = env.step(action) 
        image, tensor = process_image(pixels)
        frame_queue.append(tensor)
        score+=reward

        if(len(frame_queue) < FRAME_QUEUE_SIZE):
            continue
        else:
            frames = torch.stack(frame_queue, dim=0).to(DEVICE)
            frame_queue = []

        if state is not None:
            D.append((state, action, reward, frames))
        state = frames


        if len(D) >= SAMPLE_SIZE:
            subsamples = rng.sample(D, SAMPLE_SIZE)
            targets = torch.zeros(SAMPLE_SIZE).to(DEVICE)
            real_values = torch.zeros(SAMPLE_SIZE).to(DEVICE)

            for i, sample in enumerate(subsamples):

                s = sample[0]
                s_next = sample[3]
                r = torch.tensor(sample[2], dtype=torch.float32).to(DEVICE)
                terminal = (r != 0).int()

                next_q = Q_star.forward(s_next)
                # Target is best Q value
                desired_q = next_q.max()
                if r == 0:
                    target_q = r + desired_q * Gamma
                else:
                    target_q = r

                action_taken = sample[1]
                Q_action = Q_star.forward(s)[action_taken]

                real_values[i] = Q_action
                targets[i] = target_q

            loss = F.mse_loss(targets, real_values)

            ADAM.zero_grad()
            loss.backward()
            ADAM.step()
        
        Epsilon -= 0.00000001
        running = not (terminated or truncated)

    Q = copy.deepcopy(Q_star)

    performance.append(score)
    print("Episode Fisnished", episode, " Reward", score)


plt.plot([i for i in range(len(performance))], performance)
plt.show()

env.close()

