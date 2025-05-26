import gymnasium as gym
import ale_py
import pandas as pd
import numpy as np
import torch
from torch import nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import random as rng
from collections import deque

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((100,100)),
    T.ToTensor()
    ])

NUM_EPISODES = 1
FRAME_QUEUE_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNet(nn.Module):

    def __init__(self, num_actions):
        super().__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(16, 256),
                nn.ReLU(),
                nn.Linear(256, num_actions)
                )
    
    def forward(self, x):
        return self.model(x)

def to_gray_to_binary(rgb):
    return (np.dot(rgb[:,:,:3], [0.2989, 0.5870, 0.1140])>128).astype(np.uint8)

def process_image(rgb):
    binary = to_gray_to_binary(pixels)
    binary = binary[34:194, :]
    return binary, transform(torch.tensor(binary)).squeeze(0)


gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="human")

D = deque(maxlen=100000)
Q = NeuralNet(env.action_space.n).to(DEVICE)
Epsilon = 0.4
Gamma = 0.4
SAMPLE_SIZE = 10

for episode in range(NUM_EPISODES):
    
    pixels, info = env.reset()
    running = True
    reward = 0

    image, tensor = process_image(pixels)
    frame_queue = [tensor, tensor, tensor, tensor]
    frames = torch.stack(frame_queue, dim=0).to(DEVICE)

    while running:

        state = frames
        if (rng.random() < Epsilon):
            action = env.action_space.sample()
        else:
            output = Q.forward(frames)
            output = torch.argmax(output, dim=1)
            counts = torch.bincount(output)
            action = torch.argmax(counts)

        pixels, reward, terminated, truncated, info = env.step(action) 

        image, tensor = process_image(pixels)

        frame_queue = frame_queue[1:]
        frame_queue.append(tensor)
        frames = torch.stack(frame_queue, dim=0).to(DEVICE)
        D.append((state, action, reward, frames))

        if len(D) >= SAMPLE_SIZE:
            subsamples = rng.sample(D, SAMPLE_SIZE)

            for sample in subsamples:

                s = sample[0]
                s_next = sample[3]
                r = torch.tensor(sample[2], dtype=torch.float32).to(DEVICE)
                a = torch.tensor(sample[1], dtype=torch.int64).to(DEVICE)
                terminal = (r != 0).int()

                target_q = Q.forward(s_next).max(1)[0]
                y = r + target_q * Gamma * (1-terminal)

                Q_state = Q.forward(s)
                a_index = torch.full((32,),a).to(DEVICE)
                Q_real = Q_state[:, action]
                loss = nn.functional.mse_loss(Q_real, y)

                loss.backward()

        running = not (terminated or truncated)


plt.imshow(image)
plt.show()

env.close()

