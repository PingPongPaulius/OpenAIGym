import gymnasium as gym
import ale_py
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt


transform = T.Compose([
    T.ToPILImage(),
    T.Resize((100,100)),
    T.ToTensor()
    ])

def to_gray_to_binary(rgb):
    return (np.dot(rgb[:,:,:3], [0.2989, 0.5870, 0.1140])>128).astype(np.uint8)

def process_image(rgb):
    binary = to_gray_to_binary(pixels)
    binary = binary[34:194, :]
    return binary, transform(torch.tensor(binary)).squeeze(0)


gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5", render_mode="human")
obs, info = env.reset()

print("OBS: ", np.size(obs))
print("Info: ", info)

x,y,z = np.indices((obs.shape))
df = pd.DataFrame({"x": x.flatten(), "y": y.flatten(), "z":z.flatten(), "value":obs.flatten()})

df.to_csv("Output.csv")

NUM_EPISODES = 30
FRAME_QUEUE_SIZE = 4
frame_queue = []

for episode in range(NUM_EPISODES):
    
    action = env.action_space.sample()
    pixels, reward, terminated, truncated, info = env.step(action) 
    # Data/Image Processing
    image, tensor = process_image(pixels)
    
    if(len(frame_queue) >= FRAME_QUEUE_SIZE):
        frame_queue = frame_queue[1:]

    frame_queue.append(tensor)
    frames = torch.stack(frame_queue, dim=0)
    print(frames.shape)

    if terminated or truncated:
        pixels, info = env.reset()


plt.imshow(image)
plt.show()

env.close()

