import numpy as np
import random
import gym
import ale_py
import collections
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

PATH = 'PATH'

ENV_LIST = ['Breakout-v5', 'Seaquest-v5']
NUM_EPISODES = 3000
INTERVAL = 20
EPS = 0.05
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
GAMMA = 0.98
BUFFER_LIMIT = 10000

# Check Performance
GAME_NAME = []
IDX_EPISODE = []
IDX_EPOCH = []
AVG_REWARD = [] # avg reward per episode
AVG_Q_VALUE = [] # avg action value
Q_VALUE = []

class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.action_size = action_size
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(7*7*64, 512)
        self.out = nn.Linear(512, self.action_size)
    
    def forward(self, x):
        # print('---'*10)
        x = x.float()
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.relu(self.fc(x.view(x.size(0), -1)))
        # print(x.shape)
        x = self.out(x)
        # print(x.shape, x)
        # print('---'*10)
        return x
    
    def epsilon_greedy(self, obs, epsilon):
        if random.random() < epsilon:
            a = np.random.randint(0, self.action_size)
            return a
        else:
            out = self.forward(obs)
            a = out.argmax().item()
            return a

def preprocess_state(state):
    # resize (210x160) to (110x84)
    s = cv2.resize(state, dsize=(84, 110))
    # crop (84x84)
    s = s[16:100, :]
    # transform to input shape (1 * 84 * 84)
    s = np.array([s])
    
    return s

for env_name in ENV_LIST:
    print('==============', env_name, '==============')
    
    # Load Environment
    env = gym.make('ALE/'+env_name, 
                    obs_type='grayscale', 
                    frameskip=4,
                    render_mode='human')
    a_size = env.action_space.n
    
    # Load Model 
    q = DQN(action_size=a_size)
    q.load_state_dict(torch.load(f'{PATH}/model_dqn_{env_name}.pt'))
    q.eval()
    
    for i in range(5):
        s = preprocess_state(env.reset()[0])
        done = False
        
        # state history for using as input
        s_history = np.concatenate((s, s, s, s))
        
        score = 0
        lives = 0
        
        while not done:
            a = q.epsilon_greedy(torch.from_numpy(np.array([s_history])), EPS)
            
            s_prime, r, done, tr, info = env.step(a)
            
            # reward 
            r = 1 if r > 0 else -1 if info['lives'] < lives else 0
            
            s_prime = preprocess_state(s_prime)
            done_mask = 0 if done else 1
            
            # Stack last 4 frame to producethe input to the q-function
            s_history = np.concatenate((s_history, s))[1:]
            s_prime_history = np.concatenate((s_history, s_prime))[1:]
            
            # next stage
            s = s_prime
            score += r
            lives = info['lives']
            
            # game over
            if done:
                break
        
    env.close()
    del env