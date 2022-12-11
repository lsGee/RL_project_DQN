import numpy as np
import random
import gym
import ale_py
import collections
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
from tqdm import tqdm
import gc, warnings
warnings.filterwarnings('ignore')

PATH = 'PATH'

ENV_LIST = ['Breakout-v5', 'Seaquest-v5',
            # 'Enduro-v5', 'Pong-v5', 'Qbert-v5', 'SpaceInvaders-v5', 'BeamRider-v5'
            ]

NUM_EPISODES = 3000
INTERVAL = 20
EPS_END = 0.1
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


class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=BUFFER_LIMIT)
        # self.position = 0
    
    def size(self):
        return len(self.buffer)

    def put(self, transition):
        self.buffer.append(transition)
        # self.position = (self.position + 1) % self.BATCH_SIZE
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst, s_history_lst, s_prime_history_lst = [], [], [], [], [], [], []
        
        for transition in mini_batch:
            # next_state, reward, terminated, truncated
            s, a, r, s_prime, done_mask, s_history, s_prime_history = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])
            s_history_lst.append(s_history)
            s_prime_history_lst.append(s_prime_history)
            
        return torch.tensor(s_lst), torch.tensor(a_lst),\
               torch.tensor(r_lst), torch.tensor(s_prime_lst),\
               torch.tensor(done_mask_lst), torch.tensor(s_history_lst),\
               torch.tensor(s_prime_history_lst)

def preprocess_state(state):
    # resize (210x160) to (110x84)
    s = cv2.resize(state, dsize=(84, 110))
    # crop (84x84)
    s = s[16:100, :]
    # transform to input shape (1 * 84 * 84)
    s = np.array([s])
    
    return s

def train(q, q_target, memory, optimizer):
    global Q_VALUE
    for i in range(10):
        s, a, r, s_prime, done_mask, s_history, s_prime_history = memory.sample(BATCH_SIZE)
        
        q_out = q(s_history)
        q_a = q_out.gather(1, a)
        max_q_prime = q_target(s_prime_history).max(1)[0].unsqueeze(1)
        target = r + GAMMA * max_q_prime * done_mask
        
        # loss function (MSE)
        loss = F.mse_loss(q_a, target)
        
        # Train NN
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        Q_VALUE.extend(list(map(lambda x: x.item(), q_a)))


if __name__ == '__main__':
    for env_name in ENV_LIST:
        print('==============', env_name, '==============')
        
        # Load Environment
        env = gym.make('ALE/'+env_name, 
                       obs_type='grayscale', 
                       frameskip=4,
                       render_mode='human')
        a_size = env.action_space.n
        
        # Create 2 NNs
        q = DQN(action_size=a_size)
        q_target = DQN(action_size=a_size)
        
        # Copy parameters
        q_target.load_state_dict(q.state_dict())
        
        # Create Replay Buffer
        memory = ReplayBuffer()
        
        optimizer = optim.RMSprop(q.parameters(), lr=LEARNING_RATE)
        score = 0.0
        
        for n_epi in tqdm(range(1, NUM_EPISODES+1)):
            
            # epsilon greedy
            epsilon = max(EPS_END, 1.0 - 0.9*(n_epi/NUM_EPISODES))
            
            s = preprocess_state(env.reset()[0])
            done = False
            
            # state history for using as input
            s_history = np.concatenate((s, s, s, s))
            
            # initailize lives info
            lives = 0
            
            while not done:
                a = q.epsilon_greedy(torch.from_numpy(np.array([s_history])), epsilon)
                
                s_prime, r, done, tr, info = env.step(a)
                
                # reward 
                r = 1 if r > 0 else -1 if info['lives'] < lives else 0
                
                s_prime = preprocess_state(s_prime)
                done_mask = 0 if done else 1
                
                # Stack last 4 frame to producethe input to the q-function
                s_history = np.concatenate((s_history, s))[1:]
                s_prime_history = np.concatenate((s_history, s_prime))[1:]
                
                memory.put((s, a, r, s_prime, done_mask, s_history, s_prime_history))
                
                # next stage
                s = s_prime
                score += r
                lives = info['lives']
                
                # game over
                if done:
                    break
            
            # Train
            if memory.size() > BATCH_SIZE:
                train(q, q_target, memory, optimizer)
                
            # Copy to target network
            if n_epi % INTERVAL == 0 and n_epi != 0:
                q_target.load_state_dict(q.state_dict())
                print(f'n_episode: {n_epi:.1f}, score(avg.): {score/INTERVAL}, n_buffer: {memory.size()}, eps: {epsilon*100:.1f}%')
                
                # Record Performance metrics
                GAME_NAME.append(env_name)
                IDX_EPISODE.append(n_epi)
                IDX_EPOCH.append(int(n_epi/INTERVAL))
                AVG_REWARD.append(score/INTERVAL)
                AVG_Q_VALUE.append(np.mean(Q_VALUE))
                
                # Initialize
                Q_VALUE = []
                score = 0.0
        
        # Save Model
        torch.save(q.state_dict(), f'{PATH}/model_dqn_{env_name}.pt')
        
        env.close()
        del env
        gc.collect()
    
    
    # Save Perfomance Info.        
    result_df = pd.DataFrame({'GAME': GAME_NAME, 'EPISODE': IDX_EPISODE, 'REWARD_AVG': AVG_REWARD, 'ACTION_VAL_AVG': AVG_Q_VALUE})
    result_df.head(10)
    result_df.to_csv('{PATH}/output.csv', index=False)