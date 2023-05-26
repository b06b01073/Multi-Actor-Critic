import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from util import *

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Actor(nn.Module):
    def __init__(self, nb_states, hidden_dim, nb_actions=1, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, nb_actions)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
        self.tanh = nn.Tanh()

    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

    def reset_lstm_hidden_state(self, done=True):
        if done == True:
            self.cx = Variable(torch.zeros(1, 300)).type(FLOAT).to(device)
            self.hx = Variable(torch.zeros(1, 300)).type(FLOAT).to(device)
        else:
            self.cx = Variable(self.cx.data).type(FLOAT).to(device)
            self.hx = Variable(self.hx.data).type(FLOAT).to(device)

    def forward(self, x, hidden_state):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        hidden_state = self.rnn(x, hidden_state)
        x = self.tanh(self.fc3(x))
        return x, hidden_state

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions=1, init_w=3e-3):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, 400)
        self.fc2 = nn.Linear(400 + nb_actions, 300)
        self.fc3 = nn.Linear(300, 1)
        self.relu = nn.ReLU()
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, xs):
        x, a = xs
        out = self.fc1(x)
        out = self.relu(out)
        # debug()
        #out = self.fc2(torch.cat([out,a],dim=1)) # dim should be 1, why doesn't work?
        out = self.fc2(torch.cat([out,a], 1)) # dim should be 1, why doesn't work?
        out = self.relu(out)
        out = self.fc3(out)
        return out