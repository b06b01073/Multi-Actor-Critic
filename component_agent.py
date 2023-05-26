import numpy as np
from model import Actor, Critic
from util import *
from random_process import OrnsteinUhlenbeckProcess
from scipy.special import softmax
import torch
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim.lr_scheduler as Scheduler

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ComponentAgent:
    def __init__(self, args):
        print(f'using rnn mode: {args.rnn_mode}')

        self.init_asset = args.asset
        self.asset = args.asset
        self.data_interval = args.data_interval
        # TODO: build the nn model
        # self.net = None

        nb_actions = 1

        self.agent_type = args.agent_type
        self.input_size = args.input_size
        self.date = args.date
        ##### Create Actor Network #####
        self.actor = Actor(nb_states=self.input_size).to(device)
        self.actor_target = Actor(nb_states=self.input_size).to(device)
        ##### Create Critic Network #####
        self.critic = Critic(nb_states=self.input_size).to(device)
        self.critic_target = Critic(nb_states=self.input_size).to(device)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        # Hyper-parameters
        self.is_training = True
        self.rnn_mode = args.rnn_mode
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon_decay
        self.epsilon = 1.0
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)


        ### Optimizer and LR_scheduler ###
        beta1 = args.beta1
        beta2 = args.beta2
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.c_rate, betas=(beta1, beta2))
        self.critic_scheduler = Scheduler.StepLR(self.critic_optim, step_size=args.sch_step_size, gamma=args.sch_gamma)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.a_rate, betas=(beta1, beta2))
        self.actor_scheduler = Scheduler.StepLR(self.actor_optim, step_size=args.sch_step_size, gamma=args.sch_gamma)


        ### initialized values 
        self.total_policy_loss = 0
        self.critic_loss = 0



    def reset_asset(self):
        self.asset = self.init_asset
        

    def build_state(self, state):
        cur_open = state.iloc[1]['Open']
        prev_open = state.iloc[0]['Open']
        prev_high = state.iloc[0]['High']
        prev_close = state.iloc[0]['Close']
        prev_low = state.iloc[0]['Low']
        prev_volume = state.iloc[0]['Volume']
        if self.agent_type == 1:
            return torch.FloatTensor([cur_open, prev_open, prev_high, prev_low, prev_close, prev_volume]).to(device)

    def take_action(self, state, noise_enable=True, decay_epsilon=True):
        # TODO: select action based on the model output

        state = self.build_state(state).unsqueeze(dim=0)
        action, _ = self.actor(state)
        
        action = to_numpy(action.cpu()).squeeze(0)
        if noise_enable == True:
            action += self.is_training * max(self.epsilon, 0)*self.random_process.sample()
            action = np.clip(action, a_min=-1, a_max=1)

        # print(action)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        # return action, self.epsilon

        # action = np.random.uniform(low=-1.0, high=1)
        invested_asset = self.asset * np.abs(action)

        return action, invested_asset, state.squeeze().cpu().numpy()
    
    def reset_lstm_hidden_state(self, done=True):
        self.actor.reset_lstm_hidden_state(done)
    
    def learn(self, experiences, batch_size):
        # TODO: update the model params
        if experiences is None: # not enough samples
            return

        # update trajectory-wise
        for t in range(len(experiences) - 1): # iterate over episodes
            target_cx = Variable(torch.zeros(batch_size, 120)).type(FLOAT).to(device)
            target_hx = Variable(torch.zeros(batch_size, 120)).type(FLOAT).to(device)

            cx = Variable(torch.zeros(batch_size, 120)).type(FLOAT).to(device)
            hx = Variable(torch.zeros(batch_size, 120)).type(FLOAT).to(device)

            # we first get the data out of the sampled experience
        
            state0 = np.stack([trajectory.state0 for trajectory in experiences[t]])
            # action = np.expand_dims(np.stack((trajectory.action for trajectory in experiences[t])), axis=1)
            action = np.stack([trajectory.action for trajectory in experiences[t]])
            reward = np.stack([trajectory.reward for trajectory in experiences[t]]) 
            # reward = np.stack((trajectory.reward for trajectory in experiences[t]))
            state1 = np.stack([trajectory.state0 for trajectory in experiences[t+1]])


            with torch.no_grad():
                target_action, (target_hx, target_cx) = self.actor_target(to_tensor(state1), (target_hx, target_cx))
                next_q_value = self.critic_target([
                    to_tensor(state1),
                    target_action
                ])

                
                target_q = to_tensor(reward) + self.discount * next_q_value

            # Critic update
            current_q = self.critic([to_tensor(state0), to_tensor(action)])


            # value_loss = criterion(q_batch, target_q_batch)
            value_loss = F.smooth_l1_loss(current_q, target_q)
            value_loss /= len(experiences) # divide by trajectory length

            self.critic_optim.zero_grad()
            value_loss.backward()
            self.critic_optim.step()

            # Actor update
            action, (hx, cx) = self.actor(to_tensor(state0), (hx, cx))
            policy_loss = -self.critic([
                to_tensor(state0),
                action
            ])
            policy_loss /= len(experiences) # divide by trajectory length

            # update per trajectory

            self.actor_optim.zero_grad()
            policy_loss = policy_loss.mean()
            policy_loss.backward()
            self.actor_optim.step()


        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)


        
    def soft_update(self):
        ##### Target_Net update #####
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)


    def update_asset(self, earning):
        self.asset += earning



    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()


    # training : reset_noises
    def reset_noise(self):
        self.random_process.reset_states()

    def cuda(self):
        #device = torch.device('cuda:0')
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
    
    def load_weights(self, checkpoint_path, model_fn):
        if checkpoint_path is None: return False
        
        model_path = checkpoint_path +'/test_case/' +model_fn
        model = torch.load(model_path)

        self.actor.load_state_dict(model['actor'])
        self.critic.load_state_dict(model['critic'])

        return True

    def save_model(self, checkpoint_path, episode, ewma_reward):
        e_reward = int(np.round(ewma_reward)) #(ewma_reward,2)
        description = '_' +self.rnn_mode +'_' +'ep' +str(episode) +'_' +'rd' +str(e_reward) +'_' +str(self.date) +'.pkl'
        if self.is_BClone:
            description = '_BC' +description
        model_path = checkpoint_path +'/' +description
        torch.save({
                    'actor': self.actor.state_dict(),
                    'critic': self.critic.state_dict(),
                    # 'actor_target': self.actor_target.state_dict(),
                    # 'critic_target': self.critic_target.state_dict(),
                    # 'rnn_opt': self.rnn_optim.state_dict(),
                    # 'actor_opt': self.actor_optim.state_dict(),
                    # 'critic_opt': self.critic_optim.state_dict(),
                    }, model_path)


