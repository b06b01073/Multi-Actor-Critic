import numpy as np
from model import Actor, Critic
from util import *
from random_process import OrnsteinUhlenbeckProcess, GuassianNoise
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

        nb_states = (args.data_interval - 1) * args.input_size
 
        self.agent_type = args.agent_type
        self.input_size = args.input_size
        self.date = args.date
        ##### Create Actor Network #####
        self.actor = Actor(nb_states=nb_states, hidden_dim=args.hidden_dim).to(device)
        self.actor_target = Actor(nb_states=nb_states, hidden_dim=args.hidden_dim).to(device)
        ##### Create Critic Network #####
        self.critic = Critic(nb_states=nb_states).to(device)
        self.critic_target = Critic(nb_states=nb_states).to(device)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        # Hyper-parameters
        self.is_training = True
        self.rnn_mode = args.rnn_mode
        self.tau = args.tau
        self.discount = args.discount
        self.random_process = GuassianNoise(mu=0, sigma=0.5)


        ### Optimizer and LR_scheduler ###
        beta1 = args.beta1
        beta2 = args.beta2
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.c_rate, weight_decay=1e-3)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.a_rate, weight_decay=1e-3)


        ### initialized values 
        self.total_policy_loss = 0
        self.critic_loss = 0
        self.BC_loss = 0
        self.BC_loss_Qf = 0



        self.delay_update = 0

        self.action_freedom = 0.1

        self.noise_weight = 1

        ##### Behavior Cloning Setting #####
        self.is_BClone = args.is_BClone
        self.is_Qfilt = args.is_Qfilt
        self.use_Qfilt = args.use_Qfilt
        if self.is_BClone:
            self.lambda_Policy = args.lambda_Policy
            self.lambda_BC = 1-self.lambda_Policy
        else:
            self.lambda_Policy = 1
            self.lambda_BC = 1-self.lambda_Policy
        # self.lambda_BC = args.lambda_BC
        self.BC_loss_func = nn.MSELoss(reduce=False)
        # self.BC_loss_func = nn.BCELoss(reduce=False)

    def reset_asset(self):
        self.asset = self.init_asset
        

    def build_state(self, state):
        if state is None:
            return None

        prev_opens = [state.iloc[i]['norm_Open'] for i in range(self.data_interval - 1)]
        prev_highs = [state.iloc[i]['norm_High'] for i in range(self.data_interval - 1)]
        prev_closes = [state.iloc[i]['norm_Close'] for i in range(self.data_interval - 1)]
        prev_lows = [state.iloc[i]['norm_Low'] for i in range(self.data_interval - 1)]
        prev_volumes = [state.iloc[i]['norm_Volume'] for i in range(self.data_interval - 1)]
        prev_MA5 = [state.iloc[i]['norm_MA5'] for i in range(self.data_interval - 1)]
        prev_MA10 = [state.iloc[i]['norm_MA10'] for i in range(self.data_interval - 1)]
        if self.agent_type == 1:
            concat_data = prev_opens  + prev_highs + prev_lows + prev_closes + prev_MA5 + prev_MA10
            return torch.FloatTensor(concat_data)

    def increase_action_freedom(self):
        self.action_freedom += 0.02
        self.action_freedom = min(self.action_freedom, 1)

    def decrease_noise_weight(self):
        self.noise_weight -= 0.02
        self.noise_weight = max(self.noise_weight, 0.1)

    def take_action(self, state, hidden_state,noise_enable=True):
        # TODO: select action based on the model output

        state = self.build_state(state).to(device)
        hidden_state = torch.FloatTensor(hidden_state).to(device)
        action, hidden_state = self.actor(state, hidden_state)

        action = to_numpy(action.cpu())
        

        

        if noise_enable == True:
            noise = self.random_process.sample()
            action += noise * self.noise_weight


        # action = np.random.uniform(low=-1.0, high=1)
        action = np.clip(action, a_min=-1, a_max=1) * self.action_freedom
        # action = np.clip(action, a_min=-1, a_max=1) * 0.8 # it seems like when action_freedom = 1, the model breaks easily
        invested_asset = self.asset * np.abs(action)

        return action, invested_asset, state.squeeze().cpu().numpy(), hidden_state
    
    def reset_lstm_hidden_state(self, done=True):
        self.actor.reset_lstm_hidden_state(done)
    
    def learn(self, experiences, batch_size):
        # TODO: update the model params
        if experiences is None: # not enough samples
            return

        # update trajectory-wise
        

        action_bc = np.stack([data.action_bc for data in experiences])
        hidden_state = np.stack([data.hidden_state for data in experiences]).astype('float32')
        state0 = np.stack([data.state0 for data in experiences])
        # action = np.expand_dims(np.stack((data.action for data in experiences)), axis=1)
        action = np.stack([data.action for data in experiences])
        reward = np.stack([data.reward for data in experiences]).astype('float32')
        # reward = np.stack((data.reward for data in experiences))
        state1 = np.stack([data.state1 for data in experiences]) 
        terminal = np.stack([data.terminal1 for data in experiences])



        target_action, _ = self.actor_target(to_tensor(state1), to_tensor(hidden_state))
        next_q_value = self.critic_target([
            to_tensor(state1),
            target_action
        ])

        
        # target_q = to_tensor(reward) + self.discount * next_q_value * to_tensor(terminal).unsqueeze(dim=1)
        target_q = to_tensor(reward) + self.discount * next_q_value * to_tensor(terminal).unsqueeze(dim=1)

        

        # Critic update
        
        current_q = self.critic([to_tensor(state0), to_tensor(action)])


        # value_loss = criterion(q_batch, target_q_batch)
        value_loss = F.mse_loss(current_q, target_q.detach())

        self.critic_optim.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.2)
        self.critic_optim.step()

        # Actor update
        action, _ = self.actor(to_tensor(state0), to_tensor(hidden_state))
        policy_loss = -self.critic([
            to_tensor(state0),
            action
        ])


        ### Behavior Cloning : Estimate actor action ###
        # q_action = self.critic([xh_b0, action.cuda()])
        # => current_q
        ##### Behavior Cloning Loss #####
        if self.is_BClone:
            ### Estimate prophetic action ###
            
            q_action_bc = self.critic([to_tensor(state0), to_tensor(action_bc).unsqueeze(dim=1)])
            
            ### Q_filter & BC_loss ###
            BC_loss = self.BC_loss_func(action.float(), to_tensor(action_bc).unsqueeze(dim=1).cuda().float())
            BC_loss = torch.sum(BC_loss,dim=1).unsqueeze(1)
            BC_loss = BC_loss.to(torch.float32)

            Q_filter = torch.gt(q_action_bc, current_q)
            Q_filter = Q_filter.to(torch.float32)
            BC_loss_Qf = BC_loss.float() * (Q_filter.detach().float())
            BC_loss_Qf = BC_loss_Qf.to(torch.float32)
            if self.is_Qfilt:
                ### modified Policy loss ###
                policy_loss = (self.lambda_Policy*policy_loss) + (self.lambda_BC*BC_loss_Qf.float()) 
            else:
                ### modified Policy loss ###
                policy_loss = (self.lambda_Policy*policy_loss) + (self.lambda_BC*BC_loss.float())
                
                
        else:  ### Original Policy loss ###
            policy_loss = policy_loss
            

        

        # update per trajectory

        self.actor_optim.zero_grad()
        policy_loss = policy_loss.mean()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.2)
    
        policy_loss = policy_loss.to(torch.float32)

        policy_loss.backward()
        self.actor_optim.step()

        self.delay_update += 1

        #if self.delay_update % 20 == 0:
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return policy_loss, value_loss


        
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



