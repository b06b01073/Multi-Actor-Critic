import numpy as np
#from model import Actor, Critic
from new_model import (RNN, Actor, Critic)
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
        #print(f'using rnn mode: {args.rnn_mode}')

        self.init_asset = args.asset
        self.asset = args.asset
        self.data_interval = args.data_interval
        nb_actions = 1

        nb_states = (args.data_interval - 1) * args.input_size

        args.hidden_rnn = nb_states
 
        self.agent_type = args.agent_type
        self.input_size = args.input_size
        self.date = args.date
        # ##### Create Actor Network #####
        # self.actor = Actor(nb_states=nb_states, hidden_dim=args.hidden_dim).to(device)
        # self.actor_target = Actor(nb_states=nb_states, hidden_dim=args.hidden_dim).to(device)
        # ##### Create Critic Network #####
        # self.critic = Critic(nb_states=nb_states, hidden_dim=args.hidden_dim).to(device)
        # self.critic_target = Critic(nb_states=nb_states, hidden_dim=args.hidden_dim).to(device)



        ##### Create RNN Layer #####
        self.rnn = RNN(args).cuda()
        self.rnn_target = RNN(args).cuda()
        ##### Create Actor Network #####
        self.actor = Actor(args).cuda()
        self.actor_target = Actor(args).cuda()
        ##### Create Critic Network #####
        self.critic = Critic(args).cuda()
        self.critic_target = Critic(args).cuda()


        ##### Model Setting #####

        self.hidden_rnn = args.hidden_rnn
        self.num_layer = args.num_rnn_layer
        self.batch_size = args.batch_size


        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        # Hyper-parameter
        self.is_training = True

        self.tau = args.tau
        self.discount = args.discount
        #self.random_process = GuassianNoise(mu=0, sigma=0.15)

        # Hyper-parameters

        self.depsilon = 1.0 / args.epsilon_decay
        self.epsilon = 1.0
        self.random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=args.ou_theta, mu=args.ou_mu, sigma=args.ou_sigma)
        ### Optimizer and LR_scheduler ###
        beta1 = args.beta1
        beta2 = args.beta2
        #self.critic_optim  = Adam(self.critic.parameters(), lr=args.c_rate, weight_decay=1e-4)
        #self.actor_optim  = Adam(self.actor.parameters(), lr=args.a_rate, weight_decay=1e-4)

        self.rnn_optim = Adam(self.rnn.parameters(), lr=args.r_rate, betas=(beta1, beta2))
        self.rnn_scheduler = Scheduler.StepLR(self.rnn_optim, step_size=args.sch_step_size, gamma=args.sch_gamma)
        self.critic_optim  = Adam(self.critic.parameters(), lr=args.c_rate, betas=(beta1, beta2))
        self.critic_scheduler = Scheduler.StepLR(self.critic_optim, step_size=args.sch_step_size, gamma=args.sch_gamma)
        self.actor_optim  = Adam(self.actor.parameters(), lr=args.a_rate, betas=(beta1, beta2))
        self.actor_scheduler = Scheduler.StepLR(self.actor_optim, step_size=args.sch_step_size, gamma=args.sch_gamma)


        self.temp_observation = ['norm_Open','norm_Close','norm_High','norm_Low',
                                 'norm_Volume','norm_MA5', 'norm_MA10']

        ### initialized values 
        self.total_policy_loss = 0
        self.critic_loss = 0


        self.delay_update = 0

        self.action_freedom = 0.01


    def reset_asset(self):
        self.asset = self.init_asset
        

    def build_state(self, state):
        if state is None:
            return None
        # print(state)
        prev_opens = [state.iloc[i]['norm_Open'] for i in range(self.data_interval - 1)]
        prev_highs = [state.iloc[i]['norm_High'] for i in range(self.data_interval - 1)]
        prev_closes = [state.iloc[i]['norm_Close'] for i in range(self.data_interval - 1)]
        prev_lows = [state.iloc[i]['norm_Low'] for i in range(self.data_interval - 1)]
        prev_volumes = [state.iloc[i]['norm_Volume'] for i in range(self.data_interval - 1)]
        prev_MA5 = [state.iloc[i]['norm_MA5'] for i in range(self.data_interval - 1)]
        prev_MA10 = [state.iloc[i]['norm_MA10'] for i in range(self.data_interval - 1)]
        self.temp_observation = ['norm_Open','norm_Close','norm_High','norm_Low',
                                 'norm_Volume','norm_MA5', 'norm_MA10']
        if self.agent_type == 1:
            concat_data = prev_opens  + prev_highs + prev_lows + prev_closes + prev_MA5 + prev_MA10
            state = state[self.temp_observation][ : -1]
            #return torch.FloatTensor(concat_data)
            tensor_state = torch.from_numpy(state.values).float()
            return tensor_state
    
    def filter_state(self,state):
        if state is None:
            return None
        columns = ['norm_Open','norm_Close','norm_High','norm_Low',
                                 'norm_Volume','norm_MA5', 'norm_MA10']
        #state = state.filter(items=columns)
        #state = state[columns][ : ]
        print(state)
        tensor_state = torch.from_numpy(state.values).float()
        return tensor_state
    def get_freedom(self):
        return self.action_freedom

    def increase_action_freedom(self):
        self.action_freedom += 0.02
        self.action_freedom = min(self.action_freedom, 1)

    def take_example_action(self, winset, state):
        # ma5=state.iloc[1]['norm_MA5']-state.iloc[2]['norm_MA5']
        # action = 0
        # if state.iloc[0]['norm_Open'] > ma5 and ma5 > 0:
        #     if state.iloc[1]['norm_MA5']>state.iloc[1]['norm_MA10']:
        #         action = 0.8
        #     else:
        #         action = 0.4
        # elif state.iloc[0]['norm_Open'] < ma5 and ma5 < 0:
        #     if state.iloc[1]['norm_MA5']<state.iloc[1]['norm_MA10']:
        #         action = -0.8 
        #     else:
        #         action=-0.4
        price_change=state.iloc[0]['norm_Open']-state.iloc[0]['norm_Close']
        
        return 0.7 if price_change*winset > 0 else - 0.7
    def take_action_ori(self, state, actor_hidden_state, critic_hidden_state, noise_enable=True):
        # TODO: select action based on the model output
        ma5=state.iloc[1]['norm_MA5']-state.iloc[2]['norm_MA5']
        state = self.build_state(state).to(device)
        #print(state)
        actor_hidden_state = torch.FloatTensor(actor_hidden_state).to(device)
        action, actor_hidden_state = self.actor(state, actor_hidden_state)

        with torch.no_grad():
            critic_hidden_state = torch.FloatTensor(critic_hidden_state).to(device)
            _, critic_hidden_state = self.critic([state, action, critic_hidden_state])

        action = to_numpy(action.cpu()) * self.action_freedom
        

        

        if noise_enable == True:
            
            noise = self.random_process.sample()
            #if ma5 > 0 and noise<0:
            #    noise *= -1
            #if ma5 < 0 and noise > 0:
            #   noise *= -1
            action += noise 


        # action = np.random.uniform(low=-1.0, high=1)
        action = np.clip(action, a_min=-0.8, a_max=0.8)
        # action = np.clip(action, a_min=-1, a_max=1) * 0.8 # it seems like when action_freedom = 1, the model breaks easily
        invested_asset = self.asset * np.abs(action)

        return action, invested_asset, state.squeeze().cpu().numpy(), actor_hidden_state, critic_hidden_state
    


    

    def take_action(self, state, noise_enable=True, decay_epsilon=True):
        # TODO: select action based on the model output
        xh, _ = self.rnn(state.unsqueeze(dim=0))
        action = self.actor(xh.squeeze())
        
        action = to_numpy(action.cpu()).squeeze(0)
        action = np.clip(action, a_min=-1, a_max=1) * self.action_freedom
        if noise_enable == True:
            # print(self.is_training * max(self.epsilon, 0)*self.random_process.sample().item())
            action += self.is_training * max(self.epsilon, 0)*self.random_process.sample().item()

        # print(action)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        # return action, self.epsilon

        # action = np.random.uniform(low=-1.0, high=1)
        invested_asset = self.asset * np.abs(action)
        

        return np.clip(action, a_min=-1, a_max=1), invested_asset
    


    # def reset_lstm_hidden_state(self, done=True):
    #     self.actor.reset_lstm_hidden_state(done)
    
    def learn_ori(self, experiences, batch_size,epoch):
        # TODO: update the model params
        if experiences is None: # not enough samples
            return

        # update trajectory-wise
        

        
        actor_hidden_state = np.stack([data.actor_hidden_state for data in experiences]).astype('float32')
        critic_hidden_state = np.stack([data.critic_hidden_state for data in experiences]).astype('float32')
        state0 = np.stack([data.state0 for data in experiences])
        # action = np.expand_dims(np.stack((data.action for data in experiences)), axis=1)
        action = np.stack([data.action for data in experiences])
        reward = np.stack([data.reward for data in experiences]).astype('float32')
        # reward = np.stack((data.reward for data in experiences))
        state1 = np.stack([data.state1 for data in experiences]) 
        terminal = np.stack([data.terminal1 for data in experiences])

        with torch.no_grad():
            _, next_actor_hidden_state = self.actor_target(to_tensor(state0), to_tensor(actor_hidden_state))

        with torch.no_grad():
            _, next_critic_hidden_state = self.critic_target([
                to_tensor(state0),
                to_tensor(action),
                to_tensor(critic_hidden_state)
            ])

        target_action, _ = self.actor_target(to_tensor(state1), next_actor_hidden_state)
        next_q_value, _ = self.critic_target([
            to_tensor(state1),
            target_action,
            next_critic_hidden_state
        ])

        
        # target_q = to_tensor(reward) + self.discount * next_q_value * to_tensor(terminal).unsqueeze(dim=1)
        target_q = to_tensor(reward) + self.discount * next_q_value * to_tensor(terminal).unsqueeze(dim=1)

        

        # Critic update
        current_q, _ = self.critic([to_tensor(state0), to_tensor(action), to_tensor(critic_hidden_state)])


        # value_loss = criterion(q_batch, target_q_batch)
        value_loss = F.mse_loss(current_q, target_q.detach())

        self.critic_optim.zero_grad()
        value_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        # Actor update
        action, _ = self.actor(to_tensor(state0), to_tensor(actor_hidden_state))
        policy_loss, _ = self.critic([
            to_tensor(state0),
            action,
            to_tensor(critic_hidden_state)
        ])


        # update per trajectory

        self.actor_optim.zero_grad()
        policy_loss = -policy_loss.mean()
        
        #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        policy_loss.backward()
        self.actor_optim.step()

        self.delay_update += 1

        if self.delay_update % 20 == 0:
            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

        return policy_loss, value_loss


    def learn(self,experiences):

        if experiences is None:
            return
      
        t_len = len(experiences)
        # action_bc, state0, action, reward, done = experiences\
        

        for t in range(t_len - 1):

            a_cx = Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_rnn)).type(FLOAT).cuda()
            a_hx = Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_rnn)).type(FLOAT).cuda()
            
            #action_bc = np.stack((trajectory.action_bc for trajectory in experiences[t]))
            state0 = np.stack((trajectory.state0 for trajectory in experiences[t]))        
            action = np.stack((trajectory.action for trajectory in experiences[t]))
            done = np.stack((trajectory.terminal1 for trajectory in experiences[t]))
            # print('done:', done)
            done = to_tensor(done).unsqueeze(dim=1).type(torch.LongTensor).cuda()
            action = to_tensor(action).unsqueeze(dim=1).type(torch.FloatTensor)
            reward = np.expand_dims(np.stack((trajectory.reward for trajectory in experiences[t])), axis=1)
            reward = to_tensor(reward).cuda()
            state1 = np.stack((trajectory.state0 for trajectory in experiences[t+1]))
            # util:to_tensor : tensor = torch.from_numpy(ndarray)  tensor = tensor.to(device)
            #state0_cuda = to_tensor(state0.astype(float)).cuda()
            #state1_cuda = to_tensor(state1.astype(float)).cuda()
            #state0_cuda = self.filter_state(state0)
            #state1_cuda = self.filter_state(state1)
            #state0_cuda = torch.from_numpy(state0.astype(float))
            #state1_cuda = torch.from_numpy(state1.astype(float))
            state0_cuda = to_tensor(state0)
            state1_cuda = to_tensor(state1)

            self.update_critic(state0_cuda, a_hx,a_cx, action, reward, state1_cuda, done,t_len)
            self.update_actor(state0_cuda, a_hx,a_cx, action,t_len)

        ##### Learning rate Scheduling #####
        self.rnn_scheduler.step()
        self.critic_scheduler.step()
        self.actor_scheduler.step()

        ##### Apply Q-filter to BC loss #####
        # if (episode-1) >= (self.warmup+self.use_Qfilt):
        #     self.is_Qfilt=True


        self.total_policy_loss = []
        self.critic_loss = []



    def update_critic(self, state0_cuda, a_hx,a_cx, action, reward, state1_cuda, done,t_len):
        # with torch.no_grad():
        # ref : tensor_state = torch.from_numpy(state.values).float()
        
        xh0, _ = self.rnn(state0_cuda, a_hx)
        current_q = self.critic([xh0, action.cuda()])
        
        with torch.no_grad():
            xh1, _ = self.rnn_target(state1_cuda, a_hx)
            target_action = self.actor_target(xh1)
            target_action = target_action.detach()
            next_q_value = self.critic_target([xh1, target_action])
                
        # print(reward.shape, done.shape, next_q_value.shape)
        target_q = reward + (1-done) * self.discount * next_q_value
        
        value_loss = 0
        value_loss_total = 0
        value_loss = F.smooth_l1_loss(current_q, target_q.cuda())

        value_loss /= t_len # divide by experience length
        value_loss_total += value_loss 

        #self.critic_loss.append(value_loss_total)

        ####### update Critic per step ####### 
        self.rnn.zero_grad()
        self.actor.zero_grad()
        self.critic.zero_grad()
        value_loss.backward()
        self.critic_optim.step()
        self.rnn_optim.step()  

    def update_actor(self, state0_cuda, a_hx,a_cx, action,t_len):
        
        xh_b0, _ = self.rnn(state0_cuda, a_hx)
        behavior_action = self.actor(xh_b0)    


        ### Behavior Cloning : Estimate actor action ###
        q_action = self.critic([xh_b0, action.cuda()]) 

        ### Calculate Actor loss based on Q-value ###
        behavior_action = self.actor(xh_b0)
        actor_loss = -self.critic([xh_b0, behavior_action])
        
        # ##### Behavior Cloning Loss #####
        # if self.is_BClone:
        #     ### Estimate prophetic action ###
        #     q_action_bc = self.agent.critic([xh_b0, action_bc.cuda()])
            
        #     ### Q_filter & BC_loss ###
        #     BC_loss = self.BC_loss_func(behavior_action, action_bc.cuda())
        #     BC_loss = torch.sum(BC_loss,dim=1).unsqueeze(1)
            
        #     Q_filter = torch.gt(q_action_bc, q_action)
        #     BC_loss_Qf = BC_loss * (Q_filter.detach())
        #     if self.is_Qfilt:
        #         ### modified Policy loss ###
        #         policy_loss = (self.lambda_Policy*actor_loss) + (self.lambda_BC*BC_loss_Qf)
        #     else:
        #         ### modified Policy loss ###
        #         policy_loss = (self.lambda_Policy*actor_loss) + (self.lambda_BC*BC_loss)
                
        # else:  ### Original Policy loss ###
        #     policy_loss = actor_loss

        policy_loss = actor_loss
        
        ################## Actor loss calculation ##################

        # if self.is_BClone:
        #     BC_loss /= t_len
        #     BC_loss_total +=  BC_loss.mean()  #BC loss
        #     BC_loss_Qf  /= t_len
        #     BC_loss_Qf_total += BC_loss_Qf.mean()
        #     actor_loss /= t_len
        #     actor_loss_total += actor_loss.mean()   #actor loss
        # else:
        #     BC_loss_total = torch.zeros(1)
        #     BC_loss_Qf_total = torch.zeros(1)
        #     actor_loss_total = torch.zeros(1)

        actor_loss_total = torch.zeros(1)
        policy_loss_total = 0
        
        policy_loss /= t_len # divide by experience length
        policy_loss_total += policy_loss.mean()

        #self.total_policy_loss.append(policy_loss_total)

        ####### Update Actor ###########
        self.rnn.zero_grad()
        self.actor.zero_grad()
        self.critic.zero_grad()
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        self.rnn_optim.step()  


        self.actor_loss = actor_loss_total.item()
        #self.BC_loss = BC_loss_total.item()
        #self.BC_loss_Qf = BC_loss_Qf_total.item()



        
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
        description = '_' +'ep' +str(episode) +'_' +'rd' +str(e_reward) +'_' +str(self.date) +'.pkl'
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


    def save(self):
        torch.save(self.critic, 'critic.pth')
        torch.save(self.actor, 'actor.pth')

