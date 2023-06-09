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

        self.init_asset = args.asset
        self.asset = args.asset
        self.data_interval = args.data_interval
        nb_actions = 1

        nb_states = (args.data_interval - 1) * args.input_size

        args.hidden_rnn = nb_states
 
        self.agent_type = args.agent_type
        self.input_size = args.input_size
        self.date = args.date

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
        
        # Hyper-parameters
        self.is_training = True
        #self.rnn_mode = args.rnn_mode
        self.tau = args.tau
        self.discount = args.discount
        self.a_update_freq = args.a_update_freq

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

        self.actor_loss = 0
        self.BC_loss = 0
        self.BC_loss_Qf = 0
        self.tot_policy_loss = 0
        self.critic_loss = 0


        self.delay_update = 0

        self.action_freedom = 0.01

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

    def increase_action_freedom(self):
        self.action_freedom += 0.01
        self.action_freedom = min(self.action_freedom, 1)

    def take_action(self, state, noise_enable=True, decay_epsilon=True):

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

        invested_asset = self.asset * np.clip(action, a_min=-1, a_max=1)
        
        return np.clip(action, a_min=-1, a_max=1), invested_asset

    
    def learn(self,experiences,epoch):

        if experiences is None:
            return
      
        t_len = len(experiences)

        self.actor_loss_total = 0  #actor loss
        self.BC_loss_total = 0  #BC loss
        self.BC_loss_Qf_total = 0  #BC loss after Q-filter
        self.policy_loss_total = 0  #policy loss
        self.value_loss_total = 0  #critic loss
        
        for t in range(t_len - 1):

            #a_cx = Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_rnn)).type(FLOAT).cuda()
            a_hx = Variable(torch.zeros(self.num_layer, self.batch_size, self.hidden_rnn)).type(FLOAT).cuda()
            
            action_bc = np.stack((trajectory.action_bc for trajectory in experiences[t]))
            state0 = np.stack((trajectory.state0 for trajectory in experiences[t]))        
            action = np.stack((trajectory.action for trajectory in experiences[t]))
            done = np.stack((trajectory.terminal1 for trajectory in experiences[t]))
            done = to_tensor(done).unsqueeze(dim=1).type(torch.LongTensor).cuda()
            action = to_tensor(action).unsqueeze(dim=1).type(torch.FloatTensor)
            #print('action: ',action)
            action_bc = to_tensor(action_bc).type(torch.FloatTensor)
            #print('action_bc: ',action_bc)
            reward = np.expand_dims(np.stack((trajectory.reward for trajectory in experiences[t])), axis=1)
            reward = to_tensor(reward).cuda()
            state1 = np.stack((trajectory.state0 for trajectory in experiences[t+1]))
            state0_cuda = to_tensor(state0)
            state1_cuda = to_tensor(state1)

            self.update_critic(state0_cuda, a_hx, action, reward, state1_cuda, done,t_len)
            if t % self.a_update_freq ==0: # update Actor per 3-steps 
                self.update_actor(action_bc,state0_cuda, a_hx, action,t_len,epoch)

        ##### Learning rate Scheduling #####
        self.rnn_scheduler.step()
        self.critic_scheduler.step()
        self.actor_scheduler.step()

        ##### Target_Net update #####
        soft_update(self.rnn_target, self.rnn, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return  self.actor_loss, self.BC_loss, self.BC_loss_Qf, self.tot_policy_loss, self.critic_loss


    def update_critic(self, state0_cuda, a_hx, action, reward, state1_cuda, done,t_len):
        
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
        value_loss = F.smooth_l1_loss(current_q, target_q.cuda())

        value_loss /= t_len # divide by experience length
        self.value_loss_total += value_loss 

        ####### update Critic per step ####### 
        self.rnn.zero_grad()
        self.actor.zero_grad()
        self.critic.zero_grad()
        value_loss.backward()
        self.critic_optim.step()
        self.rnn_optim.step()  

    def update_actor(self, action_bc,state0_cuda, a_hx, action,t_len,epoch):
        
        xh_b0, _ = self.rnn(state0_cuda, a_hx)
        behavior_action = self.actor(xh_b0)    
        actor_loss = -self.critic([xh_b0, behavior_action])


        ### Behavior Cloning : Estimate actor action ###
        # same with update_critic's current_q
        # question : why not behavior_action
        q_action = self.critic([xh_b0, action.cuda()]) 

        # ##### Behavior Cloning Loss #####
        if self.is_BClone and (epoch > self.use_Qfilt):
            ### Estimate prophetic action ###
            q_action_bc = self.critic([xh_b0, action_bc.cuda()])
            
            ### Q_filter & BC_loss ###
            BC_loss = self.BC_loss_func(behavior_action, action_bc.cuda())
            BC_loss = torch.sum(BC_loss,dim=1).unsqueeze(1)
            BC_loss = -BC_loss
            
            # when the critic indicates that the expert actions perform better than the actor actions
            Q_filter = torch.gt(q_action_bc, q_action)
            BC_loss_Qf = BC_loss * (Q_filter.detach())
            if self.is_Qfilt:
                ### modified Policy loss ###
                policy_loss = (self.lambda_Policy*actor_loss) + (self.lambda_BC*BC_loss_Qf)
            else:
                ### modified Policy loss ###
                policy_loss = (self.lambda_Policy*actor_loss) + (self.lambda_BC*BC_loss)
                
        else:  ### Original Policy loss ###
            policy_loss = actor_loss

        
        ################## Actor loss calculation ##################

        if self.is_BClone and (epoch > self.use_Qfilt):
            BC_loss /= t_len
            self.BC_loss_total +=  BC_loss.mean()  #BC loss
            BC_loss_Qf  /= t_len
            self.BC_loss_Qf_total += BC_loss_Qf.mean()
            actor_loss /= t_len
            self.actor_loss_total += actor_loss.mean()   #actor loss
        else:
            self.BC_loss_total = torch.zeros(1)
            self.BC_loss_Qf_total = torch.zeros(1)
            self.actor_loss_total = torch.zeros(1)

        
        
        policy_loss /= t_len # divide by experience length
        self.policy_loss_total += policy_loss.mean()



        ####### Update Actor ###########
        self.rnn.zero_grad()
        self.actor.zero_grad()
        self.critic.zero_grad()
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()
        self.rnn_optim.step()  


        ########### Record all losses ############
        self.actor_loss = self.actor_loss_total.item()
        self.BC_loss = self.BC_loss_total.item()
        self.BC_loss_Qf = self.BC_loss_Qf_total.item()
        self.tot_policy_loss = self.policy_loss_total.item()
        self.critic_loss = self.value_loss_total.item()



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



